"""Portable entry point for the Track 4 road and bridge inspection system."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import tempfile
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath, PureWindowsPath

from prepare_assets import asset_group, sha256, verify_assets

ROOT = Path(__file__).resolve().parent
AGENT_ROOT = ROOT / "code/code"
PROFILES = {
    "online": "submission_risk_gate_teacher.yaml",
    "onsite": "submission_risk_gate_teacher_offline.yaml",
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def competition_entry():
    spec = importlib.util.spec_from_file_location("track4_competition_entry", ROOT / "code/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_metadata(entry, input_dir: Path, supplied: Path | None) -> list[dict]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"找不到输入目录 {input_dir}")
    source = supplied or input_dir / "metadata.json"
    if supplied or source.is_file():
        rows = json.loads(source.read_text(encoding="utf-8-sig"))
        if not isinstance(rows, list):
            raise ValueError("metadata 必须是 JSON 数组。")
    else:
        rows = []
        for image in entry._image_files(input_dir):
            relative = image.relative_to(input_dir)
            parts = relative.parts
            category = parts[0]
            if category == "桥梁" and len(parts) >= 3:
                bridge = parts[1]
            elif category == "轨道" and len(parts) >= 2:
                bridge = ""
            else:
                raise ValueError(f"请在 metadata 中为 {relative} 提供任务类别和可信桥名。")
            rows.append({
                "questionCategory": category, "bridgeName": bridge,
                "filename": image.name, "imageRelativePath": relative.as_posix(),
            })

    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"metadata 第 {index + 1} 行必须是对象。")
        relative = row.get("imageRelativePath")
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError(f"metadata 第 {index + 1} 行缺少 imageRelativePath。")
        path = PurePosixPath(relative.replace("\\", "/"))
        if path.is_absolute() or PureWindowsPath(relative).drive or ".." in path.parts:
            raise ValueError(f"metadata 第 {index + 1} 行需要输入目录内的相对路径。")
        if path.name != row.get("filename"):
            raise ValueError(f"metadata 第 {index + 1} 行 filename 与相对路径的文件名不一致。")
        normalized.append({**row, "imageRelativePath": path.as_posix()})
    entry._validate_metadata_coverage(input_dir, normalized)
    return normalized


def configuration(args, data_dir: Path) -> dict:
    import yaml

    path = AGENT_ROOT / "configs" / PROFILES[args.profile]
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["run_mode"] = args.mode
    config["heads_only"]["rail_weights"] = str(data_dir / "rail_head/rail_head_v1.npz")
    visual = config["full_visual"]
    visual["bridge_weights"] = str(data_dir / "bridge_all_view/head_all.npz")
    visual["support_weights"] = str(data_dir / "training_support/support_bank.npz")
    visual["save_vectors"] = False
    config["risk_gate"]["enabled"] = args.mode != "2b_only"
    review = config["selective_review"]
    review["enabled"] = args.mode == "risk_gate_teacher"
    if review["enabled"]:
        review["base_model"] = str(args.teacher_model)
        review["adapter"] = str(args.adapter)
    config["tongagents"]["enabled"] = args.tongagents
    return config


def prepare_runtime(args):
    os.environ.update({
        "TRACK4_FINAL_OFFLINE": "1",
        "TRACK4_DISABLE_SECRETS": "1",
        "TRACK4_RETRIEVAL_BACKEND": "local",
        "TRACK4_LOCAL_QWEN_VL_DEVICE": args.device,
        "TRACK4_OFFICIAL_EMBEDDING_MODEL_ID": "Qwen3-VL-Embedding-2B",
        "TRACK4_LOCAL_IMAGE_BATCH": "4",
        "TRACK4_EMBED_PREP_WORKERS": "8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    if args.embedding_model:
        verify_assets("embedding_2b", args.embedding_model, hash_large=False)
        os.environ["TRACK4_LOCAL_QWEN_VL_EMBEDDING_PATH"] = str(args.embedding_model)
        if not args.check_only and args.device == "cuda":
            import torch

            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("需要可用的 CUDA/BF16，或使用 --device cpu。")
    if args.tongagents:
        try:
            installed = version("tongagents")
        except PackageNotFoundError:
            installed = "未安装"
        if installed != "2.7.19":
            raise RuntimeError(f"需要 TongAgents 2.7.19（当前 {installed}），安装方法见 REPRODUCIBILITY.md。")
    if args.check_only and args.mode == "risk_gate_teacher":
        verify_assets("teacher_27b", args.teacher_model, hash_large=False)
        verify_assets("adapter", args.adapter)


def runtime_versions(real_embedding: bool, teacher: bool, tongagents: bool) -> dict:
    names = ["numpy", "Pillow", "PyYAML"]
    if real_embedding:
        names.extend(("torch", "torchvision", "transformers", "qwen-vl-utils"))
    if teacher:
        names.append("peft")
    if tongagents:
        names.append("tongagents")
    versions = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return {"python": platform.python_version(), **versions}


def run(args):
    start = time.perf_counter()
    sys.path.insert(0, str(AGENT_ROOT / "src"))
    from track4_contract_agent.runtime.full_visual import checked_inputs, run_full_visual

    entry = competition_entry()
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"请使用新的输出目录，已有目录为 {output}")
    if args.input_dir and (output == args.input_dir or args.input_dir in output.parents):
        raise ValueError("输出目录需要放在输入目录之外。")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".track4-", dir=output.parent) as name:
        scratch = Path(name)
        encoder = None
        support_scope = "official_training_images_only"
        if args.demo:
            from examples.demo_fixture import PixelFixtureEncoder, make_demo

            input_dir, metadata_file, data_dir = make_demo(scratch)
            if args.embedding_model is None:
                encoder = PixelFixtureEncoder()
            support_scope = "synthetic_fixture"
        else:
            input_dir, metadata_file, data_dir = args.input_dir, args.metadata, args.assets
            verify_assets("learned", data_dir)
        rows = prepare_metadata(entry, input_dir, metadata_file)
        checked_inputs(rows, input_dir)
        prepare_runtime(args)
        if args.check_only:
            print(f"校验通过：{len(rows)} 张图及其元数据。")
            return
        metadata = scratch / "runtime_metadata.json"
        write_json(metadata, rows)
        output_tmp = scratch / "output"
        config = configuration(args, data_dir)
        info = run_full_visual(
            input_dir, metadata, output_tmp, config, AGENT_ROOT,
            data_dir=data_dir, image_backend=encoder, use_tongagents=args.tongagents,
            support_scope=support_scope,
        )
        result = output_tmp / "result.json"
        location_count = entry.fill_defect_location(result) if args.fill_defect_location else 0
        run_id = uuid.uuid4().hex
        seconds = time.perf_counter() - start
        timing = output_tmp / "infer_time.json"
        write_json(timing, {"infer_time": seconds, "inter_time": seconds, "run_id": run_id})
        entry.validate_run(metadata, output_tmp, input_dir)
        receipt = {
            "run_id": run_id,
            "mode": args.mode,
            "profile": args.profile,
            "demo": args.demo,
            "image_backend": "synthetic_pixel_encoder" if encoder is not None else "Qwen3-VL-Embedding-2B",
            "device": "cpu" if encoder is not None else args.device,
            "versions": runtime_versions(encoder is None, args.mode == "risk_gate_teacher", args.tongagents),
            "asset_scope": support_scope,
            "n_input_images": len(rows),
            "n_embedded": info["n_embedded"],
            "defect_location_filled_rows": location_count,
            "config_sha256": sha256(AGENT_ROOT / "configs" / PROFILES[args.profile]),
            "metadata_sha256": sha256(metadata),
            "result_sha256": sha256(result),
            "infer_time_sha256": sha256(timing),
            "risk_gate": info.get("risk_gate"),
            "selective_review": info.get("selective_review"),
            "tongagents": info["tongagents"],
        }
        if args.embedding_model is not None:
            receipt["embedding_revision"] = asset_group("embedding_2b")["revision"]
        write_json(output_tmp / "run_manifest.json", receipt)
        if output.exists():
            raise FileExistsError(f"输出目录已被另一个进程创建 {output}")
        output_tmp.rename(output)
    print(f"{len(rows)} 张图片处理完成 · {receipt['image_backend']} · {seconds:.3f} 秒")
    print(output / "result.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="用两张合成图片和示例分类资产验证管线")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--assets", type=Path, help="包含三份 NPZ 和配套 JSON 的数据目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("2b_only", "risk_gate", "risk_gate_teacher"), default="risk_gate")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="onsite")
    parser.add_argument("--teacher-model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--tongagents", action="store_true")
    parser.add_argument("--fill-defect-location", action="store_true", help="采用归档入口的文件名位置后处理")
    parser.add_argument("--check-only", action="store_true", help="检查输入和依赖资产是否就绪")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if args.demo:
        if args.input_dir or args.metadata or args.assets or args.mode == "risk_gate_teacher":
            parser.error("--demo 使用自身的样例输入与学习资产，支持初判和风险门控。")
    elif not all((args.input_dir, args.assets, args.embedding_model)):
        parser.error("真实推理需要 --input-dir、--assets 和 --embedding-model。")
    if args.mode == "risk_gate_teacher":
        if not args.teacher_model or not args.adapter or args.device != "cuda" or platform.system() != "Linux":
            parser.error("27B 复核需要 Linux CUDA/PPU、--teacher-model 和 --adapter。")
    elif args.teacher_model or args.adapter:
        parser.error("--teacher-model 与 --adapter 用于 risk_gate_teacher 模式。")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
