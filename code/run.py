#!/usr/bin/env python3
"""Track4 stable entry: 2B full-image inspection + optional severe-risk 27B review.

========================= HOW TO CHANGE THE DATASET (数据集换路径就看这里) =========================

1) Recommended: pass the new dataset root explicitly

   python code/run.py --input-dir "/dataset/决赛数据集2/赛题4/线上测试集" --output-dir ~/track4/runs/finals2

   - --input-dir must contain 桥梁/ and 轨道/ subdirectories with images.
   - --output-dir is created automatically; keep it OUTSIDE the input directory.
   - Each run archives the previous result pair under code/.runs/<run_id>/previous_result/
     and atomically overwrites result.json + infer_time.json.

2) If you omit --input-dir

   - TRACK4_INPUT_DIR is used first:  TRACK4_INPUT_DIR="/dataset/.../线上测试集" python code/run.py
   - Otherwise /dataset and /data are scanned for roots containing 桥梁/ + 轨道/.
   - Exactly one candidate: use it. Multiple candidates: refuse to guess and exit.
     In that case pass --input-dir or set TRACK4_INPUT_DIR (recommended for a new path).

3) Metadata / bridgeName (only the 桥梁 rows need it)

   - Preferred: place metadata.json in the input root, or pass --metadata PATH.
     Each row needs: questionCategory, bridgeName, filename, imageRelativePath.
   - Alternatively organize images as 桥梁/<bridgeName>/... (name comes from the folder).
   - Or provide asset_context.json with {"bridgeName": "..."}, or set TRACK4_BRIDGE_NAME.
   - If none of the above and images are flat under 桥梁/, the run CONTINUES with
     bridgeName=<image stem> (the visible image name without extension) so the batch
     still completes. This is an image-name fallback, not a verified bridge identity.

4) Environment variables (all optional)

   TRACK4_INPUT_DIR, TRACK4_METADATA, TRACK4_OUTPUT_DIR, TRACK4_CONFIG,
   TRACK4_BRIDGE_NAME, TRACK4_METADATA_SOURCES, TRACK4_PYTHON,
   TRACK4_LOCAL_QWEN_VL_EMBEDDING_PATH

5) Defaults

   config: code/code/configs/submission_risk_gate_teacher.yaml
   output: <package>/result
   models: ~/track4_current/... first (fast local NVMe), then ~/ backups.
   After a reboot restore them with: bash ~/track4/reports/restore_models_to_root.sh

6) More examples

   python code/run.py --input-dir /data/new_test --output-dir ~/track4/runs/new_test
   python code/run.py --help          # prints this whole guide

7) Package result mirroring (the system packages ~/result)

   - Every successful non-dry run also mirrors result.json + infer_time.json into
     <package>/result (normally ~/result/result), so the packaged result
     always matches the latest run even when --output-dir points elsewhere.
   - The previous package pair is archived under
     code/.runs/<run_id>/previous_result_package/ for rollback.
   - Use --no-mirror for rehearsals that must not touch the package result.

8) defectLocation fill (output-only)

   - Empty bridge defectLocation values are filled from component-position
     filenames such as 第N跨 / N号墩 / 跨中梁底 / 墩支座. DJI-only and rail rows
     stay empty. The 左幅_/右幅_ prefix is kept.
   - This changes only defectLocation; type/description/rating/bridgeName and all
     model decisions are untouched. Run output reports defect_location_filled_rows.

Behavior of this entry:
  resolves the 2B model, builds metadata from the real input identities, runs the
  inner pipeline in a fresh .runs/<run_id> directory, validates coverage/contract,
  then atomically publishes result.json and infer_time.json from the SAME run.
  The outer entry does not load the 27B teacher itself; the inner pipeline starts it
  only for severe-risk selected rows and keeps the legal 2B result on any failure.
"""
from __future__ import annotations

import time

begin_time = time.time()

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parent          # .../track4_risk_gate_v1/code
BRANCH_ROOT = PACKAGE_ROOT.parent                       # .../track4_risk_gate_v1
INNER_ENTRY = PACKAGE_ROOT / "code" / "run.py"
DEFAULT_CONFIG = PACKAGE_ROOT / "code" / "configs" / "submission_risk_gate_teacher.yaml"
DEFAULT_OUTPUT = BRANCH_ROOT / "result"
RUN_ROOT = PACKAGE_ROOT / ".runs"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SEVEN = ("questionCategory", "bridgeName", "defectLocation", "filename",
         "defectType", "defectDescription", "ratingScale(1-5)")
DEFECT_LOCATION_PATTERNS = (
    r"第\s*\d+\s*跨", r"\d+\s*号墩", r"跨中", r"梁底", r"支座",
    r"盖梁", r"桥面", r"伸缩缝",
)


def log(message: str) -> None:
    print(f"[entry] {message}", flush=True)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _image_files(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


METADATA_NAMES = ("metadata.json", "runtime_test_metadata.json", "test_metadata.json")


def _is_test_root(path: Path) -> bool:
    return (path / "桥梁").is_dir() and (path / "轨道").is_dir()


def _candidate_input_roots(roots: Iterable[Path]) -> List[Path]:
    """Find roots that contain both 桥梁/ and 轨道/ with images.

    Training/label trees are excluded. The scan never uses a dataset name or an
    image count to choose a split: that would silently prefer the old 330 set.
    """
    found: List[Path] = []
    for base in roots:
        base = Path(base)
        if not base.is_dir():
            continue
        for root, dirs, _files in os.walk(base):
            current = Path(root)
            rel = current.relative_to(base)
            if len(rel.parts) > 5:
                dirs[:] = []
                continue
            lowered = {part.lower() for part in current.parts}
            if ("训练" in current.parts or "初赛" in current.parts
                    or "train" in lowered or "label" in lowered or "labels" in lowered):
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs
                       if not d.startswith(".")
                       and d not in {"桥梁", "轨道"}
                       and "训练" not in d
                       and "train" not in d.lower()
                       and "label" not in d.lower()]
            if _is_test_root(current):
                if _image_files(current / "桥梁") and _image_files(current / "轨道"):
                    found.append(current.resolve())
                    dirs[:] = []
    return sorted(set(found))


def find_input_dir(explicit: str, roots: Optional[Iterable[Path]] = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"input dir not found: {path}")
        if not _image_files(path):
            raise FileNotFoundError(f"input dir contains no images: {path}")
        return path
    env = str(os.environ.get("TRACK4_INPUT_DIR") or "").strip()
    if env:
        path = Path(env).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"TRACK4_INPUT_DIR not found: {path}")
        if not _image_files(path):
            raise FileNotFoundError(f"TRACK4_INPUT_DIR contains no images: {path}")
        return path
    search_roots = list(roots) if roots is not None else [Path("/dataset"), Path("/data")]
    found = _candidate_input_roots(search_roots)
    if not found:
        raise FileNotFoundError(
            "No contest input root with 桥梁/ and 轨道/ images found under "
            f"{[str(p) for p in search_roots]}. Pass --input-dir explicitly."
        )
    if len(found) == 1:
        return found[0]
    # A unique metadata marker is the only allowed tie-breaker. Directory name
    # or image count is never used: choosing online330 silently is forbidden.
    marked = [p for p in found
              if any((p / name).is_file() for name in METADATA_NAMES)]
    if len(marked) == 1:
        log(f"input root: multiple candidates, selected the only root with metadata: {marked[0]}")
        return marked[0]
    raise RuntimeError(
        "Multiple contest input roots found: " + ", ".join(str(p) for p in found)
        + ". Refusing to guess; pass --input-dir explicitly."
    )


def _normalise_metadata_rows(rows: Iterable[Dict[str, Any]], input_dir: Path,
                             nested_bridge: Dict[str, str],
                             asset_bridge: str) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for index, raw in enumerate(rows):
        category = str(raw.get("questionCategory") or raw.get("category") or "").strip()
        if category not in ("桥梁", "轨道"):
            raise ValueError(f"metadata row {index} has unsupported category: {category!r}")
        rel = str(raw.get("imageRelativePath") or raw.get("relative_path") or "").strip()
        filename = str(raw.get("filename") or "").strip()
        if not rel:
            if not filename:
                raise ValueError(f"metadata row {index} has neither imageRelativePath nor filename")
            rel = f"{category}/{filename}"
        if not filename:
            filename = Path(rel).name
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"metadata row {index} has unsafe imageRelativePath: {rel}")
        if not (input_dir / rel_path).is_file():
            raise FileNotFoundError(f"metadata row {index} image is missing: {rel}")
        bridge_name = str(raw.get("bridgeName") or "").strip()
        if category == "桥梁" and not bridge_name:
            parts = rel_path.parts
            if len(parts) >= 3 and parts[0] == "桥梁":
                bridge_name = parts[1]
            elif str(rel_path.parent) in nested_bridge:
                bridge_name = nested_bridge[str(rel_path.parent)]
            elif asset_bridge:
                bridge_name = asset_bridge
            elif str(os.environ.get("TRACK4_BRIDGE_NAME") or "").strip():
                bridge_name = str(os.environ["TRACK4_BRIDGE_NAME"]).strip()
            else:
                raise ValueError(
                    f"bridgeName is missing for {rel}. Supply metadata.json / "
                    "asset_context.json / TRACK4_BRIDGE_NAME; no hard-coded bridge is used."
                )
        output.append({
            "questionCategory": category,
            "bridgeName": bridge_name if category == "桥梁" else "",
            "defectLocation": str(raw.get("defectLocation") or ""),
            "filename": filename,
            "imageRelativePath": rel_path.as_posix(),
        })
    if not output:
        raise FileNotFoundError("No supported images found in input metadata")
    return output


def _metadata_file_candidates(input_dir: Path) -> List[Path]:
    return [
        input_dir / "metadata.json",
        input_dir / "runtime_test_metadata.json",
        input_dir / "test_metadata.json",
    ]


def _discover_bridge_names(input_dir: Path, bridge_dir: Path,
                           sources: Optional[Iterable[Path]] = None) -> Optional[Dict[str, str]]:
    """Read bridge names ONLY by exact imageRelativePath identity.

    A filename-only match is deliberately not allowed: an old online metadata
    file can contain the same basename as a new offline image. If the source
    does not cover every current bridge image by full relative path, it is
    rejected unless it is the unique metadata marker.
    """
    if sources is None:
        candidates: List[Path] = []
        env_sources = str(os.environ.get("TRACK4_METADATA_SOURCES") or "").strip()
        if env_sources:
            candidates.extend(Path(item).expanduser() for item in env_sources.split(os.pathsep) if item.strip())
        for parent in (input_dir, input_dir.parent):
            for name in METADATA_NAMES:
                candidates.append(parent / name)
        candidates.extend([
            Path("~//code/output/stageC_0904arch/runtime_test_metadata.json"),
            Path("~//code/data/infer/online_test_metadata.json"),
            Path("/data/track4/metadata.json"),
        ])
        sources = candidates
    images = _image_files(bridge_dir)
    if not images:
        return {}
    for source in dict.fromkeys(Path(p) for p in sources):
        if not source.is_file():
            continue
        try:
            rows = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        by_rel: Dict[str, str] = {}
        for raw in rows:
            category = str(raw.get("questionCategory") or raw.get("category") or "").strip()
            if category != "桥梁":
                continue
            name = str(raw.get("bridgeName") or "").strip()
            rel = str(raw.get("imageRelativePath") or raw.get("relative_path") or "").strip()
            if name and rel:
                by_rel[rel] = name
        discovered: Dict[str, str] = {}
        for image in images:
            rel = image.relative_to(input_dir).as_posix()
            name = by_rel.get(rel)
            if not name:
                break
            discovered[rel] = name
        if len(discovered) == len(images):
            log(f"bridge names: matched {len(discovered)} exact identities from {source}")
            return discovered
    return None


def _validate_metadata_coverage(input_dir: Path, rows: List[Dict[str, Any]]) -> None:
    images = _image_files(input_dir)
    rels = [str(row.get("imageRelativePath") or "") for row in rows]
    if len(rows) != len(images):
        raise ValueError(
            f"metadata rows {len(rows)} != input images {len(images)} under {input_dir}"
        )
    if len(set(rels)) != len(rels):
        raise ValueError("metadata contains duplicate imageRelativePath rows")
    missing = [image.relative_to(input_dir).as_posix()
               for image in images if image.relative_to(input_dir).as_posix() not in set(rels)]
    if missing:
        raise ValueError(
            f"metadata does not cover {len(missing)} input images (first: {missing[:3]})"
        )


def build_metadata(input_dir: Path, temp_dir: Path, explicit: str) -> Path:
    env = str(os.environ.get("TRACK4_METADATA") or "").strip()
    chosen = explicit or env
    if chosen:
        path = Path(chosen).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"metadata not found: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("metadata must be a JSON array")
        nested: Dict[str, str] = {}
        bridge_dir = input_dir / "桥梁"
        if bridge_dir.is_dir():
            for child in sorted(bridge_dir.iterdir()):
                if child.is_dir():
                    nested[child.name] = child.name
        asset_bridge = ""
        context = input_dir / "asset_context.json"
        if context.is_file():
            asset_bridge = str(json.loads(context.read_text(encoding="utf-8")).get("bridgeName") or "")
        normalised = _normalise_metadata_rows(rows, input_dir, nested, asset_bridge)
        _validate_metadata_coverage(input_dir, normalised)
        dest = temp_dir / "runtime_metadata.json"
        dest.write_text(json.dumps(normalised, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log(f"metadata: explicit {path} -> {dest} (n={len(normalised)})")
        return dest

    for candidate in _metadata_file_candidates(input_dir):
        if candidate.is_file():
            return build_metadata(input_dir, temp_dir, str(candidate))

    nested: Dict[str, str] = {}
    bridge_dir = input_dir / "桥梁"
    if bridge_dir.is_dir():
        for child in sorted(bridge_dir.iterdir()):
            if child.is_dir():
                nested[child.name] = child.name
    asset_bridge = ""
    context = input_dir / "asset_context.json"
    if context.is_file():
        asset_bridge = str(json.loads(context.read_text(encoding="utf-8")).get("bridgeName") or "")

    rows: List[Dict[str, Any]] = []
    if bridge_dir.is_dir():
        subdirs = [p for p in sorted(bridge_dir.iterdir()) if p.is_dir()]
        if subdirs:
            for bridge in subdirs:
                for image in _image_files(bridge):
                    rel = image.relative_to(input_dir)
                    rows.append({"questionCategory": "桥梁", "bridgeName": bridge.name,
                                 "filename": image.name, "imageRelativePath": rel.as_posix()})
        else:
            bridge_name = asset_bridge or str(os.environ.get("TRACK4_BRIDGE_NAME") or "").strip()
            if bridge_name:
                for image in _image_files(bridge_dir):
                    rel = image.relative_to(input_dir)
                    rows.append({"questionCategory": "桥梁", "bridgeName": bridge_name,
                                 "filename": image.name, "imageRelativePath": rel.as_posix()})
            else:
                discovered = _discover_bridge_names(input_dir, bridge_dir)
                if not discovered:
                    # Fallback: a new dataset may be flat under 桥梁/ and ship no
                    # metadata. Use each image's visible name (stem) as bridgeName
                    # so the seven-field row remains filled and image-aligned.
                    log("WARNING: bridge name unknown for flat 桥梁/ images; using each "
                        "image stem as bridgeName (image-name fallback). Supply metadata.json / "
                        "asset_context.json / TRACK4_BRIDGE_NAME for exact bridgeName.")
                    for image in _image_files(bridge_dir):
                        rel = image.relative_to(input_dir)
                        rows.append({"questionCategory": "桥梁", "bridgeName": image.stem,
                                     "filename": image.name, "imageRelativePath": rel.as_posix()})
                else:
                    for image in _image_files(bridge_dir):
                        rel = image.relative_to(input_dir).as_posix()
                        rows.append({"questionCategory": "桥梁", "bridgeName": discovered[rel],
                                     "filename": image.name, "imageRelativePath": rel})
    rail_dir = input_dir / "轨道"
    if rail_dir.is_dir():
        for image in _image_files(rail_dir):
            rel = image.relative_to(input_dir)
            rows.append({"questionCategory": "轨道", "bridgeName": "",
                         "filename": image.name, "imageRelativePath": rel.as_posix()})
    if not rows:
        raise FileNotFoundError(f"No images found under {input_dir}/桥梁 or {input_dir}/轨道")
    normalised = _normalise_metadata_rows(rows, input_dir, nested, asset_bridge)
    _validate_metadata_coverage(input_dir, normalised)
    dest = temp_dir / "runtime_metadata.json"
    dest.write_text(json.dumps(normalised, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"metadata: built from input identities -> {dest} (n={len(normalised)})")
    return dest


def _python_works(path: str) -> bool:
    if not path or not os.access(path, os.X_OK):
        return False
    try:
        result = subprocess.run(
            [path, "-c", "import torch,transformers,yaml,sklearn,numpy,PIL"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
        )
        return result.returncode == 0
    except Exception:
        return False


def choose_python() -> str:
    candidates = [
        str(os.environ.get("TRACK4_PYTHON") or "").strip(),
        sys.executable,
        "~/track4_current/venv/bin/python",
        "/usr/local/bin/python3",
        shutil.which("python3") or "",
    ]
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if _python_works(candidate):
            log(f"python: {candidate}")
            return candidate
    raise RuntimeError(
        "No Python interpreter with torch/transformers/yaml/sklearn/numpy/PIL was found. "
        "Set TRACK4_PYTHON to a working interpreter."
    )


def _model_ok(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    if not (path / "scripts" / "qwen3_vl_embedding.py").is_file():
        return False
    if (path / "model.safetensors").is_file():
        return True
    return bool(list(path.glob("model-*.safetensors")))


def choose_model() -> Path:
    candidates = [
        str(os.environ.get("TRACK4_LOCAL_QWEN_VL_EMBEDDING_PATH") or "").strip(),
        str(PACKAGE_ROOT / "model_assets" / "embedding_2b"),
        str(PACKAGE_ROOT / "code" / "model_assets" / "embedding_2b"),
        str(PACKAGE_ROOT / "code" / "model_assets" / "bases" / "embedding_2b"),
        "~/track4_current/code/model_assets/bases/embedding_2b",
        "~/track4_current/code/model_assets/bases/embedding_2b",
        "~/models/Qwen3-VL-Embedding-2B",
        "/data/model/Qwen3-VL-Embedding-2B",
        "/model/Qwen3-VL-Embedding-2B",
    ]
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        path = Path(candidate).expanduser().resolve()
        if _model_ok(path):
            log(f"2B model: {path}")
            return path
    raise FileNotFoundError(
        "2B embedding model not found. Checked: " + ", ".join(seen) +
        ". Bundle it under code/model_assets/embedding_2b or set "
        "TRACK4_LOCAL_QWEN_VL_EMBEDDING_PATH."
    )


def run_inner(python: str, input_dir: Path, metadata: Path, temp_dir: Path,
              config: Path, model_dir: Path) -> Tuple[int, Path]:
    env = os.environ.copy()
    env.update({
        "TRACK4_FINAL_OFFLINE": "1",
        "TRACK4_DISABLE_SECRETS": "1",
        "TRACK4_RETRIEVAL_BACKEND": "local",
        "TRACK4_LOCAL_QWEN_VL_DEVICE": "cuda",
        "TRACK4_LOCAL_QWEN_VL_EMBEDDING_PATH": str(model_dir),
        "CUDA_VISIBLE_DEVICES": "0",
        "TRACK4_LOCAL_IMAGE_BATCH": "4",
        "TRACK4_EMBED_PREP_WORKERS": "8",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
    })
    log_path = temp_dir / "run.log"
    command = [
        python, str(INNER_ENTRY),
        "--input-dir", str(input_dir),
        "--metadata", str(metadata),
        "--output-dir", str(temp_dir),
        "--config", str(config),
        "--skip-server",
    ]
    log("command: " + " ".join(command))
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=str(PACKAGE_ROOT / "code"), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            stream.write(line)
            stream.flush()
        code = process.wait(timeout=1800)
    return code, log_path


def fill_defect_location(result_path: Path) -> int:
    """Fill empty bridge defectLocation from component-position filenames.

    Output post-process only: type/description/rating and all model decisions are
    untouched. Rail rows and DJI-only filenames remain empty. The side prefix
    (左幅_/右幅_) is kept in the location string.
    """
    rows = json.loads(result_path.read_text(encoding="utf-8"))
    changed = 0
    for row in rows:
        if row.get("questionCategory") != "桥梁":
            continue
        if str(row.get("defectLocation") or "").strip():
            continue
        stem = Path(str(row.get("filename") or "")).stem
        normalized = re.sub(r"\s*\(\d+\)\s*$", "", stem).strip()
        if normalized and any(re.search(p, normalized) for p in DEFECT_LOCATION_PATTERNS):
            row["defectLocation"] = normalized
            changed += 1
    if changed:
        result_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    log(f"defectLocation filled: {changed} bridge rows")
    return changed


def validate_run(metadata_path: Path, temp_dir: Path, input_dir: Optional[Path] = None) -> Dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if input_dir is not None:
        input_images = _image_files(input_dir)
        if len(metadata) != len(input_images):
            raise RuntimeError(
                f"metadata rows {len(metadata)} != input images {len(input_images)}"
            )
    result_path = temp_dir / "result.json"
    timing_path = temp_dir / "infer_time.json"
    heads_path = temp_dir / "heads_only.json"
    for path in (result_path, timing_path, heads_path):
        if not path.is_file():
            raise FileNotFoundError(f"inner run did not produce {path.name}")
    rows = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != len(metadata):
        raise RuntimeError(f"result row count {len(rows)} != metadata rows {len(metadata)}")
    for index, (row, meta) in enumerate(zip(rows, metadata)):
        if tuple(row) != SEVEN:
            raise RuntimeError(f"row {index} does not have exactly the seven fields")
        if row["filename"] != meta["filename"] or row["questionCategory"] != meta["questionCategory"]:
            raise RuntimeError(f"row {index} identity mismatch")
        if not all(isinstance(value, str) for value in row.values()):
            raise RuntimeError(f"row {index} has a non-string field")
    heads = json.loads(heads_path.read_text(encoding="utf-8"))
    n_rows = int(heads.get("n_rows", -1))
    n_embedded = int(heads.get("n_embedded", -1))
    n_skipped = int(heads.get("n_skipped_visual", -1))
    if n_rows != len(metadata) or n_embedded != len(metadata) or n_skipped != 0:
        raise RuntimeError(
            f"2B coverage failed: n_rows={n_rows}, n_embedded={n_embedded}, "
            f"n_skipped_visual={n_skipped}, metadata={len(metadata)}"
        )
    return {"rows": rows, "heads": heads}


def _publish_pair(source_dir: Path, dest_dir: Path, run_id: str, archive_name: str) -> None:
    """Atomically replace one result pair, archiving the previous pair first."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    old_result = dest_dir / "result.json"
    if old_result.is_file():
        backup = RUN_ROOT / run_id / archive_name
        backup.mkdir(parents=True, exist_ok=True)
        for name in ("result.json", "infer_time.json"):
            source = dest_dir / name
            if source.is_file():
                shutil.copy2(source, backup / name)
        log(f"previous result archived in {backup}")
    for name in ("result.json", "infer_time.json"):
        source = source_dir / name
        temporary = dest_dir / (name + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, dest_dir / name)


def publish(temp_dir: Path, output_dir: Path, run_id: str,
            mirror_package: bool = True) -> Tuple[Path, Path]:
    """Publish the run pair and mirror it into <package>/result by default.

    The evaluation packages ~/result contents, so the package result
    must always carry the latest successful run. The previous package pair is
    archived under code/.runs/<run_id>/previous_result_package/ for rollback.
    """
    _publish_pair(temp_dir, output_dir, run_id, "previous_result")
    package_dir = DEFAULT_OUTPUT.resolve()
    if mirror_package and output_dir.resolve() != package_dir:
        _publish_pair(temp_dir, package_dir, run_id, "previous_result_package")
        log(f"package result mirrored to {package_dir}")
    return output_dir / "result.json", output_dir / "infer_time.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input-dir", default="",
        help="New dataset root containing 桥梁/ and 轨道/ with images. "
             "Default: TRACK4_INPUT_DIR, then auto-scan /dataset and /data.")
    parser.add_argument(
        "--metadata", default="",
        help="Optional metadata JSON (questionCategory/bridgeName/filename/imageRelativePath). "
             "Default: metadata.json/runtime_test_metadata.json/test_metadata.json in the input root.")
    parser.add_argument(
        "--output-dir", default="",
        help="Output directory for result.json and infer_time.json. "
             "Default: TRACK4_OUTPUT_DIR or <package>/result. Must be outside the input directory. "
             "The pair is also mirrored to <package>/result unless --no-mirror is set.")
    parser.add_argument(
        "--config", default="",
        help="Config YAML. Default: code/code/configs/submission_risk_gate_teacher.yaml.")
    parser.add_argument("--skip-server", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-mirror", action="store_true",
        help="Do not mirror the result pair into <package>/result. "
             "Use this for rehearsals that must not touch the system-packaged result.")
    args = parser.parse_args()

    input_dir = find_input_dir(args.input_dir)
    config = Path(args.config or os.environ.get("TRACK4_CONFIG") or DEFAULT_CONFIG).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"config not found: {config}")
    output_dir = Path(
        args.output_dir or os.environ.get("TRACK4_OUTPUT_DIR") or DEFAULT_OUTPUT
    ).expanduser().resolve()

    run_id = time.strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}"
    temp_dir = (RUN_ROOT / run_id).resolve()
    temp_dir.mkdir(parents=True, exist_ok=False)
    log(f"run_id: {run_id}")
    log(f"input: {input_dir}")
    log(f"config: {config}")
    log(f"output: {output_dir}")

    metadata = build_metadata(input_dir, temp_dir, args.metadata)
    python = choose_python()
    model_dir = choose_model()
    if args.dry_run:
        import yaml
        cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        rows = json.loads(metadata.read_text(encoding="utf-8"))
        print(json.dumps({
            "dry_run": True,
            "run_mode": cfg.get("run_mode"),
            "risk_gate": bool((cfg.get("risk_gate") or {}).get("enabled")),
            "selective_review": bool((cfg.get("selective_review") or {}).get("enabled")),
            "input_dir": str(input_dir),
            "metadata": str(metadata),
            "n_rows": len(rows),
            "output_dir": str(output_dir),
            "config": str(config),
            "python": python,
            "model_2b": str(model_dir),
            "teacher_checked": False,
        }, ensure_ascii=False, indent=2), flush=True)
        return 0
    inner_code, log_path = run_inner(python, input_dir, metadata, temp_dir, config, model_dir)
    if inner_code != 0:
        failure = {"run_id": run_id, "exit_code": inner_code, "stage": "inner",
                   "log": str(log_path), "finished_at": time.time()}
        (temp_dir / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
        log(f"FAILED inner exit={inner_code}; log={log_path}")
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
        print(tail, file=sys.stderr)
        return inner_code or 1

    filled_location = fill_defect_location(temp_dir / "result.json")
    validated = validate_run(metadata, temp_dir, input_dir)
    inner_timing = json.loads((temp_dir / "infer_time.json").read_text(encoding="utf-8"))
    inner_time = float(inner_timing.get("infer_time", 0.0))
    (temp_dir / "inner_infer_time.json").write_text(
        json.dumps({"infer_time": inner_time, "inter_time": inner_timing.get("inter_time", inner_time)},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    wall = time.time() - begin_time
    timing = {"infer_time": wall, "inter_time": wall, "run_id": run_id}
    (temp_dir / "infer_time.json").write_text(json.dumps(timing, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    result_path, timing_path = publish(temp_dir, output_dir, run_id,
                                       mirror_package=not args.no_mirror)
    package_result = DEFAULT_OUTPUT.resolve() / "result.json"
    if args.no_mirror:
        log("package mirror disabled (--no-mirror)")
    elif output_dir.resolve() != DEFAULT_OUTPUT.resolve():
        log(f"package result: {package_result}")

    risk_summary = None
    risk_path = temp_dir / "risk_gate.json"
    if risk_path.is_file():
        risk_summary = json.loads(risk_path.read_text(encoding="utf-8")).get("batch")

    manifest = {
        "run_id": run_id,
        "mode": str(validated["heads"].get("run_mode") or ""),
        "input_dir": str(input_dir),
        "metadata": {"path": str(metadata), "sha256": sha256_file(metadata),
                     "rows": len(validated["rows"])},
        "config": {"path": str(config), "sha256": sha256_file(config)},
        "model_2b": {"path": str(model_dir),
                     "config_sha256": sha256_file(model_dir / "config.json"),
                     "weights_sha256": sha256_file(model_dir / "model.safetensors")
                     if (model_dir / "model.safetensors").is_file() else None},
        "python": python,
        "result_sha256": sha256_file(result_path),
        "infer_time_sha256": sha256_file(timing_path),
        "package_result": str(package_result) if not args.no_mirror else None,
        "mirror_package": not args.no_mirror,
        "infer_time_s": wall,
        "inner_infer_time_s": inner_time,
        "defect_location_filled_rows": filled_location,
        "n_rows": len(validated["rows"]),
        "n_input_images": len(_image_files(input_dir)),
        "n_embedded": int(validated["heads"].get("n_embedded", -1)),
        "n_skipped_visual": int(validated["heads"].get("n_skipped_visual", -1)),
        "risk_gate": risk_summary,
        "entry_sha256": sha256_file(Path(__file__).resolve()),
        "finished_at": time.time(),
        "log": str(log_path),
    }
    (temp_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    (temp_dir / "entrypoint_timing.json").write_text(json.dumps({
        "run_id": run_id, "begin_time": begin_time, "end_time": time.time(),
        "entry_elapsed_s": wall, "inner_inference_s": inner_time,
        "includes_metadata_model_resolution_and_publish": True,
        "python": python, "config": str(config), "model_2b": str(model_dir),
        "mirror_package": not args.no_mirror,
        "package_result": str(package_result) if not args.no_mirror else None,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "run_id": run_id,
        "mode": manifest["mode"],
        "n_rows": manifest["n_rows"],
        "n_embedded": manifest["n_embedded"],
        "infer_time_s": wall,
        "result_sha256": manifest["result_sha256"],
        "infer_time_sha256": manifest["infer_time_sha256"],
        "defect_location_filled_rows": filled_location,
        "result": str(result_path),
        "package_result": str(package_result) if not args.no_mirror else None,
        "mirror_package": not args.no_mirror,
        "audit_dir": str(temp_dir),
        "risk_gate": risk_summary,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[entry] FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
