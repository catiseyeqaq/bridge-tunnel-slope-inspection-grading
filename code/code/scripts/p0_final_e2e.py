#!/usr/bin/env python3
"""P0 final E2E receipt: no-arg/no-env run on the contest input under 16 CPU.

The container's existing cgroup has memory.limit_in_bytes = 64 GiB but the
cgroupfs is read-only, so a child cgroup cannot be created. This driver runs in
the existing 64 GiB cgroup, samples memory.usage_in_bytes, and records the
cgroup OOM counters. It does not display or export test images or per-row
predictions.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

BRANCH = Path("~/track4_risk_gate_v1")
OUTPUT = BRANCH / "result"
INPUT = Path("/dataset/决赛数据集1/赛题4/线上测试集")
TRUSTED_METADATA = Path("~//code/output/stageC_0904arch/runtime_test_metadata.json")
CGROUP = Path("/sys/fs/cgroup/memory")
REPORTS = BRANCH / "reports"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def sha_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_int(path: Path):
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def read_oom_kill() -> int:
    try:
        for line in (CGROUP / "memory.oom_control").read_text().splitlines():
            if line.startswith("oom_kill "):
                return int(line.split()[1])
    except Exception:
        pass
    return 0


def manifest_sha(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def metadata_map(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_rel, by_name = {}, {}
    for row in rows:
        category = str(row.get("questionCategory") or row.get("category") or "")
        rel = str(row.get("imageRelativePath") or "")
        filename = str(row.get("filename") or Path(rel).name)
        bridge = str(row.get("bridgeName") or "")
        by_rel[(category, rel)] = bridge
        by_name[(category, filename)] = bridge
    return rows, by_rel, by_name


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not INPUT.is_dir():
        raise SystemExit(f"contest input missing: {INPUT}")
    if not TRUSTED_METADATA.is_file():
        raise SystemExit(f"trusted metadata missing: {TRUSTED_METADATA}")

    cpu_probe = subprocess.run(
        ["taskset", "-c", "0-15", sys.executable, "-c",
         "import os; print(len(os.sched_getaffinity(0)))"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    images = [p for p in INPUT.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    input_manifest = manifest_sha(INPUT)

    mem_limit = read_int(CGROUP / "memory.limit_in_bytes")
    usage_before = read_int(CGROUP / "memory.usage_in_bytes")
    max_usage_before = read_int(CGROUP / "memory.max_usage_in_bytes")
    failcnt_before = read_int(CGROUP / "memory.failcnt")
    oom_before = read_oom_kill()

    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            value = read_int(CGROUP / "memory.usage_in_bytes")
            if value is not None:
                samples.append(value)
            stop.wait(0.5)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()

    env = {k: v for k, v in os.environ.items() if not k.startswith("TRACK4_")}
    command = ["taskset", "-c", "0-15", sys.executable, str(BRANCH / "code" / "run.py")]
    log_path = REPORTS / "p0_final_e2e.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=str(BRANCH), env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - started
    stop.set()
    thread.join(timeout=2)

    usage_after = read_int(CGROUP / "memory.usage_in_bytes")
    max_usage_after = read_int(CGROUP / "memory.max_usage_in_bytes")
    failcnt_after = read_int(CGROUP / "memory.failcnt")
    oom_after = read_oom_kill()
    sampled_peak = max(samples) if samples else None

    result_path = OUTPUT / "result.json"
    timing_path = OUTPUT / "infer_time.json"
    receipt = {
        "run_id": time.strftime("%Y%m%dT%H%M%S"),
        "command": command,
        "environment_policy": "no TRACK4_* variables; no CLI arguments",
        "exit_code": completed.returncode,
        "input": {
            "path": str(INPUT),
            "n_images": len(images),
            "manifest_sha256": input_manifest,
            "trusted_metadata": str(TRUSTED_METADATA),
        },
        "constraints": {
            "cpu_affinity_count": int(cpu_probe),
            "cpu_affinity_request": "0-15",
            "cgroup_memory_limit_bytes": mem_limit,
            "cgroup_memory_limit_GiB": round(mem_limit / 2**30, 3) if mem_limit else None,
            "cgroup_child_created": False,
            "cgroup_note": ("cgroupfs is read-only; the run used the container's existing "
                            "64 GiB cgroup. The sampled usage includes concurrent processes."),
        },
        "memory": {
            "usage_before_bytes": usage_before,
            "usage_after_bytes": usage_after,
            "sampled_peak_bytes": sampled_peak,
            "sampled_peak_GiB": round(sampled_peak / 2**30, 3) if sampled_peak else None,
            "cgroup_max_usage_before_bytes": max_usage_before,
            "cgroup_max_usage_after_bytes": max_usage_after,
            "oom_kill_before": oom_before,
            "oom_kill_after": oom_after,
            "oom_kill_delta": oom_after - oom_before,
            "failcnt_before": failcnt_before,
            "failcnt_after": failcnt_after,
            "samples": len(samples),
            "samples_median_bytes": int(statistics.median(samples)) if samples else None,
        },
        "timing": {
            "process_wall_s": wall,
            "entry_infer_time_s": None,
            "prior_timings_not_merged": [80.05842542648315, 83.10911250114441, 77.49555659294128],
        },
        "output": {
            "result_path": str(result_path),
            "infer_time_path": str(timing_path),
            "result_sha256": None,
            "infer_time_sha256": None,
            "n_rows": None,
            "n_embedded": None,
            "n_skipped_visual": None,
            "bridge_disease": None,
            "rail_disease": None,
        },
        "bridge_identity_check": {},
        "assets": {},
        "no_original_images_displayed_or_exported": True,
        "platform_submitted": False,
    }

    if completed.returncode == 0 and result_path.is_file() and timing_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        receipt["timing"]["entry_infer_time_s"] = timing.get("infer_time")
        receipt["output"].update({
            "result_sha256": sha_file(result_path),
            "infer_time_sha256": sha_file(timing_path),
            "n_rows": len(result),
            "bridge_disease": sum(1 for row in result
                                  if row["questionCategory"] == "桥梁" and row["defectType"] != "完好"),
            "rail_disease": sum(1 for row in result
                                if row["questionCategory"] == "轨道" and row["defectType"] != "完好"),
        })
        runs = sorted([p for p in (BRANCH / "code" / ".runs").iterdir() if p.is_dir()])
        latest = runs[-1]
        heads = json.loads((latest / "heads_only.json").read_text(encoding="utf-8"))
        receipt["output"]["n_embedded"] = heads.get("n_embedded")
        receipt["output"]["n_skipped_visual"] = heads.get("n_skipped_visual")
        generated_metadata = latest / "runtime_metadata.json"
        receipt["input"]["generated_metadata"] = str(generated_metadata)
        receipt["input"]["generated_metadata_sha256"] = sha_file(generated_metadata)
        generated_rows, generated_rel, generated_name = metadata_map(generated_metadata)
        trusted_rows, trusted_rel, trusted_name = metadata_map(TRUSTED_METADATA)
        matched = mismatched = missing = 0
        mismatch_indices = []
        for index, row in enumerate(generated_rows):
            if row["questionCategory"] != "桥梁":
                continue
            rel = row["imageRelativePath"]
            filename = row["filename"]
            expected = trusted_rel.get(("桥梁", rel)) or trusted_name.get(("桥梁", filename))
            if expected is None:
                missing += 1
                mismatch_indices.append(index)
            elif expected != row["bridgeName"]:
                mismatched += 1
                mismatch_indices.append(index)
            else:
                matched += 1
        receipt["bridge_identity_check"] = {
            "generated_rows": len(generated_rows),
            "trusted_rows": len(trusted_rows),
            "bridge_rows": sum(1 for row in generated_rows if row["questionCategory"] == "桥梁"),
            "matched": matched,
            "mismatched": mismatched,
            "missing": missing,
            "mismatch_row_indices": mismatch_indices[:10],
            "all_bridge_identities_match": missing == 0 and mismatched == 0,
        }

    base = Path("~/models/Qwen3-VL-Embedding-2B/model.safetensors")
    heads = {
        "bridge_head": BRANCH / "code/code/data/bridge_all_view/head_all.npz",
        "rail_head": BRANCH / "code/code/data/rail_head/rail_head_v1.npz",
        "support_bank": BRANCH / "code/code/data/training_support/support_bank.npz",
    }
    receipt["assets"] = {
        "embedding_2b": {
            "path": str(base), "realpath": str(base.resolve()),
            "persistent_under_workspace": str(base.resolve()).startswith("~//"),
            "sha256": sha_file(base) if base.is_file() else None,
            "bytes": base.stat().st_size if base.is_file() else None,
        },
        **{name: {
            "path": str(path), "realpath": str(path.resolve()),
            "persistent_under_workspace": str(path.resolve()).startswith("~//"),
            "sha256": sha_file(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        } for name, path in heads.items()},
    }

    receipt_path = REPORTS / "P0_FINAL_E2E_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "exit_code": receipt["exit_code"],
        "entry_infer_time_s": receipt["timing"]["entry_infer_time_s"],
        "process_wall_s": receipt["timing"]["process_wall_s"],
        "result_sha256": receipt["output"]["result_sha256"],
        "n_rows": receipt["output"]["n_rows"],
        "n_embedded": receipt["output"]["n_embedded"],
        "bridge_identity": receipt["bridge_identity_check"],
        "sampled_peak_GiB": receipt["memory"]["sampled_peak_GiB"],
        "oom_kill_delta": receipt["memory"]["oom_kill_delta"],
        "receipt": str(receipt_path),
    }, ensure_ascii=False, indent=2))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
