#!/usr/bin/env python3
"""P2-A E2E receipt: gate -> preflight only. Never starts the 27B.

Runs the normal entry with a preflight config on a real 330-image set, samples
the container cgroup, and records the gate/review states. It does not display
or export images or per-row predictions.
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config = Path(args.config)
    if output_dir.exists():
        raise SystemExit(f"output dir already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    images = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]

    usage_before = read_int(CGROUP / "memory.usage_in_bytes")
    max_before = read_int(CGROUP / "memory.max_usage_in_bytes")
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
    command = ["taskset", "-c", "0-15", sys.executable, str(BRANCH / "code" / "run.py"),
               "--input-dir", str(input_dir), "--output-dir", str(output_dir),
               "--config", str(config)]
    if args.metadata:
        command.extend(["--metadata", args.metadata])
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRACK4_")}
    log_path = REPORTS / f"p2a_{args.name}.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=str(BRANCH), env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - started
    stop.set()
    thread.join(timeout=2)

    result_path = output_dir / "result.json"
    timing_path = output_dir / "infer_time.json"
    manifest_path = output_dir / "run_manifest.json"
    receipt = {
        "name": args.name,
        "phase": "P2-A_preflight_only",
        "real_teacher_run": False,
        "command": command,
        "config": {"path": str(config), "sha256": sha_file(config)},
        "input": {"path": str(input_dir), "n_images": len(images),
                  "manifest_sha256": manifest_sha(input_dir),
                  "metadata": args.metadata},
        "exit_code": completed.returncode,
        "timing": {"process_wall_s": wall, "entry_infer_time_s": None},
        "cgroup": {
            "limit_bytes": read_int(CGROUP / "memory.limit_in_bytes"),
            "usage_before": usage_before,
            "sampled_peak_bytes": max(samples) if samples else None,
            "usage_after": read_int(CGROUP / "memory.usage_in_bytes"),
            "max_usage_before": max_before,
            "max_usage_after": read_int(CGROUP / "memory.max_usage_in_bytes"),
            "oom_kill_before": oom_before,
            "oom_kill_after": read_oom_kill(),
            "failcnt_before": failcnt_before,
            "failcnt_after": read_int(CGROUP / "memory.failcnt"),
            "samples": len(samples),
            "median_bytes": int(statistics.median(samples)) if samples else None,
        },
        "output": {},
        "risk_gate": None,
        "selective_review": None,
        "actual_calls": None,
        "preflight": None,
    }
    if completed.returncode == 0 and result_path.is_file() and timing_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        receipt["timing"]["entry_infer_time_s"] = timing.get("infer_time")
        receipt["output"] = {
            "result_path": str(result_path), "result_sha256": sha_file(result_path),
            "infer_time_sha256": sha_file(timing_path), "n_rows": len(result),
        }
        run_id = timing.get("run_id")
        audit_dir = (BRANCH / "code" / ".runs" / str(run_id)) if run_id else None
        if audit_dir is None or not audit_dir.is_dir():
            audit_dir = sorted([p for p in (BRANCH / "code" / ".runs").iterdir() if p.is_dir()])[-1]
        receipt["audit_dir"] = str(audit_dir)
        manifest = json.loads((audit_dir / "run_manifest.json").read_text(encoding="utf-8"))
        heads = json.loads((audit_dir / "heads_only.json").read_text(encoding="utf-8"))
        receipt["output"]["n_embedded"] = heads.get("n_embedded")
        receipt["output"]["n_skipped_visual"] = heads.get("n_skipped_visual")
        receipt["risk_gate"] = heads.get("risk_gate")
        receipt["selective_review"] = heads.get("selective_review")
        calls = heads.get("tongagents", {}).get("actual_calls", [])
        receipt["actual_calls"] = [
            {"tool": call.get("tool"), "completed": call.get("completed"),
             "error": call.get("error")} for call in calls]
        review_dir = audit_dir / "selective_review"
        if (review_dir / "preflight.json").is_file():
            receipt["preflight"] = json.loads((review_dir / "preflight.json").read_text(encoding="utf-8"))
            receipt["preflight"]["sha256"] = sha_file(review_dir / "preflight.json")
        if (review_dir / "summary.json").is_file():
            receipt["selective_review"] = json.loads((review_dir / "summary.json").read_text(encoding="utf-8"))

    out = REPORTS / f"P2A_E2E_{args.name}.json"
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "name": args.name, "exit_code": receipt["exit_code"],
        "entry_infer_time_s": receipt["timing"]["entry_infer_time_s"],
        "result_sha256": receipt["output"].get("result_sha256"),
        "n_rows": receipt["output"].get("n_rows"),
        "risk_gate": {k: (receipt["risk_gate"] or {}).get(k) for k in
                      ("n_risk_detected", "n_teacher_triggered", "n_teacher_selected")},
        "selective_review": {k: (receipt["selective_review"] or {}).get(k) for k in
                             ("teacher_selected", "teacher_started", "teacher_completed",
                              "teacher_accepted", "status", "skip_reason")},
        "preflight": {k: (receipt["preflight"] or {}).get(k) for k in
                      ("resource_ok", "assets_ok", "sha_ok", "gpu_free_gib",
                       "time_estimate_s", "time_ok", "reasons", "model_loaded")},
        "actual_calls": receipt["actual_calls"],
        "receipt": str(out),
    }, ensure_ascii=False, indent=2))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
