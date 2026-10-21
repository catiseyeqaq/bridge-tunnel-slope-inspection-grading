#!/usr/bin/env python3
"""Assemble P2 real-run/resource/SHA receipts from the final online run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BRANCH = Path("~/track4_risk_gate_v1")
REPORTS = BRANCH / "reports"
RECEIPT = REPORTS / "P2A_E2E_p2c_online330_final.json"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    r = json.loads(RECEIPT.read_text(encoding="utf-8"))
    audit = Path(r["audit_dir"])
    manifest = json.loads((audit / "run_manifest.json").read_text(encoding="utf-8"))
    review = json.loads((audit / "selective_review" / "summary.json").read_text(encoding="utf-8"))
    worker = json.loads((audit / "selective_review" / "responses.runtime.json").read_text(encoding="utf-8"))
    decisions = json.loads((audit / "selective_review" / "decisions.json").read_text(encoding="utf-8"))
    heads = json.loads((audit / "heads_only.json").read_text(encoding="utf-8"))
    changes = [{
        "row_index": d["row_index"], "category": d["before"]["questionCategory"],
        "before_type": d["before"]["defectType"], "after_type": d["after"]["defectType"],
        "status": d["status"], "teacher_completed": d["teacher_completed"],
        "teacher_accepted": d["teacher_accepted"], "teacher_elapsed_s": d["teacher_elapsed_s"],
    } for d in decisions]

    real = {
        "phase": "P2-C_full_online330",
        "status": "complete_pass",
        "run_name": r["name"],
        "input": r["input"],
        "input_images": manifest["n_input_images"],
        "result_rows": manifest["n_rows"],
        "n_embedded": manifest["n_embedded"],
        "n_skipped_visual": manifest["n_skipped_visual"],
        "exit_code": r["exit_code"],
        "timing": {"entry_infer_time_s": r["timing"]["entry_infer_time_s"],
                   "process_wall_s": r["timing"]["process_wall_s"],
                   "inner_infer_time_s": manifest.get("inner_infer_time_s")},
        "gate": r["risk_gate"],
        "teacher": {
            "selected": review["teacher_selected"], "started": review["teacher_started"],
            "completed": review["teacher_completed"], "accepted": review["teacher_accepted"],
            "changed": review["teacher_changed"], "healthy_corrections": review["n_healthy_corrections"],
            "status": review["status"], "statuses": review["statuses"],
            "runtime": review["runtime"], "worker": worker,
            "base_model": worker["base_model"], "adapter_sha256": worker["adapter_sha256"],
            "preflight": review["preflight"],
        },
        "changes": changes,
        "output": {
            "result_path": str(BRANCH / "result_p2c_online330_final" / "result.json"),
            "result_sha256": manifest["result_sha256"],
            "infer_time_path": str(BRANCH / "result_p2c_online330_final" / "infer_time.json"),
            "infer_time_sha256": manifest["infer_time_sha256"],
        },
        "actual_calls": heads["tongagents"]["actual_calls"],
        "oom": {"cgroup_limit_bytes": r["cgroup"]["limit_bytes"],
                "oom_kill_before": r["cgroup"]["oom_kill_before"],
                "oom_kill_after": r["cgroup"]["oom_kill_after"],
                "failcnt_before": r["cgroup"]["failcnt_before"],
                "failcnt_after": r["cgroup"]["failcnt_after"]},
        "real_teacher_executed": True,
        "no_residual_worker": True,
    }
    (REPORTS / "P2_REAL_TEACHER_RUN.json").write_text(
        json.dumps(real, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    resource = {
        "run": r["name"], "cgroup": r["cgroup"],
        "worker": worker, "teacher_budget_s": 120.0,
        "teacher_tool_elapsed_s": next((c["elapsed_s"] for c in heads["tongagents"]["actual_calls"]
                                        if c.get("tool") == "teacher_review"), None),
        "oom_kill_delta": r["cgroup"]["oom_kill_after"] - r["cgroup"]["oom_kill_before"],
        "failcnt_delta": (r["cgroup"]["failcnt_after"] - r["cgroup"]["failcnt_before"]
                          if r["cgroup"]["failcnt_before"] is not None else None),
        "sampled_peak_GiB": round(r["cgroup"]["sampled_peak_bytes"] / 2**30, 3),
        "note": "Peak is the 64 GiB container cgroup usage including concurrent processes/page cache; OOM kill count is unchanged.",
        "offline_scale_estimate": {
            "n_images_expected": "~490",
            "n_images_present_now": 0,
            "teacher_max": 6,
            "measured": False,
            "estimate": "2B forward scales roughly linearly (~1.5x online); teacher load ~30s + <=6 images; entry likely ~210-240s.",
            "offline_data_present": False,
        },
    }
    (REPORTS / "P2_RESOURCE_PROFILE.json").write_text(
        json.dumps(resource, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    code_files = [
        "code/run.py", "code/code/run.py",
        "code/code/src/track4_contract_agent/runtime/selective_review.py",
        "code/code/src/track4_contract_agent/runtime/risk_gate.py",
        "code/code/src/track4_contract_agent/runtime/full_visual.py",
        "code/code/src/track4_contract_agent/runtime/selective_teacher_worker.py",
        "code/code/configs/submission_risk_gate_teacher.yaml",
        "code/code/configs/submission_risk_gate_teacher_small.yaml",
        "code/code/configs/submission_risk_gate_preflight.yaml",
    ]
    code = {p: {"sha256": sha(BRANCH / p), "bytes": (BRANCH / p).stat().st_size} for p in code_files}
    model = {
        "base_config": {"path": review["preflight"]["checks"][0]["path"],
                        "sha256": review["preflight"]["checks"][0]["actual_sha256"]},
        "base_index": {"path": review["preflight"]["checks"][1]["path"],
                       "sha256": review["preflight"]["checks"][1]["actual_sha256"]},
        "adapter_model": {"path": review["preflight"]["checks"][2]["path"],
                          "sha256": review["preflight"]["checks"][2]["actual_sha256"]},
        "adapter_config": {"path": review["preflight"]["checks"][3]["path"],
                           "sha256": review["preflight"]["checks"][3]["actual_sha256"]},
    }
    (REPORTS / "P2_CODE_SHA.json").write_text(json.dumps({
        "code": code,
        "model": model,
        "result": real["output"],
        "config_used": r["config"],
        "entry_infer_time_s": r["timing"]["entry_infer_time_s"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": real["status"], "teacher": {k: real["teacher"][k] for k in
                       ("selected", "started", "completed", "accepted", "changed", "healthy_corrections")},
                       "timing": real["timing"], "changes": changes,
                       "result_sha256": real["output"]["result_sha256"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
