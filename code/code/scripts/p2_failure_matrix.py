#!/usr/bin/env python3
"""P2 failure matrix. Preflight and parser tests; no real 27B is loaded here."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import yaml

from track4_contract_agent.runtime.selective_review import (
    apply_selective_review, checked_config, parse_and_merge, preflight_teacher,
)

BRANCH = Path("~/track4_risk_gate_v1")
REPORTS = BRANCH / "reports"
CONFIG = BRANCH / "code/code/configs/submission_risk_gate_teacher.yaml"


def rail_row():
    return {"questionCategory": "轨道", "defectType": "裂缝(混凝土裂缝)",
            "defectDescription": "", "ratingScale(1-5)": ""}


def bridge_row():
    return {"questionCategory": "桥梁", "bridgeName": "范家坪1号大桥（右幅）",
            "defectType": "渗水/泛碱", "defectDescription": "既有描述",
            "ratingScale(1-5)": "2"}


class FakeBridge:
    healthy_by_view = {"aerial": "桥面铺装平整，外观完好"}
    healthy_medoid = "桥面铺装平整，外观完好"
    type_medoid = {"渗水/泛碱": "桥面局部渗水泛碱", "黄色水痕": "桥面黄色水痕",
                   "锈蚀/碳化": "构件表面锈蚀碳化"}


def main() -> int:
    cfg = checked_config(yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["selective_review"])
    rows = []
    preflight_cases = [
        ("preflight_pass_4_selected", cfg, 4, 90.0),
        ("time_budget_insufficient_6", cfg, 6, 90.0),
        ("gpu_free_insufficient", cfg, 4, 10.0),
    ]
    for name, base, n, gpu in preflight_cases:
        result = preflight_teacher(base, n, gpu_free_gib=gpu)
        rows.append({
            "scenario": name, "kind": "preflight", "simulated": True,
            "real_model_loaded": False, "expected": "resource_ok",
            "actual": {"resource_ok": result["resource_ok"], "reasons": result["reasons"],
                       "time_estimate_s": result["time_estimate_s"],
                       "gpu_free_gib": result["gpu_free_gib"]},
            "passed": result["resource_ok"] if name == "preflight_pass_4_selected" else not result["resource_ok"],
        })

    missing = dict(cfg)
    missing["base_model"] = "/nonexistent/teacher_27b"
    missing["adapter"] = "/nonexistent/teacher_lora"
    result = preflight_teacher(missing, 2, gpu_free_gib=90.0)
    rows.append({"scenario": "weights_missing", "kind": "preflight", "simulated": True,
                 "real_model_loaded": False, "expected": "assets_ok=false",
                 "actual": {"assets_ok": result["assets_ok"], "reasons": result["reasons"]},
                 "passed": not result["assets_ok"] and "teacher_assets_missing" in result["reasons"]})

    bad_sha = dict(cfg)
    bad_sha["adapter_sha256"] = "0" * 64
    result = preflight_teacher(bad_sha, 2, gpu_free_gib=90.0)
    rows.append({"scenario": "adapter_sha_mismatch", "kind": "preflight", "simulated": True,
                 "real_model_loaded": False, "expected": "sha_ok=false",
                 "actual": {"sha_ok": result["sha_ok"], "reasons": result["reasons"]},
                 "passed": not result["sha_ok"] and "teacher_sha_mismatch" in result["reasons"]})

    # Zero selected: no asset check, no worker.
    with tempfile.TemporaryDirectory() as tmp:
        _, summary = apply_selective_review(
            [rail_row()], [{"question_category": "轨道", "embedding_ok": True,
                            "probability_margin": 0.9, "entropy": 0.1}],
            ["image.jpg"], {"轨道": {"裂缝(混凝土裂缝)", "破损"}}, None,
            {**cfg, "mode": "preflight"}, Path(tmp), selected_rows=[], run_mode="preflight")
        rows.append({"scenario": "zero_selected", "kind": "scheduler", "simulated": True,
                     "real_model_loaded": False, "expected": "no_teacher_selected",
                     "actual": {"status": summary["status"], "teacher_started": summary["teacher_started"]},
                     "passed": summary["status"] == "no_teacher_selected" and not summary["teacher_started"]})

    # Worker timeout/failure: fake runner, never a real TongAgents call.
    for status in ("wall_budget_exhausted", "worker_failed"):
        with tempfile.TemporaryDirectory() as tmp:
            def fake_runner(request_path, response_path, cfg, worker_path, _status=status):
                return {"status": _status, "returncode": -15, "elapsed_s": 120.0,
                        "log": "simulated.log", "simulated": True}
            merged, summary = apply_selective_review(
                [rail_row()], [{"question_category": "轨道", "embedding_ok": True,
                                "probability_margin": 0.9, "entropy": 0.1}],
                ["image.jpg"], {"轨道": {"裂缝(混凝土裂缝)", "破损"}}, None,
                {**cfg, "mode": "real"}, Path(tmp), selected_rows=[0],
                run_mode="real", worker_runner=fake_runner)
            rows.append({"scenario": "teacher_" + status, "kind": "worker", "simulated": True,
                         "real_model_loaded": False, "expected": "keep_2b",
                         "actual": {"status": summary["status"], "teacher_started": summary["teacher_started"],
                                    "teacher_completed": summary["teacher_completed"],
                                    "row_unchanged": merged[0] == rail_row()},
                         "passed": (summary["status"] == status and summary["teacher_completed"] == 0
                                    and merged[0] == rail_row())})

    labels = {"完好", "渗水/泛碱", "黄色水痕", "锈蚀/碳化"}
    rail_labels = {"完好", "裂缝(混凝土裂缝)", "破损"}
    evidence = {"question_category": "桥梁", "embedding_ok": True, "p_disease": 0.85,
                "threshold": 0.8, "visual_view": "aerial", "type_margin": 0.3}
    parser_cases = [
        ("invalid_json", "{", "invalid_json", bridge_row(), evidence, labels),
        ("teacher_uncertain", json.dumps({"type": "完好", "rating": "", "evidence": "看不清",
                                          "uncertain": True}), "teacher_uncertain", bridge_row(), evidence, labels),
        ("outside_closed_labels", json.dumps({"type": "不存在", "rating": "", "evidence": "图",
                                              "uncertain": False}), "outside_closed_labels", bridge_row(), evidence, labels),
        ("missing_visual_evidence", json.dumps({"type": "完好", "rating": "", "evidence": "",
                                                "uncertain": False}), "missing_visual_evidence", bridge_row(), evidence, labels),
        ("invalid_rating", json.dumps({"type": "渗水/泛碱", "rating": "9", "evidence": "图",
                                       "uncertain": False}), "invalid_rating", bridge_row(), evidence, labels),
        ("healthy_correction_eligible", json.dumps({"type": "完好", "rating": "", "evidence": "平整",
                                                    "uncertain": False}), "accepted_changed", bridge_row(), evidence, labels),
        ("healthy_correction_ineligible", json.dumps({"type": "完好", "rating": "", "evidence": "平整",
                                                      "uncertain": False}),
         "healthy_correction_student_evidence_insufficient", bridge_row(),
         {**evidence, "p_disease": 0.95}, labels),
        ("rail_disease_to_healthy_blocked", json.dumps({"type": "完好", "rating": "", "evidence": "平整",
                                                        "uncertain": False}),
         "healthy_correction_student_evidence_insufficient", rail_row(),
         {"question_category": "轨道", "embedding_ok": True}, rail_labels),
        ("rail_valid_change", json.dumps({"type": "破损", "rating": "", "evidence": "破损",
                                          "uncertain": False}), "accepted_changed", rail_row(),
         {"question_category": "轨道", "embedding_ok": True}, rail_labels),
    ]
    for name, raw, expected, row, ev, allowed in parser_cases:
        merged, status, _ = parse_and_merge(raw, row, ev, allowed, FakeBridge(), cfg)
        rows.append({"scenario": name, "kind": "parser", "simulated": True,
                     "real_model_loaded": False, "expected": expected,
                     "actual": {"status": status, "row_unchanged": merged == row},
                     "passed": status == expected})

    rows.append({"scenario": "system_level_oom", "kind": "system", "simulated": True,
                 "real_model_loaded": False, "expected": "not_injected",
                 "actual": {"note": "Parent OOM-kill cannot be reliably recovered; final real run oom_kill=0."},
                 "passed": None, "blocking_if_unrecoverable": True})

    out = REPORTS / "P2_FAILURE_MATRIX.json"
    out.write_text(json.dumps({"phase": "P2", "cases": rows,
                               "passed": sum(1 for r in rows if r["passed"] is True),
                               "failed": sum(1 for r in rows if r["passed"] is False),
                               "not_run": sum(1 for r in rows if r["passed"] is None)},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Real TongAgents calls from the final online run, kept separate from simulations.
    receipt = json.loads((REPORTS / "P2A_E2E_p2c_online330_final.json").read_text(encoding="utf-8"))
    heads = json.loads((Path(receipt["audit_dir"]) / "heads_only.json").read_text(encoding="utf-8"))
    calls = heads.get("tongagents", {}).get("actual_calls", [])
    calls_out = {
        "real_run": "p2c_online330_final",
        "real_calls": calls,
        "real_teacher_review_calls": sum(1 for c in calls if c.get("tool") == "teacher_review"),
        "real_embed_calls": sum(1 for c in calls if c.get("tool") == "embed_head_inspect"),
        "teacher_jobs": 4, "teacher_completed": 4,
        "simulated_cases": [{"scenario": r["scenario"], "simulated": True,
                             "recorded_as_real_tongagents_call": False}
                            for r in rows if r["simulated"]],
        "policy": "Simulated cases are not written to actual_calls/responses.log and must not be cited as real teacher inference.",
    }
    (REPORTS / "P2_TONGAGENTS_CALLS.json").write_text(
        json.dumps(calls_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": sum(1 for r in rows if r["passed"] is True),
                      "failed": sum(1 for r in rows if r["passed"] is False),
                      "not_run": sum(1 for r in rows if r["passed"] is None),
                      "real_calls": calls, "out": str(out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
