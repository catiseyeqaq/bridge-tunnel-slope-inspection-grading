#!/usr/bin/env python3
"""P1.1 replay: historical330, online330 and a normal input with no risk.

Read-only. No model is loaded. Thresholds were derived on dev+lockbox with
train-only heads; online330 is replayed for reporting only and is never used to
tune thresholds.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from track4_contract_agent.runtime.risk_gate import DEFAULTS, evaluate  # noqa: E402


def load_evidence(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda item: int(item.get("row_index", len(rows))))
    return rows


def reference_summary(path: Path, report):
    routing = json.loads(path.read_text(encoding="utf-8"))
    selected = [item for item in routing if item.get("selected")]
    ref_rows = sorted(int(item["row_index"]) for item in selected)
    gate_rows = sorted(int(item["row_index"]) for item in report["selected"])
    audits = [item for item in selected
              if any("audit" in str(reason) for reason in item.get("reasons", []))]
    return {
        "routing_path": str(path),
        "reference_selected": ref_rows,
        "gate_selected": gate_rows,
        "overlap": sorted(set(ref_rows) & set(gate_rows)),
        "reference_only": sorted(set(ref_rows) - set(gate_rows)),
        "gate_only": sorted(set(gate_rows) - set(ref_rows)),
        "reference_audit_rows": sorted(int(item["row_index"]) for item in audits),
        "reference_selected_count": len(ref_rows),
        "gate_selected_count": len(gate_rows),
        "reference_rows": [
            {"row_index": int(item["row_index"]), "priority": item.get("priority"),
             "reasons": item.get("reasons", []), "rank": item.get("rank")}
            for item in sorted(selected, key=lambda value: value.get("rank", 0))
        ],
    }


def normal_input():
    rows = []
    for _ in range(10):
        rows.append({
            "question_category": "桥梁", "embedding_ok": True,
            "p_disease": 0.10, "threshold": 0.8,
            "type_margin": 0.90, "nearest_similarity": 0.99,
            "training_support_similarity": 0.95, "training_support_threshold": 0.7252,
            "low_training_support": False, "head_label": "完好",
        })
    for _ in range(10):
        rows.append({
            "question_category": "轨道", "embedding_ok": True,
            "probability_margin": 0.90, "entropy": 0.10,
            "training_support_similarity": 0.95, "training_support_threshold": 0.5886,
            "low_training_support": False, "head_label": "裂缝",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", required=True)
    parser.add_argument("--online", required=True)
    parser.add_argument("--historical-routing", required=True)
    parser.add_argument("--online-routing", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    datasets = {}
    all_rows = []
    for name, evidence_path, routing_path in (
        ("historical330", Path(args.historical), Path(args.historical_routing)),
        ("online330", Path(args.online), Path(args.online_routing)),
    ):
        evidence = load_evidence(evidence_path)
        report = evaluate(evidence, DEFAULTS)
        datasets[name] = {
            "evidence": str(evidence_path.resolve()),
            "n_rows": len(evidence),
            "batch": report["batch"],
            "selected": report["selected"],
            "reference": reference_summary(routing_path, report),
            "candidates": report["candidates"],
        }
        for candidate in report["candidates"]:
            all_rows.append({
                "dataset": name,
                "row_index": candidate["row_index"],
                "category": candidate["category"],
                "risk_detected": candidate["risk_detected"],
                "teacher_triggered": candidate["teacher_triggered"],
                "teacher_selected": candidate["teacher_selected"],
                "budget_overflow": candidate["budget_overflow"],
                "risk_priority": candidate["risk_priority"],
                "trigger_reasons": ";".join(candidate["trigger_reasons"]),
                "severe_reasons": ";".join(candidate["severe_reasons"]),
                "selection_status": candidate["selection_status"],
                "review_status": candidate["review_status"],
            })

    normal = evaluate(normal_input(), DEFAULTS)
    datasets["normal_input"] = {
        "evidence": "synthetic: high-confidence, support-sufficient bridge/rail rows",
        "n_rows": normal["batch"]["n_rows"],
        "batch": normal["batch"],
        "selected": normal["selected"],
        "reference": None,
        "candidates": normal["candidates"],
    }
    checks = {
        "normal_input_teacher_triggered": normal["batch"]["n_teacher_triggered"],
        "normal_input_teacher_selected": normal["batch"]["n_teacher_selected"],
        "normal_input_ok": normal["batch"]["n_teacher_triggered"] == 0
                           and normal["batch"]["n_teacher_selected"] == 0,
        "historical_by_category": datasets["historical330"]["batch"]["by_category"],
        "online_by_category": datasets["online330"]["batch"]["by_category"],
    }
    output = {
        "version": "p1_1",
        "threshold_source": {
            "data": "dev+lockbox (historical validation), train-only heads",
            "method": "1% tail quantiles for severe boundary/margin/similarity/entropy; support shortfall 0.10",
            "thresholds": DEFAULTS,
            "online330_used_for_tuning": False,
            "online330_labels_available": False,
            "labels_used_for_thresholds": False,
            "labels_used_for_diagnostics": True,
            "note": "Classifier scores are not token probabilities; nearest-neighbour similarity is not calibrated OOD.",
        },
        "datasets": datasets,
        "checks": checks,
        "notes": [
            "P1.1 is audit-only: teacher_started is false in every dataset.",
            "Only teacher_triggered rows are eligible; no fixed healthy/positive audits.",
            "Selection is not correctness; no gold labels are used for online330.",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "historical330": {
            "batch": {k: datasets["historical330"]["batch"][k] for k in (
                "n_rows", "n_risk_detected", "n_teacher_triggered", "n_teacher_selected",
                "n_budget_overflow", "n_uncertainty", "n_low_training_support",
                "n_dual_triggered", "n_dual_severe", "allocation")},
            "reference": datasets["historical330"]["reference"],
        },
        "online330": {
            "batch": {k: datasets["online330"]["batch"][k] for k in (
                "n_rows", "n_risk_detected", "n_teacher_triggered", "n_teacher_selected",
                "n_budget_overflow", "n_uncertainty", "n_low_training_support",
                "n_dual_triggered", "n_dual_severe", "allocation")},
            "reference": datasets["online330"]["reference"],
        },
        "normal_input": datasets["normal_input"]["batch"],
        "checks": checks,
        "out": str(out_path),
        "csv": str(csv_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
