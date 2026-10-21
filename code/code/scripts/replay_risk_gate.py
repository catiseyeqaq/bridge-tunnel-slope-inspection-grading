#!/usr/bin/env python3
"""Offline replay of the P1 risk gate on frozen evidence logs.

No model is loaded. The script reads `visual_evidence.jsonl` produced by a
previous r1 run, evaluates the risk gate, and (optionally) compares the selected
rows with the frozen r1 `routing.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from track4_contract_agent.runtime.risk_gate import evaluate  # noqa: E402


def load_evidence(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rows.sort(key=lambda item: int(item.get("row_index", len(rows))))
    return rows


def reference_summary(path: Path, gate_selected):
    routing = json.loads(path.read_text(encoding="utf-8"))
    selected = [item for item in routing if item.get("selected")]
    ref_rows = sorted(int(item["row_index"]) for item in selected)
    gate_rows = sorted(int(item["row_index"]) for item in gate_selected)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "submission_risk_gate.yaml"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference-routing", default="")
    args = parser.parse_args()

    import yaml
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    gate_config = dict(config.get("risk_gate") or {})
    evidence = load_evidence(Path(args.evidence))
    report = evaluate(evidence, gate_config)

    output = {
        "evidence": str(Path(args.evidence).resolve()),
        "config": str(Path(args.config).resolve()),
        "run_mode": config.get("run_mode"),
        "gate_enabled": bool(gate_config.get("enabled")),
        "thresholds": report["thresholds"],
        "budget": report["budget"],
        "batch": report["batch"],
        "selected": report["selected"],
        "candidates": report["candidates"],
        "reference": reference_summary(Path(args.reference_routing), report["selected"])
        if args.reference_routing else None,
        "notes": [
            "Offline replay only; no GPU inference and no 27B process was started.",
            "Only risk-triggered rows are eligible; there are no fixed healthy/positive audits.",
            "Selection is not correctness; no gold labels are used here.",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "evidence": str(args.evidence),
        "n_rows": report["batch"]["n_rows"],
        "n_triggered": report["batch"]["n_triggered"],
        "n_selected": report["batch"]["n_selected"],
        "n_budget_exhausted": report["batch"]["n_budget_exhausted"],
        "n_uncertainty": report["batch"]["n_uncertainty"],
        "n_low_training_support": report["batch"]["n_low_training_support"],
        "low_support_fraction": report["batch"]["low_support_fraction"],
        "batch_shift_suspected": report["batch"]["batch_shift_suspected"],
        "by_reason": report["batch"]["by_reason"],
        "selected": [
            {"row_index": row["row_index"], "category": row["category"],
             "risk_priority": row["risk_priority"], "reasons": row["trigger_reasons"]}
            for row in report["selected"]
        ],
        "reference": output["reference"],
        "out": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
