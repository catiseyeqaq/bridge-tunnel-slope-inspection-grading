#!/usr/bin/env python3
"""Derive severe-risk thresholds from dev/lockbox with train-only heads.

Read-only analysis. The online330 log is NOT used here. Gold labels are used
only to describe the error rate of the candidate tail, not to fit a model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from track4_contract_agent.runtime.full_visual import AllViewBridgeHead  # noqa: E402
from track4_contract_agent.runtime.rail_embed_head import RailEmbedHead  # noqa: E402
from track4_contract_agent.runtime.training_support import TrainingSupport  # noqa: E402

CACHE = Path("~//code/data/multimodal_gold_index/train_v1/image_vectors.f16.npy")
TABLE = Path("~//exp_b0_7152/reports/e0c_data_table.json")
BRIDGE_HEAD = Path("~//exp_b0_7152/b0_package/code/data/bridge_all_view/head_train.npz")
RAIL_HEAD = Path("~//exp_b0_7152/b0_package/code/data/rail_head/rail_head_train_v15.npz")
SUPPORT = ROOT / "data" / "training_support" / "support_bank.npz"


def quantiles(values, qs):
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(arr):
        return {}
    return {str(q): float(np.quantile(arr, q)) for q in qs}


def main() -> int:
    rows = json.loads(TABLE.read_text())
    x = np.load(CACHE).astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    bridge = AllViewBridgeHead.load(BRIDGE_HEAD)
    rail = RailEmbedHead.load(RAIL_HEAD)
    support = TrainingSupport(SUPPORT)

    bundle = {"bridge": {"dev": [], "lockbox": []}, "rail": {"dev": [], "lockbox": []}}
    for split in ("dev", "lockbox"):
        br = [r for r in rows if r["split"] == split and r["category"] == "桥梁"]
        rr = [r for r in rows if r["split"] == split and r["category"] == "轨道"]
        if br:
            vec = x[[r["row"] for r in br]]
            decisions = bridge.decide(vec)
            evidence = []
            for r, d in zip(br, decisions):
                evidence.append({
                    "question_category": "桥梁",
                    "p_disease": d["p_disease"], "threshold": bridge.threshold,
                    "type_margin": d["type_margin"], "nearest_similarity": d["nearest_similarity"],
                })
            support.annotate(vec, evidence)
            for r, d, e in zip(br, decisions, evidence):
                bundle["bridge"][split].append({
                    "row": r["row"],
                    "gray_distance": abs(float(d["p_disease"]) - float(bridge.threshold)),
                    "type_margin": float(d["type_margin"]),
                    "nearest_similarity": float(d["nearest_similarity"]),
                    "support_shortfall": float(e["training_support_threshold"]) - float(e["training_support_similarity"]),
                    "p_disease": float(d["p_disease"]),
                    "gold_disease": r["defect_type"] != "完好",
                    "pred_disease": float(d["p_disease"]) >= bridge.threshold,
                    "gold_type": r["defect_type"], "pred_type": d["type"],
                })
        if rr:
            vec = x[[r["row"] for r in rr]]
            decisions = rail.predict_detailed(vec)
            evidence = []
            for r, d in zip(rr, decisions):
                evidence.append({
                    "question_category": "轨道",
                    "probability_margin": d["probability_margin"], "entropy": d["entropy"],
                })
            support.annotate(vec, evidence)
            for r, d, e in zip(rr, decisions, evidence):
                bundle["rail"][split].append({
                    "row": r["row"],
                    "margin": float(d["probability_margin"]),
                    "entropy": float(d["entropy"]),
                    "support_shortfall": float(e["training_support_threshold"]) - float(e["training_support_similarity"]),
                    "gold_type": r["defect_type"], "pred_type": d["label"],
                    "exact": r["defect_type"] == d["label"],
                })

    report = {"scope": "dev+lockbox, train-only heads, no online330 used", "source": {
        "bridge_head": str(BRIDGE_HEAD), "rail_head": str(RAIL_HEAD), "support": str(SUPPORT),
        "vectors": str(CACHE), "table": str(TABLE)},
        "splits": {}}
    for split in ("dev", "lockbox"):
        b = bundle["bridge"][split]
        r = bundle["rail"][split]
        report["splits"][split] = {
            "bridge": {
                "n": len(b),
                "gray_distance": quantiles([v["gray_distance"] for v in b], [0.005, 0.01, 0.02, 0.05, 0.10, 0.50]),
                "type_margin": quantiles([v["type_margin"] for v in b], [0.005, 0.01, 0.02, 0.05, 0.10, 0.50]),
                "nearest_similarity": quantiles([v["nearest_similarity"] for v in b], [0.005, 0.01, 0.02, 0.05, 0.10, 0.50]),
                "support_shortfall": quantiles([v["support_shortfall"] for v in b], [0.50, 0.90, 0.95, 0.98, 0.99, 0.995]),
                "presence_error_rate": float(np.mean([v["gold_disease"] != v["pred_disease"] for v in b])),
            },
            "rail": {
                "n": len(r),
                "margin": quantiles([v["margin"] for v in r], [0.005, 0.01, 0.02, 0.05, 0.10, 0.50]),
                "entropy": quantiles([v["entropy"] for v in r], [0.50, 0.90, 0.95, 0.98, 0.99, 0.995]),
                "support_shortfall": quantiles([v["support_shortfall"] for v in r], [0.50, 0.90, 0.95, 0.98, 0.99, 0.995]),
                "exact_error_rate": float(np.mean([not v["exact"] for v in r])),
            },
        }

    # Candidate severe thresholds = the 1% tail of dev+lockbox combined.
    combined = {
        "bridge_gray_distance": [v["gray_distance"] for s in ("dev", "lockbox") for v in bundle["bridge"][s]],
        "bridge_type_margin": [v["type_margin"] for s in ("dev", "lockbox") for v in bundle["bridge"][s]],
        "bridge_nearest_similarity": [v["nearest_similarity"] for s in ("dev", "lockbox") for v in bundle["bridge"][s]],
        "rail_margin": [v["margin"] for s in ("dev", "lockbox") for v in bundle["rail"][s]],
        "rail_entropy": [v["entropy"] for s in ("dev", "lockbox") for v in bundle["rail"][s]],
        "support_shortfall_bridge": [v["support_shortfall"] for s in ("dev", "lockbox") for v in bundle["bridge"][s]],
        "support_shortfall_rail": [v["support_shortfall"] for s in ("dev", "lockbox") for v in bundle["rail"][s]],
    }
    report["candidate_severe_thresholds_1pct"] = {
        "bridge_gray_distance_max": float(np.quantile(combined["bridge_gray_distance"], 0.01)),
        "bridge_type_margin_min": float(np.quantile(combined["bridge_type_margin"], 0.01)),
        "bridge_nearest_similarity_min": float(np.quantile(combined["bridge_nearest_similarity"], 0.01)),
        "rail_margin_min": float(np.quantile(combined["rail_margin"], 0.01)),
        "rail_entropy_max": float(np.quantile(combined["rail_entropy"], 0.99)),
        "support_shortfall_bridge_min": float(np.quantile(combined["support_shortfall_bridge"], 0.99)),
        "support_shortfall_rail_min": float(np.quantile(combined["support_shortfall_rail"], 0.99)),
        "support_shortfall_bridge_max": float(np.quantile(combined["support_shortfall_bridge"], 0.99)),
        "support_shortfall_rail_max": float(np.quantile(combined["support_shortfall_rail"], 0.99)),
        "note": "1% tail of dev+lockbox with train-only heads; gold labels used only for diagnostics, not for online330 tuning.",
    }

    # Diagnostic: error rate inside the 1% severe tail vs overall.
    bt = report["candidate_severe_thresholds_1pct"]
    br_all = [v for s in ("dev", "lockbox") for v in bundle["bridge"][s]]
    rr_all = [v for s in ("dev", "lockbox") for v in bundle["rail"][s]]
    br_tail = [v for v in br_all if v["gray_distance"] <= bt["bridge_gray_distance_max"]]
    rr_tail = [v for v in rr_all if v["margin"] <= bt["rail_margin_min"]]
    report["tail_diagnostics"] = {
        "bridge_tail_n": len(br_tail),
        "bridge_tail_presence_error": float(np.mean([v["gold_disease"] != v["pred_disease"] for v in br_tail])) if br_tail else None,
        "bridge_overall_presence_error": float(np.mean([v["gold_disease"] != v["pred_disease"] for v in br_all])),
        "rail_tail_n": len(rr_tail),
        "rail_tail_exact_error": float(np.mean([not v["exact"] for v in rr_tail])) if rr_tail else None,
        "rail_overall_exact_error": float(np.mean([not v["exact"] for v in rr_all])),
    }

    out = ROOT.parent / "reports" / "gate_threshold_study.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("scope", "candidate_severe_thresholds_1pct", "tail_diagnostics")},
                     ensure_ascii=False, indent=2))
    print("out", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
