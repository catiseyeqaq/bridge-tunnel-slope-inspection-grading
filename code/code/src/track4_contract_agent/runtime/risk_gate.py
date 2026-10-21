"""Risk Gate V1.1: three-level trigger semantics and category-aware quotas.

Levels (kept separate on purpose):
  * risk_detected   : an ordinary risk signal (r1 thresholds) is present.
  * teacher_triggered: a *severe* signal (or high-confidence + severe support
                       shortfall) makes the 27B insurance worthwhile.
  * teacher_selected : the row actually receives one of the limited review slots.

Ordinary risk does NOT start the teacher. Cross-category raw scores are never
compared: each category is triggered independently and quota is allocated from
per-category triggered counts. There are no fixed healthy/positive audits.

Thresholds for the severe tail come from dev+lockbox with train-only heads
(1% tail; support shortfall 0.10). The online330 expected output is not used to
tune them. Classifier scores are not token probabilities; nearest-neighbour
similarity is not a calibrated OOD probability.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

VERSION = "risk_gate_v1_1"
HEALTHY = "完好"

DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "max_records": 6,
    "max_fraction": 0.05,
    "wall_budget_s": 120.0,
    # Ordinary r1 thresholds.
    "bridge_gray_width": 0.10,
    "bridge_type_margin": 0.15,
    "bridge_similarity": 0.55,
    "rail_margin": 0.15,
    "rail_entropy": 1.4,
    # Severe thresholds: 1% tail of dev+lockbox with train-only heads.
    "bridge_gray_width_severe": 0.01,
    "bridge_type_margin_severe": 0.005,
    "bridge_similarity_severe": 0.45,
    "rail_margin_severe": 0.005,
    "rail_entropy_severe": 2.25,
    # "Clearly insufficient" support: similarity at least 0.10 below threshold.
    "support_shortfall_severe": 0.10,
}


def checked_config(raw: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    if raw:
        cfg.update({k: v for k, v in dict(raw).items() if k in DEFAULTS})
    for key in ("max_records",):
        value = int(cfg[key])
        if value < 0:
            raise ValueError(f"risk_gate.{key} must be >= 0")
        cfg[key] = value
    numeric = [k for k in DEFAULTS if k not in ("enabled", "max_records")]
    for key in numeric:
        value = float(cfg[key])
        if value < 0:
            raise ValueError(f"risk_gate.{key} must be >= 0")
        cfg[key] = value
    if cfg["max_fraction"] > 1:
        raise ValueError("risk_gate.max_fraction must be <= 1")
    if cfg["wall_budget_s"] <= 0:
        raise ValueError("risk_gate.wall_budget_s must be > 0")
    return cfg


def _support_shortfall(e: Mapping[str, Any]) -> float:
    similarity = e.get("training_support_similarity")
    threshold = e.get("training_support_threshold")
    if similarity is None or threshold is None:
        return 0.0
    return float(threshold) - float(similarity)


def _low_support(e: Mapping[str, Any]) -> bool:
    if "low_training_support" in e:
        return bool(e["low_training_support"])
    return _support_shortfall(e) > 0.0


def _prioritise(values: List[float], cap: float) -> float:
    """Map a severe distance to 0..1 (larger = more severe)."""
    if cap <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - values[0] / cap))


def evaluate_row(e: Mapping[str, Any], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    if not e.get("embedding_ok", True):
        raise ValueError("2B full-visual coverage failed before risk gating")
    category = str(e.get("question_category") or "")
    if category not in ("桥梁", "轨道"):
        raise ValueError(f"Unsupported review category: {category!r}")

    ordinary: List[str] = []
    ordinary_groups: List[str] = []
    severe: List[str] = []
    severity_scores: List[float] = []

    shortfall = _support_shortfall(e)
    low_support = _low_support(e)
    if low_support:
        ordinary.append("low_training_support")
        ordinary_groups.append("low_training_support")

    if category == "桥梁":
        p = float(e["p_disease"])
        threshold = float(e["threshold"])
        distance = abs(p - threshold)
        margin = float(e["type_margin"])
        similarity = float(e["nearest_similarity"])
        high_confidence = distance > float(cfg["bridge_gray_width"]) and (
            str(e.get("head_label") or "") == HEALTHY
            or margin >= float(cfg["bridge_type_margin"])
        )
        if distance <= float(cfg["bridge_gray_width"]):
            ordinary.append("bridge_presence_boundary")
            ordinary_groups.append("uncertainty")
        if p >= 0.5 and margin < float(cfg["bridge_type_margin"]):
            ordinary.append("bridge_case_vote_disagreement")
            ordinary_groups.append("uncertainty")
        if p >= 0.5 and similarity < float(cfg["bridge_similarity"]):
            ordinary.append("bridge_low_disease_case_similarity")
            ordinary_groups.append("uncertainty")

        if distance <= float(cfg["bridge_gray_width_severe"]):
            severe.append("bridge_presence_boundary_severe")
            severity_scores.append(1.0 - distance / max(float(cfg["bridge_gray_width_severe"]), 1e-9))
        if p >= 0.5 and margin < float(cfg["bridge_type_margin_severe"]):
            severe.append("bridge_case_vote_disagreement_severe")
            severity_scores.append(1.0 - margin / max(float(cfg["bridge_type_margin_severe"]), 1e-9))
        if p >= 0.5 and similarity < float(cfg["bridge_similarity_severe"]):
            severe.append("bridge_low_disease_case_similarity_severe")
            severity_scores.append(1.0 - similarity / max(float(cfg["bridge_similarity_severe"]), 1e-9))
        if high_confidence and shortfall >= float(cfg["support_shortfall_severe"]):
            severe.append("high_confidence_low_training_support")
            severity_scores.append(shortfall / max(float(cfg["support_shortfall_severe"]), 1e-9))
    else:
        margin = float(e["probability_margin"])
        entropy = float(e["entropy"])
        high_confidence = margin >= float(cfg["rail_margin"]) and entropy <= float(cfg["rail_entropy"])
        if margin < float(cfg["rail_margin"]):
            ordinary.append("rail_low_class_margin")
            ordinary_groups.append("uncertainty")
        if entropy > float(cfg["rail_entropy"]):
            ordinary.append("rail_high_class_entropy")
            ordinary_groups.append("uncertainty")
        if margin < float(cfg["rail_margin_severe"]):
            severe.append("rail_low_class_margin_severe")
            severity_scores.append(1.0 - margin / max(float(cfg["rail_margin_severe"]), 1e-9))
        if entropy > float(cfg["rail_entropy_severe"]):
            severe.append("rail_high_class_entropy_severe")
            severity_scores.append((entropy - float(cfg["rail_entropy_severe"])) / max(1.0, 4.0 - float(cfg["rail_entropy_severe"])))
        if high_confidence and shortfall >= float(cfg["support_shortfall_severe"]):
            severe.append("high_confidence_low_training_support")
            severity_scores.append(shortfall / max(float(cfg["support_shortfall_severe"]), 1e-9))

    ordinary = list(dict.fromkeys(ordinary))
    ordinary_groups = list(dict.fromkeys(ordinary_groups))
    severe = list(dict.fromkeys(severe))
    risk_detected = bool(ordinary)
    teacher_triggered = bool(severe)
    return {
        "category": category,
        "risk_detected": risk_detected,
        "teacher_triggered": teacher_triggered,
        "trigger_reasons": ordinary,
        "risk_groups": ordinary_groups,
        "severe_reasons": severe,
        "risk_priority": round(float(max(severity_scores, default=0.0)), 6),
        "high_confidence": high_confidence,
        "support_shortfall": round(shortfall, 6),
    }


def _allocate(limit: int, n_bridge: int, n_rail: int) -> Dict[str, Any]:
    """Allocate quota from per-category triggered counts only.

    Both categories triggered and limit >= 2: one each, then Hamilton
    proportional allocation of the remainder, capped by each category's
    triggered count; unused quota is redistributed to the other category.
    Only one category triggered: all quota to it (no forced other-category slot).
    """
    if n_bridge <= 0 and n_rail <= 0:
        return {"bridge": 0, "rail": 0, "method": "no_trigger"}
    if n_bridge <= 0:
        return {"bridge": 0, "rail": min(limit, n_rail), "method": "rail_only"}
    if n_rail <= 0:
        return {"bridge": min(limit, n_bridge), "rail": 0, "method": "bridge_only"}
    if limit <= 1:
        # No cross-category score comparison: use triggered counts, tie bridge.
        if n_bridge >= n_rail:
            return {"bridge": min(1, n_bridge), "rail": 0, "method": "limit_one_bridge"}
        return {"bridge": 0, "rail": min(1, n_rail), "method": "limit_one_rail"}

    quota_b = quota_r = 1
    remaining = limit - 2
    total = n_bridge + n_rail
    if remaining > 0:
        ideal_b = remaining * n_bridge / total
        ideal_r = remaining - ideal_b
        add_b, add_r = math.floor(ideal_b), math.floor(ideal_r)
        leftover = remaining - add_b - add_r
        frac_b, frac_r = ideal_b - add_b, ideal_r - add_r
        if leftover:
            if frac_b >= frac_r:
                add_b += leftover
            else:
                add_r += leftover
        quota_b += add_b
        quota_r += add_r

    # Cap at the number of triggered rows; redistribute unused slots.
    quota_b = min(quota_b, n_bridge)
    quota_r = min(quota_r, n_rail)
    unused = limit - quota_b - quota_r
    while unused > 0:
        if quota_b < n_bridge and (quota_r >= n_rail or (n_bridge - quota_b) >= (n_rail - quota_r)):
            quota_b += 1
        elif quota_r < n_rail:
            quota_r += 1
        else:
            break
        unused -= 1
    return {"bridge": quota_b, "rail": quota_r, "method": "both_half_proportional_capped"}


def evaluate(evidence: List[Mapping[str, Any]], raw_config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    cfg = checked_config(raw_config)
    n = len(evidence)
    if not cfg["enabled"]:
        return {
            "version": VERSION, "enabled": False,
            "thresholds": {k: cfg[k] for k in DEFAULTS if k.startswith(("bridge_", "rail_", "support_"))},
            "budget": {k: cfg[k] for k in ("max_records", "max_fraction", "wall_budget_s")},
            "batch": {"enabled": False, "n_rows": n, "n_risk_detected": 0,
                      "n_teacher_triggered": 0, "n_teacher_selected": 0,
                      "n_budget_overflow": 0, "teacher_started": False,
                      "teacher_skip_reason": "gate_disabled",
                      "allocation": {"bridge": 0, "rail": 0, "method": "gate_disabled"}},
            "candidates": [], "selected": [],
        }

    fraction_cap = math.floor(n * cfg["max_fraction"])
    if n and cfg["max_fraction"] > 0:
        fraction_cap = max(1, fraction_cap)
    limit = min(fraction_cap, cfg["max_records"])

    decisions: List[Dict[str, Any]] = []
    for row_index, e in enumerate(evidence):
        d = evaluate_row(e, cfg)
        decisions.append({
            "row_index": row_index,
            "category": d["category"],
            "risk_detected": d["risk_detected"],
            "teacher_triggered": d["teacher_triggered"],
            "teacher_selected": False,
            "budget_overflow": False,
            "trigger_reasons": d["trigger_reasons"],
            "risk_groups": d["risk_groups"],
            "severe_reasons": d["severe_reasons"],
            "risk_priority": d["risk_priority"],
            "high_confidence": d["high_confidence"],
            "support_shortfall": d["support_shortfall"],
            "quota_category": "",
            "selection_status": "not_detected" if not d["risk_detected"] else (
                "detected_not_triggered" if not d["teacher_triggered"] else "triggered_not_selected"),
            "review_status": "not_applicable" if not d["risk_detected"] else (
                "not_triggered" if not d["teacher_triggered"] else "not_started_budget"),
        })

    bridge_trig = [d for d in decisions if d["category"] == "桥梁" and d["teacher_triggered"]]
    rail_trig = [d for d in decisions if d["category"] == "轨道" and d["teacher_triggered"]]
    allocation = _allocate(limit, len(bridge_trig), len(rail_trig))

    def take(rows: List[Dict[str, Any]], quota: int, category: str) -> None:
        rows.sort(key=lambda d: (-d["risk_priority"], str(evidence[d["row_index"]].get("vector_sha256") or ""), d["row_index"]))
        for d in rows[:quota]:
            d["teacher_selected"] = True
            d["quota_category"] = category
            d["selection_status"] = "selected"
            d["review_status"] = "not_started_p1"

    take(bridge_trig, allocation["bridge"], "bridge")
    take(rail_trig, allocation["rail"], "rail")
    for d in decisions:
        if d["teacher_triggered"] and not d["teacher_selected"]:
            d["budget_overflow"] = True
            d["selection_status"] = "triggered_not_selected"
            d["review_status"] = "not_started_budget"

    selected = [d for d in decisions if d["teacher_selected"]]
    n_triggered = sum(d["teacher_triggered"] for d in decisions)
    n_detected = sum(d["risk_detected"] for d in decisions)
    by_category: Dict[str, Dict[str, int]] = {}
    for category in ("桥梁", "轨道"):
        rows = [d for d in decisions if d["category"] == category]
        by_category[category] = {
            "n_rows": len(rows),
            "n_risk_detected": sum(d["risk_detected"] for d in rows),
            "n_teacher_triggered": sum(d["teacher_triggered"] for d in rows),
            "n_low_training_support": sum(1 for d in rows if "low_training_support" in d["risk_groups"]),
            "n_severe_support_triggered": sum(
                1 for d in rows if "high_confidence_low_training_support" in d["severe_reasons"]),
            "n_dual_triggered": sum(1 for d in rows if
                                    "uncertainty" in d["risk_groups"]
                                    and "low_training_support" in d["risk_groups"]),
            "n_dual_severe": sum(1 for d in rows if
                                 any(r.startswith(("bridge_", "rail_")) for r in d["severe_reasons"])
                                 and "high_confidence_low_training_support" in d["severe_reasons"]),
            "n_teacher_selected": sum(d["teacher_selected"] for d in rows),
            "n_budget_overflow": sum(d["budget_overflow"] for d in rows),
        }
    dual = sum(1 for d in decisions if
               "uncertainty" in d["risk_groups"] and "low_training_support" in d["risk_groups"])
    dual_severe = sum(1 for d in decisions if
                      any(r.startswith(("bridge_", "rail_")) for r in d["severe_reasons"])
                      and "high_confidence_low_training_support" in d["severe_reasons"])
    batch = {
        "enabled": True,
        "n_rows": n,
        "n_risk_detected": n_detected,
        "n_teacher_triggered": n_triggered,
        "n_teacher_selected": len(selected),
        "n_budget_overflow": max(0, n_triggered - len(selected)),
        "n_uncertainty": sum(1 for d in decisions if "uncertainty" in d["risk_groups"]),
        "n_low_training_support": sum(1 for d in decisions if "low_training_support" in d["risk_groups"]),
        "n_severe_support_triggered": sum(
            1 for d in decisions if "high_confidence_low_training_support" in d["severe_reasons"]),
        "n_dual_triggered": dual,
        "n_dual_severe": dual_severe,
        "by_category": by_category,
        "allocation": {**allocation, "limit": limit},
        "teacher_started": False,
        "teacher_skip_reason": ("no_teacher_triggered" if n_triggered == 0 else "p1_no_teacher_started"),
        "teacher_required_if_p2": bool(selected),
        "max_records": cfg["max_records"],
        "max_fraction": cfg["max_fraction"],
        "wall_budget_s": cfg["wall_budget_s"],
        "threshold_source": ("severe = 1% tail on dev+lockbox with train-only heads; "
                             "support shortfall 0.10; no online330 tuning"),
    }
    return {
        "version": VERSION, "enabled": True,
        "thresholds": {k: cfg[k] for k in DEFAULTS if k.startswith(("bridge_", "rail_", "support_"))},
        "budget": {k: cfg[k] for k in ("max_records", "max_fraction", "wall_budget_s")},
        "batch": batch,
        "candidates": decisions,
        "selected": selected,
    }


__all__ = ["VERSION", "DEFAULTS", "checked_config", "evaluate_row", "evaluate"]
