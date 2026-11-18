"""Budgeted, image-grounded second pass. No image-name or bridge-name routing.

Uncertainty is a routing heuristic, not a calibrated probability of correctness.
The trained teacher is fallible; every rejected/unfinished review retains the head.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BRIDGE, RAIL, HEALTHY = "桥梁", "轨道", "完好"
RATING = "ratingScale(1-5)"
DEFAULTS = {
    "enabled": False, "max_records": 6, "max_fraction": 0.05,
    "wall_budget_s": 120.0, "per_image_s": 12.0, "first_image_s": 25.0, "max_new_tokens": 128,
    "bridge_gray_width": 0.10, "bridge_type_margin": 0.15,
    "bridge_similarity": 0.55, "rail_margin": 0.15, "rail_entropy": 1.4,
    "base_model": "model_assets/bases/teacher_27b",
    "adapter": "model_assets/teacher_clean_full_20261011",
    "adapter_sha256": "458c2160eb78d6e14a475609100582bf4cde8155ba5757fb552c602cbf54b7a8",
    "adapter_config_sha256": "02df6134b191db1a1be55c510772bbaba664b29e2d9efb7b3ef722578a6c7099",
    "base_config_sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
    "base_index_sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
    "healthy_audit_records": 2, "positive_audit_records": 2,
    "conditional_healthy_correction": True,
    "mode": "real",
    "teacher_preflight_gpu_gib": 60.0,
    "teacher_preflight_load_s": 45.0,
}


def checked_config(raw):
    cfg = DEFAULTS | dict(raw or {})
    if not isinstance(cfg["enabled"], bool):
        raise ValueError("selective_review.enabled must be boolean")
    for key in ("max_records", "max_new_tokens", "healthy_audit_records", "positive_audit_records"):
        if type(cfg[key]) is not int or cfg[key] < (1 if key == "max_new_tokens" else 0):
            raise ValueError(key)
    for key in ("max_fraction", "wall_budget_s", "per_image_s", "first_image_s", "bridge_gray_width",
                "bridge_type_margin", "bridge_similarity", "rail_margin", "rail_entropy"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(key)
    if cfg["max_fraction"] > 1 or min(cfg["wall_budget_s"], cfg["per_image_s"], cfg["first_image_s"]) <= 0:
        raise ValueError("Invalid review fraction/time budget")
    if type(cfg["conditional_healthy_correction"]) is not bool:
        raise ValueError("conditional_healthy_correction must be boolean")
    if str(cfg.get("mode") or "real") not in {"real", "preflight"}:
        raise ValueError("selective_review.mode must be real or preflight")
    for key in ("teacher_preflight_gpu_gib", "teacher_preflight_load_s"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(key)
    # Unconditional healthy overrides remain prohibited.
    if cfg.get("allow_disease_to_healthy") or cfg.get("aerial_healthy_gate"):
        raise ValueError("Unconditional healthy overrides are not supported")
    return cfg


def uncertainty(e, cfg):
    """All views are eligible; type ambiguity is meaningful near/above presence."""
    reasons, scores = [], []
    if not e.get("embedding_ok"):
        raise ValueError("2B full visual coverage must succeed before review")
    if e.get("low_training_support"):
        reasons.append("low_training_vector_support")
        scores.append(2.0 + float(e["training_support_threshold"]) - float(e["training_support_similarity"]))
    if e["question_category"] == BRIDGE:
        p, threshold = float(e["p_disease"]), float(e["threshold"])
        distance = abs(p - threshold)
        width = cfg["bridge_gray_width"]
        if distance <= width:
            reasons.append("bridge_presence_boundary")
            scores.append(1.0 - distance / max(width, 1e-9))
        # Disease-only case distance is not an all-training OOD detector.
        if p >= 0.5:
            margin, sim = float(e["type_margin"]), float(e["nearest_similarity"])
            if margin < cfg["bridge_type_margin"]:
                reasons.append("bridge_case_vote_disagreement")
                scores.append(1.0 + cfg["bridge_type_margin"] - margin)
            if sim < cfg["bridge_similarity"]:
                reasons.append("bridge_low_disease_case_similarity")
                scores.append(1.0 + cfg["bridge_similarity"] - sim)
    elif e["question_category"] == RAIL:
        margin, entropy = float(e["probability_margin"]), float(e["entropy"])
        if margin < cfg["rail_margin"]:
            reasons.append("rail_low_class_margin")
            scores.append(1.0 + cfg["rail_margin"] - margin)
        if entropy > cfg["rail_entropy"]:
            reasons.append("rail_high_class_entropy")
            scores.append(1.0 + (entropy - cfg["rail_entropy"]) / max(1.0, math.log(22)))
    else:
        raise ValueError("Unsupported review category")
    return max(scores, default=0.0), reasons


def route(evidence, raw):
    cfg = checked_config(raw)
    fraction_cap = math.floor(len(evidence) * cfg["max_fraction"])
    if evidence and cfg["max_fraction"] > 0:
        fraction_cap = max(1, fraction_cap)
    limit = min(fraction_cap, cfg["max_records"]) if cfg["enabled"] else 0
    decisions = []
    for i, e in enumerate(evidence):
        score, reasons = uncertainty(e, cfg)
        decisions.append({"row_index": i, "priority": score, "reasons": reasons,
                          "selected": False, "rank": 0})
    order_key = lambda d: (-d["priority"], evidence[d["row_index"]]["vector_sha256"], d["row_index"])
    ordered = sorted((d for d in decisions if d["reasons"]),
                     key=order_key)
    # Reserve a small audit for confidently healthy images. Visual-view strata
    # give aerial images a chance; none of these decisions forces them healthy.
    healthy = [d for d in decisions if evidence[d["row_index"]].get("head_label")==HEALTHY]
    healthy.sort(key=lambda d:(evidence[d["row_index"]]["vector_sha256"],d["row_index"]))
    # On a small test batch, preserve at least one genuine uncertainty/OOD slot.
    audit_limit=min(max(0,limit-(1 if ordered else 0)),cfg["healthy_audit_records"])
    audit=[]
    for predicate in (lambda e:e['question_category']==BRIDGE and e.get('visual_view')=='aerial',
                      lambda e:not(e['question_category']==BRIDGE and e.get('visual_view')=='aerial')):
        pool=[d for d in healthy if predicate(evidence[d['row_index']]) and d not in audit]
        if pool and len(audit)<audit_limit:audit.append(pool[0])
    audit += [d for d in healthy if d not in audit][:max(0,audit_limit-len(audit))]
    if not healthy and not ordered:
        other=[d for d in decisions if evidence[d['row_index']].get('head_label')]
        other.sort(key=lambda d:(evidence[d['row_index']]['vector_sha256'],d['row_index']))
        audit=other[:audit_limit]
    audit_ids={d['row_index'] for d in audit}
    for d in audit:
        reason='healthy_prediction_audit' if evidence[d['row_index']].get('head_label')==HEALTHY else 'confident_prediction_audit'
        d['reasons']=list(d['reasons'])+[reason]
    # Reserve bounded positive audits across visually predicted strata. This
    # can catch false positives missed by the generic ambiguity ranking.
    positives=[d for d in decisions if d["row_index"] not in audit_ids and evidence[d["row_index"]].get("head_label") not in (None,HEALTHY)]
    def positive_distance(d):
        e=evidence[d["row_index"]]
        if e["question_category"]==BRIDGE:
            return abs(float(e["p_disease"])-float(e["threshold"]))/max(cfg["bridge_gray_width"],1e-9)
        return float(e["probability_margin"])/max(cfg["rail_margin"],1e-9)
    positives.sort(key=lambda d:(positive_distance(d),*order_key(d)))
    positive_limit=min(cfg["positive_audit_records"],max(0,limit-len(audit)-(1 if ordered else 0)))
    positive=[]; strata=set()
    for d in positives:
        e=evidence[d["row_index"]]
        stratum=(e["question_category"], e.get("visual_view","rail"))
        if stratum not in strata and len(positive)<positive_limit:
            positive.append(d);strata.add(stratum)
    positive += [d for d in positives if d not in positive][:max(0,positive_limit-len(positive))]
    for d in positive:
        d["reasons"]=list(d["reasons"])+["positive_prediction_audit"]
    reserved_ids=audit_ids|{d["row_index"] for d in positive}
    ordered=[d for d in ordered if d["row_index"] not in reserved_ids]
    uncertain_limit=max(0,limit-len(audit)-len(positive))
    bridge = [d for d in ordered if evidence[d["row_index"]]["question_category"] == BRIDGE]
    rail = [d for d in ordered if evidence[d["row_index"]]["question_category"] == RAIL]
    chosen = bridge[:math.ceil(uncertain_limit / 2)]
    chosen += rail[:max(0, uncertain_limit - len(chosen))]
    used = {d["row_index"] for d in chosen}
    chosen += [d for d in ordered if d["row_index"] not in used][:max(0, uncertain_limit - len(chosen))]
    chosen += positive+audit
    for rank, d in enumerate(chosen, 1):
        d.update(selected=True, rank=rank)
    return decisions


def make_prompt(row, evidence, labels):
    # The teacher must actually inspect the image independently. No student
    # prediction, neighbour label, filename, bridge identity or gold is supplied.
    return (
        f"请独立检查这张{row['questionCategory']}巡检图片，只根据图中可见信息填写巡检台账。\n"
        "本任务登记病害，也登记划痕、沥青摊铺接缝等可见外观痕迹。"
        "不要把正常标线、阴影、正常伸缩缝误判为损伤。图像证据不足就标记不确定。\n"
        "type只能从以下闭集选一个完整标签：" + json.dumps(sorted(labels), ensure_ascii=False) + "\n"
        "轨道rating必须为空字符串；桥梁完好rating为空；桥梁非完好rating填写1至5的字符串，"
        "看图无法确定标度时rating留空。uncertain表示无法可靠确定是否完好或具体类型。"
        "图像模糊、遮挡或分辨率不足时不能仅因未见损伤就判完好，应返回uncertain=true。"
        "只有检查区域清晰可见且没有闭集中的登记缺陷时才返回完好。\n"
        "仅输出一个JSON对象，不含思考或Markdown。键恰为type,rating,evidence,uncertain。"
        "evidence用一句短句描述本图可见依据，uncertain为JSON布尔值。"
    )


def healthy_correction_eligible(evidence, raw_config):
    """A conservative student-evidence condition, not a correctness probability.

    Bridge: presence boundary only; type disagreement does not prove absence.
    Rail: healthy downgrades remain blocked after a labelled counterexample.
    View, filename, bridge identity and leaderboard feedback are not inputs.
    """
    cfg=checked_config(raw_config)
    if not cfg["conditional_healthy_correction"] or not evidence.get("embedding_ok"):
        return False
    if evidence["question_category"]==BRIDGE:
        p=float(evidence["p_disease"]); threshold=float(evidence["threshold"])
        return threshold <= p <= threshold+cfg["bridge_gray_width"]
    return False


def parse_and_merge(raw, row, evidence, labels, bridge, raw_config=None):
    """Keep identity/wording fixed; a model-produced JSON is still only a proposal."""
    keep = dict(row)
    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate JSON key")
            obj[key] = value
        return obj
    try:
        obj = json.loads(raw, object_pairs_hook=unique_object)
    except (ValueError, TypeError):
        return keep, "invalid_json", None
    if not isinstance(obj, dict) or set(obj) != {"type", "rating", "evidence", "uncertain"}:
        return keep, "invalid_schema", obj
    if not all(isinstance(obj[k], str) for k in ("type", "rating", "evidence")) or type(obj["uncertain"]) is not bool:
        return keep, "invalid_field_types", obj
    if obj["type"] not in labels:
        return keep, "outside_closed_labels", obj
    if obj["uncertain"]:
        return keep, "teacher_uncertain", obj
    if not obj["evidence"].strip():
        return keep, "missing_visual_evidence", obj
    cat, label, rating = row["questionCategory"], obj["type"], obj["rating"]
    # The rail ledger has no grade field. Enforce its empty field locally,
    # exactly as bridge identity and representative wording are rendered locally.
    # Preserve the unmodified teacher proposal in the audit for diagnosis.
    if cat == RAIL:
        rating = ""
    if label == HEALTHY and rating != "":
        return keep, "invalid_rating", obj
    if cat == BRIDGE and label != HEALTHY and rating not in ("1", "2", "3", "4", "5"):
        # The teacher can recognize a type without inventing a grade. Preserve
        # the existing case-vote grade, with this source recorded separately.
        fallback = row[RATING] or str(evidence.get('rating') or '')
        if rating=='' and fallback in ('1','2','3','4','5'):
            rating=fallback
        else:
            return keep, "invalid_rating", obj
    if row["defectType"] != HEALTHY and label == HEALTHY:
        if not healthy_correction_eligible(evidence,raw_config):
            return keep, "healthy_correction_student_evidence_insufficient", obj
    if cat == BRIDGE:
        if label == HEALTHY:
            desc = bridge.healthy_by_view.get(evidence.get("visual_view"), bridge.healthy_medoid)
        else:
            desc = bridge.type_medoid.get(label)
        if not desc:
            return keep, "missing_fixed_description", obj
        keep["defectDescription"] = desc
    keep["defectType"], keep[RATING] = label, rating
    return keep, "accepted_changed" if keep != row else "accepted_unchanged", obj


def preflight_teacher(cfg, selected_n, *, gpu_free_gib=None):
    """Check assets/SHA/GPU/time without importing or loading the 27B model.

    This is a P2-A preflight. It never constructs a model, never starts a
    worker and never writes TongAgents call logs.
    """
    reasons = []
    checks = []

    def check(name, path, expected):
        path = Path(path)
        exists = path.is_file()
        actual = None
        ok = False
        if exists:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            ok = (expected == actual) if expected else True
        checks.append({"name": name, "path": str(path), "exists": exists,
                       "expected_sha256": expected, "actual_sha256": actual, "ok": ok})
        return exists, ok

    base = Path(cfg["base_model"])
    adapter = Path(cfg["adapter"])
    base_config_ok, base_config_sha_ok = check("base_config", base / "config.json",
                                               cfg.get("base_config_sha256"))
    base_index_ok, base_index_sha_ok = check("base_index", base / "model.safetensors.index.json",
                                             cfg.get("base_index_sha256"))
    adapter_ok, adapter_sha_ok = check("adapter_model", adapter / "adapter_model.safetensors",
                                       cfg.get("adapter_sha256"))
    adapter_cfg_ok, adapter_cfg_sha_ok = check("adapter_config", adapter / "adapter_config.json",
                                               cfg.get("adapter_config_sha256"))
    assets_ok = all([base_config_ok, base_index_ok, adapter_ok, adapter_cfg_ok])
    sha_ok = all([base_config_sha_ok, base_index_sha_ok, adapter_sha_ok, adapter_cfg_sha_ok])
    if not assets_ok:
        reasons.append("teacher_assets_missing")
    if not sha_ok:
        reasons.append("teacher_sha_mismatch")

    if gpu_free_gib is None:
        try:
            import torch

            if torch.cuda.is_available():
                # The worker loads onto logical device 0 after CUDA_VISIBLE_DEVICES.
                gpu_free_gib = round(torch.cuda.mem_get_info(0)[0] / 2**30, 3)
        except (ImportError, OSError, RuntimeError):
            gpu_free_gib = None
    gpu_ok = None if gpu_free_gib is None else gpu_free_gib >= float(cfg["teacher_preflight_gpu_gib"])
    if gpu_ok is None:
        reasons.append("gpu_query_unavailable")
    elif not gpu_ok:
        reasons.append("gpu_free_insufficient")

    estimate = (float(cfg["teacher_preflight_load_s"]) + float(cfg["first_image_s"])
                + max(0, int(selected_n) - 1) * float(cfg["per_image_s"]))
    time_ok = float(cfg["wall_budget_s"]) >= estimate
    if not time_ok:
        reasons.append("time_budget_insufficient")

    resource_ok = bool(assets_ok and sha_ok and time_ok and gpu_ok is True)
    return {
        "selected_n": int(selected_n),
        "base_model": str(base),
        "adapter": str(adapter),
        "assets_ok": bool(assets_ok),
        "sha_ok": bool(sha_ok),
        "gpu_free_gib": gpu_free_gib,
        "gpu_required_gib": float(cfg["teacher_preflight_gpu_gib"]),
        "gpu_ok": gpu_ok,
        "time_budget_s": float(cfg["wall_budget_s"]),
        "time_estimate_s": round(estimate, 3),
        "time_ok": bool(time_ok),
        "resource_ok": resource_ok,
        "reasons": reasons,
        "checks": checks,
        "model_loaded": False,
        "teacher_started": False,
        "note": "P2-A preflight only; no 27B import/load/worker call.",
    }


def run_worker(request_path, response_path, cfg, worker_path):
    """Own child only, including its loading in the budget; no resident service."""
    start = time.perf_counter()
    log_path = response_path.with_suffix(".log")
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    cmd = [str(cfg.get("python") or sys.executable), str(worker_path),
           str(request_path), str(response_path)]
    status, returncode = "completed", None
    with log_path.open("wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=(os.name == "posix"))
        try:
            returncode = proc.wait(timeout=cfg["wall_budget_s"])
            if returncode:
                status = "worker_failed"
        except subprocess.TimeoutExpired:
            status = "wall_budget_exhausted"
            if os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                proc.terminate()
            try:
                returncode = proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()
                returncode = proc.wait(timeout=5)
    return {"status": status, "returncode": returncode,
            "elapsed_s": time.perf_counter() - start, "log": log_path.name}


def apply_selective_review(rows, evidence, paths, allowed, bridge, raw_config, output_dir,
                           *, worker_runner=None, selected_rows=None, run_mode=None):
    start = time.perf_counter()
    cfg = checked_config(raw_config)
    mode = str(run_mode or cfg.get("mode") or "real")
    if mode not in {"real", "preflight"}:
        raise ValueError("run_mode must be real or preflight")
    if not (len(rows) == len(evidence) == len(paths)):
        raise ValueError("Review alignment mismatch")

    root = Path(output_dir) / "selective_review"
    root.mkdir(exist_ok=False)
    if selected_rows is None:
        decisions = route(evidence, cfg)
        selected = sorted((d for d in decisions if d["selected"]), key=lambda d: d["rank"])
        routing_source = "legacy_route"
    else:
        selected_rows = [int(i) for i in selected_rows]
        if len(set(selected_rows)) != len(selected_rows):
            raise ValueError("Duplicate teacher_selected row")
        if any(i < 0 or i >= len(rows) for i in selected_rows):
            raise ValueError("teacher_selected row out of range")
        selected = [{"row_index": i, "rank": rank,
                     "reasons": ["teacher_selected_by_risk_gate"],
                     "priority": None}
                    for rank, i in enumerate(selected_rows, 1)]
        decisions = selected
        routing_source = "risk_gate"
    (root / "routing.json").write_text(
        json.dumps({"source": routing_source, "selected": selected}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    summary = {
        "enabled": cfg["enabled"], "mode": mode, "routing_source": routing_source,
        "n_eligible": len(selected), "n_selected": len(selected),
        "teacher_selected": len(selected), "teacher_started": False,
        "teacher_completed": 0, "teacher_accepted": 0, "teacher_changed": 0,
        "n_completed": 0, "n_changed": 0,
        "max_records": cfg["max_records"], "max_fraction": cfg["max_fraction"],
        "wall_budget_s": cfg["wall_budget_s"], "filename_routing": False,
        "uncertainty_is_accuracy": False, "quality_status": "experimental_not_proven",
        "conditional_healthy_correction": cfg["conditional_healthy_correction"],
        "positive_audit_records": cfg["positive_audit_records"],
        "healthy_audit_records": cfg["healthy_audit_records"],
    }
    if not selected:
        summary.update(status="no_teacher_selected", skip_reason="no_teacher_selected",
                       elapsed_s=time.perf_counter() - start)
        (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return rows, summary

    if mode == "preflight":
        preflight = preflight_teacher(cfg, len(selected))
        (root / "preflight.json").write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        audit = []
        for d in selected:
            i = d["row_index"]
            audit.append({
                **d, "status": "not_started_preflight",
                "before": rows[i], "after": dict(rows[i]),
                "teacher_proposal": None, "teacher_started": False,
                "teacher_completed": False, "teacher_accepted": False,
                "review_status": "not_started_preflight",
                "teacher_elapsed_s": None,
            })
        summary.update(
            status=("preflight_passed_not_started" if preflight["resource_ok"]
                    else "preflight_failed_not_started"),
            skip_reason=("p2a_preflight_only" if preflight["resource_ok"]
                         else "preflight_failed:" + ",".join(preflight["reasons"])),
            preflight=preflight, teacher_started=False, teacher_completed=0,
            teacher_accepted=0, teacher_changed=0,
            statuses=dict(Counter(a["status"] for a in audit)),
            elapsed_s=time.perf_counter() - start,
        )
        (root / "decisions.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return rows, summary

    preflight = preflight_teacher(cfg, len(selected))
    (root / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not preflight["resource_ok"]:
        audit = []
        for d in selected:
            i = d["row_index"]
            audit.append({
                **d, "status": "not_started_preflight",
                "before": rows[i], "after": dict(rows[i]),
                "teacher_proposal": None, "teacher_started": False,
                "teacher_completed": False, "teacher_accepted": False,
                "review_status": "not_started_preflight",
                "teacher_elapsed_s": None,
            })
        summary.update(
            status="preflight_failed_not_started",
            skip_reason="preflight_failed:" + ",".join(preflight["reasons"]),
            preflight=preflight, teacher_started=False, teacher_completed=0,
            teacher_accepted=0, teacher_changed=0,
            statuses=dict(Counter(a["status"] for a in audit)),
            elapsed_s=time.perf_counter() - start,
        )
        (root / "decisions.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return rows, summary

    jobs = []
    for d in selected:
        i = d["row_index"]
        jobs.append({"row_index": i, "image_path": paths[i],
                     "prompt": make_prompt(rows[i], evidence[i], allowed[rows[i]["questionCategory"]])})
    payload = {"config": cfg, "jobs": jobs}
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    payload["request_sha256"] = digest
    request_path, response_path = root / "request.json", root / "responses.jsonl"
    request_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        runner = worker_runner or run_worker
        runtime = runner(request_path, response_path, cfg, Path(__file__).with_name("selective_teacher_worker.py"))
    except (OSError, subprocess.SubprocessError) as exc:
        runtime = {"status": "worker_launch_failed", "error": type(exc).__name__}
    teacher_started = runtime.get("status") not in {"worker_launch_failed"}
    responses = {}
    if response_path.exists():
        for line in response_path.read_bytes().splitlines(keepends=True):
            if not line.endswith(b'\n'):
                continue
            try:
                item = json.loads(line.decode('utf-8'))
                i = item["row_index"]
                if type(i) is int and item.get("request_sha256") == digest:
                    responses.setdefault(i, []).append(item)
            except (ValueError, KeyError, TypeError):
                continue
    merged, audit = [dict(r) for r in rows], []
    for d in selected:
        i = d["row_index"]
        found = responses.get(i, [])
        status, proposal = "review_unfinished_keep_head", None
        completed = False
        if len(found) > 1:
            status = "duplicate_response_keep_head"
        elif found and found[0].get("ok") is True:
            summary["n_completed"] += 1
            completed = True
            merged[i], status, proposal = parse_and_merge(found[0].get("raw"), rows[i], evidence[i],
                                                         allowed[rows[i]["questionCategory"]], bridge, cfg)
        elif found:
            status = "teacher_error_keep_head"
        accepted = status.startswith("accepted")
        audit.append({**d, "healthy_correction_eligible":healthy_correction_eligible(evidence[i],cfg),
                      "status": status, "before": rows[i], "after": merged[i],
                      "teacher_proposal": proposal,
                      "teacher_started": teacher_started, "teacher_completed": completed,
                      "teacher_accepted": accepted, "teacher_status": status,
                      "review_status": status,
                      "rating_source": ('not_applicable' if rows[i]['questionCategory']==RAIL or merged[i]['defectType']==HEALTHY
                          else ('teacher_proposal' if status.startswith('accepted') and proposal and proposal.get('rating') in ('1','2','3','4','5')
                                else ('retained_case_vote' if status.startswith('accepted') and proposal and proposal.get('rating')=='' else 'head'))),
                      "teacher_elapsed_s": found[0].get("elapsed_s") if len(found) == 1 else None})
    accepted_count = sum(a["teacher_accepted"] for a in audit)
    changed_count = sum(a["status"] == "accepted_changed" for a in audit)
    summary.update(n_healthy_corrections=sum(a["before"]["defectType"]!=HEALTHY and a["after"]["defectType"]==HEALTHY for a in audit),
                   runtime=runtime, n_changed=changed_count, preflight=preflight,
                   teacher_started=teacher_started, teacher_completed=summary["n_completed"],
                   teacher_accepted=accepted_count, teacher_changed=changed_count,
                   status=runtime.get("status", "completed"),
                   skip_reason=(None if runtime.get("status") == "completed" else runtime.get("status")),
                   statuses=dict(Counter(a["status"] for a in audit)),
                   elapsed_s=time.perf_counter()-start)
    (root / "decisions.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged, summary
