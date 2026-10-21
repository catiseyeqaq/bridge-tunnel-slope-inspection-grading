"""All-image visual inference. Image names never select a disease decision path."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from track4_contract_agent.runtime.bridge_case_head import BridgeCaseHead

SEVEN = ("questionCategory", "bridgeName", "defectLocation", "filename",
         "defectType", "defectDescription", "ratingScale(1-5)")
HEALTHY = "完好"
BRIDGE = "桥梁"
RAIL = "轨道"


class AllViewBridgeHead(BridgeCaseHead):
    """The same presence LR and disease kNN, fitted across all bridge views.

    View prediction chooses healthy wording only. It cannot suppress perception
    or force a healthy disease label. Inputs to decide() are image vectors only.
    """
    def __init__(self, arrays, meta):
        super().__init__(arrays, meta)
        self.view_coef = np.asarray(arrays["view_coef"], dtype=np.float32)
        self.view_intercept = np.asarray(arrays["view_intercept"], dtype=np.float32)
        self.view_classes = [str(v) for v in arrays["view_classes"]]
        self.healthy_by_view = dict(meta["healthy_by_view"])
        self.model_meta = dict(meta)
        if self.hd_coef.shape != (1, 2048) or self.view_coef.shape != (len(self.view_classes), 2048):
            raise ValueError("All-view head dimensions do not match the 2B embedding contract")
        if not 0 < self.threshold < 1 or not len(self.dis_vectors):
            raise ValueError("Invalid all-view head threshold or empty disease cases")

    def decide(self, vectors):
        decisions = super().decide(vectors)
        x = np.asarray(vectors, dtype=np.float32)
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
        logits = x @ self.view_coef.T + self.view_intercept
        prob = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob /= prob.sum(axis=1, keepdims=True)
        for decision, pp in zip(decisions, prob):
            j = int(pp.argmax())
            decision["visual_view"] = self.view_classes[j]
            decision["view_probability"] = float(pp[j])
        return decisions


def checked_inputs(meta, input_dir):
    """Validate full identities and every image before any result can be written."""
    root = Path(input_dir).resolve()
    if not isinstance(meta, list):
        raise ValueError("Input metadata must be a JSON array of row objects")
    identities, ids, paths = set(), set(), []
    for i, m in enumerate(meta):
        if not isinstance(m, dict):
            raise ValueError(f"Input metadata row {i} must be an object")
        if not isinstance(m.get("questionCategory"), str) or not isinstance(m.get("filename"), str):
            raise ValueError(f"Category and filename must be strings at index {i}")
        if m["questionCategory"] == BRIDGE and (not isinstance(m.get("bridgeName"), str) or not m["bridgeName"].strip()):
            raise ValueError(f"Bridge identity is missing at index {i}; provide the current input's legal bridgeName")
        if not isinstance(m.get("imageRelativePath"), str) or not m["imageRelativePath"].strip():
            raise FileNotFoundError(f"Missing imageRelativePath at index {i}")
        cat, name, fn = (str(m.get(k) or "") for k in ("questionCategory", "bridgeName", "filename"))
        identity = (cat, name, fn)
        if cat not in (BRIDGE, RAIL) or not fn or identity in identities:
            raise ValueError(f"Unsupported or duplicate row identity at index {i}")
        identities.add(identity)
        record_id = str(m.get("record_id") or "row:" + str(i))
        if record_id in ids:
            raise ValueError(f"Duplicate record_id at index {i}")
        ids.add(record_id)
        rel = Path(str(m.get("imageRelativePath") or ""))
        path = (root / rel).resolve()
        if rel.is_absolute() or root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Invalid or missing image at index {i}: {rel}")
        paths.append(str(path))
    if not paths:
        raise ValueError("Empty input metadata")
    return paths


def checked_vectors(vectors, count):
    if len(vectors) != count or any(v is None for v in vectors):
        raise RuntimeError(f"Visual coverage incomplete: received {len(vectors)}/{count} vectors")
    x = np.asarray(vectors, dtype=np.float32)
    if x.shape != (count, 2048) or not np.isfinite(x).all() or np.any(np.linalg.norm(x, axis=1) < 1e-8):
        raise RuntimeError("Visual coverage failed: malformed, non-finite or zero image vector")
    return x


def infer_rows(meta, vectors, rail_head, bridge_head):
    """Position-aligned predictions; metadata is used for output identity only."""
    x = checked_vectors(vectors, len(meta))
    result, evidence = [None] * len(meta), [None] * len(meta)
    for category in (RAIL, BRIDGE):
        positions = [i for i, m in enumerate(meta) if m["questionCategory"] == category]
        if not positions:
            continue
        decisions = (rail_head.predict_detailed(x[positions]) if category == RAIL
                     else bridge_head.decide(x[positions]))
        if len(decisions) != len(positions):
            raise RuntimeError("Head returned incomplete decisions")
        for i, d in zip(positions, decisions):
            m = meta[i]
            if category == RAIL:
                label, rating, description = d["label"], "", ""
            else:
                label = d["type"] if d["p_disease"] >= bridge_head.threshold else HEALTHY
                rating = d["rating"] if label != HEALTHY else ""
                description = (bridge_head.type_medoid.get(label, "") if label != HEALTHY
                               else bridge_head.healthy_by_view.get(d["visual_view"], bridge_head.healthy_medoid))
                if not description:
                    raise RuntimeError(f"Missing training-derived description for bridge label {label}")
            result[i] = dict(zip(SEVEN, (category, "" if category == RAIL else str(m.get("bridgeName") or ""),
                                        "", str(m["filename"]), label, description, rating)))
            evidence[i] = {"row_index": i, "record_id": str(m.get("record_id") or "row:" + str(i)),
                           "question_category": category, "embedding_ok": True,
                           "route_source": "rail_linear_head" if category == RAIL else "bridge_all_view_head",
                           "head_label": label, "head_rating": rating, "is_aerial_default": False,
                           "vector_sha256": hashlib.sha256(x[i].tobytes()).hexdigest(), **d}
            if category == BRIDGE:
                evidence[i]["threshold"] = bridge_head.threshold
    if any(r is None for r in result) or any(e is None for e in evidence):
        raise RuntimeError("A row escaped visual inference")
    return result, evidence


def validate_result(rows, meta, allowed):
    if len(rows) != len(meta):
        raise RuntimeError("Result row count mismatch")
    for i, (r, m) in enumerate(zip(rows, meta)):
        cat = m["questionCategory"]
        if tuple(r) != SEVEN or r["filename"] != m["filename"] or r["questionCategory"] != cat:
            raise RuntimeError(f"Seven-field identity contract failed at {i}")
        if not all(isinstance(v, str) for v in r.values()) or r["defectType"] not in allowed[cat]:
            raise RuntimeError(f"Closed label/string contract failed at {i}")
        if r["ratingScale(1-5)"] not in ("", "1", "2", "3", "4", "5"):
            raise RuntimeError(f"Rating contract failed at {i}")
        if cat == RAIL and any(r[k] for k in ("bridgeName", "defectLocation", "defectDescription", "ratingScale(1-5)")):
            raise RuntimeError(f"Rail empty-field contract failed at {i}")
        if cat == BRIDGE and (r["bridgeName"] != str(m.get("bridgeName") or "") or not r["defectDescription"]):
            raise RuntimeError(f"Bridge identity/description contract failed at {i}")
        if cat == BRIDGE and (not r["bridgeName"].strip() or (r["defectType"] == HEALTHY and r["ratingScale(1-5)"])):
            raise RuntimeError(f"Bridge identity/healthy-rating contract failed at {i}")


def run_full_visual(input_dir: Path, metadata: Path, output_dir: Path, config: Mapping[str, Any], agent_root: Path):
    started = time.perf_counter()
    meta = json.loads(Path(metadata).read_text(encoding="utf-8"))
    paths = checked_inputs(meta, input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "result.json").exists():
        raise FileExistsError("Use a fresh output directory; refusing to leave stale successful results")
    from track4_contract_agent.models.local_qwen_vl_retrieval import get_local_embedder
    def _allowed_labels(root):
        result={}
        for category,name in ((BRIDGE,'bridge_allowed_labels.json'),(RAIL,'rail_allowed_labels.json')):
            d=json.loads((root/'data/ontology'/name).read_text(encoding='utf-8'))
            result[category]=d['allowed_labels']
        return result
    # Fresh JPEGs for every image in this run; no prediction/vector reuse.
    from concurrent.futures import ThreadPoolExecutor
    from track4_contract_agent.models.image_embedding import resolve_embedder
    early_cfg = dict(config["heads_only"])
    embedder = resolve_embedder(str(early_cfg.get("embedder") or "qwen3_vl_embedding_2b"))
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="fresh-jpeg") as pool:
        pending = pool.submit(embedder._prepare_data_urls, paths)
        get_local_embedder()  # Still entirely within the run.py timer.
        embedder._prepared_image_urls = dict(zip(paths, pending.result()))
    loaded = time.perf_counter()
    from track4_contract_agent.runtime.rail_embed_head import RailEmbedHead
    from track4_contract_agent.runtime.tongagents_bridge import bootstrap_tongagents, heads_with_shell, review_with_shell
    cfg = dict(config["heads_only"])
    full_cfg = dict(config["full_visual"])
    rail = RailEmbedHead.load(agent_root / cfg["rail_weights"])
    bridge = AllViewBridgeHead.load(agent_root / full_cfg["bridge_weights"])
    from track4_contract_agent.runtime.training_support import TrainingSupport
    support=TrainingSupport(agent_root/full_cfg['support_weights'])
    allowed = _allowed_labels(agent_root)
    shell = bootstrap_tongagents(dict(config))
    # embedder above holds only JPEGs freshly prepared within this entry.
    holder = {}

    def inspect_all():
        before = time.perf_counter()
        # Keep rail image batches identical to the legacy rail-only call.
        # Mixing a bridge image into a partially filled rail batch can change
        # padding and flip borderline bf16 predictions despite unchanged heads.
        # Category grouping never skips a row and does not inspect filenames.
        vectors_by_row = [None] * len(meta)
        for category in (RAIL, BRIDGE):
            positions = [i for i, m in enumerate(meta) if m["questionCategory"] == category]
            if not positions:
                continue
            batch_vectors = checked_vectors(embedder.embed_image_paths([paths[i] for i in positions]), len(positions))
            for i, vector in zip(positions, batch_vectors):
                vectors_by_row[i] = vector
        vectors = checked_vectors(vectors_by_row, len(meta))
        after_embed = time.perf_counter()
        rows, evidence = infer_rows(meta, vectors, rail, bridge)
        support_info=support.annotate(vectors,evidence)
        validate_result(rows, meta, allowed)
        risk_summary = None
        risk_report = None
        risk_cfg = dict(config.get("risk_gate") or {})
        if risk_cfg.get("enabled", False):
            from track4_contract_agent.runtime.risk_gate import evaluate as evaluate_risk
            risk_report = evaluate_risk(evidence, risk_cfg)
            (output_dir / "risk_gate.json").write_text(
                json.dumps(risk_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            risk_summary = risk_report["batch"]
        holder.update(rows=rows, evidence=evidence, vectors=vectors, risk_report=risk_report)
        return {"mode": "full_visual", "run_mode": str(config.get("run_mode") or ""),
                "n_rows": len(meta), "n_embedded": len(vectors),
                "n_skipped_visual": 0, "n_aerial_default": 0,
                "training_support":support_info,
                "risk_gate": risk_summary,
                "batch_grouping": "category_only; rail batch order retained; no filename routing",
                "embedding_and_preprocess_s": after_embed - before,
                "head_and_contract_s": time.perf_counter() - after_embed,
                "rail_rows": sum(m["questionCategory"] == RAIL for m in meta),
                "bridge_rows": sum(m["questionCategory"] == BRIDGE for m in meta),
                "bridge_disease": sum(r["questionCategory"] == BRIDGE and r["defectType"] != HEALTHY for r in rows)}

    info = heads_with_shell(inspect_all)
    # The SDK bridge may catch errors; never allow an empty or partial tool result.
    if "rows" not in holder or not isinstance(info, dict) or info.get("n_embedded") != len(meta):
        raise RuntimeError("TongAgents visual tool did not complete all rows")
    review_cfg = dict(config.get("selective_review") or {})
    for key in ('base_model','adapter'):
        value=review_cfg.get(key)
        if value and not Path(value).is_absolute():review_cfg[key]=str((agent_root/value).resolve())
    if review_cfg.get("enabled", False):
        risk_report = holder.get("risk_report")
        selected_rows = ([int(d["row_index"]) for d in risk_report.get("selected", [])]
                         if risk_report is not None else None)
        if selected_rows == []:
            # P2-A: zero gate-selected rows means no asset check, no teacher
            # import/load, no worker and no TongAgents teacher call.
            info["selective_review"] = {
                "enabled": True,
                "mode": str(review_cfg.get("mode") or "real"),
                "routing_source": "risk_gate",
                "teacher_selected": 0, "teacher_started": False,
                "teacher_completed": 0, "teacher_accepted": 0, "teacher_changed": 0,
                "status": "no_teacher_selected", "skip_reason": "no_teacher_selected",
                "asset_check_skipped": True,
            }
            info["final_bridge_disease"] = info.get("bridge_disease")
        else:
            from track4_contract_agent.runtime.selective_review import apply_selective_review
            (output_dir / "head_result.json").write_text(
                json.dumps(holder["rows"], ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            review_mode = str(review_cfg.get("mode") or "real")
            if review_mode == "preflight":
                # Preflight is a direct Python call: it must not be recorded as
                # a real TongAgents teacher_review invocation.
                holder['rows'], review_info = apply_selective_review(
                    holder['rows'], holder['evidence'], paths, allowed, bridge,
                    review_cfg, output_dir, selected_rows=selected_rows,
                    run_mode="preflight")
                info['selective_review'] = review_info
            else:
                def review_selected():
                    holder['rows'],review_info=apply_selective_review(
                        holder['rows'],holder['evidence'],paths,allowed,bridge,review_cfg,output_dir,
                        selected_rows=selected_rows, run_mode="real")
                    return review_info
                info['selective_review']=review_with_shell(review_selected)
            validate_result(holder["rows"], meta, allowed)
            info["final_bridge_disease"] = sum(r["questionCategory"] == BRIDGE and
                                               r["defectType"] != HEALTHY for r in holder["rows"])
    else:
        info["final_bridge_disease"] = info.get("bridge_disease")
    from track4_contract_agent.models.local_qwen_vl_retrieval import PROFILE_SECONDS
    info["runtime_profile"] = dict(PROFILE_SECONDS)
    info.update(tongagents=shell.snapshot(), model_load_and_prefetch_s=loaded-started,
                bridge_head_sha256=hashlib.sha256((agent_root/full_cfg["bridge_weights"]).read_bytes()).hexdigest(),
                decision_inputs="image_embedding_only; category selects rail vs bridge",
                filename_routing=False, bridge_training=bridge.model_meta.get("training_scope"))
    if full_cfg.get("save_vectors", False):
        np.save(output_dir / "visual_vectors.f32.npy", holder["vectors"])
    (output_dir / "visual_evidence.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in holder["evidence"]), encoding="utf-8")
    (output_dir / "heads_only.json").write_text(json.dumps(info, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    # Publish result only after coverage and the entire contract have succeeded.
    tmp = output_dir / "result.json.tmp"
    tmp.write_text(json.dumps(holder["rows"], ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    os.replace(tmp, output_dir / "result.json")
    return info
