"""Image-vector classification and training-case voting."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from typing import Any,Dict,List,Mapping,Sequence,Tuple
import numpy as np

class BridgeCaseHead:
    def __init__(self, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> None:
        self.hd_coef = np.asarray(arrays["hd_coef"], dtype=np.float32)
        self.hd_intercept = np.asarray(arrays["hd_intercept"], dtype=np.float32)
        self.dis_vectors = np.asarray(arrays["dis_vectors"], dtype=np.float32)
        self.dis_types = [str(t) for t in arrays["dis_types"]]
        self.dis_ratings = [str(r) for r in arrays["dis_ratings"]]
        self.threshold = float(meta["threshold"])
        self.k = int(meta["k"])
        self.type_medoid = dict(meta["type_medoid"])
        self.healthy_medoid = str(meta["healthy_medoid"])
        self.aerial_healthy_medoid = str(meta.get("aerial_healthy_medoid") or meta["healthy_medoid"])

    @classmethod
    def load(cls, path: str | Path) -> "BridgeCaseHead":
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(arrays, meta)

    def decide(self, vectors: np.ndarray) -> List[Dict[str, Any]]:
        x = np.asarray(vectors, dtype=np.float32)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        logit = x @ self.hd_coef.T + self.hd_intercept
        p_dis = 1.0 / (1.0 + np.exp(-logit[:, 0]))
        sims = x @ self.dis_vectors.T
        top = np.argsort(-sims, axis=1)[:, : self.k]
        out = []
        for p, row, idx in zip(p_dis, sims, top):
            tv: Dict[str, float] = defaultdict(float)
            rv: Dict[str, float] = defaultdict(float)
            for j in idx:
                tv[self.dis_types[j]] += float(row[j])
                rv[self.dis_ratings[j]] += float(row[j])
            typ = max(tv, key=tv.get)
            rating = max(rv, key=rv.get) or "2"
            ranked = sorted(tv.items(), key=lambda item: (-item[1], item[0]))
            total_vote = sum(max(0.0, score) for _, score in ranked) or 1.0
            top_candidates = [
                {"label": label, "score": float(score), "vote_share": float(max(0.0, score) / total_vote)}
                for label, score in ranked[:5]
            ]
            type_margin = (
                top_candidates[0]["vote_share"] - top_candidates[1]["vote_share"]
                if len(top_candidates) > 1 else 1.0
            )
            out.append({
                "p_disease": float(p), "type": typ, "rating": rating,
                "nearest_similarity": float(row[idx[0]]) if len(idx) else None,
                "type_margin": float(type_margin), "top_candidates": top_candidates,
            })
        return out
