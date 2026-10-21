"""Image-vector classification and training-case voting."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from typing import Any,Dict,List,Mapping,Sequence,Tuple
import numpy as np

class RailEmbedHead:
    def __init__(self, coef: np.ndarray, intercept: np.ndarray, classes: Sequence[str]) -> None:
        self.coef = np.asarray(coef, dtype=np.float32)
        self.intercept = np.asarray(intercept, dtype=np.float32)
        self.classes = [str(c) for c in classes]
        if self.coef.shape[0] != len(self.classes):
            raise ValueError("rail head coef/classes mismatch")

    @classmethod
    def load(cls, path: str | Path) -> "RailEmbedHead":
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(z["coef"], z["intercept"], list(z["classes"]))

    def probabilities(self, vectors: np.ndarray) -> np.ndarray:
        x = np.asarray(vectors, dtype=np.float32)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        logits = x @ self.coef.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob

    def predict(self, vectors: np.ndarray) -> Tuple[List[str], np.ndarray]:
        prob = self.probabilities(vectors)
        idx = prob.argmax(axis=1)
        return [self.classes[i] for i in idx], prob[np.arange(len(idx)), idx]

    def predict_detailed(self, vectors: np.ndarray) -> List[Dict[str, Any]]:
        prob = self.probabilities(vectors)
        order = np.argsort(-prob, axis=1)
        out = []
        for row, idx in zip(prob, order):
            top1 = int(idx[0])
            top2 = int(idx[1]) if len(idx) > 1 else top1
            entropy = float(-np.sum(row * np.log(np.clip(row, 1e-12, 1.0))))
            out.append({
                "label": self.classes[top1], "top1_probability": float(row[top1]),
                "top2_probability": float(row[top2]) if len(idx) > 1 else 0.0,
                "probability_margin": float(row[top1] - row[top2]) if len(idx) > 1 else 1.0,
                "entropy": entropy,
                "top_candidates": [
                    {"label": self.classes[int(i)], "probability": float(row[int(i)])}
                    for i in idx[:5]
                ],
            })
        return out
