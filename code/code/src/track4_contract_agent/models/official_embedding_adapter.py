# -*- coding: utf-8 -*-
"""Formal Embedding / Reranker adapters (local Qwen3-VL or China open-source API).

Allowed models (competition policy):
  - Qwen3-VL-Embedding-2B / 8B
  - Qwen3-VL-Reranker-2B / 8B

Finals offline: set local model dirs (or rely on /data/model defaults) and
TRACK4_RETRIEVAL_BACKEND=local — no official HTTP gateway required.

Forbidden backends for *large / foundation* retrieval (do not wire here):
  OpenCLIP, DINOv2, non-China closed large LLMs/VLMs.

The pipeline is pure-VLM (no YOLO detector in the architecture);
they are not embedding/rerank backends.

These adapters retrieve / rank evidence only. They must not be the sole
decision path for formal defectType (DENSE_ONLY_DECISION = FORBIDDEN).
"""

from __future__ import annotations

import base64
import io
import os
import threading
from pathlib import Path
from typing import Any, Optional, Sequence

import requests
from PIL import Image

from track4_contract_agent.models.local_qwen_vl_retrieval import (
    embed_image_data_urls_local,
    embed_texts_local,
    embedding_model_dir,
    local_retrieval_ready,
    local_reranker_ready,
    rerank_local,
    reranker_model_dir,
)
from track4_contract_agent.models.official_credentials import load_official_credentials

DEFAULT_EMBEDDING_MODEL = "Qwen3-VL-Embedding-8B"
DEFAULT_RERANKER_MODEL = "Qwen3-VL-Reranker-8B"
DEFAULT_LOCAL_EMBEDDING_MODEL = "Qwen3-VL-Embedding-2B"
DEFAULT_LOCAL_RERANKER_MODEL = "Qwen3-VL-Reranker-2B"


def _retrieval_backend_mode() -> str:
    raw = os.environ.get("TRACK4_RETRIEVAL_BACKEND", "auto").strip().lower()
    if raw in {"local", "offline"}:
        return "local"
    if raw in {"official", "remote", "api"}:
        return "official"
    # 决赛离线环境：禁止任何形式的官方 HTTP 回落，宁可报错也不越境调用。
    if os.environ.get("TRACK4_FINAL_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return "local"
    return "auto"


def _use_local_embedding() -> bool:
    mode = _retrieval_backend_mode()
    if mode == "local":
        return local_retrieval_ready()
    if mode == "official":
        return False
    return local_retrieval_ready()


def _use_local_reranker() -> bool:
    mode = _retrieval_backend_mode()
    if mode == "local":
        return local_reranker_ready()
    if mode == "official":
        return False
    return local_reranker_ready()

try:
    _RERANK_MAX_INFLIGHT = max(1, int(os.environ.get("TRACK4_RERANK_MAX_INFLIGHT", "4") or 4))
except ValueError:
    _RERANK_MAX_INFLIGHT = 4
_rerank_gate = threading.BoundedSemaphore(_RERANK_MAX_INFLIGHT)


def jpeg_data_url(
    image_path: str | Path,
    *,
    longest_edge: int = 1024,
    quality: int = 85,
) -> str:
    """Resize to longest_edge and return a JPEG data URL for the official gateway."""
    path = Path(image_path)
    with Image.open(path) as src:
        im = src.convert("RGB")
        w, h = im.size
        scale = longest_edge / max(w, h)
        if scale < 1:
            im = im.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.BILINEAR,
            )
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class OfficialEmbeddingAdapter:
    """Qwen3-VL-Embedding via local weights or OpenAI-compatible /embeddings API."""

    backend_name = "official_embedding"
    family = "qwen_china_opensource"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout_s: float = 60.0,
    ) -> None:
        key, base = load_official_credentials()
        self._api_key = api_key if api_key is not None else key
        self._api_base = (api_base if api_base is not None else base).rstrip("/")
        self._model_id = (
            model_id
            or os.environ.get("TRACK4_OFFICIAL_EMBEDDING_MODEL_ID", "")
            or DEFAULT_EMBEDDING_MODEL
        )
        self._timeout_s = float(timeout_s)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def configuration_ready(self) -> bool:
        if _use_local_embedding():
            return True
        return bool(self._api_key and self._api_base and self._model_id)

    @property
    def uses_local_backend(self) -> bool:
        return _use_local_embedding()

    # alias used by older Shadow callers
    @property
    def configured(self) -> bool:
        return self.configuration_ready

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def api_base(self) -> str:
        return self._api_base

    @property
    def model(self) -> str:
        return self._model_id

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    def health_check(self) -> bool:
        if not self.configuration_ready:
            return False
        probe = self.embed_texts(["健康检查"])
        return bool(probe.get("ok"))

    def embed_texts(self, texts: list[str]) -> dict[str, Any]:
        if not self.configuration_ready:
            return {"ok": False, "error": "NOT_CONFIGURED"}
        if not texts:
            return {"ok": True, "status_code": 200, "data": {"data": []}, "n_input": 0}
        if self.uses_local_backend:
            try:
                vectors = embed_texts_local(texts)
                if len(vectors) != len(texts):
                    return {"ok": False, "error": "LOCAL_EMBED_COUNT_MISMATCH"}
                model_id = self._model_id or DEFAULT_LOCAL_EMBEDDING_MODEL
                return {
                    "ok": True,
                    "status_code": 200,
                    "data": {
                        "data": [
                            {"index": i, "embedding": v} for i, v in enumerate(vectors)
                        ]
                    },
                    "n_input": len(texts),
                    "model_id": model_id,
                    "backend": "local_qwen_vl_embedding",
                }
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "error": type(e).__name__ + ": " + str(e)[:200]}
        url = f"{self._api_base}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self._model_id, "input": texts}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self._timeout_s)
            data = resp.json() if resp.content else {}
            return {
                "ok": resp.status_code == 200,
                "status_code": resp.status_code,
                "data": data,
                "n_input": len(texts),
                "model_id": self._model_id,
                "backend": self.backend_name,
            }
        except Exception as e:  # noqa: BLE001 — callers decide fail-closed
            return {"ok": False, "error": type(e).__name__ + ": " + str(e)[:200]}

    def embed_image_data_urls(self, data_urls: list[str]) -> dict[str, Any]:
        """Embed images. Official gateway needs ``{"image": data_url}`` per item.

        OpenAI-style ``image_url`` collapses to a constant vector on this gateway
        (verified 2026-08-12). One image per request.
        """
        if not self.configuration_ready:
            return {"ok": False, "error": "NOT_CONFIGURED"}
        if not data_urls:
            return {"ok": True, "status_code": 200, "data": {"data": []}, "n_input": 0}
        if self.uses_local_backend:
            try:
                vectors = embed_image_data_urls_local(data_urls)
                out: list[dict[str, Any]] = []
                for i, vec in enumerate(vectors):
                    if vec is None:
                        return {
                            "ok": False,
                            "error": "LOCAL_IMAGE_EMBED_FAILED",
                            "failed_index": i,
                            "n_input": len(data_urls),
                        }
                    out.append({"index": i, "embedding": vec})
                return {
                    "ok": True,
                    "status_code": 200,
                    "data": {"data": out},
                    "n_input": len(data_urls),
                    "model_id": self._model_id or DEFAULT_LOCAL_EMBEDDING_MODEL,
                    "backend": "local_qwen_vl_embedding",
                }
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "error": type(e).__name__ + ": " + str(e)[:200]}
        vectors: list[list[float]] = []
        last_status = 200
        url = f"{self._api_base}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        for i, data_url in enumerate(data_urls):
            payload = {"model": self._model_id, "input": [{"image": data_url}]}
            try:
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=self._timeout_s
                )
                last_status = resp.status_code
                data = resp.json() if resp.content else {}
                if resp.status_code != 200:
                    return {
                        "ok": False,
                        "status_code": resp.status_code,
                        "data": data,
                        "n_input": len(data_urls),
                        "failed_index": i,
                        "error": f"HTTP_{resp.status_code}",
                    }
                item = ((data.get("data") or [{}])[0].get("embedding"))
                if not isinstance(item, list) or not item:
                    return {
                        "ok": False,
                        "error": "EMPTY_EMBEDDING",
                        "failed_index": i,
                        "n_input": len(data_urls),
                    }
                vectors.append(item)
            except Exception as e:  # noqa: BLE001
                return {
                    "ok": False,
                    "error": type(e).__name__ + ": " + str(e)[:200],
                    "failed_index": i,
                    "n_input": len(data_urls),
                }
        return {
            "ok": True,
            "status_code": last_status,
            "data": {"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]},
            "n_input": len(data_urls),
            "model_id": self._model_id,
            "backend": self.backend_name,
        }

    def embed_image_paths(
        self,
        paths: Sequence[str],
        *,
        longest_edge: int = 1024,
        quality: int = 85,
    ) -> dict[str, Any]:
        urls: list[str] = []
        for path in paths:
            try:
                urls.append(jpeg_data_url(path, longest_edge=longest_edge, quality=quality))
            except (OSError, ValueError) as exc:
                return {
                    "ok": False,
                    "error": f"IMAGE_ENCODE_FAILED:{type(exc).__name__}",
                    "path": str(path),
                }
        return self.embed_image_data_urls(urls)

    def extract_vectors(self, embed_result: dict[str, Any]) -> list[list[float]]:
        """Return embedding vectors in input order; empty list on failure."""
        if not embed_result.get("ok"):
            return []
        data = (embed_result.get("data") or {}).get("data") or []
        data = sorted(data, key=lambda x: x.get("index", 0))
        out: list[list[float]] = []
        for item in data:
            emb = item.get("embedding")
            if isinstance(emb, list):
                out.append(emb)
        return out

    def get_runtime_metadata(self) -> dict[str, Any]:
        local_dir = embedding_model_dir()
        return {
            "backend": "local_qwen_vl_embedding" if self.uses_local_backend else self.backend_name,
            "family": self.family,
            "model_id": self._model_id,
            "api_base_set": bool(self._api_base),
            "local_model_dir": str(local_dir) if local_dir else "",
            "configured": self.configuration_ready,
            "sole_decision_forbidden": True,
        }


class OfficialRerankerAdapter:
    """Official /rerank for Qwen3-VL-Reranker-8B (gateway may return bare list)."""

    backend_name = "official_reranker"
    family = "qwen_china_opensource"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout_s: float = 60.0,
    ) -> None:
        key, base = load_official_credentials()
        self._api_key = api_key if api_key is not None else key
        self._api_base = (api_base if api_base is not None else base).rstrip("/")
        self._model_id = (
            model_id
            or os.environ.get("TRACK4_OFFICIAL_RERANKER_MODEL_ID", "")
            or DEFAULT_RERANKER_MODEL
        )
        self._timeout_s = float(timeout_s)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def configuration_ready(self) -> bool:
        if _use_local_reranker():
            return True
        return bool(self._api_key and self._api_base and self._model_id)

    @property
    def configured(self) -> bool:
        return self.configuration_ready

    @property
    def uses_local_backend(self) -> bool:
        return _use_local_reranker()

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def api_base(self) -> str:
        return self._api_base

    @property
    def model(self) -> str:
        return self._model_id

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    def health_check(self) -> bool:
        if not self.configuration_ready:
            return False
        r = self.rerank("probe", ["文档甲", "文档乙"], top_n=2)
        return bool(r.get("ok"))

    def probe_endpoints(self, query: str, documents: list[str]) -> dict[str, Any]:
        """Try common OpenAI-compatible rerank paths; return first success."""
        if not self.configuration_ready:
            return {"ok": False, "error": "NOT_CONFIGURED"}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        candidates = [
            ("/rerank", {"model": self._model_id, "query": query, "documents": documents}),
            ("/v1/rerank", {"model": self._model_id, "query": query, "documents": documents}),
        ]
        results = []
        for path, payload in candidates:
            url = f"{self._api_base}{path}"
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self._timeout_s)
                results.append(
                    {
                        "path": path,
                        "status_code": resp.status_code,
                        "ok": resp.status_code == 200,
                        "body_preview": (resp.text or "")[:300],
                    }
                )
                if resp.status_code == 200:
                    return {"ok": True, "path": path, "status_code": 200, "tried": results}
            except Exception as e:  # noqa: BLE001
                results.append({"path": path, "ok": False, "error": str(e)[:160]})
        return {"ok": False, "tried": results}

    def rerank(
        self, query: str, documents: list[str], *, top_n: int | None = None
    ) -> dict[str, Any]:
        if not self.configuration_ready:
            return {"ok": False, "error": "NOT_CONFIGURED"}
        if not documents:
            return {"ok": True, "ranked": [], "model_id": self._model_id}
        if self.uses_local_backend:
            try:
                ranked = rerank_local(query, documents, top_n=top_n)
                return {
                    "ok": True,
                    "status_code": 200,
                    "ranked": ranked,
                    "raw_keys": ["local"],
                    "model_id": self._model_id or DEFAULT_LOCAL_RERANKER_MODEL,
                    "backend": "local_qwen_vl_reranker",
                }
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "error": type(e).__name__ + ": " + str(e)[:200]}
        url = f"{self._api_base}/rerank"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._model_id,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n
        _rerank_gate.acquire()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self._timeout_s)
            raw = resp.json() if resp.content else {}
            if resp.status_code != 200:
                return {"ok": False, "status_code": resp.status_code, "data": raw}
            ranked = parse_rerank_response(raw, documents)
            raw_keys = list(raw.keys()) if isinstance(raw, dict) else ["__list__"]
            return {
                "ok": True,
                "status_code": 200,
                "ranked": ranked,
                "raw_keys": raw_keys,
                "model_id": self._model_id,
                "backend": self.backend_name,
            }
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": type(e).__name__ + ": " + str(e)[:200]}
        finally:
            _rerank_gate.release()

    def get_runtime_metadata(self) -> dict[str, Any]:
        local_dir = reranker_model_dir()
        return {
            "backend": "local_qwen_vl_reranker" if self.uses_local_backend else self.backend_name,
            "family": self.family,
            "model_id": self._model_id,
            "api_base_set": bool(self._api_base),
            "local_model_dir": str(local_dir) if local_dir else "",
            "configured": self.configuration_ready,
            "sole_decision_forbidden": True,
        }


def parse_rerank_response(raw: Any, documents: list[str]) -> list[dict[str, Any]]:
    """Parse list-or-dict rerank payloads from the official gateway."""
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = raw.get("results") or raw.get("data") or raw.get("output") or []
    else:
        items = []
    if not isinstance(items, list):
        return [{"index": i, "document": d, "score": None} for i, d in enumerate(documents)]
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        idx = it.get("index")
        if idx is None and "document" in it:
            doc = it.get("document")
            if isinstance(doc, dict):
                doc = doc.get("text") or doc.get("document")
            try:
                idx = documents.index(doc) if isinstance(doc, str) else None
            except ValueError:
                idx = None
        score = it.get("relevance_score", it.get("score", it.get("relevance")))
        doc_text = None
        if isinstance(idx, int) and 0 <= idx < len(documents):
            doc_text = documents[idx]
        elif isinstance(it.get("document"), str):
            doc_text = it["document"]
        out.append({"index": idx, "document": doc_text, "score": score})
    out = [x for x in out if x.get("document")]
    out.sort(
        key=lambda x: (
            -(float(x["score"]) if isinstance(x.get("score"), (int, float)) else -1e9),
            str(x.get("document")),
        )
    )
    return out


# Backward-compatible aliases for Shadow scripts / skills
OfficialEmbeddingClient = OfficialEmbeddingAdapter
OfficialRerankerClient = OfficialRerankerAdapter
