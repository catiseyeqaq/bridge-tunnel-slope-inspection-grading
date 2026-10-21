# -*- coding: utf-8 -*-
"""Frozen Qwen visual feature extraction with per-run image preparation."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import List, Optional, Protocol, Sequence

from PIL import Image

from track4_contract_agent.models.official_embedding_adapter import (
    DEFAULT_EMBEDDING_MODEL,
    OfficialEmbeddingAdapter,
    jpeg_data_url,
)

TEXT_DIM = 256
IMAGE_DIM = 128
QWEN_BACKEND = "qwen3_vl_embedding_8b"
QWEN_BACKEND_2B = "qwen3_vl_embedding_2b"
HASH_BACKEND = "local_hash"


def l2_normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec)) or 1e-12
    return [float(x) / norm for x in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(sum(float(a[i]) * float(b[i]) for i in range(n)))


def _ngrams(text: str, n: int) -> List[str]:
    s = "".join(str(text or "").split())
    if not s:
        return []
    if len(s) < n:
        return [s]
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def hash_text(text: str, *, dim: int = TEXT_DIM) -> List[float]:
    vec = [0.0] * dim
    grams = _ngrams(text, 1)[:80] + _ngrams(text, 2) + _ngrams(text, 3)[:40]
    for gram in grams:
        digest = hashlib.md5(gram.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "little") % dim
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vec[idx] += sign
    return l2_normalize(vec)


def hash_image(image_path: str, *, dim: int = IMAGE_DIM) -> List[float]:
    """dHash (64) + 4³ RGB histogram (64). Offline, no API."""
    path = Path(image_path)
    with Image.open(path) as src:
        rgb = src.convert("RGB")
        gray = rgb.convert("L").resize((9, 8), Image.LANCZOS)
        pixels = list(gray.getdata())
        dhash: List[float] = []
        for y in range(8):
            row = pixels[y * 9 : (y + 1) * 9]
            for x in range(8):
                dhash.append(1.0 if row[x] > row[x + 1] else -1.0)
        small = rgb.resize((32, 32), Image.LANCZOS)
        hist = [0.0] * 64
        for r, g, b in small.getdata():
            hist[(r // 64) * 16 + (g // 64) * 4 + (b // 64)] += 1.0
    total = sum(hist) or 1.0
    hist = [v / total for v in hist]
    raw = dhash + hist
    if dim < len(raw):
        raw = raw[:dim]
    elif dim > len(raw):
        raw = raw + [0.0] * (dim - len(raw))
    return l2_normalize(raw)


class Embedder(Protocol):
    backend_name: str

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        ...

    def embed_image_paths(self, paths: Sequence[str]) -> List[Optional[List[float]]]:
        ...


class LocalHasher:
    """Deterministic local embedder used when Qwen embedding is off."""

    backend_name = HASH_BACKEND
    text_dim = TEXT_DIM
    image_dim = IMAGE_DIM

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [hash_text(t) for t in texts]

    def embed_image_paths(self, paths: Sequence[str]) -> List[Optional[List[float]]]:
        out: List[Optional[List[float]]] = []
        for path in paths:
            if not path or not Path(path).is_file():
                out.append(None)
                continue
            try:
                out.append(hash_image(path))
            except (OSError, ValueError):
                out.append(None)
        return out


class QwenVLEmbedder:
    """Qwen3-VL-Embedding-8B via the official OpenAI-compatible /embeddings API."""

    backend_name = QWEN_BACKEND
    model_id = DEFAULT_EMBEDDING_MODEL

    def __init__(
        self,
        adapter: Optional[OfficialEmbeddingAdapter] = None,
        *,
        batch_size: int = 8,
        longest_edge: int = 1024,
        require_ready: bool = True,
    ) -> None:
        self.adapter = adapter or OfficialEmbeddingAdapter()
        self.batch_size = max(1, int(batch_size))
        self.longest_edge = int(longest_edge)
        self.require_ready = bool(require_ready)

    def _ensure(self) -> None:
        if self.require_ready and not self.adapter.configuration_ready:
            raise RuntimeError(
                "Qwen3-VL-Embedding-8B is not configured "
                "(TRACK4_OFFICIAL_API_KEY / TRACK4_OFFICIAL_API_BASE)"
            )

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        self._ensure()
        if not texts:
            return []
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = [str(t or "") for t in texts[i : i + self.batch_size]]
            result = self.adapter.embed_texts(chunk)
            vectors = self.adapter.extract_vectors(result)
            if len(vectors) != len(chunk):
                raise RuntimeError(
                    "Qwen3-VL-Embedding-8B text batch failed: "
                    f"{result.get('error') or result.get('status_code')}"
                )
            out.extend(l2_normalize(v) for v in vectors)
        return out

    def embed_image_paths(self, paths: Sequence[str]) -> List[Optional[List[float]]]:
        """JPEG preprocessing is unchanged; local offline images may be batched."""
        self._ensure()
        import time
        from track4_contract_agent.models.local_qwen_vl_retrieval import PROFILE_SECONDS
        t_prepare = time.perf_counter()
        cache = getattr(self, "_prepared_image_urls", None)
        prepared = [cache[str(p)] for p in paths] if cache is not None else self._prepare_data_urls(paths)
        PROFILE_SECONDS["jpeg_prepare_s"] = PROFILE_SECONDS.get("jpeg_prepare_s", 0.0) + time.perf_counter() - t_prepare
        out: List[Optional[List[float]]] = [None] * len(prepared)
        valid = [(i, u) for i, u in enumerate(prepared) if u is not None]
        if not valid:
            return out
        batch_size = max(1, int(os.environ.get("TRACK4_LOCAL_IMAGE_BATCH", "4") or 4))
        use_local_batch = bool(getattr(self.adapter, "uses_local_backend", False)) and batch_size > 1
        if not use_local_batch:
            for i, data_url in valid:
                result = self.adapter.embed_image_data_urls([data_url])
                vectors = self.adapter.extract_vectors(result)
                if vectors:
                    out[i] = l2_normalize(vectors[0])
            return out
        from track4_contract_agent.models.local_qwen_vl_retrieval import (
            _data_url_to_image, get_local_embedder, _tensor_to_vectors,
        )
        embedder = get_local_embedder()
        import torch
        from concurrent.futures import ThreadPoolExecutor
        chunks = [valid[b:b + batch_size] for b in range(0, len(valid), batch_size)]
        def prepare(chunk):
            images = [_data_url_to_image(data_url) for _, data_url in chunk]
            conversations = [embedder.format_model_input(image=image) for image in images]
            return embedder._preprocess_inputs(conversations)
        # Exactly the official process operations, with one CPU batch ahead.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="next-image-batch") as pool:
            pending = pool.submit(prepare, chunks[0])
            for i, chunk in enumerate(chunks):
                processed_inputs = pending.result()
                if i + 1 < len(chunks):
                    pending = pool.submit(prepare, chunks[i + 1])
                processed_inputs = {k: v.to(embedder.model.device) for k, v in processed_inputs.items()}
                outputs = embedder.forward(processed_inputs)
                embeddings = embedder._pooling_last(outputs['last_hidden_state'], outputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
                vectors = _tensor_to_vectors(embeddings)
                if len(vectors) != len(chunk):
                    raise RuntimeError("Pipeline returned incomplete vectors")
                for (pos, _), vector in zip(chunk, vectors):
                    out[pos] = l2_normalize(vector)
        return out

    def _prepare_data_urls(self, paths: Sequence[str]) -> List[Optional[str]]:
        """把“读文件+解码+缩放到 JPEG”从 GPU 循环里拆出来并行做完。

        jpeg_data_url 是（文件字节, 长边, 编码参数）的纯函数，并行不改变任何像素，
        且下面仍按原顺序逐图调 process，向量与串行版本逐字相同。
        单图失败仍沿用同一个 (OSError, ValueError) 返回 None，语义不变。
        """
        workers = max(1, int(os.environ.get("TRACK4_EMBED_PREP_WORKERS", "8")))

        def one(path: str) -> Optional[str]:
            if not path or not Path(path).is_file():
                return None
            try:
                return jpeg_data_url(path, longest_edge=self.longest_edge)
            except (OSError, ValueError):
                return None

        items = list(paths)
        if workers == 1 or len(items) <= 1:
            return [one(p) for p in items]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
            return list(pool.map(one, items))


def resolve_embedder(name: str = QWEN_BACKEND, **kwargs) -> Embedder:
    key = str(name or QWEN_BACKEND).strip().lower().replace("-", "_")
    if key in {"hash", "local_hash", "local", HASH_BACKEND}:
        return LocalHasher()
    if key in {
        "qwen",
        "qwen3",
        "official",
        QWEN_BACKEND,
        QWEN_BACKEND_2B,
        "qwen3_vl_embedding_8b",
        "qwen3_vl_embedding_2b",
        "qwen3_vl_embedding",
        DEFAULT_EMBEDDING_MODEL.lower().replace("-", "_"),
        "Qwen3-VL-Embedding-2B".lower().replace("-", "_"),
    }:
        return QwenVLEmbedder(**kwargs)
    raise ValueError(f"unknown embedder: {name}")


def top_k_cosine(
    query: Sequence[float],
    ids: Sequence[str],
    matrix: Sequence[Sequence[float]],
    *,
    top_k: int = 10,
) -> List[tuple[str, float]]:
    scored: List[tuple[str, float]] = []
    q = list(query)
    for doc_id, vec in zip(ids, matrix):
        if not vec:
            continue
        scored.append((doc_id, cosine(q, vec)))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:top_k]
