# -*- coding: utf-8 -*-
"""Local Qwen3-VL-Embedding / Reranker inference (offline finals).

Loads bundled scripts from the ModelScope/HF model directory. Default device is
CPU so SGLang can keep the PPU for the 27B VLM. Evidence only — never writes
formal defect fields.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from contextlib import contextmanager

from PIL import Image

_DEFAULT_EMB_PATH = Path("/data/model/Qwen3-VL-Embedding-2B")
_DEFAULT_RR_PATH = Path("/data/model/Qwen3-VL-Reranker-2B")
_FALLBACK_ROOT = Path("~/models")

# TRACK4_PROFILE=1 时才累加；纯计时，不影响任何向量或输出。
PROFILE_SECONDS: Dict[str, float] = {"image_decode": 0.0, "model_forward": 0.0}

_lock = threading.Lock()
_embedder: Any = None
_reranker: Any = None


def _env_device() -> str:
    raw = os.environ.get("TRACK4_LOCAL_QWEN_VL_DEVICE", "cpu").strip().lower()
    if raw in {"gpu", "cuda"}:
        return "cuda"
    return "cpu"


def _resolve_model_dir(env_key: str, default_name: str) -> Optional[Path]:
    raw = os.environ.get(env_key, "").strip()
    if raw:
        explicit=Path(raw)
        if not (explicit/'config.json').is_file():
            raise FileNotFoundError(f'Explicit model asset is missing: {explicit}')
        return explicit
    candidates: List[Path] = []
    if raw:
        candidates.append(Path(raw))
    candidates.append(Path("/data/model") / default_name)
    candidates.append(_FALLBACK_ROOT / default_name)
    for p in candidates:
        if (p / "config.json").is_file():
            return p
    return None


def embedding_model_dir() -> Optional[Path]:
    return _resolve_model_dir(
        "TRACK4_LOCAL_QWEN_VL_EMBEDDING_PATH", "Qwen3-VL-Embedding-2B"
    )


def reranker_model_dir() -> Optional[Path]:
    return _resolve_model_dir(
        "TRACK4_LOCAL_QWEN_VL_RERANKER_PATH", "Qwen3-VL-Reranker-2B"
    )


def local_retrieval_ready() -> bool:
    return embedding_model_dir() is not None


def local_reranker_ready() -> bool:
    return reranker_model_dir() is not None


def _ensure_transformers_compat() -> None:
    """Shim missing helpers referenced by bundled Qwen3-VL scripts."""
    try:
        import transformers.utils.generic as generic

        if not hasattr(generic, "check_model_inputs"):

            def check_model_inputs(fn):  # type: ignore[no-untyped-def]
                return fn

            generic.check_model_inputs = check_model_inputs  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


def _load_module(model_dir: Path, script_name: str, mod_name: str) -> Any:
    _ensure_transformers_compat()
    script = model_dir / "scripts" / script_name
    if not script.is_file():
        raise FileNotFoundError(f"missing script: {script}")
    spec = importlib.util.spec_from_file_location(mod_name, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _dtype_kwargs() -> Dict[str, Any]:
    import torch

    device = _env_device()
    if device == "cpu":
        return {"torch_dtype": torch.float32}
    return {"torch_dtype": torch.bfloat16}


@contextmanager
def _hide_cuda_if_cpu(device: str):
    """Bundled Qwen3VLEmbedder/Reranker always `.to(cuda if available)`.

    Hide CUDA during construction so CPU mode does not allocate ~8GB PPU
    cache beside SGLang, then leak it after `.to('cpu')`.
    """
    import torch

    if device != "cpu":
        yield
        return
    original = torch.cuda.is_available
    torch.cuda.is_available = lambda: False  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.cuda.is_available = original  # type: ignore[method-assign]
        try:
            if original():
                torch.cuda.empty_cache()
        except Exception:
            pass


def get_local_embedder() -> Any:
    global _embedder
    with _lock:
        if _embedder is not None:
            return _embedder
        model_dir = embedding_model_dir()
        if model_dir is None:
            raise RuntimeError("local Qwen3-VL-Embedding path not found")
        _t_import = time.perf_counter()
        mod = _load_module(model_dir, "qwen3_vl_embedding.py", "qwen3_vl_embedding_local")
        PROFILE_SECONDS["module_import_s"] = time.perf_counter() - _t_import
        _t_construct = time.perf_counter()
        device = _env_device()
        kwargs = _dtype_kwargs()
        with _hide_cuda_if_cpu(device):
            embedder = mod.Qwen3VLEmbedder(str(model_dir), **kwargs)
        embedder.model = embedder.model.to(device)
        embedder.model.eval()
        PROFILE_SECONDS["construct_and_move_s"] = time.perf_counter() - _t_construct
        _original_preprocess = embedder._preprocess_inputs
        _original_forward = embedder.forward
        def _timed_preprocess(conversations):
            t = time.perf_counter()
            out = _original_preprocess(conversations)
            PROFILE_SECONDS["official_preprocess_s"] = PROFILE_SECONDS.get("official_preprocess_s", 0.0) + time.perf_counter() - t
            return out
        def _timed_forward(inputs):
            import torch
            if device == "cuda": torch.cuda.synchronize()
            t = time.perf_counter()
            out = _original_forward(inputs)
            if device == "cuda": torch.cuda.synchronize()
            PROFILE_SECONDS["forward_sync_s"] = PROFILE_SECONDS.get("forward_sync_s", 0.0) + time.perf_counter() - t
            return out
        embedder._preprocess_inputs = _timed_preprocess
        embedder.forward = _timed_forward
        _embedder = embedder
        print(
            f"[RETRIEVAL] Qwen3-VL-Embedding loaded device={device} dir={model_dir}",
            flush=True,
        )
        return _embedder


def get_local_reranker() -> Any:
    global _reranker
    with _lock:
        if _reranker is not None:
            return _reranker
        model_dir = reranker_model_dir()
        if model_dir is None:
            raise RuntimeError("local Qwen3-VL-Reranker path not found")
        mod = _load_module(model_dir, "qwen3_vl_reranker.py", "qwen3_vl_reranker_local")
        device = _env_device()
        kwargs = _dtype_kwargs()
        with _hide_cuda_if_cpu(device):
            reranker = mod.Qwen3VLReranker(str(model_dir), **kwargs)
        reranker.model = reranker.model.to(device)
        reranker.score_linear = reranker.score_linear.to(device)
        reranker.model.eval()
        reranker.score_linear.eval()
        _reranker = reranker
        return _reranker


def _tensor_to_vectors(tensor: Any) -> List[List[float]]:
    import torch

    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().float().numpy()
    else:
        arr = tensor
    if getattr(arr, "ndim", 0) == 1:
        arr = arr.reshape(1, -1)
    return [row.tolist() for row in arr]


def embed_texts_local(texts: Sequence[str]) -> List[List[float]]:
    if not texts:
        return []
    embedder = get_local_embedder()
    inputs = [{"text": str(t or "")} for t in texts]
    vecs = embedder.process(inputs)
    return _tensor_to_vectors(vecs)


def _data_url_to_image(data_url: str) -> Image.Image:
    if "," not in data_url:
        raise ValueError("invalid data url")
    payload = data_url.split(",", 1)[1]
    raw = base64.b64decode(payload)
    with Image.open(io.BytesIO(raw)) as src:
        return src.convert("RGB")


def embed_image_data_urls_local(data_urls: Sequence[str]) -> List[Optional[List[float]]]:
    if not data_urls:
        return []
    embedder = get_local_embedder()
    profile = str(os.environ.get("TRACK4_PROFILE", "")).strip().lower() in {"1", "true", "yes", "on"}
    out: List[Optional[List[float]]] = []
    for url in data_urls:
        try:
            t0 = time.perf_counter() if profile else 0.0
            image = _data_url_to_image(url)
            if profile:
                t1 = time.perf_counter()
                PROFILE_SECONDS["image_decode"] += t1 - t0
            vecs = embedder.process([{"image": image}])
            if profile:
                PROFILE_SECONDS["model_forward"] += time.perf_counter() - t1
            rows = _tensor_to_vectors(vecs)
            out.append(rows[0] if rows else None)
        except (OSError, ValueError, RuntimeError):
            out.append(None)
    return out


def rerank_local(
    query: str,
    documents: Sequence[str],
    *,
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    docs = [str(d or "") for d in documents]
    if not docs:
        return []
    reranker = get_local_reranker()
    payload = {
        "query": {"text": query},
        "documents": [{"text": d} for d in docs],
    }
    scores = reranker.process(payload)
    if hasattr(scores, "tolist"):
        score_list = scores.tolist()
    else:
        score_list = list(scores)
    ranked = [
        {"index": i, "document": docs[i], "score": score_list[i]}
        for i in range(min(len(docs), len(score_list)))
    ]
    ranked.sort(
        key=lambda x: (
            -(float(x["score"]) if isinstance(x.get("score"), (int, float)) else -1e9),
            str(x.get("document")),
        )
    )
    if top_n is not None:
        ranked = ranked[: max(1, int(top_n))]
    return ranked
