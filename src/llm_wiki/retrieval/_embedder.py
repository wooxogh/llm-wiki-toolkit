"""Shared local embedder for the vault tooling.

Local, offline, NO API key. Backend: sentence-transformers on whichever torch
device is available — Apple-Silicon GPU (MPS) by default, CPU or CUDA via
`WIKI_EMBED_DEVICE`. An unavailable device is downgraded rather than raised, but
never silently: see `_resolve_device`.

Model: Qwen3-Embedding-0.6B (2025 multilingual, strong KO/EN, 1024-dim).
Qwen3-Embedding uses a task *instruction* on the query side; passages are raw.
Ask `dimension()` for the vector width — never hardcode it, or a store gets
written at one width and queried at another.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import warnings

warnings.filterwarnings("ignore")
import numpy as np
# NOTE: sentence_transformers/torch are imported lazily inside _get() so that
# the server-backed path (recall.py -> embed_query -> HTTP) stays light
# (numpy+urllib only, ~0.2s) and never pays the ~2s torch import.

MODEL = "Qwen/Qwen3-Embedding-0.6B"  # 1024-dim, multilingual
# Requested device. Not necessarily the one used — see _resolve_device.
DEVICE = os.environ.get("WIKI_EMBED_DEVICE", "auto")  # auto | mps | cpu | cuda
PORT = int(os.environ.get("WIKI_EMBED_PORT", "8477"))  # resident embed_server.py
# query-side task instruction (Qwen3-Embedding convention); passages get none.
_QUERY_INSTRUCTION = "Instruct: Given a search query, retrieve relevant knowledge-base pages\nQuery: "

_model = None
_model_device = None


# Attention is O(seq_len^2): an un-capped ~3,000-token chunk inside a batch of 16 asks
# MPS for a ~10 GiB score tensor in one shot, and the MPS caching allocator does not
# hand memory back promptly -> "MPS backend out of memory (MPS allocated: 38.12 GiB,
# tried to allocate 12.84 GiB)" (measured 2026-07-27; it had already forced a CPU
# fallback once before). Chunks are ~700 chars (~280 tokens), so 512 truncates only
# the long tail, and a smaller batch bounds the peak linearly.
MAX_SEQ_TOKENS = int(os.environ.get("WIKI_EMBED_MAX_SEQ", "512"))
BATCH_SIZE = int(os.environ.get("WIKI_EMBED_BATCH", "4"))


def _resolve_device(requested: str = None) -> str:
    """The torch device to actually use, downgrading an unavailable one to CPU.

    `auto` prefers CUDA, then MPS, then CPU. An explicitly requested unavailable
    accelerator falls back to CPU and says so on stderr.
    """
    requested = DEVICE if requested is None else requested
    if requested == "cpu":
        return "cpu"
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    available = {
        "mps": lambda: torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available,
    }.get(requested)
    if available is None:
        return requested  # an exotic device the user named on purpose; let torch judge
    if available():
        return requested
    print(f"[embedder] {requested!r} is not available on this machine — falling back to "
          f"CPU (much slower). Set WIKI_EMBED_DEVICE=cuda (or cpu) to choose explicitly.",
          file=sys.stderr)
    return "cpu"


def _get(device: str | None = None):
    global _model, _model_device
    resolved = _resolve_device(device)
    if _model is None or _model_device != resolved:
        from sentence_transformers import SentenceTransformer  # lazy: heavy import
        _model = SentenceTransformer(MODEL, device=resolved)
        _model.max_seq_length = MAX_SEQ_TOKENS
        _model_device = resolved
    return _model


def dimension() -> int:
    """Vector width of `MODEL`, asked of the loaded model rather than hardcoded.

    A hardcoded width is how a store ends up written at one dimension and
    queried at another: the arrays never meet until a cosine multiply fails, or
    worse, an empty store is written at a stale width and the mismatch only
    surfaces on the first real query. Loads the model if it is not loaded yet,
    so the only caller that pays for this is the one with no vectors to measure.
    """
    return int(_get().get_sentence_embedding_dimension())


def _encode(texts: list[str], device: str | None = None,
            show_progress: bool = False) -> np.ndarray:
    # RSS still climbs ~1.6 -> ~3.9 GB over a 1,557-chunk rebuild. Periodic
    # `torch.mps.empty_cache()` between slices was tried and **measured as no help**
    # (3.81 GB without vs 3.98 GB with) — don't re-add it. The peak is bounded and
    # the run completes; the OOM came from seq_len/batch, not from cache retention.
    v = _get(device).encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=show_progress and sys.stderr.isatty(),
    )
    return np.asarray(v, dtype=np.float32)  # already L2-normalized -> dot == cosine


def embed_passages(texts: list[str], device: str | None = None,
                   show_progress: bool = False) -> np.ndarray:
    return _encode(texts, device, show_progress)


def embed_query_local(text: str, device: str | None = None) -> np.ndarray:
    """Embed a query in-process (loads the model). Used by embed_server.py and
    as the fallback when no resident server is running."""
    return _encode([_QUERY_INSTRUCTION + text], device)[0]


def embed_query(text: str, device: str | None = None) -> np.ndarray:
    """Server-first: hit the resident embed_server (warm, ~0.1s) if up; else
    fall back to an in-process load (~7-9s cold)."""
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/embed",
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return np.asarray(json.load(r)["vector"], dtype=np.float32)
    except Exception:
        return embed_query_local(text, device)
