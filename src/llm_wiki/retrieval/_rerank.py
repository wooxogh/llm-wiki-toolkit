"""Optional local cross-encoder reranker (key-free, offline).

Off by default. When enabled, hybrid retrieval fetches a larger candidate pool
and this reorders it by a query-document cross-encoder score — targets the
measured gap where the right page is in the top-k but not ranked #1
(HARD gold: Hit@1 64% vs Hit@8 100%, 2026-07-24).

Model: BAAI/bge-reranker-v2-m3 (multilingual, strong Korean). Lazy-loaded so the
default retrieval path never pays for it. Overridable via WIKI_RERANK_MODEL.
"""
from __future__ import annotations

import os

_MODEL_NAME = os.environ.get("WIKI_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
_model = None


def _get():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(_MODEL_NAME, max_length=512)
    return _model


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """Cross-encoder relevance score per text (higher = more relevant)."""
    if not texts:
        return []
    return [float(s) for s in _get().predict([(query, t) for t in texts])]
