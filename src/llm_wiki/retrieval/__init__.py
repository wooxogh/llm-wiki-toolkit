"""Hybrid retrieval core: local embedder, optional reranker, chunk indexer.

Kept as a separate subpackage because it is the ONE part of the tool that may
import torch/sentence-transformers, and only lazily inside functions (see
_embedder.py, _rerank.py). tests/ must never import this package's heavy
optional dependencies; dense scores are always injected by the caller.
"""
from __future__ import annotations
