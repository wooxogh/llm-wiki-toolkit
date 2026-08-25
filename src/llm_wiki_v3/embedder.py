"""Local Qwen embedding and optional cross-encoder reranking."""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from .chunking.runtime import QwenSentenceEmbedder


QUERY_INSTRUCTION = "Instruct: Given a search query, retrieve relevant knowledge-base chunks\nQuery: "


class QwenEmbedder:
    def __init__(self, model_id: str, *, device: str = "auto", batch_size: int = 32) -> None:
        self.model_id = model_id
        self._delegate = QwenSentenceEmbedder(
            model_id,
            device=device,
            batch_size=batch_size,
            sort_by_length=True,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._delegate.encode(texts)

    def encode_query(self, query: str) -> np.ndarray:
        return self._delegate.encode([QUERY_INSTRUCTION + query])[0]


class CrossEncoderReranker:
    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.environ.get("WIKI_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_id, max_length=512)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores = self._load().predict([(query, text) for text in texts])
        return [float(score) for score in scores]
