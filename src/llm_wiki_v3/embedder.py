"""Local Qwen embedding for retrieval and indexing."""
from __future__ import annotations

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
