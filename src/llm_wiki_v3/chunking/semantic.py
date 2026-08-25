"""Gate V2 feature calculations shared by production inference and tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


EPSILON = 1e-12


class EmbeddingProvider(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class BoundaryContext:
    vectors: np.ndarray
    valid_mask: np.ndarray


class BoundaryVerifier(Protocol):
    def predict(self, contexts: Sequence[BoundaryContext]) -> list[float]: ...


def contextual_cosine_curve(embeddings: Sequence[Sequence[float]], window_size: int = 2) -> np.ndarray:
    """Cosine between mean-pooled left/right contexts for every sentence gap."""
    if window_size < 1:
        raise ValueError("window_size must be positive.")
    vectors = np.asarray(embeddings, dtype=np.float64)
    if len(vectors) < 2:
        return np.empty(0, dtype=np.float64)
    prefix = np.concatenate(
        (np.zeros((1, vectors.shape[1]), dtype=np.float64), np.cumsum(vectors, axis=0))
    )
    gaps = np.arange(len(vectors) - 1)
    left_start = np.maximum(0, gaps - window_size + 1)
    left_end = gaps + 1
    right_start = gaps + 1
    right_end = np.minimum(len(vectors), gaps + window_size + 1)
    left_mean = (prefix[left_end] - prefix[left_start]) / (left_end - left_start)[:, None]
    right_mean = (prefix[right_end] - prefix[right_start]) / (right_end - right_start)[:, None]
    denominator = np.linalg.norm(left_mean, axis=1) * np.linalg.norm(right_mean, axis=1)
    numerator = np.sum(left_mean * right_mean, axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def local_valleys(similarities: Sequence[float]) -> np.ndarray:
    """Return plateau-safe, deterministic two-sided local valley flags."""
    scores = np.asarray(similarities, dtype=np.float64)
    valleys = np.zeros(len(scores), dtype=bool)
    start = 0
    while start < len(scores):
        end = start
        while end + 1 < len(scores) and abs(scores[end + 1] - scores[start]) <= EPSILON:
            end += 1
        if start > 0 and end + 1 < len(scores):
            value = scores[start]
            if value < scores[start - 1] - EPSILON and value < scores[end + 1] - EPSILON:
                valleys[(start + end) // 2] = True
        start = end + 1
    return valleys


def boundary_context(
    embeddings: Sequence[Sequence[float]],
    gap_index: int,
    window_size: int = 3,
) -> BoundaryContext:
    """Build the checkpoint's fixed [-3,-2,-1,+1,+2,+3] input context."""
    vectors = np.asarray(embeddings, dtype=np.float32)
    if not 0 <= gap_index < len(vectors) - 1:
        raise IndexError("gap_index must point between two existing sentences.")
    indexes = list(range(gap_index - window_size + 1, gap_index + 1))
    indexes.extend(range(gap_index + 1, gap_index + window_size + 1))
    rows: list[np.ndarray] = []
    mask: list[bool] = []
    for index in indexes:
        if 0 <= index < len(vectors):
            rows.append(vectors[index])
            mask.append(True)
        else:
            rows.append(np.zeros(vectors.shape[1], dtype=np.float32))
            mask.append(False)
    return BoundaryContext(np.stack(rows), np.asarray(mask, dtype=bool))
