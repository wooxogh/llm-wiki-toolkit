"""Lazy Qwen embedding and Small-V3 Attention checkpoint runtime."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .semantic import BoundaryContext


def resolve_device(requested: str) -> str:
    # PyTorch reads this setting while importing/initializing MPS. Set it
    # before the torch import so unsupported MPS ops alone can fall back.
    if requested in {"auto", "mps"} and sys.platform == "darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class QwenSentenceEmbedder:
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        batch_size: int = 32,
        max_seq_length: int = 1024,
        sort_by_length: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.sort_by_length = sort_by_length
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, device=self.device)
            self._model.max_seq_length = self.max_seq_length
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        original_texts = list(texts)
        # SentenceTransformer pads every batch to its longest sequence. Sorting
        # by an inexpensive character-length proxy prevents one long chunk from
        # inflating compute for many short chunks, then restores source order.
        order = sorted(range(len(original_texts)), key=lambda index: (len(original_texts[index]), index)) if self.sort_by_length else list(range(len(original_texts)))
        ordered_texts = [original_texts[index] for index in order]
        ordered_vectors = self._load_model().encode(
            ordered_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(original_texts) > self.batch_size,
        )
        ordered_vectors = np.asarray(ordered_vectors, dtype=np.float32)
        if not self.sort_by_length:
            return ordered_vectors
        vectors = np.empty_like(ordered_vectors)
        vectors[order] = ordered_vectors
        return vectors


@dataclass(frozen=True)
class _AttentionConfig:
    input_dimension: int = 1024
    model_dimension: int = 256
    layers: int = 2
    heads: int = 4
    feedforward_dimension: int = 1024
    dropout: float = 0.1


def _build_attention_model(config: _AttentionConfig):
    import torch
    from torch import nn

    class AttentionBoundaryVerifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(config.input_dimension, config.model_dimension)
            self.boundary_token = nn.Parameter(torch.zeros(1, 1, config.model_dimension))
            self.position = nn.Parameter(torch.zeros(1, 7, config.model_dimension))
            layer = nn.TransformerEncoderLayer(
                d_model=config.model_dimension,
                nhead=config.heads,
                dim_feedforward=config.feedforward_dimension,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
            self.classifier = nn.Linear(config.model_dimension, 1)

        def forward(self, vectors, valid_mask):
            projected = self.projection(vectors)
            boundary = self.boundary_token.expand(projected.size(0), -1, -1)
            sequence = torch.cat((projected[:, :3], boundary, projected[:, 3:]), dim=1)
            boundary_mask = torch.ones(
                (valid_mask.size(0), 1), dtype=torch.bool, device=valid_mask.device
            )
            mask = torch.cat((valid_mask, boundary_mask), dim=1)
            encoded = self.encoder(sequence + self.position, src_key_padding_mask=~mask)
            return self.classifier(encoded[:, 3]).squeeze(-1)

    return AttentionBoundaryVerifier()


class SmallV3BoundaryVerifier:
    """Inference-only wrapper around ``small_v3_50_normal.pt``."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str = "auto",
        batch_size: int = 128,
    ) -> None:
        import torch

        self.device = resolve_device(device)
        self.batch_size = batch_size
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Small-V3 checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        raw_config = payload.get("config", {})
        config = _AttentionConfig(
            input_dimension=int(raw_config.get("input_dimension", 1024)),
            model_dimension=int(raw_config.get("model_dimension", 256)),
            layers=int(raw_config.get("layers", 2)),
            heads=int(raw_config.get("heads", 4)),
            feedforward_dimension=int(raw_config.get("feedforward_dimension", 1024)),
            dropout=float(raw_config.get("dropout", 0.1)),
        )
        self.input_dimension = config.input_dimension
        self.model = _build_attention_model(config).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def predict(self, contexts: Sequence[BoundaryContext]) -> list[float]:
        import torch

        if not contexts:
            return []
        if any(context.vectors.shape != (6, self.input_dimension) for context in contexts):
            shapes = sorted({tuple(context.vectors.shape) for context in contexts})
            raise ValueError(
                f"Checkpoint expects boundary contexts shaped (6, {self.input_dimension}); got {shapes}."
            )
        probabilities: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(contexts), self.batch_size):
                batch = contexts[start : start + self.batch_size]
                vectors = torch.from_numpy(np.stack([item.vectors for item in batch])).to(self.device)
                masks = torch.from_numpy(np.stack([item.valid_mask for item in batch])).to(self.device)
                scores = torch.sigmoid(self.model(vectors, masks)).cpu().tolist()
                probabilities.extend(float(score) for score in scores)
        return probabilities
