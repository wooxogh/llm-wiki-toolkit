"""Configuration and default artifact paths for Chunker V3."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PACKAGE_DIR.parent / "assets" / "small_v3_50_normal.pt"
DEFAULT_CACHE_ROOT = Path(
    os.environ.get("LLM_WIKI_CACHE", Path.home() / ".cache" / "llm-wiki-v3")
).expanduser()
DEFAULT_CACHE_DIR = DEFAULT_CACHE_ROOT / "sentence_embeddings"


@dataclass(frozen=True)
class ChunkerConfig:
    """Runtime policy for structural parsing and semantic refinement.

    ``boundary_keep_threshold`` is the main tuning control. A lower value keeps
    more candidate boundaries and therefore produces finer chunks. The default
    0.66 is the validation-selected operating point of Small-V3-50-Normal.
    """

    boundary_keep_threshold: float = 0.66
    candidate_budget: float = 0.50
    gate_window_size: int = 2
    attention_context_window: int = 3
    embedding_batch_size: int = 32
    inference_batch_size: int = 128
    device: str = "auto"
    model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    cache_directory: Path = DEFAULT_CACHE_DIR

    def __post_init__(self) -> None:
        if not 0.0 <= self.boundary_keep_threshold <= 1.0:
            raise ValueError("boundary_keep_threshold must be in [0, 1].")
        if not 0.0 < self.candidate_budget <= 1.0:
            raise ValueError("candidate_budget must be in (0, 1].")
        if self.gate_window_size != 2:
            raise ValueError(
                "Small-V3-50-Normal was trained with gate_window_size=2; "
                "changing it requires a separately validated checkpoint."
            )
        if self.attention_context_window != 3:
            raise ValueError(
                "The Small Attention checkpoint architecture requires "
                "attention_context_window=3."
            )
        if self.embedding_batch_size < 1 or self.inference_batch_size < 1:
            raise ValueError("Batch sizes must be positive.")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["checkpoint_path"] = str(self.checkpoint_path)
        payload["cache_directory"] = str(self.cache_directory)
        return payload
