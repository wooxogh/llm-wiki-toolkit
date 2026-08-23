"""Two-stage hybrid candidate discovery; never compare every concept pair."""
from __future__ import annotations

from pathlib import Path

from llm_wiki.v2 import concept_index
from llm_wiki.v2.models import Concept


def discover(vault: Path | None, seed: Concept, concepts: list[Concept],
            top_k: int = 10) -> list[tuple[float, Concept]]:
    """Fuse dense/text concept ranking with cheap document/topic priors."""
    ranked = concept_index.search(vault, seed.text, k=max(top_k * 4, 20), concepts=concepts)
    scored: list[tuple[float, Concept]] = []
    for score, candidate in ranked:
        if candidate.id == seed.id:
            continue
        prior = 0.002 if candidate.document_id == seed.document_id else 0.0
        if seed.primary_topic_id and seed.primary_topic_id == candidate.primary_topic_id:
            prior += 0.001
        scored.append((score + prior, candidate))
    return sorted(scored, key=lambda row: (-row[0], row[1].id))[:top_k]
