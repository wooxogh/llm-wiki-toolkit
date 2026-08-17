"""Content-addressed cache for pairwise relation/temporal LLM calls.

Mirrors concept_extraction.py's cache: Concept.id is already a content hash
(concept_store._concept_id), so (source.id, target.id) is a safe key without
needing a separate hash field.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from llm_wiki.v2 import artifacts
from llm_wiki.v2.models import Concept, RelationProposal


def cached_call(kind: str, source: Concept, target: Concept, prompt_version: str,
                model_identity: str, compute: Callable[[], RelationProposal | None],
                vault: Path | None = None) -> RelationProposal | None:
    key = hashlib.sha256(
        f"{kind}|{source.id}|{target.id}|{prompt_version}|{model_identity}".encode()
    ).hexdigest()
    cache = artifacts.artifact_path(f"cache/relations/{key}.json", vault)
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return RelationProposal.from_dict(raw) if raw is not None else None
    proposal = compute()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(proposal.to_dict() if proposal is not None else None, ensure_ascii=False),
        encoding="utf-8",
    )
    return proposal
