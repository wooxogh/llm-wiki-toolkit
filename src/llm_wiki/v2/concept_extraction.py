"""Validated concept extraction and content-addressed LLM response cache."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from llm_wiki.v2 import artifacts
from llm_wiki.v2.llm_adapter import UserLLMAdapter
from llm_wiki.v2.models import Chunk, ConceptProposal
from llm_wiki.v2.schemas import CONCEPT_PROMPT_VERSION


class ExtractionError(ValueError):
    pass


def extract(chunk: Chunk, adapter: UserLLMAdapter, model_identity: str = "offline") -> list[ConceptProposal]:
    """Return only schema-valid, source-grounded atomic concept proposals."""
    key = hashlib.sha256(f"{chunk.content_hash}|{CONCEPT_PROMPT_VERSION}|{model_identity}".encode()).hexdigest()
    cache = artifacts.artifact_path(f"cache/concepts/{key}.json")
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return _validate(raw, chunk)
    raw = adapter.extract_concepts(chunk)
    proposals = _validate(raw, chunk)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([p.__dict__ for p in proposals], ensure_ascii=False), encoding="utf-8")
    return proposals


def _validate(raw: object, chunk: Chunk) -> list[ConceptProposal]:
    if not isinstance(raw, list):
        raise ExtractionError("concept extraction must return a list")
    out: list[ConceptProposal] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        data = item.__dict__ if isinstance(item, ConceptProposal) else item
        if not isinstance(data, dict):
            raise ExtractionError("concept extraction item must be an object")
        required = ("text", "summary", "source_quote", "confidence")
        if any(not isinstance(data.get(k), str) for k in required[:3]):
            raise ExtractionError("concept extraction item has invalid text fields")
        try:
            confidence = float(data["confidence"])
        except (TypeError, ValueError) as exc:
            raise ExtractionError("concept confidence must be numeric") from exc
        quote, text = data["source_quote"].strip(), data["text"].strip()
        if not quote or quote not in chunk.text:
            continue
        if not text or len(text) < 3 or not 0 <= confidence <= 1:
            continue
        if not _meaningfully_grounded(text, quote):
            continue
        key = (text, quote)
        if key not in seen:
            seen.add(key)
            out.append(ConceptProposal(text, data["summary"].strip(), quote, confidence))
    return out


def _meaningfully_grounded(text: str, quote: str) -> bool:
    """Cheap deterministic guard: a quote must support some asserted content.

    This cannot prove semantic equivalence, but it rejects unrelated claims such
    as ``The moon is green`` paired with an arbitrary source quote. Provider
    output is still a proposal, never automatic truth.
    """
    tokens = lambda value: set(re.findall(r"[A-Za-z0-9_]+|[가-힣]{2,}", value.lower()))
    ignored = {"this", "that", "there", "using", "uses", "used", "with", "from", "the", "and", "is", "are"}
    claim = tokens(text) - ignored
    source = tokens(quote) - ignored
    return bool(claim & source)
