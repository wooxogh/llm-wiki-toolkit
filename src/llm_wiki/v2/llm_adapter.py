"""LLM seam for v2.

Core v2 code depends on this Protocol only. Adapters may call Codex, Claude, or a
local model, but they cannot mutate storage directly.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Protocol

from llm_wiki import config
from llm_wiki.v2.models import Chunk, Concept, ConceptProposal, PlacementProposal, RelationProposal
from llm_wiki.v2.schemas import (CONCEPT_EXTRACTION_INSTRUCTION,
                                 CONCEPT_EXTRACTION_JSON_SCHEMA, CONCEPT_PROMPT_VERSION,
                                 PLACEMENT_INSTRUCTION, PLACEMENT_JSON_SCHEMA,
                                 PLACEMENT_PROMPT_VERSION, RELATION_CLASSIFICATION_INSTRUCTION,
                                 RELATION_JSON_SCHEMA, RELATION_PROMPT_VERSION,
                                 TEMPORAL_JSON_SCHEMA, TEMPORAL_PROMPT_VERSION,
                                 TEMPORAL_RESOLUTION_INSTRUCTION, RelationType)


class UserLLMAdapter(Protocol):
    def extract_concepts(self, chunk: Chunk) -> list[ConceptProposal]:
        ...

    def place_concept(self, concept: Concept, tree_candidates: list[dict]) -> PlacementProposal:
        ...

    def classify_relation(self, source: Concept, target: Concept) -> RelationProposal | None:
        ...

    def resolve_temporal(self, source: Concept, target: Concept) -> RelationProposal | None:
        ...


MAX_RETRIES = int(os.environ.get("WIKI_V2_LLM_MAX_RETRIES", "3"))
RETRY_BASE_DELAY = float(os.environ.get("WIKI_V2_LLM_RETRY_BASE_DELAY", "2.0"))


def _retry(call: Callable[[], dict], sleep=time.sleep) -> dict:
    last_exc: RuntimeError | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return call()
        except RuntimeError as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                sleep(RETRY_BASE_DELAY * (2 ** attempt))
    raise last_exc


class CommandUserLLMAdapter:
    """Provider-neutral JSON-lines adapter for Codex, Claude, or a local command.

    ``WIKI_V2_LLM_COMMAND`` receives one JSON request on stdin and must emit one
    JSON object on stdout. The command owns provider authentication; this project
    never stores API keys. Set ``WIKI_V2_LLM_MODEL`` to include model identity in
    artifact cache keys.
    """
    def __init__(self, command: str | None = None, model_identity: str | None = None):
        self.command = command or os.environ.get("WIKI_V2_LLM_COMMAND")
        if not self.command:
            raise RuntimeError("WIKI_V2_LLM_COMMAND is not configured")
        self.model_identity = model_identity or os.environ.get("WIKI_V2_LLM_MODEL", self.command)

    def _call(self, task: str, payload: dict) -> dict:
        return _retry(lambda: self._call_once(task, payload))

    def _call_once(self, task: str, payload: dict) -> dict:
        request = json.dumps({"task": task, "payload": payload}, ensure_ascii=False)
        result = subprocess.run(shlex.split(self.command), input=request, text=True,
                                encoding="utf-8", capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError(f"User LLM command failed ({result.returncode}): {result.stderr.strip()}")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("User LLM command did not return JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError("User LLM command must return a JSON object")
        return value

    def extract_concepts(self, chunk: Chunk) -> list[ConceptProposal]:
        data = self._call("extract_concepts", {
            "prompt_version": CONCEPT_PROMPT_VERSION,
            "instruction": CONCEPT_EXTRACTION_INSTRUCTION,
            "json_schema": CONCEPT_EXTRACTION_JSON_SCHEMA,
            "chunk": chunk.to_dict(),
        })
        rows = data.get("concepts", [])
        return [ConceptProposal(**row) for row in rows if isinstance(row, dict)]

    def place_concept(self, concept: Concept, tree_candidates: list[dict]) -> PlacementProposal:
        data = self._call("place_concept", {
            "prompt_version": PLACEMENT_PROMPT_VERSION,
            "instruction": PLACEMENT_INSTRUCTION,
            "json_schema": PLACEMENT_JSON_SCHEMA,
            "concept": concept.to_dict(),
            "tree_candidates": tree_candidates,
        })
        data = dict(data)
        data["secondary_topic_ids"] = list(dict.fromkeys(data.get("secondary_topic_ids", [])))
        return PlacementProposal(**data)

    def classify_relation(self, source: Concept, target: Concept) -> RelationProposal | None:
        data = self._call("classify_relation", {
            "prompt_version": RELATION_PROMPT_VERSION,
            "instruction": RELATION_CLASSIFICATION_INSTRUCTION,
            "json_schema": RELATION_JSON_SCHEMA,
            "source": source.to_dict(), "target": target.to_dict(),
        })
        return RelationProposal.from_dict(data["proposal"]) if data.get("proposal") else None

    def resolve_temporal(self, source: Concept, target: Concept) -> RelationProposal | None:
        data = self._call("resolve_temporal", {
            "prompt_version": TEMPORAL_PROMPT_VERSION,
            "instruction": TEMPORAL_RESOLUTION_INSTRUCTION,
            "json_schema": TEMPORAL_JSON_SCHEMA,
            "direction": "source_newer_to_target_older",
            "source": source.to_dict(), "target": target.to_dict(),
        })
        return RelationProposal.from_dict(data["proposal"]) if data.get("proposal") else None


class AgentCLIUserLLMAdapter(CommandUserLLMAdapter):
    """Invoke the user's authenticated Codex or Claude CLI directly."""

    def __init__(self, agent: str, vault: Path | None = None):
        if agent not in {"codex", "claude"}:
            raise ValueError("v2 agent must be 'codex' or 'claude'")
        self.agent = agent
        self.vault = Path(vault).resolve() if vault is not None else Path.cwd()
        self.model_identity = os.environ.get("WIKI_V2_LLM_MODEL", f"{agent}-cli-default")

    def _call(self, task: str, payload: dict) -> dict:
        return _retry(lambda: self._call_once(task, payload))

    def _call_once(self, task: str, payload: dict) -> dict:
        executable = shutil.which(self.agent)
        if not executable:
            raise RuntimeError(
                f"v2 agent '{self.agent}' is configured but its CLI is not installed or not on PATH"
            )
        request = {"task": task, "payload": payload}
        prompt = (
            "You are the semantic reasoning adapter for llm-wiki v2. "
            "Do not inspect files, run tools, or mutate any state. Evaluate only the JSON request below. "
            "Return only one JSON object matching payload.json_schema exactly.\n\n"
            + json.dumps(request, ensure_ascii=False)
        )
        schema = payload.get("json_schema", {"type": "object"})
        if self.agent == "codex":
            return self._call_codex(executable, prompt, schema)
        return self._call_claude(executable, prompt, schema)

    def _call_codex(self, executable: str, prompt: str, schema: dict) -> dict:
        with tempfile.TemporaryDirectory(prefix="llm-wiki-v2-") as directory:
            schema_path = Path(directory) / "schema.json"
            output_path = Path(directory) / "response.json"
            schema_path.write_text(
                json.dumps(_codex_output_schema(schema), ensure_ascii=False), encoding="utf-8")
            argv = [
                executable, "exec", "--cd", str(self.vault), "--sandbox", "read-only",
                "--skip-git-repo-check", "--ephemeral", "--ignore-rules", "--color", "never",
                "--output-schema", str(schema_path), "--output-last-message", str(output_path), "-",
            ]
            result = subprocess.run(argv, input=prompt, text=True, encoding="utf-8",
                                    capture_output=True, timeout=300)
            if result.returncode:
                raise RuntimeError(f"Codex CLI failed ({result.returncode}): {result.stderr.strip()}")
            raw = output_path.read_text(encoding="utf-8") if output_path.exists() else result.stdout
        return _parse_agent_json(raw, "Codex")

    def _call_claude(self, executable: str, prompt: str, schema: dict) -> dict:
        argv = [executable, "-p", prompt, "--output-format", "json",
                "--json-schema", json.dumps(schema, ensure_ascii=False)]
        result = subprocess.run(argv, text=True, encoding="utf-8", capture_output=True, timeout=300)
        if result.returncode:
            raise RuntimeError(f"Claude CLI failed ({result.returncode}): {result.stderr.strip()}")
        return _parse_agent_json(result.stdout, "Claude")


def _codex_output_schema(schema: dict) -> dict:
    """Return the Codex structured-output subset without weakening our contract.

    Codex rejects ``uniqueItems`` in ``--output-schema``. The full provider-neutral
    schema remains in the prompt, and placement responses are deduplicated after
    parsing, so only the transport schema loses this unsupported annotation.
    """
    return {
        key: (_codex_output_schema(value) if isinstance(value, dict)
              else [_codex_output_schema(item) if isinstance(item, dict) else item for item in value]
              if isinstance(value, list) else value)
        for key, value in schema.items()
        if key != "uniqueItems"
    }


def _parse_agent_json(raw: str, agent: str) -> dict:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{agent} CLI did not return valid JSON") from exc
    if isinstance(value, dict) and isinstance(value.get("structured_output"), dict):
        value = value["structured_output"]
    elif isinstance(value, dict) and isinstance(value.get("result"), str):
        try:
            nested = json.loads(value["result"])
        except json.JSONDecodeError:
            nested = None
        if isinstance(nested, dict):
            value = nested
    if not isinstance(value, dict):
        raise RuntimeError(f"{agent} CLI must return one JSON object")
    return value


def default_adapter(vault: Path | None = None) -> UserLLMAdapter:
    """Resolve advanced command override, configured agent CLI, then offline rules."""
    if os.environ.get("WIKI_V2_LLM_COMMAND"):
        return CommandUserLLMAdapter()
    agent = config.load(vault).v2_agent
    return AgentCLIUserLLMAdapter(agent, vault) if agent else RuleBasedUserLLMAdapter()


class RuleBasedUserLLMAdapter:
    """Small deterministic adapter for tests and offline smoke runs."""

    def extract_concepts(self, chunk: Chunk) -> list[ConceptProposal]:
        proposals: list[ConceptProposal] = []
        for sentence in _sentences(chunk.text):
            claim = _normalize_claim(sentence)
            if not claim:
                continue
            proposals.append(ConceptProposal(
                text=claim,
                summary=claim[:180],
                source_quote=sentence.strip(),
                confidence=0.72,
            ))
        return proposals

    def place_concept(self, concept: Concept, tree_candidates: list[dict]) -> PlacementProposal:
        topics = [(row.get("id", ""), row.get("label", "")) if isinstance(row, dict) else (row, row)
                  for row in tree_candidates]
        best = _best_topic(concept.text, topics)
        if best:
            return PlacementProposal(concept_id=concept.id, primary_topic_id=best, confidence=0.70)
        label = concept.heading_path[-1] if concept.heading_path else _topic_label(concept.text)
        return PlacementProposal(concept_id=concept.id, primary_topic_id=None,
                                 create_topic_label=label, confidence=0.55)

    def classify_relation(self, source: Concept, target: Concept) -> RelationProposal | None:
        if source.id == target.id:
            return None
        source_l, target_l = source.text.lower(), target.text.lower()
        relation = None
        confidence = 0.0
        if source_l == target_l:
            relation, confidence = RelationType.DUPLICATE_OF, 0.99
        elif _token_overlap(source_l, target_l) >= 0.45:
            relation, confidence = RelationType.SUPPORTS, 0.91
        elif any(word in source_l for word in ("not ", "no longer", "deprecated")):
            relation, confidence = RelationType.CONTRADICTS, 0.74
        if not relation:
            return None
        return RelationProposal(
            id=f"proposal:{source.id}:{relation.value}:{target.id}",
            source_concept_id=source.id,
            target_concept_id=target.id,
            relation=relation.value,
            confidence=confidence,
            evidence=source.source_quote,
            same_subject=True,
            same_scope=True,
            temporal_change_possible=False,
            reason="deterministic lexical overlap",
        )

    def resolve_temporal(self, source: Concept, target: Concept) -> RelationProposal | None:
        if any(word in source.text.lower() for word in ("replaces", "supersedes", "no longer")):
            return RelationProposal(
                id=f"proposal:{source.id}:SUPERSEDES:{target.id}",
                source_concept_id=source.id,
                target_concept_id=target.id,
                relation=RelationType.SUPERSEDES.value,
                confidence=0.78,
                evidence=source.source_quote,
                same_subject=True,
                same_scope=True,
                temporal_change_possible=True,
                reason="source contains explicit replacement language",
            )
        return None


def _sentences(text: str) -> list[str]:
    cleaned = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    parts = re.split(r"(?<=[.!?。！？다요함임])\s+|\n+", cleaned)
    return [p.strip(" -\t") for p in parts if p.strip(" -\t")]


def _normalize_claim(sentence: str) -> str | None:
    s = " ".join(sentence.split())
    if len(s) < 8:
        return None
    lower = s.lower()
    claim_markers = (
        " is ", " are ", " uses ", " use ", " should ", " must ", "replaces",
        "supersedes", "no longer", "deprecated", "결정", "사용", "이다", "한다",
        "됩니다", "합니다", "필요", "권장",
    )
    if not any(marker in lower or marker in s for marker in claim_markers):
        return None
    return s


def _topic_label(text: str) -> str:
    words = [w.strip(".,:;()[]{}").lower() for w in text.split() if len(w) > 2]
    return " ".join(words[:3]).title() if words else "General"


def _best_topic(text: str, topics: list[tuple[str, str]]) -> str | None:
    if not topics:
        return None
    text_tokens = set(re.findall(r"[\w가-힣]+", text.lower()))
    scored = []
    for topic_id, label in topics:
        topic_tokens = set(re.findall(r"[\w가-힣]+", label.lower()))
        scored.append((len(text_tokens & topic_tokens), topic_id))
    score, topic_id = max(scored, key=lambda row: row[0])
    return topic_id if score else None


def _token_overlap(a: str, b: str) -> float:
    left = set(re.findall(r"[\w가-힣]+", a))
    right = set(re.findall(r"[\w가-힣]+", b))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
