"""Schema constants and enums for the v2 artifact layer."""
from __future__ import annotations

from enum import StrEnum

ARTIFACT_SCHEMA_VERSION = "llm-wiki-v2.1"
CHUNK_SCHEMA_VERSION = "heading-aware-700.v1"
CONCEPT_SCHEMA_VERSION = "atomic-concept.v1"
CONCEPT_PROMPT_VERSION = "concept-extraction.v2"
PLACEMENT_PROMPT_VERSION = "concept-placement.v2"
RELATION_PROMPT_VERSION = "relation-classification.v2"
TEMPORAL_PROMPT_VERSION = "temporal-resolution.v1"
DEFAULT_CHUNK_TARGET_CHARS = 700
DEFAULT_RELATION_CANDIDATE_TOPK = 10
DEFAULT_SAFE_RELATION_MIN_CONFIDENCE = 0.90
DEFAULT_REQUIRE_USER_APPROVAL = ("CONTRADICTS", "SUPERSEDES", "OVERRIDES")


class ConceptState(StrEnum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    DISPUTED = "DISPUTED"
    DUPLICATE = "DUPLICATE"
    ARCHIVED = "ARCHIVED"


class NodeType(StrEnum):
    TOPIC = "TOPIC"
    COLLECTION = "COLLECTION"
    DOCUMENT = "DOCUMENT"
    CONCEPT = "CONCEPT"


class EdgeType(StrEnum):
    PARENT_OF = "PARENT_OF"
    PRIMARY_TOPIC_OF = "PRIMARY_TOPIC_OF"
    SECONDARY_TOPIC_OF = "SECONDARY_TOPIC_OF"
    CONTAINS_DOCUMENT = "CONTAINS_DOCUMENT"
    DOCUMENT_HAS_CONCEPT = "DOCUMENT_HAS_CONCEPT"
    RELATES_TO = "RELATES_TO"


class RelationType(StrEnum):
    SUPPORTS = "SUPPORTS"
    COMPLEMENTS = "COMPLEMENTS"
    CONTRADICTS = "CONTRADICTS"
    SUPERSEDES = "SUPERSEDES"
    OVERRIDES = "OVERRIDES"
    DUPLICATE_OF = "DUPLICATE_OF"


SAFE_RELATIONS = frozenset({
    RelationType.SUPPORTS,
    RelationType.COMPLEMENTS,
    RelationType.DUPLICATE_OF,
})
RISKY_RELATIONS = frozenset({
    RelationType.CONTRADICTS,
    RelationType.SUPERSEDES,
    RelationType.OVERRIDES,
})

CONCEPT_EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["concepts"],
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "summary", "source_quote", "confidence"],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "summary": {"type": "string"},
                    "source_quote": {"type": "string", "minLength": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    },
}

CONCEPT_EXTRACTION_INSTRUCTION = """Extract zero or more Atomic Concepts from exactly this source chunk.

An Atomic Concept is the smallest independently truth-judgeable assertion. Return one concept per independent assertion. Split combined claims. Do not return headings, keywords, questions, TODOs, or unsupported inferences. If the chunk contains no factual, decision, rule, or measurable assertion, return an empty concepts array.

For every concept, source_quote must be an exact contiguous quote from the chunk. text may normalize wording only when it preserves the quote's meaning, subject, scope, conditions, modality, and time/version. Do not add facts. Return JSON only matching the supplied schema."""

PLACEMENT_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["concept_id", "primary_topic_id", "secondary_topic_ids", "create_topic_label",
                 "collection_id", "create_collection_label", "collection_type", "confidence", "reason"],
    "properties": {
        "concept_id": {"type": "string", "minLength": 1},
        "primary_topic_id": {"type": ["string", "null"]},
        "secondary_topic_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "create_topic_label": {"type": ["string", "null"]},
        "collection_id": {"type": ["string", "null"]},
        "create_collection_label": {"type": ["string", "null"]},
        "collection_type": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
}

PLACEMENT_INSTRUCTION = """Propose placement for one Atomic Concept without mutating storage.

Choose an existing Topic when possible. Topic candidates include ids, labels, parents, and state. You may propose one new Topic only when no candidate fits. A Concept has exactly one primary Topic and zero or more distinct secondary Topics. Collection placement is optional and applies to the source Document, not to a copy of the Concept. Use an existing Collection for a recurring document series, or propose a new Collection with a concise type. Respect current user placement unless there is a clear semantic reason to change it. Return JSON only matching the supplied schema."""

RELATION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["proposal"],
    "properties": {
        "proposal": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "source_concept_id", "target_concept_id", "relation",
                                 "confidence", "evidence", "same_subject", "same_scope",
                                 "temporal_change_possible", "reason"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "source_concept_id": {"type": "string", "minLength": 1},
                        "target_concept_id": {"type": "string", "minLength": 1},
                        "relation": {"enum": [r.value for r in RelationType]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence": {"type": "string", "minLength": 1},
                        "same_subject": {"type": "boolean"},
                        "same_scope": {"type": "boolean"},
                        "temporal_change_possible": {"type": "boolean"},
                        "reason": {"type": "string", "minLength": 1},
                    },
                },
            ],
        },
    },
}

RELATION_CLASSIFICATION_INSTRUCTION = """Compare exactly two Atomic Concepts and return either no proposal or one typed relation proposal.

Judge same_subject and same_scope explicitly. CONTRADICTS means the same subject and scope cannot both be true. SUPERSEDES means the source is the newer revision of the target; OVERRIDES means source priority/scope wins rather than time. SUPPORTS, COMPLEMENTS, and DUPLICATE_OF are safe semantic links. evidence must be an exact quote from either supplied Concept. Never approve or mutate state. Risky relations always go to human review regardless of confidence. Return JSON only matching the supplied schema."""

TEMPORAL_JSON_SCHEMA = RELATION_JSON_SCHEMA

TEMPORAL_RESOLUTION_INSTRUCTION = """Resolve whether the source Concept is a temporal or scoped replacement for the target Concept.

Return only SUPERSEDES, OVERRIDES, or no proposal. SUPERSEDES requires same subject, same scope, incompatible values or an explicit update, and either a newer source timestamp or explicit version/revision evidence. Direction is always newer source -> older target. OVERRIDES requires a clear priority/scope rule. Include exact quoted evidence and a short reason. Do not infer recency from file order and never approve or mutate state. Return JSON only matching the supplied schema."""
