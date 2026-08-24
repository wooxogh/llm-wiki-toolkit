"""Task profiles declare which capabilities a suite is scored on.

A profile is the suite's contract with the runner. The runner scores exactly
the declared capabilities, so a metric missing from a report is missing because
the profile does not support it, never because a field happened to be empty.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    capabilities: frozenset[str]


# Capability -> the BenchmarkCase attributes or label keys the capability needs.
# "labels.x" denotes a required key inside BenchmarkCase.labels.
_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "retrieval": ("evidence_ids",),
    "fine_retrieval": ("fine_evidence_ids",),
    "answer": ("labels.answers",),
    "multi_slot_answer": ("labels.answer_slots",),
    "citations": ("evidence_ids",),
    "abstention": ("expects_abstention",),
    "label": ("labels.label",),
    "distractor_rejection": ("labels.distractor_answers",),
    "sub_claim_labels": ("labels.sub_claims", "labels.sub_claim_labels"),
}

CAPABILITIES = frozenset(_REQUIREMENTS)

PROFILES: dict[str, Profile] = {
    profile.name: profile
    for profile in (
        Profile("memory_qa", frozenset({"retrieval", "fine_retrieval", "answer", "abstention"})),
        Profile("retrieval_qa", frozenset({"retrieval", "answer", "citations"})),
        Profile("multi_slot_retrieval_qa", frozenset({"retrieval", "multi_slot_answer", "citations"})),
        Profile("counterfactual_qa", frozenset({"retrieval", "answer", "distractor_rejection"})),
        Profile("temporal_discrimination", frozenset({"answer", "distractor_rejection"})),
        Profile("grounded_verification", frozenset({"label"})),
        Profile("claim_decomposition", frozenset({"sub_claim_labels"})),
    )
}


def get_profile(name: str) -> Profile:
    """Return a registered profile or fail clearly."""
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown profile: {name}") from error


def capability_requirements(capability: str) -> tuple[str, ...]:
    """Return the case fields a capability requires to be scoreable."""
    try:
        return _REQUIREMENTS[capability]
    except KeyError as error:
        raise ValueError(f"unknown capability: {capability}") from error
