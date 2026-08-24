import pytest

from llm_wiki_bench.profiles import (
    CAPABILITIES,
    PROFILES,
    capability_requirements,
    get_profile,
)


def test_every_profile_declares_known_capabilities():
    for profile in PROFILES.values():
        assert profile.capabilities
        assert profile.capabilities <= CAPABILITIES


def test_get_profile_returns_declared_capabilities():
    assert get_profile("grounded_verification").capabilities == frozenset({"label"})


def test_get_profile_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown profile: nope"):
        get_profile("nope")


def test_every_capability_declares_its_required_case_fields():
    for capability in CAPABILITIES:
        assert capability_requirements(capability)


def test_capability_requirements_rejects_unknown_capability():
    with pytest.raises(ValueError, match="unknown capability: nope"):
        capability_requirements("nope")


def test_expected_profiles_are_registered():
    assert set(PROFILES) == {
        "memory_qa",
        "retrieval_qa",
        "multi_slot_retrieval_qa",
        "counterfactual_qa",
        "temporal_discrimination",
        "grounded_verification",
        "claim_decomposition",
    }
