import pytest

from llm_wiki_bench.schema import (
    DATASETS,
    BenchmarkCase,
    Prediction,
    validate_case,
    validate_prediction,
)


def test_valid_answerable_case_and_prediction_validate() -> None:
    case = BenchmarkCase(
        id="case-1",
        dataset="factlens",
        split="test",
        profile="retrieval_qa",
        prompt="What does the evidence establish?",
        labels={"answers": ("A supported claim",)},
        context=["Evidence passage"],
        evidence_ids=["doc-1"],
        metadata={"source": "fixture"},
    )
    prediction = Prediction(
        case_id="case-1",
        answer="A supported claim",
        ranked_evidence_ids=["doc-1"],
        cited_evidence_ids=["doc-1"],
        latency_ms=12.5,
    )

    validate_case(case)
    validate_prediction(prediction)
    assert case.context == ("Evidence passage",)
    assert prediction.ranked_evidence_ids == ("doc-1",)


def test_case_copies_and_immutably_exposes_mapping_inputs() -> None:
    labels = {
        "label": "entailment",
        "answer": "A supported claim",
        "nested": {"key": "value"},
        "items": ["item-1"],
        "tags": {"tag-1"},
    }
    metadata = {"source": "fixture"}
    case = BenchmarkCase(
        id="case-1",
        dataset="factlens",
        split="test",
        profile="grounded_verification",
        prompt="What does the evidence establish?",
        labels=labels,
        metadata=metadata,
    )

    labels["answer"] = "Mutated caller value"
    metadata["source"] = "Mutated caller value"

    assert case.labels == {
        "label": "entailment",
        "answer": "A supported claim",
        "nested": {"key": "value"},
        "items": ("item-1",),
        "tags": frozenset({"tag-1"}),
    }
    assert case.metadata == {"source": "fixture"}
    with pytest.raises(TypeError):
        case.labels["answer"] = "Mutated case value"
    with pytest.raises(TypeError):
        case.labels["nested"]["key"] = "Mutated nested case value"
    with pytest.raises(AttributeError):
        case.labels["items"].append("item-2")
    with pytest.raises(AttributeError):
        case.labels["tags"].add("tag-2")
    with pytest.raises(TypeError):
        case.metadata["source"] = "Mutated case value"


@pytest.mark.parametrize("case_kwargs", [{"id": "   "}, {"dataset": "unknown"}, {"prompt": ""}])
def test_invalid_case_identity_dataset_or_prompt_is_rejected(case_kwargs: dict) -> None:
    values = {
        "id": "case-1",
        "dataset": "factlens",
        "split": "test",
        "profile": "grounded_verification",
        "prompt": "What does the evidence establish?",
        "labels": {"label": "entailment"},
    }
    values.update(case_kwargs)

    with pytest.raises(ValueError):
        BenchmarkCase(**values)


@pytest.mark.parametrize(
    "case_kwargs",
    [{"context": None}, {"evidence_ids": None}, {"dataset": []}],
)
def test_invalid_case_container_inputs_raise_value_error(case_kwargs: dict) -> None:
    values = {
        "id": "case-1",
        "dataset": "factlens",
        "split": "test",
        "profile": "grounded_verification",
        "prompt": "What does the evidence establish?",
        "labels": {"label": "entailment"},
    }
    values.update(case_kwargs)

    with pytest.raises(ValueError):
        BenchmarkCase(**values)


def test_duplicate_case_evidence_ids_are_rejected() -> None:
    with pytest.raises(ValueError):
        BenchmarkCase(
            id="case-1",
            dataset="factlens",
            split="test",
            profile="retrieval_qa",
            prompt="What does the evidence establish?",
            labels={"answers": ("A supported claim",)},
            evidence_ids=["doc-1", "doc-1"],
        )


def test_negative_prediction_latency_is_rejected() -> None:
    with pytest.raises(ValueError):
        Prediction(case_id="case-1", latency_ms=-0.1)


def test_non_numeric_prediction_latency_is_rejected() -> None:
    with pytest.raises(ValueError):
        Prediction(case_id="case-1", latency_ms="fast")


def test_abstained_prediction_carrying_answer_is_rejected() -> None:
    with pytest.raises(ValueError):
        Prediction(case_id="case-1", answer="An answer", abstained=True)


def test_case_requires_a_known_profile():
    with pytest.raises(ValueError, match="unknown profile: nope"):
        BenchmarkCase(
            id="c1",
            dataset="vitaminc",
            split="test",
            profile="nope",
            prompt="claim",
            labels={"label": "entailment"},
        )


def test_declared_capability_without_data_is_an_error_not_a_silent_skip():
    """grounded_verification declares `label`; omitting it must fail."""
    with pytest.raises(ValueError, match="grounded_verification requires labels.label"):
        BenchmarkCase(
            id="c1",
            dataset="vitaminc",
            split="test",
            profile="grounded_verification",
            prompt="claim",
            labels={},
        )


def test_retrieval_profile_requires_non_empty_evidence_ids():
    with pytest.raises(ValueError, match="retrieval_qa requires evidence_ids"):
        BenchmarkCase(
            id="c1",
            dataset="rgb_base",
            split="en",
            profile="retrieval_qa",
            prompt="q",
            labels={"answers": ("a",)},
            evidence_ids=(),
        )


def test_memory_qa_case_accepts_two_evidence_granularities():
    case = BenchmarkCase(
        id="q1",
        dataset="longmemeval",
        split="oracle",
        profile="memory_qa",
        prompt="where is the key?",
        labels={"answers": ("the blue vase",)},
        evidence_ids=("s1",),
        fine_evidence_ids=("s1:2",),
        expects_abstention=False,
    )
    assert case.fine_evidence_ids == ("s1:2",)
    assert case.expects_abstention is False


def test_case_no_longer_accepts_a_task_field():
    with pytest.raises(TypeError):
        BenchmarkCase(
            id="c1",
            dataset="vitaminc",
            split="test",
            profile="grounded_verification",
            task="verification",
            prompt="claim",
            labels={"label": "entailment"},
        )


def test_prediction_carries_sub_claim_labels():
    prediction = Prediction(case_id="c1", sub_claim_labels=("true", "false"))
    assert prediction.sub_claim_labels == ("true", "false")


def test_expects_abstention_must_be_boolean():
    with pytest.raises(ValueError, match="expects_abstention must be a boolean"):
        BenchmarkCase(
            id="q1",
            dataset="longmemeval",
            split="oracle",
            profile="memory_qa",
            prompt="q",
            labels={"answers": ("a",)},
            evidence_ids=("s1",),
            fine_evidence_ids=("s1:0",),
            expects_abstention="yes",
        )


def test_seven_suites_are_registered():
    assert DATASETS == frozenset(
        {
            "longmemeval",
            "hoh",
            "vitaminc",
            "factlens",
            "rgb_base",
            "rgb_integration",
            "rgb_counterfactual",
        }
    )
