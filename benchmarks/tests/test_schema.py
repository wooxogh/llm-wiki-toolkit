import pytest

from llm_wiki_bench.schema import (
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
        task="answer",
        prompt="What does the evidence establish?",
        labels={"answer": "A supported claim"},
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


@pytest.mark.parametrize("case_kwargs", [{"id": "   "}, {"dataset": "unknown"}, {"prompt": ""}])
def test_invalid_case_identity_dataset_or_prompt_is_rejected(case_kwargs: dict) -> None:
    values = {
        "id": "case-1",
        "dataset": "factlens",
        "split": "test",
        "task": "answer",
        "prompt": "What does the evidence establish?",
        "labels": {"answer": "A supported claim"},
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
            task="answer",
            prompt="What does the evidence establish?",
            labels={"answer": "A supported claim"},
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
