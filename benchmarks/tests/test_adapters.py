from pathlib import Path

import pytest

from llm_wiki_bench.adapters.factlens import FactLensAdapter
from llm_wiki_bench.adapters.hoh import HoHAdapter
from llm_wiki_bench.adapters.longmemeval import LongMemEvalAdapter
from llm_wiki_bench.adapters.rgb import RGBAdapter
from llm_wiki_bench.adapters.vitaminc import VitaminCAdapter


FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.mark.parametrize(
    ("adapter", "fixture", "dataset", "task", "labels"),
    [
        (LongMemEvalAdapter(), "longmemeval.jsonl", "longmemeval", "memory_qa", {"answers": ("the blue vase",)}),
        (HoHAdapter(), "hoh.jsonl", "hoh", "multi_hop_qa", {"answers": ("Larkspur",)}),
        (VitaminCAdapter(), "vitaminc.jsonl", "vitaminc", "verification", {"label": "entailment"}),
        (RGBAdapter(), "rgb.jsonl", "rgb", "rag_qa", {"answers": ("2022",)}),
    ],
)
def test_required_adapter_normalizes_its_fixture_contract(
    adapter, fixture: str, dataset: str, task: str, labels: dict
) -> None:
    case = adapter.load(FIXTURES / fixture)[0]

    assert case.dataset == dataset
    assert case.task == task
    assert case.labels == labels
    assert case.metadata["source_version"] == f"{dataset}-fixture-v1"
    assert case.metadata["source_path"] == str(FIXTURES / fixture)
    assert case.metadata["source_record"] == 1


def test_hoh_preserves_all_two_hop_evidence_ids() -> None:
    case = HoHAdapter().load(FIXTURES / "hoh.jsonl")[0]

    assert case.evidence_ids == ("hop-1", "hop-2")


def test_vitaminc_unknown_label_names_its_source_record(tmp_path: Path) -> None:
    source = tmp_path / "vitaminc.jsonl"
    source.write_text(
        '{"id":"vc-bad","split":"test","source_version":"v1","claim":"Claim","evidence":"Evidence","evidence_id":"e-1","label":"MAYBE"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"record 1.*MAYBE"):
        VitaminCAdapter().load(source)


def test_factlens_normalizes_its_verification_fixture() -> None:
    case = FactLensAdapter().load(FIXTURES / "factlens.jsonl")[0]

    assert case.dataset == "factlens"
    assert case.task == "factual_consistency"
    assert case.labels == {"label": "supported"}
    assert case.evidence_ids == ("fl-source-1",)
