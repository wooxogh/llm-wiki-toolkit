import pytest

from llm_wiki_bench.registry import (
    OPTIONAL_SUITES,
    REQUIRED_SUITES,
    enabled_adapters,
    get_adapter,
)


def test_every_suite_is_registered():
    for name in (*REQUIRED_SUITES, *OPTIONAL_SUITES):
        assert get_adapter(name).name == name


def test_required_suites_are_the_six_non_optional_ones():
    assert REQUIRED_SUITES == (
        "hoh",
        "longmemeval",
        "rgb_base",
        "rgb_counterfactual",
        "rgb_integration",
        "vitaminc",
    )


def test_factlens_is_the_only_optional_suite():
    assert OPTIONAL_SUITES == ("factlens",)
    assert get_adapter("factlens").required is False


def test_every_adapter_declares_a_profile_and_container():
    for name in (*REQUIRED_SUITES, *OPTIONAL_SUITES):
        adapter = get_adapter(name)
        assert adapter.profile
        assert adapter.container in {"json_array", "jsonl", "csv", "parquet"}
        assert adapter.evidence_id_origin in {"upstream", "synthesized"}


def test_unknown_suite_fails_clearly():
    with pytest.raises(ValueError, match="unknown benchmark adapter: rgb"):
        get_adapter("rgb")


def test_factlens_is_enabled_only_when_configured():
    without = enabled_adapters({"datasets": {}})
    assert [adapter.name for adapter in without] == list(REQUIRED_SUITES)
    with_factlens = enabled_adapters(
        {"datasets": {"factlens": {"path": "benchmarks/fixtures/factlens.csv"}}}
    )
    assert "factlens" in [adapter.name for adapter in with_factlens]
