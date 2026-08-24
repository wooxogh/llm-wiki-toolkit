from pathlib import Path

import pytest

from llm_wiki_bench.registry import enabled_adapters, get_adapter


def test_registry_returns_each_required_adapter() -> None:
    assert [get_adapter(name).name for name in ("longmemeval", "hoh", "vitaminc", "rgb")] == [
        "longmemeval",
        "hoh",
        "vitaminc",
        "rgb",
    ]


def test_enabled_adapters_keeps_factlens_unconfigured_without_error() -> None:
    enabled = enabled_adapters({"datasets": {}})

    assert [adapter.name for adapter in enabled] == ["longmemeval", "hoh", "vitaminc", "rgb"]
    assert get_adapter("factlens").required is False


def test_enabled_adapters_includes_factlens_when_it_has_a_path(tmp_path: Path) -> None:
    enabled = enabled_adapters({"datasets": {"factlens": {"path": str(tmp_path / "factlens.jsonl")}}})

    assert [adapter.name for adapter in enabled] == ["longmemeval", "hoh", "vitaminc", "rgb", "factlens"]


def test_registry_rejects_unknown_adapter_name() -> None:
    with pytest.raises(ValueError, match="unknown benchmark adapter"):
        get_adapter("unknown")
