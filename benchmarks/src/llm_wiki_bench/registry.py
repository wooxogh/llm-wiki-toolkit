"""Named benchmark adapter discovery."""

from __future__ import annotations

from typing import Any

from .adapters import (
    BenchmarkAdapter,
    FactLensAdapter,
    HoHAdapter,
    LongMemEvalAdapter,
    RGBBaseAdapter,
    RGBCounterfactualAdapter,
    RGBIntegrationAdapter,
    VitaminCAdapter,
)


_ADAPTERS: dict[str, BenchmarkAdapter] = {
    adapter.name: adapter
    for adapter in (
        LongMemEvalAdapter(),
        HoHAdapter(),
        VitaminCAdapter(),
        RGBBaseAdapter(),
        RGBIntegrationAdapter(),
        RGBCounterfactualAdapter(),
        FactLensAdapter(),
    )
}


def get_adapter(name: str) -> BenchmarkAdapter:
    """Return a registered adapter or fail clearly for an unsupported suite."""
    try:
        return _ADAPTERS[name]
    except KeyError as error:
        raise ValueError(f"unknown benchmark adapter: {name}") from error


def enabled_adapters(config: dict[str, Any]) -> list[BenchmarkAdapter]:
    """Return required suites plus FactLens only when it has a configured path."""
    adapters = [adapter for adapter in _ADAPTERS.values() if adapter.required]
    datasets = config.get("datasets", {})
    factlens = datasets.get("factlens", {}) if isinstance(datasets, dict) else {}
    if isinstance(factlens, dict) and factlens.get("path"):
        adapters.append(get_adapter("factlens"))
    return adapters
