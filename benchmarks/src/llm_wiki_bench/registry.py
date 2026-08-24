"""Named benchmark adapter discovery.

The registry is the single source of truth for which suites exist; the runner
no longer keeps its own list.
"""

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

REQUIRED_SUITES: tuple[str, ...] = tuple(
    sorted(name for name, adapter in _ADAPTERS.items() if adapter.required)
)
OPTIONAL_SUITES: tuple[str, ...] = tuple(
    sorted(name for name, adapter in _ADAPTERS.items() if not adapter.required)
)


def get_adapter(name: str) -> BenchmarkAdapter:
    """Return a registered adapter or fail clearly for an unsupported suite."""
    try:
        return _ADAPTERS[name]
    except KeyError as error:
        raise ValueError(f"unknown benchmark adapter: {name}") from error


def enabled_adapters(config: dict[str, Any]) -> list[BenchmarkAdapter]:
    """Return required suites plus each optional suite that has a configured path."""
    adapters = [get_adapter(name) for name in REQUIRED_SUITES]
    datasets = config.get("datasets", {})
    if not isinstance(datasets, dict):
        return adapters
    for name in OPTIONAL_SUITES:
        details = datasets.get(name, {})
        if isinstance(details, dict) and details.get("path"):
            adapters.append(get_adapter(name))
    return adapters
