"""Small terminal progress helpers shared by long-running CLI commands."""
from __future__ import annotations

import sys
from collections.abc import Mapping

from tqdm import tqdm


class PhaseProgress:
    """Render sequential build phases as compact tqdm progress bars."""

    def __init__(self, phases: Mapping[str, tuple[str, str]]) -> None:
        self._phases = phases
        self._phase: str | None = None
        self._bar = None

    def update(self, phase: str, completed: int, total: int, label: str = "") -> None:
        if total <= 0:
            return
        if self._phase != phase:
            self.close()
            description, unit = self._phases.get(phase, (phase, "item"))
            self._bar = tqdm(
                total=total,
                desc=description,
                unit=unit,
                dynamic_ncols=True,
                leave=True,
                disable=None,
                file=sys.stderr,
            )
            self._phase = phase
        elif self._bar.total != total:
            self._bar.total = total
            self._bar.refresh()

        if label:
            compact = label.replace("\n", " ")
            self._bar.set_postfix_str(compact[-60:], refresh=False)
        self._bar.update(max(0, min(completed, total) - self._bar.n))

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
        self._bar = None
        self._phase = None

