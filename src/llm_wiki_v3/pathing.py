"""Cross-platform vault path helpers."""
from __future__ import annotations

from pathlib import Path


def relative_to_root(path: Path, root: Path) -> Path:
    """Return a vault-relative path without needlessly rewriting Windows paths.

    GitHub's Windows runner can expose the temporary root through an 8.3 alias
    such as ``RUNNER~1`` while ``Path.resolve()`` expands a child to the long
    account name. Trying the lexical paths first keeps both sides in the same
    representation; resolved paths remain a fallback for ordinary callers.
    """

    error: ValueError | None = None
    for candidate, candidate_root in ((path, root), (path.absolute(), root.absolute())):
        try:
            return candidate.relative_to(candidate_root)
        except ValueError as exc:
            error = exc
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        error = exc
    raise error
