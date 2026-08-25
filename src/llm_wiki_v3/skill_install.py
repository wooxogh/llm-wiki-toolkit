"""Install the packaged LLM-Wiki skill for Codex, Claude Code, or both."""
from __future__ import annotations

import argparse
import os
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Iterable


def targets(
    provider: str,
    *,
    home: Path | None = None,
    codex_home: Path | None = None,
) -> list[Path]:
    resolved_home = (home or Path.home()).expanduser()
    values: list[Path] = []
    if provider in {"codex", "both"}:
        resolved_codex_home = codex_home or Path(
            os.environ.get("CODEX_HOME", resolved_home / ".codex")
        )
        values.append(resolved_codex_home.expanduser() / "skills" / "llm-wiki-v3")
    if provider in {"claude", "both"}:
        values.append(resolved_home / ".claude" / "skills" / "llm-wiki-v3")
    return values


def _copy_resources(source, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            _copy_resources(item, target)
        else:
            with item.open("rb") as source_file, target.open("wb") as target_file:
                shutil.copyfileobj(source_file, target_file)


def install(
    provider: str,
    *,
    home: Path | None = None,
    codex_home: Path | None = None,
) -> list[Path]:
    source = files("llm_wiki_v3").joinpath("resources", "llm-wiki-v3")
    destinations = targets(provider, home=home, codex_home=codex_home)
    for destination in destinations:
        _copy_resources(source, destination)
    return destinations


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("codex", "claude", "both"), default="both")
    args = parser.parse_args(list(argv) if argv is not None else None)
    for destination in install(args.provider):
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
