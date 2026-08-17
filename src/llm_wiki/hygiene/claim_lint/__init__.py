#!/usr/bin/env python3
"""Flag claims that read like guesses and carry no measurement.

The vault's founding rule is *measured facts only*. Every other invariant here
is machine-checked (frontmatter schema, index drift, embedding staleness,
community synthesis, retrieval quality), but the one rule the vault is
actually *about* was left entirely to human and LLM discipline. This closes
that gap in the cheapest way that still helps: look for the *shape* of an
unmeasured claim — hedging language with no number, date, PR, or code path near
it.

Deliberately a WARNING, never an error:
  - Hedging is legitimate when a page records uncertainty on purpose
    ("the retry ceiling under real load is **unverified**"). That is a
    measured statement about the absence of measurement.
  - A gate that blocks a commit on prose would get worked around, not obeyed.
  - It is a heuristic. It will miss confident-sounding fabrications entirely —
    those need a reader, not a regex.

The hedge/uncertainty/evidence *vocabulary* is language-specific, but the
insight ("hedging with no measurement nearby") is not. That split lives here
as logic plus per-language pattern packs under `patterns/`, selected by
`[lint] packs` in wiki.toml (see `load_packs`).

  wiki-lint                    # whole vault
  wiki-lint --page <id>
"""
from __future__ import annotations

import argparse
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from llm_wiki import config
from llm_wiki.paths import VAULT_ROOT, content_paths, relative

PACK_DIR = Path(__file__).resolve().parent / "patterns"

# A measurement anchor that no language owns: ISO dates, ratios, percentages,
# grouped thousands, common units, PR references, and code paths. NOT "any
# digit" — a bare number is usually the value being *proposed* ("raise
# concurrency to 60"), which is the claim, not evidence for it. Units require a
# preceding digit for the same reason a bare Korean unit word like 쪽 ("side")
# or 건 (a common counter, also "case/incident") is ordinary prose, not a
# measurement, unless a number is actually attached to it.
_UNIVERSAL_UNITS = "ms|s\\b|KB|MB|GB|rps|req|vCPU|tok"


def _universal_evidence(extra_units: str = "") -> str:
    """Language-neutral evidence anchors, with the unit alternation extended by
    any language-specific units (e.g. ko.toml's 초/분/건/회/개/행/쪽/페이지).

    Units are folded into THIS digit-anchored group rather than appended as
    free-standing top-level alternatives — appending a single-character/word
    unit like 쪽 or 건 at the top level would match it anywhere in prose, not
    just after a number, and reintroduce the exact false-positive class this
    lint exists to avoid.
    """
    units = f"{_UNIVERSAL_UNITS}|{extra_units}" if extra_units else _UNIVERSAL_UNITS
    return (
        r"\d{4}-\d{2}-\d{2}"                              # ISO date
        r"|\d+\s*/\s*\d+"                                 # ratio, e.g. 29/31
        r"|\d+(?:\.\d+)?\s*%"                              # percentage
        r"|\d{1,3}(?:,\d{3})+"                             # grouped thousands
        rf"|\d+(?:\.\d+)?\s*(?:{units})"
        r"|#\d+"                                           # PR reference
        r"|`[^`]*[/.][^`]*`"                                # code path
    )


_STRUCK = re.compile(r"~~.+?~~", re.S)
_FENCE = re.compile(r"```.*?```", re.DOTALL)

_EMPTY = re.compile(r"(?!x)x")  # matches nothing; used when an axis has no patterns


class UnknownPack(Exception):
    """A configured lint pack has no patterns/<name>.toml."""


@dataclass(frozen=True)
class Claim:
    lineno: int
    line: str


@dataclass(frozen=True)
class Patterns:
    hedge: re.Pattern
    literal_sight: re.Pattern | None
    deliberate_uncertainty: re.Pattern
    evidence: re.Pattern


def load_packs(names: tuple[str, ...] | None = None) -> Patterns:
    """Compile one alternation per axis across the named packs.

    Each pack is a TOML file with up to five string keys — `hedge`,
    `literal_sight`, `deliberate_uncertainty`, `evidence`, `evidence_units` —
    each a regex alternation in that pack's language. Packs compose: naming
    more than one combines their patterns per axis with `|`, so a vault that
    writes in two languages can lint both at once (see `test_packs_compose`).
    `evidence_units` is folded into the digit-anchored unit group inside
    `evidence` rather than appended alongside it — see `_universal_evidence`.

    `names` defaults to this vault's `[lint] packs` (`config.load().lint_packs`).
    An unknown pack name is a hard error — a lint that quietly checks nothing
    for a misspelled pack name is worse than one that fails loudly.
    """
    names = tuple(names) if names is not None else config.load().lint_packs
    axes = ("hedge", "literal_sight", "deliberate_uncertainty", "evidence", "evidence_units")
    parts: dict[str, list[str]] = {axis: [] for axis in axes}
    for name in names:
        path = PACK_DIR / f"{name}.toml"
        if not path.is_file():
            raise UnknownPack(f"no lint pack named {name!r} in {PACK_DIR}")
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        for axis in axes:
            value = data.get(axis, "")
            if value:
                parts[axis].append(value)
    evidence_pattern = _universal_evidence("|".join(parts["evidence_units"]))
    return Patterns(
        hedge=re.compile("|".join(parts["hedge"])) if parts["hedge"] else _EMPTY,
        literal_sight=re.compile("|".join(parts["literal_sight"])) if parts["literal_sight"] else None,
        deliberate_uncertainty=(
            re.compile("|".join(parts["deliberate_uncertainty"])) if parts["deliberate_uncertainty"] else _EMPTY
        ),
        evidence=re.compile("|".join([evidence_pattern, *parts["evidence"]])),
    )


def is_hedged(text: str, patterns: Patterns | None = None) -> bool:
    patterns = patterns or load_packs()
    if patterns.deliberate_uncertainty.search(text):
        return False
    if not patterns.hedge.search(text):
        return False
    # A pack may declare a sight/seeming homonym (Korean 보인다 is the one
    # example today: "~로 보인다" hedges, but "X 이 보인다" means X is
    # visible). If this occurrence reads as literal sight, and removing that
    # sight-reading match leaves no other hedge signal, it is an assertion
    # about visibility, not a guess. See patterns/ko.toml for why this
    # carve-out exists.
    if patterns.literal_sight is not None and patterns.literal_sight.search(text):
        without_sight = patterns.literal_sight.sub("", text)
        if not patterns.hedge.search(without_sight):
            return False
    return True


def has_evidence(text: str, patterns: Patterns | None = None) -> bool:
    patterns = patterns or load_packs()
    return bool(patterns.evidence.search(text))


def find_unmeasured_claims(body: str, window: int = 1, patterns: Patterns | None = None) -> list[Claim]:
    """Hedged lines with no evidence on the line or within `window` lines.

    The neighbour window matters: a common house style puts the claim on one
    line and the measurement on the next ("...seems to be the bottleneck.
    \\nmeasured: 3.7s -> 0.41s").
    """
    patterns = patterns or load_packs()
    text = _STRUCK.sub("", _FENCE.sub("", body))
    lines = text.splitlines()
    out: list[Claim] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith(">"):  # quoted speech is data, not a claim
            continue
        if not is_hedged(stripped, patterns):
            continue
        lo, hi = max(0, i - window), min(len(lines), i + window + 1)
        if any(has_evidence(lines[j], patterns) for j in range(lo, hi)):
            continue
        out.append(Claim(lineno=i + 1, line=stripped))
    return out


def lint_vault(vault: Path = VAULT_ROOT) -> dict:
    """id -> (relative path, claims), for pages with at least one unmeasured claim."""
    from llm_wiki import build_index

    # The packs must come from THIS vault's own wiki.toml, not the current
    # process's ambient config — content_paths(vault)/relative(path, vault)
    # below already resolve against `vault`, and a lint that silently used a
    # different vault's language selection would check the wrong words.
    patterns = load_packs(config.load(vault).lint_packs)
    out: dict = {}
    for path in content_paths(vault):
        fm = build_index.parse_frontmatter(path) or {}
        body = path.read_text(encoding="utf-8").split("\n---", 1)[-1]
        claims = find_unmeasured_claims(body, patterns=patterns)
        if claims:
            out[fm.get("id") or path.stem] = (relative(path, vault), claims)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", help="lint a single page id")
    ap.add_argument("--vault", type=Path, default=VAULT_ROOT)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    found = lint_vault(args.vault.resolve())
    if args.page:
        found = {k: v for k, v in found.items() if k == args.page}

    total = sum(len(c) for _, c in found.values())
    for _id, (rel, claims) in sorted(found.items(), key=lambda kv: -len(kv[1][1]))[:args.limit]:
        print(f"\n! {_id}  ({rel}) — {len(claims)} claim(s)")
        for c in claims[:3]:
            print(f"    L{c.lineno}: {c.line[:110]}")
    print(f"\n{total} unmeasured-looking claim(s) across {len(found)} page(s) "
          f"— WARNING only; hedging can be correct, read before editing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
