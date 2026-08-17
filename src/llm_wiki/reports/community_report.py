#!/usr/bin/env python3
"""GraphRAG-style community summaries for the vault, adapted to this project's
constraints (offline, no API key needed at query time, measured-facts-only).

Two layers, cleanly separated so the deterministic part never depends on an LLM:

  1. EXTRACTIVE (this tool, deterministic): for each link-graph community
     (from graph_report), emit a grounded digest = the member pages' own
     `summary:` frontmatter lines. Zero hallucination — it only aggregates
     facts already written on the pages.

  2. ABSTRACTIVE (optional sidecar `community_summaries.json`): a 1-2 sentence
     synthesis per community, keyed by a stable signature (hash of member
     ids). Written by whatever process synthesizes community summaries
     (typically an LLM session working through the vault's ingest workflow)
     and marked GENERATED. When membership changes the signature changes, so
     a stale synthesis is dropped and flagged "awaiting synthesis" until
     regenerated.

This enables GraphRAG-style global/thematic queries ("what does the whole
corpus say about a given workaround technique") without breaking the offline,
key-free recall path.

  python -m llm_wiki.reports.community_report            # print COMMUNITIES.md to stdout
  python -m llm_wiki.reports.community_report --write     # write COMMUNITIES.md
  python -m llm_wiki.reports.community_report --stale     # list signatures needing a synthesis
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from llm_wiki.paths import VAULT_ROOT, content_paths, content_root
from llm_wiki.reports.graph_report import analyze, collect, parse_page

OUT = content_root() / "COMMUNITIES.md"
SIDECAR = content_root() / "community_summaries.json"  # optional, hand- or LLM-authored, committed

# Decided by verification (2026-07-24): communities of size <= 2 get ~0
# marginal value from an abstractive synthesis (member-recall@3n = 1.0 —
# retrieval already finds both members, and reading two pages is trivial). So
# the synthesis *requirement* (flagged as awaiting / --stale) only kicks in
# once size >= MIN_SYNTH_SIZE. An existing synthesis still renders regardless
# of size.
MIN_SYNTH_SIZE = 3


def page_summaries(vault: Path = VAULT_ROOT) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in content_paths(vault):
        fm, _ = parse_page(p)
        if fm.get("id"):
            out[fm["id"]] = (fm.get("summary") or "").strip()
    return out


def signature(members: list[str]) -> str:
    return hashlib.sha1("\n".join(sorted(members)).encode("utf-8")).hexdigest()[:12]


def load_sidecar(vault: Path = VAULT_ROOT) -> dict:
    """The abstractive-synthesis sidecar for `vault`, or `{}` if it has none.

    A missing sidecar is the normal state for a freshly created vault — every
    community simply renders without a `synthesis`, nothing errors.
    """
    path = content_root(vault) / "community_summaries.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def build(comms: list[dict], min_synth: int = MIN_SYNTH_SIZE, vault: Path = VAULT_ROOT):
    side = load_sidecar(vault)
    rows = []
    for c in comms:
        sig = signature(c["members"])
        rows.append({**c, "sig": sig, "synthesis": side.get(sig),
                     "needs_synth": c["size"] >= min_synth})
    return rows


def community_rows(vault: Path = VAULT_ROOT) -> list[dict]:
    """Every community in `vault`, each annotated with its signature and the
    sidecar synthesis (if any). Read-only."""
    return build(analyze(collect(vault))["communities"], vault=vault)


def stale_communities(vault: Path = VAULT_ROOT) -> list[dict]:
    """Communities large enough to require an abstractive synthesis that do not
    have one for their *current* membership signature.

    Membership changes change the signature, so an old synthesis for a mutated
    community shows up here rather than being served as if it still described
    the members. Callers treat a non-empty result as fatal — a stale synthesis
    is a confidently-wrong answer, not a gap.
    """
    return [
        {"sig": r["sig"], "label": r["label"], "hub": r["hub"], "members": r["members"]}
        for r in community_rows(vault)
        if not r["synthesis"] and r["needs_synth"]
    ]


def render_md(rows: list[dict], vault: Path = VAULT_ROOT) -> str:
    sums = page_summaries(vault)
    L = ["# COMMUNITIES — link-graph community summaries (GraphRAG-style)", ""]
    L.append("> The EXTRACTIVE digest is generated deterministically by `community_report` "
              "(aggregates member `summary:` fields, zero hallucination).")
    L.append("> `Synthesis (GENERATED)` is an abstractive synthesis stored per signature "
              "in `community_summaries.json`.")
    L.append("")
    need = [r for r in rows if not r["synthesis"] and r["needs_synth"]]
    if need:
        L.append(f"WARNING: {len(need)} communities awaiting synthesis: " +
                  ", ".join(f"`{r['label']}`({r['sig']})" for r in need))
        L.append("")
    for k, r in enumerate(rows, 1):
        mix = ", ".join(f"{d}x{n}" for d, n in r["domains"].items())
        L.append(f"## C{k} · `{r['label']}` ({r['size']}) — hub `{r['hub']}`")
        L.append(f"<!-- sig:{r['sig']} -->")
        L.append(f"*{mix}*")
        L.append("")
        if r["synthesis"]:
            L.append(f"> **Synthesis (GENERATED):** {r['synthesis']}")
        elif r["needs_synth"]:
            L.append("> **Synthesis (GENERATED):** _awaiting synthesis_")
        else:
            L.append(f"> _small (size < {MIN_SYNTH_SIZE}) — the evidence summary below is "
                      f"enough, abstractive synthesis skipped_")
        L.append("")
        L.append("Evidence (member page summaries):")
        for m in r["members"]:
            s = sums.get(m, "")
            L.append(f"- `{m}` — {s}" if s else f"- `{m}`")
        L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--stale", action="store_true", help="list communities lacking a synthesis")
    args = ap.parse_args()

    rows = community_rows()

    if args.stale:
        for r in stale_communities():
            print(json.dumps(r, ensure_ascii=False))
        return 0

    md = render_md(rows)
    print(md)
    if args.write:
        OUT.write_text(md + "\n", encoding="utf-8")
        print(f"\n[written] {OUT.relative_to(content_root())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
