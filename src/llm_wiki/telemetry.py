#!/usr/bin/env python3
"""Opt-in recall telemetry: what was asked, what came back, was it useful.

The gold corpus can only contain questions someone thought to write down. Real
queries are the only source of the ones nobody anticipated — but recording them
silently would turn a local knowledge base into a query log. So this is strictly
opt-in, local-only, and gitignored:

  WIKI_RECALL_TELEMETRY=1 wiki-recall "..."
  wiki-recall "..." --telemetry --label useful

What is stored: the query, the filters used, the returned page IDs, latency, the
decision, and an optional human label. What is NOT stored: page bodies,
snippets, or any retrieved text — the IDs are enough to reconstruct a gold case,
and page content in a log adds exposure without adding signal.

  python3 -m llm_wiki.telemetry propose      # candidate gold cases -> stdout

`propose` only ever prints. Promoting a case into the gold corpus is a human
decision: an unreviewed query/answer pair is precisely the kind of unverified
claim the vault's own rules forbid.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from llm_wiki.paths import VAULT_ROOT, content_root

ENV_FLAG = "WIKI_RECALL_TELEMETRY"
LABELS = ("useful", "not-useful", "wrong", "none-correct")


@dataclass(frozen=True)
class RecallEvent:
    query: str
    filters: dict = field(default_factory=dict)
    result_ids: tuple = ()
    latency_ms: float = 0.0
    decision: str = "list"
    useful: object = None  # label string, True/False, or None

    def as_json(self) -> dict:
        return {
            "query": self.query,
            "filters": dict(self.filters),
            "result_ids": list(self.result_ids),
            "latency_ms": self.latency_ms,
            "decision": self.decision,
            "useful": self.useful,
        }


def default_path(vault: Path = VAULT_ROOT) -> Path:
    return content_root(vault) / ".local" / "recall-events.jsonl"


def enabled(flag: bool = False) -> bool:
    """Opt-in only: an explicit CLI flag, or WIKI_RECALL_TELEMETRY=1."""
    return bool(flag) or os.environ.get(ENV_FLAG, "") == "1"


def append_event(path: Path, event: RecallEvent) -> None:
    """Append one JSON object per line, atomically enough for concurrent recalls.

    O_APPEND makes each write land at the current end of file, so two recalls
    racing cannot interleave into a corrupt line.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event.as_json(), ensure_ascii=False) + "\n"
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def read_events(path: Path) -> tuple:
    """Return (events, errors). A malformed line is reported, never dropped
    silently — a log that quietly loses records is worse than no log."""
    p = Path(path)
    if not p.exists():
        return [], []
    events, errors = [], []
    for n, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            events.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            errors.append(f"{p.name} line {n}: malformed JSON ({exc.msg})")
    return events, errors


def propose_cases(path: Path) -> list:
    """Candidate gold cases from explicitly labeled events only.

    An unlabeled event says nothing about whether recall was right, so it cannot
    become a gold case. `wrong`/`not-useful` events are proposed too — they mark
    a real gap — but with no expected id, flagged for a human to fill in.
    """
    events, _ = read_events(path)
    seen, out = set(), []
    for ev in events:
        label = ev.get("useful")
        if label not in LABELS:
            continue
        q = (ev.get("query") or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        results = ev.get("result_ids") or []
        if label == "useful":
            case = {"q": q, "expect": results[:1], "expect_none": False,
                    "split": "test", "category": "direct", "difficulty": "easy",
                    "needs_human_label": False}
        elif label == "none-correct":
            case = {"q": q, "expect": [], "expect_none": True, "split": "test",
                    "category": "negative", "difficulty": "easy",
                    "needs_human_label": False}
        else:  # wrong | not-useful — a known gap with no verified answer yet
            case = {"q": q, "expect": [], "expect_none": False, "split": "test",
                    "category": "indirect", "difficulty": "hard",
                    "needs_human_label": True, "returned": results[:3]}
        out.append(case)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["propose", "show"])
    ap.add_argument("--path", type=Path, default=default_path())
    args = ap.parse_args()

    if args.command == "propose":
        print(json.dumps(propose_cases(args.path), ensure_ascii=False, indent=2))
        return 0

    events, errors = read_events(args.path)
    print(f"{len(events)} event(s) in {args.path}")
    for e in errors:
        print(f"WARNING: {e}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
