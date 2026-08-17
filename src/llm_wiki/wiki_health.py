#!/usr/bin/env python3
"""Read-only health gate over the vault's derived artifacts.

Canonical Markdown pages are the source of truth; index.yaml, .embeddings/, and
community_summaries.json are all *derived*. Each can silently fall out of sync
with the pages, and every one of those drifts degrades recall in a way that
looks like a working system returning wrong answers. This command turns each
drift class into a named, exit-coded failure.

  wiki-health --mode full        # local: includes .embeddings/
  wiki-health --mode ci          # CI: skips local-only artifacts
  wiki-health --mode full --json

`full` is for the machine that owns the embedding store; `ci` runs where
`.embeddings/` is gitignored and absent, so embedding checks would be noise.

This command NEVER writes: it does not rebuild the index, re-embed, or touch a
page body. It only reports. Exit 1 iff at least one issue has severity "error".
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

from llm_wiki import build_index
from llm_wiki import config
from llm_wiki.paths import (VAULT_ROOT, content_paths, content_root, embeddings_dir,
                            index_path, page_hash, relative)

MODES = ("full", "ci")
# CTX_CHARS in embed_index: a summary longer than this gets truncated into every
# chunk prefix, so it is both a retrieval and a page-hygiene signal.
SUMMARY_WARN_CHARS = 300


@dataclass(frozen=True)
class HealthIssue:
    code: str
    severity: str  # "error" | "warning"
    detail: str


def _error(code: str, detail: str) -> HealthIssue:
    return HealthIssue(code=code, severity="error", detail=detail)


def _warning(code: str, detail: str) -> HealthIssue:
    return HealthIssue(code=code, severity="warning", detail=detail)


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------


def check_index_health(vault: Path) -> list[HealthIssue]:
    pages = build_index.collect_pages(vault)
    errors = build_index.validate(pages, vault)
    if errors:
        # An invalid page makes the *expected* index undefined, so comparing
        # against the committed one would only add a confusing second failure.
        return [_error("index-invalid", f"{len(errors)} frontmatter error(s): {errors[0]}")]
    if build_index.check_index(vault, index_path(vault)):
        return [_error("index-stale", build_index.STALE_INDEX)]
    return []


# --------------------------------------------------------------------------
# embeddings (full mode only)
# --------------------------------------------------------------------------


def _expected_identity() -> str:
    """`MODEL|CHUNK_SCHEMA|META_SCHEMA` as the current code would write it.

    Imported lazily and defensively: embed_index pulls in numpy and _embedder,
    and this must not become the reason `--mode ci` needs them.
    """
    try:
        from llm_wiki.retrieval.embed_index import CHUNK_SCHEMA, META_SCHEMA
        from llm_wiki.retrieval._embedder import MODEL
    except Exception:
        return ""
    return f"{MODEL}|{CHUNK_SCHEMA}|{META_SCHEMA}"


def check_embedding_health(vault: Path) -> list[HealthIssue]:
    emb = embeddings_dir(vault)
    vectors, meta_path = emb / "vectors.npy", emb / "meta.json"
    pages_path, model_path = emb / "pages.json", emb / "model.txt"

    if not (vectors.exists() and meta_path.exists() and pages_path.exists()):
        # One issue, not one per page — an absent store is a single fact.
        return [_error("embedding-store-missing",
                       f"no embedding store at {relative(emb, vault)} — run wiki-embed --full")]

    issues: list[HealthIssue] = []
    if not model_path.exists():
        issues.append(_error("embedding-model-missing",
                             "model.txt absent — store identity (model|chunk-schema) unknown"))
    else:
        stored = model_path.read_text(encoding="utf-8").strip()
        expected = _expected_identity()
        # A model or chunk/meta-schema change makes every stored vector
        # meaningless, but leaves the file present and every hash matching — so
        # an existence check alone reports a fully stale store as healthy.
        if expected and stored != expected:
            issues.append(_error(
                "embedding-identity-stale",
                f"store built as {stored!r} but code expects {expected!r} — "
                f"run wiki-embed --full"))

    current = {}
    for path in content_paths(vault):
        fm = build_index.parse_frontmatter(path) or {}
        current[fm.get("id") or path.stem] = page_hash(path.read_text(encoding="utf-8"))
    stored = json.loads(pages_path.read_text(encoding="utf-8"))

    missing = sorted(set(current) - set(stored))
    if missing:
        issues.append(_error("embedding-page-missing",
                             f"{len(missing)} page(s) never embedded: {', '.join(missing[:5])}"))
    stale = sorted(i for i in set(current) & set(stored) if current[i] != stored[i])
    if stale:
        issues.append(_error("embedding-page-stale",
                             f"{len(stale)} page(s) changed since embedding: {', '.join(stale[:5])}"))
    deleted = sorted(set(stored) - set(current))
    if deleted:
        issues.append(_error("embedding-page-deleted",
                             f"{len(deleted)} embedded page(s) no longer exist: {', '.join(deleted[:5])}"))

    # mmap: row count only. Never pull ~8 MB of float32 into memory for a shape check.
    import numpy as np

    rows = int(np.load(vectors, mmap_mode="r").shape[0])
    n_meta = len(json.loads(meta_path.read_text(encoding="utf-8")))
    if rows != n_meta:
        issues.append(_error("embedding-row-mismatch",
                             f"vectors.npy has {rows} row(s) but meta.json has {n_meta} record(s)"))
    return issues


# --------------------------------------------------------------------------
# community synthesis
# --------------------------------------------------------------------------


def check_community_health(vault: Path) -> list[HealthIssue]:
    from llm_wiki.reports import community_report

    stale = community_report.stale_communities(vault)
    if not stale:
        return []
    labels = ", ".join(f"{c['label']}({c['sig']})" for c in stale[:5])
    return [_error("community-synthesis-stale",
                   f"{len(stale)} community/communities await a grounded synthesis: {labels} — "
                   f"add community_summaries.json at the vault's content root, keyed by the signature "
                   f"shown above (e.g. \"{stale[0]['sig']}\"), with a short grounded synthesis "
                   f"of that community's member pages as the value")]


# --------------------------------------------------------------------------
# content warnings (never fatal)
# --------------------------------------------------------------------------


def check_report_health(vault: Path) -> list[HealthIssue]:
    """GRAPH_REPORT.md / COMMUNITIES.md are generated from the same pages as
    index.yaml, so a stale one is the same class of silently-wrong artifact.

    An *absent* report is not drift — a vault that never generated them is not
    lying about anything; regenerating is the ingest pipeline's job.
    """
    from llm_wiki.reports import community_report, graph_report

    stale = []
    root = content_root(vault)
    graph_path = root / "GRAPH_REPORT.md"
    if graph_path.exists():
        expected = graph_report.render_md(graph_report.analyze(graph_report.collect(vault)))
        if graph_path.read_text(encoding="utf-8").strip() != expected.strip():
            stale.append("GRAPH_REPORT.md")

    comm_path = root / "COMMUNITIES.md"
    if comm_path.exists():
        expected = community_report.render_md(community_report.community_rows(vault), vault)
        if comm_path.read_text(encoding="utf-8").strip() != expected.strip():
            stale.append("COMMUNITIES.md")

    if not stale:
        return []
    return [_error("report-stale",
                   f"generated report(s) differ from the current pages: {', '.join(stale)} — "
                   f"run python -m llm_wiki.reports.graph_report --write / "
                   f"python -m llm_wiki.reports.community_report --write")]


def check_content_warnings(vault: Path) -> list[HealthIssue]:
    """Knowledge-hygiene signals. All non-fatal: they describe how good the
    knowledge is, not whether the artifacts are consistent, and a vault should
    not fail CI for having an orphan page."""
    from llm_wiki.reports import graph_report

    out: list[HealthIssue] = []
    pages = graph_report.collect(vault)

    dangling = sorted(
        f"{_id} → [[{t}]]" for _id, m in pages.items() for t in m["dangling"]
    )
    if dangling:
        out.append(_warning("dangling-wikilink",
                            f"{len(dangling)} unresolved wikilink(s): {'; '.join(dangling[:5])}"))

    if pages:
        graph_report.build_degrees(pages)
        orphans = sorted(i for i, m in pages.items() if m["indeg"] == 0)
        if orphans:
            pct = len(orphans) * 100 // len(pages)
            out.append(_warning(
                "orphan-pages",
                f"{len(orphans)}/{len(pages)} ({pct}%) page(s) have no inbound link: "
                f"{', '.join(orphans[:5])}"))

    from llm_wiki.hygiene import claim_lint

    unmeasured = claim_lint.lint_vault(vault)
    if unmeasured:
        total = sum(len(c) for _, c in unmeasured.values())
        worst = sorted(unmeasured.items(), key=lambda kv: -len(kv[1][1]))[:3]
        out.append(_warning(
            "unmeasured-claim",
            f"{total} hedged claim(s) with no nearby measurement across "
            f"{len(unmeasured)} page(s): " + ", ".join(f"{i}({len(c)})" for i, (_, c) in worst)
            + " — heuristic; hedging can be correct, read before editing"))

    long_summaries = []
    for path in content_paths(vault):
        fm = build_index.parse_frontmatter(path) or {}
        summary = " ".join((fm.get("summary") or "").split())
        if len(summary) > SUMMARY_WARN_CHARS:
            long_summaries.append(f"{fm.get('id') or path.stem} ({len(summary)})")
    if long_summaries:
        out.append(_warning(
            "summary-too-long",
            f"{len(long_summaries)} summary/summaries over {SUMMARY_WARN_CHARS} chars — "
            f"contextual retrieval prefixes it onto EVERY chunk of the page: "
            f"{', '.join(long_summaries[:5])}"))

    return out


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------


def check_health(vault: Path = VAULT_ROOT, mode: str = "full",
                 v2_only: bool = False) -> list[HealthIssue]:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(MODES)}")
    issues: list[HealthIssue] = []
    if not v2_only:
        issues = check_index_health(vault)
        if issues and issues[0].code == "index-invalid":
            # Legacy downstream checks all assume parseable frontmatter.
            return issues
        if mode == "full":
            issues.extend(check_embedding_health(vault))
        issues.extend(check_report_health(vault))
        issues.extend(check_community_health(vault))
        issues.extend(check_content_warnings(vault))
    try:
        cfg = config.load(vault)
    except config.ConfigError:
        cfg = None
    v2_root = content_root(vault) / ".llm_wiki_v2"
    # v2 artifacts are lightweight local JSON/NumPy data; unlike the legacy
    # page embedding store they are valid CI inputs and must not be skipped.
    if v2_only or v2_root.exists() or (cfg and cfg.v2_enabled):
        from llm_wiki.v2.health import check_v2_health
        issues.extend(_error("v2-health", issue) for issue in check_v2_health(vault))
    return issues


def exit_code(issues: list[HealthIssue]) -> int:
    return 1 if any(i.severity == "error" for i in issues) else 0


def report(vault: Path = VAULT_ROOT, mode: str = "full", v2_only: bool = False) -> dict:
    issues = check_health(vault, mode, v2_only)
    return {
        "ok": exit_code(issues) == 0,
        "mode": mode,
        "scope": "v2" if v2_only else "all",
        "errors": [asdict(i) for i in issues if i.severity == "error"],
        "warnings": [asdict(i) for i in issues if i.severity == "warning"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=list(MODES))
    ap.add_argument("--v2", action="store_true",
                    help="check only v2 source/Concept/index/NET artifacts; no YAML frontmatter required")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--vault", type=Path, default=VAULT_ROOT)
    args = ap.parse_args()

    data = report(args.vault.resolve(), args.mode, args.v2)
    if args.json:
        print(json.dumps(data, ensure_ascii=False))
    else:
        for e in data["errors"]:
            print(f"ERROR [{e['code']}] {e['detail']}", file=sys.stderr)
        for w in data["warnings"]:
            print(f"WARNING [{w['code']}] {w['detail']}", file=sys.stderr)
        print(f"{'OK: healthy' if data['ok'] else 'ERROR: unhealthy'} (mode={args.mode}, scope={data['scope']}, "
              f"{len(data['errors'])} error(s), {len(data['warnings'])} warning(s))")
    return 0 if data["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
