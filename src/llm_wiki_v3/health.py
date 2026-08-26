"""Artifact health, evidence packaging, and approved hygiene actions."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, load
from .hygiene import apply_decision, apply_events, read_events
from .indexing import build, collect_markdown
from .io import configure_stdio, read_json, read_jsonl
from .models import HealthIssue
from .pathing import relative_to_root


def _issue(severity: str, code: str, detail: str) -> HealthIssue:
    return HealthIssue(severity, code, detail)


def inspect(config: Config) -> list[HealthIssue]:
    issues: list[HealthIssue] = []
    manifest_path = config.artifact_dir / "manifest.json"
    chunks_path = config.artifact_dir / "chunks.jsonl"
    vectors_path = config.artifact_dir / "vectors.npy"
    if not manifest_path.is_file() or not chunks_path.is_file() or not vectors_path.is_file():
        return [_issue("error", "index-missing", f"run wiki-embed for {config.root}")]

    try:
        manifest = read_json(manifest_path, {}) or {}
        base_chunks = read_jsonl(chunks_path)
        chunks = apply_events(base_chunks, read_events(config))
        vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [_issue("error", "index-invalid", str(exc))]

    if len(vectors) != len(chunks):
        issues.append(_issue("error", "vector-row-mismatch", f"vectors={len(vectors)} chunks={len(chunks)}"))
    if int(manifest.get("chunk_count", -1)) != len(chunks):
        issues.append(_issue("error", "manifest-chunk-mismatch", f"manifest={manifest.get('chunk_count')} chunks={len(chunks)}"))

    current_hashes = {}
    for path in collect_markdown(config):
        relative = relative_to_root(path, config.root).as_posix()
        current_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if current_hashes != manifest.get("documents", {}):
        issues.append(_issue("error", "index-stale", "Markdown files changed since wiki-embed"))

    by_id = {str(chunk.get("id")): chunk for chunk in chunks}
    for chunk in chunks:
        source_path = config.root / str(chunk.get("source_path") or "")
        if not source_path.is_file():
            issues.append(_issue("error", "source-missing", f"{chunk.get('id')}: {source_path}"))
            continue
        source = source_path.read_text(encoding="utf-8")
        start, end = int(chunk.get("source_start", -1)), int(chunk.get("source_end", -1))
        if start < 0 or end < start or end > len(source):
            issues.append(_issue("error", "source-span-invalid", f"{chunk.get('id')}: {start}:{end}"))
            continue
        exact = source[start:end]
        if exact != chunk.get("source_text"):
            issues.append(_issue("error", "source-span-drift", str(chunk.get("id"))))
        digest = hashlib.sha256(exact.encode("utf-8")).hexdigest()
        if digest != chunk.get("content_hash"):
            issues.append(_issue("error", "content-hash-drift", str(chunk.get("id"))))

    tree = read_json(config.artifact_dir / "tree.json", {}) or {}
    tree_ids = {
        str(chunk_id)
        for node in tree.get("nodes", [])
        for chunk_id in node.get("chunk_ids", [])
    }
    unknown_tree_ids = tree_ids - set(by_id)
    if unknown_tree_ids:
        issues.append(_issue("error", "tree-dangling-chunk", ", ".join(sorted(unknown_tree_ids)[:5])))
    if set(by_id) - tree_ids:
        issues.append(_issue("error", "tree-missing-chunk", f"{len(set(by_id) - tree_ids)} chunk(s) absent"))

    sparse_rows = read_json(config.artifact_dir / "sparse_index" / "documents.json", []) or []
    sparse_ids = [str(row.get("chunk_id")) for row in sparse_rows]
    chunk_ids = [str(chunk.get("id")) for chunk in chunks]
    if sparse_ids != chunk_ids:
        issues.append(_issue("error", "sparse-index-mismatch", f"sparse={len(sparse_ids)} chunks={len(chunk_ids)}"))

    graph_path = config.artifact_dir / "knn_graph.npz"
    if not graph_path.is_file():
        issues.append(_issue("error", "knn-missing", str(graph_path)))
    else:
        graph = np.load(graph_path, allow_pickle=False)
        indices = np.asarray(graph["indices"])
        scores = np.asarray(graph["scores"])
        if len(indices) != len(chunks) or indices.shape != scores.shape:
            issues.append(_issue("error", "knn-shape-mismatch", f"indices={indices.shape} scores={scores.shape} chunks={len(chunks)}"))
        elif indices.size and (int(indices.min()) < 0 or int(indices.max()) >= len(chunks)):
            issues.append(_issue("error", "knn-index-out-of-range", f"valid rows: 0..{len(chunks) - 1}"))

    for event in read_events(config):
        event_id = str(event.get("event_id") or "unknown-event")
        event_type = event.get("type")
        if event_type == "partial_supersede":
            fields = ["old_chunk_id"] + (["superseding_chunk_id"] if event.get("superseding_chunk_id") else [])
            for field in fields:
                if str(event.get(field)) not in by_id:
                    issues.append(_issue("error", "event-dangling-chunk", f"{event_id}: {field}={event.get(field)}"))
            if event.get("resolution_source"):
                source = config.root / str(event["resolution_source"])
                if not source.is_file():
                    issues.append(_issue("error", "resolution-source-missing", f"{event_id}: {source}"))
                old = by_id.get(str(event.get("old_chunk_id")))
                claims = old.get("superseded_claims") if old else []
                if not any(claim.get("superseded_by_chunk_id") for claim in claims or []):
                    issues.append(_issue("error", "resolution-not-embedded", f"{event_id}: run wiki-embed"))
        elif event_type == "dispute":
            for chunk_id in event.get("chunk_ids") or []:
                if str(chunk_id) not in by_id:
                    issues.append(_issue("error", "event-dangling-chunk", f"{event_id}: {chunk_id}"))
        elif event_type == "error_correction":
            old = by_id.get(str(event.get("old_chunk_id")))
            source = config.root / str(event.get("correction_source") or "")
            if old is None:
                issues.append(_issue("error", "event-dangling-chunk", f"{event_id}: old chunk missing"))
            if not source.is_file():
                issues.append(_issue("error", "correction-source-missing", f"{event_id}: {source}"))
            if old is not None and not old.get("replaced_by"):
                issues.append(_issue("error", "correction-not-embedded", f"{event_id}: run wiki-embed"))

    return issues


def _semantic_comparison_candidates(config: Config) -> list[dict[str, Any]]:
    chunks = apply_events(read_jsonl(config.artifact_dir / "chunks.jsonl"), read_events(config))
    graph_path = config.artifact_dir / "knn_graph.npz"
    if not graph_path.is_file() or not chunks:
        return []
    graph = np.load(graph_path, allow_pickle=False)
    indices = np.asarray(graph["indices"], dtype=np.int64)
    scores = np.asarray(graph["scores"], dtype=np.float32)
    candidates = []
    seen = set()
    for left in range(min(len(chunks), len(indices))):
        if not chunks[left].get("searchable", True):
            continue
        for right, score in zip(indices[left], scores[left], strict=True):
            right = int(right)
            pair = tuple(sorted((left, right)))
            if pair in seen or float(score) < config.review_similarity:
                continue
            seen.add(pair)
            if not chunks[right].get("searchable", True):
                continue
            candidates.append(
                {
                    "type": "semantic_comparison_candidate",
                    "is_contradiction": None,
                    "instruction": "The host LLM must compare scope, time, and claims; this similarity is not a contradiction judgment.",
                    "cosine_similarity": round(float(score), 7),
                    "left": chunks[left],
                    "right": chunks[right],
                }
            )
    candidates.sort(key=lambda row: (-row["cosine_similarity"], row["left"]["id"], row["right"]["id"]))
    return candidates


def review_bundles(config: Config, *, limit: int = 50) -> list[dict[str, Any]]:
    """Return whole-vault semantic comparison candidates for periodic audits."""
    return _semantic_comparison_candidates(config)[:limit]


def _linked_scope_ids(chunk: dict[str, Any]) -> dict[str, str]:
    links: dict[str, str] = {}
    for field in ("previous_chunk_id", "next_chunk_id"):
        target_id = str(chunk.get(field) or "")
        if target_id:
            links[target_id] = "adjacent"
    for claim in chunk.get("superseded_claims") or []:
        target_id = str(claim.get("superseded_by_chunk_id") or "")
        if target_id:
            links[target_id] = "related"
    for claim in chunk.get("supersedes") or []:
        target_id = str(claim.get("chunk_id") or "")
        if target_id:
            links[target_id] = "related"
    for target_id in [*(chunk.get("replaced_by") or []), *(chunk.get("corrects") or [])]:
        if str(target_id):
            links[str(target_id)] = "related"
    for dispute in chunk.get("disputes") or []:
        for target_id in dispute.get("counterpart_chunk_ids") or []:
            if str(target_id):
                links[str(target_id)] = "related"
    return links


def review_query_scope(config: Config, chunk_ids: list[str], *, limit: int | None = None) -> dict[str, Any]:
    """Limit semantic-comparison evidence to a retrieved chunk neighborhood.

    The scope starts with retrieval hits, then adds only their direct document
    neighbors and hygiene-linked chunks. It never runs an embedder or an LLM.
    """
    candidates = _semantic_comparison_candidates(config)
    by_id = {
        str(chunk["id"]): chunk
        for chunk in apply_events(read_jsonl(config.artifact_dir / "chunks.jsonl"), read_events(config))
    }
    reasons: dict[str, set[str]] = {}
    seed_ids = []
    for chunk_id in dict.fromkeys(map(str, chunk_ids)):
        if chunk_id in by_id:
            reasons.setdefault(chunk_id, set()).add("retrieved")
            seed_ids.append(chunk_id)
    for chunk_id in seed_ids:
        for linked_id, reason in _linked_scope_ids(by_id[chunk_id]).items():
            if linked_id in by_id:
                reasons.setdefault(linked_id, set()).add(reason)

    scoped = []
    for candidate in candidates:
        left_id = str(candidate["left"]["id"])
        right_id = str(candidate["right"]["id"])
        matches = []
        for chunk_id in (left_id, right_id):
            if chunk_id in reasons:
                matches.append({"chunk_id": chunk_id, "scope_reasons": sorted(reasons[chunk_id])})
        if not matches:
            continue
        scoped.append(
            {
                **candidate,
                "scope_matches": matches,
                "_direct_retrieval": int(left_id in seed_ids or right_id in seed_ids),
                "_both_in_scope": int(left_id in reasons and right_id in reasons),
            }
        )
    scoped.sort(
        key=lambda row: (
            -row["_direct_retrieval"],
            -row["_both_in_scope"],
            -row["cosine_similarity"],
            row["left"]["id"],
            row["right"]["id"],
        )
    )
    for candidate in scoped:
        candidate.pop("_direct_retrieval")
        candidate.pop("_both_in_scope")
    resolved_limit = config.review_query_limit if limit is None else limit
    return {
        "scope": "query",
        "seed_chunk_ids": seed_ids,
        "context_chunk_ids": sorted(reasons),
        "matched_candidate_count": len(scoped),
        "candidates": scoped[:resolved_limit],
    }


def _query_seed_ids(config: Config, query: str, *, k: int) -> list[str]:
    from .service import read_daemon_state, request

    daemon_response = request(
        config.artifact_dir,
        {"action": "search", "query": query, "k": k},
    )
    if daemon_response is not None:
        if not daemon_response.get("ok"):
            raise RuntimeError(str(daemon_response.get("error") or "daemon search failed"))
        results = daemon_response["result"].get("results") or []
        return [str(item.get("chunk", {}).get("id")) for item in results if item.get("chunk", {}).get("id")]
    if read_daemon_state(config.artifact_dir) is not None:
        raise RuntimeError("wiki-daemon state exists but it is not reachable; run wiki-daemon status or restart it")
    from .search import SearchEngine

    return [str(hit.chunk["id"]) for hit in SearchEngine(config).search(query, k=k).hits]


def report(config: Config) -> dict[str, Any]:
    issues = inspect(config)
    return {
        "ok": not any(issue.severity == "error" for issue in issues),
        "errors": [issue.to_dict() for issue in issues if issue.severity == "error"],
        "warnings": [issue.to_dict() for issue in issues if issue.severity == "warning"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and curate an LLM-Wiki V3 vault")
    parser.add_argument("--vault", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    review = subparsers.add_parser("review", help="emit evidence pairs for the host LLM")
    review.add_argument("--vault", type=Path, default=None)
    review.add_argument("--json", action="store_true")
    review.add_argument("--scope", choices=("global", "query"), default="global")
    review.add_argument("--limit", type=int, default=None)
    review.add_argument("--query", type=str, default=None)
    review.add_argument("--chunk-id", action="append", default=[])
    review.add_argument("--k", type=int, default=8)
    apply_parser = subparsers.add_parser("apply", help="apply a user-approved decision JSON")
    apply_parser.add_argument("decision", type=Path)
    apply_parser.add_argument("--vault", type=Path, default=None)
    apply_parser.add_argument("--json", action="store_true")
    apply_parser.add_argument("--no-rebuild", action="store_true")
    return parser


def main() -> int:
    configure_stdio()
    args = _parser().parse_args()
    try:
        config = load(args.vault)
        if args.command == "review":
            if args.limit is not None and args.limit <= 0:
                raise ValueError("--limit must be positive")
            if args.scope == "global":
                if args.query or args.chunk_id:
                    raise ValueError("--query and --chunk-id require --scope query")
                payload = {"scope": "global", "candidates": review_bundles(config, limit=args.limit or 50)}
            else:
                if args.query and args.chunk_id:
                    raise ValueError("use either --query or --chunk-id with --scope query")
                if args.k <= 0:
                    raise ValueError("--k must be positive")
                if not args.query and not args.chunk_id:
                    raise ValueError("--scope query requires --query or at least one --chunk-id")
                chunk_ids = args.chunk_id or _query_seed_ids(config, args.query, k=args.k)
                payload = review_query_scope(config, chunk_ids, limit=args.limit)
            print(json.dumps(payload, ensure_ascii=False) if args.json else json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        if args.command == "apply":
            decision = json.loads(args.decision.read_text(encoding="utf-8"))
            chunks = apply_events(read_jsonl(config.artifact_dir / "chunks.jsonl"), read_events(config))
            event = apply_decision(config, decision, chunks)
            rebuild = None
            if not args.no_rebuild:
                from .service import read_daemon_state, request

                daemon_response = request(config.artifact_dir, {"action": "embed"}, timeout=3600.0)
                if daemon_response is None:
                    if read_daemon_state(config.artifact_dir) is not None:
                        raise RuntimeError("wiki-daemon state exists but it is not reachable; run wiki-daemon status or restart it")
                    rebuild = build(config)
                elif daemon_response.get("ok"):
                    rebuild = dict(daemon_response["result"]["manifest"])
                else:
                    raise RuntimeError(str(daemon_response.get("error") or "daemon embedding failed"))
            payload = {"applied": event, "rebuild": rebuild}
            print(json.dumps(payload, ensure_ascii=False) if args.json else json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        payload = report(config)
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"wiki-health: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        for issue in payload["errors"]:
            print(f"ERROR [{issue['code']}] {issue['detail']}", file=sys.stderr)
        for issue in payload["warnings"]:
            print(f"WARNING [{issue['code']}] {issue['detail']}")
        print("OK: healthy" if payload["ok"] else "ERROR: unhealthy")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
