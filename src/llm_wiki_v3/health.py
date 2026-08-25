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
        relative = path.resolve().relative_to(config.root).as_posix()
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


def review_bundles(config: Config, *, limit: int = 50) -> list[dict[str, Any]]:
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
    return candidates[:limit]


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
    review.add_argument("--limit", type=int, default=50)
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
            payload = {"candidates": review_bundles(config, limit=args.limit)}
            print(json.dumps(payload, ensure_ascii=False) if args.json else json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        if args.command == "apply":
            decision = json.loads(args.decision.read_text(encoding="utf-8"))
            chunks = apply_events(read_jsonl(config.artifact_dir / "chunks.jsonl"), read_events(config))
            event = apply_decision(config, decision, chunks)
            rebuild = None if args.no_rebuild else build(config)
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
