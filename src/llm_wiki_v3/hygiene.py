"""Append-only hygiene events and deterministic metadata overlays."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from .config import Config
from .io import append_jsonl, read_jsonl


EVENT_TYPES = {"partial_supersede", "error_correction", "dispute"}


def events_path(config: Config) -> Path:
    return config.artifact_dir / "hygiene" / "events.jsonl"


def read_events(config: Config) -> list[dict[str, Any]]:
    return read_jsonl(events_path(config))


def _base_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    updated = dict(chunk)
    updated.update(
        status="active",
        searchable=True,
        superseded_claims=[],
        supersedes=[],
        disputed=False,
        disputes=[],
        retracted_at=None,
        corrected_at=chunk.get("corrected_at"),
        replaced_by=[],
        corrects=list(chunk.get("corrects") or []),
        correction_event_ids=list(chunk.get("correction_event_ids") or []),
    )
    return updated


def apply_events(chunks: Iterable[dict[str, Any]], events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [_base_metadata(chunk) for chunk in chunks]
    by_id = {row["id"]: row for row in rows}
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row.get("source_path") or ""), []).append(row)

    for event in events:
        event_type = event.get("type")
        if event_type == "partial_supersede":
            old = by_id.get(str(event.get("old_chunk_id")))
            direct_id = str(event.get("superseding_chunk_id") or "")
            successors = [by_id[direct_id]] if direct_id in by_id else []
            if event.get("resolution_source"):
                successors.extend(by_source.get(str(event["resolution_source"]), []))
            successors = list({row["id"]: row for row in successors}.values())
            if old is not None:
                for successor in successors or [None]:
                    old["superseded_claims"].append({
                        "claim_id": event.get("claim_id"),
                        "quote": event.get("quote"),
                        "chunk_text_start": event.get("chunk_text_start"),
                        "chunk_text_end": event.get("chunk_text_end"),
                        "superseded_by_chunk_id": successor["id"] if successor is not None else direct_id or None,
                        "replacement_quote": event.get("replacement_quote"),
                        "decided_at": event.get("decided_at"),
                        "reason": event.get("reason"),
                        "event_id": event.get("event_id"),
                    })
            for new in successors:
                new["supersedes"].append({
                        "chunk_id": event.get("old_chunk_id"),
                        "claim_id": event.get("claim_id"),
                        "event_id": event.get("event_id"),
                })
        elif event_type == "dispute":
            chunk_ids = [str(value) for value in event.get("chunk_ids") or []]
            for chunk_id in chunk_ids:
                row = by_id.get(chunk_id)
                if row is None:
                    continue
                row["status"] = "disputed"
                row["disputed"] = True
                row["disputes"].append(
                    {
                        "event_id": event.get("event_id"),
                        "claim_quote": event.get("claim_quotes", {}).get(chunk_id),
                        "counterpart_chunk_ids": [value for value in chunk_ids if value != chunk_id],
                        "reason": event.get("reason"),
                        "decided_at": event.get("decided_at"),
                    }
                )
        elif event_type == "error_correction":
            old = by_id.get(str(event.get("old_chunk_id")))
            correction_source = str(event.get("correction_source") or "")
            replacements = by_source.get(correction_source, [])
            replacement_ids = [row["id"] for row in replacements]
            if old is not None:
                old["status"] = "retracted"
                old["searchable"] = False
                old["retracted_at"] = event.get("decided_at")
                old["replaced_by"] = replacement_ids
                old["error"] = {
                    "quote": event.get("quote"),
                    "chunk_text_start": event.get("chunk_text_start"),
                    "chunk_text_end": event.get("chunk_text_end"),
                    "reason": event.get("reason"),
                    "decision_event_id": event.get("event_id"),
                }
            for replacement in replacements:
                replacement["kind"] = "correction"
                replacement["corrected_at"] = event.get("decided_at")
                replacement["corrects"] = list(dict.fromkeys([*replacement["corrects"], event.get("old_chunk_id")]))
                replacement["correction_event_ids"] = list(
                    dict.fromkeys([*replacement["correction_event_ids"], event.get("event_id")])
                )
    return rows


def validate_decision(decision: dict[str, Any], chunks: Iterable[dict[str, Any]]) -> None:
    event_type = decision.get("type")
    if event_type not in EVENT_TYPES:
        raise ValueError(f"type must be one of {sorted(EVENT_TYPES)}")
    if decision.get("user_approved") is not True:
        raise ValueError("decision must contain user_approved=true")
    by_id = {str(chunk.get("id")): chunk for chunk in chunks}
    referenced = []
    if event_type in {"partial_supersede", "error_correction"}:
        referenced.append(str(decision.get("old_chunk_id") or ""))
    if event_type == "partial_supersede":
        if decision.get("superseding_chunk_id"):
            referenced.append(str(decision["superseding_chunk_id"]))
        elif not str(decision.get("replacement_text") or "").strip():
            raise ValueError("partial_supersede requires superseding_chunk_id or replacement_text")
    if event_type == "dispute":
        referenced.extend(str(value) for value in decision.get("chunk_ids") or [])
    missing = [chunk_id for chunk_id in referenced if chunk_id not in by_id]
    if missing:
        raise ValueError(f"decision references unknown chunk(s): {', '.join(missing)}")

    old_id = decision.get("old_chunk_id")
    if old_id and decision.get("expected_content_hash"):
        actual = by_id[str(old_id)].get("content_hash")
        if actual != decision["expected_content_hash"]:
            raise ValueError("old chunk content changed after review; create a fresh decision")
    if event_type == "partial_supersede":
        quote = str(decision.get("quote") or "")
        if not quote or quote not in str(by_id[str(old_id)].get("text") or ""):
            raise ValueError("partial_supersede quote must occur in the old chunk")
    if event_type == "error_correction" and not str(decision.get("corrected_text") or "").strip():
        raise ValueError("error_correction requires a self-contained corrected_text")


def _event_id(decision: dict[str, Any], now: str) -> str:
    seed = f"{decision.get('type')}|{decision.get('old_chunk_id')}|{now}|{decision.get('reason')}"
    return "event:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]


def _correction_filename(event_id: str) -> str:
    return "correction-" + re.sub(r"[^a-z0-9]+", "-", event_id.lower()).strip("-") + ".md"


def apply_decision(config: Config, decision: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    validate_decision(decision, chunks)
    now = datetime.now(timezone.utc).astimezone().isoformat()
    event = {key: value for key, value in decision.items() if key != "user_approved"}
    event["event_id"] = str(decision.get("event_id") or _event_id(decision, now))
    event["decided_at"] = str(decision.get("decided_at") or now)
    if event.get("old_chunk_id") and event.get("quote"):
        reviewed = next(chunk for chunk in chunks if chunk["id"] == event["old_chunk_id"])
        start = str(reviewed.get("text") or "").find(str(event["quote"]))
        event.setdefault("chunk_text_start", start)
        event.setdefault("chunk_text_end", start + len(str(event["quote"])))

    if event["type"] in {"error_correction", "partial_supersede"} and (
        event["type"] == "error_correction" or not event.get("superseding_chunk_id")
    ):
        config.correction_dir.mkdir(parents=True, exist_ok=True)
        target = config.correction_dir / _correction_filename(event["event_id"])
        if target.exists():
            raise FileExistsError(f"correction source already exists: {target}")
        old = next(chunk for chunk in chunks if chunk["id"] == event["old_chunk_id"])
        relative_target = target.resolve().relative_to(config.root).as_posix()
        is_error = event["type"] == "error_correction"
        frontmatter = {
            "llm_wiki_v3_kind": "correction" if is_error else "supersede_resolution",
            "correction_event_id": event["event_id"],
            "corrects_chunk_ids": [event["old_chunk_id"]] if is_error else [],
            "supersedes_chunk_ids": [] if is_error else [event["old_chunk_id"]],
            "created_at": event["decided_at"],
            "source_document": old.get("source_path"),
        }
        rendered = "---\n" + yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip() + "\n---\n\n"
        body = event["corrected_text"] if is_error else event["replacement_text"]
        rendered += str(body).strip() + "\n"
        target.write_text(rendered, encoding="utf-8", newline="\n")
        event["correction_source" if is_error else "resolution_source"] = relative_target

    append_jsonl(events_path(config), event)
    return event
