"""Deterministic heading-aware chunking for v2."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from llm_wiki.v2.models import Chunk
from llm_wiki.v2.schemas import DEFAULT_CHUNK_TARGET_CHARS

FRONTMATTER_RE = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class TextBlock:
    text: str
    start: int
    end: int
    heading_path: list[str]


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def body_span(raw: str) -> tuple[str, int]:
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return raw, 0
    return raw[match.end():], match.end()


def _line_spans(text: str, base: int = 0) -> list[tuple[str, int, int]]:
    spans = []
    pos = 0
    for keep in text.splitlines(keepends=True):
        start = base + pos
        end = start + len(keep)
        spans.append((keep, start, end))
        pos += len(keep)
    if text and not text.endswith(("\n", "\r")):
        return spans
    return spans


def _heading_level(line: str) -> tuple[int, str] | None:
    match = HEADING_RE.match(line.strip())
    if not match:
        return None
    return len(match.group(1)), match.group(2).strip()


def _heading_path_before(text: str) -> list[str]:
    path: list[str] = []
    for line in text.splitlines():
        heading = _heading_level(line)
        if not heading:
            continue
        level, title = heading
        path = path[: level - 1]
        while len(path) < level - 1:
            path.append("")
        path.append(title)
    return [p for p in path if p]


def _split_by_heading(block: TextBlock, level: int) -> list[TextBlock]:
    lines = _line_spans(block.text, block.start)
    starts: list[int] = []
    for i, (line, _, _) in enumerate(lines):
        heading = _heading_level(line)
        if heading and heading[0] == level:
            starts.append(i)
    if not starts:
        return [block]
    prefix_text = ""
    if starts[0] != 0:
        prefix_text = "".join(line for line, _, _ in lines[:starts[0]])
        has_prefix_body = any(
            line.strip() and not _heading_level(line) for line in prefix_text.splitlines()
        )
        if has_prefix_body:
            starts.insert(0, 0)

    result = []
    for idx, line_idx in enumerate(starts):
        next_idx = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        start = lines[line_idx][1]
        end = lines[next_idx - 1][2]
        text = block.text[start - block.start:end - block.start].strip()
        if not text:
            continue
        prefix = block.text[: start - block.start] if line_idx != starts[0] else prefix_text
        result.append(TextBlock(text=text, start=start, end=end, heading_path=_heading_path_before(prefix + text)))
    return result


def _split_paragraphs(block: TextBlock, target_chars: int) -> list[TextBlock]:
    parts: list[TextBlock] = []
    for match in re.finditer(r"\S(?:.*?)(?=\r?\n\s*\r?\n|\Z)", block.text, re.DOTALL):
        text = match.group(0).strip()
        if not text:
            continue
        start = block.start + match.start()
        end = block.start + match.end()
        path = block.heading_path or _heading_path_before(text)
        parts.append(TextBlock(text=text, start=start, end=end, heading_path=path))
    return parts or [block]


def split_markdown(raw: str, target_chars: int = DEFAULT_CHUNK_TARGET_CHARS) -> list[TextBlock]:
    """Split Markdown using the v2 deterministic rules from the migration spec."""
    body, offset = body_span(raw)
    root = TextBlock(text=body.strip(), start=offset + len(body) - len(body.lstrip()),
                     end=offset + len(body.rstrip()), heading_path=[])
    if not root.text:
        return []

    h1_blocks = _split_by_heading(root, 1)
    output: list[TextBlock] = []
    for h1 in h1_blocks:
        candidates = [h1]
        if len(h1.text) > target_chars:
            candidates = _split_by_heading(h1, 2)
        for candidate in candidates:
            if len(candidate.text) <= target_chars:
                output.append(candidate)
            else:
                output.extend(_split_paragraphs(candidate, target_chars))
    return output


def chunk_document(
    document_id: str,
    path: Path | str,
    raw: str,
    target_chars: int = DEFAULT_CHUNK_TARGET_CHARS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for ordinal, block in enumerate(split_markdown(raw, target_chars), start=1):
        digest = _hash(block.text)
        chunks.append(Chunk(
            id=f"{document_id}:chunk:{ordinal:04d}:{digest[:10]}",
            document_id=document_id,
            path=Path(path).as_posix(),
            heading_path=block.heading_path,
            ordinal=ordinal,
            text=block.text,
            source_start=block.start,
            source_end=block.end,
            content_hash=digest,
            target_chars=target_chars,
            oversized=len(block.text) > target_chars,
        ))
    return chunks
