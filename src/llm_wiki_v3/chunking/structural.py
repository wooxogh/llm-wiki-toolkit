"""Markdown-preserving structural parsing and sentence span extraction.

The rules preserve source offsets, heading paths, and frontmatter handling
required by the LLM-Wiki artifact contract.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .models import SentenceSpan, StructuralBlock


HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
TABLE_SEPARATOR = re.compile(r"^\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$")
FENCE_START = re.compile(r"^\s{0,3}(?P<fence>`{3,}|~{3,})")
PROTECTED_FENCE_START = re.compile(r"(?m)^[ \t]{0,3}(?:`{3,}|~{3,})")
PROTECTED_TABLE_START = re.compile(r"(?m)^[ \t]*\|(?:[^|]*\|)+[ \t]*$")
THEMATIC_BREAK = re.compile(r"^\s{0,3}(?P<char>[-*_])(?:\s*(?P=char)){2,}\s*$")
UNORDERED_LIST_ITEM = re.compile(r"^\s{0,3}[-+*]\s+")
ORDERED_LIST_ITEM = re.compile(r"^\s{0,3}\d{1,9}[.)]\s+")
HTML_BLOCK_START = re.compile(r"^\s{0,3}<(?P<tag>[A-Za-z][A-Za-z0-9-]*)(?:\s|>|/>)")
SENTENCE_END = re.compile(r"[.!?。！？](?:[\"'”’)}\]]*)(?=\s+|$)")


@dataclass(frozen=True)
class _RawBlock:
    kind: str
    text: str


def parse_markdown_blocks(markdown: str) -> list[StructuralBlock]:
    """Return source-aligned blocks with the active Markdown heading path."""
    blocks: list[StructuralBlock] = []
    heading_stack: list[str] = []
    search_cursor = 0

    frontmatter = _frontmatter(markdown)
    if frontmatter is not None:
        text, source_end = frontmatter
        blocks.append(StructuralBlock(0, "frontmatter", text, 0, source_end, None, ()))
        search_cursor = source_end

    for raw in _split_raw_blocks(markdown[search_cursor:]):
        source_start, source_end = _find_source_span(markdown, raw.text, search_cursor)
        if source_start < 0:
            raise ValueError(f"Could not align structural block kind={raw.kind!r} with the Markdown source.")
        # Composite blocks are assembled from source lines so their internal blank
        # line count can differ from the source. Once aligned, preserve the source
        # substring itself as the authoritative text and provenance span.
        source_text = markdown[source_start:source_end]
        heading_level = None
        heading_match = next(
            (match for line in source_text.splitlines() if (match := HEADING.match(line.strip()))),
            None,
        )
        if raw.kind == "heading" and heading_match is not None:
            heading_level = len(heading_match.group(1))
            label = heading_match.group(2).strip()
            heading_stack[heading_level - 1 :] = []
            while len(heading_stack) < heading_level - 1:
                heading_stack.append("")
            heading_stack.append(label)
        blocks.append(
            StructuralBlock(
                index=len(blocks),
                kind=raw.kind,
                text=source_text,
                source_start=source_start,
                source_end=source_end,
                heading_level=heading_level,
                heading_path=tuple(item for item in heading_stack if item),
            )
        )
        search_cursor = source_end
    return blocks


def _find_source_span(markdown: str, text: str, search_cursor: int) -> tuple[int, int]:
    """Locate an assembled block while retaining its exact original whitespace.

    ``_split_raw_blocks`` deliberately joins compatible prose/code/table blocks.
    That join uses a canonical blank line, whereas authored Markdown may contain
    zero, one, or several blank lines between the same non-blank source lines.
    The fast exact lookup covers normal blocks. The fallback makes only
    inter-line blank whitespace flexible, then returns the original span.
    """
    exact_start = markdown.find(text, search_cursor)
    if exact_start >= 0:
        return exact_start, exact_start + len(text)
    # Treat a run of blank lines as one flexible separator. This is only the
    # fallback locator; the captured source substring below still keeps every
    # original blank line exactly as authored.
    lines = re.split(r"\n(?:[ \t]*\n)*", text)
    if not lines:
        return -1, -1
    separator = r"\n(?:[ \t]*\n)*"
    pattern = separator.join(re.escape(line) for line in lines)
    match = re.compile(pattern).search(markdown, search_cursor)
    return (match.start(), match.end()) if match is not None else (-1, -1)


def sentence_spans(block: StructuralBlock) -> list[SentenceSpan]:
    """Split one prose block while retaining exact source coordinates."""
    spans: list[SentenceSpan] = []
    # Markdown soft-wrapped lines are still one prose sentence. Scan the full
    # structural block so a line break cannot create an artificial gap.
    local_start = 0
    for match in SENTENCE_END.finditer(block.text):
        _append_sentence_span(spans, block, block.text, 0, local_start, match.end())
        local_start = match.end()
        while local_start < len(block.text) and block.text[local_start].isspace():
            local_start += 1
    _append_sentence_span(spans, block, block.text, 0, local_start, len(block.text))
    return spans


def _append_sentence_span(
    spans: list[SentenceSpan],
    block: StructuralBlock,
    line: str,
    line_offset: int,
    local_start: int,
    local_end: int,
) -> None:
    raw = line[local_start:local_end]
    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw) - len(raw.rstrip())
    start = local_start + leading
    end = local_end - trailing
    if start >= end:
        return
    text = line[start:end]
    cleaned = clean_markdown_text(text)
    if not cleaned:
        return
    spans.append(
        SentenceSpan(
            text=text,
            embedding_text=cleaned,
            source_start=block.source_start + line_offset + start,
            source_end=block.source_start + line_offset + end,
        )
    )


def clean_markdown_text(text: str) -> str:
    """Match the experiment's readable Markdown cleanup for Qwen input."""
    line = text.strip()
    line = re.sub(r"^\s{0,3}>\s?", "", line)
    line = re.sub(r"^\s*(?:[-*+] |\d+[.)] )", "", line)
    line = re.sub(r"^\[[ xX]\]\s+", "", line)
    line = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    line = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", line)
    line = re.sub(r"`([^`]*)`", r"\1", line)
    line = re.sub(r"[*_~]{1,3}([^*_~]+)[*_~]{1,3}", r"\1", line)
    line = re.sub(r"<[^>]+>", " ", line)
    return " ".join(line.split()).strip()


def normalize_prose_text(text: str) -> str:
    """Join Markdown soft wraps while retaining paragraph breaks for retrieval text."""
    paragraphs = re.split(r"\r?\n[ \t]*\r?\n", text)
    return "\n\n".join(
        re.sub(r"[ \t]*\r?\n[ \t]*", " ", paragraph).strip()
        for paragraph in paragraphs
    )


def normalize_chunk_text(text: str, kind: str) -> str:
    """Normalize prose portions while preserving protected Markdown blocks exactly."""
    if not kind.startswith("paragraph"):
        return text
    if kind == "paragraph":
        return normalize_prose_text(text)

    protected_starts = [
        match.start()
        for pattern in (PROTECTED_FENCE_START, PROTECTED_TABLE_START)
        if (match := pattern.search(text)) is not None
    ]
    if not protected_starts:
        return normalize_prose_text(text)

    protected_start = min(protected_starts)
    prose = normalize_prose_text(text[:protected_start].rstrip())
    protected = text[protected_start:]
    return f"{prose}\n\n{protected}" if prose else protected


def _frontmatter(markdown: str) -> tuple[str, int] | None:
    lines = markdown.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None
    offset = len(lines[0])
    for line in lines[1:]:
        content_end = offset + len(line.rstrip("\r\n"))
        if line.strip() == "---":
            return markdown[:content_end], content_end
        offset += len(line)
    return None


def _split_raw_blocks(markdown: str) -> list[_RawBlock]:
    lines = markdown.splitlines()
    chunks: list[_RawBlock] = []
    paragraph_lines: list[str] = []
    pending_prefix_lines: list[str] = []

    def append_chunk(kind: str, block_lines: Sequence[str], merge_with_previous: bool = False) -> None:
        nonlocal pending_prefix_lines
        merged_lines = _strip_outer_blank_lines((*pending_prefix_lines, *block_lines))
        pending_prefix_lines = []
        if not merged_lines:
            return
        text = "\n".join(merged_lines)
        if merge_with_previous and chunks and chunks[-1].kind != "heading":
            previous = chunks[-1]
            chunks[-1] = _RawBlock(_merge_chunk_kind(previous.kind, kind), f"{previous.text}\n\n{text}")
        else:
            chunks.append(_RawBlock(kind, text))

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            append_chunk("paragraph", paragraph_lines)
            paragraph_lines = []

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            if pending_prefix_lines:
                pending_prefix_lines.append(line)
            index += 1
            continue
        if THEMATIC_BREAK.match(stripped):
            flush_paragraph()
            pending_prefix_lines.append(line)
            index += 1
            continue
        if HEADING.match(stripped):
            flush_paragraph()
            append_chunk("heading", (line,))
            index += 1
            continue
        fence_match = FENCE_START.match(line)
        if fence_match:
            flush_paragraph()
            fence_lines, index = _collect_fenced_code(lines, index, fence_match)
            append_chunk("code_fence", fence_lines, merge_with_previous=True)
            continue
        if _is_table_row(stripped):
            flush_paragraph()
            table_lines, index = _collect_table(lines, index)
            append_chunk("table", table_lines, merge_with_previous=True)
            continue
        html_match = HTML_BLOCK_START.match(line)
        if html_match:
            flush_paragraph()
            html_lines, index = _collect_html_block(lines, index, html_match)
            append_chunk("html_block", html_lines)
            continue
        if _is_list_item(line):
            flush_paragraph()
            list_lines, index = _collect_list(lines, index)
            append_chunk("list", list_lines)
            continue
        paragraph_lines.append(line)
        index += 1

    flush_paragraph()
    if pending_prefix_lines:
        append_chunk("thematic_break", ())
    return chunks


def _merge_chunk_kind(previous_kind: str, appended_kind: str) -> str:
    if appended_kind in previous_kind.split("+"):
        return previous_kind
    return f"{previous_kind}+{appended_kind}"


def _strip_outer_blank_lines(lines: Sequence[str]) -> tuple[str, ...]:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return tuple(lines[start:end])


def _collect_fenced_code(lines: Sequence[str], start: int, match: re.Match[str]) -> tuple[tuple[str, ...], int]:
    fence = match.group("fence")
    closing = re.compile(rf"^\s{{0,3}}{re.escape(fence[0])}{{{len(fence)},}}\s*$")
    block_lines = [lines[start]]
    index = start + 1
    while index < len(lines):
        block_lines.append(lines[index])
        if closing.match(lines[index]):
            index += 1
            break
        index += 1
    return tuple(block_lines), index


def _collect_table(lines: Sequence[str], start: int) -> tuple[tuple[str, ...], int]:
    table_lines: list[str] = []
    index = start
    while index < len(lines) and _is_table_row(lines[index].strip()):
        table_lines.append(lines[index])
        index += 1
    return tuple(table_lines), index


def _collect_html_block(lines: Sequence[str], start: int, match: re.Match[str]) -> tuple[tuple[str, ...], int]:
    tag = match.group("tag").lower()
    closing = re.compile(rf"</{re.escape(tag)}\s*>", re.IGNORECASE)
    block_lines = [lines[start]]
    index = start + 1
    if closing.search(lines[start]) or lines[start].rstrip().endswith("/>"):
        return tuple(block_lines), index
    while index < len(lines):
        block_lines.append(lines[index])
        if closing.search(lines[index]):
            index += 1
            break
        if not lines[index].strip():
            index += 1
            break
        index += 1
    return tuple(block_lines), index


def _collect_list(lines: Sequence[str], start: int) -> tuple[tuple[str, ...], int]:
    list_lines: list[str] = []
    index = start
    while index < len(lines):
        line = lines[index]
        if _is_list_item(line) or _is_list_continuation(line):
            list_lines.append(line)
            index += 1
        else:
            break
    return tuple(list_lines), index


def _is_list_item(line: str) -> bool:
    return bool(UNORDERED_LIST_ITEM.match(line) or ORDERED_LIST_ITEM.match(line))


def _is_list_continuation(line: str) -> bool:
    return bool(line.strip() and (line.startswith("  ") or line.startswith("\t")))


def _is_table_row(stripped: str) -> bool:
    if TABLE_SEPARATOR.match(stripped):
        return True
    if stripped.count("|") < 2:
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    return len(cells) >= 2 and any(cells)
