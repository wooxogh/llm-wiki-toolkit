"""Text normalization shared by sparse and structural retrieval."""
from __future__ import annotations

import re


TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\uac00-\ud7a3]+")


def tokenize(value: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall((value or "").lower()):
        tokens.append(token)
        if len(token) > 1 and all("\uac00" <= char <= "\ud7a3" for char in token):
            tokens.extend(token[index:index + 2] for index in range(len(token) - 1))
    return list(dict.fromkeys(tokens))


def fielded_text(chunk: dict) -> str:
    heading = " ".join(chunk.get("heading_path") or [])
    document = str(chunk.get("document_id") or "")
    body = str(chunk.get("text") or "")
    return " ".join([document] * 3 + [heading] * 2 + [body])


def embedding_text(chunk: dict) -> str:
    heading = " > ".join(chunk.get("heading_path") or [])
    prefix = f"Document: {chunk.get('document_id', '')}"
    if heading:
        prefix += f"\nHeading: {heading}"
    return f"{prefix}\n{chunk.get('text', '')}".strip()
