#!/usr/bin/env python3
"""Build the semantic-recall vector store for the vault.

Local, offline, NO API key: sentence-transformers running Qwen3-Embedding-0.6B
(1024-dim, multilingual) — see _embedder.py. Chunks every page body and stores
one vector per chunk in .embeddings/ so recall can do cosine search.

INCREMENTAL by default: only re-embeds pages whose content changed since the
last run (tracked by a per-page hash in .embeddings/pages.json), reuses cached
vectors for unchanged pages, and drops vectors for deleted pages.

FULL rebuild when: --full is passed, OR today is the weekly-full weekday
(default Sunday), OR the model changed, OR no prior store exists. This lets a
single daily ingest call self-manage: cheap incremental on weekdays, one full
rebuild weekly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
from llm_wiki.paths import VAULT_ROOT, content_paths, embeddings_dir, page_hash, relative
from llm_wiki.retrieval._embedder import MODEL, dimension, embed_passages

VAULT = VAULT_ROOT          # config root: content_paths/relative resolve the redirect
OUT = embeddings_dir()      # content root: the store sits beside the pages
CHUNK_CHARS = 700
CTX_CHARS = 300      # cap on the context blurb prefixed to every chunk, so a long summary cannot inflate chunks
# Bump when the chunk-construction scheme changes (page_chunks). The incremental
# hash tracks RAW page text, so a chunking change alone would NOT re-embed —
# folding this into the stored identity forces a full rebuild instead.
CHUNK_SCHEMA = "ctx-v3"  # v3 = v2 + oversized-paragraph splitting (2026-07-27, MPS OOM fix)
# The *record* shape in meta.json is a second, independent axis: chunk text can
# be unchanged while retrieval needs new fields on every row. An incremental run
# reuses old rows verbatim, so without this in the identity the store would end
# up half old-schema / half new-schema and filters would behave differently per
# page. v2 = + projects/tags/confidence/status/updated/summary (2026-07-31).
META_SCHEMA = "meta-v2"
WEEKLY_FULL_DOW = 6  # Sunday (Mon=0 .. Sun=6)


def parse(path: Path):
    text = path.read_text(encoding="utf-8")
    fm, body = {}, text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        import yaml
        fm = yaml.safe_load(text[3:end]) or {}
        body = text[end + 4:]
    return fm, body, text


def _split_long(p: str) -> list[str]:
    """A single paragraph can blow past CHUNK_CHARS on its own — split it.

    Measured 2026-07-27: paragraph-only splitting left a 7,603-char chunk (~3,000
    tokens) and 116 chunks over 2,000 chars, which (a) OOM'd MPS via O(seq^2)
    attention and (b) is a poor retrieval unit anyway — one vector averaging many
    topics retrieves worse than several focused ones. Prefer sentence/line breaks
    near the cap; hard-cut only if a single sentence exceeds it.
    """
    out, rest = [], p
    while len(rest) > CHUNK_CHARS:
        window = rest[:CHUNK_CHARS]
        cut = max(window.rfind(". "), window.rfind("\n"), window.rfind("다. "), window.rfind("· "))
        if cut < CHUNK_CHARS // 3:      # no usable break -> hard cut at the cap
            cut = CHUNK_CHARS
        out.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        out.append(rest)
    return out


def chunk(body: str) -> list[str]:
    paras = [q for p in re.split(r"\n\s*\n", body) if p.strip() for q in _split_long(p.strip())]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 > CHUNK_CHARS and cur:
            chunks.append(cur); cur = p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks or [""]


def page_chunks(fm: dict, body: str, _id: str) -> list[str]:
    # Contextual Retrieval (Anthropic): situate EVERY chunk in its document, not
    # just the first — else chunks after #0 lose the page-level context. We use
    # the measured `summary:` as the context blurb (key-free, deterministic).
    # NOTE: the context prefix is added to EVERY chunk, so a long summary would
    #   inflate every chunk of that page (measured 2026-07-27: longest summary
    #   2,554 chars, 18/179 pages over 300 chars -> even after splitting
    #   paragraphs at 700 chars, a chunk was left at 3,299 chars). The blurb only
    #   needs to say which document a chunk belongs to, so it is capped.
    summary = " ".join((fm.get("summary") or "").split())
    if len(summary) > CTX_CHARS:
        summary = summary[:CTX_CHARS].rstrip() + "…"
    ctx = f"{_id}. {summary}".strip()
    return [f"{ctx}\n{c}" for c in chunk(body)]


SNIPPET_CHARS = 240


def _iso(value) -> str:
    """YAML turns a bare 2026-07-31 into a date object; meta.json is JSON."""
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()[:10]
    return str(value) if value else ""


def _str_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(x) for x in value]


def chunk_metadata(fm: dict, _id: str, rel: str, chunk_no: int, text: str) -> dict:
    """The per-chunk record stored in .embeddings/meta.json.

    Retrieval reads ONLY this file — it never re-opens the page — so anything a
    filter, a fielded BM25 field, or a policy decision needs must be carried
    here. Previously only layer/domain were, which is why `status: superseded`
    pages could not be excluded and why tags contributed nothing to sparse
    scoring.

    `status` defaults to "active" so an unlabelled page stays reachable: the
    default filter drops superseded pages, and a missing label is not evidence
    of supersession.
    """
    return {
        "id": _id,
        "path": rel,
        "layer": fm.get("layer"),
        "domain": fm.get("domain"),
        "chunk": chunk_no,
        "projects": _str_list(fm.get("projects")),
        "tags": _str_list(fm.get("tags")),
        "confidence": fm.get("confidence") or "",
        "status": fm.get("status") or "active",
        "updated": _iso(fm.get("updated")),
        "summary": " ".join((fm.get("summary") or "").split()),
        "text": text,  # full chunk text for sparse (BM25)
        "snippet": text[:SNIPPET_CHARS].replace("\n", " "),
    }


def collect() -> dict:
    """id -> (frontmatter, body, raw_text, vault-relative path)."""
    pages = {}
    for p in content_paths(VAULT):
        fm, body, raw = parse(p)
        _id = fm.get("id", p.stem)
        pages[_id] = (fm, body, raw, relative(p, VAULT))
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="force full rebuild")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    pages = collect()
    hashes = {i: page_hash(v[2]) for i, v in pages.items()}

    identity = f"{MODEL}|{CHUNK_SCHEMA}|{META_SCHEMA}"
    prior_model = (OUT / "model.txt").read_text().strip() if (OUT / "model.txt").exists() else None
    prior_pages = json.loads((OUT / "pages.json").read_text()) if (OUT / "pages.json").exists() else {}
    have_store = (OUT / "vectors.npy").exists() and (OUT / "meta.json").exists()

    is_weekly_full = dt.date.today().weekday() == WEEKLY_FULL_DOW
    full = args.full or is_weekly_full or prior_model != identity or not have_store

    kept_vecs, kept_meta = None, []
    if full:
        changed = set(pages)
        reason = "forced" if args.full else ("weekly" if is_weekly_full else ("identity-change" if prior_model != identity else "no-store"))
        print(f"[full rebuild: {reason}]")
    else:
        old_vecs = np.load(OUT / "vectors.npy")
        old_meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
        changed = {i for i in pages if prior_pages.get(i) != hashes[i]}
        keep_ids = set(pages) - changed
        keep_rows = [k for k, m in enumerate(old_meta) if m["id"] in keep_ids]
        kept_vecs = old_vecs[keep_rows] if keep_rows else None
        kept_meta = [old_meta[k] for k in keep_rows]
        deleted = set(prior_pages) - set(pages)
        print(f"[incremental] changed/new={len(changed)} unchanged={len(keep_ids)} deleted={len(deleted)}")

    # embed changed/new pages
    new_meta, new_texts = [], []
    for _id in sorted(changed):
        fm, body, _raw, rel = pages[_id]
        body_chunks = chunk(body)  # clean chunk text (for BM25 + snippet)
        for i, text in enumerate(page_chunks(fm, body, _id)):
            new_texts.append(text)
            new_meta.append(chunk_metadata(fm, _id, rel, i, body_chunks[i]))
    new_vecs = embed_passages(new_texts, show_progress=True) if new_texts else None

    # combine
    parts = [v for v in (kept_vecs, new_vecs) if v is not None and len(v)]
    # No vectors at all (empty vault, or every page deleted). The store still has
    # to be the width a *query* will be, or the next `vecs @ query` cannot even be
    # attempted — so ask the embedder instead of hardcoding a number, which is how
    # this file once shipped a 384-wide store for a 1024-dim model.
    vecs = np.vstack(parts) if parts else np.zeros((0, dimension()), dtype=np.float32)
    meta = kept_meta + new_meta

    np.save(OUT / "vectors.npy", vecs)
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    (OUT / "pages.json").write_text(json.dumps(hashes, ensure_ascii=False), encoding="utf-8")
    (OUT / "model.txt").write_text(identity, encoding="utf-8")
    print(f"OK: {len(meta)} chunks / {len(pages)} pages -> .embeddings/ "
          f"(dim={vecs.shape[1]}, embedded {len(new_texts)} new)")


if __name__ == "__main__":
    main()
