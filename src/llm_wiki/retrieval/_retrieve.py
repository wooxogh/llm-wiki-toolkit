"""Shared retrieval for the vault: dense (embeddings) + sparse (BM25) hybrid.

Used by recall.py and eval.py so both rank identically. Local, no API key.

- dense: cosine over Qwen3 embeddings (query via the resident embed_server).
- sparse: BM25 over a *fielded* document — id and tags repeated 3x, summary 2x,
  body 1x — so exact/rare tokens (API names, fields like `rowFilterKey`,
  `orderStatusCode`) and the page's own labels both carry weight.
- fusion: Reciprocal Rank Fusion (scale-free) of the two chunk rankings, then
  aggregate to the best chunk per page.

mode="hybrid" (default) fuses both; mode="dense" reproduces pure cosine.

Filtering happens BEFORE rank enumeration. RRF scores a document by its *rank*,
so a filtered-out chunk that still occupied ranks 0-1 would silently push every
admissible result two places down the fused score. Excluded chunks must not
exist as far as the ranking is concerned.

Test seam: `rank_from_scores()` is pure — it takes score arrays and returns the
ranking. `search()` is the only part that embeds a query or loads the store, so
the whole ranking policy is testable without torch or the embed server.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llm_wiki.paths import content_root, embeddings_dir

# Both are the CONTENT root: `.embeddings/` sits beside the pages, and every
# `meta["path"]` is content-root-relative (see paths.relative).
VAULT = content_root()
EMB = embeddings_dir()
RRF_K = 60  # standard RRF damping
# dense is the stronger semantic signal; weight it above sparse so BM25 helps
# exact-token queries without burying purely-conceptual ones. Tunable via env.
DENSE_W = float(os.environ.get("WIKI_DENSE_W", "2.0"))
SPARSE_W = float(os.environ.get("WIKI_SPARSE_W", "1.0"))

# bigram (default) | kiwi — see tokenize(). Default is bigram so behaviour is
# identical with or without the optional kiwipiepy dependency installed.
TOKENIZER = os.environ.get("WIKI_TOKENIZER", "bigram")
_TOKEN = re.compile(r"[a-z0-9_]+|[가-힣]+")
_HANGUL = re.compile(r"^[가-힣]+$")
NGRAM_N = 2
_cache: dict = {}


_kiwi = None


def _get_kiwi():
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return _kiwi


def tokenize_kiwi(s: str) -> list[str]:
    """Morphological tokens via Kiwi — the linguistically correct alternative to
    character bigrams for Korean.

    Korean glues particles onto stems (verb-stem + particle), which is the whole
    reason the bigram hack exists. A real analyser splits them properly and
    leaves Latin identifiers (`orderStatusCode`, `rowFilterKey`) intact.

    REJECTED alternative, kept for re-measurement — do not switch to this as the
    default without re-running the measurement below.

    MEASURED AND REJECTED (2026-07-31). Switching to this **hurt**:
    pattern Hit@3 60.5% → 50.0%, overall Hit@3 77.1% → 73.9%, hard Hit@3
    79.5% → 74.8% (`WIKI_TOKENIZER=kiwi wiki-gate`).

    Why the "linguistically correct" option loses here: bigrams are imprecise
    but give *fuzzy partial* matching, and this gold set is dominated by
    indirect/conceptual queries (118/188) whose wording differs from the page's.
    Morphology buys precision and pays for it in paraphrase recall.

    Kept as an env-switchable alternative so the rejection stays reproducible —
    re-measure if the corpus shifts toward direct, same-wording lookups.

    Optional dependency. `tokenize()` keeps the bigram default so the system
    behaves identically whether or not kiwipiepy is installed.
    """
    out = [t.form.lower() for t in _get_kiwi().tokenize(s or "")]
    return list(dict.fromkeys(t for t in out if len(t) > 1 or t.isalnum()))


def tokenize(s: str) -> list[str]:
    """Word tokens, plus character bigrams for Hangul words.

    Korean is agglutinative and written without spaces between a stem and its
    particles, so whole-word BM25 matching is brittle: a query for a bare stem
    and a page writing the stem-plus-particle form share no token at all.
    Character bigrams bridge that.

    Latin/API tokens are left exact — bigramming `orderstatuscode` would add a
    dozen meaningless terms ("or", "rd", "de", ...) that collide across
    unrelated pages and dilute precisely the rare-token matching sparse exists
    to provide. Single-character Hangul has no bigram either.
    """
    if TOKENIZER == "kiwi":
        return tokenize_kiwi(s)
    out: list[str] = []
    for tok in _TOKEN.findall((s or "").lower()):
        out.append(tok)
        if len(tok) >= NGRAM_N and _HANGUL.match(tok):
            out.extend(tok[i:i + NGRAM_N] for i in range(len(tok) - NGRAM_N + 1))
    return list(dict.fromkeys(out))  # dedupe, order-preserving


def fielded_sparse_text(meta: dict) -> str:
    """Build the BM25 document for a chunk without mutating its body text.

    Repetition is the standard key-free way to express field weights in a
    single-field BM25 index: a term appearing in the id or a tag gets 3x the
    term frequency of the same term in the body.
    """
    high = " ".join([str(meta.get("id") or ""), *[str(t) for t in (meta.get("tags") or [])]])
    medium = str(meta.get("summary") or "")
    body = str(meta.get("text") or "")
    return " ".join([high] * 3 + [medium] * 2 + [body])


@dataclass(frozen=True)
class SearchFilters:
    """Pre-rank admissibility. `status="active"` is the default on purpose:
    a superseded page is knowledge we deliberately retired, and serving it
    unasked is worse than serving nothing. `status=None` opts back in."""

    layer: str | None = None
    domain: str | None = None
    project: str | None = None
    confidence: str | None = None
    status: str | None = "active"

    def keep(self, meta: dict) -> bool:
        if self.layer and meta.get("layer") != self.layer:
            return False
        if self.domain and meta.get("domain") != self.domain:
            return False
        if self.confidence and meta.get("confidence") != self.confidence:
            return False
        if self.status and (meta.get("status") or "active") != self.status:
            return False
        if self.project and self.project not in (meta.get("projects") or []):
            return False
        return True


def rank_from_scores(meta: list, dense, sparse, filters: SearchFilters,
                     k: int = 8, mode: str = "hybrid") -> list:
    """Pure ranking: (chunk metadata, dense scores, sparse scores) -> top-k pages.

    Returns [(score, chunk_meta)] with one entry per page — the page's best
    chunk. Aggregating by best chunk (rather than summing) keeps a long page
    from outranking a precise short one just by having more chunks.
    """
    dense = np.asarray(dense, dtype=np.float64)
    candidates = [i for i, m in enumerate(meta) if filters.keep(m)]
    if not candidates:
        return []
    idx = np.asarray(candidates)

    if mode == "dense":
        best: dict = {}
        for i in idx:
            sc, m = float(dense[i]), meta[i]
            if m["id"] not in best or sc > best[m["id"]][0]:
                best[m["id"]] = (sc, m)
        return sorted(best.values(), key=lambda x: (-x[0], x[1]["id"]))[:k]

    sparse = np.asarray(sparse, dtype=np.float64)
    fused = np.zeros(len(meta))
    # np.argsort is stable ('stable' kind) so equal scores keep meta order,
    # making the whole ranking reproducible run to run.
    for rank, i in enumerate(idx[np.argsort(-dense[idx], kind="stable")]):
        fused[i] += DENSE_W / (RRF_K + rank)
    for rank, i in enumerate(idx[np.argsort(-sparse[idx], kind="stable")]):
        fused[i] += SPARSE_W / (RRF_K + rank)

    best = {}
    for i in idx[np.argsort(-fused[idx], kind="stable")]:
        m = meta[i]
        if m["id"] not in best:
            best[m["id"]] = (float(fused[i]), m)
    return sorted(best.values(), key=lambda x: (-x[0], x[1]["id"]))[:k]


def page_confidence(meta: list, dense, page_ids) -> dict:
    """Absolute semantic closeness per page = its best chunk's cosine.

    Ranking and *confidence* are different questions and need different signals.
    RRF answers "which is most relevant relative to the others" and is
    deliberately scale-free — its score depends on rank position, so the top hit
    of a nonsense query lands in the same narrow band as the top hit of a perfect
    one (measured 2026-07-31: negatives 0.0363-0.0489, positives 0.0425-0.0500 —
    overlapping, so no threshold separates them). Cosine answers "how close is
    this actually", on a fixed [-1, 1] scale that does separate them
    (negatives 0.3617-0.4786 vs positives 0.4885-0.8846).

    So: rank with RRF, decide automatic consumption with this.
    """
    dense = np.asarray(dense, dtype=np.float64)
    wanted = set(page_ids)
    best: dict = {}
    for i, m in enumerate(meta):
        _id = m["id"]
        if _id in wanted:
            sc = float(dense[i])
            if _id not in best or sc > best[_id]:
                best[_id] = sc
    return best


def _load():
    if not _cache:
        if not (EMB / "vectors.npy").exists() or not (EMB / "meta.json").exists():
            # VAULT is resolved from the working directory (via llm_wiki.paths), not
            # pinned to this file's location, so a missing store usually means "run
            # this from somewhere else" or "run this before wiki-embed" rather than
            # a broken install -- name the resolved path so either is diagnosable.
            raise RuntimeError(
                f"No embedding store found at {EMB} (resolved vault: {VAULT}). "
                "Run `wiki-embed` to build one, or set WIKI_VAULT to point at a "
                "vault that already has one."
            )
        from rank_bm25 import BM25Okapi
        vecs = np.load(EMB / "vectors.npy")
        meta = json.loads((EMB / "meta.json").read_text(encoding="utf-8"))
        if not meta or len(vecs) == 0:
            raise RuntimeError(
                f"Embedding store at {EMB} is empty (resolved vault: {VAULT}). "
                "Run `wiki-index` first so the vault has index entries, then run "
                "`wiki-embed` to rebuild .embeddings/."
            )
        bm25 = BM25Okapi([tokenize(fielded_sparse_text(m)) for m in meta])
        _cache.update(vecs=vecs, meta=meta, bm25=bm25)
    return _cache


def apply_rerank(ranked: list, query: str, vault: Path = VAULT, rerank_fn=None) -> tuple:
    """Reorder candidates with the local cross-encoder. Returns (ranked, reranked).

    The boolean is the point: automatic consumption fails closed on
    `reranked is False`. If a model-load failure quietly returned the
    un-reranked list as though it had been reranked, that contract would be
    unenforceable — the caller could not tell "the reranker agreed" from "the
    reranker never ran".

    Only import/model-load errors are absorbed. Corrupt frontmatter, a broken
    index, or a bad embedding store must still raise.
    """
    if not ranked:
        return ranked, False
    if rerank_fn is None:
        try:
            from llm_wiki.retrieval._rerank import rerank_scores as rerank_fn
        except (ImportError, OSError, RuntimeError):
            return ranked, False

    # feed FULL page body (not the thin representative chunk): measured
    # 2026-07-24 that chunk-text rerank REGRESSES (EASY Hit@1 92→73) while
    # full-page rerank lifts HARD Hit@1 64→76 and holds EASY.
    texts = []
    for _, m in ranked:
        try:
            t = (vault / m["path"]).read_text(encoding="utf-8")
            if t.startswith("---"):
                end = t.find("\n---", 3)
                t = t[end + 4:] if end != -1 else t
            texts.append(t[:2000])
        except OSError:
            texts.append(m.get("text") or m.get("snippet") or "")

    try:
        rs = rerank_fn(query, texts)
    except (ImportError, OSError, RuntimeError):
        return ranked, False

    order = sorted(range(len(ranked)), key=lambda i: -rs[i])
    return [(rs[i], ranked[i][1]) for i in order], True


def search(query: str, k: int = 8, layer: str | None = None,
           domain: str | None = None, mode: str = "hybrid",
           rerank: int = 0, project: str | None = None,
           status: str | None = "active", confidence: str | None = None,
           query_vector=None) -> list:
    """rerank>0 (opt-in): fetch `rerank` candidates from hybrid/dense, then
    reorder them with the local cross-encoder (_rerank) and return top-k.
    rerank=0 (default) leaves the fast key-free path untouched.

    `query_vector` injects a precomputed embedding (used by eval to embed once
    across modes); otherwise the query is embedded here."""
    c = _load()
    vecs, meta, bm25 = c["vecs"], c["meta"], c["bm25"]
    if query_vector is None:
        from llm_wiki.retrieval._embedder import embed_query
        query_vector = embed_query(query)
    dense = vecs @ query_vector  # per-chunk cosine
    sparse = np.asarray(bm25.get_scores(tokenize(query))) if mode != "dense" else None

    filters = SearchFilters(layer=layer, domain=domain, project=project,
                            confidence=confidence, status=status)
    pool = max(k, rerank) if rerank else k
    ranked = rank_from_scores(meta, dense, sparse, filters, k=pool, mode=mode)

    if rerank:
        ranked, _ = apply_rerank(ranked, query)
    return ranked[:k]


@dataclass(frozen=True)
class ConfidentResult:
    """Retrieval output for the automatic-consumption path.

    `hits` are in FINAL ranked order. `reranked` says whether the cross-encoder
    actually ran — the auto policy fails closed when it did not.
    """

    hits: tuple = ()
    reranked: bool = False


def search_with_confidence(query: str, rerank: int = 0, **kwargs) -> ConfidentResult:
    """search(), plus each hit's absolute cosine confidence and a rerank flag.

    Ranking order and confidence are deliberately different signals: order comes
    from RRF and (when enabled) the cross-encoder; confidence is the absolute
    cosine, which is the only one of the three on a calibratable scale.
    """
    from llm_wiki.retrieval._embedder import embed_query

    query_vector = kwargs.pop("query_vector", None)
    if query_vector is None:
        query_vector = embed_query(query)
    k = kwargs.pop("k", 8)
    # Fetch the candidate pool WITHOUT reranking, then rerank here so the caller
    # learns whether it succeeded.
    pool = max(k, rerank) if rerank else k
    ranked = search(query, k=pool, query_vector=query_vector, rerank=0, **kwargs)
    if not ranked:
        return ConfidentResult((), False)

    reranked = False
    if rerank:
        ranked, reranked = apply_rerank(ranked, query)
    ranked = ranked[:k]

    c = _load()
    dense = c["vecs"] @ query_vector
    conf = page_confidence(c["meta"], dense, {m["id"] for _, m in ranked})
    hits = tuple((score, conf.get(m["id"], float("-inf")), m) for score, m in ranked)
    return ConfidentResult(hits, reranked)
