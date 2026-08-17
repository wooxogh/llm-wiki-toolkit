"""Lightweight concept embedding/index store.

The vector backend is deterministic and local by default so v2 health and query
tests do not require a model download. A production adapter can replace this
module behind the same artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np

from llm_wiki import config
from llm_wiki.v2 import artifacts
from llm_wiki.v2.concept_store import read_concepts
from llm_wiki.v2.models import Concept

MODEL_ID = "deterministic-hash-embedding-v1"
VECTOR_DIM = 64
RRF_K = 60
DENSE_WEIGHT = 2.0
SPARSE_WEIGHT = 1.0


def tokens(text: str) -> list[str]:
    return re.findall(r"[\w가-힣]+", text.lower())


def vectorize(text: str, dim: int = VECTOR_DIM) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    for token in tokens(text):
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = -1.0 if digest[4] % 2 else 1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def _settings(vault: Path | None = None) -> tuple[str, str]:
    cfg = config.load(vault)
    backend = os.environ.get("WIKI_V2_EMBED_BACKEND", cfg.v2_embed_backend).lower()
    device = os.environ.get("WIKI_EMBED_DEVICE", cfg.v2_embed_device).lower()
    return backend, device


def _semantic_available(vault: Path | None = None) -> bool:
    # Loading a 0.6B model during a normal test/metadata build is surprising and
    # makes the local-first core depend on a model download. Semantic Qwen mode
    # is an explicit deployment choice in wiki.toml or WIKI_V2_EMBED_BACKEND.
    if _settings(vault)[0] not in {"qwen", "semantic"}:
        return False
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _model_id(vault: Path | None = None) -> str:
    return "Qwen/Qwen3-Embedding-0.6B" if _semantic_available(vault) else MODEL_ID


def _embed_passages(texts: list[str], vault: Path | None = None,
                    show_progress: bool = False) -> np.ndarray:
    if _semantic_available(vault):
        from llm_wiki.retrieval._embedder import embed_passages
        return embed_passages(texts, device=_settings(vault)[1], show_progress=show_progress)
    return np.vstack([vectorize(text) for text in texts]) if texts else np.zeros((0, VECTOR_DIM), dtype=np.float32)


def _embed_query(text: str, model: str, vault: Path | None = None) -> np.ndarray:
    if model == "Qwen/Qwen3-Embedding-0.6B":
        from llm_wiki.retrieval._embedder import embed_query
        return embed_query(text, device=_settings(vault)[1])
    return vectorize(text)


def build_index(vault: Path | None = None, show_progress: bool = False) -> int:
    root = artifacts.ensure_layout(vault) / "concept_embeddings"
    concepts = read_concepts(vault)
    matrix = _embed_passages([_index_text(c) for c in concepts], vault, show_progress)
    np.save(root / "vectors.npy", matrix.astype(np.float32))
    meta = [{
        "concept_id": c.id,
        "chunk_id": c.chunk_id,
        "document_id": c.document_id,
        "chunk_hash": c.chunk_hash,
        "text_hash": hashlib.sha1(_index_text(c).encode("utf-8")).hexdigest(),
    } for c in concepts]
    (root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "model.txt").write_text(_model_id(vault), encoding="utf-8")
    return len(concepts)


def search(vault: Path | None, query: str, k: int = 8,
           concepts: list[Concept] | None = None) -> list[tuple[float, Concept]]:
    return [(row["score"], row["concept"]) for row in search_with_signals(vault, query, k, concepts)]


def search_with_signals(vault: Path | None, query: str, k: int = 8,
                        concepts: list[Concept] | None = None) -> list[dict]:
    concepts = concepts if concepts is not None else read_concepts(vault)
    if not concepts:
        return []
    root = artifacts.artifact_path("concept_embeddings", vault)
    model = (root / "model.txt").read_text(encoding="utf-8").strip() if (root / "model.txt").exists() else MODEL_ID
    qvec = _embed_query(query, model, vault)
    sparse = _sparse_scores(query, concepts)
    dense = _dense_scores(root, qvec, concepts)
    dense_rank = _ranks(dense, include_zero=True)
    sparse_rank = _ranks(sparse, include_zero=False)
    rows = []
    for index, concept in enumerate(concepts):
        fused = 0.0
        if index in dense_rank:
            fused += DENSE_WEIGHT / (RRF_K + dense_rank[index])
        if index in sparse_rank:
            fused += SPARSE_WEIGHT / (RRF_K + sparse_rank[index])
        if fused > 0:
            rows.append({"score": fused, "dense_score": float(dense[index]),
                         "sparse_score": float(sparse[index]), "concept": concept})
    return sorted(rows, key=lambda row: (-row["score"], row["concept"].id))[:k]


def _dense_scores(root: Path, qvec: np.ndarray, concepts: list[Concept]) -> list[float]:
    vectors = root / "vectors.npy"
    meta_path = root / "meta.json"
    if vectors.exists() and meta_path.exists():
        matrix = np.load(vectors, mmap_mode="r")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        by_id = {row.get("concept_id"): index for index, row in enumerate(meta)}
        if matrix.shape[1] == qvec.shape[0]:
            return [float(matrix[by_id[concept.id]] @ qvec) if concept.id in by_id else 0.0
                    for concept in concepts]
    return [float(vectorize(_index_text(concept), qvec.shape[0]) @ qvec) for concept in concepts]


def _ranks(scores: list[float], include_zero: bool) -> dict[int, int]:
    ordered = [index for index, score in sorted(enumerate(scores), key=lambda row: (-row[1], row[0]))
               if include_zero or score > 0]
    return {index: rank for rank, index in enumerate(ordered, start=1)}


def is_stale(vault: Path | None = None) -> str | None:
    root = artifacts.artifact_path("concept_embeddings", vault)
    vectors, meta_path, model = root / "vectors.npy", root / "meta.json", root / "model.txt"
    if not vectors.exists() or not meta_path.exists() or not model.exists():
        return "concept index missing"
    if model.read_text(encoding="utf-8").strip() != _model_id(vault):
        return "concept index model identity stale"
    concepts = read_concepts(vault)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if len(meta) != len(concepts):
        return "concept index row count stale"
    rows = int(np.load(vectors, mmap_mode="r").shape[0])
    if rows != len(concepts):
        return "concept vector row count stale"
    expected_ids = [c.id for c in concepts]
    stored_ids = [m.get("concept_id") for m in meta]
    if expected_ids != stored_ids:
        return "concept index metadata order stale"
    for concept, row in zip(concepts, meta):
        if row.get("chunk_hash") != concept.chunk_hash:
            return f"concept index chunk hash stale for {concept.id}"
        if row.get("text_hash") != hashlib.sha1(_index_text(concept).encode("utf-8")).hexdigest():
            return f"concept index text hash stale for {concept.id}"
    return None


def _index_text(concept: Concept) -> str:
    return " ".join([concept.text, concept.summary, concept.source_quote])


def _sparse_scores(query: str, concepts: list[Concept]) -> list[float]:
    q = set(tokens(query))
    scores = []
    total_docs = max(1, len(concepts))
    k1, b = 1.5, 0.75
    doc_freq: dict[str, int] = {}
    concept_tokens = [tokens(_index_text(c)) for c in concepts]
    average_length = sum(len(row) for row in concept_tokens) / total_docs or 1.0
    for toks in concept_tokens:
        for token in set(toks):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    for toks in concept_tokens:
        counts = {token: toks.count(token) for token in set(toks)}
        score = 0.0
        for token in q:
            if token not in counts:
                continue
            idf = math.log(1 + (total_docs - doc_freq.get(token, 0) + 0.5) / (doc_freq.get(token, 0) + 0.5))
            tf = counts[token]
            norm = tf + k1 * (1 - b + b * len(toks) / average_length)
            score += idf * (tf * (k1 + 1)) / norm
        scores.append(score)
    return scores
