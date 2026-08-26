"""Four-channel retrieval over Text, Dense, Tree, and k-NN."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .config import Config, load
from .hygiene import apply_events, read_events
from .io import configure_stdio, read_json, read_jsonl
from .models import SearchHit
from .text import fielded_text, tokenize


CHANNELS = ("text", "dense", "tree", "knn")


class QueryEmbedder(Protocol):
    def encode_query(self, query: str) -> np.ndarray: ...


@dataclass(frozen=True)
class SearchResponse:
    hits: tuple[SearchHit, ...]


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.fromisoformat(text[:10])
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _effective_time(chunk: dict[str, Any]) -> datetime | None:
    correction_times = [
        _parse_time(item.get("decided_at"))
        for item in chunk.get("superseded_claims") or []
        if isinstance(item, dict)
    ]
    correction_times = [value for value in correction_times if value is not None]
    if correction_times:
        return max(correction_times)
    return _parse_time(chunk.get("document_modified_at")) or _parse_time(chunk.get("document_created_at"))


def _within_range(chunk: dict[str, Any], years: int | None, now: datetime) -> bool:
    if years is None:
        return True
    timestamp = _effective_time(chunk)
    if timestamp is None:
        return False
    cutoff = now.astimezone(timezone.utc) - timedelta(days=365.2425 * years)
    return timestamp >= cutoff


def _rank(scores: np.ndarray, admissible: np.ndarray, limit: int, *, positive_only: bool = False) -> list[int]:
    ordered = admissible[np.argsort(-scores[admissible], kind="stable")]
    if positive_only:
        ordered = np.asarray([index for index in ordered if scores[index] > 0], dtype=np.int64)
    return [int(index) for index in ordered[:limit]]


def _rrf(rankings: dict[str, list[int]], weights: dict[str, float], rrf_k: int) -> np.ndarray:
    length = max((max(values, default=-1) for values in rankings.values()), default=-1) + 1
    scores = np.zeros(length, dtype=np.float64)
    for channel, ranking in rankings.items():
        weight = weights[channel]
        for rank, index in enumerate(ranking, 1):
            if index >= len(scores):
                scores = np.pad(scores, (0, index + 1 - len(scores)))
            scores[index] += weight / (rrf_k + rank)
    return scores


class SearchEngine:
    def __init__(
        self,
        config: Config,
        *,
        embedder: QueryEmbedder | None = None,
        now: datetime | None = None,
    ) -> None:
        self.config = config
        self.now = now or datetime.now(timezone.utc)
        chunks_path = config.artifact_dir / "chunks.jsonl"
        vectors_path = config.artifact_dir / "vectors.npy"
        if not chunks_path.is_file() or not vectors_path.is_file():
            raise RuntimeError(f"V3 index is missing at {config.artifact_dir}; run wiki-embed")
        self.chunks = apply_events(read_jsonl(chunks_path), read_events(config))
        self.vectors = np.asarray(np.load(vectors_path, allow_pickle=False), dtype=np.float32)
        if len(self.chunks) != len(self.vectors):
            raise RuntimeError("chunks.jsonl and vectors.npy have different row counts")
        self.by_id = {str(chunk["id"]): chunk for chunk in self.chunks}
        sparse_rows = read_json(config.artifact_dir / "sparse_index" / "documents.json", []) or []
        if (
            len(sparse_rows) == len(self.chunks)
            and [row.get("chunk_id") for row in sparse_rows] == [chunk.get("id") for chunk in self.chunks]
        ):
            sparse_documents = [list(map(str, row.get("tokens") or [])) for row in sparse_rows]
        else:
            sparse_documents = [tokenize(fielded_text(chunk)) for chunk in self.chunks]
        self.bm25 = BM25Okapi(sparse_documents) if sparse_documents else None
        self.tree_nodes = (read_json(config.artifact_dir / "tree.json", {}) or {}).get("nodes", [])
        graph_path = config.artifact_dir / "knn_graph.npz"
        if graph_path.is_file():
            graph = np.load(graph_path, allow_pickle=False)
            self.knn_indices = np.asarray(graph["indices"], dtype=np.int64)
            self.knn_edge_scores = np.asarray(graph["scores"], dtype=np.float32)
        else:
            self.knn_indices = np.empty((len(self.chunks), 0), dtype=np.int64)
            self.knn_edge_scores = np.empty((len(self.chunks), 0), dtype=np.float32)
        if embedder is None:
            from .embedder import QwenEmbedder

            embedder = QwenEmbedder(
                config.model_id,
                device=config.embed_device,
                batch_size=config.embedding_batch_size,
            )
        self.embedder = embedder
        self._query_cache: dict[str, np.ndarray] = {}

    def prepare_queries(self, queries: Iterable[str]) -> None:
        for query in dict.fromkeys(queries):
            if query not in self._query_cache:
                self._query_cache[query] = np.asarray(self.embedder.encode_query(query), dtype=np.float32)

    def _admissible(self, years: int | None) -> np.ndarray:
        if years is not None and years <= 0:
            raise ValueError("--range must be a positive number of years")
        return np.asarray(
            [
                index
                for index, chunk in enumerate(self.chunks)
                if chunk.get("searchable", True)
                and chunk.get("status") != "retracted"
                and _within_range(chunk, years, self.now)
            ],
            dtype=np.int64,
        )

    def _tree_scores(self, query: str, dense: np.ndarray, admissible: np.ndarray) -> np.ndarray:
        scores = np.full(len(self.chunks), -np.inf, dtype=np.float64)
        query_tokens = set(tokenize(query))
        allowed = set(int(index) for index in admissible)
        id_to_index = {str(chunk["id"]): index for index, chunk in enumerate(self.chunks)}
        for node in self.tree_nodes:
            if node.get("kind") == "root":
                continue
            indices = [id_to_index[str(chunk_id)] for chunk_id in node.get("chunk_ids", []) if str(chunk_id) in id_to_index and id_to_index[str(chunk_id)] in allowed]
            if not indices:
                continue
            node_tokens = set(tokenize(" ".join([str(node.get("label") or ""), *map(str, node.get("path") or [])])))
            lexical = len(query_tokens & node_tokens) / max(len(query_tokens | node_tokens), 1)
            # A broad directory can contain unrelated documents. It is useful
            # only when the query names that directory; otherwise its best
            # descendant would give every sibling the same Tree score.
            if node.get("kind") == "directory" and lexical == 0:
                continue
            node_score = 0.65 * max(float(dense[index]) for index in indices) + 0.35 * lexical
            for index in indices:
                scores[index] = max(scores[index], node_score)
        return scores

    def _knn_scores(self, dense: np.ndarray, admissible: np.ndarray) -> np.ndarray:
        scores = np.full(len(self.chunks), -np.inf, dtype=np.float64)
        if not len(admissible):
            return scores
        allowed = set(int(index) for index in admissible)
        seeds = _rank(dense, admissible, min(10, len(admissible)))
        for seed_rank, seed in enumerate(seeds, 1):
            scores[seed] = max(scores[seed], float(dense[seed]) / seed_rank)
            if seed >= len(self.knn_indices):
                continue
            for neighbor, edge_score in zip(self.knn_indices[seed], self.knn_edge_scores[seed], strict=True):
                neighbor = int(neighbor)
                if neighbor not in allowed:
                    continue
                propagated = float(dense[seed]) * max(float(edge_score), 0.0) / seed_rank
                scores[neighbor] = max(scores[neighbor], propagated)
        return scores

    def _related(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        relations = []
        seen = set()
        for claim in chunk.get("superseded_claims") or []:
            target_id = str(claim.get("superseded_by_chunk_id") or "")
            target = self.by_id.get(target_id)
            if target is not None and target_id not in seen:
                relations.append({"relation": "SUPERSEDED_BY", "claim": claim, "chunk": target})
                seen.add(target_id)
        for dispute in chunk.get("disputes") or []:
            for target_id in dispute.get("counterpart_chunk_ids") or []:
                target = self.by_id.get(str(target_id))
                if target is not None and target_id not in seen:
                    relations.append({"relation": "DISPUTED_WITH", "dispute": dispute, "chunk": target})
                    seen.add(target_id)
        return relations

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        years: int | None = None,
        channels: Iterable[str] = CHANNELS,
    ) -> SearchResponse:
        channels = tuple(channels)
        unknown = set(channels) - set(CHANNELS)
        if unknown:
            raise ValueError(f"unknown search channel(s): {', '.join(sorted(unknown))}")
        admissible = self._admissible(years)
        if not len(admissible):
            return SearchResponse(())
        if query not in self._query_cache:
            self._query_cache[query] = np.asarray(self.embedder.encode_query(query), dtype=np.float32)
        query_vector = self._query_cache[query]
        if self.vectors.ndim != 2 or self.vectors.shape[1] != len(query_vector):
            raise RuntimeError("query embedding dimension does not match vectors.npy")
        dense = np.asarray(self.vectors @ query_vector, dtype=np.float64)
        text_scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float64) if self.bm25 else np.zeros(len(self.chunks))
        tree_scores = self._tree_scores(query, dense, admissible)
        knn_scores = self._knn_scores(dense, admissible)
        raw_scores = {"text": text_scores, "dense": dense, "tree": tree_scores, "knn": knn_scores}
        pool = max(k, self.config.candidate_pool)
        rankings = {
            channel: _rank(raw_scores[channel], admissible, pool, positive_only=channel == "text")
            for channel in channels
        }
        weights = {
            "text": self.config.text_weight,
            "dense": self.config.dense_weight,
            "tree": self.config.tree_weight,
            "knn": self.config.knn_weight,
        }
        fused = _rrf(rankings, weights, self.config.rrf_k)
        candidates = sorted(
            set(index for ranking in rankings.values() for index in ranking),
            key=lambda index: (-float(fused[index]), self.chunks[index]["id"]),
        )
        candidates = candidates[:k]
        hits = []
        for index in candidates:
            hits.append(
                SearchHit(
                    chunk=self.chunks[index],
                    score=float(fused[index]),
                    channel_ranks={channel: ranking.index(index) + 1 for channel, ranking in rankings.items() if index in ranking},
                    channel_scores={channel: round(float(raw_scores[channel][index]), 8) for channel in channels},
                )
            )

        for hit in hits:
            hit.related_evidence = self._related(hit.chunk)
        return SearchResponse(tuple(hits))


def _auto_decision(response: SearchResponse, config: Config) -> tuple[str, str]:
    if not response.hits:
        return "none", "no-candidates"
    top_dense = float(response.hits[0].channel_scores.get("dense", -math.inf))
    if top_dense < config.auto_none_cosine:
        return "none", "below-none-threshold"
    if response.hits[0].related_evidence or response.hits[0].chunk.get("disputed"):
        return "review", "related-hygiene-evidence"
    runner_dense = float(response.hits[1].channel_scores.get("dense", -math.inf)) if len(response.hits) > 1 else -math.inf
    if top_dense < config.auto_answer_cosine:
        return "review", "below-answer-threshold"
    if top_dense - runner_dense < config.auto_margin:
        return "review", "top-two-margin"
    return "answer", "clear-margin"


def main() -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Search an LLM-Wiki V3 vault")
    parser.add_argument("query")
    parser.add_argument("--vault", type=Path, default=None)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--range", dest="range_years", type=int, default=None, metavar="YEARS")
    parser.add_argument("--no-daemon", action="store_true", help="search in this process even when wiki-daemon is running")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        config = load(args.vault)
        daemon_payload = None
        if not args.no_daemon:
            from .service import read_daemon_state, request

            daemon_response = request(
                config.artifact_dir,
                {
                    "action": "search",
                    "query": args.query,
                    "k": args.k,
                    "range_years": args.range_years,
                    "auto": args.auto,
                },
            )
            if daemon_response is not None:
                if not daemon_response.get("ok"):
                    raise RuntimeError(str(daemon_response.get("error") or "daemon search failed"))
                daemon_payload = dict(daemon_response["result"])
            elif read_daemon_state(config.artifact_dir) is not None:
                raise RuntimeError("wiki-daemon state exists but it is not reachable; run wiki-daemon status or restart it")
        if daemon_payload is None:
            response = SearchEngine(config).search(
                args.query,
                k=args.k,
                years=args.range_years,
            )
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"wiki-search: {exc}", file=sys.stderr)
        return 1

    if daemon_payload is not None:
        payload = daemon_payload
        response = None
    else:
        payload: dict[str, Any] = {
            "query": args.query,
            "range_years": args.range_years,
            "results": [hit.to_dict() for hit in response.hits],
        }
        if args.auto:
            decision, reason = _auto_decision(response, config)
            payload.update(decision=decision, reason=reason)
    if args.json or args.auto:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        for rank, item in enumerate(payload["results"], 1):
            chunk = item["chunk"]
            flags = []
            if chunk.get("superseded_claims"):
                flags.append("partial-supersede")
            if chunk.get("disputed"):
                flags.append("disputed")
            suffix = f" [{' '.join(flags)}]" if flags else ""
            print(f"{rank}. [{float(item['score']):.6f}] {chunk['id']}{suffix}")
            print(f"   {chunk.get('source_path')} :: {' > '.join(chunk.get('heading_path') or [])}")
            print(f"   {str(chunk.get('text') or '')[:240]}")
            for evidence in item.get("related_evidence") or []:
                related = evidence["chunk"]
                print(f"   -> {evidence['relation']}: {related['id']} :: {str(related.get('text') or '')[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
