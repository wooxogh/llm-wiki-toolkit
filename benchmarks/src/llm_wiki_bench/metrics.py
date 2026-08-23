"""Pure, dependency-light metric helpers for benchmark runs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from math import isfinite, log2
from numbers import Real
from unicodedata import normalize as unicode_normalize


def score_retrieval(
    expected_ids: tuple[str, ...],
    ranked_ids: tuple[str, ...],
    ks: tuple[int, ...] = (1, 3, 8),
) -> dict[str, float | None]:
    """Score first-hit retrieval plus coverage of an entire evidence set."""
    expected = set(expected_ids)
    scores: dict[str, float | None] = {}
    if not expected:
        for k in ks:
            scores[f"recall@{k}"] = None
        scores.update({"mrr": None, "ndcg@3": None, "evidence_set_coverage": None})
        return scores

    rank = next((index for index, item in enumerate(ranked_ids, start=1) if item in expected), None)
    for k in ks:
        scores[f"recall@{k}"] = float(rank is not None and rank <= k)
    scores["mrr"] = 1.0 / rank if rank is not None else 0.0
    scores["ndcg@3"] = 1.0 / log2(rank + 1) if rank is not None and rank <= 3 else 0.0
    scores["evidence_set_coverage"] = len(expected.intersection(ranked_ids)) / len(expected)
    return scores


def score_answer(expected_answers: tuple[str, ...], answer: str | None) -> dict[str, float]:
    """Score Unicode-normalized exact match and maximum whitespace-token F1."""
    normalized_expected = tuple(_normalize_text(value) for value in expected_answers)
    normalized_answer = _normalize_text(answer or "")
    exact_match = float(bool(normalized_answer) and normalized_answer in normalized_expected)
    token_f1 = max((_token_f1(expected, normalized_answer) for expected in normalized_expected), default=0.0)
    return {"exact_match": exact_match, "token_f1": token_f1}


def score_citations(
    expected_ids: tuple[str, ...], cited_ids: tuple[str, ...]
) -> dict[str, float | None]:
    """Score evidence citations, preserving undefined no-citation precision."""
    expected, cited = set(expected_ids), set(cited_ids)
    correct = len(expected.intersection(cited))
    precision = correct / len(cited) if cited else None
    recall = correct / len(expected) if expected else None
    f1 = 0.0 if recall is not None else None
    if precision is not None and recall is not None and precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    return {"citation_precision": precision, "citation_recall": recall, "citation_f1": f1}


def score_label(expected_label: str | None, label: str | None) -> dict[str, float | None]:
    """Score a single classification label when the task supplies one."""
    if expected_label is None:
        return {"label_accuracy": None}
    return {"label_accuracy": float(label == expected_label)}


def label_confusion_matrix(rows: Iterable[tuple[str | None, str | None]]) -> dict[str, dict[str, int]]:
    """Return a deterministic expected-label by predicted-label count matrix."""
    counts: dict[str, Counter[str]] = {}
    for expected, predicted in rows:
        expected_key = expected if expected is not None else "<missing>"
        predicted_key = predicted if predicted is not None else "<missing>"
        counts.setdefault(expected_key, Counter())[predicted_key] += 1
    return {
        expected: {predicted: count for predicted, count in sorted(predictions.items())}
        for expected, predictions in sorted(counts.items())
    }


def aggregate(rows: list[dict]) -> dict[str, float | int | None]:
    """Average finite numeric fields, retaining undefined fields as ``None``."""
    keys = sorted({key for row in rows for key in row} - {"n"})
    result: dict[str, float | int | None] = {"n": len(rows)}
    for key in keys:
        values = [
            float(value)
            for row in rows
            if _is_finite_number(value := row.get(key))
        ]
        result[key] = sum(values) / len(values) if values else None
    return {key: result[key] for key in sorted(result)}


def _normalize_text(value: str) -> str:
    return " ".join(unicode_normalize("NFKC", value).casefold().split())


def _token_f1(expected: str, answer: str) -> float:
    expected_tokens, answer_tokens = expected.split(), answer.split()
    if not expected_tokens or not answer_tokens:
        return 0.0
    overlap = sum((Counter(expected_tokens) & Counter(answer_tokens)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(answer_tokens), overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def _is_finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)
