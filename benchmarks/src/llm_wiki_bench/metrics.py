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


def aggregate(rows: list[dict]) -> dict[str, float | int | dict[str, int] | None]:
    """Average finite numeric fields, retaining undefined fields as ``None``.

    Rows may carry different key sets (a profile is chosen per record, so, for
    example, abstention cases carry no ``fine_*`` keys). Each key is therefore
    averaged only over the rows that actually contributed a finite value for
    it, and that per-key contributing count is reported under ``counts`` so a
    reader never mistakes it for the top-level ``n`` (total rows).
    """
    keys = sorted({key for row in rows for key in row} - {"n", "counts"})
    result: dict[str, float | int | dict[str, int] | None] = {"n": len(rows)}
    counts: dict[str, int] = {}
    for key in keys:
        values = [
            float(value)
            for row in rows
            if _is_finite_number(value := row.get(key))
        ]
        counts[key] = len(values)
        result[key] = sum(values) / len(values) if values else None
    result["counts"] = {key: counts[key] for key in sorted(counts)}
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


def score_abstention(expects_abstention: bool, abstained: bool) -> dict[str, float]:
    """Emit per-case abstention counters.

    Precision and recall cannot be averaged per case, so they are summarized
    once over the run by ``abstention_summary``.
    """
    return {
        "abstention_tp": float(expects_abstention and abstained),
        "abstention_fp": float(not expects_abstention and abstained),
        "abstention_fn": float(expects_abstention and not abstained),
    }


def abstention_summary(rows: list[dict]) -> dict[str, float | None]:
    """Return corpus-level abstention precision, recall, and F1."""
    tp = sum(row.get("abstention_tp", 0.0) for row in rows)
    fp = sum(row.get("abstention_fp", 0.0) for row in rows)
    fn = sum(row.get("abstention_fn", 0.0) for row in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None or precision + recall == 0:
        f1 = None if precision is None or recall is None else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "abstention_f1": f1,
        "abstention_precision": precision,
        "abstention_recall": recall,
    }


def score_distractor_rejection(
    distractor_answers: tuple[str, ...], answer: str | None
) -> dict[str, float]:
    """Score whether a prediction avoided every known distractor answer.

    Reproducing an outdated or counterfactual answer is a distinct failure from
    being merely wrong, so it is measured separately from exact match.
    """
    normalized = _normalize_text(answer or "")
    if not normalized:
        return {"distractor_rejection": 1.0}
    distractors = {_normalize_text(value) for value in distractor_answers}
    return {"distractor_rejection": float(normalized not in distractors)}


def score_multi_slot_answer(
    answer_slots: tuple[tuple[str, ...], ...], answer: str | None
) -> dict[str, float]:
    """Score how many required answer slots appear in a single response."""
    normalized = _normalize_text(answer or "")
    if not answer_slots:
        return {"all_slots_matched": 0.0, "slot_coverage": 0.0}
    matched = sum(
        any(_normalize_text(alias) in normalized for alias in slot) for slot in answer_slots
    )
    coverage = matched / len(answer_slots)
    return {"all_slots_matched": float(matched == len(answer_slots)), "slot_coverage": coverage}


def score_sub_claims(
    expected: tuple[str, ...], predicted: tuple[str, ...]
) -> dict[str, float | None]:
    """Score per-sub-claim labels, undefined when the decomposition disagrees."""
    if not expected or len(expected) != len(predicted):
        return {"sub_claim_accuracy": None, "sub_claim_macro_f1": None}
    accuracy = sum(a == b for a, b in zip(expected, predicted)) / len(expected)
    per_label_f1 = []
    for label in sorted(set(expected) | set(predicted)):
        tp = sum(a == label and b == label for a, b in zip(expected, predicted))
        fp = sum(a != label and b == label for a, b in zip(expected, predicted))
        fn = sum(a == label and b != label for a, b in zip(expected, predicted))
        denominator = 2 * tp + fp + fn
        per_label_f1.append(2 * tp / denominator if denominator else 0.0)
    macro_f1 = sum(per_label_f1) / len(per_label_f1) if per_label_f1 else 0.0
    return {"sub_claim_accuracy": accuracy, "sub_claim_macro_f1": macro_f1}
