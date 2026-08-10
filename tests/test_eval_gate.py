"""Retrieval regression gate — the comparison logic only.

Why this exists: on 2026-07-31 a ranking change (page-level RRF instead of
chunk-level) dropped pattern Hit@8 from 89.5% to 65.8% and entity Hit@3 from
60% to 20%. Nothing caught it except a human re-running the eval by hand. CI
validates schema but never measures ranking, so the next such change merges
silently.

Only the *comparison* lives here. Actually measuring requires embedding
hundreds of queries, which would drag the model into the test suite — so that
half is a separate command (`wiki-gate`), and this file pins the arithmetic
with literal numbers.

One exception: `test_thresholds_override_reaches_the_false_answer_gate` below
checks *wiring*, not measurement — it stubs `retrieval_policy.load_thresholds`
so no query is ever embedded, and asserts only that `--thresholds` reaches the
path `_gated_false_answers` actually reads.
"""
from __future__ import annotations

import json

import pytest

from llm_wiki.evaluation import eval_gate
from llm_wiki.evaluation.eval_gate import compare, load_baseline, tolerance_for


def slice_(n, **hits):
    base = {"n": n, "hit@1": 0.0, "hit@3": 0.0, "hit@8": 0.0}
    base.update(hits)
    return base


# --------------------------------------------------------------------------
# tolerance: small slices need a case-sized allowance, not a flat percentage
# --------------------------------------------------------------------------


def test_tolerance_is_at_least_one_case_for_small_slices():
    # entity has n=5, so one case is 20 percentage points
    assert tolerance_for(n=5, tolerance_pp=3.0) == pytest.approx(0.20)


def test_tolerance_is_the_flat_percentage_for_large_slices():
    # domain has n=132; one case is 0.76pp, so the flat 3pp floor governs
    assert tolerance_for(n=132, tolerance_pp=3.0) == pytest.approx(0.03)


def test_tolerance_of_an_empty_slice_does_not_divide_by_zero():
    assert tolerance_for(n=0, tolerance_pp=3.0) == pytest.approx(0.03)


# --------------------------------------------------------------------------
# regression detection
# --------------------------------------------------------------------------


def test_no_change_is_no_regression():
    base = {"overall": slice_(188, **{"hit@1": 0.596})}
    assert compare(base, base) == []


def test_the_measured_page_level_rrf_disaster_is_caught():
    """The exact numbers from the 2026-07-31 near-miss."""
    baseline = {"layer:pattern": slice_(38, **{"hit@1": 0.263, "hit@8": 0.895}),
                "layer:entity": slice_(5, **{"hit@3": 0.600})}
    current = {"layer:pattern": slice_(38, **{"hit@1": 0.237, "hit@8": 0.658}),
               "layer:entity": slice_(5, **{"hit@3": 0.200})}

    found = compare(baseline, current)
    hit = {(r.slice, r.metric) for r in found}

    assert ("layer:pattern", "hit@8") in hit   # 89.5 -> 65.8 = 9 cases
    assert ("layer:entity", "hit@3") in hit    # 60 -> 20 = 2 cases
    # pattern hit@1 fell only 2.6pp = one case on n=38 -> within tolerance
    assert ("layer:pattern", "hit@1") not in hit


def test_a_single_case_drop_on_a_small_slice_is_tolerated():
    baseline = {"layer:entity": slice_(5, **{"hit@1": 1.00})}
    current = {"layer:entity": slice_(5, **{"hit@1": 0.80})}

    assert compare(baseline, current) == []


def test_a_two_case_drop_on_a_small_slice_is_a_regression():
    baseline = {"layer:entity": slice_(5, **{"hit@1": 1.00})}
    current = {"layer:entity": slice_(5, **{"hit@1": 0.60})}

    assert [r.metric for r in compare(baseline, current)] == ["hit@1"]


def test_an_exact_one_case_drop_survives_baseline_rounding():
    """Baselines are stored rounded to 6 decimals, so a true one-case drop on
    n=3 reads as 0.666667 -> 0.333333 and must not fire against a 1/3 tolerance."""
    baseline = {"layer:raw": slice_(3, **{"hit@1": 0.666667})}
    current = {"layer:raw": slice_(3, **{"hit@1": 0.333333})}

    assert compare(baseline, current) == []


def test_improvement_is_never_a_regression():
    baseline = {"layer:pattern": slice_(38, **{"hit@1": 0.263})}
    current = {"layer:pattern": slice_(38, **{"hit@1": 0.947})}

    assert compare(baseline, current) == []


def test_a_slice_that_disappeared_is_reported():
    baseline = {"layer:raw": slice_(3, **{"hit@1": 0.667})}

    found = compare(baseline, {})

    assert [r.slice for r in found] == ["layer:raw"]
    assert "missing" in found[0].detail


def test_a_new_slice_is_not_a_regression():
    assert compare({}, {"layer:new": slice_(4, **{"hit@1": 0.5})}) == []


def test_a_shrinking_corpus_is_reported_even_at_equal_rates():
    """Losing gold cases hides regressions — the rate can hold while coverage falls."""
    baseline = {"overall": slice_(188, **{"hit@1": 0.60})}
    current = {"overall": slice_(150, **{"hit@1": 0.60})}

    found = compare(baseline, current)

    assert [r.metric for r in found] == ["n"]
    assert "188" in found[0].detail and "150" in found[0].detail


def test_false_answers_must_never_increase():
    """The hard gate: zero false answers on negative/ambiguous cases."""
    baseline = {"auto": {"n": 123, "false_answers_gated": 0}}
    current = {"auto": {"n": 123, "false_answers_gated": 1}}

    found = compare(baseline, current)

    assert [r.metric for r in found] == ["false_answers_gated"]


def test_false_answers_have_no_tolerance_at_all():
    """Not even one. This is the safety property, not a quality metric."""
    baseline = {"auto": {"n": 123, "false_answers_gated": 2}}
    current = {"auto": {"n": 123, "false_answers_gated": 3}}

    assert [r.metric for r in compare(baseline, current)] == ["false_answers_gated"]


# --------------------------------------------------------------------------
# baseline io
# --------------------------------------------------------------------------


def test_baseline_round_trips(tmp_path):
    path = tmp_path / "eval_baseline.json"
    data = {"generated": "2026-07-31", "slices": {"overall": slice_(188, **{"hit@1": 0.596})}}
    path.write_text(json.dumps(data), encoding="utf-8")

    assert load_baseline(path)["overall"]["hit@1"] == pytest.approx(0.596)


def test_a_missing_baseline_is_an_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="--update"):
        load_baseline(tmp_path / "absent.json")


# --------------------------------------------------------------------------
# threshold wiring: --thresholds must reach the false-answer safety gate
# --------------------------------------------------------------------------


def test_thresholds_override_reaches_the_false_answer_gate(monkeypatch, tmp_path):
    """With an empty case list the gate loop never runs, so this checks only
    that the resolved path is the one `_gated_false_answers` hands to
    `load_thresholds` — not any retrieval or decision content."""
    seen = {}

    def fake_load_thresholds(path):
        seen["path"] = path
        return object()

    monkeypatch.setattr("llm_wiki.retrieval.retrieval_policy.load_thresholds", fake_load_thresholds)

    override = tmp_path / "vault_thresholds.json"
    result = eval_gate._gated_false_answers([], k=8, mode="hybrid", rerank=0, thresholds=override)

    assert seen["path"] == override
    assert result == {"n": 0, "false_answers_gated": 0}
