"""`--auto`'s threshold-source wiring.

`run_auto` must consult the CLI's `--thresholds` override when given one, and
the packaged `THRESHOLDS_PATH` otherwise — and it must pass whatever
`load_thresholds` returns straight into `decide()`. A refactor that quietly
read the module global instead of the resolved path, or that dropped the
override, must turn this suite red rather than pass silently.
"""
from __future__ import annotations

import argparse

from llm_wiki.retrieval import recall, retrieval_policy
from llm_wiki.retrieval._retrieve import ConfidentResult


def make_args(**over):
    base = dict(query="q", k=8, layer=None, domain=None, project=None,
                confidence=None, status="active", mode="hybrid", rerank=0,
                json=False, auto=True, thresholds=None, telemetry=False, label=None)
    base.update(over)
    return argparse.Namespace(**base)


def _stub_search_with_confidence(monkeypatch, hits=()):
    def fake(query, **kwargs):
        return ConfidentResult(hits=hits, reranked=True)

    monkeypatch.setattr("llm_wiki.retrieval._retrieve.search_with_confidence", fake)


def _capture_load_and_decide(monkeypatch, thresholds_to_return):
    """Patch load_thresholds/decide and return a dict that records what each
    call actually received, so the test can assert on wiring rather than on
    decision content."""
    seen = {}

    def fake_load_thresholds(path):
        seen["load_path"] = path
        return thresholds_to_return

    def fake_decide(candidates, thresholds, reranker_available, **kwargs):
        seen["decide_thresholds"] = thresholds
        return retrieval_policy.Decision("review", tuple(candidates), "stub")

    monkeypatch.setattr(retrieval_policy, "load_thresholds", fake_load_thresholds)
    monkeypatch.setattr(retrieval_policy, "decide", fake_decide)
    return seen


def test_thresholds_override_reaches_the_decision(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("WIKI_RECALL_TELEMETRY", raising=False)  # keep telemetry off-by-default
    override_path = tmp_path / "vault_thresholds.json"
    override_path.write_text("{}", encoding="utf-8")  # content is irrelevant; load is stubbed
    custom = retrieval_policy.Thresholds(score=0.5, margin=0.01, none=0.1, rerank=0.3)

    _stub_search_with_confidence(monkeypatch)
    seen = _capture_load_and_decide(monkeypatch, custom)

    recall.run_auto(make_args(thresholds=override_path))

    # The override path -- not the packaged default -- is what got read.
    assert seen["load_path"] == override_path
    assert seen["load_path"] != recall.THRESHOLDS_PATH
    # And decide() received exactly what load_thresholds returned for it.
    assert seen["decide_thresholds"] is custom


def test_no_override_falls_back_to_the_packaged_thresholds_path(monkeypatch, capsys):
    monkeypatch.delenv("WIKI_RECALL_TELEMETRY", raising=False)  # keep telemetry off-by-default
    _stub_search_with_confidence(monkeypatch)
    seen = _capture_load_and_decide(monkeypatch, retrieval_policy.UNCALIBRATED)

    recall.run_auto(make_args(thresholds=None))

    assert seen["load_path"] == recall.THRESHOLDS_PATH
    assert seen["decide_thresholds"] is retrieval_policy.UNCALIBRATED
