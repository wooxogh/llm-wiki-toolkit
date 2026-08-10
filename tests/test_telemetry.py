"""Opt-in event persistence, privacy boundary, and proposal extraction."""
from __future__ import annotations

import json

import pytest

from llm_wiki.telemetry import RecallEvent, append_event, enabled, propose_cases, read_events


def event(**over):
    base = dict(query="how does hybrid ranking break a tie",
                filters={"project": "project-b"},
                result_ids=("hybrid-ranking-tradeoffs",),
                latency_ms=24.1, decision="answer", useful=None)
    base.update(over)
    return RecallEvent(**base)


# --------------------------------------------------------------------------
# persistence + privacy
# --------------------------------------------------------------------------


def test_event_contains_query_metadata_but_not_page_body(tmp_path):
    ev = RecallEvent(
        query="how does hybrid ranking break a tie",
        filters={"project": "project-b"},
        result_ids=("hybrid-ranking-tradeoffs",),
        latency_ms=24.1,
        decision="answer",
        useful=True,
    )
    path = tmp_path / "recall.jsonl"
    append_event(path, ev)
    saved = json.loads(path.read_text().strip())

    assert saved["query"] == "how does hybrid ranking break a tie"
    assert "body" not in saved
    assert "snippet" not in saved
    assert "text" not in saved


def test_events_append_one_json_object_per_line(tmp_path):
    path = tmp_path / "recall.jsonl"

    append_event(path, event(query="first query"))
    append_event(path, event(query="second query"))

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert [json.loads(x)["query"] for x in lines] == ["first query", "second query"]


def test_append_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "recall.jsonl"

    append_event(path, event())

    assert path.exists()


def test_result_ids_round_trip_as_a_list(tmp_path):
    path = tmp_path / "recall.jsonl"

    append_event(path, event(result_ids=("a", "b")))

    assert json.loads(path.read_text())["result_ids"] == ["a", "b"]


# --------------------------------------------------------------------------
# opt-in gate
# --------------------------------------------------------------------------


def test_telemetry_is_off_by_default(monkeypatch):
    monkeypatch.delenv("WIKI_RECALL_TELEMETRY", raising=False)

    assert enabled(flag=False) is False


def test_the_cli_flag_enables_telemetry(monkeypatch):
    monkeypatch.delenv("WIKI_RECALL_TELEMETRY", raising=False)

    assert enabled(flag=True) is True


def test_the_environment_variable_enables_telemetry(monkeypatch):
    monkeypatch.setenv("WIKI_RECALL_TELEMETRY", "1")

    assert enabled(flag=False) is True


@pytest.mark.parametrize("value", ["0", "", "false", "no"])
def test_only_an_explicit_1_enables_via_environment(monkeypatch, value):
    monkeypatch.setenv("WIKI_RECALL_TELEMETRY", value)

    assert enabled(flag=False) is False


def test_disabled_telemetry_performs_no_write(tmp_path, monkeypatch):
    monkeypatch.delenv("WIKI_RECALL_TELEMETRY", raising=False)
    path = tmp_path / "recall.jsonl"

    if enabled(flag=False):  # pragma: no cover - guard mirrors the caller
        append_event(path, event())

    assert not path.exists()


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def test_malformed_lines_are_reported_not_silently_dropped(tmp_path):
    path = tmp_path / "recall.jsonl"
    append_event(path, event(query="well-formed"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")

    events, errors = read_events(path)

    assert [e["query"] for e in events] == ["well-formed"]
    assert len(errors) == 1
    assert "line 2" in errors[0]


def test_reading_an_absent_file_is_empty_not_an_error(tmp_path):
    assert read_events(tmp_path / "absent.jsonl") == ([], [])


# --------------------------------------------------------------------------
# gold proposals
# --------------------------------------------------------------------------


def test_proposals_include_only_explicitly_labeled_events(tmp_path):
    path = tmp_path / "recall.jsonl"
    append_event(path, event(query="no label", useful=None))
    append_event(path, event(query="labeled useful", useful="useful", result_ids=("page-a",)))

    proposals = propose_cases(path)

    assert [p["q"] for p in proposals] == ["labeled useful"]


def test_a_useful_label_proposes_its_top_result_as_the_expected_id(tmp_path):
    path = tmp_path / "recall.jsonl"
    append_event(path, event(query="a question", useful="useful", result_ids=("page-a", "page-b")))

    proposal = propose_cases(path)[0]

    assert proposal["expect"] == ["page-a"]
    assert proposal["expect_none"] is False
    assert proposal["category"] == "direct"


def test_a_none_correct_label_proposes_a_negative_case(tmp_path):
    path = tmp_path / "recall.jsonl"
    append_event(path, event(query="a question outside the wiki", useful="none-correct", result_ids=()))

    proposal = propose_cases(path)[0]

    assert proposal["expect"] == []
    assert proposal["expect_none"] is True
    assert proposal["category"] == "negative"


def test_wrong_and_not_useful_labels_do_not_propose_an_expected_id(tmp_path):
    path = tmp_path / "recall.jsonl"
    append_event(path, event(query="wrong", useful="wrong", result_ids=("bad-page",)))
    append_event(path, event(query="not useful", useful="not-useful", result_ids=("meh",)))

    proposals = propose_cases(path)

    assert [p["q"] for p in proposals] == ["wrong", "not useful"]
    assert all(p["expect"] == [] and p["needs_human_label"] for p in proposals)


def test_proposals_deduplicate_repeated_queries(tmp_path):
    path = tmp_path / "recall.jsonl"
    append_event(path, event(query="same question", useful="useful", result_ids=("page-a",)))
    append_event(path, event(query="same question", useful="useful", result_ids=("page-a",)))

    assert len(propose_cases(path)) == 1


def test_proposing_never_writes_to_the_gold_corpus(tmp_path):
    path = tmp_path / "recall.jsonl"
    gold = tmp_path / "eval_gold.json"
    gold.write_text("[]", encoding="utf-8")
    append_event(path, event(query="a question", useful="useful", result_ids=("page-a",)))

    propose_cases(path)

    assert gold.read_text(encoding="utf-8") == "[]"
