"""LongMemEval adapter for the released JSON array.

Source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
Files: longmemeval_s_cleaned.json, longmemeval_m_cleaned.json,
       longmemeval_oracle.json  (MIT)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter

_CONSUMED = {
    "question_id",
    "question_type",
    "question",
    "answer",
    "question_date",
    "haystack_dates",
    "haystack_session_ids",
    "haystack_sessions",
    "answer_session_ids",
}


class LongMemEvalAdapter(BenchmarkAdapter):
    name = "longmemeval"
    profile = "memory_qa"
    container = "json_array"
    evidence_id_origin = "upstream"

    def normalize(
        self, record: dict[str, Any], path: Path, record_number: int, split: str
    ) -> dict[str, Any]:
        question_id = str(self._required(record, "question_id", path, record_number))
        session_ids = self._required(record, "haystack_session_ids", path, record_number)
        sessions = self._required(record, "haystack_sessions", path, record_number)
        if not isinstance(session_ids, list) or not isinstance(sessions, list):
            raise ValueError(
                f"{path}: record {record_number}: haystack_session_ids and haystack_sessions must be lists"
            )
        if len(session_ids) != len(sessions):
            raise ValueError(
                f"{path}: record {record_number}: haystack_session_ids and haystack_sessions differ in length"
            )

        context: list[str] = []
        fine_evidence_ids: list[str] = []
        for session_id, turns in zip(session_ids, sessions):
            if not isinstance(turns, list):
                raise ValueError(f"{path}: record {record_number}: session {session_id} is not a list of turns")
            for turn_index, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    raise ValueError(
                        f"{path}: record {record_number}: session {session_id} turn {turn_index} is not an object"
                    )
                role = self._required(turn, "role", path, record_number)
                content = self._required(turn, "content", path, record_number)
                context.append(f"{role}: {content}")
                if turn.get("has_answer") is True:
                    fine_evidence_ids.append(f"{session_id}:{turn_index}")

        expects_abstention = question_id.endswith("_abs")
        # Use memory_qa_abstention profile for abstention questions with no answering turns.
        # This allows fine_evidence_ids to be empty, which is the correct semantics.
        has_answering_turn = len(fine_evidence_ids) > 0
        profile = None if (has_answering_turn or not expects_abstention) else "memory_qa_abstention"

        metadata = self._metadata(record, _CONSUMED)
        metadata.update(
            {
                "question_type": record.get("question_type"),
                "question_date": record.get("question_date"),
            }
        )
        result = {
            "id": question_id,
            "prompt": self._required(record, "question", path, record_number),
            "context": tuple(context),
            "evidence_ids": tuple(self._required(record, "answer_session_ids", path, record_number)),
            "fine_evidence_ids": tuple(fine_evidence_ids),
            "labels": {"answers": (self._required(record, "answer", path, record_number),)},
            "expects_abstention": expects_abstention,
            "metadata": metadata,
        }
        if profile is not None:
            result["profile"] = profile
        return result
