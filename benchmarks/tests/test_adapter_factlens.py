from pathlib import Path

import pytest

from llm_wiki_bench.adapters.factlens import FactLensAdapter

HEADER = "ind,claim,sub_claims,labels,aggregated_label\n"
ROW = (
    '0,"A represented Munich, while B represented London.",'
    '"[\'A represented Munich\', \'B represented London\']",'
    '"[\'true\', \'false\']",False\n'
)


def _write(tmp_path, body):
    path = tmp_path / "fact_lens_benchmark.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_python_repr_lists_are_parsed(tmp_path):
    case = FactLensAdapter().load(_write(tmp_path, ROW), split="benchmark").cases[0]
    assert case.profile == "claim_decomposition"
    assert case.labels["sub_claims"] == ("A represented Munich", "B represented London")
    assert case.labels["sub_claim_labels"] == ("true", "false")


def test_the_aggregated_label_is_normalized_to_a_boolean(tmp_path):
    case = FactLensAdapter().load(_write(tmp_path, ROW), split="benchmark").cases[0]
    assert case.labels["aggregated_label"] is False


def test_the_claim_is_the_prompt_and_no_evidence_is_invented(tmp_path):
    case = FactLensAdapter().load(_write(tmp_path, ROW), split="benchmark").cases[0]
    assert case.id == "0"
    assert case.prompt.startswith("A represented Munich")
    assert case.context == ()
    assert case.evidence_ids == ()


def test_sub_claims_and_labels_must_be_the_same_length(tmp_path):
    body = (
        '0,"claim","[\'one\', \'two\']","[\'true\']",True\n'
    )
    with pytest.raises(ValueError, match="record 1: sub_claims and labels differ in length"):
        FactLensAdapter().load(_write(tmp_path, body), split="benchmark")


def test_a_malformed_repr_list_names_the_record(tmp_path):
    body = '0,"claim","not a list","[\'true\']",True\n'
    with pytest.raises(ValueError, match="record 1: sub_claims is not a list of strings"):
        FactLensAdapter().load(_write(tmp_path, body), split="benchmark")


def test_factlens_stays_optional():
    assert FactLensAdapter().required is False


def test_the_committed_fixture_matches_the_released_shape():
    fixture = Path(__file__).parent.parent / "fixtures" / "factlens.csv"
    result = FactLensAdapter().load(fixture, split="benchmark")
    assert result.record_count == 2
    assert all(case.labels["sub_claims"] for case in result.cases)


def test_labels_mixing_strings_and_booleans_are_canonicalized(tmp_path):
    """The real released CSV mixes str and bool labels (502/733 rows do).

    labels is semantically boolean; both 'true'/'false' strings and True/False
    bools must canonicalize to the lowercase strings 'true'/'false'.
    """
    body = (
        '0,"claim","[\'one\', \'two\']","[True, \'false\']",True\n'
    )
    case = FactLensAdapter().load(_write(tmp_path, body), split="benchmark").cases[0]
    assert case.labels["sub_claim_labels"] == ("true", "false")


def test_an_invalid_label_still_names_the_record(tmp_path):
    body = (
        '0,"claim","[\'one\', \'two\']","[True, \'maybe\']",True\n'
    )
    with pytest.raises(ValueError, match="record 1: labels is not a list of true/false values"):
        FactLensAdapter().load(_write(tmp_path, body), split="benchmark")


def test_a_non_string_non_bool_label_still_names_the_record(tmp_path):
    body = (
        '0,"claim","[\'one\', \'two\']","[True, 3]",True\n'
    )
    with pytest.raises(ValueError, match="record 1: labels is not a list of true/false values"):
        FactLensAdapter().load(_write(tmp_path, body), split="benchmark")
