"""The shipped example vault must pass the same gates a real vault does.

This is the only test that runs against a committed vault rather than a
synthetic one — it is what keeps the README quickstart honest.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from llm_wiki import build_index, config, wiki_health
from llm_wiki.evaluation import eval_schema

VAULT = Path(__file__).resolve().parents[1] / "examples" / "vault"


def index_entries() -> list:
    return yaml.safe_load((VAULT / "index.yaml").read_text(encoding="utf-8"))["entries"]


def test_example_vault_index_is_current():
    assert build_index.write_index(VAULT, VAULT / "index.yaml") > 0
    # write_index is idempotent; a second run must not change the bytes
    before = (VAULT / "index.yaml").read_bytes()
    build_index.write_index(VAULT, VAULT / "index.yaml")
    assert (VAULT / "index.yaml").read_bytes() == before


def test_example_vault_passes_ci_health():
    issues = [i for i in wiki_health.check_health(VAULT, mode="ci") if i.severity == "error"]
    assert issues == []


def test_example_gold_validates_against_the_example_index():
    cfg = config.load(VAULT)
    cases = eval_schema.load_gold(VAULT / cfg.gold)
    assert eval_schema.validate_gold(cases, index_entries()) == []


def test_example_gold_meets_its_own_declared_minimums():
    cfg = config.load(VAULT)
    cases = eval_schema.load_gold(VAULT / cfg.gold)
    report = eval_schema.coverage_report(cases, index_entries())
    assert eval_schema.coverage_shortfalls(report, cfg.minimums) == []


def test_example_vault_declares_at_least_one_korean_page():
    """Cross-lingual recall is a real feature; the example must exercise it."""
    texts = [p.read_text(encoding="utf-8") for p in VAULT.rglob("*.md")]
    assert any(any("가" <= ch <= "힣" for ch in t) for t in texts)
