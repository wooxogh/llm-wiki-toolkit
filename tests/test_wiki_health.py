"""Artifact drift detection for llm_wiki.wiki_health, in full and CI modes.

No test here loads torch, the embedding model, or the resident embed server —
every `.embeddings/` store is written literally by the fixture builder.
"""
from __future__ import annotations

import json

import pytest

from llm_wiki import wiki_health
from conftest import (
    current_page_hashes,
    page,
    write_embedding_fixture,
    write_index_and_embedding_fixture,
    write_page,
)


def issue_codes(issues) -> set:
    return {i.code for i in issues}


def errors(issues) -> list:
    return [i for i in issues if i.severity == "error"]


def error_codes(issues) -> set:
    """Fixture vaults have no inbound links, so `orphan-pages` is expected noise
    in nearly every test. Assertions about *drift* use this instead."""
    return {i.code for i in errors(issues)}


# --------------------------------------------------------------------------
# embedding drift (full mode only)
# --------------------------------------------------------------------------


def test_full_health_reports_missing_and_stale_embeddings(vault):
    write_page(vault, "domain/research/one.md", page(id="one"))
    write_page(vault, "domain/research/two.md", page(id="two"))
    write_index_and_embedding_fixture(
        vault,
        page_hashes={"one": "wrong"},
        meta=[{"id": "one"}],
        vector_rows=1,
    )

    issues = wiki_health.check_health(vault, mode="full")

    assert error_codes(issues) == {
        "embedding-page-missing",
        "embedding-page-stale",
    }


def test_full_health_reports_deleted_embedding_ids(vault):
    write_page(vault, "domain/research/one.md", page(id="one"))
    hashes = current_page_hashes(vault)
    write_index_and_embedding_fixture(
        vault,
        page_hashes={**hashes, "removed-page": "abc123"},
        meta=[{"id": "one"}],
        vector_rows=1,
    )

    issues = wiki_health.check_health(vault, mode="full")

    assert error_codes(issues) == {"embedding-page-deleted"}
    assert "removed-page" in errors(issues)[0].detail


def test_full_health_reports_vector_metadata_row_mismatch(vault):
    write_page(vault, "domain/research/one.md", page(id="one"))
    write_index_and_embedding_fixture(
        vault,
        page_hashes=current_page_hashes(vault),
        meta=[{"id": "one"}, {"id": "one"}],
        vector_rows=5,
    )

    issues = wiki_health.check_health(vault, mode="full")

    assert error_codes(issues) == {"embedding-row-mismatch"}
    assert "5" in errors(issues)[0].detail and "2" in errors(issues)[0].detail


def test_full_health_reports_missing_model_identity(vault):
    write_page(vault, "domain/research/one.md", page(id="one"))
    write_index_and_embedding_fixture(
        vault,
        page_hashes=current_page_hashes(vault),
        meta=[{"id": "one"}],
        vector_rows=1,
    )
    (vault / ".embeddings" / "model.txt").unlink()

    issues = wiki_health.check_health(vault, mode="full")

    assert error_codes(issues) == {"embedding-model-missing"}


def test_full_health_reports_an_absent_store_once_not_per_page(vault):
    write_page(vault, "domain/research/one.md", page(id="one"))
    write_page(vault, "domain/research/two.md", page(id="two"))
    from llm_wiki import build_index

    build_index.write_index(vault, vault / "index.yaml")

    issues = wiki_health.check_health(vault, mode="full")

    assert error_codes(issues) == {"embedding-store-missing"}


def test_full_health_reports_a_stale_model_identity(vault):
    """A model or chunk-schema change makes every stored vector meaningless.
    Checking only that model.txt *exists* lets that pass as healthy."""
    write_page(vault, "domain/research/one.md", page(id="one"))
    write_index_and_embedding_fixture(
        vault,
        page_hashes=current_page_hashes(vault),
        meta=[{"id": "one"}],
        vector_rows=1,
        model="old-model|ctx-v1|meta-v1",
    )

    issues = wiki_health.check_health(vault, mode="full")

    assert error_codes(issues) == {"embedding-identity-stale"}
    assert "old-model" in errors(issues)[0].detail


def test_full_health_is_clean_on_a_consistent_vault(vault):
    write_page(vault, "domain/research/one.md", page(id="one"))
    write_index_and_embedding_fixture(
        vault,
        page_hashes=current_page_hashes(vault),
        meta=[{"id": "one"}],
        vector_rows=1,
    )

    issues = wiki_health.check_health(vault, mode="full")

    assert errors(issues) == []


# --------------------------------------------------------------------------
# index drift (both modes)
# --------------------------------------------------------------------------


def test_ci_health_skips_local_embeddings_but_rejects_stale_index(vault):
    write_page(vault, "domain/research/one.md", page(id="one"))
    (vault / "index.yaml").write_text("version: 2\nentries: []\n", encoding="utf-8")

    issues = wiki_health.check_health(vault, mode="ci")

    assert error_codes(issues) == {"index-stale"}


def test_ci_health_ignores_a_completely_absent_embedding_store(vault):
    write_page(vault, "domain/research/one.md", page(id="one"))
    from llm_wiki import build_index

    build_index.write_index(vault, vault / "index.yaml")

    assert errors(wiki_health.check_health(vault, mode="ci")) == []


def test_health_reports_invalid_frontmatter_before_index_comparison(vault):
    write_page(vault, "domain/research/one.md", "no frontmatter at all\n")

    issues = wiki_health.check_health(vault, mode="ci")

    assert error_codes(issues) == {"index-invalid"}


def test_unknown_mode_is_rejected(vault):
    with pytest.raises(ValueError, match="mode"):
        wiki_health.check_health(vault, mode="quick")


# --------------------------------------------------------------------------
# community synthesis staleness
# --------------------------------------------------------------------------


def test_health_reports_a_required_but_missing_community_synthesis(vault):
    # three mutually linked pages form one community of size >= MIN_SYNTH_SIZE
    for name in ("alpha", "beta", "gamma"):
        others = [n for n in ("alpha", "beta", "gamma") if n != name]
        body = " ".join(f"[[{o}]]" for o in others) + "\n"
        write_page(vault, f"domain/research/{name}.md", page(body, id=name))
    write_index_and_embedding_fixture(
        vault,
        page_hashes=current_page_hashes(vault),
        meta=[{"id": "alpha"}, {"id": "beta"}, {"id": "gamma"}],
        vector_rows=3,
    )

    issues = wiki_health.check_health(vault, mode="full")

    assert error_codes(issues) == {"community-synthesis-stale"}
    # The message has to be self-explanatory: a first-time user hitting this
    # needs to know the file, the key, and the shape of the value without
    # reading this module's source.
    detail = next(i.detail for i in errors(issues) if i.code == "community-synthesis-stale")
    assert "community_summaries.json" in detail
    from llm_wiki.reports import community_report

    sig = community_report.stale_communities(vault)[0]["sig"]
    assert sig in detail


def test_a_grounded_sidecar_synthesis_clears_the_community_issue(vault):
    for name in ("alpha", "beta", "gamma"):
        others = [n for n in ("alpha", "beta", "gamma") if n != name]
        body = " ".join(f"[[{o}]]" for o in others) + "\n"
        write_page(vault, f"domain/research/{name}.md", page(body, id=name))
    write_index_and_embedding_fixture(
        vault,
        page_hashes=current_page_hashes(vault),
        meta=[{"id": "alpha"}, {"id": "beta"}, {"id": "gamma"}],
        vector_rows=3,
    )
    from llm_wiki.reports import community_report

    stale = community_report.stale_communities(vault)
    assert len(stale) == 1
    (vault / "community_summaries.json").write_text(
        json.dumps({stale[0]["sig"]: "grounded synthesis text"}, ensure_ascii=False), encoding="utf-8"
    )

    assert errors(wiki_health.check_health(vault, mode="full")) == []


def test_stale_communities_reads_the_given_vault_not_the_committed_one(vault):
    from llm_wiki.reports import community_report

    write_page(vault, "domain/research/solo.md", page(id="solo"))

    assert community_report.stale_communities(vault) == []


# --------------------------------------------------------------------------
# generated reports (GRAPH_REPORT.md / COMMUNITIES.md) are optional artifacts
# --------------------------------------------------------------------------


def test_report_checks_are_silent_when_the_vault_has_no_reports(vault):
    from llm_wiki import build_index

    write_page(vault, "domain/research/one.md", page(id="one"))
    build_index.write_index(vault, vault / "index.yaml")
    codes = {i.code for i in wiki_health.check_health(vault, mode="ci")}
    assert not any(c.startswith("report-") or c.startswith("community-") for c in codes)


# --------------------------------------------------------------------------
# content warnings never fail the gate
# --------------------------------------------------------------------------


def test_dangling_wikilinks_are_warnings_not_errors(vault):
    write_page(vault, "domain/research/one.md", page("see [[nowhere]]\n", id="one"))
    write_index_and_embedding_fixture(
        vault,
        page_hashes=current_page_hashes(vault),
        meta=[{"id": "one"}],
        vector_rows=1,
    )

    issues = wiki_health.check_health(vault, mode="full")

    assert "dangling-wikilink" in issue_codes(issues)
    assert errors(issues) == []
    assert wiki_health.exit_code(issues) == 0


def test_generated_reports_are_drift_checked_like_the_index(vault):
    """GRAPH_REPORT.md and COMMUNITIES.md are generated from the same pages as
    index.yaml. Checking only index.yaml exactly was inconsistent — a stale
    report is the same class of silently-wrong artifact."""
    write_page(vault, "domain/research/one.md", page(id="one"))
    write_index_and_embedding_fixture(
        vault,
        page_hashes=current_page_hashes(vault),
        meta=[{"id": "one"}],
        vector_rows=1,
    )
    (vault / "GRAPH_REPORT.md").write_text("stale report\n", encoding="utf-8")
    (vault / "COMMUNITIES.md").write_text("stale communities\n", encoding="utf-8")

    issues = wiki_health.check_health(vault, mode="full")

    assert error_codes(issues) == {"report-stale"}
    detail = errors(issues)[0].detail
    assert "GRAPH_REPORT.md" in detail and "COMMUNITIES.md" in detail


def test_absent_generated_reports_are_not_reported_as_drift(vault):
    """A vault that has never generated them is not *drifted*; regenerating is
    the ingest pipeline's job, not a health error."""
    write_page(vault, "domain/research/one.md", page(id="one"))
    write_index_and_embedding_fixture(
        vault, page_hashes=current_page_hashes(vault),
        meta=[{"id": "one"}], vector_rows=1)

    assert errors(wiki_health.check_health(vault, mode="full")) == []


def test_oversized_summary_is_a_warning(vault):
    """Contextual retrieval prefixes the summary onto EVERY chunk of the page,
    so a long summary inflates that whole page's chunks (measured 2026-07-27)."""
    write_page(vault, "domain/research/one.md", page(id="one", summary="x" * 400))
    write_index_and_embedding_fixture(
        vault, page_hashes=current_page_hashes(vault),
        meta=[{"id": "one"}], vector_rows=1)

    issues = wiki_health.check_health(vault, mode="full")

    assert "summary-too-long" in issue_codes(issues)
    assert errors(issues) == []


def test_orphan_pages_are_a_warning_with_a_count(vault):
    write_page(vault, "domain/research/one.md", page(id="one"))
    write_page(vault, "domain/research/two.md", page(id="two"))
    write_index_and_embedding_fixture(
        vault, page_hashes=current_page_hashes(vault),
        meta=[{"id": "one"}, {"id": "two"}], vector_rows=2)

    issues = wiki_health.check_health(vault, mode="full")

    assert "orphan-pages" in issue_codes(issues)
    assert errors(issues) == []
    detail = next(i.detail for i in issues if i.code == "orphan-pages")
    assert "2/2" in detail


def test_a_linked_page_is_not_an_orphan(vault):
    write_page(vault, "domain/research/one.md", page("see [[two]]\n", id="one"))
    write_page(vault, "domain/research/two.md", page(id="two"))
    write_index_and_embedding_fixture(
        vault, page_hashes=current_page_hashes(vault),
        meta=[{"id": "one"}, {"id": "two"}], vector_rows=2)

    issues = wiki_health.check_health(vault, mode="full")
    detail = next(i.detail for i in issues if i.code == "orphan-pages")

    assert "1/2" in detail


def test_unmeasured_claims_are_a_warning_not_an_error(vault):
    """The vault's founding rule finally has a check — but hedging is sometimes
    correct, so it must never block."""
    write_page(vault, "domain/research/one.md",
               page("This path is probably the cause.\n", id="one"))
    write_index_and_embedding_fixture(
        vault, page_hashes=current_page_hashes(vault),
        meta=[{"id": "one"}], vector_rows=1)

    issues = wiki_health.check_health(vault, mode="full")

    assert "unmeasured-claim" in issue_codes(issues)
    assert errors(issues) == []


def test_a_measured_page_raises_no_unmeasured_claim_warning(vault):
    write_page(vault, "domain/research/one.md",
               page("Likely the same cause: 12/40 runs failed.\n", id="one"))
    write_index_and_embedding_fixture(
        vault, page_hashes=current_page_hashes(vault),
        meta=[{"id": "one"}], vector_rows=1)

    assert "unmeasured-claim" not in issue_codes(wiki_health.check_health(vault, mode="full"))


def test_exit_code_is_one_when_any_error_is_present(vault):
    write_page(vault, "domain/research/one.md", page(id="one"))
    (vault / "index.yaml").write_text("version: 2\nentries: []\n", encoding="utf-8")

    issues = wiki_health.check_health(vault, mode="ci")

    assert wiki_health.exit_code(issues) == 1


def test_report_json_shape_is_stable(vault):
    write_page(vault, "domain/research/one.md", page("see [[nowhere]]\n", id="one"))
    (vault / "index.yaml").write_text("version: 2\nentries: []\n", encoding="utf-8")

    report = wiki_health.report(vault, mode="ci")

    assert report["ok"] is False
    assert report["mode"] == "ci"
    assert [e["code"] for e in report["errors"]] == ["index-stale"]
    assert "dangling-wikilink" in [w["code"] for w in report["warnings"]]


def test_health_never_rewrites_canonical_pages(vault):
    path = write_page(vault, "domain/research/one.md", page("see [[nowhere]]\n", id="one"))
    before = path.read_bytes()
    write_index_and_embedding_fixture(
        vault,
        page_hashes=current_page_hashes(vault),
        meta=[{"id": "one"}],
        vector_rows=1,
    )

    wiki_health.check_health(vault, mode="full")

    assert path.read_bytes() == before
    assert not (vault / "COMMUNITIES.md").exists()
    assert not (vault / "GRAPH_REPORT.md").exists()
