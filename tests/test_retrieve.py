"""Fielded Korean tokenization, pre-rank filters, and page aggregation.

Dense scores are always *injected* — this suite must never embed anything, so
every ranking assertion here is exact rather than "roughly the right order".
"""
from __future__ import annotations

import pytest

from llm_wiki.retrieval import embed_index
from llm_wiki.retrieval._retrieve import (
    SearchFilters,
    fielded_sparse_text,
    page_confidence,
    rank_from_scores,
    tokenize,
)


# --------------------------------------------------------------------------
# chunk metadata
# --------------------------------------------------------------------------


def test_chunk_metadata_carries_retrieval_frontmatter():
    fm = {
        "projects": ["project-b"],
        "tags": ["ranking-tag", "ranking-api"],
        "confidence": "confirmed",
        "status": "active",
        "updated": "2026-07-31",
        "summary": "hybrid ranking fusion weights",
    }

    meta = embed_index.chunk_metadata(fm, "ranking", "domain/tooling/ranking.md", 0, "body text")

    assert meta["projects"] == ["project-b"]
    assert meta["tags"] == ["ranking-tag", "ranking-api"]
    assert meta["status"] == "active"
    assert meta["summary"] == "hybrid ranking fusion weights"
    assert meta["confidence"] == "confirmed"
    assert meta["updated"] == "2026-07-31"
    assert meta["id"] == "ranking"
    assert meta["path"] == "domain/tooling/ranking.md"
    assert meta["chunk"] == 0
    assert meta["text"] == "body text"


def test_chunk_metadata_defaults_absent_fields_to_retrieval_safe_values():
    meta = embed_index.chunk_metadata({}, "orphan", "raw/orphan.md", 2, "sample text")

    assert meta["projects"] == []
    assert meta["tags"] == []
    # An unlabelled page must not be silently excluded by the default filter.
    assert meta["status"] == "active"
    assert meta["summary"] == ""


def test_chunk_metadata_coerces_a_yaml_date_to_an_iso_string():
    import datetime as dt

    meta = embed_index.chunk_metadata({"updated": dt.date(2026, 7, 31)}, "p", "raw/p.md", 0, "t")

    assert meta["updated"] == "2026-07-31"
    # meta.json is JSON — a bare date object would not survive serialization.
    import json

    assert json.loads(json.dumps(meta))["updated"] == "2026-07-31"


def test_chunk_metadata_snippet_is_single_line_and_bounded():
    meta = embed_index.chunk_metadata({}, "p", "raw/p.md", 0, "first line\nsecond line\n" + "x" * 400)

    assert "\n" not in meta["snippet"]
    assert len(meta["snippet"]) <= 240


# --------------------------------------------------------------------------
# tokenization
# --------------------------------------------------------------------------


def test_korean_tokenizer_emits_words_and_ngrams():
    assert set(tokenize("검색기 임베딩")) >= {
        "검색기", "검색", "색기", "임베딩", "임베", "베딩"
    }


def test_tokenizer_keeps_latin_and_api_tokens_exact():
    tokens = tokenize("orderStatusCode and rowFilterKey")

    assert "orderstatuscode" in tokens
    assert "rowfilterkey" in tokens
    # No character ngrams for Latin — that would flood BM25 with junk terms.
    assert "or" not in tokens
    assert "ro" not in tokens


def test_tokenizer_does_not_ngram_single_character_hangul():
    assert tokenize("각") == ["각"]


def test_tokenizer_deduplicates_while_preserving_order():
    tokens = tokenize("임베딩 임베딩")

    assert tokens == list(dict.fromkeys(tokens))
    assert tokens[0] == "임베딩"


def test_kiwi_tokenizer_splits_particles_and_keeps_identifiers():
    """The bigram hack exists because Korean glues particles onto stems. A real
    analyser splits them instead — and must not shred Latin API names."""
    pytest.importorskip("kiwipiepy")
    from llm_wiki.retrieval._retrieve import tokenize_kiwi

    assert "색인" in tokenize_kiwi("색인을 무한히 돌던 문제")
    assert "orderstatuscode" in tokenize_kiwi("orderStatusCode 와 rowFilterKey")
    assert "rowfilterkey" in tokenize_kiwi("orderStatusCode 와 rowFilterKey")


def test_the_default_tokenizer_is_bigram_regardless_of_kiwi_availability():
    """Installing an optional dependency must not silently change ranking."""
    from llm_wiki.retrieval._retrieve import TOKENIZER

    assert TOKENIZER == "bigram"


def test_tokenizer_handles_empty_and_none():
    assert tokenize("") == []
    assert tokenize(None) == []


# --------------------------------------------------------------------------
# fielded sparse text
# --------------------------------------------------------------------------


def test_fielded_sparse_text_upweights_id_and_tags_over_body():
    meta = {"id": "doc-primary", "tags": ["reference"], "summary": "summary", "text": "body"}

    text = fielded_sparse_text(meta)

    assert text.count("doc-primary") == 3
    assert text.count("reference") == 3
    assert text.count("summary") == 2
    assert text.count("body") == 1


def test_fielded_sparse_text_does_not_mutate_the_body_snippet():
    meta = {"id": "p", "tags": [], "summary": "s", "text": "original body"}

    fielded_sparse_text(meta)

    assert meta["text"] == "original body"


def test_fielded_sparse_text_tolerates_missing_fields():
    text = fielded_sparse_text({"id": "p"})

    assert text.split() == ["p", "p", "p"]


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------


ACTIVE = {"id": "active-page", "status": "active", "layer": "domain",
          "domain": "research", "projects": ["project-a"], "confidence": "confirmed"}
SUPERSEDED = {"id": "old-page", "status": "superseded", "layer": "domain",
              "domain": "research", "projects": ["project-a"], "confidence": "confirmed"}


def test_default_filter_excludes_superseded_pages():
    f = SearchFilters()

    assert f.keep(ACTIVE) is True
    assert f.keep(SUPERSEDED) is False


def test_status_any_opts_back_into_superseded_pages():
    f = SearchFilters(status=None)

    assert f.keep(ACTIVE) is True
    assert f.keep(SUPERSEDED) is True


def test_project_filter_matches_membership_not_equality():
    assert SearchFilters(project="project-a").keep(ACTIVE) is True
    assert SearchFilters(project="project-b").keep(ACTIVE) is False


def test_layer_domain_and_confidence_filters_compose():
    assert SearchFilters(layer="domain", domain="research", confidence="confirmed").keep(ACTIVE) is True
    assert SearchFilters(layer="pattern").keep(ACTIVE) is False
    assert SearchFilters(domain="tooling").keep(ACTIVE) is False
    assert SearchFilters(confidence="provisional").keep(ACTIVE) is False


def test_filter_treats_a_page_with_no_status_as_active():
    assert SearchFilters().keep({"id": "p"}) is True


# --------------------------------------------------------------------------
# ranking / aggregation
# --------------------------------------------------------------------------


def chunk(_id, text="body", **over):
    m = {"id": _id, "path": f"domain/research/{_id}.md", "layer": "domain", "domain": "research",
         "chunk": 0, "text": text, "snippet": text, "tags": [], "summary": "",
         "projects": [], "status": "active", "confidence": "confirmed", "updated": "2026-07-01"}
    m.update(over)
    return m


def test_default_filter_excludes_superseded_pages_from_results():
    meta = [chunk("active-page"), chunk("old-page", status="superseded")]

    ranked = rank_from_scores(meta, dense=[0.1, 0.9], sparse=[0.1, 0.9],
                              filters=SearchFilters(), k=8)

    assert [m["id"] for _, m in ranked] == ["active-page"]


def test_page_aggregation_uses_best_chunk_not_chunk_count():
    meta = [chunk("short"), chunk("long"), chunk("long"), chunk("other")]
    dense = [0.9, 0.1, 0.1, 0.8]
    sparse = [0.9, 0.1, 0.1, 0.8]

    ranked = rank_from_scores(meta, dense=dense, sparse=sparse,
                              filters=SearchFilters(), k=3)

    assert [m["id"] for _, m in ranked] == ["short", "other", "long"]


def test_filtering_happens_before_rank_enumeration():
    """A filtered-out chunk must not consume a rank slot.

    If the filter ran after RRF rank enumeration, the two superseded chunks
    would occupy ranks 0 and 1 and push `wanted` down the fused score even
    though it is the only admissible result.
    """
    meta = [chunk("dead-1", status="superseded"), chunk("dead-2", status="superseded"),
            chunk("wanted"), chunk("other")]
    dense = [0.99, 0.98, 0.50, 0.49]
    sparse = [0.99, 0.98, 0.50, 0.49]

    ranked = rank_from_scores(meta, dense=dense, sparse=sparse,
                              filters=SearchFilters(), k=2)
    top_score = ranked[0][0]

    unfiltered = rank_from_scores(meta, dense=dense, sparse=sparse,
                                  filters=SearchFilters(status=None), k=4)
    wanted_score = next(s for s, m in unfiltered if m["id"] == "wanted")

    assert [m["id"] for _, m in ranked] == ["wanted", "other"]
    assert top_score > wanted_score


def test_chunk_count_does_not_change_a_pages_rank():
    """Duplicating a page's chunks must not improve its rank."""
    base = [chunk("rival"), chunk("target")]
    dense, sparse = [0.80, 0.75], [0.80, 0.75]
    before = rank_from_scores(base, dense, sparse, SearchFilters(), k=2)

    padded = [chunk("rival")] * 6 + [chunk("target")]
    ranked = rank_from_scores(padded, [0.80] * 6 + [0.75], [0.80] * 6 + [0.75],
                              SearchFilters(), k=2)

    assert [m["id"] for _, m in before] == [m["id"] for _, m in ranked]


def test_ranking_is_stable_for_tied_scores():
    meta = [chunk("b"), chunk("a")]
    dense = [0.5, 0.5]
    sparse = [0.5, 0.5]

    first = rank_from_scores(meta, dense, sparse, SearchFilters(), k=2)
    second = rank_from_scores(meta, dense, sparse, SearchFilters(), k=2)

    assert [m["id"] for _, m in first] == [m["id"] for _, m in second]


def test_rank_returns_empty_when_everything_is_filtered_out():
    meta = [chunk("dead", status="superseded")]

    assert rank_from_scores(meta, [0.9], [0.9], SearchFilters(), k=8) == []


def test_confidence_is_absolute_cosine_not_the_scale_free_fused_score():
    """RRF is deliberately scale-free: its score is a function of RANK, so the
    top result of a nonsense query scores about the same as the top result of a
    perfect one (measured 2026-07-31 — negatives 0.0363-0.0489 vs positives
    0.0425-0.0500, overlapping). Automatic consumption therefore has to read
    confidence off the absolute dense cosine, which does separate them.
    """
    meta = [chunk("wanted"), chunk("wanted"), chunk("other")]
    dense = [0.31, 0.88, 0.30]

    ranked = rank_from_scores(meta, dense=dense, sparse=[0.9, 0.1, 0.1],
                              filters=SearchFilters(), k=2)
    conf = page_confidence(meta, dense, {m["id"] for _, m in ranked})

    fused_top = ranked[0][0]
    assert fused_top < 0.1                      # RRF lives on a tiny fixed scale
    assert conf["wanted"] == pytest.approx(0.88)  # best chunk's absolute cosine
    assert conf["other"] == pytest.approx(0.30)


# --------------------------------------------------------------------------
# reranking: whether it ACTUALLY ran is what the auto-policy gates on
# --------------------------------------------------------------------------


def ranked_pair():
    return [(0.05, chunk("first")), (0.04, chunk("second"))]


def test_apply_rerank_reports_that_it_ran_and_reorders():
    from llm_wiki.retrieval._retrieve import apply_rerank

    out, reranked = apply_rerank(ranked_pair(), "q", rerank_fn=lambda q, t: [0.1, 0.9])

    assert reranked is True
    assert [m["id"] for _, m in out] == ["second", "first"]


@pytest.mark.parametrize("exc", [ImportError, OSError, RuntimeError])
def test_apply_rerank_reports_failure_instead_of_pretending_it_ran(exc):
    """The auto-policy fails closed on `reranked is False`. If a model-load
    failure silently returned the un-reranked list as if it had been reranked,
    the fail-closed contract would be unenforceable."""
    from llm_wiki.retrieval._retrieve import apply_rerank

    def boom(q, t):
        raise exc("model unavailable")

    out, reranked = apply_rerank(ranked_pair(), "q", rerank_fn=boom)

    assert reranked is False
    assert [m["id"] for _, m in out] == ["first", "second"]  # original order kept


def test_apply_rerank_on_empty_candidates_reports_not_reranked():
    from llm_wiki.retrieval._retrieve import apply_rerank

    assert apply_rerank([], "q", rerank_fn=lambda q, t: []) == ([], False)


def test_apply_rerank_falls_back_to_chunk_text_when_the_page_is_unreadable(tmp_path):
    """`vault` is passed explicitly (rather than relying on the `apply_rerank`
    default) so this test owns its own filesystem: the fixture's paths must not
    exist under `tmp_path`, regardless of what does or does not exist at the
    process's real vault root."""
    from llm_wiki.retrieval._retrieve import apply_rerank

    seen = {}

    def capture(q, texts):
        seen["texts"] = texts
        return [0.5] * len(texts)

    apply_rerank(ranked_pair(), "q", vault=tmp_path, rerank_fn=capture)

    # paths in the fixture do not exist under tmp_path -> the fallback returns
    # each chunk's own `text` field verbatim rather than reading the page from
    # disk. Asserting the exact value (not just "non-empty") fails if the
    # fallback did not actually run.
    assert seen["texts"] == ["body", "body"]


def test_page_confidence_ignores_pages_outside_the_requested_set():
    meta = [chunk("a"), chunk("b")]

    assert page_confidence(meta, [0.9, 0.8], {"a"}) == {"a": pytest.approx(0.9)}


# --------------------------------------------------------------------------
# missing embedding store: diagnosability, not redirection
# --------------------------------------------------------------------------


def test_load_reports_a_clear_error_when_no_embedding_store_exists(tmp_path, monkeypatch):
    """VAULT is resolved from the working directory, not pinned to this file's
    location, so a missing store is an expected, diagnosable state (wrong cwd,
    or wiki-embed never ran) rather than a broken install. The error must name
    the resolved vault path and say how to fix it, instead of a bare
    FileNotFoundError from inside numpy."""
    from llm_wiki.retrieval import _retrieve

    monkeypatch.setattr(_retrieve, "VAULT", tmp_path)
    monkeypatch.setattr(_retrieve, "EMB", tmp_path / ".embeddings")
    monkeypatch.setattr(_retrieve, "_cache", {})

    with pytest.raises(RuntimeError) as exc_info:
        _retrieve._load()

    message = str(exc_info.value)
    assert str(tmp_path) in message
    assert "wiki-embed" in message
    assert "WIKI_VAULT" in message


def test_load_reports_a_clear_error_when_embedding_store_is_empty(tmp_path, monkeypatch):
    """An empty store used to fall through to rank-bm25 and raise
    ZeroDivisionError. Name the real problem instead: wiki-index found no pages
    when wiki-embed last ran, or the store needs rebuilding."""
    import json
    import numpy as np
    from llm_wiki.retrieval import _retrieve

    emb = tmp_path / ".embeddings"
    emb.mkdir()
    np.save(emb / "vectors.npy", np.empty((0, 1024), dtype=np.float32))
    (emb / "meta.json").write_text(json.dumps([]), encoding="utf-8")

    monkeypatch.setattr(_retrieve, "VAULT", tmp_path)
    monkeypatch.setattr(_retrieve, "EMB", emb)
    monkeypatch.setattr(_retrieve, "_cache", {})

    with pytest.raises(RuntimeError) as exc_info:
        _retrieve._load()

    message = str(exc_info.value)
    assert "empty" in message
    assert "wiki-index" in message
    assert "wiki-embed" in message


def test_dense_mode_ranks_by_raw_cosine_only():
    meta = [chunk("sparse-favourite"), chunk("dense-favourite")]

    ranked = rank_from_scores(meta, dense=[0.1, 0.9], sparse=[0.9, 0.1],
                              filters=SearchFilters(), k=2, mode="dense")

    assert [m["id"] for _, m in ranked] == ["dense-favourite", "sparse-favourite"]
    assert ranked[0][0] == pytest.approx(0.9)
