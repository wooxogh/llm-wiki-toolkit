"""The graph and community reports are deterministic functions of the vault.

Every function that accepts a `vault` argument must actually read that vault
and only that vault — never fall back to the process's own committed vault
(`llm_wiki.paths.VAULT_ROOT`). Several tests here are deliberately written so
that a broken `vault` handoff (a call that silently drops the argument and
reads the default instead) makes the assertion fail, not just look odd.
"""
from __future__ import annotations

import json

from conftest import page, write_page
from llm_wiki.reports import community_report, graph_report


def linked(vault, _id: str, *targets: str, **overrides):
    body = " ".join(f"[[{t}]]" for t in targets) + "\n"
    return write_page(vault, f"domain/research/{_id}.md", page(body, id=_id, **overrides))


def isolated(vault, _id: str, **overrides):
    return write_page(vault, f"domain/research/{_id}.md", page("no links here\n", id=_id, **overrides))


# -- graph_report -------------------------------------------------------


def test_collect_reads_every_content_page(vault):
    linked(vault, "a", "b")
    linked(vault, "b")
    assert set(graph_report.collect(vault)) == {"a", "b"}


def test_wikilinks_ignore_code_spans():
    assert graph_report.wikilinks("see [[a]] but not `[[b]]`") == {"a"}


def test_degrees_count_inbound_links(vault):
    linked(vault, "a", "b")
    linked(vault, "b")
    pages = graph_report.collect(vault)
    graph_report.build_degrees(pages)
    assert pages["b"]["indeg"] == 1
    assert pages["a"]["indeg"] == 0
    assert len(pages["a"]["out"]) == 1


def test_dangling_links_are_attributed_to_their_referrer(vault):
    linked(vault, "a", "nonexistent")
    a = graph_report.analyze(graph_report.collect(vault))
    assert a["dangling"] == {"nonexistent": ["a"]}
    assert a["n_dangling"] == 1


def test_islands_have_neither_inbound_nor_outbound(vault):
    linked(vault, "a", "b")
    linked(vault, "b")
    isolated(vault, "c")
    a = graph_report.analyze(graph_report.collect(vault))
    assert a["islands"] == ["c"]
    # b has an inbound link so it is an orphan-free hub, not an island
    assert "b" not in a["islands"]


def test_communities_group_pages_that_link_each_other(vault):
    linked(vault, "a", "b")
    linked(vault, "b", "c")
    linked(vault, "c", "a")
    isolated(vault, "solo")
    a = graph_report.analyze(graph_report.collect(vault))
    assert len(a["communities"]) == 1
    assert a["communities"][0]["members"] == ["a", "b", "c"]
    assert a["communities"][0]["size"] == 3


def test_render_md_is_a_smoke_test_over_a_sparse_graph(vault):
    """An almost-empty graph must not crash render_md (no god-nodes, no
    dangling links, no bridges, no communities — every "else" branch)."""
    isolated(vault, "only")
    a = graph_report.analyze(graph_report.collect(vault))
    md = graph_report.render_md(a)
    assert "# GRAPH_REPORT" in md
    assert "## Summary" in md
    assert "only" in md


def test_render_md_handles_a_completely_empty_vault(vault):
    """An empty vault (n_pages == 0) is the normal state right after
    `pip install` — before any page exists. render_md divides by n_pages in
    two places (average out-degree, orphan percentage); both must degrade to
    sensible output instead of raising ZeroDivisionError."""
    a = graph_report.analyze(graph_report.collect(vault))
    md = graph_report.render_md(a)
    assert isinstance(md, str)
    assert "# GRAPH_REPORT" in md


def test_render_md_reports_god_nodes_and_dangling(vault):
    linked(vault, "a", "hub")
    linked(vault, "b", "hub")
    linked(vault, "hub", "missing")
    a = graph_report.analyze(graph_report.collect(vault))
    md = graph_report.render_md(a)
    assert "`hub` <- **2** inbound" in md
    assert "[[missing]]" in md


# -- community_report -----------------------------------------------------


def test_signature_is_order_independent():
    assert community_report.signature(["b", "a"]) == community_report.signature(["a", "b"])


def test_communities_below_the_synthesis_floor_are_not_stale(vault):
    linked(vault, "a")
    assert community_report.stale_communities(vault) == []


def test_load_sidecar_missing_file_returns_empty_mapping(vault):
    """A freshly created vault has no `community_summaries.json` yet. That must
    be the normal, unremarkable case: an empty mapping, not an error."""
    assert community_report.load_sidecar(vault) == {}


def test_load_sidecar_reads_from_the_vault_it_is_given(vault, tmp_path_factory):
    """Two vaults must not share sidecar state. If `load_sidecar` ever fell
    back to a module-level default instead of honouring its `vault` argument,
    this would either raise (wrong path) or read the wrong vault's file."""
    other_vault = tmp_path_factory.mktemp("other-vault")
    (other_vault / "community_summaries.json").write_text(
        json.dumps({"deadbeefcafe": "a synthesis that belongs to the other vault"}),
        encoding="utf-8",
    )
    assert community_report.load_sidecar(vault) == {}
    assert community_report.load_sidecar(other_vault) == {
        "deadbeefcafe": "a synthesis that belongs to the other vault"
    }


def test_stale_communities_honours_a_synthesis_written_for_this_vault(vault):
    """A community with a matching sidecar synthesis is not stale. This only
    holds if `stale_communities` -> `community_rows` -> `build` ->
    `load_sidecar` all thread the *same* `vault` through — if any hop silently
    dropped it and read the process's own committed vault instead (which has
    no matching sidecar entry), this community would wrongly show up as
    awaiting synthesis."""
    linked(vault, "a", "b")
    linked(vault, "b", "c")
    linked(vault, "c", "a")
    sig = community_report.signature(["a", "b", "c"])
    write_page(vault, "community_summaries.json", json.dumps({sig: "a real synthesis"}))
    assert community_report.stale_communities(vault) == []

    rows = community_report.community_rows(vault)
    assert rows[0]["synthesis"] == "a real synthesis"


def test_stale_communities_flags_a_large_community_with_no_synthesis(vault):
    linked(vault, "a", "b")
    linked(vault, "b", "c")
    linked(vault, "c", "a")
    stale = community_report.stale_communities(vault)
    assert len(stale) == 1
    assert stale[0]["members"] == ["a", "b", "c"]


def test_render_md_pulls_page_summaries_from_the_vault_it_is_given(vault):
    """render_md's evidence lines come from each member's own `summary:`
    frontmatter — read via `page_summaries(vault)`. If `vault` were not
    threaded through, this vault's summaries would never be found (the
    process's own committed vault has no such pages) and the line would
    render bare, with no summary text after the id."""
    linked(vault, "a", "b", summary="a measured fact about a")
    linked(vault, "b", summary="a measured fact about b")
    rows = community_report.community_rows(vault)
    md = community_report.render_md(rows, vault)
    assert "a measured fact about a" in md
    assert "a measured fact about b" in md


def test_render_md_flags_communities_awaiting_synthesis(vault):
    linked(vault, "a", "b")
    linked(vault, "b", "c")
    linked(vault, "c", "a")
    rows = community_report.community_rows(vault)
    md = community_report.render_md(rows, vault)
    assert "awaiting synthesis" in md


def test_community_rows_do_not_leak_between_vaults(vault, tmp_path_factory):
    """Calling community_rows on one vault and then another must not carry
    over state (e.g. a cached sidecar or a cached page-summary map)."""
    linked(vault, "a", "b")
    linked(vault, "b", "c")
    linked(vault, "c", "a")

    other_vault = tmp_path_factory.mktemp("other-vault-2")
    for d in ("domain/research", "domain/tooling", "patterns", "entities", "raw"):
        (other_vault / d).mkdir(parents=True, exist_ok=True)
    linked(other_vault, "x", "y")
    linked(other_vault, "y", "z")
    linked(other_vault, "z", "x")

    first = community_report.community_rows(vault)
    second = community_report.community_rows(other_vault)

    assert first[0]["members"] == ["a", "b", "c"]
    assert second[0]["members"] == ["x", "y", "z"]
