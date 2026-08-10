"""Exact-index and schema behaviour for llm_wiki.build_index."""
from __future__ import annotations

import pytest
import yaml

from llm_wiki import build_index, paths
from conftest import build_fm, page, page_from, valid_frontmatter, write_page

STALE = "index.yaml differs from canonical page frontmatter"


def test_check_index_accepts_a_freshly_written_index(vault):
    write_page(vault, "domain/research/one.md", page(id="one", summary="first"))
    build_index.write_index(vault, vault / "index.yaml")

    assert build_index.check_index(vault, vault / "index.yaml") == []


def test_check_index_rejects_valid_but_stale_index(vault):
    write_page(vault, "domain/research/one.md", page(id="one", summary="first"))
    build_index.write_index(vault, vault / "index.yaml")
    write_page(vault, "domain/research/one.md", page(id="one", summary="changed"))

    assert build_index.check_index(vault, vault / "index.yaml") == [STALE]


def test_check_index_rejects_a_missing_index(vault):
    write_page(vault, "domain/research/one.md", page(id="one"))

    assert build_index.check_index(vault, vault / "index.yaml") == [STALE]


def test_check_index_rejects_an_index_stale_only_by_a_new_page(vault):
    write_page(vault, "domain/research/one.md", page(id="one"))
    build_index.write_index(vault, vault / "index.yaml")
    write_page(vault, "domain/research/two.md", page(id="two"))

    assert build_index.check_index(vault, vault / "index.yaml") == [STALE]


@pytest.mark.parametrize(
    ("field", "value"),
    [("projects", "repo"), ("tags", "tag"), ("summary", ["not", "text"])],
)
def test_validate_rejects_wrong_frontmatter_types(vault, field, value):
    fm = valid_frontmatter(id="one")
    fm[field] = value
    write_page(vault, "domain/research/one.md", page_from(fm))

    errors = build_index.validate(build_index.collect_pages(vault), vault)

    assert any(f"{field} must be" in error for error in errors)


def test_validate_rejects_non_string_members_in_list_fields(vault):
    fm = valid_frontmatter(id="one", tags=["ok", 7])
    write_page(vault, "domain/research/one.md", page_from(fm))

    errors = build_index.validate(build_index.collect_pages(vault), vault)

    assert any("tags must be" in error for error in errors)


def test_validate_rejects_empty_string_members_in_list_fields(vault):
    fm = valid_frontmatter(id="one", projects=["project-b", "  "])
    write_page(vault, "domain/research/one.md", page_from(fm))

    errors = build_index.validate(build_index.collect_pages(vault), vault)

    assert any("projects must be" in error for error in errors)


@pytest.mark.parametrize("bad_id", ["Not_Kebab", "trailing-", "UPPER", "two--dashes"])
def test_validate_rejects_non_kebab_ids(vault, bad_id):
    fm = valid_frontmatter(id=bad_id)
    write_page(vault, f"domain/research/{bad_id}.md", page_from(fm))

    errors = build_index.validate(build_index.collect_pages(vault), vault)

    assert any("id must be kebab-case" in error for error in errors)


def test_validate_rejects_non_date_updated(vault):
    fm = valid_frontmatter(id="one", updated="July 31 2026")
    write_page(vault, "domain/research/one.md", page_from(fm))

    errors = build_index.validate(build_index.collect_pages(vault), vault)

    assert any("updated must be" in error for error in errors)


def test_validate_accepts_iso_date_as_string_or_date(vault):
    write_page(vault, "domain/research/one.md", page_from(valid_frontmatter(id="one", updated="2026-07-31")))
    write_page(vault, "domain/research/two.md", page_from(valid_frontmatter(id="two")))

    assert build_index.validate(build_index.collect_pages(vault), vault) == []


def test_render_index_is_relative_to_the_given_vault_not_the_module_global(vault):
    write_page(vault, "domain/tooling/one.md", page(id="one", domain="tooling"))

    rendered = build_index.render_index(vault)
    entry = yaml.safe_load(rendered)["entries"][0]

    assert entry["path"] == "domain/tooling/one.md"


def test_render_index_raises_on_validation_errors(vault):
    fm = valid_frontmatter(id="one")
    del fm["summary"]
    write_page(vault, "domain/research/one.md", page_from(fm))

    with pytest.raises(ValueError, match="summary"):
        build_index.render_index(vault)


def test_render_index_is_deterministic_across_calls(vault):
    write_page(vault, "domain/research/b.md", page(id="b"))
    write_page(vault, "domain/research/a.md", page(id="a"))
    write_page(vault, "patterns/p.md", page(id="p", layer="pattern", domain=None))

    assert build_index.render_index(vault) == build_index.render_index(vault)


def test_collect_pages_walks_every_canonical_content_dir(vault):
    write_page(vault, "domain/research/one.md", page(id="one"))
    write_page(vault, "patterns/two.md", page(id="two", layer="pattern", domain=None))
    write_page(vault, "entities/three.md", page(id="three", layer="entity", domain=None))
    write_page(vault, "raw/four.md", page(id="four", layer="raw", domain=None))
    write_page(vault, "docs/not-canonical.md", page(id="five"))

    ids = {fm["id"] for _, fm in build_index.collect_pages(vault)}

    assert ids == {"one", "two", "three", "four"}


def test_unknown_domain_is_rejected_only_when_the_vault_declares_domains(vault):
    write_page(vault, "domain/research/one.md", page(id="one", domain="nope"))
    # No wiki.toml: the vault has not declared a domain vocabulary, so any value passes.
    assert build_index.write_index(vault, vault / "index.yaml") == 1

    (vault / "wiki.toml").write_text(
        '[schema]\ndomains = ["research", "tooling"]\n', encoding="utf-8")
    errors = build_index.validate(
        [(p, build_fm(p)) for p in paths.content_paths(vault)], vault)
    assert any("domain" in e for e in errors)


def test_invalid_layer_error_names_the_offending_value(vault):
    fm = valid_frontmatter(id="one")
    fm["layer"] = "bogus"
    write_page(vault, "domain/research/one.md", page_from(fm))

    errors = build_index.validate(build_index.collect_pages(vault), vault)

    assert any("bogus" in e for e in errors)


def test_invalid_domain_error_names_the_offending_value(vault):
    write_page(vault, "domain/research/one.md", page(id="one", domain="nope"))
    (vault / "wiki.toml").write_text(
        '[schema]\ndomains = ["research", "tooling"]\n', encoding="utf-8")

    errors = build_index.validate(
        [(p, build_fm(p)) for p in paths.content_paths(vault)], vault)

    assert any("nope" in e for e in errors)


# --------------------------------------------------------------------------
# silent-invisibility guards (both classes were found live in a real vault)
# --------------------------------------------------------------------------


def test_validate_rejects_summary_truncated_by_an_unquoted_hash(vault):
    """YAML starts a comment at ' #', so `summary: measured PR #527 and more`
    silently reaches index.yaml as `measured PR` — valid YAML, lost knowledge."""
    write_page(
        vault,
        "domain/research/one.md",
        "---\n"
        "id: one\nlayer: domain\ndomain: research\nprojects:\n- project-a\n"
        "tags:\n- retrieval\nconfidence: confirmed\nstatus: active\n"
        "summary: measured in PR #527 — 1,204 chunks re-embedded\n"
        "---\n\nbody\n",
    )

    errors = build_index.validate(build_index.collect_pages(vault), vault)

    assert any("truncated" in e and "summary" in e for e in errors)


def test_validate_accepts_a_hash_inside_a_quoted_summary(vault):
    write_page(
        vault,
        "domain/research/one.md",
        "---\n"
        "id: one\nlayer: domain\ndomain: research\nprojects:\n- project-a\n"
        "tags:\n- retrieval\nconfidence: confirmed\nstatus: active\n"
        "summary: 'measured in PR #527 — 1,204 chunks re-embedded'\n"
        "---\n\nbody\n",
    )

    assert build_index.validate(build_index.collect_pages(vault), vault) == []


def test_validate_rejects_a_page_with_another_page_buried_in_its_body(vault):
    """index.yaml registers only the first frontmatter block, so concatenated
    pages are invisible to the index and to recall."""
    buried = (
        "---\nid: buried-one\nlayer: pattern\nprojects:\n- project-b\n"
        "tags:\n- gotcha\nconfidence: confirmed\nstatus: active\nsummary: buried\n---\n\nbody2\n"
    )
    write_page(vault, "patterns/host.md", page(id="host", layer="pattern", domain=None) + buried)

    errors = build_index.validate(build_index.collect_pages(vault), vault)

    assert any("buried" in e and "buried-one" in e for e in errors)


def test_validate_ignores_frontmatter_shown_inside_a_fenced_code_block(vault):
    body = "example:\n\n```yaml\n---\nid: sample\nlayer: domain\n---\n```\n"
    write_page(vault, "domain/research/one.md", page(body, id="one"))

    assert build_index.validate(build_index.collect_pages(vault), vault) == []
