"""`[vault] root` must redirect content WITHOUT discarding the rest of wiki.toml.

Every test here drives a CLI as a real subprocess, through `WIKI_VAULT`, because
that is the only path the CLIs actually take: `llm_wiki.paths` resolves the roots
once at import time, so an in-process test that passes `vault=` explicitly
exercises a different code path than `wiki-index` does. `tests/test_paths.py`
only ever passed the argument, and that is exactly why this class of bug lived:

    config.load(root)  ->  Config.root = root/<[vault] root>
    paths.VAULT_ROOT   =  that content root
    config.load(VAULT_ROOT)  ->  no wiki.toml in the subdirectory
                             ->  silently back to the built-in defaults

The vault's `content_dirs`, `layers`, `domains`, `required`, lint packs, gold
file and `[eval.minimums]` all vanished, `index.yaml` was rewritten with zero
entries, and both gates reported success — the precise silent-drift failure
`build_index` exists to prevent. So each test below asserts on an *effect that
is only possible if the redirected vault's own config was read*.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from conftest import page

REPO_ROOT = Path(__file__).resolve().parents[1]

# A vault whose every declaration differs from the built-in defaults, so nothing
# below can pass by accidentally agreeing with a default.
WIKI_TOML = """\
[vault]
root = "sub"
content_dirs = ["notes"]

[schema]
layers = ["domain"]
domains = ["research"]
required = ["id", "layer", "summary"]

[lint]
packs = ["ko"]

[eval]
gold = "cases.json"

[eval.minimums]
total = 1
recent_cases = 0
layer = { domain = 1 }
category = {}
"""

# `projects` and `tags` are deliberately absent: they are in the *default*
# `required` tuple but not in this vault's, so a page that validates at all
# proves `[schema] required` was read from the redirected vault's wiki.toml.
PAGE = """\
---
id: redirected
layer: domain
domain: research
confidence: confirmed
status: active
updated: 2026-08-01
summary: one measured line
---

청크 크기를 1400까지 올려도 괜찮아 보인다.
"""

GOLD = """\
[{"q": "what is redirected", "expect": ["redirected"], "layer": "domain",
  "domain": "research", "category": "direct", "difficulty": "easy",
  "split": "test"}]
"""


@pytest.fixture()
def redirected_vault(tmp_path: Path) -> Path:
    """A vault whose wiki.toml sits one level above its content."""
    (tmp_path / "wiki.toml").write_text(WIKI_TOML, encoding="utf-8")
    notes = tmp_path / "sub" / "notes"
    notes.mkdir(parents=True)
    (notes / "redirected.md").write_text(PAGE, encoding="utf-8")
    (tmp_path / "sub" / "cases.json").write_text(GOLD, encoding="utf-8")
    # A decoy in the *config* root's default content dir. Anything that finds
    # this has ignored both `[vault] root` and `content_dirs`.
    decoy = tmp_path / "domain" / "research"
    decoy.mkdir(parents=True)
    (decoy / "decoy.md").write_text(page(id="decoy"), encoding="utf-8")
    return tmp_path


def run_cli(vault: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=REPO_ROOT, env={**os.environ, "WIKI_VAULT": str(vault)},
        capture_output=True, text=True, timeout=60,
    )


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------


def test_index_is_written_into_the_content_root_with_the_redirected_pages(redirected_vault):
    result = run_cli(redirected_vault, "llm_wiki.build_index")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 entries" in result.stdout, result.stdout
    index = redirected_vault / "sub" / "index.yaml"
    assert index.exists(), "index.yaml must live in the content root, beside the pages"
    assert not (redirected_vault / "index.yaml").exists()
    entries = yaml.safe_load(index.read_text(encoding="utf-8"))["entries"]
    assert [e["id"] for e in entries] == ["redirected"]
    # Content-root-relative, not config-root-relative: "notes/…", never "sub/notes/…".
    assert entries[0]["path"] == "notes/redirected.md"


def test_the_decoy_outside_the_content_root_is_not_indexed(redirected_vault):
    """`domain/research/decoy.md` is a valid page under the *default* layout. It
    must stay invisible: finding it means `content_dirs` or the redirect was
    dropped, and the previous bug found it (as an error) for exactly that reason."""
    run_cli(redirected_vault, "llm_wiki.build_index")

    index = redirected_vault / "sub" / "index.yaml"
    assert "decoy" not in index.read_text(encoding="utf-8")


def test_index_check_agrees_with_what_index_just_wrote(redirected_vault):
    """--check must byte-compare the same file --write wrote. Two different
    answers to "where is index.yaml" would make the drift gate unfalsifiable."""
    assert run_cli(redirected_vault, "llm_wiki.build_index").returncode == 0

    result = run_cli(redirected_vault, "llm_wiki.build_index", "--check")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "up to date" in result.stdout


def test_index_check_still_detects_real_drift_under_a_redirect(redirected_vault):
    """The gate must not be merely quiet — a stale index in the content root has
    to fail, or `--check` returning 0 above proves nothing."""
    run_cli(redirected_vault, "llm_wiki.build_index")
    (redirected_vault / "sub" / "index.yaml").write_text(
        "version: 2\nentries: []\n", encoding="utf-8")

    result = run_cli(redirected_vault, "llm_wiki.build_index", "--check")

    assert result.returncode == 1
    assert "differs from canonical page frontmatter" in result.stderr


# --------------------------------------------------------------------------
# schema and lint packs
# --------------------------------------------------------------------------


def test_the_vaults_own_required_fields_are_honoured(redirected_vault):
    """The page carries no `projects:` and no `tags:`, both of which the built-in
    `required` tuple demands. It validating at all means `[schema] required` came
    from the redirected vault; under the bug it failed with default-schema errors."""
    result = run_cli(redirected_vault, "llm_wiki.build_index")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "missing required field" not in result.stderr


def test_the_vaults_own_layers_narrow_what_a_redirected_page_may_be(redirected_vault):
    """The paired positive: `layers = ["domain"]` is a strict subset of the
    default four, so a `layer: pattern` page must be rejected *by this vault's
    list*. Under the bug the defaults applied and `pattern` was accepted."""
    (redirected_vault / "sub" / "notes" / "extra.md").write_text(
        PAGE.replace("id: redirected", "id: extra")
            .replace("layer: domain\ndomain: research\n", "layer: pattern\n"),
        encoding="utf-8")

    result = run_cli(redirected_vault, "llm_wiki.build_index")

    assert result.returncode == 1
    assert "layer 'pattern' not in ['domain']" in result.stderr, result.stderr


def test_health_reads_the_redirected_content_and_the_vaults_lint_packs(redirected_vault):
    """A Korean hedge with no measurement is only flagged when `[lint] packs =
    ["ko"]` is read. Under the default `en` pack the page looks clean, which is
    how the bug reported "healthy (0 errors, 0 warnings)" on an empty vault."""
    run_cli(redirected_vault, "llm_wiki.build_index")

    result = run_cli(redirected_vault, "llm_wiki.wiki_health", "--mode", "ci")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "unmeasured-claim" in result.stderr, result.stderr
    assert "0 warning(s)" not in result.stdout


def test_health_fails_when_the_redirected_index_is_stale(redirected_vault):
    """The paired negative: health must be capable of failing here. A gate that
    cannot see the content root cannot see its drift either."""
    result = run_cli(redirected_vault, "llm_wiki.wiki_health", "--mode", "ci")

    assert result.returncode == 1
    assert "index-stale" in result.stderr or "index-invalid" in result.stderr


# --------------------------------------------------------------------------
# gold set and minimums
# --------------------------------------------------------------------------


def test_the_gold_file_resolves_against_the_content_root_under_its_own_minimums(redirected_vault):
    """`[eval] gold = "cases.json"` lives in the content root, and the single
    case only passes curation because this vault lowered [eval.minimums]. Both
    facts come from the same wiki.toml the bug discarded."""
    run_cli(redirected_vault, "llm_wiki.build_index")

    result = run_cli(redirected_vault, "llm_wiki.evaluation.eval", "--validate-only")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "gold set valid and complete" in result.stdout
    assert "coverage shortfall" not in result.stdout


def test_a_gold_case_naming_a_page_outside_the_content_root_still_fails(redirected_vault):
    """Validation is against the redirected index, so an id that is not in it
    must be rejected — otherwise "valid" above could just mean "checked nothing"."""
    run_cli(redirected_vault, "llm_wiki.build_index")
    (redirected_vault / "sub" / "cases.json").write_text(
        '[{"q": "what is decoy", "expect": ["decoy"], "layer": "domain", '
        '"domain": "research", "category": "direct", "difficulty": "easy", '
        '"split": "test"}]',
        encoding="utf-8")

    result = run_cli(redirected_vault, "llm_wiki.evaluation.eval", "--validate-only")

    assert result.returncode == 1
    assert "does not exist in index.yaml" in result.stdout


# --------------------------------------------------------------------------
# the common case must not have changed
# --------------------------------------------------------------------------


def test_a_vault_with_no_redirect_keeps_index_at_its_root(vault):
    """The overwhelmingly common shape: no `[vault] root`, so the config root and
    the content root are the same directory and nothing moves."""
    (vault / "domain" / "research" / "one.md").write_text(page(id="one"), encoding="utf-8")

    result = run_cli(vault, "llm_wiki.build_index")

    assert result.returncode == 0, result.stdout + result.stderr
    assert (vault / "index.yaml").exists()
    entries = yaml.safe_load((vault / "index.yaml").read_text(encoding="utf-8"))["entries"]
    assert [e["path"] for e in entries] == ["domain/research/one.md"]
