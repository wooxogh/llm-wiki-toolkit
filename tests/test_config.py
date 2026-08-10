"""wiki.toml loading, defaults, validation, and root discovery."""
from __future__ import annotations

from pathlib import Path

import pytest

from llm_wiki import config


def write_toml(root: Path, text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    p = root / config.CONFIG_FILENAME
    p.write_text(text, encoding="utf-8")
    return p


def test_defaults_apply_when_no_config_file(tmp_path):
    cfg = config.load(tmp_path)
    # .resolve() on purpose: on macOS pytest's tmp_path sits under a symlinked
    # /var, and load() resolves, so a bare comparison would fail spuriously.
    assert cfg.root == tmp_path.resolve()
    assert cfg.content_dirs == config.DEFAULT_CONTENT_DIRS
    assert cfg.layers == frozenset(config.DEFAULT_LAYERS)
    assert cfg.required == config.DEFAULT_REQUIRED
    assert cfg.lint_packs == config.DEFAULT_LINT_PACKS


def test_empty_domains_by_default_means_no_domain_validation(tmp_path):
    assert config.load(tmp_path).domains == frozenset()


def test_declared_domains_are_loaded(tmp_path):
    write_toml(tmp_path, '[schema]\ndomains = ["research", "tooling"]\n')
    assert config.load(tmp_path).domains == {"research", "tooling"}


def test_content_dirs_override(tmp_path):
    write_toml(tmp_path, '[vault]\ncontent_dirs = ["notes", "refs"]\n')
    assert config.load(tmp_path).content_dirs == ("notes", "refs")


def test_minimums_merge_over_defaults(tmp_path):
    write_toml(tmp_path, "[eval.minimums]\ntotal = 10\n")
    mins = config.load(tmp_path).minimums
    assert mins["total"] == 10
    # untouched axes keep their defaults
    assert mins["category"] == config.DEFAULT_MINIMUMS["category"]


def test_overriding_one_table_axis_replaces_it_entirely(tmp_path):
    write_toml(tmp_path, "[eval.minimums]\nlayer = { pattern = 30 }\n")
    mins = config.load(tmp_path).minimums
    # the whole `layer` table is replaced, not merged key-by-key
    assert mins["layer"] == {"pattern": 30}
    # other table axes are untouched and keep their defaults
    assert mins["category"] == config.DEFAULT_MINIMUMS["category"]


def test_vault_root_is_resolved_relative_to_the_config_file(tmp_path):
    write_toml(tmp_path, '[vault]\nroot = "sub"\n')
    assert config.load(tmp_path).root == (tmp_path / "sub").resolve()


def test_config_dir_stays_where_wiki_toml_is_when_root_redirects(tmp_path):
    """The two roots must be separately addressable. Collapsing them is what let
    `config.load(cfg.root)` find no wiki.toml and silently return the built-in
    defaults while every gate still reported success."""
    write_toml(tmp_path, '[vault]\nroot = "sub"\ncontent_dirs = ["notes"]\n')

    cfg = config.load(tmp_path)

    assert cfg.config_dir == tmp_path.resolve()
    assert cfg.root == (tmp_path / "sub").resolve()


def test_reloading_from_config_dir_round_trips_the_whole_config(tmp_path):
    """`config.load(cfg.config_dir)` must reproduce `cfg`, because that is what
    every consumer does with the `vault` handle it was given. Reloading from
    `cfg.root` instead is the bug: it reads no file at all."""
    write_toml(tmp_path, '[vault]\nroot = "sub"\ncontent_dirs = ["notes"]\n'
                         '[lint]\npacks = ["ko"]\n')

    cfg = config.load(tmp_path)

    assert config.load(cfg.config_dir) == cfg
    assert config.load(cfg.root).content_dirs == config.DEFAULT_CONTENT_DIRS  # the trap
    assert config.load(cfg.root).lint_packs == config.DEFAULT_LINT_PACKS


def test_config_dir_equals_root_with_no_redirect(tmp_path):
    """The common case: no `[vault] root`, so nothing is split and the two roots
    are the same directory."""
    write_toml(tmp_path, '[lint]\npacks = ["ko"]\n')

    cfg = config.load(tmp_path)

    assert cfg.config_dir == cfg.root == tmp_path.resolve()


def test_vault_root_wrong_type_is_rejected(tmp_path):
    write_toml(tmp_path, "[vault]\nroot = 5\n")
    with pytest.raises(config.ConfigError, match="root"):
        config.load(tmp_path)


def test_layers_may_not_be_emptied(tmp_path):
    write_toml(tmp_path, "[schema]\nlayers = []\n")
    with pytest.raises(config.ConfigError, match="layers"):
        config.load(tmp_path)


def test_required_may_not_drop_id(tmp_path):
    write_toml(tmp_path, '[schema]\nrequired = ["layer", "summary"]\n')
    with pytest.raises(config.ConfigError, match="id"):
        config.load(tmp_path)


def test_wrong_type_is_rejected(tmp_path):
    write_toml(tmp_path, '[vault]\ncontent_dirs = "domain"\n')
    with pytest.raises(config.ConfigError, match="content_dirs"):
        config.load(tmp_path)


def test_unknown_key_is_rejected(tmp_path):
    write_toml(tmp_path, "[schema]\nlayerz = []\n")
    with pytest.raises(config.ConfigError, match="layerz"):
        config.load(tmp_path)


def test_malformed_toml_is_reported_with_path(tmp_path):
    write_toml(tmp_path, "[schema\n")
    with pytest.raises(config.ConfigError, match=config.CONFIG_FILENAME):
        config.load(tmp_path)


def test_find_root_prefers_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("WIKI_VAULT", str(tmp_path))
    assert config.find_root(Path("/nowhere")) == tmp_path.resolve()


def test_find_root_walks_up_to_the_config_file(tmp_path, monkeypatch):
    monkeypatch.delenv("WIKI_VAULT", raising=False)
    write_toml(tmp_path, "")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert config.find_root(nested) == tmp_path.resolve()


def test_find_root_falls_back_to_start_when_nothing_is_found(tmp_path, monkeypatch):
    monkeypatch.delenv("WIKI_VAULT", raising=False)
    nested = tmp_path / "a"
    nested.mkdir()
    assert config.find_root(nested) == nested.resolve()
