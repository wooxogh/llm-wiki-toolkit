"""Vault-root discovery and content enumeration honour per-vault config."""
from __future__ import annotations

from llm_wiki import paths


def make(vault, rel: str, text: str = "x"):
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_content_paths_are_deterministic_in_declared_dir_order(tmp_path):
    make(tmp_path, "patterns/b.md")
    make(tmp_path, "domain/a.md")
    make(tmp_path, "domain/z.md")
    rels = [paths.relative(p, tmp_path) for p in paths.content_paths(tmp_path)]
    assert rels == ["domain/a.md", "domain/z.md", "patterns/b.md"]


def test_content_dirs_follow_the_vaults_own_config(tmp_path):
    (tmp_path / "wiki.toml").write_text('[vault]\ncontent_dirs = ["notes"]\n', encoding="utf-8")
    make(tmp_path, "notes/a.md")
    make(tmp_path, "domain/ignored.md")
    assert [paths.relative(p, tmp_path) for p in paths.content_paths(tmp_path)] == ["notes/a.md"]


def test_missing_content_dirs_are_skipped_not_an_error(tmp_path):
    make(tmp_path, "domain/a.md")
    assert [d.name for d in paths.content_dirs(tmp_path)] == ["domain"]


def test_page_hash_is_stable_and_content_addressed():
    assert paths.page_hash("abc") == paths.page_hash("abc")
    assert paths.page_hash("abc") != paths.page_hash("abd")


def test_relative_is_posix_style(tmp_path):
    p = make(tmp_path, "domain/sub/a.md")
    assert paths.relative(p, tmp_path) == "domain/sub/a.md"


def test_content_paths_with_vault_root_redirect(tmp_path):
    (tmp_path / "wiki.toml").write_text('[vault]\nroot = "sub"\n', encoding="utf-8")
    make(tmp_path, "sub/domain/a.md")
    make(tmp_path, "domain/ignored.md")
    pages = paths.content_paths(tmp_path)
    rels = [paths.relative(p, tmp_path) for p in pages]
    assert rels == ["domain/a.md"]


def test_relative_uses_effective_root_from_vault_redirect(tmp_path):
    (tmp_path / "wiki.toml").write_text('[vault]\nroot = "sub"\n', encoding="utf-8")
    p = make(tmp_path, "sub/domain/a.md")
    assert paths.relative(p, tmp_path) == "domain/a.md"


def test_content_root_follows_the_redirect_and_index_sits_beside_the_pages(tmp_path):
    """`index.yaml` and `.embeddings/` belong to the content, so they live in the
    content root. Only `wiki.toml` stays in the config root."""
    (tmp_path / "wiki.toml").write_text('[vault]\nroot = "sub"\n', encoding="utf-8")

    assert paths.content_root(tmp_path) == (tmp_path / "sub").resolve()
    assert paths.index_path(tmp_path) == (tmp_path / "sub" / "index.yaml").resolve()
    assert paths.embeddings_dir(tmp_path) == (tmp_path / "sub" / ".embeddings").resolve()


def test_content_root_is_the_vault_itself_without_a_redirect(tmp_path):
    """No `[vault] root` (no wiki.toml at all here): every artifact stays exactly
    where it is today. This is the shape the whole existing suite describes."""
    assert paths.content_root(tmp_path) == tmp_path.resolve()
    assert paths.index_path(tmp_path) == (tmp_path / "index.yaml").resolve()
    assert paths.embeddings_dir(tmp_path) == (tmp_path / ".embeddings").resolve()
