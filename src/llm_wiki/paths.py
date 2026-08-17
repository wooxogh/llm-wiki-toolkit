"""Vault-root discovery and canonical content-path enumeration.

Every tool used to re-derive the vault root and re-declare its content
directories independently. That is fine until a test wants to point a tool at a
synthetic vault — then the module global wins and the test silently reads the
committed vault instead. One shared, *parameterized* helper removes that whole
class of accident.

Two roots, not one
------------------
`VAULT_ROOT` is the **config** root: the directory holding `wiki.toml`. It is
the handle every tool passes around as `vault`, and the only value
`config.load()` accepts.

`CONTENT_ROOT` (`content_root(vault)`) is where the pages actually live. The two
are the same directory unless the vault sets `[vault] root`, which redirects
content into a subdirectory. Everything derived from the pages — `index.yaml`,
`.embeddings/`, the generated reports, the gold set, the baselines — lives
beside the content, under the content root. Only `wiki.toml` lives in the config
root. Use `index_path()` / `embeddings_dir()` rather than joining a name onto a
`vault` argument, which would silently put the artifact in the wrong place under
a redirect.

Pure stdlib on purpose: this is imported by the lightweight CI path, which must
never pull in numpy, torch, or sentence-transformers.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from llm_wiki import config

_CONFIG = config.load()
VAULT_ROOT = _CONFIG.config_dir
CONTENT_ROOT = _CONFIG.root


def vault_root() -> Path:
    """The vault this process is operating on (the config root)."""
    return VAULT_ROOT


def content_root(vault: Path | None = None) -> Path:
    """Where `vault`'s pages and derived artifacts live.

    Equal to `vault` itself unless that vault's wiki.toml sets `[vault] root`.
    Read from the *target* vault's config, so a tool pointed at a synthetic or
    example vault honours its redirect rather than this process's.
    """
    if vault is None:
        return CONTENT_ROOT
    return config.load(vault).root


def index_path(vault: Path | None = None) -> Path:
    """`index.yaml` for `vault` — in the content root, beside the pages it indexes.

    Single-sourced because three tools must agree on it byte for byte:
    `build_index` writes it, `wiki_health` byte-compares it, and `sync_cache`
    and `eval` read it. Under `[vault] root` they would otherwise disagree.
    """
    return content_root(vault) / "index.yaml"


def embeddings_dir(vault: Path | None = None) -> Path:
    """`.embeddings/` for `vault` — in the content root, beside the pages."""
    return content_root(vault) / ".embeddings"


def content_dirs(vault: Path | None = None) -> list[Path]:
    """Existing canonical content directories under the vault's effective content root.

    Read from that vault's own wiki.toml, not from this process's, so a tool
    pointed at a synthetic or example vault honours *its* layout. The effective
    content root is determined by [vault] root in wiki.toml if present.
    """
    cfg = config.load(Path(vault) if vault is not None else VAULT_ROOT)
    return [cfg.root / d for d in cfg.content_dirs if (cfg.root / d).exists()]


def content_paths(vault: Path | None = None) -> list[Path]:
    """Every canonical Markdown page under `vault`, deterministically ordered.

    Order is (declared dir order, then sorted path) so that any artifact built
    from this list is byte-stable across machines and filesystems.
    """
    paths: list[Path] = []
    for base in content_dirs(vault):
        paths.extend(sorted(base.rglob("*.md")))
    return paths


def relative(path: Path, vault: Path | None = None) -> str:
    """POSIX-style vault-relative path, as stored in index.yaml and meta.json."""
    return Path(path).relative_to(content_root(vault)).as_posix()


def page_hash(raw_text: str) -> str:
    """Per-page content identity used by the incremental embedding store.

    Single-sourced here because two independent tools depend on it agreeing:
    embed_index.py *writes* these hashes into .embeddings/pages.json, and
    wiki_health.py *re-derives* them to detect a store that has silently gone
    stale. If the two ever drifted apart, health would report phantom staleness
    (or worse, miss real staleness) forever.
    """
    return hashlib.sha1(raw_text.encode("utf-8")).hexdigest()
