"""Canonical temporary-vault fixture builder.

Every test in this suite runs against a *synthetic* vault under tmp_path — never
the committed one. That keeps the suite fast, hermetic, and safe to run in CI
where `.embeddings/` (gitignored) does not exist.

Hard constraint (see CONTRIBUTING.md): nothing here may import torch,
sentence-transformers, or reach the resident embedding server. Dense scores are
always injected by the caller.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

# The package is installed editable (`pip install -e .`), so llm_wiki is already
# importable. The repo root still needs to be on sys.path for `integrations/`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def valid_frontmatter(**overrides) -> dict:
    """A minimal frontmatter dict that passes build_index.validate()."""
    fm = {
        "id": "one",
        "layer": "domain",
        "domain": "research",
        "projects": ["project-a"],
        "tags": ["retrieval"],
        "confidence": "confirmed",
        "status": "active",
        "updated": "2026-01-15",
        "summary": "measured summary",
    }
    fm.update(overrides)
    return fm


def page_from(fm: dict, body: str = "body text\n") -> str:
    dumped = yaml.safe_dump(fm, allow_unicode=True, sort_keys=False, width=1000)
    return f"---\n{dumped}---\n\n{body}"


def page(body: str = "body text\n", **overrides) -> str:
    return page_from(valid_frontmatter(**overrides), body)


def write_page(vault: Path, rel: str, text: str) -> Path:
    path = vault / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


CURRENT_IDENTITY = None  # resolved lazily; see _identity()


def _identity() -> str:
    """The store identity the current code would write.

    Fixtures default to this so a test only trips `embedding-identity-stale` when
    it deliberately writes a different one.
    """
    global CURRENT_IDENTITY
    if CURRENT_IDENTITY is None:
        from llm_wiki import wiki_health

        CURRENT_IDENTITY = wiki_health._expected_identity() or "test-model|ctx-v3|meta-v2"
    return CURRENT_IDENTITY


def write_embedding_fixture(
    vault: Path,
    page_hashes: dict,
    meta: list,
    vector_rows: int,
    model: str = None,
    dim: int = 4,
) -> Path:
    """Write a .embeddings/ store literally, without embedding anything.

    vector_rows is decoupled from len(meta) on purpose so tests can construct
    the row-count-mismatch drift class directly.
    """
    import numpy as np

    out = vault / ".embeddings"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", np.zeros((vector_rows, dim), dtype=np.float32))
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    (out / "pages.json").write_text(json.dumps(page_hashes, ensure_ascii=False), encoding="utf-8")
    (out / "model.txt").write_text(model if model is not None else _identity(), encoding="utf-8")
    return out


def write_index_and_embedding_fixture(
    vault: Path,
    page_hashes: dict,
    meta: list,
    vector_rows: int,
    model: str = None,
) -> None:
    """Fresh index.yaml + a literal embedding store, so a health test isolates
    exactly the embedding drift class it is about."""
    from llm_wiki import build_index

    build_index.write_index(vault, vault / "index.yaml")
    write_embedding_fixture(vault, page_hashes, meta, vector_rows, model=model)


def current_page_hashes(vault: Path) -> dict:
    """The hashes a correct embedding store would carry for this vault."""
    from llm_wiki import paths

    return {
        (build_fm(p) or {}).get("id", p.stem): paths.page_hash(p.read_text(encoding="utf-8"))
        for p in paths.content_paths(vault)
    }


def build_fm(path: Path) -> dict:
    from llm_wiki import build_index

    return build_index.parse_frontmatter(path) or {}


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    """An empty but structurally valid vault root."""
    for d in ("domain/research", "domain/tooling", "patterns", "entities", "raw"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path
