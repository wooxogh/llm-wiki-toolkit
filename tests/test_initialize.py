from __future__ import annotations

import io
from pathlib import Path

import pytest

from llm_wiki.initialize import (InitializationRequired, ensure_initialized,
                                 initialize, main, resolve_target)


def test_interactive_first_run_creates_minimal_codex_config(tmp_path):
    output = io.StringIO()
    answers = iter(["1", "cpu"])

    path = ensure_initialized([], environ={}, cwd=tmp_path, interactive=True,
                              input_fn=lambda _: next(answers), output=output)

    assert path == tmp_path / "wiki.toml"
    assert path.read_text(encoding="utf-8") == (
        '[vault]\ncontent_dirs = ["."]\n\n'
        '[v2]\nenabled = true\nagent = "codex"\n'
        'embed_backend = "qwen"\nembed_device = "cpu"\n'
    )
    assert "first-run" in output.getvalue()
    assert "Agent: codex" in output.getvalue()


def test_interactive_first_run_accepts_claude(tmp_path):
    answers = iter(["claude", "cuda"])
    path = ensure_initialized([], environ={}, cwd=tmp_path, interactive=True,
                              input_fn=lambda _: next(answers), output=io.StringIO())
    assert 'agent = "claude"' in path.read_text(encoding="utf-8")
    assert 'embed_device = "cuda"' in path.read_text(encoding="utf-8")


def test_empty_selection_defaults_to_codex(tmp_path):
    path = initialize(tmp_path, input_fn=lambda _: "", output=io.StringIO())
    assert 'agent = "codex"' in path.read_text(encoding="utf-8")


def test_existing_config_is_never_replaced(tmp_path):
    path = tmp_path / "wiki.toml"
    path.write_text('[v2]\nagent = "claude"\n', encoding="utf-8")

    assert initialize(tmp_path, "codex", output=io.StringIO()) == path
    assert path.read_text(encoding="utf-8") == '[v2]\nagent = "claude"\n'


def test_noninteractive_first_run_requires_explicit_init(tmp_path):
    with pytest.raises(InitializationRequired, match="wiki-init.*--agent codex"):
        ensure_initialized([], environ={}, cwd=tmp_path, interactive=False)
    assert not (tmp_path / "wiki.toml").exists()


def test_help_does_not_initialize(tmp_path):
    path = ensure_initialized(["--help"], environ={}, cwd=tmp_path, interactive=False)
    assert path == tmp_path / "wiki.toml"
    assert not path.exists()


def test_explicit_vault_wins_over_environment(tmp_path):
    explicit = tmp_path / "explicit"
    from_env = tmp_path / "env"
    assert resolve_target(["--vault", str(explicit)], {"WIKI_VAULT": str(from_env)}) == explicit
    assert resolve_target([f"--vault={explicit}"], {"WIKI_VAULT": str(from_env)}) == explicit


def test_wiki_init_noninteractive_agent_option(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["wiki-init", "--vault", str(tmp_path),
                                     "--agent", "claude", "--device", "mps"])
    assert main() == 0
    text = (tmp_path / "wiki.toml").read_text(encoding="utf-8")
    assert 'agent = "claude"' in text
    assert 'embed_backend = "qwen"' in text
    assert 'embed_device = "mps"' in text
