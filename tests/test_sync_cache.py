"""Claude/Codex managed-block preservation and target routing.

The cache is a *pointer* layer: sync must be able to rewrite the block it owns
without ever touching the personal memories that live beside it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from integrations.agent_memory import sync_cache
from integrations.agent_memory.sync_cache import END, START, memory_dir, sync_project

TARGETS = ("claude", "codex")


def target_memory(home: Path, project: Path, target: str) -> Path:
    slug = str(project).replace("\\", "-").replace("/", "-")
    return home / f".{target}" / "projects" / slug / "memory"


def promoted_entry(_id="canonical-id", **over):
    entry = {"id": _id, "layer": "domain", "domain": "research",
             "path": f"domain/research/{_id}.md", "summary": "measured summary",
             "cache_stub": f"project_{_id.replace('-', '_')}.md"}
    entry.update(over)
    return entry


# --------------------------------------------------------------------------
# target routing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS)
def test_memory_dir_uses_a_lowercase_dot_directory_per_target(tmp_path, target):
    project = tmp_path / "dev" / "example-project"

    resolved = memory_dir(project, target=target, home=tmp_path)

    assert resolved == target_memory(tmp_path, project, target)
    assert f".{target}" in resolved.parts


def test_the_default_target_is_claude_for_backward_compatibility(tmp_path):
    project = tmp_path / "dev" / "example-project"

    assert memory_dir(project, home=tmp_path) == target_memory(tmp_path, project, "claude")


def test_an_unknown_target_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="target"):
        memory_dir(tmp_path / "p", target="cursor", home=tmp_path)


def test_the_slug_is_the_absolute_path_with_slashes_replaced(tmp_path):
    resolved = memory_dir(Path("/opt/example/dev/proj"), target="codex", home=tmp_path)

    assert resolved.parent.name == "-opt-example-dev-proj"


# --------------------------------------------------------------------------
# managed block
# --------------------------------------------------------------------------


def test_both_targets_preserve_personal_content_outside_managed_block(tmp_path):
    project = tmp_path / "dev" / "example-project"
    for target in TARGETS:
        memory = target_memory(tmp_path, project, target)
        memory.mkdir(parents=True)
        (memory / "MEMORY.md").write_text("personal preference\n", encoding="utf-8")

    sync_project(project, target="both", home=tmp_path, entries=[promoted_entry()])

    for target in TARGETS:
        text = (target_memory(tmp_path, project, target) / "MEMORY.md").read_text()
        assert "personal preference" in text
        assert START in text
        # exactly one pointer entry — repeated syncs must not accumulate copies
        assert text.count("[[canonical-id]]") == 1


def test_repeated_sync_is_idempotent(tmp_path):
    project = tmp_path / "dev" / "example-project"
    entries = [promoted_entry()]

    sync_project(project, target="claude", home=tmp_path, entries=entries)
    first = (target_memory(tmp_path, project, "claude") / "MEMORY.md").read_text()
    sync_project(project, target="claude", home=tmp_path, entries=entries)
    second = (target_memory(tmp_path, project, "claude") / "MEMORY.md").read_text()

    assert first == second


def test_a_removed_page_disappears_from_the_managed_block(tmp_path):
    project = tmp_path / "dev" / "example-project"
    sync_project(project, target="claude", home=tmp_path,
                 entries=[promoted_entry("gone"), promoted_entry("stays")])

    sync_project(project, target="claude", home=tmp_path, entries=[promoted_entry("stays")])

    text = (target_memory(tmp_path, project, "claude") / "MEMORY.md").read_text()
    assert "stays" in text
    assert "gone" not in text


def test_sync_creates_the_memory_directory_when_absent(tmp_path):
    project = tmp_path / "dev" / "example-project"

    sync_project(project, target="codex", home=tmp_path, entries=[promoted_entry()])

    assert (target_memory(tmp_path, project, "codex") / "MEMORY.md").exists()


def test_only_the_requested_target_is_written(tmp_path):
    project = tmp_path / "dev" / "example-project"

    sync_project(project, target="codex", home=tmp_path, entries=[promoted_entry()])

    assert (target_memory(tmp_path, project, "codex") / "MEMORY.md").exists()
    assert not target_memory(tmp_path, project, "claude").exists()


def test_the_managed_block_is_delimited_by_both_markers(tmp_path):
    project = tmp_path / "dev" / "example-project"

    sync_project(project, target="claude", home=tmp_path, entries=[promoted_entry()])
    text = (target_memory(tmp_path, project, "claude") / "MEMORY.md").read_text()

    assert text.index(START) < text.index("canonical-id") < text.index(END)


# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------


def test_promoted_pages_get_a_pointer_stub_per_target(tmp_path):
    project = tmp_path / "dev" / "example-project"

    sync_project(project, target="both", home=tmp_path, entries=[promoted_entry()])

    for target in TARGETS:
        stub = target_memory(tmp_path, project, target) / "project_canonical_id.md"
        assert "type: pointer" in stub.read_text(encoding="utf-8")


def test_a_page_without_a_cache_stub_is_not_written_as_a_file(tmp_path):
    project = tmp_path / "dev" / "example-project"
    entry = promoted_entry()
    del entry["cache_stub"]

    sync_project(project, target="claude", home=tmp_path, entries=[entry])

    memory = target_memory(tmp_path, project, "claude")
    assert [p.name for p in memory.iterdir()] == ["MEMORY.md"]


def test_dry_run_reports_targets_without_writing(tmp_path, capsys):
    project = tmp_path / "dev" / "example-project"

    sync_project(project, target="both", home=tmp_path,
                 entries=[promoted_entry()], dry=True)

    printed = capsys.readouterr().out
    assert str(target_memory(tmp_path, project, "claude")) in printed
    assert str(target_memory(tmp_path, project, "codex")) in printed
    assert not target_memory(tmp_path, project, "claude").exists()
    assert not target_memory(tmp_path, project, "codex").exists()


def test_entries_are_filtered_to_the_project_when_read_from_the_index(tmp_path):
    entries = [
        {"id": "mine", "projects": ["example-project"], "layer": "domain",
         "domain": "research", "path": "domain/research/mine.md", "summary": "s"},
        {"id": "theirs", "projects": ["other-project"], "layer": "domain",
         "domain": "tooling", "path": "domain/tooling/theirs.md", "summary": "s"},
    ]

    selected = sync_cache.entries_for_project(entries, "example-project")

    assert [e["id"] for e in selected] == ["mine"]
