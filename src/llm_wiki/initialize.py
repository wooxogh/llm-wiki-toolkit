"""First-run vault initialization for installed llm-wiki commands."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence, TextIO

CONFIG_FILENAME = "wiki.toml"
AGENTS = ("codex", "claude")
EMBED_DEVICES = ("auto", "cuda", "mps", "cpu")


class InitializationRequired(RuntimeError):
    """A non-interactive command needs an explicitly initialized vault."""


def resolve_target(argv: Sequence[str] | None = None,
                   environ: Mapping[str, str] | None = None,
                   cwd: Path | None = None) -> Path:
    """Resolve an explicit --vault, WIKI_VAULT, nearest config, or cwd."""
    argv = tuple(sys.argv[1:] if argv is None else argv)
    environ = os.environ if environ is None else environ
    explicit = _option_value(argv, "--vault")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if environ.get("WIKI_VAULT"):
        return Path(environ["WIKI_VAULT"]).expanduser().resolve()
    current = (cwd or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / CONFIG_FILENAME).is_file():
            return candidate
    return current


def initialize(vault: Path, agent: str | None = None, *,
               embed_device: str | None = None,
               input_fn: Callable[[str], str] | None = None,
               output: TextIO | None = None) -> Path:
    """Create the minimal v2 wiki.toml without replacing an existing config."""
    output = output or sys.stdout
    target = Path(vault).expanduser().resolve()
    path = target / CONFIG_FILENAME
    if path.exists():
        print(f"Already initialized: {path}", file=output)
        return path
    selected = _normalize_agent(agent) if agent else _select_agent(input_fn, output)
    device = _normalize_device(embed_device) if embed_device else (
        "auto" if agent else _select_device(input_fn, output)
    )
    target.mkdir(parents=True, exist_ok=True)
    contents = (
        '[vault]\n'
        'content_dirs = ["."]\n\n'
        '[v2]\n'
        'enabled = true\n'
        f'agent = "{selected}"\n'
        'embed_backend = "qwen"\n'
        f'embed_device = "{device}"\n'
    )
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(contents)
    except FileExistsError:
        print(f"Already initialized: {path}", file=output)
        return path
    print(f"Initialized llm-wiki vault: {target}", file=output)
    print(f"Agent: {selected}", file=output)
    print(f"Embedding: qwen on {device}", file=output)
    if shutil.which(selected) is None:
        print(f"Warning: `{selected}` CLI was not found on PATH. Install/login before AI-backed v2 commands.",
              file=output)
    return path


def ensure_initialized(argv: Sequence[str] | None = None, *,
                       environ: Mapping[str, str] | None = None,
                       cwd: Path | None = None,
                       interactive: bool | None = None,
                       input_fn: Callable[[str], str] | None = None,
                       output: TextIO | None = None) -> Path:
    """Initialize on the first interactive command, or give automation a fix."""
    argv = tuple(sys.argv[1:] if argv is None else argv)
    target = resolve_target(argv, environ, cwd)
    path = target / CONFIG_FILENAME
    if path.is_file() or any(arg in {"-h", "--help", "--version"} for arg in argv):
        return path
    output = output or sys.stdout
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive:
        raise InitializationRequired(
            f"No {CONFIG_FILENAME} found for vault: {target}\n"
            f"Run `wiki-init --vault \"{target}\" --agent codex` "
            "(or `--agent claude`) before this non-interactive command."
        )
    print("llm-wiki first-run setup", file=output)
    print(f"Vault: {target}", file=output)
    return initialize(target, input_fn=input_fn, output=output)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="wiki-init",
        description="Create a minimal v2 wiki.toml for a Markdown vault.",
    )
    parser.add_argument("--vault", type=Path, help="vault directory; defaults to WIKI_VAULT or cwd")
    parser.add_argument("--agent", choices=AGENTS, help="skip the interactive Codex/Claude menu")
    parser.add_argument("--device", choices=EMBED_DEVICES,
                        help="Qwen embedding device; defaults to auto with --agent")
    args = parser.parse_args()
    target = (args.vault.expanduser().resolve() if args.vault else resolve_target(()))
    if not args.agent and not (sys.stdin.isatty() and sys.stdout.isatty()):
        parser.error("non-interactive initialization requires --agent codex or --agent claude")
    initialize(target, args.agent, embed_device=args.device)
    return 0


def _select_agent(input_fn: Callable[[str], str] | None, output: TextIO) -> str:
    reader = input_fn or input
    print("AI agent를 선택하세요:", file=output)
    print("  1. Codex (권장)", file=output)
    print("  2. Claude", file=output)
    while True:
        answer = reader("선택 [1]: ").strip().lower()
        if answer in {"", "1", "codex"}:
            return "codex"
        if answer in {"2", "claude"}:
            return "claude"
        print("1 또는 2를 입력하세요.", file=output)


def _normalize_agent(value: str) -> str:
    agent = value.strip().lower()
    if agent not in AGENTS:
        raise ValueError("agent must be 'codex' or 'claude'")
    return agent


def _select_device(input_fn: Callable[[str], str] | None, output: TextIO) -> str:
    reader = input_fn or input
    recommended = _detected_device()
    print("Concept embedding 장치를 선택하세요:", file=output)
    print(f"  1. {recommended.upper()} (감지됨, 권장)", file=output)
    print("  2. CUDA (NVIDIA GPU)", file=output)
    print("  3. CPU", file=output)
    print("  4. MPS (Apple Silicon)", file=output)
    print("  5. AUTO", file=output)
    choices = {"": recommended, "1": recommended, "2": "cuda", "3": "cpu",
               "4": "mps", "5": "auto", "auto": "auto", "cuda": "cuda",
               "cpu": "cpu", "mps": "mps"}
    while True:
        answer = reader(f"선택 [1={recommended}]: ").strip().lower()
        if answer in choices:
            return choices[answer]
        print("1~5 또는 auto/cuda/cpu/mps를 입력하세요.", file=output)


def _detected_device() -> str:
    if shutil.which("nvidia-smi"):
        return "cuda"
    if sys.platform == "darwin":
        return "mps"
    return "cpu"


def _normalize_device(value: str) -> str:
    device = value.strip().lower()
    if device not in EMBED_DEVICES:
        raise ValueError("embed device must be auto, cuda, mps, or cpu")
    return device


def _option_value(argv: Sequence[str], option: str) -> str | None:
    for index, value in enumerate(argv):
        if value == option and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(option + "="):
            return value.split("=", 1)[1]
    return None
