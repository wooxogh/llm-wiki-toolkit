"""Vault configuration: built-in defaults plus an optional wiki.toml override.

Every value here used to be a module-level constant in whichever tool happened
to need it, which meant the tools only worked on the one vault they were written
for. Reading them from the vault root instead makes the vault, not the checkout,
the thing that carries its own shape.

TOML rather than YAML on purpose: this project's own frontmatter contract is
YAML, and an unquoted `#` there once silently truncated seven pages. Config is
small and adversarially edited by hand; the stricter format is worth it. It is
stdlib from 3.11 on, so this adds no dependency.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_FILENAME = "wiki.toml"

DEFAULT_CONTENT_DIRS = ("domain", "patterns", "entities", "raw")
DEFAULT_LAYERS = ("domain", "pattern", "entity", "raw")
DEFAULT_REQUIRED = ("id", "layer", "projects", "tags", "confidence", "status", "summary")
DEFAULT_LINT_PACKS = ("en",)
DEFAULT_MINIMUMS = {
    "total": 150,
    "recent_cases": 30,
    "layer": {"domain": 100, "pattern": 25, "entity": 5, "raw": 2},
    "domain": {},
    "category": {"ambiguous": 10, "negative": 8},
}

# `id` and `layer` are not optional: the whole vault addressing scheme (filename
# == id, [[id]] wikilinks, per-layer ranking) collapses without them.
_UNDROPPABLE = ("id", "layer")

_KNOWN = {
    "vault": {"root", "content_dirs"},
    "schema": {"layers", "domains", "required"},
    "lint": {"packs"},
    "eval": {"gold", "minimums"},
    "ingest": {"repos", "prompt_file"},
}


class ConfigError(Exception):
    """The vault's wiki.toml is unreadable or says something impossible."""


@dataclass(frozen=True)
class Config:
    """Where the vault's configuration was found, and what shape it declares.

    `config_dir` and `root` are two fields on purpose. `config_dir` is the
    directory `wiki.toml` was read from; `root` is the *content* root it points
    at, which is the same directory unless `[vault] root` redirects it.

    Consumers must not confuse the two. `config.load(cfg.config_dir)`
    round-trips; `config.load(cfg.root)` under a redirect finds no `wiki.toml`
    in the subdirectory and silently falls back to the built-in defaults —
    discarding `content_dirs`, `layers`, `required`, lint packs, gold, and
    `[eval.minimums]` while every gate still reports success. So: config
    lookups resolve against `config_dir`; content enumeration, vault-relative
    paths, and generated artifacts resolve against `root`.
    """

    config_dir: Path
    root: Path
    content_dirs: tuple
    layers: frozenset
    domains: frozenset
    required: tuple
    lint_packs: tuple
    minimums: dict
    gold: str
    ingest_repos: tuple
    ingest_prompt_file: str | None


def find_root(start: Path | None = None) -> Path:
    """Locate the vault root: WIKI_VAULT, else the nearest ancestor with wiki.toml.

    Falls back to `start` itself so a vault with no config file still works —
    the defaults are a complete configuration, not a partial one.
    """
    env = os.environ.get("WIKI_VAULT")
    if env:
        return Path(env).expanduser().resolve()
    cur = (start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / CONFIG_FILENAME).is_file():
            return candidate
    return cur


def _table(raw: dict, name: str) -> dict:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a table, got {type(value).__name__}")
    unknown = set(value) - _KNOWN[name]
    if unknown:
        raise ConfigError(f"[{name}] has unknown key(s): {', '.join(sorted(unknown))}")
    return value


def _str_tuple(value, where: str) -> tuple:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{where} must be a list of strings")
    return tuple(value)


def _merge_minimums(override: dict) -> dict:
    """Layer a `[eval.minimums]` override onto the defaults, axis by axis.

    A scalar axis (e.g. `total`) is overridden value-for-value. A table axis
    (e.g. `layer`, `category`) is *replaced entirely*, not merged key-by-key:
    writing `layer = { pattern = 30 }` drops the default `domain`/`entity`/`raw`
    floors for that axis rather than keeping them alongside the new `pattern`
    value. This is intentional — declaring an axis in wiki.toml means declaring
    all of it, so the vault owner states a complete floor for that axis rather
    than silently inheriting entries they never wrote down.
    """
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_MINIMUMS.items()}
    for key, value in override.items():
        if key not in merged:
            raise ConfigError(f"[eval.minimums] has unknown axis: {key}")
        if isinstance(merged[key], dict):
            if not isinstance(value, dict):
                raise ConfigError(f"[eval.minimums] {key} must be a table")
            merged[key] = dict(value)
        else:
            if not isinstance(value, int):
                raise ConfigError(f"[eval.minimums] {key} must be an integer")
            merged[key] = value
    return merged


def load(root: Path | None = None) -> Config:
    """Read `<root>/wiki.toml` if present, layered over the built-in defaults.

    `root` is the *config* directory — the one holding `wiki.toml`. Pass
    `Config.config_dir`, never `Config.root`, when reloading a vault's config.
    """
    config_dir = Path(root).resolve() if root is not None else find_root()
    path = config_dir / CONFIG_FILENAME
    raw: dict = {}
    if path.is_file():
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path.name} is not valid TOML: {exc}") from exc
        unknown = set(raw) - set(_KNOWN)
        if unknown:
            raise ConfigError(f"{path.name} has unknown table(s): {', '.join(sorted(unknown))}")

    vault, schema = _table(raw, "vault"), _table(raw, "schema")
    lint, evaluation, ingest = _table(raw, "lint"), _table(raw, "eval"), _table(raw, "ingest")

    content_dirs = (_str_tuple(vault["content_dirs"], "[vault] content_dirs")
                    if "content_dirs" in vault else DEFAULT_CONTENT_DIRS)
    if not content_dirs:
        raise ConfigError("[vault] content_dirs must not be empty")

    layers = (_str_tuple(schema["layers"], "[schema] layers")
              if "layers" in schema else DEFAULT_LAYERS)
    if not layers:
        raise ConfigError("[schema] layers must not be empty")

    domains = (_str_tuple(schema["domains"], "[schema] domains")
               if "domains" in schema else ())

    required = (_str_tuple(schema["required"], "[schema] required")
                if "required" in schema else DEFAULT_REQUIRED)
    missing = [f for f in _UNDROPPABLE if f not in required]
    if missing:
        raise ConfigError(f"[schema] required must include: {', '.join(missing)}")

    packs = (_str_tuple(lint["packs"], "[lint] packs")
             if "packs" in lint else DEFAULT_LINT_PACKS)

    minimums = _merge_minimums(evaluation.get("minimums", {}))
    gold = evaluation.get("gold", "eval_gold.json")
    if not isinstance(gold, str):
        raise ConfigError("[eval] gold must be a string")

    repos = _str_tuple(ingest["repos"], "[ingest] repos") if "repos" in ingest else ()
    prompt_file = ingest.get("prompt_file")
    if prompt_file is not None and not isinstance(prompt_file, str):
        raise ConfigError("[ingest] prompt_file must be a string")

    content_root = config_dir
    if "root" in vault:
        if not isinstance(vault["root"], str):
            raise ConfigError("[vault] root must be a string")
        content_root = (config_dir / vault["root"]).resolve()

    return Config(
        config_dir=config_dir,
        root=content_root,
        content_dirs=content_dirs,
        layers=frozenset(layers),
        domains=frozenset(domains),
        required=required,
        lint_packs=packs,
        minimums=minimums,
        gold=gold,
        ingest_repos=repos,
        ingest_prompt_file=prompt_file,
    )
