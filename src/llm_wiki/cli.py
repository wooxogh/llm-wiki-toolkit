"""First-run-aware console entry points with lazy imports."""
from __future__ import annotations

import importlib
import sys
from collections.abc import Callable

from llm_wiki.initialize import InitializationRequired, ensure_initialized


def _run(module_name: str) -> int:
    try:
        ensure_initialized()
    except InitializationRequired as exc:
        print(exc, file=sys.stderr)
        return 2
    main: Callable[[], int | None] = getattr(importlib.import_module(module_name), "main")
    result = main()
    return int(result or 0)


def index_main() -> int:
    return _run("llm_wiki.build_index")


def health_main() -> int:
    return _run("llm_wiki.wiki_health")


def recall_main() -> int:
    return _run("llm_wiki.retrieval.recall")


def embed_main() -> int:
    return _run("llm_wiki.retrieval.embed_index")


def eval_main() -> int:
    return _run("llm_wiki.evaluation.eval")


def gate_main() -> int:
    return _run("llm_wiki.evaluation.eval_gate")


def lint_main() -> int:
    return _run("llm_wiki.hygiene.claim_lint")


def concepts_main() -> int:
    return _run("llm_wiki.v2.concepts_cli")


def net_main() -> int:
    return _run("llm_wiki.v2.net_cli")


def review_main() -> int:
    return _run("llm_wiki.v2.review_cli")
