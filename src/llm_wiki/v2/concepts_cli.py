"""CLI for v2 concept artifacts."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from llm_wiki.progress import PhaseProgress
from llm_wiki.v2.concept_index import build_index
from llm_wiki.v2.concept_store import build_concepts
from llm_wiki import config
from llm_wiki.v2.schemas import DEFAULT_CHUNK_TARGET_CHARS


def main() -> int:
    ap = argparse.ArgumentParser(prog="wiki-concepts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build")
    build.add_argument("--vault", type=Path)
    build.add_argument("--chunk-target", type=int, default=None)
    build.add_argument("--changed", action="store_true", help="rebuild only documents whose content hash changed")
    args = ap.parse_args()
    if args.cmd == "build":
        cfg = config.load(args.vault)
        backend = os.environ.get("WIKI_V2_EMBED_BACKEND", cfg.v2_embed_backend)
        device = os.environ.get("WIKI_EMBED_DEVICE", cfg.v2_embed_device)
        target = args.chunk_target if args.chunk_target is not None else cfg.v2_chunk_target_chars
        agent = cfg.v2_agent or "offline"
        print(f"[concepts] phase 1/2: extracting Atomic Concepts with {agent} "
              "(local GPU is not used during this phase)", file=sys.stderr, flush=True)
        display = PhaseProgress({"extract": ("Concept extraction", "chunk")})

        def progress(done, total, chunk):
            display.update("extract", done, total, chunk.path)

        try:
            docs, chunks, concepts = build_concepts(
                args.vault, target, changed_only=args.changed, progress=progress)
        finally:
            display.close()
        print(f"[concepts] phase 2/2: embedding {len(concepts)} Concept(s) "
              f"with {backend} on {device}",
              file=sys.stderr, flush=True)
        indexed = build_index(args.vault, show_progress=True, changed_only=args.changed)
        print(f"built v2 concepts: {len(docs)} document(s), {len(chunks)} chunk(s), {len(concepts)} concept(s), {indexed} indexed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
