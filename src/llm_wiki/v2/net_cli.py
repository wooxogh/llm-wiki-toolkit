"""CLI for v2 NET artifacts."""
from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from llm_wiki.progress import PhaseProgress
from llm_wiki.v2.net_builder import build_net
from llm_wiki.v2.net_report import export_html, export_mermaid, render_tree
from llm_wiki.v2.net_store import NetStore
from llm_wiki.v2.tree_ops import (add_secondary_topic, change_primary_topic, create_collection, delete_topic,
                                   merge_topic, move_document, move_topic, remove_secondary_topic,
                                   rename_topic, restore_topic, undo_last)


def main() -> int:
    ap = argparse.ArgumentParser(prog="wiki-net")
    sub = ap.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build")
    build.add_argument("--vault", type=Path)
    build.add_argument("--no-ai-topic-creation", action="store_true")
    build.add_argument("--changed", action="store_true", help="reconcile only Concepts affected by source changes")
    export = sub.add_parser("export")
    export.add_argument("--vault", type=Path)
    export.add_argument("--out", type=Path)
    tree = sub.add_parser("tree", help="print the NET hierarchy in the terminal")
    tree.add_argument("--vault", type=Path)
    tree.add_argument("--show-concepts", action="store_true")
    tree.add_argument("--show-ids", action="store_true")
    tree.add_argument("--max-depth", type=int)
    tree.add_argument("--active-only", action="store_true")
    tree.add_argument("--ascii", action="store_true", dest="ascii_only")
    visualize = sub.add_parser("visualize", aliases=["graph"],
                               help="write a self-contained interactive HTML graph")
    visualize.add_argument("--vault", type=Path)
    visualize.add_argument("--out", type=Path)
    visualize.add_argument("--active-only", action="store_true")
    visualize.add_argument("--open", action="store_true", dest="open_browser")
    undo = sub.add_parser("undo")
    undo.add_argument("--vault", type=Path)
    undo.add_argument("--actor", default="user")
    collection = sub.add_parser("create-collection")
    collection.add_argument("--vault", type=Path)
    collection.add_argument("--id", required=True)
    collection.add_argument("--label", required=True)
    collection.add_argument("--parent", default="topic:knowledge")
    collection.add_argument("--type")
    collection.add_argument("--actor", default="user")
    for name in ("rename-topic", "move-topic", "merge-topic", "delete-topic", "restore-topic",
                 "move-document", "primary-topic", "add-secondary-topic", "remove-secondary-topic"):
        command = sub.add_parser(name)
        command.add_argument("--vault", type=Path)
        command.add_argument("--actor", default="user")
        command.add_argument("--id", required=True)
        command.add_argument("--target")
        command.add_argument("--label")
    args = ap.parse_args()
    if args.cmd == "build":
        display = PhaseProgress({
            "placement": ("NET placement", "concept"),
            "candidates": ("Relation candidates", "concept"),
            "relations": ("Relation analysis", "pair"),
        })
        try:
            store = build_net(
                args.vault,
                allow_ai_topic_creation=not args.no_ai_topic_creation,
                progress=display.update,
                changed_only=args.changed,
            )
        finally:
            display.close()
        print(f"built v2 NET: {len(store.nodes())} node(s), {len(store.edges())} edge(s)")
    elif args.cmd == "export":
        print(export_mermaid(args.vault, args.out))
    elif args.cmd == "tree":
        if args.max_depth is not None and args.max_depth < 0:
            ap.error("tree --max-depth must be zero or greater")
        print(render_tree(args.vault, show_concepts=args.show_concepts,
                          show_ids=args.show_ids, max_depth=args.max_depth,
                          active_only=args.active_only, ascii_only=args.ascii_only))
    elif args.cmd in {"visualize", "graph"}:
        path = export_html(args.vault, args.out, active_only=args.active_only)
        print(path)
        if args.open_browser:
            webbrowser.open(path.resolve().as_uri())
    else:
        store = NetStore(args.vault)
        if args.cmd == "undo":
            print(f"undid {undo_last(store, args.actor).op}")
        elif args.cmd == "create-collection":
            create_collection(store, args.id, args.label, args.parent, args.type, args.actor)
        elif args.cmd == "rename-topic":
            rename_topic(store, args.id, args.label or args.id, args.actor)
        elif args.cmd == "move-topic":
            move_topic(store, args.id, args.target or "topic:knowledge", args.actor)
        elif args.cmd == "merge-topic":
            if not args.target:
                ap.error("merge-topic requires --target")
            merge_topic(store, args.id, args.target, args.actor)
        elif args.cmd == "delete-topic":
            delete_topic(store, args.id, args.actor)
        elif args.cmd == "restore-topic":
            restore_topic(store, args.id, args.actor)
        elif args.cmd == "move-document":
            move_document(store, args.id, args.target or "topic:knowledge", args.actor)
        elif args.cmd == "primary-topic":
            change_primary_topic(store, args.id, args.target or "topic:knowledge", args.actor)
        elif args.cmd == "add-secondary-topic":
            add_secondary_topic(store, args.id, args.target or "topic:knowledge", args.actor)
        elif args.cmd == "remove-secondary-topic":
            if not args.target:
                ap.error("remove-secondary-topic requires --target")
            remove_secondary_topic(store, args.id, args.target, args.actor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
