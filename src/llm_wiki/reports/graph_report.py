#!/usr/bin/env python3
"""Graph-health report over the vault's [[wikilink]] graph — a knowledge-graph
analog of the "god-nodes / orphans / communities" reports popularized by
code-AST graph tools, adapted here to a *distilled-knowledge* vault instead of
a code AST.

The graph here is the one the vault already is:
  nodes  = pages (frontmatter id)
  edges  = body [[id]] wikilinks  +  frontmatter `links:` entries

What it surfaces (all hygiene signals, no auto-rewrite):
  - dangling links : [[id]] with no target page      -> write the page, or fix a typo
  - orphans        : pages with 0 inbound links       -> discoverability risk
  - islands        : pages with 0 inbound AND 0 outbound
  - god-nodes      : highest inbound degree (hubs)     -> split / index candidates
  - communities    : label-propagation clusters vs. the manual domain/ folders
  - bridge edges   : links crossing domain/layer       -> "surprising connections"

Pure stdlib + pyyaml (already a dependency). No network, no API key, no model.

  python -m llm_wiki.reports.graph_report            # print report to stdout
  python -m llm_wiki.reports.graph_report --write     # also write GRAPH_REPORT.md
  python -m llm_wiki.reports.graph_report --json      # machine-readable dump
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path

import yaml

from llm_wiki.paths import VAULT_ROOT, content_paths, content_root, embeddings_dir, relative

REPORT_PATH = content_root() / "GRAPH_REPORT.md"

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)")  # [[id]], [[id|alias]], [[id#anchor]]
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_RE = re.compile(r"`[^`\n]*`")


def strip_code(text: str) -> str:
    """Drop fenced ``` blocks and `inline code` so [[x]] written *as documentation*
    (e.g. explaining the link schema) is not miscounted as a real edge."""
    return _INLINE_RE.sub("", _FENCE_RE.sub("", text))


def wikilinks(body: str) -> set[str]:
    return {m.strip() for m in WIKILINK_RE.findall(strip_code(body))}


def parse_page(path: Path) -> tuple[dict, str]:
    """Return (frontmatter_dict, body_text)."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm = yaml.safe_load(text[3:end])
    body = text[end + 4 :]
    return (fm if isinstance(fm, dict) else {}), body


def collect(vault: Path = VAULT_ROOT) -> dict[str, dict]:
    """id -> {path, fm, out: set[id or dangling str], dangling: set[str]}."""
    pages: dict[str, dict] = {}
    raw: list[tuple[str, Path, dict, str]] = []
    for p in content_paths(vault):
        fm, body = parse_page(p)
        _id = fm.get("id") or p.stem
        raw.append((_id, p, fm, body))

    known = {r[0] for r in raw}
    for _id, p, fm, body in raw:
        # edges from body wikilinks + structured `links:` frontmatter field
        targets = wikilinks(body)
        targets |= {str(x).strip() for x in (fm.get("links") or [])}
        targets.discard(_id)  # ignore self-links
        out = {t for t in targets if t in known}
        dangling = {t for t in targets if t not in known}
        pages[_id] = {
            "path": relative(p, vault),
            "layer": fm.get("layer"),
            "domain": fm.get("domain"),
            "status": fm.get("status"),
            "confidence": fm.get("confidence"),
            "out": out,
            "dangling": dangling,
        }
    return pages


def build_degrees(pages: dict[str, dict]) -> None:
    for meta in pages.values():
        meta["indeg"] = 0
    for meta in pages.values():
        for t in meta["out"]:
            pages[t]["indeg"] += 1


def label_propagation(pages: dict[str, dict], rounds: int = 20) -> dict[str, int]:
    """Deterministic undirected label propagation -> community id per page.

    Deterministic: nodes processed in sorted id order; ties broken by lowest
    label. No RNG.
    """
    adj: dict[str, set[str]] = defaultdict(set)
    for src, meta in pages.items():
        for t in meta["out"]:
            adj[src].add(t)
            adj[t].add(src)
    ids = sorted(pages)
    label = {i: idx for idx, i in enumerate(ids)}  # start: every node its own
    for _ in range(rounds):
        changed = False
        for i in ids:
            if not adj[i]:
                continue
            counts: dict[int, int] = defaultdict(int)
            for nb in adj[i]:
                counts[label[nb]] += 1
            best = min((-c, lbl) for lbl, c in counts.items())[1]
            if label[i] != best:
                label[i] = best
                changed = True
        if not changed:
            break
    return label


# Generic id tokens that carry no topic signal (workflow/lifecycle words, dates).
# These come from this project's own kebab-case id convention (e.g. a
# "<topic>-phase2-interview" style id), not from any indexed content.
_LABEL_STOP = {
    "pat", "phase1", "phase2", "phase3", "v2", "execution", "interview", "planning",
    "session", "not", "fix", "the", "and", "design", "review", "raw", "metrics",
    "analysis", "recon", "state", "lessons", "gotchas", "verdict", "doc", "api",
    "non", "not", "can", "does", "still", "under", "from", "with", "all",
}


def label_community(members: list[str]) -> str:
    """Deterministic offline label = 1-2 most frequent topic tokens across member
    ids. (An LLM could auto-label communities instead; this stays key-free.)"""
    cnt: dict[str, int] = defaultdict(int)
    for m in members:
        for tok in m.split("-"):
            if tok.isdigit() or len(tok) <= 2 or tok in _LABEL_STOP:
                continue
            cnt[tok] += 1
    top = [t for t, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:2] if _ > 1]
    return "/".join(top) if top else "misc"


def connected_components(pages: dict[str, dict]) -> list[set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for src, meta in pages.items():
        for t in meta["out"]:
            adj[src].add(t)
            adj[t].add(src)
    seen: set[str] = set()
    comps: list[set[str]] = []
    for start in sorted(pages):
        if start in seen:
            continue
        comp: set[str] = set()
        q = deque([start])
        seen.add(start)
        while q:
            n = q.popleft()
            comp.add(n)
            for nb in adj[n]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        comps.append(comp)
    return comps


def analyze(pages: dict[str, dict]) -> dict:
    build_degrees(pages)
    n = len(pages)
    edges = sum(len(m["out"]) for m in pages.values())

    orphans = sorted(i for i, m in pages.items() if m["indeg"] == 0)
    islands = sorted(i for i, m in pages.items() if m["indeg"] == 0 and not m["out"])
    god = sorted(pages.items(), key=lambda kv: (-kv[1]["indeg"], kv[0]))
    god_nodes = [(i, m["indeg"]) for i, m in god if m["indeg"] > 0][:12]

    dangling: dict[str, list[str]] = defaultdict(list)
    for i, m in pages.items():
        for t in sorted(m["dangling"]):
            dangling[t].append(i)

    bridges = []  # (src, dst, kind) crossing domain or layer
    for src, m in pages.items():
        for t in m["out"]:
            tm = pages[t]
            if m.get("domain") and tm.get("domain") and m["domain"] != tm["domain"]:
                bridges.append((src, t, f'{m["domain"]}->{tm["domain"]}'))
            elif m.get("layer") != tm.get("layer"):
                bridges.append((src, t, f'{m.get("layer")}->{tm.get("layer")}'))

    comps = connected_components(pages)
    labels = label_propagation(pages)
    comm: dict[int, list[str]] = defaultdict(list)
    for i, lbl in labels.items():
        if pages[i]["out"] or pages[i]["indeg"]:  # skip fully-isolated
            comm[lbl].append(i)
    communities = []
    for members in sorted((sorted(v) for v in comm.values() if len(v) >= 2), key=len, reverse=True):
        doms: dict[str, int] = defaultdict(int)
        for m in members:
            doms[pages[m].get("domain") or pages[m].get("layer")] += 1
        hub = max(members, key=lambda m: (pages[m]["indeg"], m))
        communities.append({
            "label": label_community(members),
            "hub": hub,
            "size": len(members),
            "domains": dict(sorted(doms.items(), key=lambda x: -x[1])),
            "members": members,
        })

    return {
        "n_pages": n,
        "n_edges": edges,
        "n_dangling": sum(len(v) for v in dangling.values()),
        "orphans": orphans,
        "islands": islands,
        "god_nodes": god_nodes,
        "dangling": {k: sorted(v) for k, v in sorted(dangling.items())},
        "bridges": sorted(bridges),
        "n_components": len(comps),
        "largest_component": len(max(comps, key=len)) if comps else 0,
        "communities": communities,
        "pages": pages,
    }


def render_md(a: dict) -> str:
    n = a["n_pages"]
    L = []
    L.append("# GRAPH_REPORT — vault link-graph health")
    L.append("")
    L.append("> GENERATED by `llm_wiki.reports.graph_report`. Hygiene signals only — no auto-rewrite.")
    L.append("> Graph = pages (nodes) + `[[wikilink]]`/`links:` (edges). Code is NOT indexed by design.")
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append(f"- Nodes (pages): **{n}**")
    avg_out = a["n_edges"] / n if n else 0.0
    L.append(f"- Edges (links): **{a['n_edges']}**  (average out-degree {avg_out:.2f})")
    orph = len(a["orphans"])
    orph_pct = (orph * 100 // n) if n else 0
    L.append(f"- Orphans (inbound 0): **{orph} / {n}  ({orph_pct}%)** — unreachable by graph traversal")
    L.append(f"- Islands (inbound and outbound both 0): **{len(a['islands'])}**")
    L.append(f"- Dangling links: **{a['n_dangling']}**")
    L.append(f"- Connected components: **{a['n_components']}** (largest: {a['largest_component']} nodes)")
    L.append(f"- Communities (>=2 members, label propagation): **{len(a['communities'])}**")
    L.append("")

    L.append("## God-nodes (highest inbound — hub / split-or-index candidates)")
    L.append("")
    if a["god_nodes"]:
        for i, deg in a["god_nodes"]:
            L.append(f"- `{i}` <- **{deg}** inbound")
    else:
        L.append("_(none — the graph is too sparse)_")
    L.append("")

    L.append("## Dangling links (pages to write, or typo candidates)")
    L.append("")
    if a["dangling"]:
        for tgt, srcs in a["dangling"].items():
            L.append(f"- `[[{tgt}]]` — referenced by: {', '.join(f'`{s}`' for s in srcs)}")
    else:
        L.append("_(none)_")
    L.append("")

    L.append("## Bridge edges (crossing domain/layer — unexpected connections)")
    L.append("")
    if a["bridges"]:
        for s, t, kind in a["bridges"]:
            L.append(f"- `{s}` -> `{t}`  ({kind})")
    else:
        L.append("_(none)_")
    L.append("")

    L.append("## Communities (auto-labelled · link topology vs. manual domain/ folders)")
    L.append("")
    L.append("> Label = the most frequent topic token(s) across member ids (deterministic, offline); "
              "hub = highest inbound degree within the community.")
    L.append("")
    if a["communities"]:
        for k, c in enumerate(a["communities"], 1):
            mix = ", ".join(f"{d}x{n}" for d, n in c["domains"].items())
            L.append(f"- **C{k} · `{c['label']}`** ({c['size']}) — hub `{c['hub']}` · {mix}")
            L.append(f"  - {', '.join(f'`{m}`' for m in c['members'])}")
    else:
        L.append("_(no connected clusters — the graph is effectively disconnected)_")
    L.append("")

    L.append("## Orphans (inbound 0 — discoverability risk, first 30)")
    L.append("")
    for i in a["orphans"][:30]:
        tag = " (island)" if i in set(a["islands"]) else ""
        L.append(f"- `{i}`{tag}")
    if orph > 30:
        L.append(f"- … and {orph - 30} more")
    L.append("")
    return "\n".join(L)


def suggest_backlinks(pages: dict[str, dict], topk: int = 3, tau: float = 0.55) -> list[dict]:
    """For every orphan (inbound 0), propose the semantically-nearest pages as
    candidate backlink *sources* — i.e. "add [[orphan]] inside page N". Evidence
    = cosine over the existing page embeddings (an inferred-edge idea, grounded
    in this project's own retriever). numpy is imported lazily so the core
    report still runs without ever touching the embedding store.

    A candidate is dropped if the two pages are already linked in either direction.

    This operates on the process's own vault (`VAULT_ROOT`), not a `vault`
    argument, because it reads a pre-built `.embeddings/` store that is itself
    tied to one vault checkout — there is no synthetic-vault variant of it.
    """
    import numpy as np  # lazy: only --suggest needs it

    EMB = embeddings_dir()
    if not (EMB / "vectors.npy").exists():
        raise SystemExit("no .embeddings/ — run: wiki-embed")
    import json as _json

    vecs = np.load(EMB / "vectors.npy")
    meta = _json.loads((EMB / "meta.json").read_text(encoding="utf-8"))

    # aggregate chunk vectors -> one L2-normalized vector per page
    by_page: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        by_page[m["id"]].append(i)
    ids = [i for i in sorted(by_page) if i in pages]
    mat = np.stack([vecs[by_page[i]].mean(axis=0) for i in ids])
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    # contiguous float32 copy + errstate: numpy's SIMD matmul emits spurious
    # divide-by-zero/overflow RuntimeWarnings on some views though the inputs
    # are finite and tiny (values <=0.2) and the result is correct.
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sim = np.nan_to_num(mat @ mat.T)
    idx = {i: k for k, i in enumerate(ids)}

    # symmetric "already linked" set (either direction)
    linked: set[frozenset] = set()
    for src, m in pages.items():
        for t in m["out"]:
            linked.add(frozenset((src, t)))

    build_degrees(pages)
    out = []
    for orphan in sorted(i for i in ids if pages[i]["indeg"] == 0):
        row = sim[idx[orphan]]
        order = row.argsort()[::-1]
        cands = []
        for k in order:
            src = ids[k]
            if src == orphan or float(row[k]) < tau:
                continue
            if frozenset((src, orphan)) in linked:
                continue
            cands.append({"source": src, "cos": round(float(row[k]), 3)})
            if len(cands) >= topk:
                break
        if cands:
            out.append({"orphan": orphan, "candidates": cands})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write GRAPH_REPORT.md")
    ap.add_argument("--json", action="store_true", help="dump machine-readable JSON")
    ap.add_argument("--suggest", action="store_true",
                    help="propose backlink candidates for orphans (uses .embeddings; needs numpy)")
    ap.add_argument("--tau", type=float, default=0.55, help="min cosine for --suggest (default 0.55)")
    args = ap.parse_args()

    pages = collect()

    if args.suggest:
        sug = suggest_backlinks(pages, tau=args.tau)
        if args.json:
            print(json.dumps(sug, ensure_ascii=False, indent=2))
            return 0
        print(f"# Backlink candidates for {len(sug)} orphans (cos >= {args.tau})\n")
        for s in sug:
            cs = "  ".join(f"{c['source']}({c['cos']})" for c in s["candidates"])
            print(f"[[{s['orphan']}]]  <- add in:  {cs}")
        return 0

    a = analyze(pages)

    if args.json:
        dump = {k: v for k, v in a.items() if k != "pages"}
        print(json.dumps(dump, ensure_ascii=False, indent=2))
        return 0

    md = render_md(a)
    print(md)
    if args.write:
        REPORT_PATH.write_text(md + "\n", encoding="utf-8")
        print(f"\n[written] {REPORT_PATH.relative_to(content_root())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
