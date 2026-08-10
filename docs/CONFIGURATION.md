# Configuration

Two things are configurable: **where the vault is**, and **what shape its pages
and gold set must have**. Both are answered by a single optional file,
`wiki.toml`, at the vault root. The built-in defaults are a complete
configuration, not a partial one — a vault with no `wiki.toml` works.

## Finding the vault root

`llm_wiki.config.find_root()` resolves, in order:

1. **`WIKI_VAULT`** — used as given (expanded and resolved). Nothing else is
   consulted.
2. **The nearest ancestor containing `wiki.toml`**, starting at the current
   directory and walking up.
3. **The current directory.**

The root is resolved **once, at import time** (`llm_wiki.paths.VAULT_ROOT`).
Setting `WIKI_VAULT` after importing the package has no effect on that process —
which is why tests that need a different vault spawn a subprocess rather than
monkeypatching. For the same reason, every command in the docs sets `WIKI_VAULT`
in the environment before the process starts.

```bash
export WIKI_VAULT="$PWD/examples/vault"     # explicit
cd ~/notes/my-vault && wiki-index           # ancestor search finds ./wiki.toml
```

### Config root and content root

For almost every vault these are the same directory and there is nothing to
think about. They separate only when `wiki.toml` sets `[vault] root`:

| | what it is | how to get it |
|---|---|---|
| **config root** | the directory holding `wiki.toml`. `WIKI_VAULT` names *this* one, and it is the `vault` handle every command and API takes. | `paths.VAULT_ROOT`, `Config.config_dir` |
| **content root** | where the pages and everything derived from them live. Equal to the config root unless `[vault] root` redirects it. | `paths.content_root(vault)`, `Config.root` |

The rule for which is which: **config lookups resolve against the config root;
pages, vault-relative paths, and generated artifacts resolve against the content
root.** So `wiki.toml` is the only file that stays behind in the config root —
`index.yaml`, `.embeddings/`, the generated reports, the gold set, the baseline,
and `auto_thresholds.json` all sit beside the pages in the content root, and
every `path:` in `index.yaml` and `meta.json` is relative to the content root
(`notes/a.md`, never `sub/notes/a.md`).

Keeping them distinct matters more than it looks. `config.load()` takes a
*config* root; handing it a content root finds no `wiki.toml` there and silently
returns the built-in defaults, which is a config file that appears to be honoured
and is not. `paths.index_path()` and `paths.embeddings_dir()` exist so no caller
has to remember which root to join a filename onto.

## `wiki.toml`

Copy `wiki.toml.example` to your vault root and change only what differs.
Unknown tables and unknown keys inside a known table are **errors**, not ignored —
a typo in a config key that silently changes nothing is worse than a loud
failure.

### `[vault]`

| key | type | default | meaning |
|---|---|---|---|
| `content_dirs` | list of strings | `["domain", "patterns", "entities", "raw"]` | directories scanned for pages. Only those that exist are used. Must not be empty. |
| `root` | string | *(unset)* | redirect the *content* root to a subdirectory of the config root. Rarely needed; useful when the vault content lives under a repo that also holds other things. `wiki.toml` stays put; everything else — `index.yaml`, `.embeddings/`, gold, reports — moves with the content. |

Page order is (declared directory order, then sorted path), so every artifact
built from that list is byte-stable across machines and filesystems.

### `[schema]`

| key | type | default | meaning |
|---|---|---|---|
| `layers` | list of strings | `["domain", "pattern", "entity", "raw"]` | allowed values of `layer:`. Must not be empty. |
| `domains` | list of strings | `[]` | allowed values of `domain:`. **Empty means no domain-value validation at all** — the field is still required when `layer: domain`, but its value is unconstrained. |
| `required` | list of strings | `["id", "layer", "projects", "tags", "confidence", "status", "summary"]` | frontmatter fields that must be present and non-empty. |

`id` and `layer` cannot be dropped from `required`. The whole addressing scheme —
filename == id, `[[id]]` wikilinks, per-layer ranking and evaluation slices —
collapses without them, so removing them is rejected rather than honoured.

### `[lint]`

| key | type | default | meaning |
|---|---|---|---|
| `packs` | list of strings | `["en"]` | `claim_lint` language packs. `en` and `ko` ship. |

Packs compose: `packs = ["en", "ko"]` lints both languages in one pass. An
unknown pack name is a hard error — a linter that quietly checks nothing because
of a misspelling is worse than one that fails loudly. See
[HYGIENE.md](HYGIENE.md#claim-lint) for what a pack contains and how to add one.

### `[eval]`

| key | type | default | meaning |
|---|---|---|---|
| `gold` | string | `"eval_gold.json"` | the vault's gold file. Absolute, or relative to the content root. |
| `minimums` | table | see below | curation floors the gold set must meet for `wiki-eval --validate-only` to pass. |

### `[eval.minimums]`

```toml
[eval.minimums]
total = 150
recent_cases = 30
layer    = { domain = 100, pattern = 25, entity = 5, raw = 2 }
domain   = {}
category = { ambiguous = 10, negative = 8 }
```

Those are the defaults, and they are sized for a corpus of a couple of hundred
pages. Scale them to yours — `examples/vault/wiki.toml` lowers `total` to 10 for
a 9-page vault, which is the whole reason these are configurable rather than
hard-coded.

**A table axis is replaced entirely, not merged key by key.** Writing
`layer = { pattern = 30 }` *drops* the default `domain`, `entity`, and `raw`
floors rather than keeping them alongside the new value. Scalar axes (`total`,
`recent_cases`) are overridden value for value. This is intentional: declaring an
axis means declaring all of it, so the vault owner states a complete floor rather
than silently inheriting entries they never wrote down.

The default `domain` axis is empty on purpose. A vault's own domain names are
unknowable to this package, so shipping one author's domain quotas as a universal
requirement would be nonsense.

### `[ingest]`

| key | type | default | meaning |
|---|---|---|---|
| `repos` | list of strings | `[]` | absolute paths the daily ingest scans for new commits. Empty (the default) omits the authoring step entirely. |
| `prompt_file` | string | *(packaged default)* | the authoring prompt. Absolute, or relative to the content root. |

See [INGEST.md](INGEST.md).

## The frontmatter contract

`index.yaml` is generated from page frontmatter. The frontmatter is the single
source of truth; you edit pages, never the index.

```yaml
---
id: hybrid-ranking-tradeoffs      # kebab-case, globally unique, == filename stem
layer: domain                     # one of [schema] layers
domain: tooling                   # required when layer == domain
projects: [some-repo]             # list of strings
tags: [retrieval, bm25]           # list of strings
confidence: confirmed             # confirmed | provisional
status: active                    # active | superseded
updated: 2026-01-18               # ISO date (YYYY-MM-DD)
links: [cross-encoder-rerank-effect]   # optional; each must resolve to a real id
summary: 'One measured line, exposed in the index and prefixed to every chunk.'
---
```

Enforced by `wiki-index` (all of these are errors, not warnings):

- **Filename stem == `id`**, so `[[id]]` wikilinks resolve.
- **`id` is kebab-case**: `[a-z0-9]` groups joined by single hyphens.
- **`id` is unique** across the vault.
- **Types are checked, not just presence.** `projects: repo` (a bare string)
  instead of `projects: [repo]` passes a presence check and then quietly behaves
  as a four-element list downstream in retrieval filters. Rejected at the schema
  boundary instead.
- **`updated` must be an ISO date.** PyYAML parses a bare `2026-01-18` into a
  date object and a quoted one into a string; both are accepted, prose is not.
- **`confidence` ∈ {confirmed, provisional}**, **`status` ∈ {active, superseded}**.
- **`links:` entries must resolve** to an existing page id.
- **No buried frontmatter blocks.** A second `---` block pasted into a page body
  is invisible to the index, to recall, and to the link graph, while sitting in
  the vault looking filed. Fenced code blocks are exempt so a page can document
  frontmatter as an example.

Non-fatal warnings: `wiki-index` prints dangling `[[wikilinks]]` and the orphan
count. `wiki-health` reports both of those plus summaries over 300 characters and
hedged claims with no nearby measurement.

### Always quote `summary:`

In YAML, a `#` preceded by whitespace starts a comment. So this:

```yaml
summary: fixed in PR #527 — the retry ceiling was the real cause
```

parses to `fixed in PR`, and the rest never reaches the index. It is valid YAML,
so neither the parser nor a presence check notices.

This is not hypothetical. Measured on a real vault (2026-08-04): seven pages were
in this state, and on one of them a **5,501-character summary had been truncated
to 73 characters**. That matters more than it sounds: the summary is the contextual
prefix attached to *every chunk of that page*, so a truncated summary degrades
the searchability of the entire page, not just its index entry.

`wiki-index` now detects the truncation and fails with the field name, the parsed
length, the raw length, and the fix. The habit that makes it a non-issue is
simpler: **always single-quote `summary:`** (double an interior `'`). Leading
`>`, `|`, and a `:` followed by a space are safe inside quotes for the same
reason.

## Files at the vault root

Everything in this table lives in the **content root** (see [Config root and
content root](#config-root-and-content-root)) — except `wiki.toml` itself, which
is what defines the config root. Without `[vault] root` the two are the same
directory and the distinction never comes up.

| path | generated? | committed? | what it is |
|---|---|---|---|
| `wiki.toml` | no | yes | this file — the one entry that stays in the config root |
| `index.yaml` | **yes** (`wiki-index`) | yes | the page index; byte-compared by `--check` |
| `GRAPH_REPORT.md` | **yes** (`graph_report --write`) | optional | link-graph hygiene |
| `COMMUNITIES.md` | **yes** (`community_report --write`) | optional | community digests |
| `community_summaries.json` | no (hand- or LLM-authored) | yes | abstractive synthesis per community signature — see [HYGIENE.md](HYGIENE.md#community-synthesis) |
| `eval_gold.json` | no | yes | the gold set (name configurable) |
| `eval_baseline.json` | **yes** (`wiki-gate --update`) | yes | the retrieval-regression baseline |
| `auto_thresholds.json` | **yes** (`wiki-eval --calibrate`) | yes | this vault's `--auto` thresholds; overrides the packaged default |
| `.embeddings/` | **yes** (`wiki-embed`) | **no** | local vector store; gitignored |
| `.local/` | yes | **no** | opt-in telemetry; gitignored |
| `.contradictions.md`, `.compaction.md` | **yes** | **no** | hygiene candidate lists; gitignored |

A generated-and-committed artifact only stays honest if something compares it
against a fresh render. `wiki-health` does exactly that for `index.yaml`,
`GRAPH_REPORT.md`, and `COMMUNITIES.md`. An *absent* report is not drift — a
vault that never generated them is not claiming anything — but a present, stale
one is an error.

## Environment variables

| variable | default | effect |
|---|---|---|
| `WIKI_VAULT` | — | vault **config** root (the directory holding `wiki.toml`); highest precedence |
| `WIKI_TOKENIZER` | `bigram` | `kiwi` switches the sparse tokenizer to the morphological analyser (measured worse — see [RETRIEVAL.md](RETRIEVAL.md#tokenization)) |
| `WIKI_DENSE_W` | `2.0` | dense weight in RRF fusion |
| `WIKI_SPARSE_W` | `1.0` | sparse weight in RRF fusion |
| `WIKI_EMBED_DEVICE` | `mps` | `mps` \| `cpu` \| `cuda`. An unavailable accelerator falls back to CPU with a warning on stderr, so the default works off Apple Silicon. |
| `WIKI_EMBED_MAX_SEQ` | `512` | embedder max sequence length (memory guard) |
| `WIKI_EMBED_BATCH` | `4` | embedder batch size (memory guard) |
| `WIKI_EMBED_PORT` | `8477` | port of the optional resident embedding server |
| `WIKI_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | cross-encoder model id |
| `WIKI_RECALL_TELEMETRY` | unset | `1` enables opt-in local recall logging |
