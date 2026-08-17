---
id: example-vault-walkthrough
layer: entity
projects: ["llm-wiki"]
tags: ["example-vault", "fixture", "quickstart"]
confidence: confirmed
status: active
updated: 2026-01-18
summary: 'What this example vault under examples/vault/ is for and how it is put together: nine pages, its own lowered eval minimums, and a gold set that CI validates against a live index.'
links: ["hybrid-ranking-tradeoffs", "cross-encoder-rerank-effect", "embedding-store-layout", "vault-schema-contract", "generated-artifact-must-be-byte-checked", "calibrate-and-report-on-different-splits", "one-index-serves-every-query-language", "unmeasured-claims-are-worse-than-gaps"]
---

# What this example vault is for

A knowledge-vault toolchain that ships with no content has nothing for its
own commands to run against. `wiki-index`, `wiki-health`, and `wiki-eval`
would have no vault to validate, and a quickstart in a README could not be
copy-pasted and actually run. This directory (`examples/vault/`) exists to
fix that: it is a small, real vault — not mocked, not a fixture that only a
test imports — that the packaged commands can be pointed at directly with
`WIKI_VAULT=$PWD/examples/vault`, and that CI runs the real gates against on
every change to the toolchain.

## Shape

Nine pages across three layers:

- **domain** (4): `hybrid-ranking-tradeoffs` and `cross-encoder-rerank-effect`
  and `embedding-store-layout` under the `tooling` domain, plus
  `vault-schema-contract` under `research`.
- **pattern** (4): `generated-artifact-must-be-byte-checked`,
  `calibrate-and-report-on-different-splits`, and two pages written in
  Korean — `one-index-serves-every-query-language` and
  `unmeasured-claims-are-worse-than-gaps`.
- **entity** (1): this page.

The two Korean pages are deliberate, not incidental. Cross-lingual recall —
one embedding index serving queries in more than one language — is a real,
load-bearing feature of this toolchain (see
`one-index-serves-every-query-language` for how it works), and an example
vault that never actually contains a non-English page would not exercise it.
Their ids and filenames are still ASCII kebab-case, because `wiki-index`
rejects a non-ASCII id and requires the filename stem to equal the id — only
the body, `summary`, and `tags` are Korean.

## Why this vault declares its own config

`wiki.toml` in this directory sets `[schema] domains` to `["tooling",
"research"]` (this vault's actual two domains) and lowers `[eval.minimums]`
well below the tool's built-in defaults. The built-in defaults assume a
corpus in the hundreds of pages; a 9-page, 12-gold-case example cannot meet
them, and forcing it to would mean either inflating this vault with filler
content or disabling the gate entirely — both worse than the actual fix,
which is that minimums are per-vault configuration rather than a constant
baked into the tool. Every real vault is expected to set its own floors the
same way once it has enough curated gold cases to justify them.

## The gold set

`eval_gold.json` has 12 cases, split across `calibration` and `test`, with
at least one `ambiguous` case (a query with more than one defensible
answer among this vault's own pages) and at least one `negative` case (a
query this vault has no page for, where the correct behavior is to return
nothing). Two queries are in Korean, matching the two Korean pages above.
`wiki-eval --validate-only` checks the gold set's structure and its coverage
against this vault's own declared minimums without embedding anything, which
is what makes it safe to run in CI where no embedding store exists.

## What CI actually verifies here

Three commands, all against this directory: `wiki-index --check` (the
committed `index.yaml` must be byte-identical to a fresh render of the nine
pages' frontmatter — see `generated-artifact-must-be-byte-checked` for why
that has to be a byte check and not just a validity check), `wiki-health
--mode ci` (schema, link, and drift checks that do not require an embedding
store), and `wiki-eval --validate-only` (gold-set structure and coverage).
None of the three touches `.embeddings/`, which this vault does not commit —
building it locally is `wiki-embed --full`, exercising
`embedding-store-layout`'s four files end to end.
