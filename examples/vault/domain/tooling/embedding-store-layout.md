---
id: embedding-store-layout
layer: domain
domain: tooling
projects: ["llm-wiki"]
tags: ["embeddings", "index", "drift"]
confidence: confirmed
status: active
updated: 2026-01-12
summary: 'The four files under .embeddings/ each carry a different drift risk, and a health check has to name each one rather than only checking that the directory exists.'
links: ["vault-schema-contract"]
---

# The four files in `.embeddings/`, and how each goes stale

The vector store this toolchain builds is not one artifact but four,
because each one can independently fall out of sync with the pages while
still existing on disk — an "is the directory there" check would pass in
every one of these cases while recall was already serving wrong or missing
results.

- **`vectors.npy`** — one row per chunk, the dense vectors themselves.
- **`meta.json`** — one record per chunk (same row order as `vectors.npy`):
  id, layer, domain, tags, status, summary, and the chunk text. Retrieval
  reads only this file at query time; it never re-opens a page's Markdown.
  Anything a filter or a ranking decision needs has to be carried here, or it
  is invisible to search regardless of what the page itself says.
- **`pages.json`** — a content hash per page, used to decide incremental
  re-embedding: a page whose hash has not changed is skipped and its old
  vectors are reused as-is.
- **`model.txt`** — an identity string of the form
  `model-name|chunk-schema-version|meta-schema-version`. This exists because
  the *shape* of a chunk or a metadata record can change even when a page's
  raw text has not, and the incremental path above would otherwise never
  notice — a chunking-scheme change with no page edits would leave every
  vector "correctly" reused under a scheme it was never built for.

Each file corresponds to a distinct drift class that a health check has to
name separately rather than folding into one generic "store looks stale"
signal:

- vectors and meta rows can disagree in count (a partial write, or a merge
  gone wrong) — a shape mismatch that existence checks miss entirely.
- a page can be edited after being embedded (`pages.json` hash no longer
  matches the live page) — stale content, not a missing file.
- a page can be embedded but later deleted from the vault (`pages.json` has
  an id the vault no longer does) — an orphaned vector nobody asked for.
- the identity in `model.txt` can predate the code that reads the store
  (chunk or metadata schema changed) — every vector present, every hash
  matching, and the whole store still meaningless, because a fully-correct
  presence-and-hash check cannot see a schema change.

Treating these as four separately-diagnosable failures rather than one
"store is dirty" flag is what lets a health check say exactly what to do
next (re-embed one page vs. rebuild the whole store) instead of only "run
something and see". The same file-by-file discipline is why the vault's
generated index has its own dedicated byte check — see
`generated-artifact-must-be-byte-checked`.
