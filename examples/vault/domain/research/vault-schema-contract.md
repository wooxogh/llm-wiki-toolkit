---
id: vault-schema-contract
layer: domain
domain: research
projects: ["llm-wiki"]
tags: ["frontmatter", "schema", "yaml"]
confidence: confirmed
status: active
updated: 2026-01-13
summary: 'The frontmatter contract every page in this vault has to satisfy: seven required fields, filename equal to id, and why an unquoted summary can silently lose most of its text.'
links: ["generated-artifact-must-be-byte-checked"]
---

# What every page's frontmatter has to declare

This toolchain treats a page's YAML frontmatter as the single source of
truth and everything else (the generated index, the embedding store, the
link-graph reports) as derived from it. That only works if every page's
frontmatter is shaped the same way, so the schema is enforced, not just
documented:

- **`id`** — ASCII kebab-case (lowercase letters, digits, single hyphens).
  Not a style preference: `[[id]]` wikilinks, the generated index, and the
  ranking filters all address a page by this string, and a non-ASCII or
  irregular id would make some of those addressing paths behave differently
  from others.
- **The filename stem must equal `id`.** A page's file has to be
  discoverable *by* its id (so an Obsidian-style `[[id]]` link resolves to a
  real file) as well as *containing* that id in its frontmatter. Letting
  these drift apart — a file renamed without updating its own `id`, or vice
  versa — breaks that bidirectional guarantee silently: the page still
  parses, still has a valid id, and still fails to be the thing any link to
  it expects to find.
- **`layer`, `projects`, `tags`, `confidence`, `status`, `summary`** — the
  remaining six required fields. `domain` joins that list specifically when
  `layer: domain`, because a domain page without a domain is a
  classification the corpus cannot use, while a pattern or entity page has
  no domain to declare in the first place.
- **`links`** (optional) must resolve to ids that actually exist. A link to
  a typo'd id is not a warning here — it is treated the same as any other
  broken reference, because a corpus that lets dangling links accumulate
  quietly loses the one thing a link graph is for.

One frontmatter failure mode is worth calling out on its own because it is
invisible to both the YAML parser and a naive schema check: an unquoted
`summary:` value containing a bare ` #` parses as valid YAML — the `#` just
starts a comment there — so everything after it is silently dropped before
the schema check ever sees it. The page still has *a* summary, still passes
"is this field present and non-empty", and still loses most of its actual
text. That matters more than an ordinary truncated field would, because this
toolchain prefixes each page's summary onto every one of that page's chunks
for retrieval context — so a truncated summary quietly degrades that page's
searchability everywhere, not just wherever the summary itself is displayed.
The fix is mechanical: wrap the value in quotes, which is why every page in
this example vault (including this one) quotes its `summary:`.
