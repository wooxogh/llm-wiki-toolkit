---
name: llm-wiki-v3
description: Search and maintain a local Markdown knowledge base with wiki-search, wiki-health, wiki-embed, and wiki-eval. Use when an answer should consult the user's LLM-Wiki vault or when the user reports conflicting, superseded, disputed, or incorrect wiki information.
---

# LLM-Wiki V3

Treat Markdown and `_wiki_corrections/` as source material. Treat `.llm_wiki_v3/`
as derived data. The CLI retrieves evidence; you interpret it and obtain the
user's decision.

## Search

Run `wiki-search "<question>" --auto --json`. Read every returned chunk's
`related_evidence`, especially `SUPERSEDED_BY` and `DISPUTED_WITH` records.

- `answer`: use the returned evidence with source attribution.
- `review`: inspect the listed evidence and explain the uncertainty.
- `none`: do not invent a wiki answer.

Use `--range N` only when the user asks for recent material. Use `--rerank` for
high-value or ambiguous searches when `--auto` is not already active.

## Hygiene

Semantic similarity is only a comparison candidate. It does not prove a
contradiction or an error. Compare exact claims, scope, version, environment,
and timestamps before asking the user.

Never apply a hygiene decision until the user has explicitly answered. Then
create a decision JSON matching [references/decisions.md](references/decisions.md)
and run:

```text
wiki-health apply <decision.json> --json
wiki-health --json
```

For partial supersession, keep the old chunk searchable. Link an existing
successor or create a new resolution chunk from the user's answer. For factual
errors, create a self-contained replacement chunk and retract the old chunk.
Do not rewrite the original Markdown for an error correction.

## Maintenance

Run `wiki-embed` after source documents change. Run `wiki-health` after every
rebuild or hygiene action. Use `wiki-eval` only with a maintained gold file;
do not tune thresholds on test results.

