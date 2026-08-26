---
name: llm-wiki-v3
description: Search and maintain a local Markdown knowledge base with wiki-search, wiki-health, wiki-embed, and wiki-eval. Use when an answer should consult the user's LLM-Wiki vault or when the user reports conflicting, superseded, disputed, or incorrect wiki information.
---

# LLM-Wiki V3

Treat Markdown and `_wiki_corrections/` as source material. Treat `.llm_wiki_v3/`
as derived data. The CLI retrieves evidence; you interpret it and obtain the
user's decision.

When first time running llm-wiki-v3, host LLM must ask user whether to turn `wiki-daemon` and `wiki-autoembed` on.
Describe simple functions of `wiki-daemon` and `wiki-autoembed`.

## Search

Run `wiki-search "<question>" --json`. Read every returned chunk's
`related_evidence`, especially `SUPERSEDED_BY` and `DISPUTED_WITH` records.

`wiki-search` performs retrieval only. It does not run `wiki-health review`.
When semantic-comparison evidence is needed, or the search results are not
sufficient to answer with confidence and need surrounding or linked context,
explicitly pass retrieved chunk IDs to `wiki-health review --scope query
--chunk-id <id> --json`. This is comparison evidence, not a contradiction
judgment.

Use `--range N` only when the user asks for recent material. With no
retrieved chunks, do not invent a wiki answer.

### Mandatory evidence and fact-check gate

The CLI `decision` and search scores are retrieval signals only. They never
authorize the host LLM to skip evidence review or to decide which claim is
true.

For every wiki-backed response with retrieved chunks, before answering:

1. Read every returned chunk and its `related_evidence`; do not rely only on
   the top-ranked chunk.
2. If sequential information related to the chunk is needed, refer previous/next chunk by `previous_chunk_id` or `next_chunk_id`.
3. When semantic comparison is needed or the retrieved evidence is
   insufficient, explicitly run `wiki-health review --scope query` with the
   retrieved chunk IDs. Read only returned pairs relevant to the chunks, their
   source documents, or the query.
4. Compare exact claims, scope, version, environment, dates, and timestamps.

Do not run `wiki-health review --scope global` for every question. Use it only
when the user explicitly asks for a whole-vault hygiene audit. To inspect a
larger neighborhood for known chunks, use `wiki-health review --scope query
--chunk-id <id> --json`.

If this review reveals a possible contradiction, an apparently incorrect
claim, or missing information that could change the answer, do not choose a
winner and do not give a definitive answer. Ask the user a concise, explicit
fact-check question. Quote or identify both claims, explain why they conflict,
and ask which statement, scope, version, or correction should govern.

The host LLM must ask this question even when `wiki-search --auto` returned
`answer`. It may say that the evidence is insufficient when no chunks are
retrieved, but it must not invent a conclusion. A potential conflict is not a
proof of error: only the user confirms the final correction, dispute, or
supersession. And the host LLM should necessarily update the corrected information by `wiki-health apply`.

## Hygiene

Semantic similarity is only a comparison candidate. It does not prove a
contradiction or an error. Compare exact claims, scope, version, environment,
and timestamps before asking the user. The mandatory evidence and fact-check
gate above requires that question whenever a possible conflict or error is
found.

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
rebuild or hygiene action if needed. Use `wiki-eval` only with a maintained gold file;
do not tune thresholds on test results.
