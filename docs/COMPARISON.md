# Prior Art & Comparison

Research notes on adjacent open-source and closed tools, compiled while
scoping LLM-Wiki's differentiation. Findings marked "code-audited" were
verified by reading the actual source, not just marketing copy or README
claims — everything else is documentation-level and should be re-verified
before being cited as a hard fact in submission materials.

## Closed / not a direct competitor (context only)

**Toss "Topic"** (토스 기술블로그, 2026-07-30) — a six-axis trust layer
(Granularity/Faithfulness/Staleness/Canonicality/Consistency/Coverage) for
managing docs+code+internal-messenger context for LLM agents. Internal to
Toss, not open source. Cited here as evidence that the problem LLM-Wiki
targets is real and taken seriously at production scale — not as a
competitor to differentiate against.

## Direct open-source competitors

### memweave (`sachinsharma9780/memweave`)
- Markdown + SQLite, BM25 + dense hybrid search, temporal decay by file age.
- **Code-audited** (`memweave/search/postprocessor.py`): result filtering is
  a single-signal `ScoreThreshold` (default 0.35) — no equivalent to
  LLM-Wiki's multi-channel confidence contract exists in the code, only
  score-threshold filtering, MMR diversity reranking, and temporal decay.
- No contradiction/supersede handling of any kind found in the `search/`
  module.

### memsearch (`milvus-io/memsearch`)
- Official Milvus team project. Indexes markdown into a Milvus vector DB
  for search. Ships a ready-made Claude Code plugin.
- Storage/search focused; no hygiene (staleness/contradiction) layer
  identified. Depends on an external vector DB, unlike LLM-Wiki's
  file-based local store.

### Basic Memory (`basicmachines-co/basic-memory`)
- 3,000+ stars, ~57k downloads/month. MCP-native, works with Claude,
  Cursor, Obsidian, VS Code, ChatGPT.
- Optional cross-encoder reranking, bidirectional human/AI read-write on
  the same files, "team-safe cloud push/pull" in newer releases (freemium:
  local free + paid cloud team tier).
- No published evidence of a structured confidence contract (answer vs.
  needs-review vs. no-answer) or of claim-level partial-supersede tracking.

### Mem0 (`mem0ai/mem0`)
- ~45–48k GitHub stars, 14M+ downloads, $24M funding, AWS's chosen memory
  provider for their Agent SDK.
- Full pipeline: chunking, embedding, LLM-based fact extraction, entity
  linking, multi-signal (semantic + keyword + entity) retrieval.
- General-purpose agent memory, not markdown-vault-specific; no
  claim-level supersede/dispute model identified in public docs.

### MemPalace (`MemPalace/mempalace`)
- ~48k stars, launched April 2026, explosive growth. Local-first
  (ChromaDB + PyYAML only), hierarchical "memory palace" structure
  (Wings→Rooms→Halls→Closets→Drawers), reports 96.6% on LongMemEval using
  rule-based classification (no LLM calls).
- **Explicit design philosophy: store verbatim, no summarization/rewriting
  no reconciliation.** This means contradiction/staleness handling is
  out of scope by design, not an oversight.
- Independent reproduction studies (`web3guru888/mempalace-scientific-analysis`,
  `lhl/agentic-memory`) found the headline benchmark numbers partly
  attributable to the base embedding model rather than the spatial
  structure itself, and reported a lower reproduced score with ~12.4%
  information loss in the compaction step. Cited here as a reminder that
  this community actively re-audits benchmark claims — our own numbers
  should stay reproducible and conservatively framed.
- Has bi-temporal (validity-window) time tracking, which LLM-Wiki does
  not yet have at the same precision.

### Zep / Graphiti, Cognee
- Typed-edge temporal knowledge graphs (causes/contradicts/depends-on
  style relations), more precise time modeling (bi-temporal: when true vs.
  when known) than LLM-Wiki's current `status: superseded` + strikethrough
  approach.
- Not local-first by default (Zep); heavier graph-database dependency
  profile than LLM-Wiki's flat-file store.

### General-purpose RAG/knowledge platforms
**Open WebUI** and **RAGFlow** — broad LLM-usage / document-RAG platforms
with strong PDF/Excel/OCR support, general similarity/rerank-based result
scoring, and no dedicated regression-prevention gate or structured
confidence contract for individual results.

## Comparison table

| | Open WebUI | RAGFlow | Basic Memory | memweave | MemPalace | **LLM-Wiki** |
|---|---|---|---|---|---|---|
| Primary purpose | multi-LLM + doc convenience | accurate parsing of complex docs | notes as agent memory | hybrid md search | verbatim agent memory at scale | verify whether retrieved knowledge can be trusted |
| Result confidence signal | similarity/rerank | score/rerank | similarity-based | single `ScoreThreshold` (code-audited) | n/a (retrieval ranking only) | multi-channel weighted RRF (V3) |
| Regression-prevention gate | none | none | none | none | none | `wiki-gate` (blocks build on measured regression) |
| Claim-level partial supersede | no | no | not identified | no | explicitly out of scope | `superseded_claims` on individual quotes within a chunk |
| Local-first, no API key in query path | partial | self-hostable | yes | yes | yes | yes |

## What we concede

- **Typed relation graphs** (Zep/Graphiti, Cognee): more expressive than
  our current link graph; roadmap item, not yet built.
- **Bi-temporal precision**: Zep/MemPalace track validity windows more
  precisely than our current status-tag approach.
- **Scale**: Mem0 and MemPalace operate at a scale (40k+ stars, funded
  teams) LLM-Wiki has not been validated against.

## Revision note

An earlier version of this document listed a cosine+cross-encoder dual-signal
`answer/review/none` policy as the primary differentiator. That reranker was
measured and removed in V3 (latency cost outweighed the retrieval gain);
the current differentiator is the four-channel weighted RRF fusion
(Text/Dense/Tree/k-NN) plus claim-level partial supersede tracking. Keep
this file in sync with `README.md` and `docs/RETRIEVAL.md` when the
retrieval design changes again.
