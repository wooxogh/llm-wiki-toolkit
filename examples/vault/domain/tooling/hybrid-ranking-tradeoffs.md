---
id: hybrid-ranking-tradeoffs
layer: domain
domain: tooling
projects: ["llm-wiki"]
tags: ["retrieval", "bm25", "dense", "rrf"]
confidence: confirmed
status: active
updated: 2026-01-10
summary: 'Why this vault fuses BM25 and dense cosine with Reciprocal Rank Fusion instead of picking one signal, and how the dense/sparse weights trade off.'
links: ["cross-encoder-rerank-effect", "embedding-store-layout"]
---

# Fusing BM25 and dense scores instead of picking one

A pure dense retriever (cosine over sentence embeddings) is good at "this
query and this page mean roughly the same thing" even when they share no
words. A pure sparse retriever (BM25) is good at the opposite case: an exact,
rare token — a field name, an error code, an id — that a dense model may
smear across many neighbours because it optimizes for topical similarity, not
lexical identity. Neither dominates the other across a mixed workload of
conceptual questions and exact-lookup questions, so this toolchain runs both
and fuses the two rankings rather than choosing a retriever ahead of time.

The fusion method is Reciprocal Rank Fusion (RRF): each candidate gets a
score from *where it ranks* in each list, not from the two lists' raw scores.
That matters because a cosine similarity and a BM25 score live on unrelated
scales — averaging them directly would let whichever score happens to have
the larger numeric range dominate the fusion regardless of how meaningful it
actually is. RRF avoids that by only ever looking at rank position:

```
fused_score(doc) += weight / (K + rank_in_list)
```

`K` is a damping constant (this codebase uses 60, the standard value from the
RRF literature) that keeps the score from blowing up for rank 0 and keeps the
tail of the list from being totally ignored. Because the score depends only
on ordinal rank, it is scale-free: it does not matter whether the dense
signal is cosine in `[-1, 1]` or the sparse signal is an unbounded BM25
score — both lists just contribute "how far down was this document" terms
that add cleanly.

The two weights are not required to be equal. This toolchain's default
weights favor the dense signal over the sparse one (roughly 2:1), on the
reasoning that most of a mixed conceptual/lookup workload is closer to
"restate this idea in different words" than "find this exact token" — so
dense should usually get first say, with sparse acting as a correction for
the queries dense alone would miss. Both weights are configurable per
deployment via environment variables rather than hard-coded, because that
ratio is a property of a given corpus's query mix, not a universal constant.

A consequence worth calling out: RRF ranks candidates *relative to each
other*, so it never tells a caller "how confident should I be in the top
hit" — the top result of a nonsense query and the top result of a
well-matched query can land in a similar narrow RRF-score band, because RRF
only ever measures relative position, never absolute closeness. A system
that needs to decide whether to answer automatically or say "I don't know"
has to ask a different, absolute-scale question — see
`cross-encoder-rerank-effect` for how the reranking step and the confidence
decision downstream of ranking are deliberately kept separate.

See also `embedding-store-layout` for what the dense side of this fusion
actually reads from disk.
