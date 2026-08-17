# Knowledge hygiene

Five passes keep the *content* honest, as opposed to the artifacts consistent
(that is `wiki-health`'s job). Three of them are two-stage by design: a cheap
deterministic detector produces **candidates**, and a human — or an LLM session
acting as one — reads the pages and decides.

**Nothing in this directory rewrites a page automatically.** That is the single
rule the whole subsystem is built around. A false positive that silently edits a
page destroys real knowledge; a false positive on a candidate list costs someone
thirty seconds.

| pass | command | needs `.embeddings/`? | output |
|---|---|---|---|
| claim lint | `wiki-lint` | no | stdout; also a `wiki-health` warning |
| link-graph health | `python -m llm_wiki.reports.graph_report` | no | `GRAPH_REPORT.md` |
| community synthesis | `python -m llm_wiki.reports.community_report` | no | `COMMUNITIES.md` + `community_summaries.json` |
| contradiction | `python -m llm_wiki.hygiene.contradict` | **yes** | `.contradictions.md` |
| compaction | `python -m llm_wiki.hygiene.compact` | **yes** | `.compaction.md` |

## Claim lint

The founding rule is *measured facts only*. Every other invariant here is
machine-checked — frontmatter schema, index drift, embedding staleness, community
synthesis, retrieval quality — while the one rule the vault is actually *about*
was left entirely to discipline. `claim_lint` closes that gap in the cheapest way
that still helps: it looks for the **shape** of an unmeasured claim, which is
hedging language with no number, date, PR reference, or code path near it.

```bash
wiki-lint                 # whole vault
wiki-lint --page <id>     # one page
```

### Why it is a warning and always will be

- Hedging is legitimate when a page records uncertainty *on purpose* ("the retry
  ceiling under real load is unverified"). That is a measured statement about the
  absence of measurement, and packs carry a `deliberate_uncertainty` axis that
  suppresses it.
- A gate that blocks a commit on prose gets worked around, not obeyed.
- It is a heuristic. **It cannot catch a confident fabrication at all** — a
  fluent, specific, entirely invented sentence has none of the shape it looks
  for. That needs a reader, not a regex.

Read the flagged lines. Do not bulk-edit them.

### What counts as evidence

A language-neutral core: ISO dates, ratios (`29/31`), percentages, grouped
thousands, `#1234` PR references, code paths in backticks, and numbers with
units.

Units are folded into a **digit-anchored** group rather than matched
free-standing. A bare unit word — the Korean `건` is a common counter that also
means "case/incident" — would otherwise match ordinary prose everywhere and
reintroduce exactly the false positives the lint exists to avoid. A number must
actually be attached.

Evidence is accepted on the flagged line **or within one line of it**, because
the house style this was built for puts the claim on one line and the measurement
on the next.

Struck-through text (`~~…~~`), fenced code blocks, and blockquoted lines (`>`)
are excluded: retracted text is not a live claim, and quoted speech is data.

### Language packs

Packs live in `src/llm_wiki/hygiene/claim_lint/patterns/<name>.toml` and are
selected by `[lint] packs` in `wiki.toml`. `en` and `ko` ship. Packs **compose**:
`packs = ["en", "ko"]` lints both languages in a single pass. An unknown pack
name is a hard error.

A pack is up to five regex alternations:

| key | meaning |
|---|---|
| `hedge` | the guess-shaped vocabulary |
| `deliberate_uncertainty` | phrases that mark uncertainty on purpose; suppress the flag |
| `evidence` | language-specific evidence words, added to the universal set |
| `evidence_units` | unit words, folded into the digit-anchored group |
| `literal_sight` | optional homonym carve-out (see below) |

To add a language, write `<name>.toml` and add the name to `[lint] packs`. Two
things to get right:

1. **Scope your inline flags.** Write `(?i:...)`, never a leading `(?i)`. Packs
   are concatenated with `|`, and Python rejects a global inline flag anywhere
   but the start of the compiled pattern.
2. **Check your homonyms.** The `literal_sight` axis exists because the Korean
   `보인다` both hedges ("seems to be") and states literal visibility ("X is
   visible"). When a line's only hedge signal disappears once the sight-reading
   is removed, it is an assertion about visibility, not a guess.

## Link-graph health

`graph_report` treats the vault as the graph it already is: pages are nodes,
`[[id]]` wikilinks and frontmatter `links:` entries are edges. Code is not
indexed, by design.

It surfaces dangling links (a `[[id]]` with no target — write the page, or fix
the typo), orphans (no inbound link, so unreachable by traversal), islands (no
links either way), god-nodes (the highest-inbound hubs, which are split-or-index
candidates), communities, and bridge edges that cross a domain or layer boundary.

All hygiene signals, no auto-rewrite. `[[x]]` written inside fenced or inline
code is stripped first, so a page documenting the link schema does not create
phantom edges.

## Community synthesis

`community_report` is a GraphRAG-style summary layer adapted to an offline,
key-free vault, in two cleanly separated halves:

1. **Extractive (deterministic).** For each link-graph community, aggregate the
   member pages' own `summary:` lines. Zero hallucination — it only restates
   facts already on the pages.
2. **Abstractive (optional sidecar).** A one-to-two-sentence synthesis per
   community, stored in `community_summaries.json` keyed by a **signature**: a
   short hash of the sorted member ids.

The signature is the interesting part. When membership changes, the signature
changes, so a synthesis written for the old membership is dropped and the
community is flagged as awaiting synthesis rather than served as if it still
described its members. A stale synthesis is a confidently-wrong answer, not a
gap, so `wiki-health` treats it as an **error**.

### The first-run trap

This is what a new vault hits, and it looks like a bug the first time.

The writing guidance says: give every new page at least one inbound link, so it
is discoverable. Follow it consistently and your pages become one connected
component. A component of 3 or more members (`MIN_SYNTH_SIZE`) is a community
that *requires* a synthesis. So a healthy, well-linked small vault fails the
health gate until you write the sidecar:

```
$ wiki-health --mode ci
❌ [community-synthesis-stale] 1 community/communities await a grounded synthesis: vault(84ebd5e58fee) — add community_summaries.json at the vault's content root, keyed by the signature shown above (e.g. "84ebd5e58fee"), with a short grounded synthesis of that community's member pages as the value
❌ unhealthy (mode=ci, 1 error(s), 1 warning(s))
```

The fix is a JSON object mapping signature → synthesis at the vault's content root:

```json
{
  "84ebd5e58fee": "This vault's nine pages form one connected community: ..."
}
```

`examples/vault/community_summaries.json` is a complete worked example for the
shipped 9-page vault, and re-running `wiki-health --mode ci` with it present exits
0.

Workflow when the membership shifts later:

```bash
python -m llm_wiki.reports.community_report --stale   # prints only what needs writing
```

Read those communities' member pages, write a grounded synthesis for each printed
signature, and re-run. The daily ingest pipeline treats any output from that
command as fatal (see [INGEST.md](INGEST.md)), which is why a stale synthesis can
never quietly ship.

Two properties worth knowing:

- **Communities of size ≤ 2 never require a synthesis.** Verified 2026-07-24:
  member-recall@3n is 1.0 for those — retrieval already finds both members and
  reading two pages is trivial, so an abstractive synthesis adds ~0. An existing
  synthesis still renders regardless of size.
- **A missing sidecar is not an error in itself.** A vault whose communities are
  all small renders fine with no `community_summaries.json` at all.

## Contradiction candidates

```bash
python -m llm_wiki.hygiene.contradict --tau 0.78              # full audit
python -m llm_wiki.hygiene.contradict --changed id1,id2 --tau 0.75   # incremental, for ingest
```

Method: a page vector is the renormalised mean of its chunk embeddings; pairs
above a cosine threshold are emitted to `.contradictions.md`. High similarity
means *same topic*, which means *worth checking* — one page's corrected fact
versus another page still asserting the old one.

Two filters improve the signal: pairs are ordered so that any pair touching a
page carrying a correction marker (a strikethrough, a retraction word in either
language) comes first, and "companion" pairs — ids sharing their first two
kebab-case tokens, i.e. sibling documents about the same thing — are skipped,
because they are *supposed* to be similar.

### Judging a candidate

Most candidates are **not** contradictions. They are companions: different scope,
different aspect, or normal chronological progression. Precision comes from this
stage, not from the detector.

A real contradiction is the same fact asserted in opposite directions. When you
find one:

- Decide which side is current **on the evidence** — a live measurement or the
  primary artifact wins. The `updated` date is a secondary signal, not the rule.
  "Newer, therefore right" is how a careless correction becomes canon.
- On the stale page, strike the wrong claim through **and say why**. Do not
  delete it. If the page is wholly superseded, set `status: superseded` and
  `links:` to the successor.
- Add a `[[stale-id]]` backlink from the current page so the relationship is
  recorded.
- Re-run `wiki-index` and `wiki-embed`.

When in doubt, do nothing. A false positive that erases real knowledge is the
worst outcome available here.

## Compaction candidates

```bash
python -m llm_wiki.hygiene.compact --merge-tau 0.85 --size-kb 12 --corrections 6
```

Three decay signals, written to `.compaction.md`:

**UPDATE — fold resolved retractions** (pages with ≥ 6 correction markers).
Highest return of the three. Only fold corrections that are *fully settled*:
rewrite the body as current truth and compress "what was overturned, when, why"
into a trailing changelog section. Twenty inline strikethroughs become a clean
body plus five changelog lines, with zero information lost. Leave anything still
uncertain or in progress inline — an unresolved correction is live content.

**MERGE — redundant overlap.** Requires **both** a high page-vector cosine and
real textual redundancy: at least 30% of the *shorter* page's substantial
sentences (≥ 40 characters) appearing verbatim in the other.

Cosine alone was tried first and had **0% precision — 19 of 19 false positives**.
Measured 2026-07-31: every pair at cosine ≥ 0.85 shared at most 2 identical
sentences out of 48-116. A topically dense corpus makes cosine track subject
matter, so two verdict pages about the same investigation read alike without
restating each other. Normalising by the shorter page catches the case actually
worth merging: one page wholly contained in another.

When judging: genuinely duplicated → fold into the more complete page, mark the
other `status: superseded` with a link, and absorb any unique detail first.
Complementary → leave both alone. Most candidates are the latter.

**SPLIT — oversized pages** (over 12 KB). Read it: a long single-topic page is
fine, so the answer is often "leave it". Split only when several independent
topics have accumulated in one file, and cross-link the results.

Compaction is a **deliberate periodic pass**, not a daily one. Compaction carries
cost and risk, so being lazy about it is the correct default; the detectors are
cheap to re-run whenever you decide to do one.

### Principles shared by both passes

- **Conservative**: when ambiguous, do nothing. Do not flatten a subtle measured
  detail into a summary.
- **Preserve facts**: never delete. Strike through, keep a changelog, archive.
- **Evidence first**: the live measurement or the primary artifact is canonical,
  not whichever page was edited most recently.
