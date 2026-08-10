# Contributing

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests -q      # 311 passed, 1 skipped
```

The one skip is the Kiwi tokenizer test, which is `importorskip`-guarded on an
optional dependency that is in no extra.

Add `".[ml]"` only if you are changing something that actually needs the model
(embedding, reranking, or a measurement run). You do not need it to run the
suite, and the suite must never need it — see below.

## The test boundary

`tests/` must not import torch or sentence-transformers, reach the network, or
touch a resident embedding server. The whole ranking, policy, health, and
evaluation layer is pure and is exercised with injected scores; the model lives
behind `retrieval/_embedder.py` and `retrieval/_rerank.py` precisely so CI can
run without a GPU. CI enforces this with `-X importtime`. If a change makes a
test need the model, the change is in the wrong layer.

Check it the way CI does:

```bash
.venv/bin/python -X importtime -m pytest tests -q 2> imports.log
grep -cE 'torch|sentence_transformers' imports.log     # must print 0
```

**Caveat worth knowing:** `-X importtime` instruments only the pytest process. A
test that spawns a subprocess escapes the guard entirely — the child's imports
never appear in that trace. `tests/test_eval.py` does exactly this (it runs the
real CLI through `sys.executable -m ...` because the vault root is resolved once
at import time and cannot be monkeypatched afterwards). If you add a
subprocess-based test, you have to reason about the child's import graph by hand;
the automated check will not do it for you.

## Layering

The seams are load-bearing. Keep them:

- **`llm_wiki.paths` / `llm_wiki.config`** — pure stdlib. Imported by every
  lightweight path, so they must never pull in numpy, torch, or
  sentence-transformers.
- **`retrieval/_retrieve.py`** — `rank_from_scores()` is pure: score arrays in,
  ranking out. `search()` and `search_with_confidence()` are the entry points
  that embed a query and load the store. New ranking logic belongs on the pure
  side of that line.
- **`retrieval/retrieval_policy.py`** — the answer/review/none decision, testable
  with literal scores. It must stay free of retrieval imports.
- **`retrieval/_embedder.py` and `_rerank.py`** — the only modules that import
  torch, both lazily inside a function.
- **`integrations/`** — optional. The core must work with the whole directory
  deleted; nothing under `src/llm_wiki/` may import from it.

## Before you push a ranking change

Anything touching `_retrieve.py`, `embed_index.py`, `retrieval_policy.py`, the
chunker, or the gold set changes measured behaviour that the unit tests cannot
see. Run the regression gate (this one *does* need the `ml` extra and a vault
with an embedding store):

```bash
wiki-gate
```

It exits 1 when any per-layer Hit@k drops beyond tolerance, or when a single
gated false answer appears. Do not update the baseline to make it pass. Update it
only with a measured justification, recorded in the commit message.

## Documentation rules

This project's founding rule is *measured facts only*, and it ships a linter that
flags hedging carrying no measurement. Documentation that overstates what the
code does would be self-refuting.

- Every claim in a doc must be traceable to code in `src/`, to a test, or to a
  measurement recorded alongside its date and method.
- Carry measured figures across exactly. Do not round them for readability, and
  do not attach a number to a claim that never had one.
- When something was tried and rejected, record what it measured, not just that
  it was rejected.
- Corrections keep the old claim visible, with the reason it was wrong.

## Commits and PRs

- Small, single-purpose commits with a message that says *why*.
- New behaviour arrives with a test; a bug fix arrives with the test that would
  have caught it.
- Do not commit generated artifacts for a vault you do not own, and do not commit
  `.embeddings/` (it is gitignored — it is a local cache, not a shared artifact).
