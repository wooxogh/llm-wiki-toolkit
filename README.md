# LLM-Wiki V3

LLM-Wiki V3 is a local-first Markdown knowledge base for Codex and Claude Code.
It chunks documents with a bundled structural + semantic Chunker V3, then
combines BM25, Qwen dense retrieval, a directory/heading tree, and a chunk k-NN
graph. The indexed Markdown directory can live anywhere on the machine.

> Status: alpha. Keep source documents under version control and inspect
> hygiene decisions before applying them.

## Requirements

- Python 3.11 through 3.13
- About 1.2 GB for the Qwen embedding model downloaded on first use
- CUDA is used automatically when PyTorch can see a compatible GPU; CPU works
  but initial embedding is substantially slower

The 7.4 MB Small-V3 Attention checkpoint is included in the package. No API key
is required for indexing or retrieval. `--rerank` downloads and runs the local
`BAAI/bge-reranker-v2-m3` model unless `WIKI_RERANK_MODEL` is set.

## Install

```bash
git clone <repository-url>
cd llm-wiki-v3
python -m pip install ".[ml]"
```

For development:

```bash
python -m pip install -e ".[ml,dev]"
python -m pytest
```

On Windows, use `python -m llm_wiki_v3.indexing` if the Python `Scripts`
directory containing `wiki-embed.exe` is not on `PATH`.

## Configure a vault

Copy `wiki.toml.example` to `wiki.toml` inside any Markdown directory. Run the
commands there, pass `--vault`, or set `WIKI_VAULT` to that directory.

```toml
[vault]
root = "."

[v3]
artifact_dir = ".llm_wiki_v3"
model_id = "Qwen/Qwen3-Embedding-0.6B"
embed_device = "auto"
chunk_boundary_keep_threshold = 0.66
chunk_candidate_budget = 0.50
```

`chunk_boundary_keep_threshold` controls how readily Small Attention preserves
a semantic boundary: lower values create finer chunks. `chunk_candidate_budget`
is the fraction of sentence gaps considered by the Gate V2 candidate stage.
The model-validated defaults are recommended until a vault-specific evaluation
shows otherwise.

## Commands

```bash
wiki-embed --vault /path/to/markdown
wiki-search "how is retry handled?" --vault /path/to/markdown --auto
wiki-search "recent migration decisions" --vault /path/to/markdown --range 2 --rerank
wiki-health --vault /path/to/markdown
wiki-eval --vault /path/to/markdown --gold eval_gold_v3.json
```

- `wiki-embed` incrementally builds chunks, vectors, BM25 data, tree nodes, and
  k-NN links under `.llm_wiki_v3/`.
- `wiki-search` fuses text, dense, tree, and k-NN evidence with weighted RRF.
  `--auto` reports answer/review/none confidence; `--range N` limits source age.
- `wiki-health` checks artifact and provenance integrity. Its `review` command
  only prepares evidence for an agent and user; it does not decide truth.
- `wiki-eval` runs retrieval ablations against a maintained gold query file.
  It is an evaluation tool, not an automatic judge of document correctness.

Run `wiki-search --help`, `wiki-health --help`, or `wiki-eval --help` for all
options. Derived indexes can be deleted and rebuilt; Markdown remains the
source of record.

## Codex and Claude skill

The same skill definition is packaged for both hosts:

```bash
python -m llm_wiki_v3.skill_install --provider codex
python -m llm_wiki_v3.skill_install --provider claude
python -m llm_wiki_v3.skill_install --provider both
```

Restart the host after installation. The skill instructs the host LLM to use
the CLI as evidence retrieval and to ask the user before recording a hygiene
decision.

## Hygiene contract

The package never decides whether claims contradict or which claim is correct.
`wiki-health review --json` emits provenance-rich comparison material for the
host skill. Only a decision JSON containing `user_approved=true` can be applied.

- Partial supersession keeps the old chunk searchable, records the affected
  quote, and attaches the successor whenever that chunk is returned.
- A factual correction creates a new Markdown source under
  `_wiki_corrections/`, links it to the prior chunk, and excludes the retracted
  chunk from normal search.
- The original Markdown is not rewritten by the correction workflow.

See `skill/llm-wiki-v3/SKILL.md` for the agent workflow and `NOTICE.md` for model
and dataset provenance.

## Release verification

```bash
python -m build
python -m pip install --force-reinstall "dist/llm_wiki_v3-0.1.0-py3-none-any.whl[ml]"
python -m llm_wiki_v3.skill_install --help
```

The wheel includes Chunker V3 source, the Small Attention checkpoint, its
metadata, and the Codex/Claude skill. It does not require the original research
workspace or its `chunk_model` and `embedding_test` directories.

## License

Project code is MIT licensed. Downloaded models and training data retain their
own licenses; see `NOTICE.md`.
