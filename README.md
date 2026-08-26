# LLM-Wiki V3

LLM-Wiki V3 is a local-first Markdown knowledge base for Codex and Claude Code.
It chunks documents with a bundled structural + semantic Chunker V3, then
combines BM25, Qwen dense retrieval, a directory/heading tree, and a chunk k-NN
graph. The indexed Markdown directory can live anywhere on the machine.

For a Korean walkthrough from the product-level mental model down to formulas,
artifacts, invariants, and source modules, see
[`docs/ARCHITECTURE_KO.md`](docs/ARCHITECTURE_KO.md).

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
git clone https://github.com/wooxogh/llm-wiki-toolkit.git
cd llm-wiki-toolkit/llm-wiki-v3
python -m pip install ".[ml]"
```

For development:

```bash
python -m pip install -e ".[ml,dev]"
python -m pytest
```

On Windows, use `python -m llm_wiki_v3.indexing` if the Python `Scripts`
directory containing `wiki-embed.exe` is not on `PATH`.

### Existing V1/V2 installations

V3 uses the canonical `wiki-*` commands. Do not install V1/V2 and V3 into the
same Python environment: they share names such as `wiki-embed`, and the most
recent installation replaces the console script. Use a dedicated environment
for V3 or remove the older package first. The module form is also useful in
automation or for a one-off coexistence diagnosis:

```powershell
python -m llm_wiki_v3.indexing --vault C:\path\to\markdown
```

## Hugging Face authentication and model cache

The Qwen embedding model is public, so a Hugging Face token is not required.
If Hugging Face prints an unauthenticated-request warning or asks for login,
authenticate once in the same Python installation used for the `wiki-*`
commands:

```powershell
python -m pip install -U huggingface_hub
hf auth login
hf auth whoami
```

`hf auth login` stores the credential in the user-level Hugging Face cache.
The token is not saved in the vault, `wiki.toml`, or this repository, and the
subsequent `wiki-embed` and `wiki-search` processes reuse that stored
credential automatically. Do not paste the token into chat, source files, or
the vault.

Model weights are cached separately from credentials. The first command may
download the model; later commands create a new in-process model object for
their own query/indexing work, but should reuse the local model files instead
of downloading them again.

After the model has downloaded successfully, an offline session can prevent
all Hugging Face network checks:

```powershell
$env:HF_HUB_OFFLINE = "1"
wiki-search "your question" --vault C:\path\to\markdown
```

Only enable offline mode after the required model files are present locally.
Unset it with `Remove-Item Env:HF_HUB_OFFLINE` when downloading or updating a
model.

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
wiki-search "how is retry handled?" --vault /path/to/markdown --auto --json
wiki-search "recent migration decisions" --vault /path/to/markdown --range 2 --rerank
wiki-health --vault /path/to/markdown
wiki-eval --vault /path/to/markdown --gold eval_gold_v3.json
```

- `wiki-embed` incrementally builds chunks, vectors, BM25 data, tree nodes, and
  k-NN links under `.llm_wiki_v3/`.
- `wiki-search` fuses text, dense, tree, and k-NN evidence with weighted RRF.
  `--auto` reports answer/review/none retrieval confidence; `--range N` limits
  source age. The host LLM assesses evidence rather than treating this as a
  factual decision.
- `wiki-health` checks artifact and provenance integrity. Its `review` command
  only prepares evidence for an agent and user; it does not decide truth.
- `wiki-eval` runs retrieval ablations against a maintained gold query file.
  It is an evaluation tool, not an automatic judge of document correctness.

### Resident model runtime and automatic embedding

For repeated CLI searches, start one local daemon per vault. It owns the Qwen
embedding model, the Chunker V3 runtime, the optional reranker, and the loaded
retrieval index. It listens only on `127.0.0.1`; its connection state is stored
under `.llm_wiki_v3/daemon.json` and is not committed source material.

```powershell
# Keep this terminal open, or add --background to detach it.
wiki-daemon start --vault C:\path\to\markdown

# In another terminal, start the Markdown watcher.
wiki-autoembed start --vault C:\path\to\markdown --initial

# Existing commands use the daemon automatically while it is available.
wiki-search "how is retry handled?" --vault C:\path\to\markdown
wiki-embed --vault C:\path\to\markdown

wiki-autoembed status --vault C:\path\to\markdown
wiki-autoembed logs --vault C:\path\to\markdown
wiki-autoembed stop --vault C:\path\to\markdown
wiki-daemon stop --vault C:\path\to\markdown
```

`wiki-autoembed` polls recursively for Markdown additions, edits, renames, and
deletions, then waits two seconds by default to coalesce editor save events.
It never loads a model itself: it asks `wiki-daemon` to run the existing
incremental `wiki-embed` pipeline with the resident Qwen instance. Derived
directories such as `.llm_wiki_v3/` are ignored; `_wiki_corrections/` is watched
as normal source Markdown. If the daemon is restarted while changes are queued,
the watcher retains those paths and retries automatically.

In foreground mode, every detected Markdown change and completed incremental
embed is printed in that watcher terminal. A background process cannot print
into the terminal that launched it, so the same events are appended to
`.llm_wiki_v3/autoembed.log`; use `wiki-autoembed logs` to inspect them.
`wiki-daemon status --json` reports a stable runtime ID and whether Qwen,
Small-V3, and the optional reranker are currently loaded. During normal daemon
operation that runtime ID must remain unchanged across automatic embeds.

`wiki-daemon start --background` waits for Qwen and Small-V3 to finish loading
before it returns. `wiki-autoembed --initial` queues its first build and keeps
retrying if the daemon is briefly unavailable, so the two commands can be run
one after another from the same vault directory without a startup race.

When no daemon is running, `wiki-search` and `wiki-embed` retain their original
single-process behavior. Use `--no-daemon` to force that diagnostic path even
when a daemon is available. `wiki-daemon start --background` and
`wiki-autoembed start --background` run detached; use the `status` and `stop`
commands to manage them.

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
