# Ingest

The playbook for keeping a vault current, and the orchestrator that runs it
unattended.

Ingest is **optional**. Everything under `integrations/` can be deleted and the
core still works — nothing in `src/llm_wiki/` imports it. What follows is one
worked arrangement: a scheduled session that reads what changed in the
repositories you work in, writes it up as vault pages, and then rebuilds every
derived artifact deterministically.

The authoring step is done by an agent CLI session, so no API key is involved.
The pipeline itself is pure Python and shells out to the same commands you would
run by hand. Claude remains the default authoring CLI for backward
compatibility; Codex is supported by setting `[ingest] agent = "codex"` or by
passing `--agent codex` to the orchestrator.

## The procedure

### 1. Collect

For each repository listed under `[ingest] repos` in `wiki.toml`, scan commits
since the last run:

```bash
git -C <repo> log --since="<last-run>" --oneline
```

If your agent runtime keeps a per-project memory directory, check it for new or
changed files too.

> **Skip pointer stubs.** If you use the `sync_cache` integration, files it has
> already promoted are rewritten as thin pointers back into the vault. Those are
> a cache *pointing at* canonical knowledge, not new knowledge, and re-promoting
> them creates a loop. Filter them out before step 2 — `grep -L` for the pointer
> marker gives you the files that are *not* pointers.

### 2. Decide

For each genuinely new fact, decide whether it belongs in the vault at all:

- Durable, portable domain finding or state → `domain/<area>/`
- Reusable process or methodology lesson → `patterns/`
- Transient task tracking, or personal working-style preference → **not the
  vault.** That belongs in your session/personal memory tier.
- **If the id already exists** this is not a promotion, it is an *update* to that
  page, under the correction rules below.

### 3. Author

Create or update the page. The rules are the vault's, not the pipeline's (see
[AGENTS.md](../AGENTS.md#writing)):

- filename == `id`; fill every required frontmatter field; list every relevant
  repository in `projects`.
- **Measured facts only.** If it was not observed, it does not go in.
- Correcting an existing page keeps the old claim: strike it through and say why
  it was wrong.
- Link related pages with `[[id]]`.
- If new information makes an existing page obsolete, set `status: superseded` on
  it and point `links:` at the successor.

### 4. Regenerate

```bash
wiki-index                                            # index.yaml + validation
wiki-embed                                            # vectors (incremental; weekly full rebuild)
python -m llm_wiki.reports.graph_report --write       # GRAPH_REPORT.md
python -m llm_wiki.reports.community_report --write   # COMMUNITIES.md
python -m integrations.agent_memory.sync_cache --project <abs-path> --target codex   # optional Codex pointer cache
python -m integrations.agent_memory.sync_cache --project <abs-path> --target both    # migrate Claude + Codex together
```

- If `wiki-index` prints a **dangling wikilink** warning, check whether that
  `[[id]]` is a real page: fix the typo, or remove the brackets if it refers to
  something outside the vault (a personal memory file is not a vault node).
- If the **orphan** ratio is climbing, add an inbound `[[new-id]]` link from a
  related page so new work is discoverable.
- For every community printed by `community_report --stale`, read its member
  pages and write a grounded one-to-two-sentence synthesis into
  `community_summaries.json` under that signature — **from what you read, nothing
  else**. Membership changes change the signature, so old syntheses retire
  themselves and you only ever have to handle `--stale`.

### 5. Check for contradictions (optional)

```bash
python -m llm_wiki.hygiene.contradict --changed <ids> --tau 0.75
```

Read both pages of any candidate pair and act only on a real conflict. See
[HYGIENE.md](HYGIENE.md#contradiction-candidates).

### 6. Log

Add one line at the top of `log.md`: `## [YYYY-MM-DD] ingest | <summary>`.

### 7. Commit

**The pipeline does this.** An authoring session touches pages,
`community_summaries.json`, and `log.md` — and stops there. It does not run the
regeneration commands and does not commit.

## The orchestrator

`integrations/ingest/ingest_pipeline.py` owns step order, fail-fast behaviour,
and the success stamp. `run_ingest.sh` is a deliberately thin wrapper that sets
`PATH` and logging and nothing else.

```
preflight → llm → build → embed → graph → community → stale → health → commit
                                                                    │
                                              ── success stamp written here ──
                                                                    │
                                                                  push   (failure keeps the stamp)
```

```bash
python -m integrations.ingest.ingest_pipeline --dry-run --skip-llm  # print the exact commands, change nothing
python -m integrations.ingest.ingest_pipeline --skip-llm            # run only the deterministic steps
python -m integrations.ingest.ingest_pipeline                       # the daily run
python -m integrations.ingest.ingest_pipeline --agent codex         # use Codex for the authoring step
```

For Codex, the `llm` step uses `codex exec` in the vault workspace with
`--sandbox workspace-write` and `--ask-for-approval never`. Every path listed in
`[ingest] repos` is also passed as `--add-dir`, so Codex can inspect the source
repositories while writing only the vault artifacts the prompt asks it to write.

### Failure semantics

These are the parts worth reading twice, because each replaced something that
failed quietly.

- **The stamp is written last, and only after everything passed.** Any non-zero
  step aborts the run, skips the remaining steps, and leaves the stamp unwritten,
  so the next tick retries the whole day. An earlier shell version stamped
  success based on the authoring agent's exit code alone — so a day where the
  index or the embedding store failed to rebuild was recorded as done and never
  retried.
- **`stale` is fatal on output, not on exit code.** `community_report --stale`
  exits 0 while printing the communities that still need a synthesis. Checking
  only the exit code would let a stale synthesis through.
- **`preflight` refuses to start on a dirty working tree.** The commit step
  stages whole content directories, so from a dirty start it cannot tell its own
  output from a human's work in progress, and would fold uncommitted edits into a
  "daily refresh" commit. Starting clean is what makes "everything dirty at the
  end is ours" true.
- **`health` (`wiki-health --mode full`) is the final gate** for index,
  embedding, report, and community drift.
- **A failed `push` does not clear the stamp.** The knowledge is already
  committed locally; only the off-machine backup is behind. Re-running a full
  ingest tomorrow because of a network blip would be the wrong response, and the
  next successful cycle carries the backlog up anyway.
- **A stamp for today skips everything**, so a scheduler that wakes several times
  a day is safe.
- **A quiet day is a success.** The commit step exits 0 when there is nothing to
  commit. A `git add` failure, on the other hand, is not swallowed: it means the
  tree is not in the shape the pipeline assumed, so it stops without stamping.
- **The authoring step is omitted entirely when `[ingest] repos` is empty** —
  which is the default. Running an agent against no repositories would be a
  no-op with a model bill attached.

### Scheduling

`integrations/macos/install.sh` renders and loads two macOS LaunchAgents (the
daily ingest and the resident embedding server) from the templates beside it. It
refuses to run on any other platform rather than writing a plist that would
silently do nothing, and it verifies each agent is actually registered afterwards
— `launchctl load` has been observed to print an error and still exit 0.

On other platforms, call `run_ingest.sh` from cron or any equivalent scheduler.
The pipeline's own once-a-day stamp makes over-scheduling harmless.

## Not adopted: a fully unattended authoring step

The authoring step could be moved to a standalone script driving a model API
directly, which would remove the dependency on an interactive agent CLI. It has
not been done, and it would introduce the first API key in the system. Recorded
here as an option, not a recommendation.
