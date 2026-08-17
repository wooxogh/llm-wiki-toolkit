#!/bin/sh
# launchd/cron entry point for the daily wiki ingest.
#
# Deliberately thin: PATH setup and logging only. All ordering, fail-fast, and
# stamping logic lives in integrations/ingest/ingest_pipeline.py, where it is
# unit-tested — a shell version that carried this logic inline once stamped
# success based on the authoring agent's exit code alone, so a failed index or
# embedding rebuild still counted as a good day.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
VAULT="${WIKI_VAULT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LOG="$HOME/.local/share/llm-wiki/ingest.log"
mkdir -p "$(dirname "$LOG")"

{
  echo "== ingest $(date) =="
  cd "$VAULT" && exec python3 -m integrations.ingest.ingest_pipeline --vault "$VAULT" "$@"
} >> "$LOG" 2>&1
