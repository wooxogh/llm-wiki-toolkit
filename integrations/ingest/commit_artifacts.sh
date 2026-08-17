#!/bin/sh
# Commit whatever the deterministic hygiene steps regenerated.
#
# SAFETY CONTRACT: the pipeline's `preflight` step guarantees the working tree
# was clean before this run started, so everything staged here was produced by
# this run. Do NOT call this script outside that pipeline — from a dirty tree it
# would fold a human's uncommitted work into a "daily refresh" commit.
#
# Exits 0 when there is nothing to commit — a quiet day (no new knowledge) is a
# success, not a failure. A `git add` failure is NOT swallowed: it means the
# working tree is not in the shape we assumed, and the pipeline must stop
# without stamping so the next tick retries.
set -eu
VAULT="${WIKI_VAULT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$VAULT"

# Only stage paths that exist — `git add` on a missing path is a hard error under
# `set -e`, and an absent optional artifact is not a failure.
PATHS=""
for p in domain patterns entities raw index.yaml GRAPH_REPORT.md COMMUNITIES.md \
         community_summaries.json log.md; do
  [ -e "$p" ] && PATHS="$PATHS $p"
done
[ -n "$PATHS" ] || { echo "nothing to stage"; exit 0; }

# shellcheck disable=SC2086  # word splitting is intended for the path list
git add -A -- $PATHS

if git diff --cached --quiet; then
  echo "nothing to commit (quiet day)"
  exit 0
fi

git commit -q -m "chore(ingest): daily wiki refresh $(date +%Y-%m-%d)"
echo "committed daily wiki refresh"
