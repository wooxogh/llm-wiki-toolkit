#!/bin/sh
# Install the optional macOS LaunchAgents for the resident embed server and the
# daily ingest pipeline.
#
# macOS only: LaunchAgents are a launchd mechanism that does not exist on any
# other platform, so this refuses to run elsewhere rather than writing a plist
# that would silently do nothing.
#
# Usage: integrations/macos/install.sh
#   WIKI_VAULT  optional; defaults to this checkout's root (two levels up)
#   WIKI_PYTHON optional; defaults to the first `python3` on PATH
set -eu

if [ "$(uname -s)" != "Darwin" ]; then
  echo "install.sh: LaunchAgents are macOS-only; this host reports $(uname -s). Refusing to install." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VAULT="${WIKI_VAULT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON="${WIKI_PYTHON:-$(command -v python3 || true)}"

if [ -z "$PYTHON" ]; then
  echo "install.sh: no python3 found on PATH; set WIKI_PYTHON to an explicit interpreter path." >&2
  exit 1
fi

DEST="$HOME/Library/LaunchAgents"
mkdir -p "$DEST"

render() {
  # $1 = template stem under this directory, $2 = installed Label/filename
  sed -e "s#__VAULT__#$VAULT#g" -e "s#__PYTHON__#$PYTHON#g" \
      "$SCRIPT_DIR/$1.plist.template" > "$DEST/$2.plist"
  echo "wrote $DEST/$2.plist"
}

render embed-server com.llm-wiki.embed-server
render ingest com.llm-wiki.ingest

for label in com.llm-wiki.embed-server com.llm-wiki.ingest; do
  launchctl load "$DEST/$label.plist"
  # `launchctl load` has been observed to print its own error to stderr (e.g.
  # "Load failed: 5: Input/output error") and still exit 0, so its exit status
  # alone cannot be trusted as the success signal. Confirm the agent is
  # actually registered before reporting success.
  if launchctl list "$label" >/dev/null 2>&1; then
    echo "loaded $label"
  else
    echo "install.sh: launchctl load reported success but $label is not registered — see launchctl's own error above." >&2
    exit 1
  fi
done
