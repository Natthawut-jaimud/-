#!/usr/bin/env bash
# SessionStart hook: in Claude Code on the web, install the toolchain the fresh
# container lacks. No-op on a local machine (install ffmpeg yourself there).
set -uo pipefail
[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v ffmpeg >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1 \
    || echo "studio: ffmpeg install failed — run: apt-get install -y ffmpeg"
fi
if command -v uv >/dev/null; then
  (cd "$ROOT" && uv sync -q) || echo "studio: uv sync failed"
fi
exit 0
