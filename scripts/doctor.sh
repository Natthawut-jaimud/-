#!/usr/bin/env bash
# Check the studio's toolchain. Prints OK / MISSING per requirement.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ok()   { printf '  OK       %s\n' "$1"; }
miss() { printf '  MISSING  %s — %s\n' "$1" "$2"; FAIL=1; }

echo "Studio doctor"
command -v ffmpeg  >/dev/null && ok ffmpeg  || miss ffmpeg  "macOS: brew install ffmpeg | Debian/Ubuntu: sudo apt-get install -y ffmpeg"
command -v ffprobe >/dev/null && ok ffprobe || miss ffprobe "ships with ffmpeg"

if command -v node >/dev/null; then
  major="$(node -p 'process.versions.node.split(".")[0]')"
  [ "$major" -ge 22 ] && ok "node $(node -v)" || miss "node >= 22" "found $(node -v)"
else
  miss "node >= 22" "https://nodejs.org"
fi

PY="python3"; [ -x "$ROOT/.venv/bin/python" ] && PY="$ROOT/.venv/bin/python"
"$PY" -c 'import requests, librosa, matplotlib, PIL, numpy' 2>/dev/null \
  && ok "python deps ($PY)" || miss "python deps" "run: uv sync   (or pip install -e .)"

if [ -n "${ELEVENLABS_API_KEY:-}" ] || grep -q '^ELEVENLABS_API_KEY=..' "$ROOT/.env" 2>/dev/null; then
  ok "ELEVENLABS_API_KEY (Scribe transcription)"
else
  echo "  OPTIONAL ELEVENLABS_API_KEY not set — will fall back to local Whisper (fillers partly lost)"
fi

command -v yt-dlp >/dev/null && ok yt-dlp || echo "  OPTIONAL yt-dlp (only for downloading sources from URLs)"
exit "${FAIL:-0}"
