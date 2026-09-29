#!/usr/bin/env bash
# Re-vendor the studio's skills from upstream.
#   video-use   https://github.com/browser-use/video-use   (MIT)
#   hyperframes https://github.com/heygen-com/hyperframes  (Apache-2.0)
# Usage: scripts/sync-skills.sh   (run from anywhere; overwrites .claude/skills/<name>)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/.claude/skills"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Published HyperFrames skills needed for: captions, overlays on talking-head
# footage, standalone motion graphics, and the core/animation/CLI layers they use.
HF_SKILLS=(
  hyperframes hyperframes-core hyperframes-animation hyperframes-keyframes
  hyperframes-creative hyperframes-cli hyperframes-registry hyperframes-audio
  media-use talking-head-recut embedded-captions motion-graphics general-video
)
# Repo-internal HyperFrames doctrine skills (motion + caption craft).
HF_INTERNAL_SKILLS=(motion-doctrine cut-the-curve seam-craft captions-overlay)

git clone --quiet --depth 1 https://github.com/browser-use/video-use "$TMP/video-use"
git clone --quiet --depth 1 https://github.com/heygen-com/hyperframes "$TMP/hyperframes"

mkdir -p "$DEST" "$ROOT/licenses"

rm -rf "$DEST/video-use"
mkdir -p "$DEST/video-use"
cp -R "$TMP/video-use/SKILL.md" "$TMP/video-use/install.md" "$TMP/video-use/helpers" \
      "$TMP/video-use/skills" "$DEST/video-use/"
cp "$TMP/video-use/LICENSE" "$ROOT/licenses/video-use-LICENSE"

for s in "${HF_SKILLS[@]}"; do
  rm -rf "$DEST/$s"; cp -R "$TMP/hyperframes/skills/$s" "$DEST/$s"
done
for s in "${HF_INTERNAL_SKILLS[@]}"; do
  rm -rf "$DEST/$s"; cp -R "$TMP/hyperframes/.claude/skills/$s" "$DEST/$s"
done
cp "$TMP/hyperframes/LICENSE" "$ROOT/licenses/hyperframes-LICENSE"

{
  echo "video-use   $(git -C "$TMP/video-use" rev-parse HEAD)"
  echo "hyperframes $(git -C "$TMP/hyperframes" rev-parse HEAD)"
} > "$DEST/UPSTREAM.txt"

echo "Synced into $DEST:"
cat "$DEST/UPSTREAM.txt"
