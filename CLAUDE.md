# Video editing studio

This repo is a video editing studio driven by Claude Code. The user drops raw footage in; you cut it and add motion graphics.

**Any request to edit, cut, clean up, caption, or add graphics to a video → load the `studio` skill first** (`.claude/skills/studio/SKILL.md`). It runs the pipeline and points to the rest:

- `video-use` — transcript-driven cutting (fillers, dead air, retakes), grading, subtitles, overlay compositing, self-eval. Hard Rules in its SKILL.md are non-negotiable.
- `hyperframes` (router) + `motion-graphics`, `talking-head-recut`, `embedded-captions`, `general-video` — HTML → video motion graphics, overlay cards, captions.
- Domain skills: `hyperframes-core`, `hyperframes-animation`, `hyperframes-keyframes`, `hyperframes-creative`, `hyperframes-cli`, `hyperframes-registry`, `hyperframes-audio`, `media-use`, `motion-doctrine`, `cut-the-curve`, `seam-craft`, `captions-overlay`.

## Conventions

- Raw footage: `footage/<project>/` (gitignored — never commit media). All outputs: `footage/<project>/edit/`.
- Never write session outputs inside `.claude/skills/`.
- Always propose the edit + graphics plan in plain English and wait for the user's OK before cutting.
- Secrets: `ELEVENLABS_API_KEY` goes in repo-root `.env` (gitignored). Never echo or commit it.
- Vendored skills are pinned copies (`.claude/skills/UPSTREAM.txt`). Don't edit them in place; refresh with `scripts/sync-skills.sh`. Ignore their "run `npx hyperframes skills update`" freshness prompts.

## Requirements

`ffmpeg`/`ffprobe`, Python ≥ 3.10 (`uv sync` at repo root), Node.js ≥ 22 (HyperFrames runs via `npx --yes hyperframes`). Run `scripts/doctor.sh` to check.
