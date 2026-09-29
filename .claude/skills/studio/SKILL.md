---
name: studio
description: The full editing pipeline for this repo — raw footage in, finished video out. Transcribe, cut fillers / dead air / retakes (video-use), then add motion graphics, captions and designed overlay cards (HyperFrames). Use whenever the user drops a raw video and asks to edit it, clean it up, remove ums, tighten it, add graphics, captions, titles, lower-thirds, or "do the full edit".
---

# Studio pipeline

This skill is the glue. The real craft lives in two vendored skill sets — read them, don't paraphrase them:

- **`video-use`** (`.claude/skills/video-use/SKILL.md`) — transcript-driven cutting, grading, subtitles, overlay compositing, self-eval. Its 12 Hard Rules are non-negotiable.
- **HyperFrames** (`.claude/skills/hyperframes/SKILL.md` is the router) — HTML → deterministic video for motion graphics, captions and overlay cards.

Bridge scripts in this skill (`.claude/skills/studio/scripts/`):

- `words_to_scribe.py` — Whisper word array → Scribe-shaped JSON, so video-use helpers work without an ElevenLabs key.
- `edl_to_words.py` — EDL + source transcripts → output-timeline word array (`transcript.json` for HyperFrames), so the cut video never needs re-transcribing.

## Where things live

```
footage/<project>/            ← user drops raw files here (gitignored)
footage/<project>/edit/       ← every video-use output (Hard Rule 12)
footage/<project>/edit/hf/    ← HyperFrames projects for post-cut packaging
```

Run helpers from the repo root: `python .claude/skills/video-use/helpers/<name>.py ...` (or `uv run python ...`).

## Stage 0 — Preflight (once per session)

1. `ffmpeg -version`, `ffprobe -version`, `node -v` (HyperFrames needs Node 22+), `python3 -c "import librosa, PIL, matplotlib, requests"`. If Python deps are missing: `uv sync` (or `pip install -e .`) at the repo root.
2. Transcription backend:
   - **Preferred:** `ELEVENLABS_API_KEY` in env or repo-root `.env` → Scribe keeps fillers ("um", "uh") verbatim, which is what filler removal needs.
   - **Fallback:** local Whisper via `npx --yes hyperframes transcribe`. Tell the user fillers are partly normalized away, so cuts lean on silence gaps and `timeline_view` checks.
3. If `footage/<project>/edit/project.md` exists, summarize the last session in one sentence and ask whether to continue.

## Stage 1 — Cut (video-use)

Follow `video-use/SKILL.md` "The process" steps 1–8. In short:

1. Inventory: `ffprobe` each source, then transcribe.
   - Scribe: `transcribe_batch.py footage/<project>` (or `transcribe.py <file>`).
   - Whisper fallback, per source:
     ```bash
     npx --yes hyperframes transcribe <src> -d footage/<project>/edit/whisper/<stem> --json --model small.en
     python .claude/skills/studio/scripts/words_to_scribe.py \
       footage/<project>/edit/whisper/<stem>/transcript.json -o footage/<project>/edit/transcripts/<stem>.json
     ```
     Non-English → `--model small --language <iso>` (never a `.en` model).
2. `pack_transcripts.py --edit-dir footage/<project>/edit` → read `takes_packed.md`.
3. Pre-scan for fillers, false starts, retakes, dead air. Converse, then **propose the strategy and wait for OK** (Hard Rule 11). Include the motion-graphics plan (Stage 2) in the same proposal so the user approves one plan.
4. Write `edl.json`, render `--preview`, self-eval every cut boundary with `timeline_view.py` on the rendered output, fix, then final render.

Filler removal specifics: cut on word boundaries only, pad 30–200 ms, 30 ms audio fades at every seam. Keep laughs and emphasis beats. When a jump cut is visually harsh, cover it with a punch-in (scale/crop on the next segment) or a B-roll/graphic overlay rather than keeping the filler.

## Stage 2 — Motion graphics (HyperFrames)

Pick per graphic; you can mix both routes in one project.

**Route A — inside the cut (video-use overlay slots).** Full-screen cards, stat hits, kinetic titles, logo stings, anything tied to one moment.
- One `Agent` sub-agent per slot, spawned in parallel (Hard Rule 10), building in `footage/<project>/edit/animations/slot_<id>/` with the `motion-graphics` skill (and `hyperframes-core` / `hyperframes-animation` as it directs).
- Opaque full-frame: `npx --yes hyperframes render . -o render.mp4`.
- Transparent over the speaker: render `--format mov` (ProRes 4444 — ffmpeg decodes its alpha by default). A VP9 `.webm` loses alpha in `render.py` unless decoded with `-c:v libvpx-vp9`.
- Add to `edl.json` `overlays` with `start_in_output` = payoff-word output time − reveal duration.

**Route B — after the cut (package `edit/final.mp4`).** Lower-thirds, callout cards, pull-quotes, side panels, captions across the whole video.
1. `python .claude/skills/studio/scripts/edl_to_words.py footage/<project>/edit/edl.json -o footage/<project>/edit/hf/<name>/transcript.json`
2. Then follow the workflow skill with that as the transcript (skip its transcribe step):
   - Designed overlay cards → `talking-head-recut`.
   - Captions / subtitles (incl. cinematic captions behind the subject) → `embedded-captions`. If you use it, do **not** also burn `render.py` subtitles.
   - Anything else → `general-video`, via the `/hyperframes` router.

Motion craft for either route: `motion-doctrine`, `cut-the-curve`, `seam-craft`, `hyperframes-keyframes`; captions craft: `captions-overlay`. Assets / SFX / music → `media-use`; mixing → `hyperframes-audio`. Registry blocks (named looks, transitions) → `hyperframes-registry`.

`hyperframes init` templates load GSAP from `cdn.jsdelivr.net`. If the render fails with `sub_timeline_script_failure` (a sandbox that blocks the CDN), copy `.claude/skills/talking-head-recut/assets/vendor/gsap.min.js` into the slot and point the `<script src>` at the local file.

Palette and fonts come from the user. If they gave none, propose one in the Stage 1 strategy and wait for confirmation. Assert web fonts loaded before rendering.

## Stage 3 — Verify and deliver

- `ffprobe` duration/resolution vs expectation; `timeline_view.py` at every cut and at each graphic's entry/exit; loudness `ffmpeg -i out.mp4 -af ebur128=peak=true -f null -` (target −14 LUFS, ≤ −1 dBTP). You cannot listen: report numbers.
- Max 3 self-eval passes, then flag what remains.
- Append the session to `footage/<project>/edit/project.md`.

## Notes on the vendored skills

- They are pinned copies (see `.claude/skills/UPSTREAM.txt`). Skip their "keep this skill fresh / `npx hyperframes skills update`" prompts; refresh with `scripts/sync-skills.sh` instead.
- The `/hyperframes` router names creation workflows that are not vendored here (product-launch-video, faceless-explainer, pr-to-video, music-to-video, slideshow, figma, remotion-to-hyperframes). If one is really needed, install it on demand with `npx hyperframes skills update <name>` after asking the user.
