"""Convert a HyperFrames/Whisper word array into the Scribe-shaped JSON that
video-use helpers read (pack_transcripts.py, render.py, timeline_view.py).

Fallback for when there is no ELEVENLABS_API_KEY. Whisper normalizes most
fillers ("um", "uh") away, so filler removal leans more on silence gaps and
timeline_view checks than it does with Scribe.

Usage:
    npx --yes hyperframes transcribe <video> -d <tmp> --json --model small.en
    python words_to_scribe.py <tmp>/transcript.json -o <edit>/transcripts/<video_stem>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def to_scribe(words: list[dict]) -> dict:
    out: list[dict] = []
    prev_end: float | None = None
    for w in words:
        start, end = float(w["start"]), float(w["end"])
        if prev_end is not None and start > prev_end:
            out.append({"type": "spacing", "text": " ", "start": prev_end, "end": start})
        out.append({"type": "word", "text": w["text"].strip(), "start": start, "end": end,
                    "speaker_id": "speaker_0"})
        prev_end = end
    return {"text": " ".join(w["text"].strip() for w in words), "words": out}


def main() -> None:
    ap = argparse.ArgumentParser(description="Whisper word array -> Scribe-shaped transcript")
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    data = json.loads(args.input.read_text())
    words = data["words"] if isinstance(data, dict) else data
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_scribe(words), indent=2))
    print(f"{len(words)} words → {args.output}")


if __name__ == "__main__":
    main()
