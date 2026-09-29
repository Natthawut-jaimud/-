"""Map the kept words of a video-use EDL onto the OUTPUT timeline.

Writes the flat word array HyperFrames skills read as `transcript.json`
([{"id","text","start","end"}, ...]), so talking-head-recut / embedded-captions
can run on edit/final.mp4 without re-transcribing it. Offsets follow video-use
Hard Rule 5: output_time = word.start - segment_start + segment_offset.

Usage:
    python edl_to_words.py <edit>/edl.json -o <hf_project>/transcript.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def output_words(edl: dict, transcripts_dir: Path) -> list[dict]:
    words: list[dict] = []
    seg_offset = 0.0
    for r in edl["ranges"]:
        seg_start, seg_end = float(r["start"]), float(r["end"])
        tr_path = transcripts_dir / f"{r['source']}.json"
        if not tr_path.exists():
            print(f"  no transcript for {r['source']}, skipping its words", file=sys.stderr)
        else:
            for w in json.loads(tr_path.read_text()).get("words", []):
                if w.get("type", "word") != "word":
                    continue
                ws, we = w.get("start"), w.get("end")
                if ws is None or we is None or we <= seg_start or ws >= seg_end:
                    continue
                text = (w.get("text") or "").strip()
                if not text:
                    continue
                words.append({
                    "id": f"w{len(words)}",
                    "text": text,
                    "start": round(max(ws, seg_start) - seg_start + seg_offset, 3),
                    "end": round(min(we, seg_end) - seg_start + seg_offset, 3),
                })
        seg_offset += seg_end - seg_start
    return words


def main() -> None:
    ap = argparse.ArgumentParser(description="EDL + Scribe transcripts -> output-timeline word array")
    ap.add_argument("edl", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--transcripts-dir", type=Path, default=None,
                    help="Default: <edl dir>/transcripts")
    args = ap.parse_args()

    edl = json.loads(args.edl.read_text())
    tdir = args.transcripts_dir or args.edl.resolve().parent / "transcripts"
    words = output_words(edl, tdir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(words, indent=2))
    end = words[-1]["end"] if words else 0.0
    print(f"{len(words)} words → {args.output} (last word ends at {end:.2f}s)")


if __name__ == "__main__":
    main()
