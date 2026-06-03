"""`transcribe` command-line interface.

A thin CLI over the same `local_wispr_mcp.core.transcribe_file` used by the MCP
server — same model, same ffmpeg decode, same chunking, same renderers. Nothing
about transcription is reimplemented here; this only handles argument parsing
and where output goes.

    transcribe meeting.m4a                      # → ./meeting.txt
    transcribe -o /tmp a.m4a b.mp3              # → /tmp/a.txt, /tmp/b.txt
    transcribe --format srt talk.mp4            # → ./talk.srt
    transcribe --stdout note.m4a               # print transcript, write nothing
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from local_wispr_mcp.core import (
    DEFAULT_MODEL,
    model_info_dict,
    transcribe_file,
)

# CLI format name -> (core format, file extension). "txt" is accepted as a
# friendly alias for the core's "text".
_FORMATS = {
    "text": ("text", ".txt"),
    "txt": ("text", ".txt"),
    "json": ("json", ".json"),
    "srt": ("srt", ".srt"),
    "vtt": ("vtt", ".vtt"),
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description=(
            "Transcribe audio/video with NVIDIA Parakeet (MLX). Shares its engine "
            f"with the local-wispr-mcp server. Model: {DEFAULT_MODEL} "
            "(override via $PARAKEET_MODEL)."
        ),
    )
    p.add_argument(
        "audios",
        nargs="*",
        metavar="FILE",
        help="Audio/video file(s) to transcribe (anything ffmpeg can decode).",
    )
    p.add_argument(
        "-f", "--format",
        choices=sorted(_FORMATS),
        default="text",
        help="Output format (default: text/txt).",
    )
    p.add_argument(
        "-o", "--output-dir",
        metavar="DIR",
        default=".",
        help="Directory for the transcript file(s) (default: current directory). "
             "Each output is named <input-basename>.<ext>. Ignored with --stdout.",
    )
    p.add_argument(
        "--stdout",
        action="store_true",
        help="Print the transcript to stdout instead of writing a file.",
    )
    p.add_argument(
        "--model-info",
        action="store_true",
        help="Print the configured model and cache location, then exit "
             "(does not download the model).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "WARNING"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    args = _build_parser().parse_args(argv)

    if args.model_info:
        import json
        print(json.dumps(model_info_dict(), indent=2))
        return 0

    if not args.audios:
        _build_parser().error("at least one FILE is required (or use --model-info)")

    core_format, ext = _FORMATS[args.format]
    out_dir = Path(args.output_dir).expanduser()

    exit_code = 0
    for raw in args.audios:
        src = Path(raw).expanduser()
        try:
            if args.stdout:
                sys.stdout.write(transcribe_file(str(src), format=core_format))
            else:
                dest = out_dir / (src.stem + ext)
                status = transcribe_file(
                    str(src), format=core_format, save_to=str(dest)
                )
                # status already ends with a newline.
                sys.stderr.write(status)
        except Exception as e:  # noqa: BLE001 — CLI boundary: report and continue.
            print(f"transcribe: {src}: {e}", file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
