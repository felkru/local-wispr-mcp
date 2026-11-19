"""FastMCP server exposing local MLX transcription (Whisper/Parakeet) over stdio.

All transcription behaviour lives in `local_wispr_mcp.core`; the tools here are
thin MCP-facing wrappers so the server and the `transcribe` CLI stay DRY.
"""

from __future__ import annotations

import json
import logging
import os
import sys

from mcp.server.fastmcp import FastMCP

from local_wispr_mcp.audio import FfmpegMissingError, require_ffmpeg
from local_wispr_mcp.core import OutputFormat, model_info_dict, transcribe_file

log = logging.getLogger("local-wispr-mcp")

mcp = FastMCP("local-wispr-mcp")


@mcp.tool()
def transcribe(
    path: str,
    format: OutputFormat = "text",
    save_to: str | None = None,
) -> str:
    """Transcribe a local audio or video file locally on MLX (Whisper large-v3 by default).

    Any format ffmpeg can decode is supported (mp3, m4a, aac, flac, ogg, opus,
    wav, wma, aiff, webm, mp4, mkv, mov, ...). The file is decoded to 16 kHz
    mono PCM via ffmpeg before being fed to the model.

    Args:
        path: Absolute or ~-prefixed path to the audio/video file on this machine.
        format: Output format.
            - "text" (default): plain transcript, no timestamps, no scaffolding.
              Minimum tokens — use this unless you specifically need timing.
            - "json":  {"text": "...", "sentences": [{"text", "start", "end"}, ...]}
            - "srt":   SubRip subtitle file contents.
            - "vtt":   WebVTT subtitle file contents.
        save_to: Optional absolute or ~-prefixed path. If given, the transcript
            is written to this file instead of being returned inline. The tool
            then returns a short status line (path, byte count, audio duration,
            wall time). Use this for anything longer than a couple of minutes
            so the transcript doesn't flood the LLM's context. Parent
            directories are created if needed. An existing file is overwritten.

    Returns:
        When `save_to` is None: the transcript in the requested format.
        When `save_to` is set:  a short status string referencing the file.
    """
    return transcribe_file(path, format=format, save_to=save_to)


@mcp.tool()
def model_info() -> str:
    """Return which model/backend is configured and where its cache lives.

    Does NOT download the model — safe to call before first transcription.
    """
    return json.dumps(model_info_dict(), indent=2)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    try:
        require_ffmpeg()
    except FfmpegMissingError as e:
        # Don't crash the server — surface the error when a tool is called.
        log.warning(str(e))
    mcp.run()


if __name__ == "__main__":
    main()
