"""FastMCP server exposing local Parakeet transcription over stdio."""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from local_wispr_mcp.audio import (
    AudioDecodeError,
    FfmpegMissingError,
    decode_to_wav,
    require_ffmpeg,
)

DEFAULT_MODEL = os.environ.get("PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
CACHE_DIR = os.environ.get("PARAKEET_CACHE_DIR")  # None → HF default (~/.cache/huggingface)

OutputFormat = Literal["text", "json", "srt", "vtt"]

log = logging.getLogger("local-wispr-mcp")


_model = None
_model_lock = threading.Lock()


def _load_model():
    """Lazy-load the Parakeet model. First call downloads it via Hugging Face."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        # Import lazily so `--help` etc. don't pay the MLX import cost.
        from parakeet_mlx import from_pretrained

        log.info("Loading Parakeet model %s (first run downloads ~1-2 GB)...", DEFAULT_MODEL)
        kwargs: dict[str, Any] = {}
        if CACHE_DIR:
            kwargs["cache_dir"] = CACHE_DIR
        _model = from_pretrained(DEFAULT_MODEL, **kwargs)
        log.info("Model loaded.")
        return _model


def _fmt_ts_srt(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_ts_vtt(seconds: float) -> str:
    return _fmt_ts_srt(seconds).replace(",", ".")


def _render_srt(sentences) -> str:
    lines = []
    for i, sent in enumerate(sentences, start=1):
        lines.append(str(i))
        lines.append(f"{_fmt_ts_srt(sent.start)} --> {_fmt_ts_srt(sent.end)}")
        lines.append(sent.text.strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_vtt(sentences) -> str:
    lines = ["WEBVTT", ""]
    for sent in sentences:
        lines.append(f"{_fmt_ts_vtt(sent.start)} --> {_fmt_ts_vtt(sent.end)}")
        lines.append(sent.text.strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_json(result) -> str:
    import json
    payload = {
        "text": result.text,
        "sentences": [
            {"text": s.text, "start": round(s.start, 3), "end": round(s.end, 3)}
            for s in result.sentences
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


mcp = FastMCP("local-wispr-mcp")


@mcp.tool()
def transcribe(
    path: str,
    format: OutputFormat = "text",
) -> str:
    """Transcribe a local audio or video file using NVIDIA Parakeet (MLX).

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

    Returns:
        The transcript in the requested format as a single string.
    """
    require_ffmpeg()  # surface a clear error early if ffmpeg is missing

    src = Path(path).expanduser()
    if not src.is_absolute():
        src = src.resolve()
    if not src.is_file():
        raise ValueError(f"File not found: {src}")

    wav_path = decode_to_wav(src)
    try:
        model = _load_model()
        with _model_lock:
            result = model.transcribe(str(wav_path))
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except OSError:
            pass

    if format == "text":
        return result.text.strip() + "\n"
    if format == "json":
        return _render_json(result)
    if format == "srt":
        return _render_srt(result.sentences)
    if format == "vtt":
        return _render_vtt(result.sentences)
    raise ValueError(f"Unknown format: {format!r}")


@mcp.tool()
def model_info() -> str:
    """Return which Parakeet model is configured and where its cache lives.

    Does NOT download the model — safe to call before first transcription.
    """
    import json
    info = {
        "model": DEFAULT_MODEL,
        "cache_dir": CACHE_DIR or os.environ.get("HF_HUB_CACHE")
            or os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface"),
        "loaded": _model is not None,
    }
    return json.dumps(info, indent=2)


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
