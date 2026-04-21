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
    save_to: str | None = None,
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
    import time
    t0 = time.time()
    require_ffmpeg()  # surface a clear error early if ffmpeg is missing

    src = Path(path).expanduser()
    if not src.is_absolute():
        src = src.resolve()
    if not src.is_file():
        raise ValueError(f"File not found: {src}")

    wav_path = decode_to_wav(src)
    try:
        # 16 kHz mono s16 WAV ⇒ 32000 bytes/sec. The 44-byte header is a lower
        # bound we tolerate being slightly off; we only need a rough duration.
        duration_s = max(0.0, (wav_path.stat().st_size - 44) / 32000.0)
        model = _load_model()
        with _model_lock:
            # Chunk only when necessary. Short clips fit in Metal as a single
            # tensor and benefit from unified attention; long files (≳5 min on
            # a 9.5 GB Metal cap) OOM without chunking, so we hand them to the
            # library's built-in streaming with the CLI defaults.
            if duration_s > 90.0:
                result = model.transcribe(
                    str(wav_path),
                    chunk_duration=120.0,
                    overlap_duration=15.0,
                )
            else:
                result = model.transcribe(str(wav_path))
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except OSError:
            pass

    if format == "text":
        rendered = result.text.strip() + "\n"
    elif format == "json":
        rendered = _render_json(result)
    elif format == "srt":
        rendered = _render_srt(result.sentences)
    elif format == "vtt":
        rendered = _render_vtt(result.sentences)
    else:
        raise ValueError(f"Unknown format: {format!r}")

    if save_to is not None:
        dest = Path(save_to).expanduser()
        if not dest.is_absolute():
            dest = dest.resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rendered, encoding="utf-8")
        dur_min, dur_sec = divmod(int(duration_s), 60)
        return (
            f"Wrote {format} transcript to {dest} "
            f"({len(rendered):,} chars, audio {dur_min}m{dur_sec:02d}s, "
            f"transcribed in {time.time()-t0:.1f}s)\n"
        )

    return rendered


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
