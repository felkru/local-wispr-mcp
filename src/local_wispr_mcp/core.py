"""Shared MLX transcription core (Whisper or Parakeet).

This module is the single source of truth for transcription. Both the MCP
server (`server.py`) and the `transcribe` CLI (`cli.py`) call into here so the
model choice, chunking heuristics, ffmpeg decode, and output renderers never
diverge. It deliberately has NO dependency on the `mcp` package so the CLI
stays lightweight.

Three backends, selected by model name:
  - "scribe_v2" / "scribe_v1"  → ElevenLabs Speech-to-Text API (cloud; needs
    $ELEVENLABS_API_KEY or ~/.config/elevenlabs/key; diarization built in)
  - name containing "whisper"  → mlx-whisper (local)
  - anything else              → parakeet-mlx (local, the default)
Set $WISPR_MODEL to switch, e.g. WISPR_MODEL=scribe_v2 transcribe foo.m4a.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal

from local_wispr_mcp.audio import decode_to_wav, require_ffmpeg

DEFAULT_MODEL = (
    os.environ.get("WISPR_MODEL")
    or os.environ.get("PARAKEET_MODEL")  # honored for backward compatibility
    or "mlx-community/parakeet-tdt-0.6b-v3"
)
CACHE_DIR = os.environ.get("PARAKEET_CACHE_DIR")  # None → HF default (~/.cache/huggingface)

OutputFormat = Literal["text", "json", "srt", "vtt"]

log = logging.getLogger("local-wispr-mcp")

_model = None
_model_lock = threading.Lock()

if DEFAULT_MODEL.lower().startswith("scribe"):
    BACKEND = "elevenlabs"
elif "whisper" in DEFAULT_MODEL.lower():
    BACKEND = "mlx-whisper"
else:
    BACKEND = "parakeet-mlx"


def _elevenlabs_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        key_file = Path.home() / ".config" / "elevenlabs" / "key"
        if key_file.is_file():
            key = key_file.read_text().strip()
    if not key:
        raise RuntimeError(
            "ElevenLabs API key not found — set $ELEVENLABS_API_KEY or write it "
            "to ~/.config/elevenlabs/key (chmod 600)."
        )
    return key


def _transcribe_scribe(wav_path: Path):
    """ElevenLabs Speech-to-Text (Scribe). Adapts the words-level response to
    the Parakeet result shape; sentences break on speaker change or terminal
    punctuation, and carry a `speaker` attr when diarization found speakers.
    If more than one speaker is present, `.text` becomes speaker-turn lines."""
    from types import SimpleNamespace

    import httpx

    with open(wav_path, "rb") as f:
        resp = httpx.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": _elevenlabs_key()},
            data={"model_id": DEFAULT_MODEL, "diarize": "true",
                  "tag_audio_events": "false"},
            files={"file": (wav_path.name, f, "audio/wav")},
            timeout=600.0,
        )
    resp.raise_for_status()
    out = resp.json()

    sentences = []
    cur: list[str] = []
    cur_speaker = None
    cur_start = cur_end = 0.0
    def flush():
        if cur:
            sentences.append(SimpleNamespace(
                text=" ".join(cur), start=cur_start, end=cur_end,
                speaker=cur_speaker,
            ))
    for w in out.get("words") or []:
        if w.get("type") == "spacing":
            continue
        if cur and w.get("speaker_id") != cur_speaker:
            flush()
            cur = []
        if not cur:
            cur_speaker = w.get("speaker_id")
            cur_start = w.get("start", 0.0)
        cur.append(w.get("text", ""))
        cur_end = w.get("end", cur_start)
        if w.get("text", "").rstrip().endswith((".", "?", "!")):
            flush()
            cur = []
    flush()

    speakers = {s.speaker for s in sentences if s.speaker is not None}
    if len(speakers) > 1:
        text = "\n".join(f"{s.speaker}: {s.text}" for s in sentences)
    else:
        text = out.get("text", "")
    return SimpleNamespace(text=text, sentences=sentences)


def _transcribe_whisper(wav_path: Path):
    """Run mlx-whisper and adapt its dict result to the Parakeet result shape
    (`.text` + `.sentences[i].text/.start/.end`) so the renderers below work
    unchanged. mlx_whisper caches the loaded model process-wide (ModelHolder),
    so repeated calls stay hot just like the Parakeet path."""
    from types import SimpleNamespace

    import mlx_whisper

    out = mlx_whisper.transcribe(str(wav_path), path_or_hf_repo=DEFAULT_MODEL)
    return SimpleNamespace(
        text=out["text"],
        sentences=[
            SimpleNamespace(text=s["text"], start=s["start"], end=s["end"])
            for s in out.get("segments", [])
        ],
    )


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
            | ({"speaker": sp} if (sp := getattr(s, "speaker", None)) is not None else {})
            for s in result.sentences
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def transcribe_file(
    path: str,
    format: OutputFormat = "text",
    save_to: str | None = None,
) -> str:
    """Transcribe one local audio/video file with Parakeet (MLX).

    Decodes via ffmpeg to 16 kHz mono PCM, runs the model (chunking long files),
    and renders the requested format. If `save_to` is given the transcript is
    written there and a short status line is returned; otherwise the rendered
    transcript is returned inline.

    This is the shared implementation behind both the MCP `transcribe` tool and
    the `transcribe` CLI — keep all transcription behaviour here.
    """
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
        if BACKEND == "elevenlabs":
            result = _transcribe_scribe(wav_path)
        elif BACKEND == "mlx-whisper":
            global _model
            with _model_lock:
                # Whisper windows 30 s internally — no manual chunking needed.
                result = _transcribe_whisper(wav_path)
                _model = "whisper (cached inside mlx_whisper)"
        else:
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


def model_info_dict() -> dict:
    """Return configured model + cache info without downloading anything."""
    return {
        "model": DEFAULT_MODEL,
        "backend": BACKEND,
        "cache_dir": CACHE_DIR
            or os.environ.get("HF_HUB_CACHE")
            or os.environ.get("HF_HOME")
            or str(Path.home() / ".cache" / "huggingface"),
        "loaded": _model is not None,
    }
