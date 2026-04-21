"""Audio decoding via ffmpeg.

Any format ffmpeg can read (mp3, m4a, aac, flac, ogg, opus, wav, wma, aiff,
webm, mp4, mkv, mov, etc.) is decoded to a 16 kHz mono 16-bit PCM WAV in a
temporary file. The WAV path is then handed to parakeet-mlx, which is the
format its preprocessor natively expects.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class FfmpegMissingError(RuntimeError):
    pass


class AudioDecodeError(RuntimeError):
    pass


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FfmpegMissingError(
            "ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`) and retry."
        )
    return path


def decode_to_wav(src: str | Path, sample_rate: int = 16000) -> Path:
    """Decode any ffmpeg-supported audio/video file to a temp 16 kHz mono WAV.

    Returns the path to the temp WAV. Caller is responsible for deleting it.
    """
    ffmpeg = require_ffmpeg()
    src_path = Path(src).expanduser().resolve()
    if not src_path.is_file():
        raise AudioDecodeError(f"Input file does not exist: {src_path}")

    fd, tmp_path = tempfile.mkstemp(prefix="parakeet-", suffix=".wav")
    # Close the fd immediately — ffmpeg will write to the path.
    import os
    os.close(fd)

    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(src_path),
        "-vn",                 # drop any video track
        "-ac", "1",            # mono
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",  # 16-bit PCM
        "-f", "wav",
        tmp_path,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        finally:
            pass
        raise AudioDecodeError(
            f"ffmpeg failed to decode {src_path.name}: {proc.stderr.strip() or 'unknown error'}"
        )

    return Path(tmp_path)
