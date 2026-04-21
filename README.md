# local-wispr-mcp

A **local** [Model Context Protocol](https://modelcontextprotocol.io) server that exposes
[NVIDIA Parakeet](https://huggingface.co/collections/mlx-community/parakeet) automatic
speech recognition to any MCP client — Claude Code, Claude Desktop, Cursor, Zed, etc.

- Runs entirely on your machine — audio never leaves it.
- Uses [`parakeet-mlx`](https://pypi.org/project/parakeet-mlx/), a native
  [MLX](https://github.com/ml-explore/mlx) port of Parakeet, so it runs on
  Apple Silicon's GPU with no CUDA, no Docker, no cloud.
- The model (~1–2 GB) is downloaded from Hugging Face on first use, cached, and
  reused forever after. No manual setup beyond `uv sync`.
- **ffmpeg-powered**: any format ffmpeg can read works — mp3, m4a, aac, flac,
  ogg, opus, wav, wma, aiff, webm, mp4, mkv, mov, and more. Video files are
  accepted too; the audio track is extracted automatically.

> Name nods to [Wispr Flow](https://wisprflow.ai) / OpenAI Whisper — but this
> is Parakeet, locally, via MLX.

## Requirements

- **macOS on Apple Silicon** (M1/M2/M3/M4). `parakeet-mlx` uses MLX, which
  is Apple-Silicon-only. If you are on Linux or an Intel Mac this package
  will not work — use the upstream NVIDIA NeMo Parakeet implementation instead.
- **Python 3.11+**
- **ffmpeg** on `PATH` — `brew install ffmpeg`
- [**uv**](https://docs.astral.sh/uv/) (recommended) — `brew install uv`

## Install

Install the server as a user-global `uv` tool. This puts a `local-wispr-mcp`
executable on your `PATH` so every MCP client — in any working directory —
can launch it without a wrapper:

```bash
uv tool install git+https://github.com/felkru/local-wispr-mcp
```

Or from a local checkout:

```bash
git clone https://github.com/felkru/local-wispr-mcp.git
cd local-wispr-mcp
uv tool install .
```

`uv` creates a dedicated venv under `~/.local/share/uv/tools/local-wispr-mcp/`
and symlinks the entry point to `~/.local/bin/local-wispr-mcp`. Make sure
`~/.local/bin` is on your `PATH` (`uv tool update-shell` does this for you).

The first transcription call triggers a ~1–2 GB download of the default model
(`mlx-community/parakeet-tdt-0.6b-v3`) from Hugging Face. This is cached under
`~/.cache/huggingface/hub` and reused on every subsequent call.

> **Why `uv tool install` and not `uv sync` + `uv run`?** `uv sync` installs
> the project in *editable* mode using a `.pth` file, which `uv` writes with
> macOS APFS compression. Python 3.12+ treats those compressed `.pth` files
> as hidden and refuses to load them — so the package becomes un-importable
> and the MCP server fails to start. `uv tool install` performs a regular
> (non-editable) install that doesn't rely on `.pth` loading and is immune
> to the bug.

## Wire it into an MCP client

### Claude Code (CLI) — user scope

Register at user scope so the server is available in every project:

```bash
claude mcp add local-wispr --scope user -- local-wispr-mcp
```

Verify:

```bash
claude mcp get local-wispr
# → Scope: User config (available in all your projects)
# → Status: ✓ Connected
```

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "local-wispr": {
      "command": "local-wispr-mcp"
    }
  }
}
```

If Claude Desktop can't find the executable on its `PATH`, use the absolute
path `/Users/<you>/.local/bin/local-wispr-mcp` instead. Restart Claude
Desktop afterwards.

### Any other MCP client

This is a standard stdio MCP server. Start it with:

```bash
local-wispr-mcp
```

and point your client at that command.

## Tools

### `transcribe(path, format="text")`

Transcribe a local audio or video file.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | string | — | Absolute path (or `~`-prefixed) to the file on this machine. |
| `format` | `"text"` \| `"json"` \| `"srt"` \| `"vtt"` | `"text"` | Output format. |

**Output formats — when to use each:**

- **`text`** (default). Plain transcript, nothing else. This is the shortest
  possible output — optimised for LLM consumption where you just want the
  words. Use this unless you specifically need timing information.

  ```
  Hello world, this is a test.
  ```

- **`json`**. Compact JSON with sentence-level start/end timestamps. Use when
  a downstream tool or model needs to reason about timing, produce chapters,
  jump to specific moments, etc.

  ```json
  {
    "text": "Hello world, this is a test.",
    "sentences": [
      {"text": "Hello world,", "start": 0.12, "end": 0.84},
      {"text": "this is a test.", "start": 0.90, "end": 1.76}
    ]
  }
  ```

- **`srt`**. Standard SubRip subtitle file. Drop into a video player or
  YouTube upload.

- **`vtt`**. WebVTT — the HTML5 `<track>` format.

### `model_info()`

Returns the configured model id, the Hugging Face cache directory, and whether
the model is currently loaded in memory. Safe to call before first use — it
will not trigger a download.

## Configuration

All configuration is via environment variables. None are required.

| Variable | Default | Purpose |
|---|---|---|
| `PARAKEET_MODEL` | `mlx-community/parakeet-tdt-0.6b-v3` | Which Parakeet checkpoint to load. See the [mlx-community collection](https://huggingface.co/collections/mlx-community/parakeet). |
| `PARAKEET_CACHE_DIR` | unset (uses `HF_HOME` / `~/.cache/huggingface`) | Where model weights are cached on disk. |
| `LOG_LEVEL` | `INFO` | Python log level for the server. |

Set them either in your shell environment or inside the MCP client config:

```json
{
  "mcpServers": {
    "local-wispr": {
      "command": "uv",
      "args": ["run", "--directory", "/abs/path", "local-wispr-mcp"],
      "env": {
        "PARAKEET_MODEL": "mlx-community/parakeet-tdt-0.6b-v2",
        "PARAKEET_CACHE_DIR": "/Volumes/fast/hf-cache"
      }
    }
  }
}
```

## How it works

```
┌────────────┐    stdio (JSON-RPC)    ┌──────────────────┐
│ MCP client │ ─────────────────────▶ │ local-wispr-mcp  │
│ (Claude)   │                        │  FastMCP server  │
└────────────┘                        └────────┬─────────┘
                                               │
                         any audio/video ─────▶│
                                               ▼
                                      ┌──────────────────┐
                                      │ ffmpeg           │
                                      │ → 16 kHz mono WAV│
                                      └────────┬─────────┘
                                               ▼
                                      ┌──────────────────┐
                                      │ parakeet-mlx     │
                                      │ (MLX on Apple    │
                                      │  Silicon GPU)    │
                                      └────────┬─────────┘
                                               ▼
                                      text / json / srt / vtt
```

1. The MCP client calls `transcribe(path, format)` over stdio.
2. The server shells out to `ffmpeg` to decode the file into a temp 16 kHz
   mono 16-bit PCM WAV — this is the format Parakeet's preprocessor expects,
   and it makes format support equal to ffmpeg's.
3. The Parakeet model is lazy-loaded on the first call (downloaded if not in
   the HF cache), then kept resident in memory for the lifetime of the
   server process.
4. Inference runs on the Apple Silicon GPU via MLX.
5. The aligned result is rendered into the requested output format and
   returned.

The temp WAV is deleted immediately after transcription.

## Performance

On an M-series Mac, `parakeet-tdt-0.6b-v3` is typically faster than real-time
by a large margin. A 10-minute podcast transcribes in roughly 15–40 seconds
depending on the chip. Cold start adds a few seconds for model load; model
download (one-off) depends on your connection.

Clips up to ~90 seconds are transcribed in one pass so the model can use
unified attention across the whole audio. For longer files (minutes to
hours) the server activates parakeet-mlx's built-in streaming with
`chunk_duration=120s, overlap_duration=15s` — without this, a 56-minute
file would need ~11 GB of contiguous Metal buffer and OOM on most Apple
Silicon chips. You don't need to split the input yourself either way.

## Troubleshooting

**`ffmpeg not found on PATH`** — install it: `brew install ffmpeg`.

**Model download is slow or fails** — set `HF_HUB_ENABLE_HF_TRANSFER=1`
and `pip install hf-transfer` for faster transfers, or pre-download with
`huggingface-cli download mlx-community/parakeet-tdt-0.6b-v3`.

**Out of memory on the GPU** — try a smaller model, e.g.
`PARAKEET_MODEL=mlx-community/parakeet-tdt-0.6b-v2`. Or set
`PARAKEET_MODEL` to a quantised variant from the mlx-community collection.

**Transcribes but returns empty text** — the file probably contains no
speech, or the audio track is corrupted. Try playing it back first.

**Server shows `✗ Failed to connect` or startup hangs** — you're almost
certainly on the `uv sync` / `uv run` path on Python 3.12+. Switch to
`uv tool install .` (see the [Install](#install) section). To confirm
the cause: `uv run --directory /path/to/local-wispr-mcp python -c "import
local_wispr_mcp"` — if that errors with `ModuleNotFoundError`, the editable
`.pth` is being skipped by Python as a "hidden" file and you need the tool
install.

**Upgrade to a newer version** — `uv tool install git+https://github.com/felkru/local-wispr-mcp --force`.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for the original Parakeet
  models.
- [`senstella/parakeet-mlx`](https://github.com/senstella/parakeet-mlx) for
  the MLX port this server is built on.
- [Anthropic MCP](https://modelcontextprotocol.io) for the protocol.
