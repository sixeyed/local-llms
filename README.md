# Local LLM Setup

Running local coding models on an Apple Silicon **M4 Max (128GB)**, with each engine tuned for that machine. The scripts are Python (standard library only — no `pip install` needed) and still detect and print the host's capabilities on startup.

## Prerequisites

macOS on Apple Silicon. The launchers are Python but shell out to an inference engine — install the one(s) you need:

| Tool | Used by | Install |
|---|---|---|
| **Python 3** | all scripts | preinstalled on macOS (`python3`) — stdlib only, no `pip install` for the launchers themselves |
| **Homebrew** | installing `llama.cpp` / `jq` | https://brew.sh |
| **llama-server** (llama.cpp) | qwen models — `servers/run-llama-server.py` | `brew install llama.cpp` |
| **mlx_lm.server** (mlx-lm) | ling / gemma models — `servers/run-mlx-server.py` | currently a git fork — see [Running mlx_lm.server](#running-mlx_lmserver) |
| **jq** | `tests/bench-mlx.sh` only | `brew install jq` |

The launchers verify their engine is on `PATH` and print the install command if it's missing. 

GGUF / MLX model weights are auto-downloaded from [HuggingFace](https://huggingface.co) on first run.

## Quick Start

```bash
python run.py                            # default: qwen3.6-moe via llama-server
python run.py --family qwen              # qwen3.6-moe via llama-server (family default)
python run.py --family ling              # ling-2.6-flash via mlx_lm.server
python run.py --family gemma             # gemma-4-31b via mlx_lm.server
python run.py --model qwen3.6-moe        # specific model — family is inferred
python run.py --model qwen3.6-moe --port 9000   # override the default port (8083)
```

`run.py` is a thin dispatcher: pass `--family` to launch the family default, or `--model` for a specific model (family is inferred from the name). With neither, it defaults to the **qwen** family. `--port` is forwarded either way. Only one model runs at a time, so all servers share a single port (default `8083`).

The default model is **`qwen3.6-moe`** (what `--family qwen` launches).

> **Thinking is disabled by default for the Qwen3 models.** Left on, qwen3 reasoning tends to spin in a "let me investigate…" thinking loop under Cline and VS Code (Copilot agent mode) — deferring instead of committing and burning the client's tool-call budget before it ever produces a visible answer. The launcher sets `--reasoning-budget 0` so the model commits straight to output. See [Thinking mode](#thinking-mode) to change it.

## Layout

```
run.py                       # entrypoint — dispatches to the right server
stop.py                      # stop any running model server
servers/
  hardware.py                # M4 detection + capability printout (shared)
  run-llama-server.py        # qwen models via llama.cpp
  run-mlx-server.py          # ling / gemma models via mlx-lm
tests/
  test-performance.py        # tok/s benchmark against a running server
  bench-mlx.sh               # low-level mlx_lm.server benchmark
```

Connect from any client (Open WebUI, Cline, etc.):

| Qwen models (llama-server) ||
|---|---|
| **Protocol** | OpenAI API compatible |
| **Base URL** | `http://<host>:<port>/v1` (see [Client Setup](#client-setup)) |
| **Model ID** | the alias passed via `--model`: `qwen3.6`, `qwen3.6-moe`, `qwen3.5`, `qwen3.5-small`, `qwen3-coder-next` |
| **API Key** | any non-empty string (e.g. `sk-no-key-required`) |
| **Context** | up to 262K tokens (model/host-dependent — see [Hardware Configs](#hardware-configs)) |

| Ling 2.6 Flash (mlx_lm.server) ||
|---|---|
| **Protocol** | OpenAI API compatible |
| **Base URL** | `http://<host>:<port>/v1` |
| **Model ID** | `mlx-community/Ling-2.6-flash-mlx-5bit` (full HF repo id — no alias support in mlx_lm.server) |
| **API Key** | any non-empty string (e.g. `sk-no-key-required`) |
| **Context** | 262144 tokens (256K) |

| Gemma 4 (mlx_lm.server) ||
|---|---|
| **Protocol** | OpenAI API compatible |
| **Base URL** | `http://<host>:<port>/v1` |
| **Model ID** | `FakeRockert543/gemma-4-31b-it-MLX-8bit` or `FakeRockert543/gemma-4-26b-a4b-it-MLX-8bit` (full HF repo id — no alias support in mlx_lm.server) |
| **API Key** | any non-empty string (e.g. `sk-no-key-required`) |
| **Context** | 262144 tokens (256K) |

## Models

| Model | Engine | Type | Params | Active | Best for |
|-------|--------|------|--------|--------|----------|
| [qwen3.6](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) | llama.cpp | Dense | 27B | 27B | General coding, agentic tasks |
| [qwen3.6-moe](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | llama.cpp | MoE | 35B | 3B | Fast coding — beats 122B-A10B on SWE-bench Verified at ~3× decode speed |
| [qwen3.5](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF) | llama.cpp | MoE | 122B | 10B | Large context, broad knowledge |
| [qwen3.5-small](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) | llama.cpp | MoE | 35B | 3B | Lightweight, fast inference |
| [qwen3-coder-next](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) | llama.cpp | MoE | 80B | 3B | Code-specialized |
| [ling-2.6-flash](https://huggingface.co/mlx-community/Ling-2.6-flash-mlx-5bit) | mlx-lm | MoE | 104B | 7.4B | Fast agentic tasks |
| [gemma-4-31b](https://huggingface.co/FakeRockert543/gemma-4-31b-it-MLX-8bit) | mlx-lm | Dense | 31B | 31B | Cline / agentic coding — commits to multi-step tool use |
| [gemma-4-26b-a4b](https://huggingface.co/FakeRockert543/gemma-4-26b-a4b-it-MLX-8bit) | mlx-lm | MoE | 26B | ~4B | Faster Gemma 4 — higher decode than the 31B dense |

## Running llama-server

Runs [llama.cpp](https://github.com/ggerganov/llama.cpp) directly with full control over batch sizes, context, and quantization. Serves an OpenAI-compatible API. Quantization and context are tuned for the M4 Max (128GB).

```bash
# Install (macOS)
brew install llama.cpp

# Run
python servers/run-llama-server.py --model qwen3.6
python servers/run-llama-server.py --model qwen3.6-moe
python servers/run-llama-server.py --model qwen3.5
python servers/run-llama-server.py --model qwen3-coder-next

# Override the default port (8083)
python servers/run-llama-server.py --model qwen3.6 --port 9000
```

GGUF files are auto-downloaded from HuggingFace on first run.

## Running mlx_lm.server

[mlx-lm](https://github.com/ml-explore/mlx-lm) runs MLX-format models natively on Apple Silicon. Used for `ling-2.6-flash` (only published as MLX weights) and the Gemma 4 models `gemma-4-31b` / `gemma-4-26b-a4b` (PLE-safe MLX quants — see note below). M4 Max only.

```bash
# Install (macOS) - see TODO below; the released mlx-lm doesn't yet
# include the bailing_hybrid arch needed for Ling-2.6
pip3 install --user --force-reinstall --no-deps \
  "git+https://github.com/ivanfioravanti/mlx-lm.git@add-ling-2.6-flash"

# Run
python servers/run-mlx-server.py --model ling-2.6-flash
python servers/run-mlx-server.py --model gemma-4-31b
python servers/run-mlx-server.py --model gemma-4-26b-a4b
```

MLX weights are auto-downloaded from HuggingFace on first run.

> **TODO:** swap back to `pip3 install --user mlx-lm` once [PR #1227](https://github.com/ml-explore/mlx-lm/pull/1227) (adds `bailing_hybrid` for Ling-2.6) is merged and released.

Unlike llama-server, mlx_lm.server has no `--alias` option — clients must send the full HuggingFace repo id (e.g. `mlx-community/Ling-2.6-flash-mlx-5bit` or `FakeRockert543/gemma-4-31b-it-MLX-8bit`) in the request `model` field. `GET /v1/models` returns the id the server is currently serving.

> **Gemma 4 PLE-safety:** standard `mlx-community`/`unsloth` Gemma 4 MLX quants corrupt Per-Layer Embedding layers (they quantize every layer including PLE / `ScaledLinear`) and produce garbage output. `servers/run-mlx-server.py` enforces an allowlist of PLE-safe repos (`FakeRockert543/gemma-4-*-MLX-{4bit,8bit,bf16}` and `mlx-community/gemma-4-*-bf16`) and aborts launch otherwise.

## Hardware Configs

Everything here is tuned for the M4 Max (128GB):

| Model | Quant | Context |
|---|---|---|
| **qwen3.6** | UD-Q8_K_XL (35GB) | 131K |
| **qwen3.6-moe** | UD-Q8_K_XL (38GB) | 262K |
| **qwen3.5** | 122B Q4_K_XL (77GB) | 262K |
| **qwen3.5-small** | 35B-A3B BF16 (69GB) | 131K |
| **qwen3-coder-next** | Q8_K_XL (86GB) | 262K |
| **ling-2.6-flash** | MLX-5bit (72GB) | 262K |
| **gemma-4-31b** | MLX-8bit (33GB) | 262K |
| **gemma-4-26b-a4b** | MLX-8bit (29GB) | 262K |

## Client Setup

Both llama-server and mlx_lm.server expose an OpenAI-compatible API at `/v1/chat/completions`. Only one model runs at a time, so the endpoint is just `host + port`. Default port is `8083` — override with `--port` on `run.py` (or the underlying server scripts).

The servers bind to `0.0.0.0`, so connect over `localhost` on the Mac itself, or over its LAN IP from another machine. Replace `192.168.1.101` below with your Mac's actual IP (`ipconfig getifaddr en0`).

So a typical base URL is `http://192.168.1.101:8083/v1`.

### Open WebUI

1. In **Settings > Connections**, add an OpenAI-compatible connection
2. Set the **Base URL** to `http://<host>:<port>/v1` (e.g. `http://192.168.1.101:8083/v1`)
3. Enter any value for the **API Key** (e.g. `sk-no-key-required`) — llama-server doesn't validate it
4. The model appears by its alias (e.g. `qwen3.6-moe`); for Ling, the full HF repo id

### Cline (VS Code)

1. Open Cline settings and select **OpenAI Compatible** as the provider
2. Set the **Base URL** to `http://<host>:<port>/v1` (e.g. `http://192.168.1.101:8083/v1`)
3. Enter any value for the **API Key** (e.g. `sk-no-key-required`) — llama-server doesn't validate it
4. Set the **Model ID** to the alias (e.g. `qwen3.6-moe`). For mlx_lm.server models, use the full HF repo id:
   - Ling: `mlx-community/Ling-2.6-flash-mlx-5bit`
   - Gemma 4 31B: `FakeRockert543/gemma-4-31b-it-MLX-8bit`

> llama-server ignores the `Authorization` header unless started with `--api-key`. Any non-empty string works as the API key for clients that require it.

### opencode

Edit `~/.config/opencode/opencode.json` (or `opencode.json` in your project root) and add a provider block using the `@ai-sdk/openai-compatible` adapter:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local server",
      "options": {
        "baseURL": "http://192.168.1.101:8083/v1"
      },
      "models": {
        "qwen3.6-moe": { "name": "Qwen 3.6 MoE (35B-A3B)" },
        "mlx-community/Ling-2.6-flash-mlx-5bit": { "name": "Ling 2.6 Flash" },
        "FakeRockert543/gemma-4-31b-it-MLX-8bit": { "name": "Gemma 4 31B" }
      }
    }
  },
  "model": "local/qwen3.6-moe"
}
```

### VS Code (custom model endpoint)

VS Code's built-in chat now supports custom OpenAI-compatible endpoints. Edit your `settings.json` and add a `github.copilot.chat.customModels` array (or use the **Add Custom Model** UI). For qwen3.6-moe:

```json
[
  {
    "name": "http://192.168.1.101:8083/v1",
    "vendor": "customendpoint",
    "apiKey": "none",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "qwen3.6-moe",
        "name": "qwen3.6-moe",
        "url": "http://192.168.1.101:8083/v1/chat/completions",
        "toolCalling": true,
        "vision": false,
        "maxInputTokens": 250000,
        "maxOutputTokens": 16000
      }
    ]
  }
]
```

- `apiType` is always `chat-completions` for both llama-server and mlx_lm.server (neither speaks the OpenAI Responses API or Anthropic Messages format).
- For mlx_lm.server models, set `id` to the full HF repo id (e.g. `FakeRockert543/gemma-4-31b-it-MLX-8bit`) — no alias support.
- `toolCalling: true` is required for agent-mode use.
- `maxInputTokens` should be a bit under the server's `-c` context (262K). 250000 leaves headroom for VS Code's bookkeeping.
- `maxOutputTokens: 16000` is a safe default for coding work; raise only if you hit truncation on big file generation.

### Thinking mode

Qwen3 models (qwen3.6, qwen3.6-moe, qwen3.5, qwen3.5-small) ship with thinking mode capable — they can generate chain-of-thought into `message.reasoning_content` before the visible `message.content`.

`servers/run-llama-server.py` **disables thinking by default** (`reasoning_budget = 0` → `--reasoning-budget 0`) for every thinking-capable Qwen3 model, including the default `qwen3.6-moe`. Reason: under Cline and VS Code (Copilot agent mode), thinking-mode qwen3 gets stuck in a "let me investigate…" thinking loop — it defers rather than commits, often exhausting the client's max-tool-calls-per-turn before producing a visible answer. Disabling thinking makes the model commit straight to output, and quality on multi-step planning hasn't been worse in practice for this workflow.

To re-enable, change `reasoning_budget = 0` to a positive integer in the model config:

| `reasoning_budget` | Behaviour | Trade-off |
|---|---|---|
| `0` *(default here)* | thinking activates and immediately terminates — model goes straight to content | model commits; loses pure chain-of-thought |
| `1000` | thinking capped at 1000 tokens, model forced into content after | ≤18s worst-case "no output yet" at 56 t/s; some thinking benefit |
| `-1` | unrestricted | best for complex reasoning, but loops/deferral observed under agentic harnesses |

`qwen3-coder-next` is not a thinking model and doesn't have a `reasoning_budget` entry.

A blank `content` field with non-empty `reasoning_content` in client responses is the symptom of thinking eating all of `max_tokens` — bump `max_tokens`, lower `--reasoning-budget`, or disable thinking entirely (the current default).

## Performance Testing

`tests/test-performance.py` probes a single host/port via `/v1/models` to discover whatever model is currently running, then runs a coding-prompt suite against it:

```bash
python tests/test-performance.py                                          # localhost:8083
python tests/test-performance.py --server-host 192.168.1.101 --iterations 3
python tests/test-performance.py --port 9000                              # if you launched with --port 9000
```

Reports tokens/sec per prompt and an overall average for the running model.

## References

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - the inference engine
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - MLX inference for Apple Silicon
- [Unsloth GGUF quants](https://huggingface.co/unsloth) - optimized quantizations used here

## License

[MIT](LICENSE) © Elton Stoneman
