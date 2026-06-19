#!/usr/bin/env python3
"""Run mlx-lm's mlx_lm.server with M4 Max-tuned settings.

mlx-lm runs MLX-format models natively on Apple Silicon. Used for ling-2.6-flash
(only published as MLX weights) and gemma-4-31b (PLE-safe MLX quant). M4 Max only.
MLX weights are auto-downloaded from HuggingFace on first run.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

from hardware import detect_hardware, NETWORK_IP, color, C

# ============================================================
# Model defaults
# ============================================================
MODELS = {
    "ling-2.6-flash": {
        "hf_repo": "mlx-community/Ling-2.6-flash-mlx-5bit",
        "max_tokens": 262144,
        "temp": 0.7,
        "top_p": 0.8,
        "prefill_step_size": 4096,
        "prompt_cache_bytes": 17179869184,  # 16GB KV cache reuse across requests
    },
    # Gemma 4 31B dense - PLE-safe 8-bit (~35GB). Standard mlx-community/unsloth
    # quants corrupt Per-Layer Embedding layers and produce garbage.
    "gemma-4-31b": {
        "hf_repo": "FakeRockert543/gemma-4-31b-it-MLX-8bit",
        "max_tokens": 262144,
        "temp": 1.0,
        "top_p": 0.95,
        "prefill_step_size": 8192,
        "prompt_cache_bytes": 17179869184,
    },
    # Gemma 4 26B-A4B MoE - PLE-safe 8-bit (~29GB), ~4B active params per token.
    "gemma-4-26b-a4b": {
        "hf_repo": "FakeRockert543/gemma-4-26b-a4b-it-MLX-8bit",
        "max_tokens": 262144,
        "temp": 1.0,
        "top_p": 0.95,
        "prefill_step_size": 4096,
        "prompt_cache_bytes": 17179869184,
    },
}

# Allowlist of HF repo patterns known to be PLE-safe for Gemma 4.
# Standard mlx-community/unsloth quants corrupt Per-Layer Embedding layers
# (their config.json lists every layer in `quantization`, including PLE/ScaledLinear).
# FakeRockert543's MLX quants skip those layers; bf16 has no quantization to corrupt.
PLE_SAFE_PATTERNS = [
    r"^FakeRockert543/gemma-4-.+-MLX-(4bit|8bit|bf16)$",
    r"^mlx-community/gemma-4-.+-bf16$",
]


def is_ple_safe(hf_repo):
    return any(re.match(p, hf_repo) for p in PLE_SAFE_PATTERNS)


def find_mlx_server():
    """Locate mlx_lm.server (pip --user installs it outside the default PATH)."""
    found = shutil.which("mlx_lm.server")
    if found:
        return found
    fallback = os.path.expanduser("~/Library/Python/3.10/bin/mlx_lm.server")
    if os.path.exists(fallback):
        return fallback
    print(color("mlx_lm.server not found. Install with: pip3 install --user mlx-lm", C.RED))
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run mlx_lm.server (M4 Max only)")
    parser.add_argument("--model", choices=list(MODELS), default="ling-2.6-flash")
    parser.add_argument("--port", type=int, default=8083)
    args = parser.parse_args()

    mlx_server_path = find_mlx_server()

    # 5-bit weights are ~72GB on disk -- M4 Max (128GB) only.
    detect_hardware(require_m4=True)

    c = MODELS[args.model]

    # ============================================================
    # Display banner
    # ============================================================
    os.system("clear")
    line = "=" * 48
    print(color(line, C.BLUE))
    print(color("   mlx_lm.server - M4 Max Optimized", C.BLUE))
    print(color(line, C.BLUE))
    print(color(f"   Model:   {args.model}", C.CYAN))
    print(color(f"   HF repo: {c['hf_repo']}", C.CYAN))
    print(color(f"   Host:    {NETWORK_IP}:{args.port}", C.CYAN))
    print(color(line, C.BLUE))
    print()
    print(color("Configuration:", C.GREEN))
    print(color(f"  Max tokens:   {c['max_tokens']}", C.GRAY))
    print(color(f"  Sampling:     temp={c['temp']}, top_p={c['top_p']}", C.GRAY))
    print(color(f"  Prefill step: {c['prefill_step_size']}", C.GRAY))
    print(color(f"  Prompt cache: {round(c['prompt_cache_bytes'] / (1024 ** 3))}GB", C.GRAY))
    print(color("  Engine:       mlx-lm (Apple Silicon native)", C.GRAY))
    if args.model.startswith("gemma-4-"):
        print(color("  PLE-safe:     yes (FakeRockert543 PLE-aware quantization)", C.GRAY))
    print()
    print(color("Endpoints:", C.GREEN))
    print(color(f"  OpenAI-compatible: http://{NETWORK_IP}:{args.port}/v1/chat/completions", C.YELLOW))
    print()
    print(color("Press Ctrl+C to stop", C.YELLOW))
    print(color(line, C.BLUE))
    print()

    # ============================================================
    # Static PLE-safety check for Gemma 4 (allowlist of known-good repos).
    # ============================================================
    if args.model.startswith("gemma-4-") and not is_ple_safe(c["hf_repo"]):
        print(color(f"Aborting: {c['hf_repo']} is not on the PLE-safe allowlist.", C.RED))
        print(color("Standard MLX Gemma 4 quants corrupt Per-Layer Embeddings.", C.YELLOW))
        sys.exit(1)

    # ============================================================
    # Build and run mlx_lm.server
    # ============================================================
    cmd = [
        mlx_server_path,
        "--model", c["hf_repo"],
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--max-tokens", str(c["max_tokens"]),
        "--temp", str(c["temp"]),
        "--top-p", str(c["top_p"]),
        "--prefill-step-size", str(c["prefill_step_size"]),
        "--prompt-cache-bytes", str(c["prompt_cache_bytes"]),
        "--trust-remote-code",
    ]

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
