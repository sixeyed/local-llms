#!/usr/bin/env python3
"""Run llama.cpp's llama-server with M4 Max-tuned settings.

Serves an OpenAI-compatible API. GGUF files are auto-downloaded from HuggingFace
on first run. Install the engine with: brew install llama.cpp
"""
import argparse
import os
import shutil
import subprocess
import sys

from hardware import detect_hardware, NETWORK_IP, color, C

# ============================================================
# Model defaults (sampling params from unsloth recommended settings)
# ============================================================
# Thinking is disabled by default for the qwen3 family (reasoning_budget = 0).
# Observed under Copilot Agent and Cline that thinking-mode qwen3 models defer
# ("let me investigate") rather than commit, exhausting the client's
# max-tool-calls-per-turn before producing an answer. Disabling thinking makes
# the model commit faster on small/exploratory questions; quality on multi-step
# planning is no worse in practice. Set to a positive number (e.g. 1000) to
# re-enable a capped thinking budget for experiments.
MODELS = {
    "qwen3.6": {
        "hf_repo": "unsloth/Qwen3.6-27B-GGUF",
        "temp": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0,
        "predict": 32768,
        "reasoning_budget": 0,
    },
    "qwen3.6-moe": {
        "hf_repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "temp": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0,
        "predict": 32768,
        "reasoning_budget": 0,
    },
    "qwen3.5": {
        "hf_repo": "unsloth/Qwen3.5-122B-A10B-GGUF",
        "temp": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0,
        "predict": 32768,
        "reasoning_budget": 0,
    },
    "qwen3.5-small": {
        "hf_repo": "unsloth/Qwen3.5-35B-A3B-GGUF",
        "temp": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0,
        "predict": 32768,
        "reasoning_budget": 0,
    },
    "qwen3-coder-next": {
        "hf_repo": "unsloth/Qwen3-Coder-Next-GGUF",
        "temp": 1.0, "top_p": 0.95, "top_k": 40, "min_p": 0.01, "repeat_penalty": 1.0,
        "predict": 65536,
        # No reasoning_budget -- coder variant doesn't use thinking mode.
    },
}

# ============================================================
# M4 Max (128GB) per-model quant + context overrides
# ============================================================
M4_OVERRIDES = {
    "qwen3.6": {
        "hf_file": "Qwen3.6-27B-UD-Q8_K_XL.gguf",  # 35GB - best available quant, 27B dense
        "ctx": 131072, "parallel": 1,
    },
    "qwen3.6-moe": {
        "hf_file": "Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf",  # 38.5GB - 3B active, near-lossless
        "ctx": 262144, "parallel": 1,
    },
    "qwen3.5": {
        "hf_file": "UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf",  # 77GB split - 10B active params
        "ctx": 262144, "parallel": 1,
    },
    "qwen3.5-small": {
        "hf_file": "BF16/Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf",  # 69GB split - 3B active params
        "ctx": 131072, "parallel": 2,
    },
    "qwen3-coder-next": {
        "hf_file": "UD-Q8_K_XL/Qwen3-Coder-Next-UD-Q8_K_XL-00001-of-00003.gguf",  # 86GB split
        "ctx": 262144, "parallel": 1,
    },
}

# M4 Max machine-level settings
THREADS = 12
BATCH = 4096
UBATCH = 2048


def main():
    parser = argparse.ArgumentParser(description="Run llama-server (M4 Max tuned)")
    parser.add_argument("--model", choices=list(MODELS), default="qwen3.6-moe")
    parser.add_argument("--port", type=int, default=8083)
    args = parser.parse_args()

    # Check llama-server is installed
    if not shutil.which("llama-server"):
        print(color("llama-server not found. Install with: brew install llama.cpp", C.RED))
        sys.exit(1)

    detect_hardware()

    # ============================================================
    # Merge model defaults + M4 overrides into final config
    # ============================================================
    c = dict(MODELS[args.model])
    c["threads"] = THREADS
    c["batch"] = BATCH
    c["ubatch"] = UBATCH
    c.update(M4_OVERRIDES[args.model])
    repo = c["hf_repo"]

    # ============================================================
    # Display banner
    # ============================================================
    os.system("clear")
    line = "=" * 48
    print(color(line, C.BLUE))
    print(color("   llama-server - M4 Max Optimized", C.BLUE))
    print(color(line, C.BLUE))
    print(color(f"   Model:   {args.model}", C.CYAN))
    print(color(f"   HF repo: {repo}", C.CYAN))
    if c.get("hf_file"):
        print(color(f"   Quant:   {c['hf_file']}", C.CYAN))
    print(color(f"   Host:    {NETWORK_IP}:{args.port}", C.CYAN))
    print(color(line, C.BLUE))
    print()
    print(color("Configuration:", C.GREEN))
    print(color(f"  Context:      {c['ctx']} tokens", C.GRAY))
    print(color(f"  Batch:        {c['batch']} (ubatch: {c['ubatch']})", C.GRAY))
    print(color(f"  Threads:      {c['threads']}", C.GRAY))
    print(color(f"  Max predict:  {c['predict']}", C.GRAY))
    print(color(f"  Parallel:     {c['parallel']}", C.GRAY))
    print(color("  GPU layers:   all (999)", C.GRAY))
    if "reasoning_budget" in c:
        if c["reasoning_budget"] == 0:
            print(color("  Reasoning:    disabled", C.GRAY))
        else:
            print(color(f"  Reasoning:    capped at {c['reasoning_budget']} tokens", C.GRAY))
    print(color("  Flash attn:   on", C.GRAY))
    print()
    print(color("Endpoints:", C.GREEN))
    print(color(f"  OpenAI-compatible: http://{NETWORK_IP}:{args.port}/v1/chat/completions", C.YELLOW))
    print(color(f"  Health check:      http://localhost:{args.port}/health", C.YELLOW))
    print()
    print(color("Press Ctrl+C to stop", C.YELLOW))
    print(color(line, C.BLUE))
    print()

    # ============================================================
    # Build and run llama-server
    # ============================================================
    cmd = [
        "llama-server",
        "-hf", repo,
        "--alias", args.model,
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "-c", str(c["ctx"]),
        "-b", str(c["batch"]),
        "-ub", str(c["ubatch"]),
        "-t", str(c["threads"]),
        "-n", str(c["predict"]),
        "-ngl", "999",
        "--parallel", str(c["parallel"]),
        "--flash-attn", "on",
        "--cache-ram", "32768",
        "-ctxcp", "128",
        "--no-mmproj",
        "--jinja",
        "--temp", str(c["temp"]),
        "--top-p", str(c["top_p"]),
        "--top-k", str(c["top_k"]),
        "--min-p", str(c["min_p"]),
        "--repeat-penalty", str(c["repeat_penalty"]),
    ]

    if c.get("hf_file"):
        cmd += ["--hf-file", c["hf_file"]]

    if "reasoning_budget" in c:
        cmd += ["--reasoning-budget", str(c["reasoning_budget"])]

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
