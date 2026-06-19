#!/usr/bin/env python3
"""Thin dispatcher for the local model servers.

Pass --family to launch a family's default model, or --model for a specific
model (the family is inferred from the name). With neither, the default family
is qwen (qwen3.6-moe). --port is forwarded either way. Only one model runs at a
time, so all servers share a single port (default 8083).

    python run.py                          # default: qwen3.6-moe via llama-server
    python run.py --family qwen            # qwen3.6-moe via llama-server
    python run.py --family ling            # ling-2.6-flash via mlx_lm.server
    python run.py --family gemma           # gemma-4-31b via mlx_lm.server
    python run.py --model qwen3.6-moe      # specific model -- family inferred
    python run.py --model qwen3.6-moe --port 9000
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

FAMILIES = ["qwen", "ling", "gemma"]
DEFAULT_FAMILY = "qwen"  # `python run.py` with no args launches qwen3.6-moe
MODELS = [
    "qwen3.6", "qwen3.6-moe", "qwen3.5", "qwen3.5-small", "qwen3-coder-next",
    "ling-2.6-flash", "gemma-4-31b", "gemma-4-26b-a4b",
]


def runner(name):
    return [sys.executable, os.path.join(HERE, name)]


def main():
    parser = argparse.ArgumentParser(description="Dispatch a local model server")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--family", choices=FAMILIES)
    group.add_argument("--model", choices=MODELS)
    parser.add_argument("--port", type=int, default=8083)
    args = parser.parse_args()

    port = ["--port", str(args.port)]

    llama = "servers/run-llama-server.py"
    mlx = "servers/run-mlx-server.py"

    if args.model:
        if args.model.startswith("ling-") or args.model.startswith("gemma-"):
            cmd = runner(mlx) + ["--model", args.model] + port
        else:
            cmd = runner(llama) + ["--model", args.model] + port
    else:
        family = args.family or DEFAULT_FAMILY
        if family == "qwen":
            cmd = runner(llama) + port
        elif family == "ling":
            cmd = runner(mlx) + port
        else:  # gemma
            cmd = runner(mlx) + ["--model", "gemma-4-31b"] + port

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
