#!/usr/bin/env python3
"""Probe a running model server and benchmark it with a coding-prompt suite.

Discovers whatever model is currently running via /v1/models, then sends a few
coding prompts and reports tokens/sec per prompt plus an overall average.

    python tests/test-performance.py
    python tests/test-performance.py --server-host 192.168.1.101 --iterations 3
    python tests/test-performance.py --port 9000
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# Allow importing the shared module from servers/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "servers"))
from hardware import color, C  # noqa: E402

PROMPTS = [
    "Write a minimal C# hello world program",
    "Create a simple PowerShell function to list files",
    "Write a Dockerfile for a .NET 10 application",
]


def http_get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def http_post_json(url, body, timeout):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main():
    parser = argparse.ArgumentParser(description="Benchmark a running model server")
    parser.add_argument("--server-host", default="localhost")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    endpoint = f"http://{args.server_host}:{args.port}"

    print(color(f"\nProbing {endpoint} for running model...", C.CYAN))
    try:
        resp = http_get_json(f"{endpoint}/v1/models", timeout=2)
        api_id = resp["data"][0]["id"]
    except (urllib.error.URLError, KeyError, IndexError, TimeoutError):
        print(color(
            f"No server responding on {endpoint}. "
            f"Start one with: python run.py --family qwen|ling --port {args.port}",
            C.RED,
        ))
        sys.exit(1)
    print(color(f"  found: {api_id}", C.GREEN))

    chat_endpoint = f"{endpoint}/v1/chat/completions"
    results = []

    print(color(f"\n=== {api_id} ===", C.CYAN))
    print(color(f"Endpoint:   {chat_endpoint}", C.YELLOW))
    print(color(f"API id:     {api_id}", C.YELLOW))
    print(color(f"Iterations: {args.iterations}", C.YELLOW))
    print(color(f"Max tokens: {args.max_tokens}", C.YELLOW))

    for prompt in PROMPTS:
        print(color(f"\nTesting: {prompt[:50]}...", C.GRAY))

        for i in range(1, args.iterations + 1):
            body = {
                "model": api_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0.7,
                "stream": False,
            }

            start = time.monotonic()
            try:
                response = http_post_json(chat_endpoint, body, timeout=300)
                total_sec = time.monotonic() - start

                gen_tokens = response["usage"]["completion_tokens"]
                gen_rate = gen_tokens / total_sec if total_sec > 0 else 0

                results.append({
                    "model": api_id,
                    "prompt": prompt[:30] + "...",
                    "iteration": i,
                    "prompt_tokens": response["usage"]["prompt_tokens"],
                    "generated_tokens": gen_tokens,
                    "total_time": round(total_sec, 2),
                    "generation_rate": round(gen_rate, 2),
                })

                print(color(
                    f"  Iteration {i}: {gen_rate:.2f} tok/s "
                    f"({gen_tokens} tokens in {total_sec:.2f}s)", C.GREEN))
            except (urllib.error.URLError, KeyError, TimeoutError) as e:
                print(color(f"  Iteration {i}: Failed - {e}", C.RED))

    if not results:
        print(color("\nNo successful runs.", C.RED))
        sys.exit(1)

    print(color("\n=== Results ===", C.CYAN))
    header = f"{'Prompt':<35}{'Iter':>5}{'PromptTok':>11}{'GenTok':>8}{'Time':>8}{'tok/s':>9}"
    print(header)
    for r in results:
        print(f"{r['prompt']:<35}{r['iteration']:>5}{r['prompt_tokens']:>11}"
              f"{r['generated_tokens']:>8}{r['total_time']:>8}{r['generation_rate']:>9}")

    avg = sum(r["generation_rate"] for r in results) / len(results)
    print(color("\n=== Average ===", C.CYAN))
    print(color(f"  {api_id:<30} {avg:>7.2f} tok/s", C.GREEN))
    print()


if __name__ == "__main__":
    main()
