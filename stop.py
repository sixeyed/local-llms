#!/usr/bin/env python3
"""Stop any running model servers (llama-server, mlx_lm.server)."""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "servers"))
from hardware import color, C  # noqa: E402

PATTERNS = ["llama-server", "mlx_lm.server"]


def pids_for(pattern):
    result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [p for p in result.stdout.split() if p]


def main():
    stopped = 0
    for pattern in PATTERNS:
        for pid in pids_for(pattern):
            print(color(f"Killing {pattern} (pid {pid})", C.YELLOW))
            subprocess.run(["kill", pid])
            stopped += 1

    time.sleep(2)

    # Verify and force-kill anything still alive
    remaining = 0
    for pattern in PATTERNS:
        for pid in pids_for(pattern):
            print(color(f"Force-killing {pattern} (pid {pid})", C.RED))
            subprocess.run(["kill", "-9", pid])
            remaining += 1

    if stopped == 0 and remaining == 0:
        print(color("No model servers running.", C.GREEN))
    else:
        print(color(f"Stopped {stopped} server(s).", C.GREEN))


if __name__ == "__main__":
    main()
