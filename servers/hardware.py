"""Shared Apple Silicon hardware detection and capability printout.

Every config in this repo is tuned for the M4 Max (128GB). Detection still runs
on whatever machine you're on and prints what it finds; non-M4 chips get a
warning (or an error, for engines that only make sense on the M4 Max).
"""
import subprocess
import sys
from dataclasses import dataclass

# Dummy LAN IP used in the "connect from another machine" banners. Replace with
# your Mac's real IP (e.g. `ipconfig getifaddr en0`) if you serve over a network.
NETWORK_IP = "192.168.1.101"


class C:
    """Minimal ANSI colour codes for terminal output."""
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"


def color(text, c):
    """Wrap text in an ANSI colour code."""
    return f"{c}{text}{C.RESET}"


def _sysctl(key):
    """Return a sysctl value as a stripped string ("" if unavailable)."""
    try:
        return subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


@dataclass
class Hardware:
    brand: str
    cores: int
    ram_gb: int
    is_m4: bool


def detect_hardware(require_m4=False):
    """Detect the Apple Silicon chip, core count and RAM, and print them.

    Prints a capability summary. If the chip isn't an M4, prints a warning and
    continues -- unless require_m4 is True, in which case it exits.
    """
    print(color("\nDetecting hardware...", C.CYAN))
    brand = _sysctl("machdep.cpu.brand_string")

    try:
        cores = int(_sysctl("machdep.cpu.core_count"))
    except ValueError:
        cores = 0

    try:
        ram_gb = round(int(_sysctl("hw.memsize")) / (1024 ** 3))
    except ValueError:
        ram_gb = 0

    is_m4 = "M4" in brand
    hw = Hardware(brand=brand, cores=cores, ram_gb=ram_gb, is_m4=is_m4)

    print(color(f"Detected: {brand} ({cores} cores, {ram_gb}GB)", C.GREEN))

    if not is_m4:
        message = f"configs in this repo are tuned for the M4 Max. Detected: {brand or 'unknown'}."
        if require_m4:
            print(color(f"ERROR: {message}", C.RED))
            sys.exit(1)
        print(color(f"WARNING: {message} Continuing anyway.", C.YELLOW))

    return hw
