"""Live swarm progress monitor — shows in-flight state, not just completed runs.

Reads the experiment log to find the current run, tracks how long it's been
in flight, and queries Ollama /api/ps for what's actually loaded on the GPU.
Run it repeatedly (or in a loop) for a live view.
"""

from __future__ import annotations

import json
import os
import os as _os
import time
import urllib.request

# The sweep log written by `swarm.experiments`. Override with SWARM_LOG.
# Note :11434 is typically the SERIALIZED shared server (-np 1), not the
# benchmark instance — point SWARM_OLLAMA_URL at the parallel one.
LOG = _os.environ.get("SWARM_LOG", "output/sweep.log")
OLLAMA = _os.environ.get("SWARM_OLLAMA_URL", "http://localhost:11435")


def read_log() -> tuple[list[str], str | None]:
    """Return (completed_runs, current_run_in_flight)."""
    if not os.path.exists(LOG):
        return [], None
    with open(LOG) as f:
        lines = f.read().splitlines()

    completed = []
    current = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Running:"):
            current = line.strip()
            # If the next line is a completion, this run is done
            if i + 1 < len(lines) and "completion=" in lines[i + 1]:
                completed.append(line.strip() + "  →  " + lines[i + 1].strip())
                current = None
    return completed, current


def ollama_ps() -> list[dict]:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/ps", timeout=5) as r:
            data = json.loads(r.read())
        return data.get("models", [])
    except Exception as e:
        return [{"error": str(e)}]


def main() -> None:
    completed, current = read_log()
    models = ollama_ps()

    print("=" * 52)
    print("  LIVE SWARM MONITOR")
    print("=" * 52)

    print(f"\n  COMPLETED RUNS: {len(completed)}")
    for c in completed[-6:]:
        print(f"    {c}")

    if current:
        print(f"\n  🔄 IN FLIGHT: {current}")
    else:
        print("\n  ✅ No run in flight (finished or not started)")

    print("\n  GPU / OLLAMA STATE:")
    if models and "error" not in models[0]:
        for m in models:
            name = m.get("name", "?")
            size_gb = m.get("size_vram", 0) / 1e9
            proc = m.get("processor", "?")
            print(f"    • {name:24s} {size_gb:5.1f} GB  on {proc}")
    else:
        print("    (no models loaded / query failed)")

    print(f"\n  {time.strftime('%H:%M:%S')}")
    print("=" * 52)


if __name__ == "__main__":
    main()
