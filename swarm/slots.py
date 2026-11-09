"""
Live view of the inference slots — the closest thing to "list the agents".

There is no such thing as an "agent" inside Ollama. An agent in this project is
one concurrent HTTP request; what the server actually has is a fixed pool of
llama.cpp *slots* (OLLAMA_NUM_PARALLEL=64 on our benchmark instance). A slot is
the unit that holds one request's KV cache and generates its tokens, so the
honest answer to "how many agents are spun up" is "how many slots are
processing right now", plus the socket count for requests still queued.

The llama.cpp /slots endpoint is bound to 127.0.0.1 on a random internal port
and is not exposed through Ollama's API, so this probe must run on the host
itself. Set SWARM_SSH_TARGET to reach a remote server; unset, it runs locally.

Usage:
    python3 -m swarm.slots            # one snapshot
    python3 -m swarm.slots --watch    # refresh until Ctrl-C
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

# Where the Ollama/llama.cpp server runs. If it is this machine, set
# SWARM_SSH_TARGET="" (or leave it unset) and the commands run locally.
# For a remote server, export e.g. SWARM_SSH_TARGET=user@gpu-box and, if the
# key is not your default identity, SWARM_SSH_KEY=~/.ssh/id_ed25519.
SSH_TARGET = os.environ.get("SWARM_SSH_TARGET", "").strip()
SSH_KEY = os.environ.get("SWARM_SSH_KEY", "").strip()

if SSH_TARGET:
    SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    if SSH_KEY:
        SSH[1:1] = ["-i", os.path.expanduser(SSH_KEY)]
    SSH.append(SSH_TARGET)
else:
    # Run the probe locally.
    SSH = ["bash", "-c"]

# Find the llama-server belonging to our PARALLEL instance (-np > 1), read its
# slot table, and count live sockets on both Ollama ports.
REMOTE = r"""
PORT=""
NP=""
# No `set -e`: a short-lived process can vanish between pgrep and the read,
# and one missing /proc entry must not abort the whole snapshot.
for p in $(pgrep -f 'llama-server' 2>/dev/null); do
  [ -r "/proc/$p/cmdline" ] || continue
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null) || continue
  np=$(echo "$cmd" | grep -oE '\-np [0-9]+' | awk '{print $2}')
  pt=$(echo "$cmd" | grep -oE '\-\-port [0-9]+' | awk '{print $2}')
  if [ -n "$np" ] && [ "$np" -gt 1 ] 2>/dev/null; then PORT=$pt; NP=$np; fi
done
echo "PORT=${PORT:-none}"
echo "NP=${NP:-0}"
echo "CONN_BENCH=$(ss -tn 2>/dev/null | grep -c ':11435' || true)"
echo "CONN_SHARED=$(ss -tn 2>/dev/null | grep -c ':11434' || true)"
if [ -n "$PORT" ]; then
  echo "SLOTS_BEGIN"
  curl -s --max-time 5 "http://127.0.0.1:$PORT/slots" || echo "[]"
  echo
  echo "SLOTS_END"
fi
"""


def fetch() -> dict:
    out = subprocess.run(SSH + [REMOTE], capture_output=True, text=True, timeout=40)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "slot probe failed")
    meta: dict = {}
    slots_raw = ""
    in_slots = False
    for line in out.stdout.splitlines():
        if line == "SLOTS_BEGIN":
            in_slots = True
            continue
        if line == "SLOTS_END":
            in_slots = False
            continue
        if in_slots:
            slots_raw += line
        elif "=" in line:
            k, v = line.split("=", 1)
            meta[k] = v
    try:
        meta["slots"] = json.loads(slots_raw) if slots_raw.strip() else []
    except json.JSONDecodeError:
        meta["slots"] = []
    return meta


def render(m: dict) -> str:
    slots = m.get("slots", [])
    busy = [s for s in slots if s.get("is_processing")]
    total = len(slots)
    conn_b = m.get("CONN_BENCH", "?")
    conn_s = m.get("CONN_SHARED", "?")

    lines = [
        f"inference slots on :11435  (llama-server internal port {m.get('PORT')})",
        f"  configured slots : {m.get('NP')}   reported by server: {total}",
        f"  PROCESSING now   : {len(busy)}",
        f"  idle             : {total - len(busy)}",
        f"  open sockets     : {conn_b} to :11435 (ours)   {conn_s} to :11434 (shared)",
        "",
    ]
    if total:
        # Visual grid: # = generating, . = idle
        grid = "".join("#" if s.get("is_processing") else "." for s in slots)
        for i in range(0, len(grid), 32):
            lines.append(f"  [{i:>3}] {grid[i:i+32]}")
        lines.append("")
    if busy:
        lines.append(f"  {'slot':>4} {'task':>6} {'ctx':>6} {'prompt_tok':>10} {'cached':>7}")
        for s in busy[:20]:
            lines.append(
                f"  {s.get('id',''):>4} {s.get('id_task',''):>6} "
                f"{s.get('n_ctx',''):>6} {s.get('n_prompt_tokens',''):>10} "
                f"{s.get('n_prompt_tokens_cache',''):>7}"
            )
        if len(busy) > 20:
            lines.append(f"  ... and {len(busy)-20} more processing")
    else:
        lines.append("  (no slot is generating — server idle between runs)")
    # Sockets far above busy slots means requests are queued, not parallel.
    try:
        if int(conn_b) > total and total:
            lines.append(
                f"\n  ⓘ {conn_b} sockets but only {total} slots — "
                f"{int(conn_b)-total} request(s) are QUEUED waiting for a slot."
            )
    except (TypeError, ValueError):
        pass
    return "\n".join(lines)


def main() -> int:
    watch = "--watch" in sys.argv
    try:
        while True:
            try:
                print("\033[2J\033[H" if watch else "", end="")
                print(render(fetch()))
            except Exception as e:
                print(f"error: {e}")
            if not watch:
                return 0
            time.sleep(2)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
