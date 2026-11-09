#!/usr/bin/env python3
"""
Swarm smoke test: spawn N agents concurrently against Ollama.
Each agent gets a unique ID and a simple task. We measure:
  - total wall time
  - per-agent latency
  - success/failure
  - whether all agents actually ran in parallel (not serial)
"""

import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3.5:4b"
NUM_AGENTS = 60
TIMEOUT = 120  # seconds per agent


def call_agent(agent_id: int) -> dict:
    """Send a single prompt to Ollama and return the result."""
    prompt = (
        f"You are agent #{agent_id} in a swarm of {NUM_AGENTS} agents. "
        f"Reply with exactly one line in this format:\n"
        f"AGENT {agent_id} | STATUS: ready | TASK: acknowledged\n"
        f"Do not add anything else."
    )

    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 64,
        },
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = json.loads(resp.read())
        elapsed = time.monotonic() - start
        return {
            "agent": agent_id,
            "ok": True,
            "response": body.get("response", "").strip(),
            "elapsed_s": round(elapsed, 2),
            "eval_count": body.get("eval_count", 0),
        }
    except urllib.error.HTTPError as e:
        elapsed = time.monotonic() - start
        return {"agent": agent_id, "ok": False, "error": f"HTTP {e.code}: {e.reason}", "elapsed_s": round(elapsed, 2)}
    except Exception as e:
        elapsed = time.monotonic() - start
        return {"agent": agent_id, "ok": False, "error": str(e), "elapsed_s": round(elapsed, 2)}


def main():
    print(f"{'='*60}")
    print(f"  SWARM SMOKE TEST")
    print(f"  Ollama : {OLLAMA_URL}")
    print(f"  Model  : {MODEL}")
    print(f"  Agents : {NUM_AGENTS} (concurrent)")
    print(f"{'='*60}\n")

    # Quick connectivity check
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=5) as r:
            ver = json.loads(r.read())
        print(f"  Ollama version: {ver.get('version', '?')}")
    except Exception:
        print("  (could not read version, continuing)")

    print()

    t0 = time.monotonic()
    results = []

    with ThreadPoolExecutor(max_workers=NUM_AGENTS) as pool:
        futures = {pool.submit(call_agent, i): i for i in range(NUM_AGENTS)}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            status = "✓" if r["ok"] else "✗"
            print(f"  {status} Agent {r['agent']}  {r['elapsed_s']:>6.2f}s  {r.get('response', r.get('error', ''))[:80]}")

    total = time.monotonic() - t0
    ok_count = sum(1 for r in results if r["ok"])
    max_lat = max(r["elapsed_s"] for r in results)
    min_lat = min(r["elapsed_s"] for r in results)

    # Parallelism check: if truly parallel, total ≈ max_lat, not sum
    sum_lat = sum(r["elapsed_s"] for r in results)
    parallel_ratio = total / max_lat if max_lat > 0 else 0

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"  {'─'*40}")
    print(f"  Agents OK      : {ok_count}/{NUM_AGENTS}")
    print(f"  Total wall time: {total:.2f}s")
    print(f"  Min / Max lat  : {min_lat:.2f}s / {max_lat:.2f}s")
    print(f"  Sum of latencies: {sum_lat:.2f}s")
    print(f"  Parallelism    : {'YES ✓' if parallel_ratio < 1.5 else 'NO ✗ (looks serial)'}  (ratio={parallel_ratio:.2f})")
    print(f"{'='*60}")

    if ok_count < NUM_AGENTS:
        print("\n  ⚠ Some agents failed. Check the Ollama server logs.")
        for r in results:
            if not r["ok"]:
                print(f"    Agent {r['agent']}: {r.get('error')}")

    return 0 if ok_count == NUM_AGENTS else 1


if __name__ == "__main__":
    raise SystemExit(main())
