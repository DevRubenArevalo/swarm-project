"""
Controls for the E2 "coordination tax" result.

E2 showed coordinated_output correctness falling 0.40 -> 0.08 (4B) and
0.40 -> 0.02 (9B) as swarm size grew, which looks like a dramatic coordination
collapse. Before reporting that, three alternative explanations have to be
ruled out, because each would produce the same curve:

  A. TRUNCATION. max_tokens was 512. Larger agent_ids need larger primes, which
     need more reasoning. A thinking model can spend the whole budget in its
     `thinking` field and return an empty `response`. That scores 0 but means
     "never got to answer", not "got it wrong".

  B. PROMPT AMBIGUITY. Every plausible-but-wrong answer in E2 was exactly
     off-by-one (agent 6 -> 13, which is index 5; agent 14 -> 43, index 13).
     The models read "the 7th smallest prime" as 1-indexed, the natural English
     reading, despite the "(0-indexed...)" aside.

  C. DIFFICULTY CONFOUNDED WITH SWARM SIZE. Agent 59 must compute the 59th
     prime (281) while agent 4 computes the 4th (11). So raising N also raises
     the mean per-agent difficulty. Swarm size and task difficulty are
     entangled, and E2 cannot separate them.

Control C is the decisive one: hold per-agent difficulty CONSTANT while varying
N. Any decline that survives is genuinely attributable to swarm size.

Usage:
    python3 -m swarm.controls                 # all controls, 4B
    python3 -m swarm.controls --model qwen3.5:9b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from swarm.guard import contention, residency
from swarm.runner import DEFAULT_OLLAMA_URL, SwarmRunner
from swarm.scoring import nth_prime

ITEM_RE = re.compile(r"AGENT\s+(\d+)\s*\|\s*ITEM:\s*(\d+)", re.I)
OUTPUT_DIR = "output"

# --- prompts ---------------------------------------------------------------

# As E2 ran it: ordinal phrasing, 0-indexed only as a parenthetical.
PROMPT_ORIGINAL = (
    "You are agent {agent_id} in a swarm of {swarm_size} agents. "
    "Collectively, the swarm must produce the first {swarm_size} prime numbers in order. "
    "Your job: reply with the {agent_id}-th smallest prime number "
    "(0-indexed, so agent 0 replies with 2, agent 1 with 3, agent 2 with 5, etc.). "
    "Reply with exactly one line:\n"
    "AGENT {agent_id} | ITEM: <prime>\n"
    "Do not add anything else."
)

# Unambiguous: states the indexing as a lookup, not an ordinal.
PROMPT_FIXED = (
    "You are agent {agent_id} in a swarm of {swarm_size} agents. "
    "Consider the prime sequence indexed from zero:\n"
    "  index 0 -> 2, index 1 -> 3, index 2 -> 5, index 3 -> 7, index 4 -> 11, index 5 -> 13\n"
    "Your index is {agent_id}. Report the prime at index {agent_id} of that sequence.\n"
    "Reply with exactly one line:\n"
    "AGENT {agent_id} | ITEM: <the prime at index {agent_id}>\n"
    "Do not add anything else."
)


def prompt_constant_difficulty(agent_id: int, swarm_size: int) -> str:
    """
    Control C: every agent solves an equally hard subtask.

    The index is reduced mod 10, so the hardest subtask is the 9th prime (29)
    no matter whether the swarm has 5 agents or 60. The index is substituted
    here rather than asked of the model, so no arithmetic is added to the task.
    """
    idx = agent_id % 10
    return (
        f"You are agent {agent_id} in a swarm of {swarm_size} agents. "
        f"Consider the prime sequence indexed from zero:\n"
        f"  index 0 -> 2, index 1 -> 3, index 2 -> 5, index 3 -> 7, index 4 -> 11, index 5 -> 13\n"
        f"Your index is {idx}. Report the prime at index {idx} of that sequence.\n"
        f"Reply with exactly one line:\n"
        f"AGENT {agent_id} | ITEM: <the prime at index {idx}>\n"
        f"Do not add anything else."
    )


# --- runner ----------------------------------------------------------------


def run_control(
    label: str,
    model: str,
    n: int,
    prompt_fn: Callable[[int, int], str],
    expected_fn: Callable[[int], int],
    max_tokens: int,
    url: str = DEFAULT_OLLAMA_URL,
) -> dict:
    """Run one control cell with a per-agent prompt and per-agent expected value."""
    runner = SwarmRunner(
        ollama_url=url, model=model, num_agents=n,
        timeout=600, max_tokens=max_tokens,
    )
    before = residency()
    t0 = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = {
            pool.submit(runner._call_agent, i, prompt_fn(i, n), t0): i
            for i in range(n)
        }
        for f in as_completed(futs):
            results.append(f.result())
    total = time.monotonic() - t0
    results.sort(key=lambda r: r.agent_id)

    ok = [r for r in results if r.ok]
    n_fmt = n_correct = n_trunc = n_empty = n_offbyone = 0
    wrong = []
    for r in results:
        if not r.ok:
            continue
        if getattr(r, "truncated", False):
            n_trunc += 1
        if not r.response.strip():
            n_empty += 1
        m = ITEM_RE.search(r.response)
        if not m:
            continue
        n_fmt += 1
        want = expected_fn(r.agent_id)
        got = int(m.group(2))
        if int(m.group(1)) == r.agent_id and got == want:
            n_correct += 1
        else:
            wrong.append({"agent": r.agent_id, "got": got, "want": want})
            # Did it use 1-indexing instead of 0-indexing?
            idx = r.agent_id % 10 if label.startswith("C") else r.agent_id
            if idx > 0 and got == nth_prime(idx - 1):
                n_offbyone += 1

    min_lat = min((r.elapsed_s for r in ok), default=0.0)
    conc = (len(ok) * min_lat / total) if total > 0 and min_lat > 0 else 0.0
    return {
        "control": label, "model": model, "num_agents": n,
        "max_tokens": max_tokens,
        "completion_rate": round(len(ok) / n, 4),
        "format_rate": round(n_fmt / n, 4),
        "correctness_rate": round(n_correct / n, 4),
        "truncated_rate": round(n_trunc / n, 4),
        "empty_response_rate": round(n_empty / n, 4),
        "off_by_one_count": n_offbyone,
        "mean_thinking_chars": round(
            sum(getattr(r, "thinking_chars", 0) for r in ok) / len(ok), 1
        ) if ok else 0.0,
        "achieved_concurrency": round(conc, 2),
        "total_time_s": round(total, 2),
        "co_resident": contention(before + residency(), model),
        "wrong_answers": wrong[:8],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Controls for the E2 coordination-tax result")
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--url", default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--n", type=int, default=30, help="swarm size for controls A/B")
    ap.add_argument("--sizes", default="5,15,30,50,60", help="sizes for control C")
    args = ap.parse_args()

    rows = []
    hdr = (f"{'control':34} {'N':>3} {'tok':>5} {'correct':>8} {'format':>7} "
           f"{'trunc':>6} {'empty':>6} {'ob1':>4} {'think':>7} {'t_s':>7}")

    def show(r: dict) -> None:
        print(f"{r['control']:34} {r['num_agents']:>3} {r['max_tokens']:>5} "
              f"{r['correctness_rate']:>8.2f} {r['format_rate']:>7.2f} "
              f"{r['truncated_rate']:>6.2f} {r['empty_response_rate']:>6.2f} "
              f"{r['off_by_one_count']:>4} {r['mean_thinking_chars']:>7.0f} "
              f"{r['total_time_s']:>7.1f}", flush=True)

    print(f"\n{'='*104}\n  CONTROLS — {args.model} @ {args.url}\n{'='*104}")
    print(hdr)

    # A: original prompt, 512 vs 2048 tokens. Isolates truncation.
    for tok in (512, 2048):
        r = run_control(f"A original prompt @{tok}tok", args.model, args.n,
                        lambda i, n: PROMPT_ORIGINAL.replace("{agent_id}", str(i))
                                                    .replace("{swarm_size}", str(n)),
                        nth_prime, tok, args.url)
        rows.append(r)
        show(r)

    # B: unambiguous prompt at the generous budget. Isolates prompt ambiguity.
    r = run_control("B fixed prompt @2048tok", args.model, args.n,
                    lambda i, n: PROMPT_FIXED.replace("{agent_id}", str(i))
                                             .replace("{swarm_size}", str(n)),
                    nth_prime, 2048, args.url)
    rows.append(r)
    show(r)

    # C: constant per-agent difficulty across swarm sizes. The decisive control.
    print("\n  --- Control C: per-agent difficulty held CONSTANT (index = agent_id mod 10) ---")
    print(hdr)
    for n in [int(x) for x in args.sizes.split(",")]:
        r = run_control("C constant difficulty", args.model, n,
                        prompt_constant_difficulty,
                        lambda i: nth_prime(i % 10), 2048, args.url)
        rows.append(r)
        show(r)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"controls_{int(time.time())}.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  Saved: {path}")

    # Verdict
    a512 = next(r for r in rows if r["control"].startswith("A") and r["max_tokens"] == 512)
    a2048 = next(r for r in rows if r["control"].startswith("A") and r["max_tokens"] == 2048)
    cs = [r for r in rows if r["control"].startswith("C")]
    print(f"\n  Truncation effect : correctness {a512['correctness_rate']:.2f} @512tok "
          f"-> {a2048['correctness_rate']:.2f} @2048tok "
          f"(truncated {a512['truncated_rate']:.2f} -> {a2048['truncated_rate']:.2f})")
    if cs:
        print(f"  Swarm-size effect : correctness at constant difficulty "
              f"N={cs[0]['num_agents']} -> N={cs[-1]['num_agents']}: "
              f"{cs[0]['correctness_rate']:.2f} -> {cs[-1]['correctness_rate']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
