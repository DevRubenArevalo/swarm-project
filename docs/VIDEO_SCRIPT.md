# 3-Minute Video Script — Swarm Fidelity

**Target: 3:00. ~430 spoken words + pauses for on-screen figures.**
Delivery note: the whole piece is one reversal. Don't soften the setup — the
"we were wrong" turn at 0:45 only lands if the first 45 seconds sound confident.

---

### 0:00 – 0:20 — The question

> **VISUAL:** Title card, then the DGX Spark terminal with `ollama ps`.

"We wanted to answer a simple question for Track 1: how far can a cheap local
model stand in for a frontier agent?

So we built a testbed on an NVIDIA DGX Spark and ran swarms of 5 to 60
concurrent agents across three local models — 4B, 9B, and 14B — measuring where
fidelity breaks down."

---

### 0:20 – 0:45 — The result we got first

> **VISUAL:** Figure 1, orange line only. Let the collapse sit on screen.

"And we found a beautiful result. Task correctness collapsed from 40% down to
8% as the swarm grew. A clean, monotonic degradation curve. Exactly the kind of
break point we went looking for.

It was completely wrong. Every single point on that curve was an artifact of
our own measurement harness."

---

### 0:45 – 1:40 — The four artifacts

> **VISUAL:** Four items building up one at a time.

"We found four of them.

**One: our swarm was never concurrent.** The inference server was running one
request at a time, so a swarm of 60 was really a queue of 60. And here's the
part that should worry anyone doing this work — both standard parallelism
metrics report *perfect parallelism* for a fully serial queue. It's not that we
misread them. It's algebraically impossible for them to tell the difference.

**Two: our scorer couldn't fail.** It checked the shape of a response, not its
content. A confidently wrong answer scored 100%. Worse, our sycophancy test
injects a wrong answer and asks if agents copy it — but a copied wrong answer
also scored 100%. The experiment couldn't detect the one thing it existed to
measure.

**Three: a token budget pretending to be a capability limit.** These are
reasoning models. At 512 tokens, up to 97% of agents burned the whole budget
thinking and returned an empty string — which we scored as the model failing.

**Four: our task got harder as the swarm got bigger.** Agent 59 had to compute
the 59th prime; agent 4 computed the 4th. So 'fidelity versus swarm size' was
really 'fidelity versus difficulty.'"

---

### 1:40 – 2:20 — The corrected result

> **VISUAL:** Figure 1 complete — the blue series appears, near-flat at the top. Then Figure 2.

"So we fixed all four and held per-agent difficulty constant.

The collapse disappears. The 4B and 9B hold 100% correctness from 5 agents all
the way to 60. The 14B holds 98.5 to 100%.

There is no swarm-size break point. A 4B model sustained 35 effectively
concurrent agents at full accuracy. And across 480 agent responses, **not one
agent copied the injected wrong answer** — zero sycophancy.

The only real ceiling we found was throughput: the 14B model saturates at about
10 effective concurrent agents where the 4B reaches 35. Model size caps
concurrency. Swarm size doesn't."

---

### 2:20 – 3:00 — Why it matters

> **VISUAL:** The reproduction checklist, then the negative-control command
> printing `⛔ SERIALIZED` and exiting 1.

"Here's our actual contribution. All four artifacts produce curves that look
like alignment findings — degradation with scale, a capability cliff, bigger
models following instructions worse. All four are infrastructure bugs. And
they're quiet: the experiments are easy to run and the metrics never complain.

So we're releasing the harness with a negative control for each one — a
condition where the metric *must* fail. This is our concurrency check run
against the broken server: it correctly reports no parallelism and exits
non-zero. That's what earns the right to trust it on the fixed server.

If you're benchmarking local-model swarms: verify your concurrency, score
content instead of format, check your truncation reason, and hold difficulty
constant.

Cheap models stand in better than we expected. Our benchmarks don't."

---

## Shot list

| # | Visual | Source |
|---|---|---|
| 1 | Title card | — |
| 2 | `ollama ps` / Spark terminal | live |
| 3 | Figure 1, orange series only | `fig1_fidelity_curve.png` |
| 4 | Four-artifact build | slides |
| 5 | Figure 1 complete (flat blue line) | `fig1_fidelity_curve.png` |
| 6 | Figure 2, truncation mirror-image | `fig2_truncation.png` |
| 7 | Figure 3, concurrency ceiling | `fig3_concurrency_ceiling.png` |
| 8 | `--verify-concurrency` on :11434 → `⛔ SERIALIZED`, exit 1 | live terminal |
| 9 | Reproduction checklist | `REPORT.md` §7 |

## Numbers to get right on screen

- 40% → 8% (the false collapse, 4B)
- corrected, difficulty constant: 4B and 9B 100% flat N=5→60; 14B 98.5–100%
  (pooled over 4 runs at N=50/60. Do NOT say "100% on every model")
- the 14B's 6 errors are all the SAME item: index 8, answered 19, correct 23
- 0.90 / 0.97 / 0.83 truncation at 512 tokens (4B / 9B / 14B)
- 35.2 vs 9.5 effective concurrency at N=60 (4B vs 14B)
- 0 of 480 agents copied the lure (5+15+30+50+60 per model x 3 models)

---

## Note — a rendered video already exists

`video/swarm_fidelity_3min.mp4` is a finished 180.00 s presentation cut from the
dashboard datasets and the report figures (see `video/README.md`).

That video uses a **tighter 501-word narration**, kept in `video/narration.py`,
not the script above. The two differ deliberately: the script above is written
for a **human presenter on camera** with a terminal and live figures; the
rendered video is a **slide deck** that must also read correctly on mute.

Use this file if you are recording yourself. Use `video/narration.py` if you are
re-voicing the rendered video, since its lines are timed to the cuts.
