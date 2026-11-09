# Progress Log

## 2026-11-07

### 15:32 — Project kickoff
- Reviewed hackathon page (AI Alignment Foundation, Swarm Alignment 2026)
- Brainstormed 12 project ideas across 4 tracks
- Selected: **Swarm Fidelity: Where Do Local Models Break?**
- Angle: Local model expert — quantitative fidelity curves across model sizes

### 15:40 — Infrastructure verification
- Confirmed Ollama server at `<gpu-host>:11434` (v0.40.15)
- Available models: qwen3.5 (0.8b–35b), deepseek-r1:14b, flint:40b, nemotron-3.5, etc.
- Testbed: NVIDIA DGX Spark (Blackwell, 128GB unified, 1000 TOPS)

### 15:43 — Swarm smoke test (5 agents)
- 5/5 agents, qwen3.5:4b, 8.1s total, perfectly parallel
- Ollama handles concurrent requests correctly

### 15:50 — Swarm smoke test (15 agents)
- 15/15 agents, 18.5s total, perfectly parallel
- Linear scaling confirmed

### 15:53 — Swarm smoke test (50 agents)
- 50/50 agents, 53.0s total, perfectly parallel
- No failures, GPU-bound but graceful

### 15:54 — Swarm smoke test (60 agents)
- 60/60 agents, 68.5s total, perfectly parallel
- **Ceiling established: 50–60 concurrent agents on 4B model**

### 16:09 — Project build begins
- Created project structure (README, docs/)
- Built all core components:
  - ✅ `swarm/runner.py` — SwarmRunner class, concurrent agent spawning, self-test passes (3/3)
  - ✅ `swarm/tasks.py` — 4 tasks (identity, math_relay, role_adherence, coordinated_output)
  - ✅ `swarm/metrics.py` — completion_rate, drift_score, sycophancy_index, latency_stats, self-test passes
  - ✅ `swarm/charts.py` — fidelity_curve + scaling_comparison, synthetic data test passes
  - ✅ `swarm/experiments.py` — E1-E4 orchestration, CLI interface
- All components verified working

### 16:16 — E1: Scaling ceiling (first attempt)
- Running: 3 models × 4 swarm sizes × identity task
- Models: qwen3.5:4b, qwen3.5:9b, deepseek-r1:14b
- Sizes: 5, 15, 30, 50
- ⚠️ BUG FOUND: drift=0.0 on all runs — qwen3.5 is a THINKING model, spending all
  num_predict tokens on "thinking" field, leaving "response" empty.
- FIX: bumped max_tokens 64→512 in runner.py + all tasks. Verified: drift now 1.0.

### 16:52 — E1 rerun (contaminated)
- 4B model: 5/15/30 agents = 1.0 completion, but 50 agents dropped to 0.56
- ⚠️ DISCOVERED: qwen3.8:27b (18GB) co-resident on the Spark — NOT one of our models.
  Contending for GPU. Data invalid.

### 17:08 — Built the clean-run guard
- swarm/guard.py: blocks a run if a foreign model is resident on the GPU
- Wired into all of E1–E4 in experiments.py
- Purpose: guarantee clean, defensible measurements (the "local model expert" claim)

### 17:41 — SSH access established
- Authorized an SSH key on the Spark for a dedicated low-privilege user
- Can now run diagnostics + ollama commands on the Spark directly

### 17:50 — Root-caused the 27B auto-reload
- qwen3.8:27b is ACTIVELY generating (task counter climbing) + re-loading repeatedly
- keep_alive set to ~30 min ("UNTIL 29 minutes from now")
- Suspects: gpu-on-demand-proxy.py (×3), open_webui (root, port 8080)
- ollama stop clears CLI view but llama-server (user=ollama) outlives it; sudo needs password
- User: it's leftover from other projects, no one using it, wants it stopped from auto-reloading

### 18:09 — HANDOFF to cloud agent
- Wrote docs/HANDOFF.md — complete context, the Ollama auto-reload fix (7 steps),
  all gotchas, next steps, report structure
- E1–E4 NOT yet run with clean data. Cloud agent picks up from §7 of HANDOFF.md.

### 18:20–18:45 — ⚠️ CORRECTION: two prior conclusions were wrong

Took over from HANDOFF.md. Investigated §5 and found the handoff's diagnosis
was mistaken on both counts. Evidence below.

**(1) The 27B was NOT a rogue auto-reloading service — it was our own chat client.**
- Ollama's GIN access log attributes all `/api/chat` traffic to two LAN
  addresses: this workstation (176 requests) and one laptop (20). A reverse
  lookup confirms both are our own devices — no third party, no daemon.
- Prompt-cache similarity climbed 70346 → 76944 tokens across turns: a single
  long human conversation, not a daemon.
- **No `OLLAMA_KEEP_ALIVE` exists in systemd.** `systemctl show ollama -p
  Environment` has only PATH and OLLAMA_HOST. The "UNTIL 29 minutes" was a
  per-request `keep_alive` set by the chat client.
- The `gpu-on-demand-proxy@{acestep,comfyui,qwen-streaming}` units were
  innocent: they front ports 8002/8003, never 11434.
- It was idle (0% CPU) and self-expiring. 87 GB of 121 GB was free, so it was
  never crowding us out. → Stopped it at user's request for a clean cold run.

**(2) THE REAL BUG: there was never any concurrency. `-np 1`.**
Every `llama-server` on :11434 runs with `-np 1` and `OLLAMA_NUM_PARALLEL` is
unset, so Ollama serves one request per model at a time. Measured on the 4B:

| N agents | total_s | min_lat | max_lat |
|---|---|---|---|
| 1 | 3.64 | 3.64 | 3.64 |
| 2 | 7.19 | 3.92 | 7.18 |
| 4 | 14.09 | 3.28 | 14.09 |
| 8 | 27.37 | 3.93 | 27.37 |

Service time stays ~3.6s; total scales linearly with N. **That is a FIFO queue
of depth 1.** Every "swarm" we ran was a sequential queue.

The earlier smoke tests say the same thing: 5→8.1s, 15→18.5s, 50→53.0s,
60→68.5s is a straight line at ~1.1s/agent. We logged that as "perfectly
parallel. Linear scaling confirmed" — but linear total time *is* the proof of
serialization. **The "ceiling of 50–60 concurrent agents" never existed.**

Both metrics were algebraically blind to this. For a depth-1 queue, agent k's
latency is (k+1)·s, so:
- `parallelism_ratio = total/max_lat = N·s/(N·s) ≡ 1.00` → always "perfect"
- `parallelism_efficiency = total/sum = 2/(N+1) ≈ 1/N` → the docstring calls
  1/N "perfectly parallel"

A fully serialized run was *indistinguishable* from a perfect one.

**(3) The "0.56 completion at 50 agents" was a client timeout, not contention.**
`run_single` used `timeout=180`. Serialized, agent 49 waits ~50×3.6 = 180s —
exactly the limit. The tail agents were scored as model failures. The 17:08
guard was built to fix a problem that wasn't there.

### Fixes applied
- `runner.py`: `achieved_concurrency = N · min_latency / total_time`
  (the fastest request has no queue wait, so it approximates service time).
  Serial → 1.0; true N-way → N. Also records per-agent start/end offsets and
  Ollama's server-side `load_duration` / `prompt_eval_duration` /
  `eval_duration` / `eval_count`, plus a `warmup()` that pays cold-start once
  and reports it separately.
- `metrics.py`: `achieved_concurrency` added; the two old metrics kept for
  file continuity but documented as unable to detect queueing.
- `guard.py`: added `residency()`, which records co-resident models across
  both Ollama instances instead of only blocking.
- `experiments.py`: timeout 180→600; per-model warm-up; `co_resident`,
  `cold_load_s`, `timeouts` recorded per row; loud `⚠️ SERIALIZED` warning;
  new `--verify-concurrency` / `--require-concurrency` / `--url` flags.
- Benchmark target moved to **:11435**, a second user-level `ollama serve`
  with `OLLAMA_NUM_PARALLEL=64` (no sudo; user is in the `ollama` group).
  `NOPRUNE=1` so it never touches the shared model blobs.

### Negative control (validates the new metric)
`--verify-concurrency --url http://<gpu-host>:11434` on the known-serialized
server correctly reports concurrency 1.00 / 0.99 / 1.01 / 1.01 at N=1/2/4/8
and exits 1. Cold load for qwen3.5:4b = 4.53s.

## 2026-11-07 (evening) — THE RESULT: every "break point" was our own harness

### 19:01–20:42 — Full clean sweep (105 runs, zero co-residency flags)
All four experiments ran on the parallel instance `:11435` with real
concurrency, the correctness scorer, 600s timeouts and per-row residency
recording. GPU was exclusively ours the entire time.

| Exp | Runs | Verdict |
|---|---|---|
| E1 scaling ceiling | 15 | **Valid.** Flat 1.00. No break point. |
| E2 coordination tax | 30 | **Confounded.** Apparent collapse — see controls. |
| E3 drift | 45 | **Broken.** 9B scored exactly 0.000 in all 15 rounds. |
| E4 sycophancy | 15 | **Valid.** `sycophancy_rate = 0.00` everywhere. |

### E1 — no swarm-size break point
4B and 9B: `correct = 1.00` at N=5,15,30,50,60. 14B: 0.80 at N=5 and 0.98 at
N=50, 1.00 elsewhere (noise, not a trend). In every row `correct == format`
and `plausible_but_wrong == 0.00`: on an easy task these models never produce
a well-formed wrong answer — the only failure mode is not answering in shape.

Effective concurrency at N=60: 4B **35.2**, 9B **36.5**, 14B **9.5**
(136.6s vs 53.9s). Model size caps throughput; swarm size does not.

### E4 — zero sycophancy (a real null result)
Not one agent of 480 copied the injected wrong leader answer (`9999`), at any
model size or swarm size. Trustworthy only because the rewritten scorer
separates *copied the lure* from *independently wrong* from *correct*; the old
logic counted "didn't say 9999" as correct and could not have detected this.

### 20:45–21:00 — Controls: the whole E2 effect was truncation + a confound
`swarm/controls.py`, qwen3.5:4b, N=30:

| Control | correct | format | truncated | off-by-one |
|---|---|---|---|---|
| A original prompt @512 tok | 0.10 | 0.10 | **0.90** | 0 |
| A original prompt @2048 tok | **0.93** | 0.93 | 0.07 | 0 |
| B unambiguous prompt @2048 | 0.93 | 0.93 | 0.07 | 0 |

Control C — per-agent difficulty held CONSTANT (index = `agent_id % 10`),
2048 tokens, qwen3.5:4b:

| N | 5 | 15 | 30 | 50 | 60 |
|---|---|---|---|---|---|
| correct | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**Swarm size has no effect on fidelity.** E2's 0.40 → 0.08 "coordination
collapse" was per-agent difficulty rising with N (agent 59 must compute the
59th prime = 281; agent 4 the 4th = 11) on top of a token budget ~4x too small
for the required reasoning.

### E3 root cause — identical, proven directly
Probe of `role_adherence` on qwen3.5:9b:
- @512 tok: all agents `done_reason='length'`, ~1800 thinking chars, **`response=''`**
- @2048 tok: all agents `done_reason='stop'`, every role correct for `agent_id % 3`

The repaired prompt was fine. The "exactly 0.000 drift" was pure truncation.

### ⚠️ Correction to gotcha #1
The handoff says bumping `max_tokens` 64 → 512 *fixed* the thinking-model
problem. **It did not.** 512 is still far too small: these models spend
1800–3250 characters in the `thinking` field before emitting an answer.
**Use `max_tokens >= 2048` for any task requiring reasoning.** Always record
`done_reason`; `"length"` means the model never got to answer, which is not
the same as getting it wrong.

### The four artifacts (each independently produces a false break-point curve)
1. **Serialization** — `-np 1` made every swarm a FIFO queue; both parallelism
   metrics report "perfectly parallel" for a serial queue.
2. **Format-only scoring** — `drift_score` accepted any well-shaped line, so a
   wrong answer scored 1.0 and E4 could not detect sycophancy at all.
3. **Token truncation** — at 512 tokens, 90% of agents returned empty strings
   that scored as model failure.
4. **Difficulty confounded with swarm size** — raising N raised per-agent task
   difficulty, so "fidelity vs swarm size" was really "fidelity vs difficulty".

### Headline for the submission
> Across 4B–14B local models on a DGX Spark, swarm size up to 60 concurrent
> agents produced **no** measurable fidelity loss and **no** sycophancy. Every
> break point we initially measured was an artifact of the benchmark harness.
> We characterise four such artifacts, each with a negative control, and show
> that the two obvious parallelism metrics cannot detect the first one.
