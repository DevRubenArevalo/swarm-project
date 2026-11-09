# Swarm Fidelity: Where Do Local Models Break?

**Hackathon:** AI Alignment Foundation — Swarm Alignment 2026
**Date:** November 7–8, 2026
**Testbed:** NVIDIA DGX Spark (Blackwell GB10, 128 GB unified memory)
**Inference:** Ollama 0.40.15, parallel-enabled instance (`OLLAMA_NUM_PARALLEL=64`) — see Reproduction below

## The finding

We went looking for the swarm size at which small local models stop behaving
like a frontier agent. We did not find one. **Our first harness reported one
anyway — confidently and falsely, and characterising that is this project's
contribution.**

The cleanest evidence is one control that changes a single variable. Same
model, same prompt, same task, same 30-agent swarm — only the per-agent token
budget moves:

| Model | correctness @ 512 tok | @ 2048 tok |
|---|---|---|
| qwen3.5 4B | **0.10** | **0.93** |
| qwen3.5 9B | 0.03 | 0.87 |

Nothing about the task changed. 83 points of apparent capability collapse were
a property of one configuration value in our own code.

That is one of four such artifacts. Each is plausible, invisible in normal
operation, and independently sufficient to produce a convincing break-point
curve — and each gets *worse* on harder tasks, not better (REPORT.md §5.1).

| # | Artifact | What it did | Negative control |
|---|---|---|---|
| 1 | **Silent serialization** | The server ran one request at a time, so a "swarm of 60" was a queue of 60. Both standard parallelism metrics report *perfect parallelism* for a fully serial queue. | `--verify-concurrency` against the serialized server: reports 1.00, exits 1 |
| 2 | **Format-only scoring** | Validated response shape, not content. A wrong answer scored 1.0; the sycophancy test could not detect sycophancy. | `python3 -m swarm.scoring` asserts the old false-passes now fail |
| 3 | **Token truncation** | At 512 tokens, 83–97% of agents exhausted the budget inside a hidden `thinking` field and returned `""`, scored as model failure. | Control A: 512 vs 2048 tokens, same task |
| 4 | **Difficulty confounded with swarm size** | Agent *k* got a subtask of difficulty ∝ *k*, so a bigger swarm was also a harder task. | Control C: difficulty held constant while N varies |

Each artifact independently produces a convincing but false break-point curve.

## Results

| Measure | Result |
|---|---|
| Correctness vs swarm size (difficulty constant) | 4B **1.00**, 9B **1.00**, 14B **0.985–1.00** at N = 5→60 |
| 14B residual errors | all **6** are one item (prime index 8, answered 19 not 23); **0** at any other index |
| Sycophancy | **0 / 480** agents copied the injected wrong answer |
| Effective concurrency at N=60 | 4B **35.2**, 9B **36.5**, 14B **9.5** |
| Truncation at 512 vs 2048 tokens | 4B 0.90→0.07, 9B 0.97→0.13, 14B 0.83→0.17 |

Model size caps throughput; swarm size does not. A 4B model sustained 35
effectively concurrent agents at full accuracy on short-output tasks.

**Read the null narrowly.** Our agents never read each other's output, so no
mechanism exists by which swarm size *could* degrade per-agent correctness
beyond resource contention, which we controlled for. "No break point" means
this design could not have produced one — not that local-model swarms are
robust. The coordination experiment this harness makes possible is set out in
REPORT.md §5.3.

## Repository

```
docs/REPORT.md        research report (the submission)
docs/progress.md      full investigation log, including the retractions
docs/VIDEO_SCRIPT.md  3-minute video script + shot list

swarm/runner.py       concurrent runner; achieved_concurrency, done_reason, timings
swarm/scoring.py      content-level scoring + self-test
swarm/controls.py     the three artifact controls
swarm/guard.py        instance-labelled GPU residency recording
swarm/slots.py        live llama.cpp slot inspector
swarm/experiments.py  E1-E4 orchestration
swarm/figures.py      report figures
swarm/tasks.py        the 4-task suite

output/               results (JSON) and the 3 report figures (PNG)
output/retracted/     pre-correction figures, kept as evidence only

video/                3-minute presentation video (180.00s) + captions
video/build.py        slide renderer and ffmpeg assembly
video/demo/           revised cut: layout, scenes, neural narration
```

## Running it

Requires Python 3.11+, `ffmpeg`, and an Ollama server with the models pulled.

```bash
pip install -r requirements.txt
export SWARM_OLLAMA_URL=http://localhost:11435   # the parallel instance
```

No host names or keys are baked into the code. Every endpoint reads from the
environment (`SWARM_OLLAMA_URL`, `SWARM_SHARED_OLLAMA_URL`, `SWARM_SSH_TARGET`)
and defaults to `localhost`.

## Reproduction

```bash
# 1. An inference server that actually runs agents in parallel.
#    The default (-np 1) silently serializes every swarm.
OLLAMA_HOST=0.0.0.0:11435 OLLAMA_NUM_PARALLEL=64 \
OLLAMA_CONTEXT_LENGTH=2048 OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_NOPRUNE=1 ollama serve &

# 2. PROVE it is parallel before trusting any number.
python3 -m swarm.experiments --verify-concurrency --url http://localhost:11435

# 3. Negative control — the same check against a serialized server
#    MUST report ~1.00 and exit 1.
python3 -m swarm.experiments --verify-concurrency --url http://localhost:11434

# 4. Experiments (refuse to run if the server serializes).
python3 -m swarm.experiments --experiment e1 \
  --models qwen3.5:4b,qwen3.5:9b,deepseek-r1:14b \
  --sizes 5,15,30,50,60 --task identity \
  --url http://localhost:11435 --require-concurrency

# 5. The artifact controls.
python3 -m swarm.controls --model qwen3.5:4b --n 30 --sizes 5,15,30,50,60

# 6. Figures.
python3 -m swarm.figures

# Live view of real inference slots (Ollama has no "agents" — it has slots).
python3 -m swarm.slots --watch
```

## If you are benchmarking a local-model swarm

1. Verify achieved concurrency against a serialized baseline. Do **not** trust
   `total/max_latency` or `total/Σ latency` — both report perfect parallelism
   for a serial queue.
2. Score content, not format. Assert that a deliberately wrong answer fails.
3. Record `done_reason`. Treat `"length"` as *no measurement*, not as failure.
   Budget ≥2048 tokens for reasoning models.
4. Hold per-agent difficulty constant when varying swarm size.
5. Record GPU residency per row, labelled by server instance.

## Status

E1 and E4 are valid as reported. **E2 and E3 are retracted** — both were
dominated by artifacts 3 and 4; the controls establish the cause. See
[`docs/progress.md`](docs/progress.md) for the full chain of evidence.
