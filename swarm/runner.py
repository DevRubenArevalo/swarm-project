"""Swarm runner: spawn N concurrent agents against Ollama and collect results."""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

# The benchmark server. Override with SWARM_OLLAMA_URL.
# Default is :11435 — the parallel-enabled instance (OLLAMA_NUM_PARALLEL>1).
# The shared :11434 runs with -np 1 and SERIALIZES all requests to a model,
# which makes any "swarm" measurement meaningless (see docs/HANDOFF.md §5).
DEFAULT_OLLAMA_URL = os.environ.get("SWARM_OLLAMA_URL", "http://localhost:11435")


@dataclass
class AgentResult:
    agent_id: int
    ok: bool
    response: str
    elapsed_s: float
    error: str | None = None
    # Wall-clock offsets from the start of the run (seconds).
    started_s: float = 0.0
    ended_s: float = 0.0
    # Server-reported timings (nanoseconds, from Ollama's /api/generate).
    load_duration_ns: int = 0
    prompt_eval_duration_ns: int = 0
    eval_duration_ns: int = 0
    eval_count: int = 0
    # Ollama's stop reason. "length" means the token budget ran out before the
    # model finished — for a thinking model that usually means the whole budget
    # went to the `thinking` field and `response` came back empty. Without this
    # a truncation is indistinguishable from a model that cannot do the task.
    done_reason: str = ""
    thinking_chars: int = 0

    @property
    def truncated(self) -> bool:
        return self.done_reason == "length"

    @property
    def server_compute_s(self) -> float:
        """Time the server actually spent on this request (excl. queue wait)."""
        return (self.prompt_eval_duration_ns + self.eval_duration_ns) / 1e9


@dataclass
class SwarmResult:
    results: list[AgentResult] = field(default_factory=list)
    total_time_s: float = 0.0
    success_count: int = 0
    total_agents: int = 0
    parallelism_ratio: float = 0.0
    model: str = ""
    task_id: str = ""
    # Measured concurrency (see achieved_concurrency below).
    achieved_concurrency: float = 0.0
    min_latency_s: float = 0.0
    serialized: bool = False
    co_resident: list[str] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        if self.total_agents == 0:
            return 0.0
        return self.success_count / self.total_agents


class SwarmRunner:
    """Spawn N concurrent agents against an Ollama server."""

    def __init__(
        self,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model: str = "qwen3.5:4b",
        num_agents: int = 5,
        timeout: int = 120,
        temperature: float = 0.1,
        max_tokens: int = 2048,  # see docs/HANDOFF.md gotcha #1 — 512 truncates thinking models
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.num_agents = num_agents
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _call_agent(self, agent_id: int, prompt: str, t0: float | None = None) -> AgentResult:
        """Send a single prompt to Ollama and return the result."""
        if t0 is None:
            t0 = time.monotonic()
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }).encode()

        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
            end = time.monotonic()
            elapsed = end - start
            return AgentResult(
                agent_id=agent_id,
                ok=True,
                response=body.get("response", "").strip(),
                elapsed_s=round(elapsed, 3),
                started_s=round(start - t0, 3),
                ended_s=round(end - t0, 3),
                load_duration_ns=int(body.get("load_duration", 0) or 0),
                prompt_eval_duration_ns=int(body.get("prompt_eval_duration", 0) or 0),
                eval_duration_ns=int(body.get("eval_duration", 0) or 0),
                eval_count=int(body.get("eval_count", 0) or 0),
                done_reason=str(body.get("done_reason", "") or ""),
                thinking_chars=len(str(body.get("thinking", "") or "")),
            )
        except urllib.error.HTTPError as e:
            end = time.monotonic()
            elapsed = end - start
            return AgentResult(
                agent_id=agent_id, ok=False, response="",
                elapsed_s=round(elapsed, 3), error=f"HTTP {e.code}: {e.reason}",
                started_s=round(start - t0, 3), ended_s=round(end - t0, 3),
            )
        except Exception as e:
            end = time.monotonic()
            elapsed = end - start
            return AgentResult(
                agent_id=agent_id, ok=False, response="",
                elapsed_s=round(elapsed, 3), error=str(e),
                started_s=round(start - t0, 3), ended_s=round(end - t0, 3),
            )

    def run(self, task_prompt: str, task_id: str = "") -> SwarmResult:
        """Run the swarm: spawn num_agents concurrent requests."""
        t0 = time.monotonic()
        results: list[AgentResult] = []

        with ThreadPoolExecutor(max_workers=self.num_agents) as pool:
            futures = {}
            for i in range(self.num_agents):
                prompt = task_prompt.replace("{agent_id}", str(i)).replace("{swarm_size}", str(self.num_agents))
                futures[pool.submit(self._call_agent, i, prompt, t0)] = i
            for fut in as_completed(futures):
                results.append(fut.result())

        total_time = time.monotonic() - t0
        results.sort(key=lambda r: r.agent_id)
        success_count = sum(1 for r in results if r.ok)
        max_lat = max((r.elapsed_s for r in results), default=0.0)
        parallelism = total_time / max_lat if max_lat > 0 else 0.0

        # --- Measured concurrency ---------------------------------------
        # WARNING on the two legacy metrics:
        #   parallelism_ratio      = total_time / max_latency
        #   parallelism_efficiency = total_time / sum(latencies)
        # Both are ALGEBRAICALLY BLIND to server-side queueing. For a FIFO
        # queue of depth 1, agent k's latency is (k+1)*s, so:
        #   max_lat = N*s = total_time  -> ratio      == 1.00 ("perfect")
        #   sum     = s*N(N+1)/2        -> efficiency == 2/(N+1) ~ 1/N ("perfect")
        # i.e. a fully serialized run is indistinguishable from a perfectly
        # parallel one. Measured on :11434 (-np 1): N=1/2/4/8 -> 3.64/7.19/
        # 14.09/27.37s total, while per-request service time stayed ~3.6s.
        #
        # The honest measure compares total wall-clock against the work done.
        # The fastest request in a run has no queue wait, so its latency
        # approximates the uncontended service time s:
        #     achieved_concurrency = N * s / total_time
        # Serial queue -> N*s/(N*s) = 1.0.  True N-way parallel -> N.
        ok_results = [r for r in results if r.ok]
        min_lat = min((r.elapsed_s for r in ok_results), default=0.0)
        if total_time > 0 and min_lat > 0:
            achieved = (len(ok_results) * min_lat) / total_time
        else:
            achieved = 0.0
        # Flag runs the server served one-at-a-time despite N>1 agents.
        serialized = self.num_agents > 1 and achieved < 1.5

        return SwarmResult(
            results=results,
            total_time_s=round(total_time, 3),
            success_count=success_count,
            total_agents=self.num_agents,
            parallelism_ratio=round(parallelism, 3),
            model=self.model,
            task_id=task_id,
            achieved_concurrency=round(achieved, 2),
            min_latency_s=round(min_lat, 3),
            serialized=serialized,
        )

    def warmup(self) -> float:
        """
        Load the model and pay the cold-start cost BEFORE measuring.

        Returns the server-reported load_duration in seconds. Keeping this
        separate means cold-start latency is recorded as its own number
        instead of silently inflating the first swarm size in every curve.
        """
        r = self._call_agent(-1, "Reply with the single word: ready")
        return round(r.load_duration_ns / 1e9, 3)


def _self_test() -> None:
    """Quick 3-agent self-test."""
    print("SwarmRunner self-test (3 agents, qwen3.5:4b)")
    runner = SwarmRunner(num_agents=3, model="qwen3.5:4b")
    prompt = "You are agent {agent_id} in a swarm of {swarm_size}. Reply with: AGENT {agent_id} | STATUS: ready"
    result = runner.run(prompt, task_id="self_test")
    print(f"  Success: {result.success_count}/{result.total_agents}")
    print(f"  Total time: {result.total_time_s:.2f}s")
    print(f"  Parallelism: {result.parallelism_ratio:.2f}")
    for r in result.results:
        status = "✓" if r.ok else "✗"
        print(f"  {status} Agent {r.agent_id}: {r.response[:60] or r.error}")


if __name__ == "__main__":
    _self_test()
