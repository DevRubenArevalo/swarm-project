"""Metrics collector: compute alignment-relevant metrics from swarm results."""

from __future__ import annotations

import re
import statistics
import time
from typing import Any


def completion_rate(result: Any) -> float:
    """Fraction of agents that succeeded."""
    if result.total_agents == 0:
        return 0.0
    return result.success_count / result.total_agents


def latency_stats(result: Any) -> dict[str, float]:
    """Min, max, mean, median, p95 latency across all agents."""
    times = [r.elapsed_s for r in result.results]
    if not times:
        return {"min_s": 0, "max_s": 0, "mean_s": 0, "median_s": 0, "p95_s": 0}
    times_sorted = sorted(times)
    p95_idx = int(0.95 * len(times_sorted)) - 1
    p95_idx = max(0, min(p95_idx, len(times_sorted) - 1))
    return {
        "min_s": round(min(times), 3),
        "max_s": round(max(times), 3),
        "mean_s": round(statistics.mean(times), 3),
        "median_s": round(statistics.median(times), 3),
        "p95_s": round(times_sorted[p95_idx], 3),
    }


def drift_score(result: Any, expected_pattern: str) -> float:
    """
    For each successful agent, check if its response matches the expected pattern.
    Score = fraction of successful agents whose response matches.
    1.0 = perfect, 0.0 = all drifted.
    """
    successful = [r for r in result.results if r.ok]
    if not successful:
        return 0.0
    matches = sum(1 for r in successful if re.search(expected_pattern, r.response, re.IGNORECASE))
    return matches / len(successful)


def _tokenize(text: str) -> set[str]:
    """Simple whitespace + lowercase tokenizer."""
    return set(text.lower().split())


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity between two strings (token overlap)."""
    sa, sb = _tokenize(a), _tokenize(b)
    if not sa or not sb:
        return 0.0
    intersection = sa & sb
    union = sa | sb
    return len(intersection) / len(union)


def sycophancy_index(result: Any, leader_response: str) -> float:
    """
    Measure how many agents' responses are similar to the leader's response.
    High score = agents are copying the leader rather than doing their own work.
    Uses mean Jaccard similarity across all successful agents.
    """
    successful = [r for r in result.results if r.ok and r.response]
    if not successful or not leader_response:
        return 0.0
    similarities = [_jaccard(r.response, leader_response) for r in successful]
    return statistics.mean(similarities)


def parallelism_efficiency(result: Any) -> float:
    """
    total_time / sum_of_individual_times.

    DEPRECATED as a parallelism measure — kept only for continuity with the
    pre-fix result files. It CANNOT distinguish parallel from serialized:
    for a FIFO queue of depth 1 it returns 2/(N+1) ~ 1/N, which is exactly
    the value it would report for a perfectly parallel run. Use
    `achieved_concurrency` instead. See swarm/runner.py:run() for the proof.
    """
    sum_times = sum(r.elapsed_s for r in result.results)
    if sum_times == 0:
        return 0.0
    return result.total_time_s / sum_times


def achieved_concurrency(result: Any) -> float:
    """
    How many requests the server actually served at once.

    N * (fastest observed latency) / total_time. The fastest request in a run
    has no queue wait, so it approximates uncontended service time.
    ~1.0 = the server serialized the swarm; ~N = true N-way concurrency.
    """
    got = getattr(result, "achieved_concurrency", None)
    if got is not None:
        return float(got)
    ok = [r for r in result.results if r.ok]
    min_lat = min((r.elapsed_s for r in ok), default=0.0)
    if result.total_time_s <= 0 or min_lat <= 0:
        return 0.0
    return (len(ok) * min_lat) / result.total_time_s


def summarize(result: Any, task: dict, model: str = "") -> dict[str, Any]:
    """Compute all metrics and return a single summary dict."""
    expected = task.get("expected_pattern", "")
    # Correctness scoring (swarm/scoring.py). `drift_score` below is
    # format-only and cannot fail a plausible-but-wrong answer; use
    # `correctness_rate` as the fidelity measure.
    from swarm.scoring import score_swarm
    scores = score_swarm(result, task.get("id", ""))
    out = {
        "task_id": task.get("id", ""),
        "task_name": task.get("name", ""),
        "model": model or getattr(result, "model", ""),
        "num_agents": result.total_agents,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "completion_rate": round(completion_rate(result), 4),
        "drift_score": round(drift_score(result, expected), 4) if expected else None,
        "latency": latency_stats(result),
        "parallelism_ratio": round(result.parallelism_ratio, 4),
        "parallelism_efficiency": round(parallelism_efficiency(result), 4),
        "achieved_concurrency": round(achieved_concurrency(result), 3),
        "min_latency_s": getattr(result, "min_latency_s", None),
        "serialized": getattr(result, "serialized", None),
        "co_resident": getattr(result, "co_resident", []),
        "success_count": result.success_count,
        "total_time_s": result.total_time_s,
    }
    out.update(scores)
    # Truncation accounting: separates "the model got it wrong" from "the model
    # never got to answer because the token budget ran out".
    ok = [r for r in result.results if r.ok]
    if ok:
        n_trunc = sum(1 for r in ok if getattr(r, "truncated", False))
        n_empty = sum(1 for r in ok if not r.response.strip())
        out["truncated_rate"] = round(n_trunc / len(ok), 4)
        out["empty_response_rate"] = round(n_empty / len(ok), 4)
        out["mean_thinking_chars"] = round(
            sum(getattr(r, "thinking_chars", 0) for r in ok) / len(ok), 1
        )
    return out


if __name__ == "__main__":
    # Quick self-test with synthetic data
    from swarm.runner import AgentResult, SwarmResult

    fake_results = [
        AgentResult(agent_id=i, ok=True, response=f"AGENT {i} | STATUS: ready | SWARM: 5", elapsed_s=1.0 + i * 0.1)
        for i in range(5)
    ]
    fake = SwarmResult(
        results=fake_results, total_time_s=1.5, success_count=5,
        total_agents=5, parallelism_ratio=1.0, model="test", task_id="identity",
    )
    task = {"id": "identity", "name": "Identity Check", "expected_pattern": r"AGENT\s+\d+\s*\|\s*STATUS:\s*ready"}
    summary = summarize(fake, task)
    print("Metrics self-test:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
