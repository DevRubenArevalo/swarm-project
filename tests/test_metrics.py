"""The standard parallelism metrics cannot see a serial queue.

Artifact 1 in the report, and the one claim that holds independently of the
testbed: for a FIFO queue of depth 1, agent k's latency is (k+1)·s, so

    parallelism_ratio      = wall / max_latency = N·s / N·s        = 1.00
    parallelism_efficiency = wall / Σ latency   = 2/(N+1)          ≈ 1/N

Both are the textbook values for *perfect* parallelism. These tests build a
perfectly serial run and a perfectly parallel one and assert the two metrics
cannot tell them apart, while achieved_concurrency can.
"""

from __future__ import annotations

import pytest

from swarm.metrics import achieved_concurrency, completion_rate, parallelism_efficiency
from swarm.runner import AgentResult, SwarmResult

SERVICE_S = 3.0


def _swarm(latencies: list[float], total_time_s: float) -> SwarmResult:
    results = [
        AgentResult(agent_id=i, ok=True, response=f"AGENT {i} | ITEM: 2", elapsed_s=lat)
        for i, lat in enumerate(latencies)
    ]
    return SwarmResult(
        results=results,
        total_time_s=total_time_s,
        success_count=len(results),
        total_agents=len(results),
        parallelism_ratio=total_time_s / max(latencies),
        model="test",
        task_id="coordinated_output",
    )


def serial_queue(n: int, service_s: float = SERVICE_S) -> SwarmResult:
    """Depth-1 FIFO: agent k waits for the k before it."""
    return _swarm([service_s * (k + 1) for k in range(n)], service_s * n)


def true_parallel(n: int, service_s: float = SERVICE_S) -> SwarmResult:
    """Everyone starts at once and finishes together."""
    return _swarm([service_s] * n, service_s)


@pytest.mark.parametrize("n", [2, 4, 8, 30, 60])
def test_parallelism_ratio_is_1_for_both(n: int) -> None:
    assert serial_queue(n).parallelism_ratio == pytest.approx(1.0)
    assert true_parallel(n).parallelism_ratio == pytest.approx(1.0)


@pytest.mark.parametrize("n", [2, 4, 8, 30, 60])
def test_parallelism_ratio_cannot_separate_them(n: int) -> None:
    assert serial_queue(n).parallelism_ratio == pytest.approx(
        true_parallel(n).parallelism_ratio
    ), "if this ever fails, the metric gained the power the report says it lacks"


@pytest.mark.parametrize("n", [2, 4, 8, 30, 60])
def test_efficiency_on_serial_is_2_over_n_plus_1(n: int) -> None:
    assert parallelism_efficiency(serial_queue(n)) == pytest.approx(2 / (n + 1))


@pytest.mark.parametrize("n", [2, 4, 8, 30, 60])
def test_efficiency_on_parallel_is_1_over_n(n: int) -> None:
    assert parallelism_efficiency(true_parallel(n)) == pytest.approx(1 / n)


@pytest.mark.parametrize("n", [8, 30, 60])
def test_efficiency_values_are_close_enough_to_be_useless(n: int) -> None:
    # 2/(N+1) against 1/N: the gap shrinks as N grows, and at the swarm sizes
    # we actually ran it is far inside normal run-to-run noise.
    serial = parallelism_efficiency(serial_queue(n))
    parallel = parallelism_efficiency(true_parallel(n))
    assert abs(serial - parallel) < 1.1 / n


class TestAchievedConcurrency:
    """N × fastest / wall-clock — the figure a queue cannot fake."""

    @pytest.mark.parametrize("n", [2, 4, 8, 30, 60])
    def test_reports_1_for_a_serial_queue(self, n: int) -> None:
        got = serial_queue(n)
        got.achieved_concurrency = 0.0  # force recomputation from latencies
        assert achieved_concurrency(got) == pytest.approx(1.0)

    @pytest.mark.parametrize("n", [2, 4, 8, 30, 60])
    def test_reports_n_for_true_parallelism(self, n: int) -> None:
        got = true_parallel(n)
        got.achieved_concurrency = 0.0
        assert achieved_concurrency(got) == pytest.approx(float(n))

    def test_separates_what_the_others_cannot(self) -> None:
        serial, parallel = serial_queue(8), true_parallel(8)
        serial.achieved_concurrency = parallel.achieved_concurrency = 0.0
        assert achieved_concurrency(serial) < 1.1
        assert achieved_concurrency(parallel) > 7.9

    def test_zero_when_nothing_succeeded(self) -> None:
        empty = SwarmResult(results=[], total_time_s=1.0, total_agents=0)
        assert achieved_concurrency(empty) == 0.0


def test_completion_rate_handles_empty_swarm() -> None:
    assert completion_rate(SwarmResult(results=[], total_agents=0)) == 0.0
