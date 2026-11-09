"""The report's headline numbers, asserted against the committed raw data.

`python3 -m swarm.verify` checks all 42 reported figures and is what CI runs
end to end. This file covers the load-bearing ones so that a plain `pytest`
run also fails if the data stops supporting the paper, and so the statistical
helper has unit tests of its own.
"""

from __future__ import annotations

import glob
import json
import os

import pytest

from swarm.verify import wilson

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


@pytest.fixture(scope="module")
def controls() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(OUT, "controls_*.json"))):
        with open(path, encoding="utf-8") as fh:
            rows.extend(json.load(fh))
    assert rows, "no control data found in output/"
    return rows


def control_a(rows: list[dict], model: str, tokens: int) -> tuple[int, int]:
    """Agents correct / agents run, for one Control A cell."""
    cell = [r for r in rows if r["control"].startswith("A")
            and r["model"] == model and r["max_tokens"] == tokens]
    assert cell, f"no Control A row for {model} @{tokens}"
    n = sum(r["num_agents"] for r in cell)
    return sum(round(r["num_agents"] * r["correctness_rate"]) for r in cell), n


class TestWilsonInterval:
    def test_contains_the_point_estimate(self) -> None:
        lo, hi = wilson(28, 30)
        assert lo < 28 / 30 < hi

    def test_stays_inside_zero_one(self) -> None:
        # The reason we use Wilson and not Wald: at p near 0 or 1 with n=30,
        # the normal approximation runs past the boundary.
        for k in (0, 1, 29, 30):
            lo, hi = wilson(k, 30)
            assert 0.0 <= lo <= hi <= 1.0

    def test_narrows_as_n_grows(self) -> None:
        small = wilson(5, 10)
        large = wilson(500, 1000)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_handles_zero_samples(self) -> None:
        assert wilson(0, 0) == (0.0, 0.0)

    def test_matches_published_interval(self) -> None:
        lo, hi = wilson(3, 30)
        assert (round(lo, 2), round(hi, 2)) == (0.03, 0.26)


class TestControlA:
    """The single-variable control the paper leads on."""

    @pytest.mark.parametrize(
        "model, tokens, expected",
        [("qwen3.5:4b", 512, 0.10), ("qwen3.5:4b", 2048, 0.93),
         ("qwen3.5:9b", 512, 0.03), ("qwen3.5:9b", 2048, 0.87)],
    )
    def test_reported_correctness(self, controls, model, tokens, expected) -> None:
        k, n = control_a(controls, model, tokens)
        assert k / n == pytest.approx(expected, abs=0.005)

    def test_swarm_size_is_held_fixed(self, controls) -> None:
        # If this is ever false the control is no longer single-variable and
        # the paper's central claim does not follow.
        sizes = {r["num_agents"] for r in controls if r["control"].startswith("A")}
        assert sizes == {30}

    def test_the_effect_is_not_a_sampling_artifact(self, controls) -> None:
        low = wilson(*control_a(controls, "qwen3.5:4b", 512))
        high = wilson(*control_a(controls, "qwen3.5:4b", 2048))
        assert low[1] < high[0], "intervals must be disjoint"


class TestCorrectedResults:
    def test_sycophancy_null(self) -> None:
        rows: list[dict] = []
        for path in sorted(glob.glob(os.path.join(OUT, "e4_*.json"))):
            with open(path, encoding="utf-8") as fh:
                rows.extend(json.load(fh))
        assert sum(r["total_ok"] for r in rows) == 480
        assert sum(r["copied_wrong_answer"] for r in rows) == 0

    @pytest.mark.parametrize(
        "model, floor", [("qwen3.5:4b", 1.0), ("qwen3.5:9b", 1.0),
                         ("deepseek-r1:14b", 0.985)],
    )
    def test_corrected_correctness_floor(self, controls, model, floor) -> None:
        repeats: list[dict] = []
        for path in sorted(glob.glob(os.path.join(OUT, "replicate_14b_*.json"))):
            with open(path, encoding="utf-8") as fh:
                repeats.extend(json.load(fh))
        corrected = [r for r in controls if r["control"].startswith("C")] + repeats
        for n in (5, 15, 30, 50, 60):
            cell = [r for r in corrected if r["model"] == model and r["num_agents"] == n]
            if not cell:
                continue
            # The report pools replicates per cell rather than averaging rates.
            total = sum(r["num_agents"] for r in cell)
            right = sum(round(r["num_agents"] * r["correctness_rate"]) for r in cell)
            assert right / total >= floor - 0.0005, f"{model} at N={n}"
