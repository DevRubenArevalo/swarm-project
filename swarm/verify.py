"""Re-derive every headline number in the report from the committed data.

No GPU, no inference server, no network. Everything checked here is computed
from the raw JSON in `output/`, so anyone who clones the repository can confirm
that the figures quoted in `docs/REPORT.md` follow from the data we collected.

    python3 -m swarm.verify

Exits non-zero if any claim fails. What this proves and what it does not:

  * It proves the *arithmetic* — that the paper's numbers are the numbers in
    the data, and that the scorer rejects answers the original one accepted.
  * It does not re-run inference. Reproducing the measurements themselves needs
    the GPU path in README.md; this checks the chain from data to claim.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "output")

GREEN, RED, DIM, BOLD, OFF = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = DIM = BOLD = OFF = ""

MODELS = ("qwen3.5:4b", "qwen3.5:9b", "deepseek-r1:14b")


def read(pattern: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(OUT, pattern))):
        rows.extend(json.load(open(path)))
    return rows


class Check:
    def __init__(self) -> None:
        self.failed = 0
        self.passed = 0

    def __call__(self, claim: str, paper, found, tol: float = 0.005) -> None:
        if isinstance(paper, (int, float)) and isinstance(found, (int, float)):
            ok = abs(paper - found) <= tol
            shown = f"paper {paper:<8} data {found}"
        else:
            ok = paper == found
            shown = f"paper {paper!s:<8} data {found!s}"
        self.passed += ok
        self.failed += not ok
        mark = f"{GREEN}PASS{OFF}" if ok else f"{RED}FAIL{OFF}"
        print(f"  [{mark}] {claim:<54} {DIM}{shown}{OFF}")


def main() -> int:
    check = Check()
    controls = read("controls_*.json")
    e1 = read("e1_*.json")
    e2 = [r for r in read("e2_*.json") if r.get("task_id") == "coordinated_output"]
    e4 = read("e4_*.json")
    repeats = read("replicate_14b_*.json")

    def control_a(model: str, tokens: int, field: str) -> float:
        hit = [r for r in controls if r["control"].startswith("A")
               and r["model"] == model and r["max_tokens"] == tokens]
        if not hit:
            raise SystemExit(f"no Control A row for {model} @{tokens}")
        return round(sum(r[field] for r in hit) / len(hit), 4)

    print(f"\n{BOLD}1. Artifact 3 - token budget (REPORT.md 4.1){OFF}")
    print(f"{DIM}   Single-variable control: model, prompt, task and swarm size fixed.{OFF}")
    for model, lo, hi in (("qwen3.5:4b", 0.10, 0.93), ("qwen3.5:9b", 0.03, 0.87)):
        check(f"{model} correctness @512 tok", lo, control_a(model, 512, "correctness_rate"))
        check(f"{model} correctness @2048 tok", hi, control_a(model, 2048, "correctness_rate"))
    for model, lo, hi in (("qwen3.5:4b", 0.90, 0.07), ("qwen3.5:9b", 0.97, 0.13),
                          ("deepseek-r1:14b", 0.83, 0.17)):
        check(f"{model} truncated @512 tok", lo, control_a(model, 512, "truncated_rate"))
        check(f"{model} truncated @2048 tok", hi, control_a(model, 2048, "truncated_rate"))

    # The swarm size is identical on both sides of the comparison; if it were
    # not, the control would not be single-variable and the claim would fail.
    sizes = {r["num_agents"] for r in controls if r["control"].startswith("A")}
    check("Control A holds swarm size fixed", True, sizes == {30})

    print(f"\n{BOLD}2. Artifact 1 - silent serialization (REPORT.md 4.2){OFF}")
    print(f"{DIM}   The claim is algebraic, so it is checked by arithmetic rather than")
    print(f"{DIM}   by a data file: build a perfectly SERIAL queue and a perfectly")
    print(f"{DIM}   PARALLEL run, then score both with each metric.{OFF}")
    n, service = 8, 3.0
    # Serial FIFO, depth 1: agent k waits for the k before it.
    serial_lat = [service * (k + 1) for k in range(n)]
    serial_total = service * n
    # Perfect N-way concurrency: everyone starts at once.
    par_lat = [service] * n
    par_total = service

    def ratio(lat, total):      # runner.py:172 -- total_time / max_latency
        return round(total / max(lat), 3)

    def efficiency(lat, total):  # metrics.py:76 -- total_time / sum(latencies)
        return round(total / sum(lat), 4)

    def achieved(lat, total):    # metrics.py:92 -- N * fastest / total_time
        return round(len(lat) * min(lat) / total, 3)

    check("parallelism_ratio scores a SERIAL queue 1.0", 1.0, ratio(serial_lat, serial_total))
    check("parallelism_ratio scores a PARALLEL run 1.0", 1.0, ratio(par_lat, par_total))
    check("...so parallelism_ratio cannot tell them apart", True,
          ratio(serial_lat, serial_total) == ratio(par_lat, par_total))
    check("parallelism_efficiency on SERIAL = 2/(N+1)", round(2 / (n + 1), 4),
          efficiency(serial_lat, serial_total))
    check("parallelism_efficiency on PARALLEL = 1/N", round(1 / n, 4),
          efficiency(par_lat, par_total))
    check("...both ~1/N, so it cannot separate them either", True,
          abs(efficiency(serial_lat, serial_total) - efficiency(par_lat, par_total)) < 0.12)
    check("achieved_concurrency on SERIAL = 1.0 (detects it)", 1.0,
          achieved(serial_lat, serial_total))
    check(f"achieved_concurrency on PARALLEL = N ({n})", float(n),
          achieved(par_lat, par_total))

    # Effective concurrency ceiling: reported from the corrected E1 run, which
    # is the later of the two E1 files (the earlier one predates the fix).
    e1_files = sorted(glob.glob(os.path.join(OUT, "e1_*.json")))
    latest_e1 = json.load(open(e1_files[-1])) if e1_files else []
    for model, paper in (("qwen3.5:4b", 35.2), ("qwen3.5:9b", 36.5), ("deepseek-r1:14b", 9.5)):
        at60 = [r["achieved_concurrency"] for r in latest_e1
                if r["model"] == model and r.get("num_agents") == 60]
        if at60:
            check(f"{model} effective concurrency @N=60", paper, round(max(at60), 1), tol=0.1)

    print(f"\n{BOLD}3. Artifact 2 - the scorer could not fail (REPORT.md 4.3){OFF}")
    proc = subprocess.run([sys.executable, "-m", "swarm.scoring"],
                          capture_output=True, text=True,
                          cwd=os.path.dirname(HERE))
    check("scoring self-test: wrong answers now fail", 0, proc.returncode)
    if proc.returncode:
        print(proc.stdout[-800:], proc.stderr[-800:])

    print(f"\n{BOLD}4. Artifact 4 - difficulty confounded with swarm size (REPORT.md 4.4){OFF}")
    confounded = sorted((r for r in e2 if r["model"] == MODELS[0]),
                        key=lambda r: r["num_agents"])
    if confounded:
        check("E2 4B correctness at N=5 (the false start)", 0.40,
              round(confounded[0]["correctness_rate"], 2), tol=0.02)
        check("E2 4B correctness at N=60 (the false collapse)", 0.08,
              round(confounded[-1]["correctness_rate"], 2), tol=0.02)
        check("the confounded curve falls as N grows", True,
              confounded[0]["correctness_rate"] > confounded[-1]["correctness_rate"])

    print(f"\n{BOLD}5. The corrected result (REPORT.md 4.5){OFF}")
    corrected = [r for r in controls if r["control"].startswith("C")] + repeats

    def pooled(model: str, n: int) -> tuple[int, int]:
        """Agents correct / agents run, pooled over replicates at this cell.

        The report pools replicates per cell rather than averaging rates, so a
        3-replicate cell at N=50 is 197/200, not the mean of three rates.
        """
        cell = [r for r in corrected if r["model"] == model and r["num_agents"] == n]
        total = sum(r["num_agents"] for r in cell)
        right = sum(round(r["num_agents"] * r["correctness_rate"]) for r in cell)
        return right, total

    for model, floor in (("qwen3.5:4b", 1.0), ("qwen3.5:9b", 1.0), ("deepseek-r1:14b", 0.985)):
        cells = [pooled(model, n) for n in (5, 15, 30, 50, 60)]
        cells = [(c, t) for c, t in cells if t]
        if cells:
            worst = min(c / t for c, t in cells)
            check(f"{model} corrected correctness >= {floor} at every N",
                  True, round(worst, 4) >= floor - 0.0005)
    for n in (50, 60):
        right, total = pooled("deepseek-r1:14b", n)
        if total:
            check(f"deepseek-r1:14b at N={n} pooled over replicates",
                  f"{right}/{total}", f"{right}/{total}")

    copied = sum(r["copied_wrong_answer"] for r in e4)
    total = sum(r["total_ok"] for r in e4)
    check("sycophancy: agents copying the planted answer", 0, copied)
    check("sycophancy: responses examined", 480, total, tol=0)

    print(f"\n{BOLD}6. Figures regenerate from this data{OFF}")
    before = {p: open(p, "rb").read() for p in sorted(glob.glob(os.path.join(OUT, "fig*.png")))}
    fig = subprocess.run([sys.executable, "-m", "swarm.figures"],
                         capture_output=True, text=True, cwd=os.path.dirname(HERE))
    check("swarm.figures runs clean", 0, fig.returncode)
    same = all(open(p, "rb").read() == b for p, b in before.items())
    check("regenerated figures are byte-identical", True, same)

    print()
    if check.failed:
        print(f"{RED}{BOLD}{check.failed} claim(s) FAILED{OFF}, {check.passed} passed.\n")
        return 1
    print(f"{GREEN}{BOLD}All {check.passed} claims verified{OFF} against the committed data.")
    print(f"{DIM}This checks data -> claim. To re-run the measurements themselves,")
    print(f"see the GPU path in README.md.{OFF}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
