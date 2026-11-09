"""
Report figures for the Swarm Fidelity submission.

Separate from charts.py on purpose: charts.py plots `drift_score`, the
format-only metric we proved meaningless (a well-shaped wrong answer scored
1.0). These figures plot the metrics that survived scrutiny.

Palette: the validated reference instance. Categorical slots are assigned in
fixed order and never cycled; every series is direct-labelled because slot 3
(aqua) measures 2.74:1 against the light surface, below the 3:1 bar, so the
relief rule applies.

    python3 -m swarm.figures
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUTPUT_DIR = "output"

# --- validated palette (light mode) ----------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1 = "#2a78d6"   # blue   - slot 1
S2 = "#eb6834"   # orange - slot 2
S3 = "#1baf7a"   # aqua   - slot 3

MODELS = ["qwen3.5:4b", "qwen3.5:9b", "deepseek-r1:14b"]
MODEL_LABEL = {"qwen3.5:4b": "qwen3.5 4B",
               "qwen3.5:9b": "qwen3.5 9B",
               "deepseek-r1:14b": "deepseek-r1 14B"}


def _style(ax, ylabel: str = "", xlabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=10)


def _pct(v, _pos=None) -> str:
    return f"{v*100:.0f}%"


# --- data loading -----------------------------------------------------------


def load_latest(pattern: str) -> list[dict]:
    fs = sorted(glob.glob(os.path.join(OUTPUT_DIR, pattern)))
    if not fs:
        return []
    d = json.load(open(fs[-1]))
    return d if isinstance(d, list) else d.get("data", [])


def load_all_controls() -> list[dict]:
    """
    Every controls_*.json merged, plus the 14B replication runs.

    Control C cells that were run more than once are POOLED by agent-response
    rather than averaged by run, so a cell's weight reflects how many agents
    it actually measured. The 14B's N=50 and N=60 were replicated 3x each
    after the single-run figures (0.96, 0.98) proved to rest on 1-2 wrong
    answers; pooled over 4 runs they are 0.985 and 0.9875.
    """
    rows = []
    for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "controls_*.json"))):
        try:
            rows.extend(json.load(open(f)))
        except (json.JSONDecodeError, OSError):
            continue
    for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "replicate_14b_*.json"))):
        try:
            rows.extend(json.load(open(f)))
        except (json.JSONDecodeError, OSError):
            continue

    # Pool repeated Control C cells.
    pooled: dict[tuple, dict] = {}
    out = []
    for r in rows:
        if not r.get("control", "").startswith("C"):
            out.append(r)
            continue
        key = (r["model"], r["num_agents"])
        if key not in pooled:
            p = dict(r)
            p["control"] = "C constant difficulty"
            p["_runs"] = 1
            p["_agents"] = r["num_agents"]
            p["_correct"] = round(r["correctness_rate"] * r["num_agents"])
            pooled[key] = p
            out.append(p)
        else:
            p = pooled[key]
            p["_runs"] += 1
            p["_agents"] += r["num_agents"]
            p["_correct"] += round(r["correctness_rate"] * r["num_agents"])
            p["off_by_one_count"] += r.get("off_by_one_count", 0)
        p = pooled[key]
        p["correctness_rate"] = round(p["_correct"] / p["_agents"], 4)
    return out


def _ctrl(rows, model, prefix, tok=None):
    out = [r for r in rows
           if r.get("model") == model and r.get("control", "").startswith(prefix)]
    if tok is not None:
        out = [r for r in out if r.get("max_tokens") == tok]
    return sorted(out, key=lambda r: r.get("num_agents", 0))


# --- Figure 1: the headline -------------------------------------------------


def fig_fidelity(controls, e2, path):
    """
    Small multiples, one panel per model: the curve as first measured vs the
    curve with the two artifacts removed.

    Two series per panel, so slots 1 and 2. Both direct-labelled.
    """
    models = [m for m in MODELS if _ctrl(controls, m, "C")]
    if not models:
        return None
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.3),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)

    for ax, model in zip(axes, models):
        c = _ctrl(controls, model, "C")
        cx = [r["num_agents"] for r in c]
        cy = [r["correctness_rate"] for r in c]

        bad = sorted([r for r in e2
                      if r.get("model") == model
                      and r.get("task_id") == "coordinated_output"],
                     key=lambda r: r.get("num_agents", 0))
        bx = [r["num_agents"] for r in bad]
        by = [r.get("correctness_rate", 0) for r in bad]

        if bx:
            ax.plot(bx, by, color=S2, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
                    label="as first measured")
        ax.plot(cx, cy, color=S1, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4,
                label="artifacts removed")

        _style(ax, xlabel="swarm size (concurrent agents)")
        ax.set_ylim(-0.05, 1.12)
        ax.set_xticks(cx)
        ax.set_title(MODEL_LABEL.get(model, model), color=INK,
                     fontsize=11, fontweight="bold", pad=10)

        # Direct labels (relief rule + selective labelling).
        if cy:
            ax.annotate(f"{_pct(cy[-1])}", (cx[-1], cy[-1]),
                        textcoords="offset points", xytext=(-6, 10),
                        color=INK_2, fontsize=9, ha="right", fontweight="bold")
        if by:
            # Above-left of the final point: below it collides with the x-axis
            # tick label at the right edge of the panel.
            ax.annotate(f"{_pct(by[-1])}", (bx[-1], by[-1]),
                        textcoords="offset points", xytext=(-8, 11),
                        color=INK_2, fontsize=9, ha="right", fontweight="bold")

    axes[0].yaxis.set_major_formatter(FuncFormatter(_pct))
    axes[0].set_ylabel("task correctness", color=INK_2, fontsize=10)

    fig.suptitle("Swarm size does not degrade local-model fidelity",
                 color=INK, fontsize=14, fontweight="bold", y=0.99, x=0.02,
                 ha="left")
    fig.text(0.02, 0.90,
             "The decline on the left is two artifacts of our own harness: a 512-token budget that "
             "truncated\nreasoning, and a task whose per-agent difficulty rose with swarm size. "
             # "flat" would overclaim: the 14B dips to 96-98% at N=50/60.
             "Remove both and the collapse disappears.",
             color=INK_2, fontsize=9.5, ha="left", va="top")

    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.02, 0.0),
                     ncol=2, frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.86))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


# --- Figure 2: the mechanism ------------------------------------------------


def fig_truncation(controls, path):
    """Grouped bars: correctness and truncation at 512 vs 2048 tokens."""
    models = [m for m in MODELS if _ctrl(controls, m, "A", 512)]
    if not models:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    fig.patch.set_facecolor(SURFACE)

    idx = range(len(models))
    w = 0.36
    gap = 0.02  # 2px-equivalent surface gap between adjacent fills

    def grouped(ax, key, title, ylabel):
        lo, hi = [], []
        for m in models:
            a = _ctrl(controls, m, "A", 512)
            b = _ctrl(controls, m, "A", 2048)
            lo.append(a[0].get(key, 0) if a else 0)
            hi.append(b[0].get(key, 0) if b else 0)
        b1 = ax.bar([i - w/2 - gap/2 for i in idx], lo, w, color=S2,
                    zorder=3, label="512 tokens")
        b2 = ax.bar([i + w/2 + gap/2 for i in idx], hi, w, color=S1,
                    zorder=3, label="2048 tokens")
        for bars in (b1, b2):
            for bar in bars:
                ax.annotate(_pct(bar.get_height()),
                            (bar.get_x() + bar.get_width()/2, bar.get_height()),
                            textcoords="offset points", xytext=(0, 4),
                            ha="center", color=INK_2, fontsize=9,
                            fontweight="bold")
        _style(ax, ylabel=ylabel)
        ax.set_xticks(list(idx))
        ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(FuncFormatter(_pct))
        ax.set_title(title, color=INK, fontsize=11, fontweight="bold", pad=10)

    grouped(ax1, "correctness_rate", "Task correctness", "correctness")
    grouped(ax2, "truncated_rate", "Responses cut off mid-thought", "truncated")

    fig.suptitle("The break point was a token budget, not a capability limit",
                 color=INK, fontsize=14, fontweight="bold", y=0.99, x=0.02,
                 ha="left")
    fig.text(0.02, 0.90,
             "Identical task, identical swarm size (30 agents). These models emit 1800-3250 characters of "
             "hidden\nreasoning before answering; at 512 tokens most never reach an answer and score as failures.",
             color=INK_2, fontsize=9.5, ha="left", va="top")

    h, l = ax1.get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.02, 0.0),
                     ncol=2, frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.86))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


# --- Figure 3: the concurrency ceiling --------------------------------------


def fig_concurrency(e1, path):
    """Effective concurrency vs swarm size, one line per model."""
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    fig.patch.set_facecolor(SURFACE)
    colors = {MODELS[0]: S1, MODELS[1]: S2, MODELS[2]: S3}

    # The 4B and 9B finish within ~1.3 of each other, so their end labels
    # overlap at a shared offset; stagger them vertically.
    dy = {MODELS[0]: -7, MODELS[1]: 7, MODELS[2]: 0}
    drawn = False
    for model in MODELS:
        rows = sorted([r for r in e1 if r.get("model") == model],
                      key=lambda r: r.get("num_agents", 0))
        if not rows:
            continue
        x = [r["num_agents"] for r in rows]
        y = [r.get("achieved_concurrency", 0) for r in rows]
        ax.plot(x, y, color=colors[model], linewidth=2, marker="o",
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                zorder=4, label=MODEL_LABEL.get(model, model))
        ax.annotate(f"{MODEL_LABEL.get(model, model)}  {y[-1]:.1f}",
                    (x[-1], y[-1]), textcoords="offset points",
                    xytext=(10, dy.get(model, 0)), color=INK_2, fontsize=9,
                    fontweight="bold", va="center")
        drawn = True
    if not drawn:
        plt.close(fig)
        return None

    # The serialized baseline: what the shared server reported at every size.
    # Anchored mid-axis — at x=5 the caption runs through the N=5 markers.
    ax.axhline(1.0, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("serialized server (-np 1): 1.0 at every swarm size",
                (33, 1.0), textcoords="offset points", xytext=(0, 7),
                color=MUTED, fontsize=9, ha="left")

    _style(ax, ylabel="effective concurrent agents",
           xlabel="swarm size (concurrent requests issued)")
    ax.set_xticks([5, 15, 30, 50, 60])
    ax.set_xlim(0, 76)
    ax.set_title("Model size, not swarm size, caps real concurrency",
                 color=INK, fontsize=13, fontweight="bold", pad=34, loc="left")
    ax.text(0.0, 1.06,
            "Measured as N x (fastest request) / wall-clock - a figure a request queue cannot fake.",
            transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")

    leg = ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    controls = load_all_controls()
    e1 = load_latest("e1_*.json")
    e2 = load_latest("e2_*.json")

    print(f"loaded: {len(controls)} control rows, {len(e1)} E1 rows, {len(e2)} E2 rows")
    made = []
    for fn, args, name in (
        (fig_fidelity, (controls, e2), "fig1_fidelity_curve.png"),
        (fig_truncation, (controls,), "fig2_truncation.png"),
        (fig_concurrency, (e1,), "fig3_concurrency_ceiling.png"),
    ):
        p = fn(*args, os.path.join(OUTPUT_DIR, name))
        if p:
            made.append(p)
            print(f"  ✅ {p}")
        else:
            print(f"  ⏭  {name} — not enough data yet")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
