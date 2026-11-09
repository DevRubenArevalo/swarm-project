"""Chart generator: fidelity curve and scaling comparison visualizations."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_fidelity_curve(data: list[dict], output_path: str) -> None:
    """
    Generate the main fidelity curve: two subplots (completion rate, drift score).
    X-axis = swarm size, Y-axis = score (0-1), one line per model.

    Each dict in data should have:
      model (str), num_agents (int), completion_rate (float), drift_score (float)
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Group by model
    models: dict[str, list[dict]] = {}
    for d in data:
        models.setdefault(d["model"], []).append(d)

    for m_list in models.values():
        m_list.sort(key=lambda x: x["num_agents"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Swarm Fidelity: Where Do Local Models Break?", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10.colors
    for idx, (model, entries) in enumerate(sorted(models.items())):
        xs = [e["num_agents"] for e in entries]
        ys_comp = [e.get("completion_rate", 0) for e in entries]
        ys_drift = [e.get("drift_score", 0) for e in entries]
        color = colors[idx % len(colors)]

        ax1.plot(xs, ys_comp, "o-", color=color, label=model, linewidth=2, markersize=6)
        ax2.plot(xs, ys_drift, "s-", color=color, label=model, linewidth=2, markersize=6)

    # Styling
    for ax, title in [(ax1, "Completion Rate"), (ax2, "Drift Score (1.0 = on-task)")]:
        ax.set_ylabel(title)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    ax2.set_xlabel("Swarm Size (number of concurrent agents)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_scaling_comparison(data: list[dict], output_path: str) -> None:
    """
    Bar chart: max swarm size achieved per model (where completion_rate >= 0.8).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # For each model, find the max num_agents where completion_rate >= 0.8
    model_max: dict[str, int] = {}
    for d in data:
        model = d["model"]
        if d.get("completion_rate", 0) >= 0.8:
            model_max[model] = max(model_max.get(model, 0), d["num_agents"])

    if not model_max:
        print("  No data with completion_rate >= 0.8, skipping chart.")
        return

    models = sorted(model_max.keys())
    max_sizes = [model_max[m] for m in models]

    _fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, max_sizes, color=plt.cm.tab10.colors[:len(models)], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Max Swarm Size (completion ≥ 80%)")
    ax.set_title("Scaling Comparison: Max Swarm Size per Model", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, max_sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontweight="bold")

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def _synthetic_data() -> list[dict]:
    """Generate synthetic test data for chart preview."""
    import random
    random.seed(42)
    models = ["qwen3.5:4b", "qwen3.5:9b", "deepseek-r1:14b", "qwen3.5:27b", "flint:40b"]
    swarm_sizes = [5, 15, 30, 50, 60]
    data = []
    for model in models:
        # Simulate: bigger models handle larger swarms better
        base = 0.95 if "40b" in model or "27b" in model else 0.85 if "14b" in model or "9b" in model else 0.7
        for n in swarm_sizes:
            penalty = max(0, (n - 20) * 0.005)
            comp = max(0.1, base - penalty + random.uniform(-0.05, 0.05))
            drift = max(0.1, comp - random.uniform(0, 0.1))
            data.append({
                "model": model,
                "num_agents": n,
                "completion_rate": round(comp, 3),
                "drift_score": round(drift, 3),
            })
    return data


if __name__ == "__main__":
    print("Generating charts with synthetic data...")
    data = _synthetic_data()
    plot_fidelity_curve(data, "output/fidelity_curve.png")
    plot_scaling_comparison(data, "output/scaling_comparison.png")
    print("Done.")
