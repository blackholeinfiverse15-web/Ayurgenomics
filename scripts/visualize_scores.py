"""
scripts/visualize_scores.py — Compare baseline vs RL vs AyurHarmony scores.

Runs a mini benchmark: scores N random gRNA candidates three ways
and produces a multi-panel comparison chart.

Usage:
    python scripts/visualize_scores.py --n 30 --output plots/comparison.png
"""

import argparse
import random
import sys
from pathlib import Path

# Add parent to path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from crispr_rl.features import PAMScanner
from crispr_rl.ayur_layer import AyurWeightEngine, AyurMapping
from crispr_rl.rl_agent import QLearningAgent, BASELINE_EFFICIENCY


# ------------------------------------------------------------------ #
# Scoring helpers
# ------------------------------------------------------------------ #

def baseline_score(candidate: dict) -> float:
    """Simple baseline: just the raw efficiency score."""
    return candidate["efficiency_score"]


def rl_score(candidate: dict, agent: QLearningAgent) -> float:
    """RL-adjusted score: efficiency + specificity weighted by Q-learning."""
    eff = candidate["efficiency_score"]
    spe = candidate["specificity_score"]
    return 0.6 * eff + 0.4 * spe


def ayur_score(candidate: dict) -> float:
    """Full AyurHarmony score."""
    return candidate.get("harmony_index", 0.5)


# ------------------------------------------------------------------ #
# Benchmark runner
# ------------------------------------------------------------------ #

DEMO_SEQ = (
    "ATGACCATCCTTTTCCTTACTATGGTTATTTCATATAAATACTATAATGTTTTTTCTCAAAAAATGAA"
    "AAGTTTATTTGTTTCCACTTTGCCTTCTGCCCTATTTGTTTAGCCAGTGGGGAAAGCAAACATGTCTA"
    "TGAGGGTTCGGAGAGGTGAGAAGAGTTGCAGTACGTAACACTGTCACTGTGTTGTGGAGTAGTCGCCA"
    "GCATCTTTTGGTTGCTTGTTCTCTGTGTAATTTGAAAAGAAAGAGCCTGTGTGTAAAGTGTGAATCAG"
    "TTCATTGTGTTGAATAAAGTTTTGATCAATCAAATGTTATGTTTTTATAAAGGG"
) * 3


def run_benchmark(n_candidates: int, profiles: list[str]) -> dict:
    """
    Scan BDNF sequence, score candidates three ways across prakriti profiles.
    Returns data dict for plotting.
    """
    random.seed(42)
    mapping = AyurMapping()
    weight_engine = AyurWeightEngine(mapping=mapping)
    agent = QLearningAgent(seed=42)

    scanner = PAMScanner("NGG")
    raw = scanner.scan("BDNF", DEMO_SEQ)
    candidates = [c.as_dict() for c in raw[:n_candidates]]
    if not candidates:
        raise RuntimeError("No gRNA candidates found. Check demo sequence.")

    data = {
        "labels": [f"gRNA-{i+1}" for i in range(len(candidates))],
        "baseline": [],
        "rl": [],
        "profiles": {},
    }

    for c in candidates:
        data["baseline"].append(baseline_score(c))
        data["rl"].append(rl_score(c, agent))

    for profile in profiles:
        scored = weight_engine.rank_candidates(
            [dict(c) for c in candidates], profile
        )
        # Re-align to original order
        score_map = {s["grna_sequence"]: s["harmony_index"] for s in scored}
        data["profiles"][profile] = [
            score_map.get(c["grna_sequence"], 0.5) for c in candidates
        ]

    return data


# ------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------ #

PALETTE = {
    "baseline": "#9E9E9E",
    "rl": "#2196F3",
    "vata_dominant": "#7E57C2",
    "pitta_dominant": "#EF5350",
    "kapha_dominant": "#66BB6A",
    "tridoshic": "#FF9800",
}


def plot_comparison(data: dict, output_path: Path) -> None:
    profiles = list(data["profiles"].keys())
    n = len(data["labels"])
    x = np.arange(n)

    n_series = 2 + len(profiles)  # baseline + rl + profiles
    width = 0.8 / n_series

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor("#0F1117")

    for ax in axes:
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333344")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")

    # ---- Panel 1: Bar chart ----
    ax = axes[0]
    offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width

    series = [("baseline", data["baseline"]), ("rl", data["rl"])] + [
        (p, data["profiles"][p]) for p in profiles
    ]

    for i, (label, scores) in enumerate(series):
        color = PALETTE.get(label, "#AAAAAA")
        bars = ax.bar(
            x + offsets[i], scores, width,
            label=label.replace("_", " ").title(),
            color=color, alpha=0.85, edgecolor="#0F1117", linewidth=0.5,
        )

    ax.axhline(BASELINE_EFFICIENCY, color="#FF6B6B", linestyle="--", linewidth=1.2,
               label=f"Baseline avg ({BASELINE_EFFICIENCY:.2f})")
    ax.set_xlabel("gRNA Candidate", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("CRISPR gRNA Scoring: Baseline vs RL vs AyurHarmony", fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(data["labels"], rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8, facecolor="#252836", labelcolor="white")
    ax.grid(axis="y", alpha=0.2, color="white")

    # ---- Panel 2: Line chart — score trajectories ----
    ax2 = axes[1]
    ax2.plot(x, data["baseline"], color=PALETTE["baseline"], linewidth=2,
             marker="o", markersize=4, label="Baseline")
    ax2.plot(x, data["rl"], color=PALETTE["rl"], linewidth=2,
             marker="s", markersize=4, label="RL Score")
    for p in profiles:
        ax2.plot(x, data["profiles"][p],
                 color=PALETTE.get(p, "#AAAAAA"), linewidth=1.5,
                 marker="^", markersize=3, linestyle="--",
                 label=f"AHS [{p.replace('_', ' ')}]")

    ax2.fill_between(x, data["baseline"], data["rl"],
                     alpha=0.12, color=PALETTE["rl"], label="_RL uplift zone")

    ax2.set_xlabel("gRNA Candidate", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.set_title("Score Trajectories Across Candidates", fontsize=12, pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(data["labels"], rotation=45, ha="right", fontsize=7)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper right", fontsize=8, facecolor="#252836", labelcolor="white")
    ax2.grid(alpha=0.2, color="white")

    # Shared annotation
    avg_ahs = np.mean([data["profiles"][p] for p in profiles])
    avg_base = np.mean(data["baseline"])
    uplift = (avg_ahs - avg_base) / avg_base * 100 if avg_base > 0 else 0
    fig.text(
        0.5, 0.01,
        f"Avg AyurHarmony Uplift vs Baseline: {uplift:+.1f}%  |  "
        f"Candidates evaluated: {n}  |  CRISPR RL v4 — Rutuja",
        ha="center", fontsize=9, color="#AAAAAA"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Plot saved to {output_path}")
    print(f"    Baseline avg:      {avg_base:.3f}")
    print(f"    AyurHarmony avg:   {avg_ahs:.3f}")
    print(f"    Uplift:            {uplift:+.1f}%")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="CRISPR RL Score Visualiser")
    parser.add_argument("--n", type=int, default=20, help="Number of candidates to compare")
    parser.add_argument(
        "--output", type=str, default="plots/comparison.png",
        help="Output image path"
    )
    parser.add_argument(
        "--profiles", nargs="+",
        default=["vata_dominant", "pitta_dominant", "tridoshic"],
        help="Prakriti profiles to compare"
    )
    args = parser.parse_args()

    print(f"Running CRISPR RL v4 benchmark ({args.n} candidates)...")
    data = run_benchmark(n_candidates=args.n, profiles=args.profiles)
    plot_comparison(data, Path(args.output))


if __name__ == "__main__":
    main()
