from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EE" / "PC" / "pc_ee_dur50_tau4.json"
OUTPUT_PATH = REPO_ROOT / "manuscript_figures" / "figure3.pdf"
FIGSIZE = (10, 8)
FONT_SIZE = 30
TITLE_SIZE = 24
YLIM = (0, 1.1)

METHODS = ["Linear", "Bilinear", "Cubic", "4th"]
METHOD_LABELS = ["Linear", "Quadratic", "Cubic", "Quartic"]


def load_ve_statistics(data_path: Path | str) -> dict[str, dict[str, float]]:
    data = json.loads(Path(data_path).read_text())
    stats: dict[str, dict[str, float]] = {}

    for method in METHODS:
        values = np.array([position[method] for position in data.values()], dtype=float)
        stats[method] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "n_samples": int(values.size),
        }

    return stats


def compute_sic(stats: dict[str, dict[str, float]]) -> float:
    return 4 - sum(stats[method]["mean"] for method in METHODS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure 3 from the PC EE VE json file.")
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    stats = load_ve_statistics(args.data_path)
    sic_value = compute_sic(stats)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    means = [stats[method]["mean"] for method in METHODS]
    stds = [stats[method]["std"] for method in METHODS]
    colors = plt.cm.Set2(np.linspace(0, 1, len(METHODS)))

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(METHODS))

    ax.bar(
        x,
        means,
        0.6,
        yerr=stds,
        capsize=5,
        color=colors,
        alpha=0.8,
        error_kw={"elinewidth": 2.5, "capthick": 2.5},
        linewidth=0.8,
    )
    ax.set_ylabel("VE")
    ax.set_xlabel("Reconstruction order")
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_LABELS)
    ax.set_ylim(*YLIM)
    ax.set_title(
        r"$SIC = 4-\overline{VE^1}-\overline{VE^2}-\overline{VE^3}-\overline{VE^4}$"
        + rf"$= {sic_value:.3f}$",
        fontsize=TITLE_SIZE,
        pad=20,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
