from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_helpers import calculate_ve_statistics


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "manuscript_figures"
FIGSIZE = (12, 8)
FONT_SIZE = 30
YLIM = (0.4, 1.1)
METHODS = ["Linear", "Bilinear", "Cubic"]
METHOD_LABELS = {
    "Linear": "Linear",
    "Bilinear": "Quadratic",
    "Cubic": "Cubic",
}
METHOD_COLORS = {
    "Linear": "#8fce00",
    "Bilinear": "#008fce",
    "Cubic": "#ce008f",
}
FILE_PATHS = {
    "GrC": REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EE" / "GrC" / "grc_ee_dur50_tau4.json",
    "BC": REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EE" / "BC" / "bc_ee_dur25_tau4.json",
    "GoC": REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EE" / "GoC" / "goc_ee_dur50_tau4.json",
    "IO": REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EE" / "IO" / "io_ee_dur50_tau4.json",
    "SC": REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EE" / "SC" / "sc_ee_dur50_tau4.json",
    "DCN": REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EE" / "DCN" / "dcn_ee_dur50_tau4.json",
}


def load_neuron_statistics() -> list[dict[str, object]]:
    results = []
    for neuron_name, file_path in FILE_PATHS.items():
        data = json.loads(Path(file_path).read_text())
        results.append(
            {
                "neuron": neuron_name,
                "statistics": calculate_ve_statistics(data),
            }
        )
    return results


def plot_single_method(neuron_results: list[dict[str, object]], method: str, output_dir: Path) -> Path:
    neuron_names = [result["neuron"] for result in neuron_results]
    means = [result["statistics"][method]["mean"] for result in neuron_results]
    stds = [result["statistics"][method]["std"] for result in neuron_results]
    x = np.arange(len(neuron_names))

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(
        x,
        means,
        0.6,
        yerr=stds,
        capsize=10,
        color=METHOD_COLORS[method],
        alpha=0.8,
        error_kw={"elinewidth": 2.5, "capthick": 2.5},
        linewidth=2,
        label=METHOD_LABELS[method],
    )
    ax.set_xlabel("Neuron type", fontsize=28)
    ax.set_ylabel("VE", fontsize=28)
    ax.set_xticks(x)
    ax.set_xticklabels(neuron_names, rotation=15, fontsize=24)
    ax.set_ylim(*YLIM)
    ax.legend(frameon=False, fontsize=26, loc=(0.8, 1.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_path = output_dir / f"figure4g_{method.lower()}.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the three separate Figure 4g bar plots.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    neuron_results = load_neuron_statistics()

    for method in METHODS:
        plot_single_method(neuron_results, method, output_dir)


if __name__ == "__main__":
    main()
