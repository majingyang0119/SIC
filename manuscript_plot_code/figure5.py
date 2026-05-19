from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_helpers import calculate_ve_statistics


REPO_ROOT = Path(__file__).resolve().parents[1]
MORPHOLOGY_PATH = REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "morphology_complexity.json"
OUTPUT_DIR = REPO_ROOT / "manuscript_figures"
FONT_SIZE = 30
FIGSIZE = (9, 8)

VE_JSON_PATHS = {
    "GrC": REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EE" / "GrC" / "grc_ee_dur50_tau4.json",
    "BC": REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EE" / "BC" / "bc_ee_dur25_tau4.json",
    "GoC": REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EE" / "GoC" / "goc_ee_dur50_tau4.json",
    "IO": REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EE" / "IO" / "io_ee_dur50_tau4.json",
    "SC": REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EE" / "SC" / "sc_ee_dur50_tau4.json",
    "DCN": REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EE" / "DCN" / "dcn_ee_dur50_tau4.json",
    "PC": REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EE" / "PC" / "pc_ee_dur50_tau4.json",
}

LABEL_OFFSETS_DC = {
    "PC": (-28, -35),
    "GoC": (12, 0),
    "DCN": (10, 0),
    "SC": (10, 0),
    "IO": (-40, 0),
    "BC": (0, 8),
    "GrC": (-10, 14),
}

LABEL_OFFSETS_AC = {
    "PC": (-28, -35),
    "GoC": (12, 0),
    "DCN": (10, 0),
    "SC": (10, 0),
    "IO": (10, 0),
    "BC": (10, 8),
    "GrC": (10, 8),
}


def load_ve_lists() -> dict[str, list[float]]:
    ve_lists: dict[str, list[float]] = {}
    for neuron, path in VE_JSON_PATHS.items():
        data = json.loads(Path(path).read_text())
        stats = calculate_ve_statistics(data)
        ve_lists[neuron] = [
            stats["Linear"]["mean"],
            stats["Bilinear"]["mean"],
            stats["Cubic"]["mean"],
            stats["4th"]["mean"],
        ]
    return ve_lists


def load_complexities() -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    morphology = json.loads(Path(MORPHOLOGY_PATH).read_text())
    dendritic_lengths = {neuron: values["Total dendritic length(μm)"] for neuron, values in morphology.items()}
    bifurcations = {neuron: values["dendritic branch points"] for neuron, values in morphology.items()}
    dendritic_sections = {neuron: values["dendritic sections"] for neuron, values in morphology.items()}
    channel_types = {
        neuron: sum(
            1
            for channel, count in values["Channels (number of sections containing each)"].items()
            if channel != "Total channel insertions" and count > 1
        )
        for neuron, values in morphology.items()
    }

    max_length = max(dendritic_lengths.values())
    max_bifurcations = max(bifurcations.values())
    max_sections = max(dendritic_sections.values())
    max_channel_types = max(channel_types.values())

    dc = {}
    ac = {}
    for neuron in morphology:
        dc[neuron] = (
            dendritic_lengths[neuron] / max_length
            + bifurcations[neuron] / max_bifurcations
            + dendritic_sections[neuron] / max_sections
        ) / 3
        ac[neuron] = channel_types[neuron] / max_channel_types

    return dc, ac, channel_types


def compute_sic(ve_lists: dict[str, list[float]]) -> dict[str, float]:
    return {neuron: 4 - sum(values) for neuron, values in ve_lists.items()}


def annotate_points(ax: plt.Axes, labels: list[str], x_vals: list[float], y_vals: list[float], offsets: dict[str, tuple[int, int]]) -> None:
    for label, x_val, y_val in zip(labels, x_vals, y_vals):
        dx, dy = offsets.get(label, (8, 8))
        ax.annotate(label, xy=(x_val, y_val), xytext=(dx, dy), textcoords="offset points", fontsize=30)


def plot_scatter(
    x_dict: dict[str, float],
    sic_dict: dict[str, float],
    x_label: str,
    output_path: Path,
    offsets: dict[str, tuple[int, int]],
) -> None:
    common_neurons = sorted(set(x_dict) & set(sic_dict))
    x_vals = [x_dict[neuron] for neuron in common_neurons]
    sic_vals = [sic_dict[neuron] for neuron in common_neurons]
    pearson_r = float(np.corrcoef(x_vals, sic_vals)[0, 1])

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(x_vals, sic_vals, color="black", s=100)
    annotate_points(ax, common_neurons, x_vals, sic_vals, offsets)
    ax.set_xlabel(x_label)
    ax.set_ylabel("synaptic integration complexity")
    ax.set_title(f"Pearson r = {pearson_r:.3f}")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 5 complexity vs SIC scatter plots.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ve_lists = load_ve_lists()
    dc_dict, ac_dict, channel_type_counts = load_complexities()
    sic_dict = compute_sic(ve_lists)

    plot_scatter(
        dc_dict,
        sic_dict,
        x_label="passive dendrite complexity",
        output_path=output_dir / "figure5_dc_sic.pdf",
        offsets=LABEL_OFFSETS_DC,
    )
    plot_scatter(
        ac_dict,
        sic_dict,
        x_label="active channel complexity",
        output_path=output_dir / "figure5_ac_sic.pdf",
        offsets=LABEL_OFFSETS_AC,
    )


if __name__ == "__main__":
    main()
