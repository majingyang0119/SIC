from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure5 import (
    LABEL_OFFSETS_AC,
    LABEL_OFFSETS_DC,
    compute_sic,
    load_complexities,
    load_ve_lists,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FIG_PATH = REPO_ROOT / "manuscript_figures" / "figures2.pdf"
FIGSIZE = (24, 24)
FONT_SIZE = 20
TITLE_FONTSIZE = 24
ANNOTATION_FONTSIZE = 20


def compute_axis_limits(values: list[float], margin_ratio: float = 0.05) -> tuple[float, float]:
    min_val = min(values)
    max_val = max(values)
    span = max_val - min_val
    margin = span * margin_ratio if span > 0 else 0.05
    return min_val - margin, max_val + margin


def fit_linear(x_vals: list[float], y_vals: list[float]) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    y_pred = slope * np.asarray(x_vals) + intercept
    y_true = np.asarray(y_vals)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), float(r_squared)


def build_leave_one_out_table(
    dc_dict: dict[str, float],
    ac_dict: dict[str, float],
    sic_dict: dict[str, float],
) -> list[dict[str, float | str]]:
    neurons = sorted(set(dc_dict) & set(ac_dict) & set(sic_dict))
    records: list[dict[str, float | str]] = []

    for omitted_neuron in neurons:
        kept_neurons = [neuron for neuron in neurons if neuron != omitted_neuron]
        sic_vals = [sic_dict[neuron] for neuron in kept_neurons]
        dc_vals = [dc_dict[neuron] for neuron in kept_neurons]
        ac_vals = [ac_dict[neuron] for neuron in kept_neurons]

        dc_slope, dc_intercept, dc_r_squared = fit_linear(dc_vals, sic_vals)
        ac_slope, ac_intercept, ac_r_squared = fit_linear(ac_vals, sic_vals)

        records.append(
            {
                "omitted_neuron": omitted_neuron,
                "n_points": len(kept_neurons),
                "dc_slope": dc_slope,
                "dc_intercept": dc_intercept,
                "dc_r_squared": dc_r_squared,
                "ac_slope": ac_slope,
                "ac_intercept": ac_intercept,
                "ac_r_squared": ac_r_squared,
            }
        )

    return records


def plot_leave_one_out_regressions(
    dc_dict: dict[str, float],
    ac_dict: dict[str, float],
    sic_dict: dict[str, float],
    output_path: Path,
) -> None:
    neurons = sorted(set(dc_dict) & set(ac_dict) & set(sic_dict))
    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, axes = plt.subplots(4, 4, figsize=FIGSIZE, sharey=True)
    dc_xlim = compute_axis_limits([dc_dict[neuron] for neuron in neurons])
    ac_xlim = compute_axis_limits([ac_dict[neuron] for neuron in neurons])
    sic_ylim = compute_axis_limits([sic_dict[neuron] for neuron in neurons])

    panel_specs = [{"title": "All", "kept_neurons": neurons}]
    panel_specs.extend(
        {"title": f"-{omitted_neuron}", "kept_neurons": [neuron for neuron in neurons if neuron != omitted_neuron]}
        for omitted_neuron in neurons
    )

    for idx, panel_spec in enumerate(panel_specs):
        kept_neurons = panel_spec["kept_neurons"]
        sic_vals = np.asarray([sic_dict[neuron] for neuron in kept_neurons], dtype=float)
        dc_vals = np.asarray([dc_dict[neuron] for neuron in kept_neurons], dtype=float)
        ac_vals = np.asarray([ac_dict[neuron] for neuron in kept_neurons], dtype=float)

        dc_r = float(np.corrcoef(dc_vals, sic_vals)[0, 1])
        ac_r = float(np.corrcoef(ac_vals, sic_vals)[0, 1])

        col = idx // 2
        row_offset = (idx % 2) * 2
        dc_ax = axes[row_offset, col]
        ac_ax = axes[row_offset + 1, col]

        dc_ax.scatter(dc_vals, sic_vals, color="black", s=35)
        dc_ax.set_title(f"{panel_spec['title']}\nPearson r = {dc_r:.3f}", fontweight="bold", fontsize=TITLE_FONTSIZE)
        dc_ax.set_xlabel("passive dendrite complexity")
        dc_ax.set_xlim(*dc_xlim)
        dc_ax.set_ylim(*sic_ylim)

        ac_ax.scatter(ac_vals, sic_vals, color="black", s=35)
        ac_ax.set_title(f"{panel_spec['title']}\nPearson r = {ac_r:.3f}", fontweight="bold", fontsize=TITLE_FONTSIZE)
        ac_ax.set_xlabel("active channel complexity")
        ac_ax.set_xlim(*ac_xlim)
        ac_ax.set_ylim(*sic_ylim)

        for ax, x_vals, y_vals, labels, offsets in (
            (dc_ax, dc_vals, sic_vals, kept_neurons, LABEL_OFFSETS_DC),
            (ac_ax, ac_vals, sic_vals, kept_neurons, LABEL_OFFSETS_AC),
        ):
            for x_val, y_val, label in zip(x_vals, y_vals, labels):
                dx, dy = offsets.get(label, (8, 8))
                ax.annotate(label, (x_val, y_val), xytext=(dx, dy), textcoords="offset points", fontsize=ANNOTATION_FONTSIZE)
            ax.tick_params(labelsize=FONT_SIZE)

    for row in range(4):
        axes[row, 0].set_ylabel("synaptic integration complexity")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-out linear regression coefficients for DC/AC vs SIC.")
    parser.add_argument("--output-fig", default=str(OUTPUT_FIG_PATH))
    args = parser.parse_args()

    ve_lists = load_ve_lists()
    dc_dict, ac_dict, _ = load_complexities()
    sic_dict = compute_sic(ve_lists)

    build_leave_one_out_table(dc_dict=dc_dict, ac_dict=ac_dict, sic_dict=sic_dict)

    output_fig_path = Path(args.output_fig)
    output_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_leave_one_out_regressions(
        dc_dict=dc_dict,
        ac_dict=ac_dict,
        sic_dict=sic_dict,
        output_path=output_fig_path,
    )


if __name__ == "__main__":
    main()
