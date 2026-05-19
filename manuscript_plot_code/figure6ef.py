from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_helpers import load_model_traces, load_selected_trace, plot_model_panel, plot_trace_panel


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "II" / "PC" / "data" / "sim_data_2.csv"
OUTPUT_PATH = REPO_ROOT / "manuscript_figures" / "figure6ef_combined.pdf"
FIGSIZE = (20, 8)
FONT_SIZE = 20
LINEWIDTH = 8
LEGEND_FONTSIZE = 36
XLIM = (9, 55)


INH_W1, INH_W2 = 0.053,0.08700000000000001


def compute_panel_ylims(
    trace_panel: dict[str, np.ndarray], model_panel: dict[str, np.ndarray | float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    left_values = np.concatenate([trace_panel["ssp"], trace_panel["e1"], trace_panel["e2"]])
    right_values = np.concatenate([model_panel["ssp"], model_panel["linear"], model_panel["bilinear"]])

    right_ymin = float(right_values.min()) - 0.2
    right_ymax = float(right_values.max())
    right_span = right_ymax - right_ymin

    left_ymin = float(left_values.min())
    left_ymax = float(left_values.max())
    left_center = 0.5 * (left_ymin + left_ymax)
    left_span = max(left_ymax - left_ymin, right_span )

    return (
        (left_center - 0.5 * left_span, left_center + 0.5 * left_span),
        (right_ymin, right_ymax),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Fig.6e and Fig.6f side by side.")
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--inh-w1", type=float, default=INH_W1)
    parser.add_argument("--inh-w2", type=float, default=INH_W2)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    trace_panel = load_selected_trace(args.data_path, exc_w1=args.inh_w1, exc_w2=args.inh_w2)
    model_panel = load_model_traces(args.data_path, exc_w1=args.inh_w1, exc_w2=args.inh_w2)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    plot_trace_panel(
        axes[0],
        trace_panel,
        linewidth=LINEWIDTH,
        xlim=XLIM,
        legend_loc=(0.38, 0.68),
        legend_fontsize=LEGEND_FONTSIZE,
        e_label="IPSP1",
        e_color="#1B2EB6",
        i_label="IPSP2",
        i_color="#3286E7",
    )
    plot_model_panel(
        axes[1],
        model_panel,
        linewidth=LINEWIDTH,
        xlim=XLIM,
        legend_loc=(0.3, 0.76),
        legend_fontsize=LEGEND_FONTSIZE,
        show_cubic=False,
    )
    left_ylim, right_ylim = compute_panel_ylims(trace_panel, model_panel)
    axes[0].set_ylim(*left_ylim)
    axes[1].set_ylim(*right_ylim)

    fig.tight_layout(w_pad=6.0)
    fig.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
