from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_helpers import load_model_traces, load_selected_trace, plot_model_panel, plot_trace_panel


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EI" / "PC" / "data" / "sim_data_1.csv"
OUTPUT_PATH = REPO_ROOT / "manuscript_figures" / "figure6bc_combined.pdf"
FIGSIZE = (20, 8)
FONT_SIZE = 20
LINEWIDTH = 8
LEGEND_FONTSIZE = 36
XLIM = (9, 55)


EXC_W, INH_W = 0.017,0.0135625


def compute_panel_ylims(
    trace_panel: dict[str, np.ndarray], model_panel: dict[str, np.ndarray | float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    left_values = np.concatenate([trace_panel["ssp"], trace_panel["e1"], trace_panel["e2"]])
    right_values = np.concatenate([model_panel["ssp"], model_panel["linear"], model_panel["bilinear"]])

    right_ymin = float(right_values.min())
    right_ymax = float(right_values.max()) + 0.2
    right_span = right_ymax - right_ymin

    left_ymin = float(left_values.min())
    left_ymax = float(left_values.max())
    left_center = 0.5 * (left_ymin + left_ymax)
    left_span = max(left_ymax - left_ymin, right_span * 1.5)

    return (
        (left_center - 0.5 * left_span, left_center + 0.5 * left_span),
        (right_ymin, right_ymax),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Fig.6b and Fig.6c side by side.")
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--exc-w", type=float, default=EXC_W)
    parser.add_argument("--inh-w", type=float, default=INH_W)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    trace_panel = load_selected_trace(args.data_path, exc_w1=args.exc_w, exc_w2=args.inh_w)
    model_panel = load_model_traces(args.data_path, exc_w1=args.exc_w, exc_w2=args.inh_w)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    plot_trace_panel(
        axes[0],
        trace_panel,
        linewidth=LINEWIDTH,
        xlim=XLIM,
        legend_loc=(0.68, 0.68),
        legend_fontsize=LEGEND_FONTSIZE,
        e_label="EPSP",
        e_color="#F44336",
        i_label="IPSP",
        i_color="#1B2EB6",
    )
    plot_model_panel(
        axes[1],
        model_panel,
        linewidth=LINEWIDTH,
        xlim=XLIM,
        legend_loc=(0.6, 0.76),
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
