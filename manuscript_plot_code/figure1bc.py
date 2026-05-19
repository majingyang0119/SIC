from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_helpers import load_model_traces, load_selected_trace, plot_model_panel, plot_trace_panel


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EE" / "PC" / "data" / "sim_data_1.csv"
EXC_W1 = 0.017
EXC_W2 = 0.014875
OUTPUT_PATH = REPO_ROOT / "manuscript_figures" / "figure1bc_combined.pdf"
FIGSIZE = (20, 8)
FONT_SIZE = 20
LINEWIDTH = 8
LEGEND_FONTSIZE = 36
XLIM = (9, 55)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Fig.1b and Fig.1c side by side.")
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--exc-w1", type=float, default=EXC_W1)
    parser.add_argument("--exc-w2", type=float, default=EXC_W2)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    trace_panel = load_selected_trace(args.data_path, exc_w1=args.exc_w1, exc_w2=args.exc_w2)
    model_panel = load_model_traces(args.data_path, exc_w1=args.exc_w1, exc_w2=args.exc_w2)

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
    )
    plot_model_panel(
        axes[1],
        model_panel,
        linewidth=LINEWIDTH,
        xlim=XLIM,
        legend_loc=(0.6, 0.76),
        legend_fontsize=LEGEND_FONTSIZE,
    )

    fig.tight_layout(w_pad=6.0)
    fig.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
