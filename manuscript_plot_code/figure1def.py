from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_helpers import load_ve_heatmaps, plot_heatmap_panel


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EE" / "PC" / "data" / "sim_data_2.csv"
OUTPUT_PATH = REPO_ROOT / "manuscript_figures" / "figure1def_combined.pdf"
FIGSIZE = (28, 10)
FONT_SIZE = 24
COLORBAR_LABEL_SIZE = 24
COLORBAR_TICK_SIZE = 20
VMIN = 0.5
VMAX = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Fig.1d/e/f as three side-by-side VE heatmaps.")
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    heatmaps = load_ve_heatmaps(args.data_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)

    image = None
    for ax, (title, heatmap) in zip(axes, heatmaps.items()):
        image = plot_heatmap_panel(ax, heatmap, title=title, vmin=VMIN, vmax=VMAX)

    fig.subplots_adjust(left=0.05, right=0.89, bottom=0.12, top=0.90, wspace=0.45)
    right_panel_box = axes[-1].get_position()
    colorbar_axis = fig.add_axes([0.905, right_panel_box.y0, 0.012, right_panel_box.height])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("VE", fontsize=COLORBAR_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

    fig.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
