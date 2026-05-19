from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_helpers import build_analysis_table, compute_ve_time, fit_cubic, fit_kappa


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EE" / "GoC" / "data" / "sim_data_2.csv"
OUTPUT_PATH = REPO_ROOT / "manuscript_figures" / "figure4c.pdf"
FIGSIZE = (10, 6)
FONT_SIZE = 20
LINEWIDTH = 8
LEGEND_FONTSIZE = 30
LEGEND_LOC = (0.15, 0.05)


def load_max_weight_traces(data_path: Path | str) -> dict[str, np.ndarray | float]:
    df = pd.read_csv(data_path)
    df_all = build_analysis_table(df)
    df_kappa = fit_kappa(df_all).set_index("time")
    df_cubic = fit_cubic(df_all).set_index("time")

    exc_w1 = float(df["exc_w"].max())
    exc_w2 = float(df["inh_w"].max())
    df_sel = df[
        np.isclose(df["exc_w"], exc_w1) &
        np.isclose(df["inh_w"], exc_w2)
    ].copy().sort_values("time")

    time = df_sel["time"].to_numpy()
    nv = df_sel["non_stim_value"].to_numpy()
    e1 = df_sel["e_value"].to_numpy() - nv
    e2 = df_sel["i_value"].to_numpy() - nv
    ssp = df_sel["ei_value"].to_numpy() - nv
    linear = e1 + e2

    kappa = df_kappa.loc[time, "kappa"].to_numpy()
    a = df_cubic.loc[time, "a"].to_numpy()
    b = df_cubic.loc[time, "b"].to_numpy()
    c = df_cubic.loc[time, "c"].to_numpy()

    quadratic = linear + kappa * e1 * e2
    cubic = linear + (a * e1 * e2 + b * e1**2 * e2 + c * e1 * e2**2)

    return {
        "time": time,
        "ssp": ssp,
        "linear": linear,
        "quadratic": quadratic,
        "cubic": cubic,
        "ve_linear": compute_ve_time(ssp, linear),
        "ve_quadratic": compute_ve_time(ssp, quadratic),
        "ve_cubic": compute_ve_time(ssp, cubic),
        "exc_w1": exc_w1,
        "exc_w2": exc_w2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure 4a from DCN EE trace data.")
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    trace = load_max_weight_traces(args.data_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        trace["time"],
        trace["linear"],
        label=f"Linear (VE={trace['ve_linear']:.3f})",
        c="#8fce00",
        linewidth=LINEWIDTH,
    )
    ax.plot(
        trace["time"],
        trace["quadratic"],
        label=f"Quadratic (VE={trace['ve_quadratic']:.3f})",
        c="#008fce",
        linewidth=LINEWIDTH,
    )
    ax.plot(
        trace["time"],
        trace["cubic"],
        label=f"Cubic (VE={trace['ve_cubic']:.3f})",
        c="#ce008f",
        linewidth=LINEWIDTH,
    )
    ax.plot(trace["time"], trace["ssp"], label="SSP", c="black", ls="--", linewidth=LINEWIDTH)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.legend(frameon=False, loc=LEGEND_LOC, fontsize=LEGEND_FONTSIZE)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
