from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_helpers import build_analysis_table, compute_ve_time, fit_cubic, fit_kappa


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "cerebellum_neurons_data" / "final_data" / "EE" / "PC" / "data" / "sim_data_2.csv"
OUTPUT_PATH = REPO_ROOT / "manuscript_figures" / "figures1.pdf"
FIGSIZE = (24, 24)
FONT_SIZE = 18
COLORBAR_LABEL_SIZE = 20
COLORBAR_TICK_SIZE = 16
VMIN = 0.5
VMAX = 1.0
N_SPLITS = 8
TRAIN_FRACTION = 0.5
RANDOM_SEED = 0


def get_weight_grid(df_raw: pd.DataFrame) -> tuple[list[float], list[float], list[tuple[float, float]]]:
    exc_weights = sorted(df_raw["exc_w"].unique())
    inh_weights = sorted(df_raw["inh_w"].unique())
    combos = [(exc_w, inh_w) for exc_w in exc_weights for inh_w in inh_weights]
    return exc_weights, inh_weights, combos


def split_weight_combos(
    combos: list[tuple[float, float]],
    n_splits: int,
    train_fraction: float,
    random_seed: int,
) -> list[tuple[set[tuple[float, float]], set[tuple[float, float]]]]:
    rng = np.random.default_rng(random_seed)
    n_train = int(round(len(combos) * train_fraction))
    combo_array = np.array(combos, dtype=object)
    splits = []

    for _ in range(n_splits):
        train_idx = rng.choice(len(combo_array), size=n_train, replace=False)
        train_combos = {tuple(combo_array[idx]) for idx in train_idx}
        test_combos = set(combos) - train_combos
        splits.append((train_combos, test_combos))

    return splits


def fit_models_on_training_set(
    df_raw: pd.DataFrame,
    train_combos: set[tuple[float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = df_raw.apply(lambda row: (row["exc_w"], row["inh_w"]) in train_combos, axis=1)
    df_train = df_raw.loc[mask].copy()
    df_all = build_analysis_table(df_train)
    return fit_kappa(df_all), fit_cubic(df_all)


def compute_test_heatmaps(
    df_raw: pd.DataFrame,
    df_kappa: pd.DataFrame,
    df_cubic: pd.DataFrame,
    train_combos: set[tuple[float, float]],
    exc_weights: list[float],
    inh_weights: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    kappa_map = df_kappa.set_index("time")["kappa"]
    cubic_map = df_cubic.set_index("time")
    bilinear_heatmap = pd.DataFrame(np.nan, index=exc_weights, columns=inh_weights, dtype=float)
    cubic_heatmap = pd.DataFrame(np.nan, index=exc_weights, columns=inh_weights, dtype=float)

    for (exc_w1, exc_w2), df_w in df_raw.groupby(["exc_w", "inh_w"]):
        if (exc_w1, exc_w2) in train_combos:
            continue

        df_w = df_w.sort_values("time")
        nv = df_w["non_stim_value"].to_numpy()
        v1 = df_w["e_value"].to_numpy() - nv
        v2 = df_w["i_value"].to_numpy() - nv
        v_gt = df_w["ei_value"].to_numpy() - nv
        time = df_w["time"].to_numpy()

        linear = v1 + v2
        kappa = kappa_map.loc[time].to_numpy()
        a = cubic_map.loc[time, "a"].to_numpy()
        b = cubic_map.loc[time, "b"].to_numpy()
        c = cubic_map.loc[time, "c"].to_numpy()

        bilinear = linear + kappa * v1 * v2
        cubic = linear + (a * v1 * v2 + b * v1**2 * v2 + c * v1 * v2**2)

        bilinear_heatmap.loc[exc_w1, exc_w2] = compute_ve_time(v_gt, bilinear)
        cubic_heatmap.loc[exc_w1, exc_w2] = compute_ve_time(v_gt, cubic)

    return bilinear_heatmap, cubic_heatmap


def plot_holdout_heatmap(
    ax: plt.Axes,
    heatmap: pd.DataFrame,
    train_combos: set[tuple[float, float]],
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = "Reds_r",
) -> plt.AxesImage:
    image_cmap = plt.get_cmap(cmap).copy()
    image_cmap.set_bad(color="#d9d9d9")
    image = ax.imshow(
        heatmap.to_numpy(),
        origin="lower",
        aspect="equal",
        cmap=image_cmap,
        vmin=vmin,
        vmax=vmax,
    )

    n_rows, n_cols = heatmap.shape
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(np.arange(1, n_cols + 1))
    ax.set_yticklabels(np.arange(1, n_rows + 1))
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("E1 weight")
    ax.set_ylabel("E2 weight")
    ax.set_title(title)

    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 8 random half-split holdout VE heatmap pairs for quadratic and cubic models.")
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    df_raw = pd.read_csv(args.data_path)
    exc_weights, inh_weights, combos = get_weight_grid(df_raw)
    splits = split_weight_combos(
        combos=combos,
        n_splits=args.n_splits,
        train_fraction=args.train_fraction,
        random_seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, axes = plt.subplots(4, 4, figsize=FIGSIZE)

    image = None
    for split_idx, (train_combos, _) in enumerate(splits, start=1):
        df_kappa, df_cubic = fit_models_on_training_set(df_raw, train_combos)
        bilinear_heatmap, cubic_heatmap = compute_test_heatmaps(
            df_raw=df_raw,
            df_kappa=df_kappa,
            df_cubic=df_cubic,
            train_combos=train_combos,
            exc_weights=exc_weights,
            inh_weights=inh_weights,
        )

        row = (split_idx - 1) // 2
        col_offset = ((split_idx - 1) % 2) * 2
        image = plot_holdout_heatmap(
            axes[row, col_offset],
            bilinear_heatmap,
            train_combos,
            title=f"Split {split_idx}: Quadratic",
            vmin=VMIN,
            vmax=VMAX,
        )
        image = plot_holdout_heatmap(
            axes[row, col_offset + 1],
            cubic_heatmap,
            train_combos,
            title=f"Split {split_idx}: Cubic",
            vmin=VMIN,
            vmax=VMAX,
        )

    fig.subplots_adjust(left=0.05, right=0.92, bottom=0.05, top=0.96, wspace=0.38, hspace=0.38)
    colorbar_axis = fig.add_axes([0.935, 0.14, 0.015, 0.7])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Holdout VE", fontsize=COLORBAR_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

    fig.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
