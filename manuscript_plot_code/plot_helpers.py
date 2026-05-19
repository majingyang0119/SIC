from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_analysis_table(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (exc_w1, exc_w2), group in df.groupby(["exc_w", "inh_w"]):
        for _, row in group.iterrows():
            nv = row["non_stim_value"]
            ev = row["e_value"]
            iv = row["i_value"]
            eiv = row["ei_value"]
            records.append(
                {
                    "time": row["time"],
                    "exc_w1": exc_w1,
                    "exc_w2": exc_w2,
                    "v1": ev - nv,
                    "v2": iv - nv,
                    "v_m": (ev - nv) * (iv - nv),
                    "v_sc": eiv - row["lin_sum_value"],
                }
            )
    return pd.DataFrame(records)


def fit_kappa(df_all: pd.DataFrame) -> pd.DataFrame:
    records = []
    for time, df_t in df_all.groupby("time"):
        x = df_t["v_m"].to_numpy()
        y = df_t["v_sc"].to_numpy()
        kappa, *_ = np.linalg.lstsq(x[:, None], y, rcond=None)
        y_pred = x * kappa[0]
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        records.append(
            {
                "time": time,
                "kappa": kappa[0],
                "VE": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            }
        )
    return pd.DataFrame(records).sort_values("time").reset_index(drop=True)


def fit_cubic(df_all: pd.DataFrame) -> pd.DataFrame:
    records = []
    for time, df_t in df_all.groupby("time"):
        v1 = df_t["v1"].to_numpy()
        v2 = df_t["v2"].to_numpy()
        y = df_t["v_sc"].to_numpy()
        x = np.column_stack([v1 * v2, v1**2 * v2, v1 * v2**2])
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        y_pred = x @ coef
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        records.append(
            {
                "time": time,
                "a": coef[0],
                "b": coef[1],
                "c": coef[2],
                "VE": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            }
        )
    return pd.DataFrame(records).sort_values("time").reset_index(drop=True)


def compute_ve_time(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def compute_weightwise_time_ve(df_raw: pd.DataFrame, df_kappa: pd.DataFrame, df_cubic: pd.DataFrame) -> pd.DataFrame:
    kappa_map = df_kappa.set_index("time")["kappa"]
    cubic_map = df_cubic.set_index("time")
    records = []

    for (exc_w1, exc_w2), df_w in df_raw.groupby(["exc_w", "inh_w"]):
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

        records.append(
            {
                "exc_w1": exc_w1,
                "exc_w2": exc_w2,
                "VE_linear": compute_ve_time(v_gt, linear),
                "VE_bilinear": compute_ve_time(v_gt, bilinear),
                "VE_cubic": compute_ve_time(v_gt, cubic),
            }
        )

    return pd.DataFrame(records)


def load_selected_trace(data_path: Path | str, exc_w1: float, exc_w2: float) -> dict[str, np.ndarray]:
    df = pd.read_csv(data_path)
    df_sel = df[
        np.isclose(df["exc_w"], exc_w1) &
        np.isclose(df["inh_w"], exc_w2)
    ].copy()
    if df_sel.empty:
        raise ValueError(f"No trace found for exc_w1={exc_w1}, exc_w2={exc_w2} in {data_path}")

    df_sel = df_sel.sort_values("time")
    nv = df_sel["non_stim_value"].to_numpy()
    e1 = df_sel["e_value"].to_numpy() - nv
    e2 = df_sel["i_value"].to_numpy() - nv
    ssp = df_sel["ei_value"].to_numpy() - nv
    return {
        "time": df_sel["time"].to_numpy(),
        "e1": e1,
        "e2": e2,
        "ssp": ssp,
        "linear": e1 + e2,
    }


def load_model_traces(data_path: Path | str, exc_w1: float, exc_w2: float) -> dict[str, np.ndarray | float]:
    df = pd.read_csv(data_path)
    df_all = build_analysis_table(df)
    df_kappa = fit_kappa(df_all).set_index("time")
    df_cubic = fit_cubic(df_all).set_index("time")
    trace = load_selected_trace(data_path=data_path, exc_w1=exc_w1, exc_w2=exc_w2)

    time = trace["time"]
    e1 = trace["e1"]
    e2 = trace["e2"]
    linear = trace["linear"]
    ssp = trace["ssp"]

    kappa = df_kappa.loc[time, "kappa"].to_numpy()
    a = df_cubic.loc[time, "a"].to_numpy()
    b = df_cubic.loc[time, "b"].to_numpy()
    c = df_cubic.loc[time, "c"].to_numpy()

    bilinear = linear + kappa * e1 * e2
    cubic = linear + (a * e1 * e2 + b * e1**2 * e2 + c * e1 * e2**2)

    return {
        **trace,
        "bilinear": bilinear,
        "cubic": cubic,
        "ve_linear": compute_ve_time(ssp, linear),
        "ve_bilinear": compute_ve_time(ssp, bilinear),
        "ve_cubic": compute_ve_time(ssp, cubic),
    }


def load_ve_heatmaps(data_path: Path | str) -> dict[str, pd.DataFrame]:
    df_raw = pd.read_csv(data_path)
    df_all = build_analysis_table(df_raw)
    df_kappa = fit_kappa(df_all)
    df_cubic = fit_cubic(df_all)
    df_ve = compute_weightwise_time_ve(df_raw, df_kappa, df_cubic)

    return {
        "Linear": df_ve.pivot(index="exc_w1", columns="exc_w2", values="VE_linear"),
        "Quadratic": df_ve.pivot(index="exc_w1", columns="exc_w2", values="VE_bilinear"),
        "Cubic": df_ve.pivot(index="exc_w1", columns="exc_w2", values="VE_cubic"),
    }


def style_panel(
    ax: plt.Axes,
    xlim: tuple[float, float],
    legend_loc: tuple[float, float],
    legend_fontsize: float,
) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_xlim(*xlim)
    ax.legend(frameon=False, loc=legend_loc, fontsize=legend_fontsize)
    ax.axis("off")


def plot_trace_panel(
    ax: plt.Axes,
    trace: dict[str, np.ndarray],
    linewidth: float,
    xlim: tuple[float, float],
    legend_loc: tuple[float, float],
    legend_fontsize: float,
    e_label: str = "EPSP1",
    e_color: str = "#f6b26b",
    i_label: str = "EPSP2",
    i_color: str = "#f44336",
) -> None:
    ax.plot(trace["time"], trace["ssp"], label="SSP", c="black", linewidth=linewidth)
    ax.plot(trace["time"], trace["e1"], label=e_label, c=e_color, linewidth=linewidth)
    ax.plot(trace["time"], trace["e2"], label=i_label, c=i_color, linewidth=linewidth)
    style_panel(ax, xlim=xlim, legend_loc=legend_loc, legend_fontsize=legend_fontsize)


def plot_model_panel(
    ax: plt.Axes,
    trace: dict[str, np.ndarray | float],
    linewidth: float,
    xlim: tuple[float, float],
    legend_loc: tuple[float, float],
    legend_fontsize: float,
    show_cubic: bool = True,
) -> None:
    ax.plot(
        trace["time"],
        trace["linear"],
        label=f"Linear (VE={trace['ve_linear']:.3f})",
        c="#8fce00",
        linewidth=linewidth,
    )
    ax.plot(
        trace["time"],
        trace["bilinear"],
        label=f"Quadratic (VE={trace['ve_bilinear']:.3f})",
        c="#008fce",
        linewidth=linewidth,
    )
    if show_cubic:
        ax.plot(
            trace["time"],
            trace["cubic"],
            label=f"Cubic (VE={trace['ve_cubic']:.3f})",
            c="#ce008f",
            linewidth=linewidth,
        )
    ax.plot(trace["time"], trace["ssp"], label="SSP", c="black", ls="--", linewidth=linewidth)
    style_panel(ax, xlim=xlim, legend_loc=legend_loc, legend_fontsize=legend_fontsize)


def plot_heatmap_panel(
    ax: plt.Axes,
    heatmap: pd.DataFrame,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = "Reds_r",
) -> plt.AxesImage:
    image = ax.imshow(
        heatmap.to_numpy(),
        origin="lower",
        aspect="equal",
        cmap=cmap,
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


def calculate_ve_statistics(data: dict[str, dict[str, float]] | pd.DataFrame) -> dict[str, dict[str, float]]:
    if isinstance(data, pd.DataFrame):
        raw_data = data.to_dict()
    else:
        raw_data = data

    methods = ["Linear", "Bilinear", "Cubic", "4th"]
    stats: dict[str, dict[str, float]] = {}
    for method in methods:
        values = np.array([position[method] for position in raw_data.values()], dtype=float)
        stats[method] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "n_samples": int(values.size),
        }
    return stats
