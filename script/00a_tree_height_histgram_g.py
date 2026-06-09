from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================
PROJECT_DIR = Path(r"H:\Python Project\TreeExtraction")
INPUT_CSV = PROJECT_DIR / "processed_height" / "tree_features_height.csv"
OUTPUT_FIG = PROJECT_DIR / "processed_height" / "fig_tree_height_histogram_fieldwise_60m.png"

FLIGHT_HEIGHT = 20
MIN_TREE_HEIGHT = 0.5
FIELDS_TO_PLOT = [1, 2, 3]
BIN_WIDTH = 0.2

# Style based on the reference code you provided.
TITLE_SIZE = 24
LABEL_SIZE = 22
TICK_SIZE = 18
LEGEND_SIZE = 16
SPINE_WIDTH = 2.0

BAR_FACE_COLOR = "#8FA8B8"
BAR_EDGE_COLOR = "black"
BAR_ALPHA = 0.28
GRID_ALPHA = 0.35


# =========================
# Utility functions
# =========================
def ensure_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"输入表缺少必要列：{missing}")


def load_and_filter_data(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_csv}")

    df = pd.read_csv(input_csv)

    required_cols = ["flight_height", "Tree Height", "Field"]
    ensure_required_columns(df, required_cols)

    df["flight_height"] = pd.to_numeric(df["flight_height"], errors="coerce")
    df["Tree Height"] = pd.to_numeric(df["Tree Height"], errors="coerce")
    df["Field"] = pd.to_numeric(df["Field"], errors="coerce")

    df = df.dropna(subset=["flight_height", "Tree Height", "Field"]).copy()

    # Keep records with flight_height = 20.
    df = df[df["flight_height"] == FLIGHT_HEIGHT].copy()

    # Remove trees below 0.5 m.
    df = df[df["Tree Height"] >= MIN_TREE_HEIGHT].copy()

    # Keep only Field 1, 2, and 3.
    df = df[df["Field"].isin(FIELDS_TO_PLOT)].copy()
    df["Field"] = df["Field"].astype(int)

    if df.empty:
        raise ValueError("筛选后无有效数据，请检查输入表内容。")

    return df


def compute_bins(df: pd.DataFrame, value_col: str, bin_width: float) -> np.ndarray:
    vmin = np.floor(df[value_col].min() / bin_width) * bin_width
    vmax = np.ceil(df[value_col].max() / bin_width) * bin_width
    return np.arange(vmin, vmax + bin_width, bin_width)


def gaussian_kde_manual(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Manually implement Gaussian KDE to avoid using seaborn or scipy.
    Return probability density values.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n < 2:
        return np.zeros_like(grid, dtype=float)

    std = np.std(x, ddof=1)
    if std == 0:
        return np.zeros_like(grid, dtype=float)

    # Silverman's rule of thumb
    bandwidth = 1.06 * std * (n ** (-1 / 5))
    if bandwidth <= 0:
        return np.zeros_like(grid, dtype=float)

    diff = (grid[:, None] - x[None, :]) / bandwidth
    density = np.exp(-0.5 * diff ** 2).sum(axis=1) / (n * bandwidth * np.sqrt(2 * np.pi))
    return density


def style_axis(ax: plt.Axes, show_ylabel: bool) -> None:
    if show_ylabel:
        ax.set_ylabel("Frequency", fontsize=LABEL_SIZE, fontweight="bold")
        ax.tick_params(axis="y", labelsize=TICK_SIZE)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    ax.set_xlabel("Tree Height (m)", fontsize=LABEL_SIZE, fontweight="bold")
    ax.tick_params(axis="x", labelsize=TICK_SIZE)

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)

    ax.grid(axis="y", linestyle="--", alpha=GRID_ALPHA)


def plot_fieldwise_tree_height_histogram(df: pd.DataFrame, output_fig: Path) -> None:
    bins = compute_bins(df, "Tree Height", BIN_WIDTH)
    panel_labels = ["a)", "b)", "c)"]

    # Use a three-panel layout consistent with the reference code.
    fig, axes = plt.subplots(
        1, 3,
        figsize=(20, 7.2),
        dpi=300,
        sharex=True,
        sharey=True,
        squeeze=False
    )
    axes = axes[0]

    for i, field in enumerate(FIELDS_TO_PLOT):
        ax = axes[i]
        sub = df[df["Field"] == field].copy()

        if sub.empty:
            ax.set_title(
                f"{panel_labels[i]} Field {field}",
                fontsize=TITLE_SIZE,
                fontweight="bold",
                pad=12
            )
            ax.text(
                0.5, 0.5, "No data",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=LABEL_SIZE
            )
            style_axis(ax, show_ylabel=(i == 0))
            continue

        heights = sub["Tree Height"].to_numpy(dtype=float)

        mean_h = heights.mean()
        median_h = np.median(heights)
        max_h = heights.max()
        min_h = heights.min()

        # Histogram
        counts, hist_bins, _ = ax.hist(
            heights,
            bins=bins,
            color=BAR_FACE_COLOR,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=1.2,
            alpha=BAR_ALPHA
        )

        # Manual KDE, scaled to histogram frequency
        x_grid = np.linspace(bins[0], bins[-1], 400)
        kde_density = gaussian_kde_manual(heights, x_grid)
        kde_scaled = kde_density * len(heights) * BIN_WIDTH
        ax.plot(x_grid, kde_scaled, color="#4C78A8", linewidth=2.2)

        # Statistic lines
        ax.axvline(mean_h, color="orange", linestyle="dashed", linewidth=2.0,
                   label=f"Mean: {mean_h:.2f} m")
        ax.axvline(median_h, color="green", linestyle="dashed", linewidth=2.0,
                   label=f"Median: {median_h:.2f} m")
        ax.axvline(max_h, color="purple", linestyle="dashed", linewidth=2.0,
                   label=f"Max: {max_h:.2f} m")
        ax.axvline(min_h, color="red", linestyle="dashed", linewidth=2.0,
                   label=f"Min: {min_h:.2f} m")

        ax.set_title(
            f"{panel_labels[i]} Field {field}",
            fontsize=TITLE_SIZE,
            fontweight="bold",
            pad=12
        )

        style_axis(ax, show_ylabel=(i == 0))

        legend = ax.legend(
            fontsize=LEGEND_SIZE,
            frameon=True,
            loc="upper right"
        )
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_facecolor("white")

        ymax = max([ax.get_ylim()[1] for ax in axes])
        for ax in axes:
            ax.set_ylim(0, ymax * 1.07)    

    axes[0].set_ylim(bottom=0)

    plt.tight_layout(rect=[0.02, 0.02, 0.995, 0.98], w_pad=2.2)
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved to: {output_fig}")


# =========================
# Main program
# =========================
def main() -> None:
    print(f"读取输入表：{INPUT_CSV}")
    df = load_and_filter_data(INPUT_CSV)

    print("\n筛选后样本数：")
    print(df.groupby("Field")["Tree Height"].count())

    plot_fieldwise_tree_height_histogram(df, OUTPUT_FIG)


if __name__ == "__main__":
    main()
