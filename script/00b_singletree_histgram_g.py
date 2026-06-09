from pathlib import Path

import numpy as np
import laspy
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================
PROJECT_DIR = Path(r"H:\Python Project\TreeExtraction")
INPUT_LAS = PROJECT_DIR / "singletreeexample" / "0_2_4_60_2510.las"
OUTPUT_FIG = PROJECT_DIR / "singletreeexample" / "single_tree_vertical_profile.png"

BIN_HEIGHT = 0.2
NORM_BINS = 20

# Style based on the reference code you provided.
TITLE_SIZE = 22
LABEL_SIZE = 20
TICK_SIZE = 16
LEGEND_SIZE = 15
SPINE_WIDTH = 2.0
LINE_WIDTH = 2.4
GRID_ALPHA = 0.35

HIST_FACE_COLOR = "#8FA8B8"
HIST_EDGE_COLOR = "black"
HIST_ALPHA = 0.35
PROFILE_COLOR = "#4C78A8"


# =========================
# Utility functions
# =========================
def load_las_points(las_path: Path) -> np.ndarray:
    """Read all points from a LAS file and return the z elevation array."""
    if not las_path.exists():
        raise FileNotFoundError(f"找不到 LAS 文件：{las_path}")

    las = laspy.read(las_path)
    z = np.asarray(las.z, dtype=float)

    if z.size == 0:
        raise ValueError("LAS 文件中没有点。")

    z = z[np.isfinite(z)]
    if z.size == 0:
        raise ValueError("LAS 文件中没有有效高程点。")

    return z


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
    if std <= 0:
        return np.zeros_like(grid, dtype=float)

    # Silverman's rule of thumb
    bandwidth = 1.06 * std * (n ** (-1 / 5))
    if bandwidth <= 0:
        return np.zeros_like(grid, dtype=float)

    diff = (grid[:, None] - x[None, :]) / bandwidth
    density = np.exp(-0.5 * diff ** 2).sum(axis=1) / (n * bandwidth * np.sqrt(2 * np.pi))
    return density


def normalize_height(z: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Normalize tree height to [0, 1].
    Use min-max normalization, which is intuitive for a single tree.
    Return normalized_z, z_min, and z_max.
    """
    z_min = float(np.min(z))
    z_max = float(np.max(z))

    if np.isclose(z_max, z_min):
        raise ValueError("该单株树点云高度没有变化，无法归一化。")

    norm_z = (z - z_min) / (z_max - z_min)
    return norm_z, z_min, z_max


def compute_vertical_histogram(z: np.ndarray, bin_height: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin points by real height and return:
    counts: point count in each layer
    edges: bin boundaries
    centers: bin centers
    """
    z_min = np.floor(np.min(z) / bin_height) * bin_height
    z_max = np.ceil(np.max(z) / bin_height) * bin_height
    edges = np.arange(z_min, z_max + bin_height, bin_height)

    counts, edges = np.histogram(z, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    return counts, edges, centers


def compute_normalized_profile(norm_z: np.ndarray, n_bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Split normalized height into 20 bins from maximum to minimum.
    Return:
    profile: relative frequency in each bin
    centers_desc: bin centers from top to bottom (1 -> 0)
    """
    # First bin from 0 to 1.
    edges = np.linspace(0, 1, n_bins + 1)
    counts, _ = np.histogram(norm_z, bins=edges)

    # Convert to relative frequency.
    profile = counts / counts.sum() if counts.sum() > 0 else counts.astype(float)

    centers = (edges[:-1] + edges[1:]) / 2

    # User requested 20 bins from maximum to minimum.
    profile_desc = profile[::-1]
    centers_desc = centers[::-1]

    return profile_desc, centers_desc


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
    ax.grid(linestyle="--", alpha=GRID_ALPHA)


# =========================
# Plotting functions
# =========================
def plot_single_tree_profiles(z: np.ndarray, output_fig: Path) -> None:
    norm_z, z_min, z_max = normalize_height(z)

    counts, edges, centers = compute_vertical_histogram(z, BIN_HEIGHT)
    profile_desc, centers_desc = compute_normalized_profile(norm_z, NORM_BINS)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 8),
        dpi=300,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.05}
    )

    # -------------------------
    # Subplot 1: real-height histogram with fitted curve.
    # -------------------------
    ax1 = axes[0]

    # Horizontal histogram: tree height is on the y-axis.
    ax1.hist(
        z,
        bins=edges,
        orientation="horizontal",
        color=HIST_FACE_COLOR,
        edgecolor=HIST_EDGE_COLOR,
        linewidth=1.2,
        alpha=HIST_ALPHA,
        label=f"Histogram ({BIN_HEIGHT:.1f} m bins)"
    )

    # KDE, scaled to the histogram count scale.
    y_grid = np.linspace(edges[0], edges[-1], 400)
    kde_density = gaussian_kde_manual(z, y_grid)
    kde_scaled = kde_density * len(z) * BIN_HEIGHT
    ax1.plot(
        kde_scaled,
        y_grid,
        color=PROFILE_COLOR,
        linewidth=LINE_WIDTH,
        label="Fitted curve"
    )

    ax1.set_title("a)", fontsize=TITLE_SIZE, fontweight="bold", pad=10)
    ax1.set_xlabel("Frequency", fontsize=LABEL_SIZE, fontweight="bold")
    ax1.set_ylabel("Tree height (m)", fontsize=LABEL_SIZE, fontweight="bold")

    style_axis(ax1)

    legend1 = ax1.legend(loc="upper right", fontsize=LEGEND_SIZE, frameon=True)
    legend1.get_frame().set_alpha(1.0)
    legend1.get_frame().set_facecolor("white")

    # -------------------------
    # Subplot 2: 20-bin normalized vertical profile.
    # -------------------------
    ax2 = axes[1]

    ax2.plot(
        profile_desc,
        centers_desc,
        color=PROFILE_COLOR,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=5.5,
        label="20-bin profile"
    )

    ax2.set_title("b)", fontsize=TITLE_SIZE, fontweight="bold", pad=10)
    ax2.set_xlabel("Relative return frequency", fontsize=LABEL_SIZE, fontweight="bold")
    ax2.set_ylabel("Normalized tree height", fontsize=LABEL_SIZE, fontweight="bold")
    ax2.set_ylim(0, 1)

    style_axis(ax2)

    legend2 = ax2.legend(loc="upper right", fontsize=LEGEND_SIZE, frameon=True)
    legend2.get_frame().set_alpha(1.0)
    legend2.get_frame().set_facecolor("white")

    # Save.
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"LAS file: {INPUT_LAS}")
    print(f"Total points: {len(z)}")
    print(f"Min height: {z_min:.3f} m")
    print(f"Max height: {z_max:.3f} m")
    print(f"Tree height range: {z_max - z_min:.3f} m")
    print(f"Figure saved to: {output_fig}")


# =========================
# Main program
# =========================
def main() -> None:
    z = load_las_points(INPUT_LAS)
    plot_single_tree_profiles(z, OUTPUT_FIG)


if __name__ == "__main__":
    main()
