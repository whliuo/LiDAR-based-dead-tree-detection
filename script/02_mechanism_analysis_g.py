import math
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


# =========================
# Configuration
# Directory structure:
# Root directory/
#   ├─ script/
#   │   └─ current script.py
#   ├─ processed/
#   │   ├─ tree_features.csv
#   │   └─ results_mechanism/
#   └─ segtree/
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

PROCESSED_DIR = ROOT_DIR / "new_processed"
INPUT_CSV = PROCESSED_DIR / "tree_features.csv"

OUTPUT_DIR = PROCESSED_DIR / "results_mechanism"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_DEAD = 0
CLASS_ALIVE = {1}

PROFILE_COLS = [f"p{i}" for i in range(1, 21)]
MAIN_FEATURES = ["rHDP", "rVRC", "UCRR", "LCRR", "VRI", "VE"]
##TABLE1_FEATURES = MAIN_FEATURES + ["skewness", "kurtosisr", "EOHR", "Tree Height"]
TABLE1_FEATURES = MAIN_FEATURES

# Figure 3 colors
DEAD_COLOR = "#8FA8B8"
ALIVE_COLOR = "#A9B8A1"
ALPHA_FILL = 0.45


# =========================
# Utility functions
# =========================
def label_class(class_value: int) -> str:
    if class_value == CLASS_DEAD:
        return "dead"
    elif class_value in CLASS_ALIVE:
        return "non-dead"
    else:
        return "other"


def ensure_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"输入表缺少必要列：{missing}")


def compute_median_iqr(x: np.ndarray) -> str:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return ""
    median = np.median(x)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    return f"{median:.4f} ± {iqr:.4f}"


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta
    The return value is approximately in the range [-1, 1].
    Positive values indicate that x tends to be greater than y.
    Negative values indicate that x tends to be less than y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if x.size == 0 or y.size == 0:
        return np.nan

    diff = x[:, None] - y[None, :]
    gt = np.sum(diff > 0)
    lt = np.sum(diff < 0)
    n = x.size * y.size
    return float((gt - lt) / n)


def direction_in_dead(dead_vals: np.ndarray, alive_vals: np.ndarray) -> str:
    dead_med = np.nanmedian(dead_vals)
    alive_med = np.nanmedian(alive_vals)
    if dead_med > alive_med:
        return "Higher"
    elif dead_med < alive_med:
        return "Lower"
    return "Equal"


def significance_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


# =========================
# Load data
# =========================
def load_data(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_csv}")

    required_cols = ["ID", "Class", "Field"] + PROFILE_COLS + TABLE1_FEATURES
    df = pd.read_csv(input_csv)
    ensure_required_columns(df, required_cols)

    df["status"] = df["Class"].apply(label_class)
    df = df[df["status"].isin(["dead", "non-dead"])].copy()

    if df.empty:
        raise ValueError("过滤后没有 dead / non-dead 样本。请检查 Class 编码。")

    return df


# =========================
# Figure 1a / 1b: overall normalized vertical profiles
# Use vertical profiles consistently.
# mode = "SE" or "STD"
# =========================

def plot_overall_profiles(df: pd.DataFrame, output_path: Path, mode: str = "SE") -> None:
    dead_df = df[df["status"] == "dead"]
    alive_df = df[df["status"] == "non-dead"]

    dead_mean = dead_df[PROFILE_COLS].mean(axis=0).to_numpy(dtype=float)
    alive_mean = alive_df[PROFILE_COLS].mean(axis=0).to_numpy(dtype=float)

    dead_std = dead_df[PROFILE_COLS].std(axis=0).to_numpy(dtype=float)
    alive_std = alive_df[PROFILE_COLS].std(axis=0).to_numpy(dtype=float)

    dead_n = max(len(dead_df), 1)
    alive_n = max(len(alive_df), 1)

    mode_upper = mode.upper()
    if mode_upper == "SE":
        dead_band = dead_std / np.sqrt(dead_n)
        alive_band = alive_std / np.sqrt(alive_n)
        dead_label = "Dead (mean ± SE)"
        alive_label = "Non-dead (mean ± SE)"
        title = "Normalized vertical profiles (mean ± SE)"
    elif mode_upper == "STD":
        dead_band = dead_std
        alive_band = alive_std
        dead_label = "Dead (mean ± SD)"
        alive_label = "Non-dead (mean ± SD)"
        title = "Normalized vertical profiles (mean ± SD)"
    else:
        raise ValueError("mode 必须是 'SE' 或 'STD'")

    y = np.linspace(0.025, 0.975, 20)

    plt.figure(figsize=(7, 9))
    plt.plot(dead_mean, y, linewidth=2.6, label=dead_label)
    plt.plot(alive_mean, y, linewidth=2.6, label=alive_label)

    plt.fill_betweenx(y, dead_mean - dead_band, dead_mean + dead_band, alpha=0.2)
    plt.fill_betweenx(y, alive_mean - alive_band, alive_mean + alive_band, alpha=0.2)

    plt.xlabel("Mean relative return proportion", fontsize=18, fontweight="bold")
    plt.ylabel("Relative tree height", fontsize=18, fontweight="bold")
    plt.title(title, fontsize=22, fontweight="bold", pad=18)
    plt.ylim(0, 1)

    if mode_upper == "STD":
        ax = plt.gca()
        ax.set_xlim(-0.02, 0.4)
        ax.set_xticks(np.arange(0, 0.41, 0.1))
        ax.set_xticklabels([f"{v:.1f}" if v > 0 else "0" for v in np.arange(0, 0.41, 0.1)], fontsize=18)
    else:
        plt.xticks(fontsize=17)

    plt.yticks(fontsize=17)
    plt.legend(fontsize=16)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Figure 1c: field-wise normalized vertical profiles
# Use vertical profiles consistently.
# mean ± STD
# =========================

def plot_fieldwise_profiles(df: pd.DataFrame, output_path: Path) -> None:
    fields = sorted(df["Field"].dropna().unique().tolist())
    n_fields = len(fields)

    if n_fields == 0:
        return

    fig, axes = plt.subplots(1, n_fields, figsize=(7 * n_fields, 7.2), squeeze=False)
    axes = axes[0]

    y = np.linspace(0.025, 0.975, 20)

    panel_labels = ["a)", "b)", "c)"]

    for i, (ax, field) in enumerate(zip(axes, fields)):
        sub = df[df["Field"] == field]
        dead_df = sub[sub["status"] == "dead"]
        alive_df = sub[sub["status"] == "non-dead"]

        if len(dead_df) > 0:
            dead_mean = dead_df[PROFILE_COLS].mean(axis=0).to_numpy(dtype=float)
            dead_std = dead_df[PROFILE_COLS].std(axis=0).to_numpy(dtype=float)
            ax.plot(dead_mean, y, linewidth=2.6, label="Dead (mean ± STD)")
            ax.fill_betweenx(y, dead_mean - dead_std, dead_mean + dead_std, alpha=0.2)

        if len(alive_df) > 0:
            alive_mean = alive_df[PROFILE_COLS].mean(axis=0).to_numpy(dtype=float)
            alive_std = alive_df[PROFILE_COLS].std(axis=0).to_numpy(dtype=float)
            ax.plot(alive_mean, y, linewidth=2.6, label="Non-dead (mean ± STD)")
            ax.fill_betweenx(y, alive_mean - alive_std, alive_mean + alive_std, alpha=0.2)

        ax.set_title(f"{panel_labels[i]} Field {field}", fontsize=21, fontweight="bold", pad=12)

        # Keep the y-axis label and tick labels only on the first subplot.
        if i == 0:
            ax.set_ylabel("Relative tree height", fontsize=18, fontweight="bold")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

        ax.set_ylim(0, 1)

        ax.set_xlim(-0.02, 0.55)
        ax.set_xticks(np.arange(0, 0.51, 0.1))
        ax.set_xticklabels([f"{v:.1f}" if v > 0 else "0" for v in np.arange(0, 0.51, 0.1)])

        ax.tick_params(axis="both", labelsize=16)
        ax.legend(fontsize=14)

        for spine in ax.spines.values():
            spine.set_linewidth(2.0)

    # Leave space for the shared x-axis label at the bottom.
    fig.subplots_adjust(wspace=0.1, bottom=0.14)

    # Shared x-axis label for the whole figure.
    fig.text(
        0.5, 0.07,
        "Mean relative return proportion",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# =========================
# Table 1: initial version
# =========================
def build_mechanism_table(df: pd.DataFrame, feature_cols: List[str], output_csv: Path) -> pd.DataFrame:
    dead_df = df[df["status"] == "dead"]
    alive_df = df[df["status"] == "non-dead"]

    rows: List[Dict] = []

    for feat in feature_cols:
        dead_vals = dead_df[feat].to_numpy(dtype=float)
        alive_vals = alive_df[feat].to_numpy(dtype=float)

        dead_vals = dead_vals[np.isfinite(dead_vals)]
        alive_vals = alive_vals[np.isfinite(alive_vals)]

        if len(dead_vals) == 0 or len(alive_vals) == 0:
            row = {
                "Feature": feat,
                "Dead (median ± IQR)": "",
                "Non-dead (median ± IQR)": "",
                "Mann-Whitney U": np.nan,
                "p-value": np.nan,
                "Cliffs delta": np.nan,
                "Direction in dead trees": "",
            }
        else:
            u_stat, p_val = mannwhitneyu(dead_vals, alive_vals, alternative="two-sided")
            cd = cliffs_delta(dead_vals, alive_vals)
            direction = direction_in_dead(dead_vals, alive_vals)

            row = {
                "Feature": feat,
                "Dead (median ± IQR)": compute_median_iqr(dead_vals),
                "Non-dead (median ± IQR)": compute_median_iqr(alive_vals),
                "Mann-Whitney U": float(u_stat),
                "p-value": float(p_val),
                "Cliffs delta": float(cd),
                "Direction in dead trees": direction,
            }

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return out_df


# =========================
# Figure 2: feature boxplots
# =========================
def plot_feature_boxplots(df: pd.DataFrame, feature_cols: List[str], output_path: Path) -> None:
    n = len(feature_cols)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.6 * nrows), squeeze=False)
    axes = axes.flatten()

    dead_df = df[df["status"] == "dead"]
    alive_df = df[df["status"] == "non-dead"]

    rng = np.random.default_rng(42)
    panel_labels = ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)", "i)"]

    for i, feat in enumerate(feature_cols):
        ax = axes[i]

        dead_vals = dead_df[feat].to_numpy(dtype=float)
        alive_vals = alive_df[feat].to_numpy(dtype=float)

        dead_vals = dead_vals[np.isfinite(dead_vals)]
        alive_vals = alive_vals[np.isfinite(alive_vals)]

        data = [dead_vals, alive_vals]

        ax.boxplot(
            data,
            labels=["Dead", "Non-dead"],
            widths=0.5,
            showfliers=False,
            boxprops=dict(linewidth=2.5),
            whiskerprops=dict(linewidth=2.2),
            capprops=dict(linewidth=2.2),
            medianprops=dict(linewidth=3.0),
        )

        if len(dead_vals) > 0:
            x_dead = 1 + rng.normal(0, 0.04, size=len(dead_vals))
            ax.scatter(x_dead, dead_vals, alpha=0.45, s=24)

        if len(alive_vals) > 0:
            x_alive = 2 + rng.normal(0, 0.04, size=len(alive_vals))
            ax.scatter(x_alive, alive_vals, alpha=0.35, s=24)

        if len(dead_vals) > 0 and len(alive_vals) > 0:
            try:
                _, p_val = mannwhitneyu(dead_vals, alive_vals, alternative="two-sided")
                stars = significance_stars(float(p_val))
                if p_val >= 0.001:
                    p_text = f"p = {p_val:.3g} ({stars})"
                else:
                    p_text = f"p < 0.001 ({stars})"
                ax.text(0.03, 0.96, p_text, transform=ax.transAxes, ha="left", va="top", fontsize=15)
            except Exception:
                pass

        title_prefix = panel_labels[i] if i < len(panel_labels) else ""
        ax.set_title(f"{title_prefix} {feat}", fontsize=22, pad=12, fontweight="bold")
        ax.set_ylabel("Feature value", fontsize=20)
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)

        for spine in ax.spines.values():
            spine.set_linewidth(2.0)

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(h_pad=2.5, w_pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Figure 3: feature violin plots
# =========================
def plot_feature_violins(df: pd.DataFrame, feature_cols: List[str], output_path: Path) -> None:
    n = len(feature_cols)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.6 * nrows), squeeze=False)
    axes = axes.flatten()

    dead_df = df[df["status"] == "dead"]
    alive_df = df[df["status"] == "non-dead"]

    panel_labels = ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)", "i)"]

    for i, feat in enumerate(feature_cols):
        ax = axes[i]

        dead_vals = dead_df[feat].to_numpy(dtype=float)
        alive_vals = alive_df[feat].to_numpy(dtype=float)

        dead_vals = dead_vals[np.isfinite(dead_vals)]
        alive_vals = alive_vals[np.isfinite(alive_vals)]

        data = [dead_vals, alive_vals]

        if len(dead_vals) > 0 or len(alive_vals) > 0:
            parts = ax.violinplot(
                data,
                positions=[1, 2],
                widths=0.7,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )

            violin_colors = [DEAD_COLOR, ALIVE_COLOR]
            for body, color in zip(parts["bodies"], violin_colors):
                body.set_facecolor(color)
                body.set_edgecolor("black")
                body.set_alpha(ALPHA_FILL)
                body.set_linewidth(2.0)

            if "cmedians" in parts:
                parts["cmedians"].set_color("black")
                parts["cmedians"].set_linewidth(3.0)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Dead", "Non-dead"], fontsize=20)

        if len(dead_vals) > 0 and len(alive_vals) > 0:
            try:
                _, p_val = mannwhitneyu(dead_vals, alive_vals, alternative="two-sided")
                stars = significance_stars(float(p_val))
                if p_val >= 0.001:
                    p_text = f"p = {p_val:.3g} ({stars})"
                else:
                    p_text = f"p < 0.001 ({stars})"
                ax.text(0.03, 0.96, p_text, transform=ax.transAxes, ha="left", va="top", fontsize=15)
            except Exception:
                pass

        title_prefix = panel_labels[i] if i < len(panel_labels) else ""
        ax.set_title(f"{title_prefix} {feat}", fontsize=22, pad=12, fontweight="bold")
        ax.set_ylabel("Feature value", fontsize=20, fontweight="bold")
        ax.tick_params(axis="y", labelsize=20)

        for spine in ax.spines.values():
            spine.set_linewidth(2.0)

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Main program
# =========================
def main():
    print(f"脚本目录：{SCRIPT_DIR}")
    print(f"根目录：{ROOT_DIR}")
    print(f"输入表：{INPUT_CSV}")
    print(f"输出目录：{OUTPUT_DIR}")

    df = load_data(INPUT_CSV)

    print("\n样本统计：")
    print(df["status"].value_counts())

    print("\nField × status：")
    print(pd.crosstab(df["Field"], df["status"]))

    # Table 1
    table1_path = OUTPUT_DIR / "table1_mechanism.csv"
    table1_df = build_mechanism_table(df, TABLE1_FEATURES, table1_path)
    print(f"\nTable 1 已保存：{table1_path}")
    print(table1_df)

    # Figure 1a: mean ± SE
    fig1a_path = OUTPUT_DIR / "fig1a_overall_profile_SE.png"
    plot_overall_profiles(df, fig1a_path, mode="SE")
    print(f"Figure 1a 已保存：{fig1a_path}")

    # Figure 1b: mean ± STD
    fig1b_path = OUTPUT_DIR / "fig1b_overall_profile_STD.png"
    plot_overall_profiles(df, fig1b_path, mode="STD")
    print(f"Figure 1b 已保存：{fig1b_path}")

    # Figure 1c: field-wise mean ± STD
    fig1c_path = OUTPUT_DIR / "fig1c_fieldwise_profile_STD.png"
    plot_fieldwise_profiles(df, fig1c_path)
    print(f"Figure 1c 已保存：{fig1c_path}")

    # Figure 2
    fig2_path = OUTPUT_DIR / "fig2_feature_boxplots.png"
    plot_feature_boxplots(df, MAIN_FEATURES, fig2_path)
    print(f"Figure 2 已保存：{fig2_path}")

    # Figure 3
    fig3_path = OUTPUT_DIR / "fig3_feature_violins.png"
    plot_feature_violins(df, MAIN_FEATURES, fig3_path)
    print(f"Figure 3 已保存：{fig3_path}")

    print("\n机制分析完成。")


if __name__ == "__main__":
    main()
