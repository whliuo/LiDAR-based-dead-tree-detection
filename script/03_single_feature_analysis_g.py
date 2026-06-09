from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold


# =========================
# Configuration
# =========================
PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_DIR / "new_processed" / "tree_features.csv"
OUTPUT_DIR = PROJECT_DIR / "new_processed" / "single_feature"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_DEAD = 0
CLASS_ALIVE = {1}

FEATURES = [
    "rHDP",
    "rVRC",
    "UCRR",
    "LCRR",
    "VRI",
    "VE",
]

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10
TOP_N_ROC = len(FEATURES)   # Figure 4 shows all six features

# =========================
# Shared plotting style parameters
# =========================
TITLE_SIZE = 18
LABEL_SIZE = 15
XTICK_SIZE = 14
YTICK_SIZE = 14
ANNOT_SIZE = 12   # Increase value labels above bars by one size
LEGEND_SIZE = 11

SPINE_LINEWIDTH = 2.0
BAR_EDGE_LINEWIDTH = 1.8
LINEWIDTH_MAIN = 2.5
LINEWIDTH_REF = 1.2   # Keep Figure 4's 1:1 reference line lightweight

# Figure 3 low-saturation multi-color bar palette
BAR_COLORS_SOFT = [
    "#B7CFEA",  # softer blue
    "#F6CFAB",  # softer orange
    "#BED7BF",  # softer green
    "#E3C4D3",  # softer pink-purple
    "#D8CBEF",  # softer lavender
    "#EEE1AE",  # softer yellow
    "#C5E0DE",  # softer cyan
    "#E3CDBB",  # softer brown

]

MAIN_BLUE = "#1f77b4"
MAIN_ORANGE = "#ff7f0e"


# =========================
# Utility functions
# =========================
def label_class(class_value: int) -> str:
    if class_value == CLASS_DEAD:
        return "dead"
    elif class_value in CLASS_ALIVE:
        return "non-dead"
    return "other"


def ensure_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"输入表缺少必要列：{missing}")


def style_axes(ax, xlabel: str = None, ylabel: str = None, title: str = None) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=10)

    ax.tick_params(axis="x", labelsize=XTICK_SIZE, rotation=0)
    ax.tick_params(axis="y", labelsize=YTICK_SIZE)

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINEWIDTH)


def compute_metrics_from_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Return sensitivity and specificity.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return float(sensitivity), float(specificity)


def find_best_threshold_by_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Select the best threshold on training probabilities using Youden's index.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    return float(thresholds[best_idx])


def threshold_from_original_feature(
    x_train: np.ndarray,
    y_train: np.ndarray,
    positive_if_greater: bool
) -> Tuple[float, float, float, float]:
    """
    Scan thresholds on the original training feature values and return:
    best_threshold, best_sensitivity, best_specificity, best_youden

    This is mainly used to report interpretable original-feature thresholds.
    """
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)

    valid = np.isfinite(x_train)
    x_train = x_train[valid]
    y_train = y_train[valid]

    unique_vals = np.unique(x_train)
    if unique_vals.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    if unique_vals.size == 1:
        return float(unique_vals[0]), np.nan, np.nan, np.nan

    best_thr = np.nan
    best_sens = np.nan
    best_spec = np.nan
    best_j = -np.inf

    for thr in unique_vals:
        if positive_if_greater:
            y_pred = (x_train >= thr).astype(int)
        else:
            y_pred = (x_train <= thr).astype(int)

        sens, spec = compute_metrics_from_confusion(y_train, y_pred)
        if np.isnan(sens) or np.isnan(spec):
            continue

        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_thr = float(thr)
            best_sens = float(sens)
            best_spec = float(spec)

    return best_thr, best_sens, best_spec, best_j


# =========================
# Load data
# =========================
def load_data(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_csv}")

    df = pd.read_csv(input_csv)
    ensure_required_columns(df, ["ID", "Class"] + FEATURES)

    df["status"] = df["Class"].apply(label_class)
    df = df[df["status"].isin(["dead", "non-dead"])].copy()

    if df.empty:
        raise ValueError("过滤后没有 dead / non-dead 样本，请检查 Class 编码。")

    df["y"] = (df["status"] == "dead").astype(int)
    return df


# =========================
# Strict cross-validated single-feature analysis
# =========================
def cross_validated_single_feature_analysis(df: pd.DataFrame, feature: str) -> Dict:
    sub = df[[feature, "y"]].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()

    if sub.empty or sub["y"].nunique() < 2:
        return {
            "Feature": feature,
            "Logistic coefficient sign": "",
            "Mean CV AUC": np.nan,
            "Mean CV Sensitivity": np.nan,
            "Mean CV Specificity": np.nan,
            "Mean CV Youden threshold (feature)": np.nan,
            "Mean CV Youden index": np.nan,
            "roc_mean_fpr": None,
            "roc_mean_tpr": None,
        }

    X = sub[[feature]].to_numpy(dtype=float)
    y = sub["y"].to_numpy(dtype=int)

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    )

    aucs = []
    sens_list = []
    spec_list = []
    thr_list = []
    youden_list = []
    coef_signs = []

    mean_fpr = np.linspace(0, 1, 200)
    interp_tprs = []

    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

        coef = float(model.coef_[0][0])
        coef_signs.append(coef)

        y_train_prob = model.predict_proba(X_train)[:, 1]
        best_prob_thr = find_best_threshold_by_youden(y_train, y_train_prob)

        y_test_prob = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_prob >= best_prob_thr).astype(int)

        if len(np.unique(y_test)) == 2:
            auc_val = roc_auc_score(y_test, y_test_prob)
            aucs.append(float(auc_val))

            fpr, tpr, _ = roc_curve(y_test, y_test_prob)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)

        sens, spec = compute_metrics_from_confusion(y_test, y_test_pred)
        sens_list.append(sens)
        spec_list.append(spec)

        positive_if_greater = coef > 0
        feat_thr, _, _, feat_youden = threshold_from_original_feature(
            X_train.ravel(),
            y_train,
            positive_if_greater=positive_if_greater
        )
        thr_list.append(feat_thr)
        youden_list.append(feat_youden)

    coef_mean = np.nanmean(coef_signs)
    coef_sign = "+" if coef_mean > 0 else "-" if coef_mean < 0 else "0"

    if len(interp_tprs) > 0:
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[-1] = 1.0
    else:
        mean_tpr = None

    return {
        "Feature": feature,
        "Logistic coefficient sign": coef_sign,
        "Mean CV AUC": float(np.nanmean(aucs)) if len(aucs) > 0 else np.nan,
        "Mean CV Sensitivity": float(np.nanmean(sens_list)) if len(sens_list) > 0 else np.nan,
        "Mean CV Specificity": float(np.nanmean(spec_list)) if len(spec_list) > 0 else np.nan,
        "Mean CV Youden threshold (feature)": float(np.nanmean(thr_list)) if len(thr_list) > 0 else np.nan,
        "Mean CV Youden index": float(np.nanmean(youden_list)) if len(youden_list) > 0 else np.nan,
        "roc_mean_fpr": mean_fpr,
        "roc_mean_tpr": mean_tpr,
    }


# =========================
# Figure 3: CV AUC ranking
# Low-saturation multi-color bars with larger value labels.
# =========================
def plot_auc_ranking(results_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = results_df.copy()
    plot_df = plot_df[np.isfinite(plot_df["Mean CV AUC"])].sort_values("Mean CV AUC", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5.6))

    x = np.arange(len(plot_df))
    y = plot_df["Mean CV AUC"].to_numpy(dtype=float)

    bar_colors = [BAR_COLORS_SOFT[i % len(BAR_COLORS_SOFT)] for i in range(len(plot_df))]
    bars = ax.bar(
        x,
        y,
        width=0.5,
        color=bar_colors,
        edgecolor="black",
        linewidth=BAR_EDGE_LINEWIDTH,
        alpha=0.88
    )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=LINEWIDTH_REF)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Feature"].tolist(), rotation=0)

    style_axes(
        ax,
        ylabel="Mean CV AUC",
        title="Cross-validated single-feature discrimination ranking"
    )

    for rect, v in zip(bars, y):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            v + 0.01,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=ANNOT_SIZE,
            fontweight="normal"
        )

    ax.set_ylim(0, min(1.05, max(0.7, y.max() + 0.08)))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Figure 4: mean ROC curves, showing all six features.
# Increase x-axis and y-axis font sizes by one size.
# =========================
def plot_top_feature_rocs(results: List[Dict], output_path: Path, top_n: int = 6) -> None:
    valid_results = [
        r for r in results
        if r["roc_mean_fpr"] is not None and r["roc_mean_tpr"] is not None and np.isfinite(r["Mean CV AUC"])
    ]
    valid_results = sorted(valid_results, key=lambda r: r["Mean CV AUC"], reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(7.4, 6.2))

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, r in enumerate(valid_results):
        color = default_colors[i % len(default_colors)]
        ax.plot(
            r["roc_mean_fpr"],
            r["roc_mean_tpr"],
            linewidth=LINEWIDTH_MAIN,
            color=color,
            label=f'{r["Feature"]} (Mean CV AUC = {r["Mean CV AUC"]:.3f})'
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.0,
        color="black"
    )

    # Increase axis font sizes separately for Figure 4.
    ax.set_xlabel("False Positive Rate", fontsize=LABEL_SIZE + 1)
    ax.set_ylabel("True Positive Rate", fontsize=LABEL_SIZE + 1)
    ax.set_title(
        "Cross-validated ROC curves of six individual features",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=10
    )

    ax.tick_params(axis="x", labelsize=XTICK_SIZE, rotation=0)
    ax.tick_params(axis="y", labelsize=YTICK_SIZE)

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINEWIDTH)

    ax.legend(fontsize=LEGEND_SIZE, frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Main program
# =========================
def main():
    print(f"读取输入表：{INPUT_CSV}")
    df = load_data(INPUT_CSV)

    print("\n样本统计：")
    print(df["status"].value_counts())
    print("\nClass × status：")
    print(pd.crosstab(df["Class"], df["status"]))

    results = []
    for feat in FEATURES:
        print(f"严格交叉验证分析特征：{feat}")
        result = cross_validated_single_feature_analysis(df, feat)
        results.append(result)

    table_rows = []
    for r in results:
        table_rows.append({
            "Feature": r["Feature"],
            "Logistic coefficient sign": r["Logistic coefficient sign"],
            "Mean CV AUC": r["Mean CV AUC"],
            "Mean CV Sensitivity": r["Mean CV Sensitivity"],
            "Mean CV Specificity": r["Mean CV Specificity"],
            "Mean CV Youden threshold (feature)": r["Mean CV Youden threshold (feature)"],
            "Mean CV Youden index": r["Mean CV Youden index"],
        })

    results_df = pd.DataFrame(table_rows).sort_values("Mean CV AUC", ascending=False)
    table2_path = OUTPUT_DIR / "table2_single_feature_cv.csv"
    results_df.to_csv(table2_path, index=False, encoding="utf-8-sig")
    print(f"\n严格版 Table 2 已保存：{table2_path}")
    print(results_df)

    fig3_path = OUTPUT_DIR / "fig3_auc_ranking_cv.png"
    plot_auc_ranking(results_df, fig3_path)
    print(f"Figure 3 已保存：{fig3_path}")

    fig4_path = OUTPUT_DIR / "fig4_top_feature_roc_cv.png"
    plot_top_feature_rocs(results, fig4_path, top_n=TOP_N_ROC)
    print(f"Figure 4 已保存：{fig4_path}")

    print("\n严格交叉验证版单特征分析完成。")


if __name__ == "__main__":
    main()
