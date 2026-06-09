from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# Configuration
# =========================
PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_DIR / "new_processed" / "tree_features.csv"
OUTPUT_CSV = PROJECT_DIR / "new_processed" / "table4_lofo.csv"
FIG_DIR = PROJECT_DIR / "new_processed" / "figures" / "main"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_DEAD = 0
CLASS_ALIVE = {1}
RANDOM_STATE = 42

FIELD_COL = "Field"

PROFILE_FEATURES = [f"p{i}" for i in range(1, 21)]
FULL_MECHANISM_FEATURES = ["rHDP", "rVRC", "UCRR", "LCRR", "VRI", "VE"]

LOFO_FEATURE_SETS = {
    "VRI": ["VRI"],
    "Mechanism-based structural features": FULL_MECHANISM_FEATURES,
    "Full normalized vertical profile": PROFILE_FEATURES,
}

FEATURE_PLOT_ORDER = [
    "VRI",
    "Mechanism-based structural features",
    "Full normalized vertical profile",
]

MODEL_ORDER = ["Logistic regression", "Random forest"]

# Low-saturation colors, following the Figure 2 palette reference.
BAR_COLORS = {
    "Logistic regression": "#8FA8B8",   # Low-saturation blue-gray
    "Random forest": "#D9A27A",         # Low-saturation orange-brown
}
EDGE_COLOR = "black"
BAR_LINEWIDTH = 2.0
BAR_ALPHA = 0.82


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


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan

    return {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1-score": float(f1),
        "Specificity": float(spec),
        "AUC": float(auc) if np.isfinite(auc) else np.nan,
    }


def find_best_threshold_by_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    return float(thresholds[best_idx])


def build_models() -> Dict[str, object]:
    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return {
        "Logistic regression": lr,
        "Random forest": rf,
    }


# =========================
# Load data
# =========================
def load_data(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_csv}")

    df = pd.read_csv(input_csv)

    required_cols = ["ID", "Class", FIELD_COL] + PROFILE_FEATURES + FULL_MECHANISM_FEATURES
    ensure_required_columns(df, required_cols)

    df[FIELD_COL] = df[FIELD_COL].astype(str)

    df["status"] = df["Class"].apply(label_class)
    df = df[df["status"].isin(["dead", "non-dead"])].copy()

    if df.empty:
        raise ValueError("过滤后没有 dead / non-dead 样本，请检查 Class 编码。")

    df["y"] = (df["status"] == "dead").astype(int)
    return df


# =========================
# Main LOFO analysis
# =========================
def run_lofo(df: pd.DataFrame) -> pd.DataFrame:
    models = build_models()
    fields = sorted(df[FIELD_COL].dropna().unique().tolist())

    if len(fields) < 2:
        raise ValueError("Field 数量不足，无法进行 leave-one-field-out。")

    results = []

    for test_field in fields:
        train_df = df[df[FIELD_COL] != test_field].copy()
        test_df = df[df[FIELD_COL] == test_field].copy()

        train_fields = sorted(train_df[FIELD_COL].unique().tolist())
        train_field_str = " + ".join(train_fields)
        test_field_str = str(test_field)

        y_train = train_df["y"].to_numpy(dtype=int)
        y_test = test_df["y"].to_numpy(dtype=int)

        n_test = len(test_df)
        n_dead_test = int(np.sum(y_test == 1))
        n_alive_test = int(np.sum(y_test == 0))

        for feature_set_name, feature_cols in LOFO_FEATURE_SETS.items():
            X_train = train_df[feature_cols].to_numpy(dtype=float)
            X_test = test_df[feature_cols].to_numpy(dtype=float)

            for model_name, base_model in models.items():
                model = clone(base_model)
                model.fit(X_train, y_train)

                y_train_prob = model.predict_proba(X_train)[:, 1]
                best_thr = find_best_threshold_by_youden(y_train, y_train_prob)

                y_test_prob = model.predict_proba(X_test)[:, 1]
                y_test_pred = (y_test_prob >= best_thr).astype(int)

                metrics = compute_metrics(y_test, y_test_pred, y_test_prob)

                row = {
                    "Training fields": train_field_str,
                    "Test field": test_field_str,
                    "Feature set": feature_set_name,
                    "Model": model_name,
                    "Test N": n_test,
                    "Test dead N": n_dead_test,
                    "Test non-dead N": n_alive_test,
                    "Threshold": best_thr,
                }
                row.update(metrics)
                results.append(row)

    return pd.DataFrame(results)


# =========================
# Figure 5: combined 2 x 3 panel figure
# First row: AUC; second row: Recall.
# =========================
def plot_lofo_combined_figure(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    feature_titles = {
        "VRI": "VRI",
        "Mechanism-based structural features": "Mechanism-based \nstructural features",
        "Full normalized vertical profile": "Full normalized \nvertical profile",
    }

    metrics = [
        ("AUC", "AUC"),
        ("Recall", "Recall"),
    ]

    panel_labels = ["a)", "b)", "c)", "d)", "e)", "f)"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11.2), squeeze=False)

    label_idx = 0

    for row_idx, (metric_col, ylabel) in enumerate(metrics):
        for col_idx, feature_name in enumerate(FEATURE_PLOT_ORDER):
            ax = axes[row_idx, col_idx]
            sub_df = df[df["Feature set"] == feature_name].copy()

            fields = sorted(sub_df["Test field"].drop_duplicates().tolist())
            x = np.arange(len(fields))
            width = 0.34

            if metric_col == "AUC":
                ymin, ymax = 0.99, 1.0007
                yticks = [0.99, 0.995, 1.00]
            else:
                ymin, ymax = 0.50, 1.04
                yticks = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

            ax.set_ylim(ymin, ymax)
            ax.set_yticks(yticks)

            for j, model_name in enumerate(MODEL_ORDER):
                sub_model = sub_df[sub_df["Model"] == model_name].copy()
                sub_model = sub_model.set_index("Test field").reindex(fields)

                vals = sub_model[metric_col].to_numpy(dtype=float)
                xpos = x + (j - 0.5) * width

                bars = ax.bar(
                    xpos,
                    vals,
                    width=width,
                    color=BAR_COLORS[model_name],
                    edgecolor=EDGE_COLOR,
                    linewidth=BAR_LINEWIDTH,
                    alpha=BAR_ALPHA,
                    label=model_name,
                )

                offset = (ymax - ymin) * 0.010

                for rect, val in zip(bars, vals):
                    if np.isfinite(val):
                        ax.text(
                            rect.get_x() + rect.get_width() / 2,
                            val + offset,
                            f"{val:.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=12,
                        )

            ax.set_xticks(x)
            ax.set_xticklabels([f"Field {f}" for f in fields], fontsize=18)

            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=20, fontweight="bold")
                ax.tick_params(axis="y", labelsize=18)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)

            ax.tick_params(axis="x", labelsize=18)

            ax.set_title(
                f"{panel_labels[label_idx]} {feature_titles[feature_name]}",
                fontsize=20,
                fontweight="bold",
                pad=12,
            )
            label_idx += 1

            for spine in ax.spines.values():
                spine.set_linewidth(2.0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.87, 0.488),
        fontsize=18,
        frameon=False,
    )

    plt.tight_layout(rect=[0.03, 0.03, 0.87, 0.98], h_pad=2.2, w_pad=2.6)
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

    print("\nField × status：")
    print(pd.crosstab(df[FIELD_COL], df["status"]))

    print("\nClass × status：")
    print(pd.crosstab(df["Class"], df["status"]))

    print("\n开始 leave-one-field-out 分析...")
    result_df = run_lofo(df)

    result_df = result_df.sort_values(
        by=["Feature set", "Test field", "Model"]
    ).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nTable 4 已保存：{OUTPUT_CSV}")
    print(result_df)

    fig5_combined = FIG_DIR / "fig5_lofo_auc_recall_combined.png"
    plot_lofo_combined_figure(
        result_df,
        output_path=fig5_combined,
    )
    print(f"Figure 5 已保存：{fig5_combined}")

    print("\nLOFO 分析完成。")


if __name__ == "__main__":
    main()
