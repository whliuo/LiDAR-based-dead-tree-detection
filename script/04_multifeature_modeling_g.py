from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# Configuration
# =========================
PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_DIR / "new_processed" / "tree_features.csv"
OUTPUT_CSV = PROJECT_DIR / "new_processed"/ "04multifeature modeling"/ "table3_multifeature_cv.csv"

CLASS_DEAD = 0
CLASS_ALIVE = {1}

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10

PROFILE_FEATURES = [f"p{i}" for i in range(1, 21)]

FULL_MECHANISM_FEATURES = ["rHDP", "rVRC", "UCRR", "LCRR", "VRI", "VE"]

FEATURE_SETS = {
    "Redistribution feature (VRI)": ["VRI"],
    "Position features": ["rHDP", "rVRC"],
    "Regional allocation features": ["UCRR", "LCRR"],
    "Position + allocation": ["rHDP", "rVRC", "UCRR", "LCRR"],
    "Position + allocation + redistribution": ["rHDP", "rVRC", "UCRR", "LCRR", "VRI"],
    "Mechanism-based structural features": ["rHDP", "rVRC", "UCRR", "LCRR", "VRI", "VE"],
    "Full normalized vertical profile": PROFILE_FEATURES,
}


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


# =========================
# Load data
# =========================
def load_data(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_csv}")

    df = pd.read_csv(input_csv)

    required_cols = ["ID", "Class"] + PROFILE_FEATURES + FULL_MECHANISM_FEATURES
    ensure_required_columns(df, required_cols)

    df["status"] = df["Class"].apply(label_class)
    df = df[df["status"].isin(["dead", "non-dead"])].copy()

    if df.empty:
        raise ValueError("过滤后没有 dead / non-dead 样本，请检查 Class 编码。")

    df["y"] = (df["status"] == "dead").astype(int)
    return df


# =========================
# Build models
# =========================
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
# Main analysis
# =========================
def run_repeated_cv(df: pd.DataFrame) -> pd.DataFrame:
    models = build_models()
    results = []

    y = df["y"].to_numpy(dtype=int)

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    for feature_set_name, feature_cols in FEATURE_SETS.items():
        print(f"\n特征组：{feature_set_name}")
        X_all = df[feature_cols].to_numpy(dtype=float)

        for model_name, base_model in models.items():
            print(f"  模型：{model_name}")
            fold_metrics = []

            for train_idx, test_idx in splitter.split(X_all, y):
                X_train, X_test = X_all[train_idx], X_all[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = clone(base_model)
                model.fit(X_train, y_train)

                y_train_prob = model.predict_proba(X_train)[:, 1]
                best_thr = find_best_threshold_by_youden(y_train, y_train_prob)

                y_test_prob = model.predict_proba(X_test)[:, 1]
                y_test_pred = (y_test_prob >= best_thr).astype(int)

                metrics = compute_metrics(y_test, y_test_pred, y_test_prob)
                fold_metrics.append(metrics)

            summary = {
                "Feature set": feature_set_name,
                "Model": model_name,
            }
            for metric_name in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
                "Specificity",
                "AUC",
            ]:
                vals = [m[metric_name] for m in fold_metrics if np.isfinite(m[metric_name])]
                summary[metric_name] = float(np.mean(vals)) if len(vals) > 0 else np.nan
                summary[f"{metric_name} SD"] = (
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
                )

            results.append(summary)

    return pd.DataFrame(results)


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

    print("\n开始多特征分类分析...")
    result_df = run_repeated_cv(df)

    result_df = result_df.sort_values(
        by=["Feature set", "AUC", "Recall"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nTable 3 已保存：{OUTPUT_CSV}")
    print(result_df)


if __name__ == "__main__":
    main()
