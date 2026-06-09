import os
import math
import glob
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import laspy


# =========================
# Configuration
# Directory structure:
# Root directory/
#   ├─ script/
#   │   └─ current script.py
#   └─ segtree/
#       └─ *.las
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
INPUT_DIR = ROOT_DIR / "dataset1"

# Output directory
OUTPUT_DIR = ROOT_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output paths
OUTPUT_CSV = OUTPUT_DIR / "tree_features.csv"
OUTPUT_XLSX = OUTPUT_DIR / "tree_features.xlsx"

EPS = 1e-9
N_BINS = 20


# =========================
# Filename parsing
# Filename format:
# Class_FieldID_TreeType_Height_TreeID.las
#
# Examples:
# 0_Field1_Valencia_3.25_001.las
# 4_Field2_Hamlin_2.87_015.las
# 5_Field3_SugarBelle_4.10_102.las
#
# Notes:
# 1) TreeType can contain underscores and still be parsed.
# 2) The "Height" field in the filename is no longer used for Tree Height.
#    Only Class, Field, Tree Type, and ID are retained.
# =========================
def parse_filename(file_path: Path) -> Dict[str, str]:
    stem = file_path.stem
    parts = stem.split("_")

    if len(parts) < 5:
        raise ValueError(
            f"文件名格式不符合要求，至少应为 5 段：Class_Field_TreeType_Height_ID，当前文件：{file_path.name}"
        )

    class_str = parts[0].strip()
    field_str = parts[1].strip()
    tree_id_str = parts[-1].strip()

    tree_type_parts = parts[2:-2]
    if len(tree_type_parts) == 0:
        raise ValueError(f"Tree Type 解析失败，当前文件：{file_path.name}")
    tree_type_str = "_".join(tree_type_parts).strip()

    try:
        class_val = int(class_str)
    except Exception as e:
        raise ValueError(f"Class 无法解析为整数：{class_str}，文件：{file_path.name}") from e

    return {
        "ID": tree_id_str,
        "Class": class_val,
        "Field": field_str,
        "Tree Type": tree_type_str,
    }


# =========================
# Statistical utility functions
# =========================
def safe_entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    if probs.size == 0:
        return np.nan
    return float(-np.sum(probs * np.log(probs)))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return np.nan
    return float(np.sum(values * weights) / w_sum)


def weighted_skewness(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted skewness based on a discrete distribution
    using bin centers and bin frequencies.
    """
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return np.nan

    mu = np.sum(weights * values) / w_sum
    var = np.sum(weights * (values - mu) ** 2) / w_sum
    if var <= EPS:
        return 0.0

    sigma = math.sqrt(var)
    skew = np.sum(weights * ((values - mu) / sigma) ** 3) / w_sum
    return float(skew)


def weighted_kurtosis(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted kurtosis based on a discrete distribution using bin centers
    and bin frequencies. This is regular kurtosis, without subtracting 3.
    Subtract 3 at the end if excess kurtosis is needed.
    """
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return np.nan

    mu = np.sum(weights * values) / w_sum
    var = np.sum(weights * (values - mu) ** 2) / w_sum
    if var <= EPS:
        return 0.0

    sigma = math.sqrt(var)
    kurt = np.sum(weights * ((values - mu) / sigma) ** 4) / w_sum
    return float(kurt)


# =========================
# Tree height calculation
# Tree Height = z99 - z1
# =========================
def compute_tree_height(z: np.ndarray) -> float:
    if z is None or len(z) == 0:
        raise ValueError("点云 z 为空")

    z = np.asarray(z, dtype=np.float64)
    z = z[np.isfinite(z)]

    if z.size == 0:
        raise ValueError("点云 z 全部为无效值")

    z1 = np.percentile(z, 1)
    z99 = np.percentile(z, 99)

    if abs(z99 - z1) <= EPS:
        return 0.0

    return float(z99 - z1)


# =========================
# Feature calculation
# Input: z values for a single tree.
# Output: p1~p20, rHDP, rVRC, UCRR, LRR, CCI, VE, skewness, kurtosisr, EOHR
#
# Details:
# 1) Relative height normalization:
#    r = (z - z1) / (z99 - z1)
#    z1 and z99 are the 1st and 99th percentiles.
#    Values are clipped to [0, 1].
#
# 2) 20-bin profile:
#    Split [0, 1] into 20 equal-width bins.
#
# 3) rHDP:
#    Bin center corresponding to the maximum frequency.
#
# 4) rVRC:
#    Normalized vertical return centroid.
#
# 5) UCRR:
#    Upper canopy return ratio = bins 15~20.
#
# 6) LCRR:
#    Lower canopy return ratio = bins 1~6.
#
# 7) VRI:
#    LCRR / (UCRR + EPS)
#
# 8) VE:
#    vertical entropy
#
# 9) skewness / kurtosisr:
#    Weighted skewness/kurtosis based on the 20-bin profile.
#
# 10) EOHR:
#    rH90 - rH10
# =========================
def compute_features_from_z(z: np.ndarray) -> Dict[str, float]:
    if z is None or len(z) == 0:
        raise ValueError("点云 z 为空")

    z = np.asarray(z, dtype=np.float64)
    z = z[np.isfinite(z)]

    if z.size == 0:
        raise ValueError("点云 z 全部为无效值")

    z1 = np.percentile(z, 1)
    z99 = np.percentile(z, 99)

    if abs(z99 - z1) <= EPS:
        raise ValueError("z99 与 z1 几乎相等，无法做归一化")

    r = (z - z1) / (z99 - z1)
    r = np.clip(r, 0.0, 1.0)

    # 20-bin normalized profile
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    counts, _ = np.histogram(r, bins=edges)
    total = counts.sum()

    if total <= 0:
        raise ValueError("20-bin profile 统计结果为空")

    probs = counts.astype(np.float64) / total

    # bin centers: 0.025, 0.075, ..., 0.975
    centers = (edges[:-1] + edges[1:]) / 2.0

    # p1 ~ p20
    profile_dict = {f"p{i+1}": float(probs[i]) for i in range(N_BINS)}

    # rHDP
    max_idx = int(np.argmax(probs))
    rHDP = float(centers[max_idx])

    # rVRC
    rVRC = weighted_mean(centers, probs)

    # UCRR: bins 15~20 -> index 14:20
    UCRR = float(np.sum(probs[14:20]))

    # LCRR: bins 1~6 -> index 0:6
    LCRR = float(np.sum(probs[0:6]))

    # VRI
    VRI = float(LCRR / (UCRR + EPS))

    # VE
    VE = safe_entropy(probs)

    # skewness / kurtosisr
    skewness_val = weighted_skewness(centers, probs)
    kurtosisr_val = weighted_kurtosis(centers, probs)

    # EOHR = rH90 - rH10
    rH10 = float(np.percentile(r, 10))
    rH90 = float(np.percentile(r, 90))
    EOHR = float(rH90 - rH10)

    out = {}
    out.update(profile_dict)
    out.update(
        {
            "rHDP": rHDP,
            "rVRC": rVRC,
            "UCRR": UCRR,
            "LCRR": LCRR,
            "VRI": VRI,
            "VE": VE,
            "skewness": skewness_val,
            "kurtosisr": kurtosisr_val,
            "EOHR": EOHR,
        }
    )
    return out


# =========================
# Read LAS
# =========================
def read_las_z(file_path: Path) -> np.ndarray:
    las = laspy.read(file_path)
    z = np.asarray(las.z, dtype=np.float64)
    return z


# =========================
# Main workflow
# =========================
def build_feature_table(input_dir: Path) -> pd.DataFrame:
    if not input_dir.exists():
        raise FileNotFoundError(f"输入文件夹不存在：{input_dir}")

    las_files = sorted(glob.glob(str(input_dir / "*.las")))
    if len(las_files) == 0:
        raise FileNotFoundError(f"在 {input_dir} 中未找到 .las 文件")

    rows: List[Dict] = []
    failed_files: List[Tuple[str, str]] = []

    for idx, file_str in enumerate(las_files, start=1):
        file_path = Path(file_str)
        print(f"[{idx}/{len(las_files)}] 处理：{file_path.name}")

        try:
            meta = parse_filename(file_path)
            z = read_las_z(file_path)

            tree_height = compute_tree_height(z)
            feats = compute_features_from_z(z)

            row = {
                "ID": meta["ID"],
                "Class": meta["Class"],
                "Field": meta["Field"],
                "Tree Type": meta["Tree Type"],
                "Tree Height": tree_height,
            }
            row.update(feats)
            rows.append(row)

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            failed_files.append((file_path.name, err_msg))
            print(f"  -> 失败：{err_msg}")
            # traceback.print_exc()

    if len(rows) == 0:
        raise RuntimeError("所有文件都处理失败，未生成任何结果。")

    df = pd.DataFrame(rows)

    ordered_cols = [
        "ID",
        "Class",
        "Field",
        "Tree Type",
        "Tree Height",
        "p1", "p2", "p3", "p4", "p5",
        "p6", "p7", "p8", "p9", "p10",
        "p11", "p12", "p13", "p14", "p15",
        "p16", "p17", "p18", "p19", "p20",
        "rHDP",
        "rVRC",
        "UCRR",
        "LCRR",
        "VRI",
        "VE",
        "skewness",
        "kurtosisr",
        "EOHR",
    ]

    missing_cols = [c for c in ordered_cols if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"结果表缺少以下列：{missing_cols}")

    df = df[ordered_cols]

    df = df.sort_values(by=["Field", "Class", "ID"], kind="stable").reset_index(drop=True)

    if failed_files:
        print("\n以下文件处理失败：")
        for name, msg in failed_files:
            print(f" - {name}: {msg}")

        fail_df = pd.DataFrame(failed_files, columns=["file_name", "error"])
        fail_csv = OUTPUT_DIR / "tree_features_failed_files.csv"
        fail_df.to_csv(fail_csv, index=False, encoding="utf-8-sig")
        print(f"\n失败文件列表已保存：{fail_csv}")

    return df


def main():
    print(f"脚本目录：{SCRIPT_DIR}")
    print(f"根目录：{ROOT_DIR}")
    print(f"输入目录：{INPUT_DIR}")
    print("开始处理 LAS 文件并计算特征...\n")

    df = build_feature_table(INPUT_DIR)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nCSV 已保存：{OUTPUT_CSV}")

    try:
        df.to_excel(OUTPUT_XLSX, index=False)
        print(f"Excel 已保存：{OUTPUT_XLSX}")
    except Exception as e:
        print(f"Excel 保存失败，但 CSV 已成功保存。原因：{e}")

    print("\n处理完成。")
    print(f"总样本数：{len(df)}")
    print("\nClass 统计：")
    print(df["Class"].value_counts(dropna=False).sort_index())

    print("\nField × Class 统计：")
    print(pd.crosstab(df["Field"], df["Class"]))


if __name__ == "__main__":
    main()
