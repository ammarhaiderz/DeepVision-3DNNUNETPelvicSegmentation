#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


BINARY_DATASETS = {
    "LH": "Dataset009_anatomy_LH",
    "RH": "Dataset010_anatomy_RH",
    "S": "Dataset011_anatomy_S",
}


def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "nnUNet").exists() or (parent / "README.md").exists():
            return parent
    raise FileNotFoundError("Could not find project root.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
NNUNET_RAW = PROJECT_ROOT / "nnUNet" / "nnUNet_raw"
OUT_DIR = PROJECT_ROOT / "analysis"


def load_split_names() -> dict[str, str]:
    split_path = PROJECT_ROOT / "train_val_test.json"
    if not split_path.is_file():
        return {}

    with split_path.open("r", encoding="utf-8") as f:
        split = json.load(f)

    mapping = {}
    for split_name in ("train", "val", "test"):
        for case_id in split.get(split_name, []):
            mapping[str(case_id)] = split_name
    return mapping


def read_label(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()[::-1]
    return arr, spacing


def bbox(mask: np.ndarray):
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return {
            "bbox_z": 0,
            "bbox_y": 0,
            "bbox_x": 0,
        }

    z_size = int(coords[0].max() - coords[0].min() + 1)
    y_size = int(coords[1].max() - coords[1].min() + 1)
    x_size = int(coords[2].max() - coords[2].min() + 1)
    return {
        "bbox_z": z_size,
        "bbox_y": y_size,
        "bbox_x": x_size,
    }


def case_id_from_label(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def analyze_label(path: Path, bone: str, dataset_name: str, folder_split: str, split_map: dict[str, str]):
    arr, spacing = read_label(path)
    mask = arr == 1
    unique_labels = sorted(int(x) for x in np.unique(arr))
    foreground_voxels = int(mask.sum())
    total_voxels = int(arr.size)
    spacing_z, spacing_y, spacing_x = [float(x) for x in spacing]
    voxel_volume_mm3 = spacing_z * spacing_y * spacing_x
    case_id = case_id_from_label(path)

    row = {
        "bone": bone,
        "dataset": dataset_name,
        "folder_split": folder_split,
        "split": split_map.get(case_id, folder_split),
        "case": case_id,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "unique_labels": " ".join(str(x) for x in unique_labels),
        "is_binary": int(set(unique_labels).issubset({0, 1})),
        "is_empty": int(foreground_voxels == 0),
        "foreground_voxels": foreground_voxels,
        "total_voxels": total_voxels,
        "foreground_fraction": foreground_voxels / total_voxels if total_voxels else 0.0,
        "foreground_volume_mm3": foreground_voxels * voxel_volume_mm3,
        "spacing_z": spacing_z,
        "spacing_y": spacing_y,
        "spacing_x": spacing_x,
    }
    row.update(bbox(mask))
    return row


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    groups = sorted({(row["bone"], row["dataset"], row["folder_split"]) for row in rows})

    for bone, dataset, folder_split in groups:
        group = [
            row for row in rows
            if row["bone"] == bone
            and row["dataset"] == dataset
            and row["folder_split"] == folder_split
        ]
        voxels = [int(row["foreground_voxels"]) for row in group]
        fractions = [float(row["foreground_fraction"]) for row in group]
        summaries.append(
            {
                "bone": bone,
                "dataset": dataset,
                "folder_split": folder_split,
                "n_cases": len(group),
                "empty_cases": int(sum(int(row["is_empty"]) for row in group)),
                "non_binary_cases": int(sum(1 - int(row["is_binary"]) for row in group)),
                "voxels_min": int(min(voxels)) if voxels else 0,
                "voxels_p05": percentile(voxels, 5),
                "voxels_median": percentile(voxels, 50),
                "voxels_mean": float(np.mean(voxels)) if voxels else 0.0,
                "voxels_p95": percentile(voxels, 95),
                "voxels_max": int(max(voxels)) if voxels else 0,
                "foreground_fraction_mean": float(np.mean(fractions)) if fractions else 0.0,
            }
        )

    return summaries


def add_outlier_flags(rows: list[dict]) -> None:
    for bone in sorted({row["bone"] for row in rows}):
        bone_rows = [row for row in rows if row["bone"] == bone]
        values = np.asarray([int(row["foreground_voxels"]) for row in bone_rows], dtype=np.float64)
        if values.size == 0:
            continue

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr

        for row in bone_rows:
            value = int(row["foreground_voxels"])
            row["is_volume_outlier"] = int(value < low or value > high)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write for {path}")

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    split_map = load_split_names()
    rows = []

    for bone, dataset_name in BINARY_DATASETS.items():
        dataset_root = NNUNET_RAW / dataset_name
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {dataset_root}")

        for folder_name, folder_split in (("labelsTr", "train_val"), ("labelsTs", "test")):
            label_dir = dataset_root / folder_name
            if not label_dir.is_dir():
                raise FileNotFoundError(f"Missing label folder: {label_dir}")

            for label_path in sorted(label_dir.glob("*.nii.gz")):
                rows.append(analyze_label(label_path, bone, dataset_name, folder_split, split_map))

    add_outlier_flags(rows)
    summary_rows = summarize(rows)

    detail_csv = OUT_DIR / "binary_anatomy_eda.csv"
    summary_csv = OUT_DIR / "binary_anatomy_eda_summary.csv"
    write_csv(detail_csv, rows)
    write_csv(summary_csv, summary_rows)

    print(f"Saved detail CSV:  {detail_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print("\nSummary:")
    for row in summary_rows:
        print(
            f"{row['bone']} {row['folder_split']}: "
            f"n={row['n_cases']}, empty={row['empty_cases']}, "
            f"non_binary={row['non_binary_cases']}, "
            f"median_voxels={row['voxels_median']:.0f}, "
            f"mean_voxels={row['voxels_mean']:.0f}"
        )

    flagged = [
        row for row in rows
        if int(row["is_empty"]) or not int(row["is_binary"]) or int(row["is_volume_outlier"])
    ]
    print(f"\nFlagged cases: {len(flagged)}")
    for row in flagged[:30]:
        print(
            f"{row['bone']} {row['split']} {row['case']}: "
            f"voxels={row['foreground_voxels']}, "
            f"empty={row['is_empty']}, binary={row['is_binary']}, "
            f"outlier={row['is_volume_outlier']}"
        )


if __name__ == "__main__":
    main()
