#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from scipy.spatial.distance import cdist
from tqdm import tqdm


BINARY_DATASETS = {
    9: {
        "bone": "LH",
        "dataset_name": "Dataset009_anatomy_LH",
        "out_csv": "binary_anatomy_LH.csv",
    },
    10: {
        "bone": "RH",
        "dataset_name": "Dataset010_anatomy_RH",
        "out_csv": "binary_anatomy_RH.csv",
    },
    11: {
        "bone": "S",
        "dataset_name": "Dataset011_anatomy_S",
        "out_csv": "binary_anatomy_S.csv",
    },
}


def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "nnUNet").exists() or (parent / "README.md").exists():
            return parent
    raise FileNotFoundError("Could not find project root.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one binary anatomy model before merging."
    )
    parser.add_argument(
        "--dataset_id",
        type=int,
        required=True,
        choices=sorted(BINARY_DATASETS.keys()),
        help="Binary anatomy dataset ID: 9=LH, 10=RH, 11=S.",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=None,
        help="Prediction folder. Defaults to predictions/<dataset_name>.",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Ground-truth folder. Defaults to nnUNet_raw/<dataset_name>/labelsTs.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path. Defaults to evaluation/binary_anatomy_<bone>.csv.",
    )
    parser.add_argument("--ignore_missing", action="store_true")
    return parser.parse_args()


def resolve_path(path_str: str | None, default: Path) -> Path:
    if path_str is None:
        return default

    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def list_image_files(folder: Path) -> list[Path]:
    return sorted(
        list(folder.glob("*.nii.gz"))
        + list(folder.glob("*.nii"))
        + list(folder.glob("*.mha"))
    )


def case_id(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii") or name.endswith(".mha"):
        return name[:-4]
    return path.stem


def read_image(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    spacing = np.asarray(img.GetSpacing()[::-1], dtype=np.float32)
    return arr, spacing


def dice_score(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    denom = int(gt_mask.sum()) + int(pred_mask.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(gt_mask, pred_mask).sum() / denom)


def iou_score(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    union = int(np.logical_or(gt_mask, pred_mask).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(gt_mask, pred_mask).sum() / union)


def surface_points(mask: np.ndarray, spacing: np.ndarray, sample_size: int = 10000):
    if not mask.any():
        return np.empty((0, 3), dtype=np.float32)

    struct = generate_binary_structure(3, 1)
    eroded = binary_erosion(mask, structure=struct)
    surface = binary_dilation(mask, structure=struct) & ~eroded
    points = np.argwhere(surface)

    if points.shape[0] > sample_size:
        rng = np.random.default_rng(2024)
        idx = rng.choice(points.shape[0], size=sample_size, replace=False)
        points = points[idx]

    return points.astype(np.float32) * spacing


def hd95_and_assd(gt_mask: np.ndarray, pred_mask: np.ndarray, spacing: np.ndarray):
    gt_points = surface_points(gt_mask, spacing)
    pred_points = surface_points(pred_mask, spacing)

    if gt_points.size == 0 and pred_points.size == 0:
        return 0.0, 0.0
    if gt_points.size == 0 or pred_points.size == 0:
        return float("inf"), float("inf")

    distances = cdist(gt_points, pred_points, metric="euclidean").astype(np.float32)
    gt_to_pred = np.min(distances, axis=1)
    pred_to_gt = np.min(distances, axis=0)

    hd95 = max(np.percentile(gt_to_pred, 95), np.percentile(pred_to_gt, 95))
    assd = (np.mean(gt_to_pred) + np.mean(pred_to_gt)) / 2
    return float(hd95), float(assd)


def format_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return f"{value:.4f}"


def mean_finite(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def main():
    args = parse_args()
    dataset_config = BINARY_DATASETS[args.dataset_id]
    dataset_name = dataset_config["dataset_name"]

    pred_dir = resolve_path(
        args.pred_dir,
        PROJECT_ROOT / "predictions" / dataset_name,
    )
    gt_dir = resolve_path(
        args.gt_dir,
        PROJECT_ROOT / "nnUNet" / "nnUNet_raw" / dataset_name / "labelsTs",
    )
    out_csv = resolve_path(
        args.out_csv,
        PROJECT_ROOT / "evaluation" / dataset_config["out_csv"],
    )

    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction folder not found: {pred_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth folder not found: {gt_dir}")

    pred_map = {case_id(p): p for p in list_image_files(pred_dir)}
    gt_map = {case_id(p): p for p in list_image_files(gt_dir)}

    rows = []
    for cid in tqdm(sorted(gt_map.keys()), desc=f"Evaluating {dataset_config['bone']}"):
        pred_path = pred_map.get(cid)
        if pred_path is None:
            if args.ignore_missing:
                continue
            raise FileNotFoundError(f"Missing prediction for case {cid} in {pred_dir}")

        gt, spacing = read_image(gt_map[cid])
        pred, _ = read_image(pred_path)

        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch for {cid}: gt {gt.shape}, pred {pred.shape}")

        gt_mask = gt == 1
        pred_mask = pred == 1
        hd95, assd = hd95_and_assd(gt_mask, pred_mask, spacing)

        rows.append(
            {
                "case": cid,
                "dice": dice_score(gt_mask, pred_mask),
                "iou": iou_score(gt_mask, pred_mask),
                "hd95": hd95,
                "assd": assd,
                "gt_voxels": int(gt_mask.sum()),
                "pred_voxels": int(pred_mask.sum()),
            }
        )

    if not rows:
        raise RuntimeError("No cases were evaluated.")

    average = {
        "case": "AVERAGE",
        "dice": mean_finite([r["dice"] for r in rows]),
        "iou": mean_finite([r["iou"] for r in rows]),
        "hd95": mean_finite([r["hd95"] for r in rows]),
        "assd": mean_finite([r["assd"] for r in rows]),
        "gt_voxels": mean_finite([float(r["gt_voxels"]) for r in rows]),
        "pred_voxels": mean_finite([float(r["pred_voxels"]) for r in rows]),
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case", "dice", "iou", "hd95", "assd", "gt_voxels", "pred_voxels"]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "case": row["case"],
                    "dice": format_metric(row["dice"]),
                    "iou": format_metric(row["iou"]),
                    "hd95": format_metric(row["hd95"]),
                    "assd": format_metric(row["assd"]),
                    "gt_voxels": row["gt_voxels"],
                    "pred_voxels": row["pred_voxels"],
                }
            )

        writer.writerow(
            {
                "case": "AVERAGE",
                "dice": format_metric(average["dice"]),
                "iou": format_metric(average["iou"]),
                "hd95": format_metric(average["hd95"]),
                "assd": format_metric(average["assd"]),
                "gt_voxels": format_metric(average["gt_voxels"]),
                "pred_voxels": format_metric(average["pred_voxels"]),
            }
        )

    print(f"Saved binary anatomy evaluation CSV: {out_csv}")


if __name__ == "__main__":
    main()
