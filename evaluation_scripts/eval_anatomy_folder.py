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


DEFAULT_DATASET = "Dataset002_anatomy_network"
CLASS_NAMES = {
    1: "LH",
    2: "RH",
    3: "S",
}


def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "nnUNet").exists() or (parent / "README.md").exists():
            return parent
    raise FileNotFoundError("Could not find project root.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate anatomy-only nnU-Net predictions for Dataset002."
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=str(PROJECT_ROOT / "predictions" / DEFAULT_DATASET),
        help="Folder containing predicted anatomy masks.",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=str(
            PROJECT_ROOT
            / "nnUNet"
            / "nnUNet_raw"
            / DEFAULT_DATASET
            / "labelsTs"
        ),
        help="Folder containing anatomy ground-truth masks.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=str(PROJECT_ROOT / "evaluation" / "anatomy_predictions.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--ignore_missing", action="store_true")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
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


def evaluate_case(gt: np.ndarray, pred: np.ndarray, spacing: np.ndarray) -> dict:
    metrics = {}

    for label, name in CLASS_NAMES.items():
        gt_mask = gt == label
        pred_mask = pred == label

        hd95, assd = hd95_and_assd(gt_mask, pred_mask, spacing)
        metrics[f"{name}_dice"] = dice_score(gt_mask, pred_mask)
        metrics[f"{name}_iou"] = iou_score(gt_mask, pred_mask)
        metrics[f"{name}_hd95"] = hd95
        metrics[f"{name}_assd"] = assd

    metrics["mean_dice"] = mean_finite([metrics[f"{n}_dice"] for n in CLASS_NAMES.values()])
    metrics["mean_iou"] = mean_finite([metrics[f"{n}_iou"] for n in CLASS_NAMES.values()])
    metrics["mean_hd95"] = mean_finite([metrics[f"{n}_hd95"] for n in CLASS_NAMES.values()])
    metrics["mean_assd"] = mean_finite([metrics[f"{n}_assd"] for n in CLASS_NAMES.values()])
    return metrics


def main():
    args = parse_args()
    pred_dir = resolve_path(args.pred_dir)
    gt_dir = resolve_path(args.gt_dir)
    out_csv = resolve_path(args.out_csv)

    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction folder not found: {pred_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth folder not found: {gt_dir}")

    pred_map = {case_id(p): p for p in list_image_files(pred_dir)}
    gt_map = {case_id(p): p for p in list_image_files(gt_dir)}

    rows = []
    for cid in tqdm(sorted(gt_map.keys()), desc="Evaluating anatomy cases"):
        pred_path = pred_map.get(cid)
        if pred_path is None:
            if args.ignore_missing:
                continue
            raise FileNotFoundError(f"Missing prediction for case {cid} in {pred_dir}")

        gt, spacing = read_image(gt_map[cid])
        pred, _ = read_image(pred_path)

        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch for {cid}: gt {gt.shape}, pred {pred.shape}")

        row = {"case": cid}
        row.update(evaluate_case(gt, pred, spacing))
        rows.append(row)

    if not rows:
        raise RuntimeError("No cases were evaluated.")

    fieldnames = ["case"]
    for name in CLASS_NAMES.values():
        fieldnames.extend([f"{name}_dice", f"{name}_iou", f"{name}_hd95", f"{name}_assd"])
    fieldnames.extend(["mean_dice", "mean_iou", "mean_hd95", "mean_assd"])

    average = {"case": "AVERAGE"}
    for field in fieldnames[1:]:
        average[field] = mean_finite([float(row[field]) for row in rows])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"case": row["case"], **{k: format_metric(row[k]) for k in fieldnames[1:]}})
        writer.writerow({"case": "AVERAGE", **{k: format_metric(average[k]) for k in fieldnames[1:]}})

    print(f"Saved anatomy evaluation CSV: {out_csv}")


if __name__ == "__main__":
    main()
