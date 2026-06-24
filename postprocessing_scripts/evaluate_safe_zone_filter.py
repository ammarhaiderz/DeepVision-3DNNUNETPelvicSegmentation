#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree
from tqdm import tqdm

from safe_zone_filter import apply_safe_zone_context, build_safe_zone_context


BINARY_DATASETS = {
    9: {"bone": "LH", "dataset_name": "Dataset009_anatomy_LH"},
    10: {"bone": "RH", "dataset_name": "Dataset010_anatomy_RH"},
    11: {"bone": "S", "dataset_name": "Dataset011_anatomy_S"},
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Filter distant binary prediction artifacts relative to the "
            "largest 3D connected component and compare metrics."
        )
    )
    parser.add_argument(
        "--dataset_id",
        type=int,
        required=True,
        choices=sorted(BINARY_DATASETS),
    )
    parser.add_argument(
        "--distance_mm",
        type=float,
        nargs="+",
        default=[10.0, 20.0, 30.0, 40.0],
        help="Safe-zone distances to evaluate in millimetres.",
    )
    parser.add_argument("--pred_dir", default=None)
    parser.add_argument("--gt_dir", default=None)
    parser.add_argument("--out_root", default=None)
    parser.add_argument(
        "--write_distance_mm",
        type=float,
        default=None,
        help=(
            "Write NIfTI masks for this distance. Defaults to the distance "
            "with the highest average Dice."
        ),
    )
    return parser.parse_args()


def resolve(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def case_id(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def image_map(folder: Path) -> dict[str, Path]:
    return {case_id(path): path for path in sorted(folder.glob("*.nii.gz"))}


def same_geometry(first: sitk.Image, second: sitk.Image) -> bool:
    return (
        first.GetSize() == second.GetSize()
        and np.allclose(first.GetSpacing(), second.GetSpacing())
        and np.allclose(first.GetOrigin(), second.GetOrigin())
        and np.allclose(first.GetDirection(), second.GetDirection())
    )


def overlap_metrics(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    intersection = int(np.logical_and(gt, pred).sum())
    denominator = int(gt.sum()) + int(pred.sum())
    union = int(np.logical_or(gt, pred).sum())
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    iou = 1.0 if union == 0 else intersection / union
    return float(dice), float(iou)


def surface_points(mask: np.ndarray, spacing_zyx: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.empty((0, 3), dtype=np.float32)
    surface = mask & ~binary_erosion(
        mask, structure=generate_binary_structure(3, 1)
    )
    return np.argwhere(surface).astype(np.float32) * spacing_zyx


def surface_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing_zyx: np.ndarray,
) -> tuple[float, float]:
    gt_points = surface_points(gt, spacing_zyx)
    pred_points = surface_points(pred, spacing_zyx)
    if not len(gt_points) and not len(pred_points):
        return 0.0, 0.0
    if not len(gt_points) or not len(pred_points):
        return float("inf"), float("inf")
    gt_to_pred = cKDTree(pred_points).query(gt_points, k=1, workers=1)[0]
    pred_to_gt = cKDTree(gt_points).query(pred_points, k=1, workers=1)[0]
    hd95 = max(
        np.percentile(gt_to_pred, 95),
        np.percentile(pred_to_gt, 95),
    )
    assd = (gt_to_pred.mean() + pred_to_gt.mean()) / 2
    return float(hd95), float(assd)


def overlap_and_error_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
) -> dict[str, float | int]:
    dice, iou = overlap_metrics(gt, pred)
    return {
        "dice": dice,
        "iou": iou,
        "false_positive_voxels": int(np.logical_and(pred, ~gt).sum()),
        "false_negative_voxels": int(np.logical_and(gt, ~pred).sum()),
        "prediction_voxels": int(pred.sum()),
    }


def write_mask(mask: np.ndarray, reference: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = sitk.GetImageFromArray(mask.astype(np.uint8))
    output.CopyInformation(reference)
    sitk.WriteImage(output, str(path))


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def main():
    args = parse_args()
    config = BINARY_DATASETS[args.dataset_id]
    pred_dir = resolve(
        args.pred_dir,
        PROJECT_ROOT / "predictions" / config["dataset_name"],
    )
    gt_dir = resolve(
        args.gt_dir,
        PROJECT_ROOT
        / "nnUNet"
        / "nnUNet_raw"
        / config["dataset_name"]
        / "labelsTs",
    )
    out_root = resolve(
        args.out_root,
        PROJECT_ROOT / "safe_zone_experiments" / config["bone"],
    )
    distances = sorted(set(float(value) for value in args.distance_mm))
    if any(value < 0 for value in distances):
        raise ValueError("All safe-zone distances must be non-negative.")

    pred_map = image_map(pred_dir)
    gt_map = image_map(gt_dir)
    missing = sorted(set(gt_map) - set(pred_map))
    if missing:
        raise FileNotFoundError(f"Missing predictions for: {missing}")

    rows = []
    for cid in tqdm(sorted(gt_map), desc=f"Safe-zone sweep {config['bone']}"):
        pred_image = sitk.ReadImage(str(pred_map[cid]))
        gt_image = sitk.ReadImage(str(gt_map[cid]))
        if not same_geometry(pred_image, gt_image):
            raise ValueError(f"Prediction/ground-truth geometry mismatch: {cid}")
        prediction = sitk.GetArrayFromImage(pred_image) > 0
        gt = sitk.GetArrayFromImage(gt_image) > 0
        spacing_zyx = np.asarray(pred_image.GetSpacing()[::-1], dtype=np.float32)
        original = overlap_and_error_metrics(gt, prediction)
        context = build_safe_zone_context(
            prediction, spacing_zyx, max(distances)
        )

        for distance_mm in distances:
            filtered, _, removed = apply_safe_zone_context(
                prediction, context, distance_mm
            )
            filtered_metrics = overlap_and_error_metrics(gt, filtered)
            rows.append(
                {
                    "case": cid,
                    "distance_mm": distance_mm,
                    **{f"original_{key}": value for key, value in original.items()},
                    **{
                        f"filtered_{key}": value
                        for key, value in filtered_metrics.items()
                    },
                    "dice_change": filtered_metrics["dice"] - original["dice"],
                    "iou_change": filtered_metrics["iou"] - original["iou"],
                    "removed_voxels": int(removed.sum()),
                    "largest_component_voxels": int(context.largest.sum()),
                }
            )

    summaries = []
    for distance_mm in distances:
        selected = [row for row in rows if row["distance_mm"] == distance_mm]
        summary = {"distance_mm": distance_mm, "cases": len(selected)}
        for field in (
            "original_dice",
            "filtered_dice",
            "dice_change",
            "original_iou",
            "filtered_iou",
            "iou_change",
            "original_false_positive_voxels",
            "filtered_false_positive_voxels",
            "original_false_negative_voxels",
            "filtered_false_negative_voxels",
            "removed_voxels",
        ):
            summary[field] = finite_mean([float(row[field]) for row in selected])
        summary["cases_dice_improved"] = sum(
            row["dice_change"] > 0 for row in selected
        )
        summary["cases_dice_worsened"] = sum(
            row["dice_change"] < 0 for row in selected
        )
        summaries.append(summary)

    best = max(summaries, key=lambda row: row["filtered_dice"])
    write_distance = (
        float(args.write_distance_mm)
        if args.write_distance_mm is not None
        else float(best["distance_mm"])
    )
    if write_distance not in distances:
        raise ValueError(
            f"write_distance_mm={write_distance} was not evaluated: {distances}"
        )

    distance_name = f"{write_distance:g}mm"
    mask_root = out_root / distance_name
    comparison_rows = []
    for cid in sorted(gt_map):
        reference = sitk.ReadImage(str(pred_map[cid]))
        gt_image = sitk.ReadImage(str(gt_map[cid]))
        prediction = sitk.GetArrayFromImage(reference) > 0
        gt = sitk.GetArrayFromImage(gt_image) > 0
        spacing_zyx = np.asarray(reference.GetSpacing()[::-1], dtype=np.float32)
        context = build_safe_zone_context(
            prediction, spacing_zyx, write_distance
        )
        filtered, safe_zone, removed = apply_safe_zone_context(
            prediction,
            context,
            write_distance,
            return_safe_zone=True,
        )
        original_hd95, original_assd = surface_metrics(
            gt, prediction, spacing_zyx
        )
        filtered_hd95, filtered_assd = surface_metrics(
            gt, filtered, spacing_zyx
        )
        original = overlap_and_error_metrics(gt, prediction)
        filtered_metrics = overlap_and_error_metrics(gt, filtered)
        comparison_rows.append(
            {
                "case": cid,
                "distance_mm": write_distance,
                **{f"original_{key}": value for key, value in original.items()},
                "original_hd95": original_hd95,
                "original_assd": original_assd,
                **{
                    f"filtered_{key}": value
                    for key, value in filtered_metrics.items()
                },
                "filtered_hd95": filtered_hd95,
                "filtered_assd": filtered_assd,
                "dice_change": filtered_metrics["dice"] - original["dice"],
                "iou_change": filtered_metrics["iou"] - original["iou"],
                "hd95_change": filtered_hd95 - original_hd95,
                "assd_change": filtered_assd - original_assd,
                "removed_voxels": int(removed.sum()),
                "largest_component_voxels": int(context.largest.sum()),
            }
        )
        write_mask(filtered, reference, mask_root / "filtered" / f"{cid}.nii.gz")
        write_mask(
            context.largest,
            reference,
            mask_root / "largest" / f"{cid}.nii.gz",
        )
        write_mask(safe_zone, reference, mask_root / "safe_zone" / f"{cid}.nii.gz")
        write_mask(removed, reference, mask_root / "removed" / f"{cid}.nii.gz")

    out_root.mkdir(parents=True, exist_ok=True)
    detailed_csv = out_root / "case_metrics.csv"
    summary_csv = out_root / "threshold_summary.csv"
    comparison_csv = mask_root / "metrics_comparison.csv"
    with detailed_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with comparison_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    with (out_root / "experiment_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "dataset_id": args.dataset_id,
                "bone": config["bone"],
                "prediction_dir": str(pred_dir),
                "ground_truth_dir": str(gt_dir),
                "distances_mm": distances,
                "best_by_average_dice": best,
                "written_distance_mm": write_distance,
                "written_masks": str(mask_root),
            },
            file,
            indent=2,
        )

    print(f"Detailed metrics: {detailed_csv}")
    print(f"Threshold summary: {summary_csv}")
    print(f"Full metric comparison: {comparison_csv}")
    print(
        f"Best average Dice: {best['filtered_dice']:.4f} at "
        f"{best['distance_mm']:g} mm "
        f"(change {best['dice_change']:+.4f})"
    )
    print(f"Wrote visualization masks for {write_distance:g} mm: {mask_root}")


if __name__ == "__main__":
    main()
