#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree

from common import (
    FINETUNE_DATASET_NAME,
    nnunet_paths,
    project_root,
    strip_nii_suffix,
)


def read_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    spacing = np.asarray(image.GetSpacing()[::-1], dtype=np.float32)
    return array > 0, spacing


def overlap_metrics(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    intersection = int(np.logical_and(gt, pred).sum())
    denom = int(gt.sum()) + int(pred.sum())
    union = int(np.logical_or(gt, pred).sum())
    dice = 1.0 if denom == 0 else 2.0 * intersection / denom
    iou = 1.0 if union == 0 else intersection / union
    return float(dice), float(iou)


def surface_points(mask: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.empty((0, 3), dtype=np.float32)
    surface = mask & ~binary_erosion(
        mask, structure=generate_binary_structure(3, 1)
    )
    return np.argwhere(surface).astype(np.float32) * spacing


def surface_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: np.ndarray,
) -> tuple[float, float]:
    gt_points = surface_points(gt, spacing)
    pred_points = surface_points(pred, spacing)
    if not len(gt_points) and not len(pred_points):
        return 0.0, 0.0
    if not len(gt_points) or not len(pred_points):
        return float("inf"), float("inf")

    gt_to_pred = cKDTree(pred_points).query(gt_points, k=1)[0]
    pred_to_gt = cKDTree(gt_points).query(pred_points, k=1)[0]
    hd95 = max(
        np.percentile(gt_to_pred, 95),
        np.percentile(pred_to_gt, 95),
    )
    assd = (gt_to_pred.mean() + pred_to_gt.mean()) / 2
    return float(hd95), float(assd)


def prediction_map(folder: Path) -> dict[str, Path]:
    return {
        strip_nii_suffix(path.name): path
        for path in sorted(folder.glob("*.nii.gz"))
    }


def main() -> None:
    nnunet_raw, _, _ = nnunet_paths()
    labels = nnunet_raw / FINETUNE_DATASET_NAME / "labelsTs"
    artifacts = project_root() / "lh_pretraining_smoke_test" / "artifacts"
    predictions = {
        "erased": prediction_map(artifacts / "predictions" / "erased_input"),
        "normal": prediction_map(artifacts / "predictions" / "normal_input"),
    }
    if not labels.is_dir():
        raise FileNotFoundError(f"Missing binary test labels: {labels}")

    rows = []
    for gt_path in sorted(labels.glob("*.nii.gz")):
        case_id = strip_nii_suffix(gt_path.name)
        gt, spacing = read_mask(gt_path)
        row = {"case": case_id}
        for input_name, pred_map in predictions.items():
            pred_path = pred_map.get(case_id)
            if pred_path is None:
                raise FileNotFoundError(
                    f"Missing {input_name} prediction for {case_id}"
                )
            pred, pred_spacing = read_mask(pred_path)
            if pred.shape != gt.shape or not np.allclose(pred_spacing, spacing):
                raise ValueError(
                    f"Prediction geometry mismatch for {case_id} ({input_name})"
                )
            dice, iou = overlap_metrics(gt, pred)
            hd95, assd = surface_metrics(gt, pred, spacing)
            row.update(
                {
                    f"{input_name}_dice": dice,
                    f"{input_name}_iou": iou,
                    f"{input_name}_hd95": hd95,
                    f"{input_name}_assd": assd,
                    f"{input_name}_false_positive_voxels": int(
                        np.logical_and(pred, ~gt).sum()
                    ),
                    f"{input_name}_false_negative_voxels": int(
                        np.logical_and(gt, ~pred).sum()
                    ),
                }
            )
        row["dice_change_normal_minus_erased"] = (
            row["normal_dice"] - row["erased_dice"]
        )
        rows.append(row)

    if not rows:
        raise RuntimeError("No test cases were evaluated.")

    numeric_fields = [key for key in rows[0] if key != "case"]
    average = {"case": "AVERAGE"}
    for field in numeric_fields:
        finite = [
            float(row[field])
            for row in rows
            if np.isfinite(float(row[field]))
        ]
        average[field] = float(np.mean(finite)) if finite else float("nan")

    output = artifacts / "evaluation_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["case", *numeric_fields])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(average)
    print(f"Saved comparison: {output}")


if __name__ == "__main__":
    main()
