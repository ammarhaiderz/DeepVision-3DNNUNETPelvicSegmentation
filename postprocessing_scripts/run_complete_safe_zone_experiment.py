#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from tqdm import tqdm

from safe_zone_filter import apply_safe_zone_context, build_safe_zone_context


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BONES = {
    "LH": {
        "dataset_name": "Dataset009_anatomy_LH",
        "label": 1,
        "color": "#e15759",
    },
    "RH": {
        "dataset_name": "Dataset010_anatomy_RH",
        "label": 2,
        "color": "#4e79a7",
    },
    "S": {
        "dataset_name": "Dataset011_anatomy_S",
        "label": 3,
        "color": "#59a14f",
    },
}
MERGE_PRIORITY = ("LH", "RH", "S")
METRIC_NAMES = (
    "dice",
    "iou",
    "hd95",
    "assd",
    "false_positive_voxels",
    "false_negative_voxels",
    "prediction_voxels",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete safe-zone experiment for LH, RH, sacrum, and "
            "their merged multiclass prediction at every requested distance."
        )
    )
    parser.add_argument(
        "--distance_mm",
        type=float,
        nargs="+",
        default=[5, 10, 15, 20, 30, 40],
    )
    parser.add_argument(
        "--out_root",
        default="safe_zone_experiments/all_distances",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=3,
        help="Marching-cubes step size used for interactive HTML files.",
    )
    parser.add_argument(
        "--skip_visualizations",
        action="store_true",
        help="Generate masks and CSVs without interactive HTML files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NIfTI and HTML outputs.",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def case_id(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def image_map(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return {case_id(path): path for path in sorted(folder.glob("*.nii.gz"))}


def same_geometry(first: sitk.Image, second: sitk.Image) -> bool:
    return (
        first.GetSize() == second.GetSize()
        and np.allclose(first.GetSpacing(), second.GetSpacing())
        and np.allclose(first.GetOrigin(), second.GetOrigin())
        and np.allclose(first.GetDirection(), second.GetDirection())
    )


def write_image(
    array: np.ndarray,
    reference: sitk.Image,
    path: Path,
    overwrite: bool,
) -> None:
    if path.is_file() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    output = sitk.GetImageFromArray(array.astype(np.uint8))
    output.CopyInformation(reference)
    sitk.WriteImage(output, str(path))


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


def evaluate_mask(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing_zyx: np.ndarray,
) -> dict[str, float | int]:
    dice, iou = overlap_metrics(gt, pred)
    hd95, assd = surface_metrics(gt, pred, spacing_zyx)
    return {
        "dice": dice,
        "iou": iou,
        "hd95": hd95,
        "assd": assd,
        "false_positive_voxels": int(np.logical_and(pred, ~gt).sum()),
        "false_negative_voxels": int(np.logical_and(gt, ~pred).sum()),
        "prediction_voxels": int(pred.sum()),
    }


def comparison_fields() -> list[str]:
    fields = ["case", "distance_mm"]
    fields.extend(f"original_{name}" for name in METRIC_NAMES)
    fields.extend(f"filtered_{name}" for name in METRIC_NAMES)
    fields.extend(
        [
            "dice_change",
            "iou_change",
            "hd95_change",
            "assd_change",
            "removed_voxels",
            "largest_component_voxels",
        ]
    )
    return fields


def comparison_row(
    cid: str,
    distance_mm: float,
    original: dict[str, float | int],
    filtered: dict[str, float | int],
    removed_voxels: int,
    largest_component_voxels: int,
) -> dict[str, float | int | str]:
    return {
        "case": cid,
        "distance_mm": distance_mm,
        **{f"original_{name}": original[name] for name in METRIC_NAMES},
        **{f"filtered_{name}": filtered[name] for name in METRIC_NAMES},
        "dice_change": filtered["dice"] - original["dice"],
        "iou_change": filtered["iou"] - original["iou"],
        "hd95_change": filtered["hd95"] - original["hd95"],
        "assd_change": filtered["assd"] - original["assd"],
        "removed_voxels": removed_voxels,
        "largest_component_voxels": largest_component_voxels,
    }


def finite_mean(values) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def average_row(rows: list[dict], distance_mm: float) -> dict:
    average = {"case": "AVERAGE", "distance_mm": distance_mm}
    for field in comparison_fields()[2:]:
        average[field] = finite_mean(row[field] for row in rows)
    return average


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def merge_masks(masks: dict[str, np.ndarray]) -> np.ndarray:
    merged = np.zeros(next(iter(masks.values())).shape, dtype=np.uint8)
    for bone in MERGE_PRIORITY:
        merged[masks[bone]] = BONES[bone]["label"]
    return merged


def merged_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing_zyx: np.ndarray,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    per_class = {}
    for bone, config in BONES.items():
        values = evaluate_mask(
            gt == config["label"],
            pred == config["label"],
            spacing_zyx,
        )
        per_class[bone] = values
        for name, value in values.items():
            result[f"{bone}_{name}"] = value
    for name in METRIC_NAMES:
        result[f"mean_{name}"] = finite_mean(
            per_class[bone][name] for bone in BONES
        )
    return result


def merged_comparison_fields() -> list[str]:
    fields = ["case", "distance_mm"]
    metric_fields = [
        *(f"{bone}_{name}" for bone in BONES for name in METRIC_NAMES),
        *(f"mean_{name}" for name in METRIC_NAMES),
    ]
    fields.extend(f"original_{name}" for name in metric_fields)
    fields.extend(f"filtered_{name}" for name in metric_fields)
    for bone in BONES:
        fields.extend(
            [
                f"{bone}_dice_change",
                f"{bone}_iou_change",
                f"{bone}_hd95_change",
                f"{bone}_assd_change",
            ]
        )
    fields.extend(
        [
            "mean_dice_change",
            "mean_iou_change",
            "mean_hd95_change",
            "mean_assd_change",
            "removed_foreground_voxels",
        ]
    )
    return fields


def merged_comparison_row(
    cid: str,
    distance_mm: float,
    original: dict[str, float | int],
    filtered: dict[str, float | int],
    removed_foreground_voxels: int,
) -> dict[str, float | int | str]:
    row = {
        "case": cid,
        "distance_mm": distance_mm,
        **{f"original_{name}": value for name, value in original.items()},
        **{f"filtered_{name}": value for name, value in filtered.items()},
        "removed_foreground_voxels": removed_foreground_voxels,
    }
    for bone in BONES:
        for name in ("dice", "iou", "hd95", "assd"):
            row[f"{bone}_{name}_change"] = (
                filtered[f"{bone}_{name}"] - original[f"{bone}_{name}"]
            )
    for name in ("dice", "iou", "hd95", "assd"):
        row[f"mean_{name}_change"] = (
            filtered[f"mean_{name}"] - original[f"mean_{name}"]
        )
    return row


def mesh_from_mask(mask: np.ndarray, spacing_zyx, step_size: int):
    if not mask.any():
        return None
    padded = np.pad(mask.astype(np.uint8), 1)
    try:
        vertices, faces, _, _ = marching_cubes(
            padded,
            level=0.5,
            spacing=spacing_zyx,
            step_size=step_size,
            allow_degenerate=False,
        )
    except RuntimeError:
        if step_size == 1:
            return None
        try:
            vertices, faces, _, _ = marching_cubes(
                padded,
                level=0.5,
                spacing=spacing_zyx,
                step_size=1,
                allow_degenerate=False,
            )
        except RuntimeError:
            return None
    vertices -= np.asarray(spacing_zyx)
    return vertices, faces


def add_mesh(
    figure,
    row: int,
    col: int,
    mask: np.ndarray,
    spacing_zyx,
    step_size: int,
    name: str,
    color: str,
    opacity: float,
):
    import plotly.graph_objects as go

    mesh = mesh_from_mask(mask, spacing_zyx, step_size)
    if mesh is None:
        return
    vertices, faces = mesh
    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 2],
            y=vertices[:, 1],
            z=vertices[:, 0],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name=name,
            color=color,
            opacity=opacity,
            flatshading=True,
            showscale=False,
        ),
        row=row,
        col=col,
    )


def finish_figure(figure, title: str, columns: int) -> None:
    figure.update_layout(
        title=title,
        height=850,
        width=max(1600, columns * 430),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    for index in range(1, columns + 1):
        scene = "scene" if index == 1 else f"scene{index}"
        figure.update_layout(
            {
                scene: dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    aspectmode="data",
                )
            }
        )


def binary_figure(
    bone: str,
    cid: str,
    distance_mm: float,
    gt: np.ndarray,
    original: np.ndarray,
    largest: np.ndarray,
    safe_zone: np.ndarray,
    filtered: np.ndarray,
    removed: np.ndarray,
    spacing_zyx,
    step_size: int,
):
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=1,
        cols=5,
        specs=[[{"type": "scene"}] * 5],
        subplot_titles=[
            "Ground Truth",
            "Original Prediction",
            "Largest + Safe Zone",
            "Filtered Prediction",
            "Removed Artifacts",
        ],
        horizontal_spacing=0.01,
    )
    color = BONES[bone]["color"]
    shell = safe_zone & ~largest
    add_mesh(figure, 1, 1, gt, spacing_zyx, step_size, "GT", color, 0.75)
    add_mesh(
        figure, 1, 2, original, spacing_zyx, step_size,
        "Original", "#4e79a7", 0.75,
    )
    add_mesh(
        figure, 1, 3, largest, spacing_zyx, step_size,
        "Largest component", "#f28e2b", 0.9,
    )
    add_mesh(
        figure, 1, 3, shell, spacing_zyx, step_size,
        "Safe-zone shell", "#edc948", 0.16,
    )
    add_mesh(
        figure, 1, 4, filtered, spacing_zyx, step_size,
        "Filtered", "#59a14f", 0.8,
    )
    add_mesh(
        figure, 1, 5, removed, spacing_zyx, step_size,
        "Removed", "#e15759", 0.9,
    )
    finish_figure(
        figure,
        f"{bone} Safe Zone: {cid}, {distance_mm:g} mm",
        5,
    )
    return figure


def add_multiclass(
    figure,
    col: int,
    array: np.ndarray,
    spacing_zyx,
    step_size: int,
    prefix: str,
    opacity: float = 0.75,
):
    for bone, config in BONES.items():
        add_mesh(
            figure,
            1,
            col,
            array == config["label"],
            spacing_zyx,
            step_size,
            f"{prefix} {bone}",
            config["color"],
            opacity,
        )


def merged_figure(
    cid: str,
    distance_mm: float,
    gt: np.ndarray,
    original: np.ndarray,
    filtered: np.ndarray,
    largest: dict[str, np.ndarray],
    safe_zone: dict[str, np.ndarray],
    removed: np.ndarray,
    spacing_zyx,
    step_size: int,
):
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=1,
        cols=5,
        specs=[[{"type": "scene"}] * 5],
        subplot_titles=[
            "Ground Truth",
            "Original Merged",
            "Per-Bone Largest + Zones",
            "Filtered Merged",
            "Removed Artifacts",
        ],
        horizontal_spacing=0.01,
    )
    add_multiclass(figure, 1, gt, spacing_zyx, step_size, "GT")
    add_multiclass(figure, 2, original, spacing_zyx, step_size, "Original")
    for bone, config in BONES.items():
        add_mesh(
            figure, 1, 3, largest[bone], spacing_zyx, step_size,
            f"{bone} largest", config["color"], 0.9,
        )
        add_mesh(
            figure,
            1,
            3,
            safe_zone[bone] & ~largest[bone],
            spacing_zyx,
            step_size,
            f"{bone} safe-zone shell",
            config["color"],
            0.10,
        )
    add_multiclass(figure, 4, filtered, spacing_zyx, step_size, "Filtered")
    add_multiclass(figure, 5, removed, spacing_zyx, step_size, "Removed", 0.9)
    finish_figure(
        figure,
        f"Merged Safe Zone: {cid}, {distance_mm:g} mm",
        5,
    )
    return figure


def validate_inputs():
    prediction_maps = {}
    binary_gt_maps = {}
    for bone, config in BONES.items():
        prediction_maps[bone] = image_map(
            PROJECT_ROOT / "predictions" / config["dataset_name"]
        )
        binary_gt_maps[bone] = image_map(
            PROJECT_ROOT
            / "nnUNet"
            / "nnUNet_raw"
            / config["dataset_name"]
            / "labelsTs"
        )
    merged_gt_map = image_map(
        PROJECT_ROOT
        / "nnUNet"
        / "nnUNet_raw"
        / "Dataset002_anatomy_network"
        / "labelsTs"
    )
    case_sets = [
        set(case_map)
        for case_map in (*prediction_maps.values(), *binary_gt_maps.values())
    ]
    case_sets.append(set(merged_gt_map))
    if len({frozenset(values) for values in case_sets}) != 1:
        raise ValueError("Prediction and ground-truth case IDs do not match.")
    return prediction_maps, binary_gt_maps, merged_gt_map, sorted(case_sets[0])


def main():
    args = parse_args()
    distances = sorted(set(float(value) for value in args.distance_mm))
    if not distances or any(value < 0 for value in distances):
        raise ValueError("Distances must contain non-negative values.")
    if args.step_size < 1:
        raise ValueError("step_size must be at least 1.")
    out_root = resolve(args.out_root)
    prediction_maps, binary_gt_maps, merged_gt_map, cases = validate_inputs()

    rows_by_distance_bone: dict[tuple[float, str], list[dict]] = defaultdict(list)
    merged_rows_by_distance: dict[float, list[dict]] = defaultdict(list)
    original_binary_cache: dict[tuple[str, str], dict] = {}
    original_merged_cache: dict[str, dict] = {}

    for cid in tqdm(cases, desc="Complete safe-zone experiment", unit="case"):
        predictions = {}
        binary_gt = {}
        references = {}
        contexts = {}
        spacing_zyx = None

        for bone in BONES:
            pred_image = sitk.ReadImage(str(prediction_maps[bone][cid]))
            gt_image = sitk.ReadImage(str(binary_gt_maps[bone][cid]))
            if not same_geometry(pred_image, gt_image):
                raise ValueError(f"{bone} prediction/GT geometry mismatch: {cid}")
            if references:
                reference = next(iter(references.values()))
                if not same_geometry(reference, pred_image):
                    raise ValueError(f"Binary prediction geometry mismatch: {cid}")
            references[bone] = pred_image
            predictions[bone] = sitk.GetArrayFromImage(pred_image) > 0
            binary_gt[bone] = sitk.GetArrayFromImage(gt_image) > 0
            spacing_zyx = np.asarray(
                pred_image.GetSpacing()[::-1], dtype=np.float32
            )
            contexts[bone] = build_safe_zone_context(
                predictions[bone], spacing_zyx, max(distances)
            )
            original_binary_cache[(cid, bone)] = evaluate_mask(
                binary_gt[bone], predictions[bone], spacing_zyx
            )

        merged_gt_image = sitk.ReadImage(str(merged_gt_map[cid]))
        if not same_geometry(next(iter(references.values())), merged_gt_image):
            raise ValueError(f"Merged ground-truth geometry mismatch: {cid}")
        merged_gt = sitk.GetArrayFromImage(merged_gt_image)
        original_merged = merge_masks(predictions)
        original_merged_cache[cid] = merged_metrics(
            merged_gt, original_merged, spacing_zyx
        )

        for distance_mm in distances:
            distance_root = out_root / f"{distance_mm:g}mm"
            filtered_masks = {}
            largest_masks = {}
            safe_zones = {}
            removed_masks = {}

            for bone in BONES:
                filtered, safe_zone, removed = apply_safe_zone_context(
                    predictions[bone],
                    contexts[bone],
                    distance_mm,
                    return_safe_zone=True,
                )
                filtered_masks[bone] = filtered
                largest_masks[bone] = contexts[bone].largest
                safe_zones[bone] = safe_zone
                removed_masks[bone] = removed
                filtered_metrics = evaluate_mask(
                    binary_gt[bone], filtered, spacing_zyx
                )
                row = comparison_row(
                    cid,
                    distance_mm,
                    original_binary_cache[(cid, bone)],
                    filtered_metrics,
                    int(removed.sum()),
                    int(contexts[bone].largest.sum()),
                )
                rows_by_distance_bone[(distance_mm, bone)].append(row)

                bone_root = distance_root / bone
                write_image(
                    predictions[bone],
                    references[bone],
                    bone_root / "original" / f"{cid}.nii.gz",
                    args.overwrite,
                )
                write_image(
                    filtered,
                    references[bone],
                    bone_root / "filtered" / f"{cid}.nii.gz",
                    args.overwrite,
                )
                write_image(
                    contexts[bone].largest,
                    references[bone],
                    bone_root / "largest" / f"{cid}.nii.gz",
                    args.overwrite,
                )
                write_image(
                    safe_zone,
                    references[bone],
                    bone_root / "safe_zone" / f"{cid}.nii.gz",
                    args.overwrite,
                )
                write_image(
                    removed,
                    references[bone],
                    bone_root / "removed" / f"{cid}.nii.gz",
                    args.overwrite,
                )
                if not args.skip_visualizations:
                    html = bone_root / "visualizations" / f"{cid}.html"
                    if args.overwrite or not html.is_file():
                        html.parent.mkdir(parents=True, exist_ok=True)
                        figure = binary_figure(
                            bone,
                            cid,
                            distance_mm,
                            binary_gt[bone],
                            predictions[bone],
                            contexts[bone].largest,
                            safe_zone,
                            filtered,
                            removed,
                            spacing_zyx,
                            args.step_size,
                        )
                        figure.write_html(str(html), include_plotlyjs="cdn")

            filtered_merged = merge_masks(filtered_masks)
            removed_merged = np.zeros(original_merged.shape, dtype=np.uint8)
            largest_merged = np.zeros(original_merged.shape, dtype=np.uint8)
            safe_zone_merged = np.zeros(original_merged.shape, dtype=np.uint8)
            for bone in MERGE_PRIORITY:
                removed_merged[removed_masks[bone]] = BONES[bone]["label"]
                largest_merged[largest_masks[bone]] = BONES[bone]["label"]
                safe_zone_merged[safe_zones[bone]] = BONES[bone]["label"]
            merged_filtered_metrics = merged_metrics(
                merged_gt, filtered_merged, spacing_zyx
            )
            merged_rows_by_distance[distance_mm].append(
                merged_comparison_row(
                    cid,
                    distance_mm,
                    original_merged_cache[cid],
                    merged_filtered_metrics,
                    int(np.logical_and(original_merged > 0, filtered_merged == 0).sum()),
                )
            )
            merged_root = distance_root / "merged"
            reference = next(iter(references.values()))
            write_image(
                original_merged,
                reference,
                merged_root / "original" / f"{cid}.nii.gz",
                args.overwrite,
            )
            write_image(
                filtered_merged,
                reference,
                merged_root / "filtered" / f"{cid}.nii.gz",
                args.overwrite,
            )
            write_image(
                removed_merged,
                reference,
                merged_root / "removed" / f"{cid}.nii.gz",
                args.overwrite,
            )
            write_image(
                largest_merged,
                reference,
                merged_root / "largest" / f"{cid}.nii.gz",
                args.overwrite,
            )
            write_image(
                safe_zone_merged,
                reference,
                merged_root / "safe_zone" / f"{cid}.nii.gz",
                args.overwrite,
            )
            if not args.skip_visualizations:
                html = merged_root / "visualizations" / f"{cid}.html"
                if args.overwrite or not html.is_file():
                    html.parent.mkdir(parents=True, exist_ok=True)
                    figure = merged_figure(
                        cid,
                        distance_mm,
                        merged_gt,
                        original_merged,
                        filtered_merged,
                        largest_masks,
                        safe_zones,
                        removed_merged,
                        spacing_zyx,
                        args.step_size,
                    )
                    figure.write_html(str(html), include_plotlyjs="cdn")

    summary_rows = []
    for distance_mm in distances:
        distance_root = out_root / f"{distance_mm:g}mm"
        for bone in BONES:
            rows = rows_by_distance_bone[(distance_mm, bone)]
            rows_with_average = [*rows, average_row(rows, distance_mm)]
            write_csv(
                rows_with_average,
                distance_root / bone / "evaluation" / "metrics_comparison.csv",
                comparison_fields(),
            )
            average = rows_with_average[-1]
            summary_rows.append(
                {
                    "distance_mm": distance_mm,
                    "prediction": bone,
                    "original_dice": average["original_dice"],
                    "filtered_dice": average["filtered_dice"],
                    "dice_change": average["dice_change"],
                    "original_iou": average["original_iou"],
                    "filtered_iou": average["filtered_iou"],
                    "iou_change": average["iou_change"],
                    "original_hd95": average["original_hd95"],
                    "filtered_hd95": average["filtered_hd95"],
                    "hd95_change": average["hd95_change"],
                    "original_assd": average["original_assd"],
                    "filtered_assd": average["filtered_assd"],
                    "assd_change": average["assd_change"],
                }
            )

        merged_rows = merged_rows_by_distance[distance_mm]
        merged_fields = merged_comparison_fields()
        merged_average = {"case": "AVERAGE", "distance_mm": distance_mm}
        for field in merged_fields[2:]:
            merged_average[field] = finite_mean(
                row[field] for row in merged_rows
            )
        write_csv(
            [*merged_rows, merged_average],
            distance_root / "merged" / "evaluation" / "metrics_comparison.csv",
            merged_fields,
        )
        summary_rows.append(
            {
                "distance_mm": distance_mm,
                "prediction": "merged_mean",
                "original_dice": merged_average["original_mean_dice"],
                "filtered_dice": merged_average["filtered_mean_dice"],
                "dice_change": merged_average["mean_dice_change"],
                "original_iou": merged_average["original_mean_iou"],
                "filtered_iou": merged_average["filtered_mean_iou"],
                "iou_change": merged_average["mean_iou_change"],
                "original_hd95": merged_average["original_mean_hd95"],
                "filtered_hd95": merged_average["filtered_mean_hd95"],
                "hd95_change": merged_average["mean_hd95_change"],
                "original_assd": merged_average["original_mean_assd"],
                "filtered_assd": merged_average["filtered_mean_assd"],
                "assd_change": merged_average["mean_assd_change"],
            }
        )

    summary_fields = list(summary_rows[0])
    write_csv(summary_rows, out_root / "all_distances_summary.csv", summary_fields)
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "experiment_manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "distances_mm": distances,
                "cases": cases,
                "merge_priority_low_to_high": list(MERGE_PRIORITY),
                "metrics": list(METRIC_NAMES),
                "visualizations": not args.skip_visualizations,
                "step_size": args.step_size,
                "output_root": str(out_root),
            },
            file,
            indent=2,
        )
    print(f"Complete safe-zone experiment saved to: {out_root}")
    print(f"Cross-distance summary: {out_root / 'all_distances_summary.csv'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Existing completed outputs were kept.", file=sys.stderr)
        raise
