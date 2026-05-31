#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from original_eval_script import evaluate_3d_single_case


DEFAULT_GT_DATASET = "Dataset001_original_data"
DEFAULT_GT_SUBDIR = "labelsTs"


def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "nnUNet").exists():
            return parent
    raise FileNotFoundError(
        "Could not find project root. Expected a parent folder containing 'nnUNet'."
    )


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate restored prediction folders.\n\n"
            "Default behavior:\n"
            "- reads all valid subfolders inside ./restorations\n"
            "- uses ./nnUNet/nnUNet_raw/Dataset001_original_data/labelsTs as GT\n"
            "- writes one CSV per restoration folder into ./evaluation\n\n"
            "You can also evaluate a single folder by passing --pred_dir."
        )
    )
    parser.add_argument(
        "--pred_root",
        type=str,
        default=None,
        help="Root folder containing restoration subfolders.",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=None,
        help="Evaluate only this one prediction folder.",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Ground-truth folder. Default: nnUNet/nnUNet_raw/Dataset001_original_data/labelsTs",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="Output folder for CSV files. Default: ./evaluation",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional explicit CSV path when using --pred_dir.",
    )
    parser.add_argument("--ignore_missing", action="store_true")
    parser.add_argument("--verbose_cases", action="store_true")
    return parser.parse_args()


def resolve_user_path(path_str: str | None, default: Path) -> Path:
    if path_str is None:
        return default

    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_paths(args):
    pred_root = resolve_user_path(args.pred_root, PROJECT_ROOT / "restorations")
    gt_dir = resolve_user_path(
        args.gt_dir,
        PROJECT_ROOT / "nnUNet" / "nnUNet_raw" / DEFAULT_GT_DATASET / DEFAULT_GT_SUBDIR,
    )
    out_root = resolve_user_path(args.out_root, PROJECT_ROOT / "evaluation")
    return pred_root, gt_dir, out_root


def list_image_files(folder: Path):
    return sorted(
        list(folder.glob("*.nii.gz")) +
        list(folder.glob("*.nii")) +
        list(folder.glob("*.mha"))
    )


def folder_contains_prediction_files(folder: Path) -> bool:
    return len(list_image_files(folder)) > 0


def is_valid_prediction_folder(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    if folder.name.startswith("."):
        return False
    if folder.name == "__pycache__":
        return False
    if not folder_contains_prediction_files(folder):
        return False
    return True


def read_image_and_spacing(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()[::-1]
    return arr, spacing


def count_instances(arr):
    labels = np.unique(arr)
    labels = labels[labels > 0]
    return int(len(labels))


def format_metric(x):
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf"
    return f"{x:.3f}"


def mean_ignore_nan(values):
    vals = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def get_case_id(path: Path):
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    if name.endswith(".mha"):
        return name[:-4]
    return path.stem


def evaluate_folder(
    pred_dir: Path,
    gt_dir: Path,
    out_csv: Path,
    ignore_missing: bool,
    verbose_cases: bool,
):
    pred_files = list_image_files(pred_dir)
    gt_files = list_image_files(gt_dir)

    pred_map = {get_case_id(p): p for p in pred_files}
    gt_map = {get_case_id(p): p for p in gt_files}
    case_ids = sorted(gt_map.keys())

    rows = []
    iou_a_vals, iou_f_vals = [], []
    assd_a_vals, assd_f_vals = [], []
    hd95_a_vals, hd95_f_vals = [], []

    print(f"\nEvaluating folder: {pred_dir.name}")
    print(f"GT folder: {gt_dir}")
    print(f"Cases to check: {len(case_ids)}\n")

    for cid in tqdm(case_ids, desc=f"Cases ({pred_dir.name})"):
        if cid not in pred_map:
            if ignore_missing:
                continue
            raise FileNotFoundError(f"Missing prediction for {cid} in {pred_dir}")

        gt, spacing = read_image_and_spacing(gt_map[cid])
        pred, _ = read_image_and_spacing(pred_map[cid])

        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch for {cid}: pred {pred.shape}, gt {gt.shape}")

        pred_count = count_instances(pred)
        gt_count = count_instances(gt)

        metrics = evaluate_3d_single_case(
            gt_volume=gt,
            pred_volume=pred,
            spacing=spacing,
            verbose=verbose_cases,
        )

        iou_a = metrics["anatomical_iou"]
        iou_f = metrics["fracture_iou"]
        assd_a = metrics["anatomical_assd"]
        assd_f = metrics["fracture_assd"]
        hd95_a = metrics["anatomical_hd95"]
        hd95_f = metrics["fracture_hd95"]

        rows.append(
            {
                "case": cid,
                "pred_count": pred_count,
                "gt_count": gt_count,
                "IoU-A ↑": iou_a,
                "IoU-F ↑": iou_f,
                "ASSD-A ↓": assd_a,
                "ASSD-F ↓": assd_f,
                "HD95-A ↓": hd95_a,
                "HD95-F ↓": hd95_f,
            }
        )

        iou_a_vals.append(iou_a)
        iou_f_vals.append(iou_f)
        assd_a_vals.append(assd_a)
        assd_f_vals.append(assd_f)
        hd95_a_vals.append(hd95_a)
        hd95_f_vals.append(hd95_f)

    fieldnames = [
        "case",
        "pred_count",
        "gt_count",
        "IoU-A ↑",
        "IoU-F ↑",
        "ASSD-A ↓",
        "ASSD-F ↓",
        "HD95-A ↓",
        "HD95-F ↓",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            writer.writerow(
                {
                    "case": r["case"],
                    "pred_count": int(r["pred_count"]),
                    "gt_count": int(r["gt_count"]),
                    "IoU-A ↑": format_metric(r["IoU-A ↑"]),
                    "IoU-F ↑": format_metric(r["IoU-F ↑"]),
                    "ASSD-A ↓": format_metric(r["ASSD-A ↓"]),
                    "ASSD-F ↓": format_metric(r["ASSD-F ↓"]),
                    "HD95-A ↓": format_metric(r["HD95-A ↓"]),
                    "HD95-F ↓": format_metric(r["HD95-F ↓"]),
                }
            )

        if rows:
            writer.writerow(
                {
                    "case": "AVERAGE",
                    "pred_count": format_metric(np.mean([r["pred_count"] for r in rows])),
                    "gt_count": format_metric(np.mean([r["gt_count"] for r in rows])),
                    "IoU-A ↑": format_metric(mean_ignore_nan(iou_a_vals)),
                    "IoU-F ↑": format_metric(mean_ignore_nan(iou_f_vals)),
                    "ASSD-A ↓": format_metric(mean_ignore_nan(assd_a_vals)),
                    "ASSD-F ↓": format_metric(mean_ignore_nan(assd_f_vals)),
                    "HD95-A ↓": format_metric(mean_ignore_nan(hd95_a_vals)),
                    "HD95-F ↓": format_metric(mean_ignore_nan(hd95_f_vals)),
                }
            )

    print(f"Saved CSV: {out_csv}")


def main():
    args = parse_args()
    pred_root, gt_dir, out_root = resolve_paths(args)

    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth folder not found: {gt_dir}")

    if args.pred_dir:
        pred_dir = resolve_user_path(args.pred_dir, PROJECT_ROOT / "restorations")
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"Prediction folder not found: {pred_dir}")
        if not folder_contains_prediction_files(pred_dir):
            raise FileNotFoundError(f"No prediction image files found in: {pred_dir}")

        out_csv = (
            resolve_user_path(args.out_csv, out_root / f"{pred_dir.name}.csv")
            if args.out_csv
            else out_root / f"{pred_dir.name}.csv"
        )

        evaluate_folder(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            out_csv=out_csv,
            ignore_missing=args.ignore_missing,
            verbose_cases=args.verbose_cases,
        )
        return

    if not pred_root.is_dir():
        raise FileNotFoundError(f"Restorations root not found: {pred_root}")

    pred_dirs = sorted([p for p in pred_root.iterdir() if is_valid_prediction_folder(p)])

    if not pred_dirs:
        print(f"No valid restoration folders found in: {pred_root}")
        return

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Restorations root: {pred_root}")
    print(f"GT dir: {gt_dir}")
    print(f"Evaluation output root: {out_root}")

    skipped_dirs = sorted(
        [p for p in pred_root.iterdir() if p.is_dir() and not is_valid_prediction_folder(p)]
    )
    if skipped_dirs:
        print("\nSkipping non-prediction folders:")
        for p in skipped_dirs:
            print(f"  - {p.name}")

    for pred_dir in pred_dirs:
        out_csv = out_root / f"{pred_dir.name}.csv"
        evaluate_folder(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            out_csv=out_csv,
            ignore_missing=args.ignore_missing,
            verbose_cases=args.verbose_cases,
        )


if __name__ == "__main__":
    main()