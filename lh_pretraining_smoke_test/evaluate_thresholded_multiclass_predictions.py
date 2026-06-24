#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ENV = PROJECT_ROOT / "nnunet_env"
PROJECT_PYTHON = PROJECT_ENV / "bin" / "python"

if Path(sys.prefix).resolve() != PROJECT_ENV.resolve():
    if not PROJECT_PYTHON.is_file():
        raise FileNotFoundError(
            f"Project Python environment not found: {PROJECT_PYTHON}"
        )
    os.execv(
        str(PROJECT_PYTHON),
        [str(PROJECT_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )


from common import PLANS_NAME, nnunet_paths, project_root, strip_nii_suffix  # noqa: E402


DATASET_NAME = "Dataset018_PENGWIN_multiclass_anatomy"
LABEL_NAMES = {
    "1": "class_1_hip",
    "2": "class_2_hip",
    "3": "sacrum",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create thresholded multiclass predictions from saved nnU-Net "
            "probability maps and evaluate them."
        )
    )
    parser.add_argument(
        "--checkpoint_stem",
        default="checkpoint_best",
        help="Evaluation folder name under artifacts/pengwin_multiclass_test.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.6],
        help=(
            "Foreground confidence thresholds. A voxel is set to background "
            "unless its best foreground class probability is at least this value."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite thresholded predictions that already exist.",
    )
    return parser.parse_args()


def command_path(name: str) -> str:
    path = PROJECT_ENV / "bin" / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing command in project environment: {path}")
    return str(path)


def run(command: list[str], env: dict[str, str]) -> None:
    print("RUN:", " ".join(command))
    subprocess.run(command, check=True, env=env)


def threshold_folder_name(threshold: float) -> str:
    return f"threshold_{threshold:.2f}".replace(".", "_")


def load_probabilities(path: Path) -> np.ndarray:
    with np.load(path) as data:
        key = "probabilities" if "probabilities" in data else "softmax"
        if key not in data:
            raise KeyError(
                f"No probabilities/softmax array in {path}. Available keys: {data.files}"
            )
        probabilities = data[key]
    if probabilities.ndim != 4:
        raise ValueError(f"Expected probabilities shape (C, Z, Y, X), got {probabilities.shape}")
    if probabilities.shape[0] < 4:
        raise ValueError(
            f"Expected background + 3 foreground channels, got {probabilities.shape[0]}"
        )
    return probabilities


def write_thresholded_predictions(
    probabilities_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    threshold: float,
    overwrite: bool,
) -> int:
    probability_files = sorted(probabilities_dir.glob("*.npz"))
    if not probability_files:
        raise FileNotFoundError(
            "No .npz probability maps found. Rerun evaluate_multiclass_anatomy.py "
            "with --save_probabilities first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for npz_path in probability_files:
        case = strip_nii_suffix(npz_path.name)
        output_path = output_dir / f"{case}.nii.gz"
        if output_path.exists() and not overwrite:
            continue

        reference_path = reference_dir / f"{case}.nii.gz"
        if not reference_path.is_file():
            raise FileNotFoundError(f"Reference prediction missing: {reference_path}")

        probabilities = load_probabilities(npz_path)
        foreground = probabilities[1:]
        best_foreground = np.argmax(foreground, axis=0).astype(np.uint8) + 1
        best_foreground_probability = np.max(foreground, axis=0)
        prediction = np.where(
            best_foreground_probability >= threshold,
            best_foreground,
            0,
        ).astype(np.uint8)

        reference_image = sitk.ReadImage(str(reference_path))
        output_image = sitk.GetImageFromArray(prediction)
        output_image.CopyInformation(reference_image)
        sitk.WriteImage(output_image, str(output_path))
        written += 1

    return written


def read_summary(summary_path: Path) -> dict[str, float | str]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    row: dict[str, float | str] = {}
    foreground = data.get("foreground_mean", {})
    row["foreground_dice"] = float(foreground.get("Dice", float("nan")))
    row["foreground_iou"] = float(foreground.get("IoU", float("nan")))
    for label, metrics in sorted(data.get("mean", {}).items(), key=lambda item: int(item[0])):
        name = LABEL_NAMES.get(label, f"class_{label}")
        row[f"{name}_dice"] = float(metrics.get("Dice", float("nan")))
        row[f"{name}_iou"] = float(metrics.get("IoU", float("nan")))
    return row


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    if not rows:
        return
    fieldnames = ["threshold"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    for threshold in args.thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1: {threshold}")

    nnunet_raw, nnunet_preprocessed, nnunet_results = nnunet_paths()
    dataset = nnunet_raw / DATASET_NAME
    labels_ts = dataset / "labelsTs"
    dataset_json = dataset / "dataset.json"
    plans = nnunet_preprocessed / DATASET_NAME / f"{PLANS_NAME}.json"

    base_output = (
        project_root()
        / "lh_pretraining_smoke_test"
        / "artifacts"
        / "pengwin_multiclass_test"
        / args.checkpoint_stem
    )
    probabilities_dir = base_output / "predictions"
    for path in (labels_ts, dataset_json, plans, probabilities_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_raw)
    env["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    env["nnUNet_results"] = str(nnunet_results)

    rows: list[dict[str, float | str]] = []
    for threshold in args.thresholds:
        threshold_dir = base_output / threshold_folder_name(threshold)
        predictions = threshold_dir / "predictions"
        summary = threshold_dir / "summary.json"

        written = write_thresholded_predictions(
            probabilities_dir=probabilities_dir,
            reference_dir=probabilities_dir,
            output_dir=predictions,
            threshold=threshold,
            overwrite=args.overwrite,
        )
        print(f"Threshold {threshold:.2f}: wrote {written} prediction(s) to {predictions}")

        run(
            [
                command_path("nnUNetv2_evaluate_folder"),
                str(labels_ts),
                str(predictions),
                "-djfile",
                str(dataset_json),
                "-pfile",
                str(plans),
                "-o",
                str(summary),
            ],
            env,
        )

        row = {"threshold": f"{threshold:.2f}", **read_summary(summary)}
        rows.append(row)
        print(
            f"Threshold {threshold:.2f}: "
            f"Dice={row['foreground_dice']:.4f}, IoU={row['foreground_iou']:.4f}"
        )

        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "threshold": threshold,
            "source_probabilities": str(probabilities_dir),
            "predictions": str(predictions),
            "summary": str(summary),
            "rule": (
                "argmax over foreground classes 1..3, then background if the "
                "best foreground probability is below threshold"
            ),
        }
        with (threshold_dir / "threshold_manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
            file.write("\n")

    csv_path = base_output / "threshold_summary.csv"
    write_csv(rows, csv_path)
    print(f"\nThreshold summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
