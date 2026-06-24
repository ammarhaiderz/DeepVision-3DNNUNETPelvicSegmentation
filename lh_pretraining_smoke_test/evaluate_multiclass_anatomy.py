#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


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


from common import PLANS_NAME, TRAINER_NAME, nnunet_paths, project_root  # noqa: E402


DATASET_ID = 18
DATASET_NAME = "Dataset018_PENGWIN_multiclass_anatomy"
LABEL_NAMES = {
    "1": "class_1_hip",
    "2": "class_2_hip",
    "3": "sacrum",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict and evaluate Dataset018 multiclass anatomy test set."
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fold", default="0")
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument("--continue_prediction", action="store_true")
    parser.add_argument("--save_probabilities", action="store_true")
    parser.add_argument("--num_processes_preprocessing", type=int, default=3)
    parser.add_argument("--num_processes_segmentation_export", type=int, default=3)
    parser.add_argument("--disable_tta", action="store_true")
    return parser.parse_args()


def command_path(name: str) -> str:
    path = PROJECT_ENV / "bin" / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing command in project environment: {path}")
    return str(path)


def run(command: list[str], env: dict[str, str]) -> None:
    print("RUN:", " ".join(command))
    subprocess.run(command, check=True, env=env)


def select_visible_devices(inherited: str | None, first_gpu: int) -> str:
    if first_gpu < 0:
        raise ValueError("gpu must be at least 0.")
    if inherited:
        allocated = [item.strip() for item in inherited.split(",") if item.strip()]
        if first_gpu >= len(allocated):
            raise ValueError(
                f"Requested logical GPU {first_gpu}, but CUDA_VISIBLE_DEVICES "
                f"only contains {len(allocated)} allocated GPU(s): {inherited}"
            )
        return allocated[first_gpu]
    return str(first_gpu)


def print_summary(summary_path: Path) -> None:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"\nSummary: {summary_path}")
    print(f"Cases: {len(data.get('metric_per_case', []))}")
    foreground = data.get("foreground_mean", {})
    print(
        "Foreground mean: "
        f"Dice={foreground.get('Dice', float('nan')):.4f}, "
        f"IoU={foreground.get('IoU', float('nan')):.4f}"
    )
    print("Class means:")
    for label, metrics in sorted(data.get("mean", {}).items(), key=lambda item: int(item[0])):
        name = LABEL_NAMES.get(label, f"class_{label}")
        print(
            f"  {label} {name}: "
            f"Dice={metrics.get('Dice', float('nan')):.4f}, "
            f"IoU={metrics.get('IoU', float('nan')):.4f}"
        )


def main() -> None:
    args = parse_args()
    nnunet_raw, nnunet_preprocessed, nnunet_results = nnunet_paths()

    dataset = nnunet_raw / DATASET_NAME
    images_ts = dataset / "imagesTs"
    labels_ts = dataset / "labelsTs"
    dataset_json = dataset / "dataset.json"
    plans = nnunet_preprocessed / DATASET_NAME / f"{PLANS_NAME}.json"
    for path in (images_ts, labels_ts, dataset_json, plans):
        if not path.exists():
            raise FileNotFoundError(f"Required evaluation artifact missing: {path}")

    checkpoint_stem = Path(args.checkpoint).stem
    output = (
        project_root()
        / "lh_pretraining_smoke_test"
        / "artifacts"
        / "pengwin_multiclass_test"
        / checkpoint_stem
    )
    predictions = output / "predictions"
    summary = output / "summary.json"
    output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_raw)
    env["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    env["nnUNet_results"] = str(nnunet_results)
    env["CUDA_VISIBLE_DEVICES"] = select_visible_devices(
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        args.gpu,
    )
    print(f"Selected CUDA device: {env['CUDA_VISIBLE_DEVICES']}")

    predict_command = [
        command_path("nnUNetv2_predict"),
        "-i",
        str(images_ts),
        "-o",
        str(predictions),
        "-d",
        str(DATASET_ID),
        "-c",
        "3d_fullres",
        "-p",
        PLANS_NAME,
        "-f",
        args.fold,
        "-tr",
        TRAINER_NAME,
        "-chk",
        args.checkpoint,
        "-npp",
        str(args.num_processes_preprocessing),
        "-nps",
        str(args.num_processes_segmentation_export),
    ]
    if args.save_probabilities:
        predict_command.append("--save_probabilities")
    if args.continue_prediction:
        predict_command.append("--continue_prediction")
    if args.disable_tta:
        predict_command.append("--disable_tta")
    run(predict_command, env)

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

    metadata = {
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "checkpoint": args.checkpoint,
        "fold": args.fold,
        "test_cases": len(list(labels_ts.glob("*.nii.gz"))),
        "predictions": str(predictions),
        "summary": str(summary),
        "label_note": (
            "Labels are numeric 1,2,3. Current left/right naming is uncertain; "
            "use class_1_hip/class_2_hip until visually confirmed."
        ),
    }
    with (output / "evaluation_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    print_summary(summary)


if __name__ == "__main__":
    main()
