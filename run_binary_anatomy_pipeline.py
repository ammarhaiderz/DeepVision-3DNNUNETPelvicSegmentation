#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path


BINARY_DATASET_IDS = [9, 10, 11]


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("\n" + "=" * 80)
    print("RUN:", " ".join(cmd))
    print("=" * 80 + "\n")
    subprocess.run(cmd, check=True, env=env)


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("nnUNet_n_proc_DA", "2")
    return env


def add_dataset_steps(steps: list[list[str]]) -> None:
    steps.append(["python", "label_engineering_scripts/generate_dataset_anatomy_binary_bones.py"])


def add_split_steps(steps: list[list[str]]) -> None:
    for dataset_id in BINARY_DATASET_IDS:
        steps.append(
            [
                "python",
                "train_val_test_nnunet_scripts/create_train_val_split.py",
                "--dataset_id",
                str(dataset_id),
            ]
        )


def add_preprocess_steps(steps: list[list[str]]) -> None:
    for dataset_id in BINARY_DATASET_IDS:
        steps.append(
            [
                "python",
                "train_val_test_nnunet_scripts/preprocess.py",
                "--dataset_id",
                str(dataset_id),
            ]
        )


def add_train_steps(steps: list[list[str]], gpu: str, fold: str) -> None:
    for dataset_id in BINARY_DATASET_IDS:
        steps.append(
            [
                "python",
                "train_val_test_nnunet_scripts/train.py",
                "--dataset_id",
                str(dataset_id),
                "--fold",
                fold,
                "--gpu",
                gpu,
            ]
        )


def add_predict_steps(steps: list[list[str]], gpu: str) -> None:
    for dataset_id in BINARY_DATASET_IDS:
        steps.append(
            [
                "python",
                "train_val_test_nnunet_scripts/predict.py",
                "--dataset_id",
                str(dataset_id),
                "--gpu",
                gpu,
            ]
        )


def add_binary_eval_steps(steps: list[list[str]]) -> None:
    for dataset_id in BINARY_DATASET_IDS:
        steps.append(
            [
                "python",
                "evaluation_scripts/eval_binary_anatomy_folder.py",
                "--dataset_id",
                str(dataset_id),
            ]
        )


def add_merge_and_main_eval_steps(steps: list[list[str]]) -> None:
    steps.append(["python", "postprocessing_scripts/merge_binary_anatomy_predictions.py"])
    steps.append(
        [
            "python",
            "evaluation_scripts/eval_anatomy_folder.py",
            "--pred_dir",
            "predictions/merged_binary_anatomy",
            "--gt_dir",
            "nnUNet/nnUNet_raw/Dataset002_anatomy_network/labelsTs",
            "--out_csv",
            "evaluation/merged_binary_anatomy.csv",
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the separate binary-bone anatomy experiment."
    )
    parser.add_argument("--gpu", default="0", help="GPU id passed to training/prediction.")
    parser.add_argument("--fold", default="0", help="nnU-Net fold to train.")
    parser.add_argument("--skip_dataset_generation", action="store_true")
    parser.add_argument("--skip_splits", action="store_true")
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_predict", action="store_true")
    parser.add_argument("--skip_binary_eval", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--skip_main_eval", action="store_true")
    parser.add_argument(
        "--only_evaluate",
        action="store_true",
        help="Run binary evaluation, merge, and merged main-task evaluation only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    steps: list[list[str]] = []

    if args.only_evaluate:
        add_binary_eval_steps(steps)
        if not args.skip_merge:
            steps.append(["python", "postprocessing_scripts/merge_binary_anatomy_predictions.py"])
        if not args.skip_main_eval:
            steps.append(
                [
                    "python",
                    "evaluation_scripts/eval_anatomy_folder.py",
                    "--pred_dir",
                    "predictions/merged_binary_anatomy",
                    "--gt_dir",
                    "nnUNet/nnUNet_raw/Dataset002_anatomy_network/labelsTs",
                    "--out_csv",
                    "evaluation/merged_binary_anatomy.csv",
                ]
            )
    else:
        if not args.skip_dataset_generation:
            add_dataset_steps(steps)
        if not args.skip_splits:
            add_split_steps(steps)
        if not args.skip_preprocess:
            add_preprocess_steps(steps)
        if not args.skip_train:
            add_train_steps(steps, args.gpu, args.fold)
        if not args.skip_predict:
            add_predict_steps(steps, args.gpu)
        if not args.skip_binary_eval:
            add_binary_eval_steps(steps)
        if not args.skip_merge:
            steps.append(["python", "postprocessing_scripts/merge_binary_anatomy_predictions.py"])
        if not args.skip_main_eval:
            steps.append(
                [
                    "python",
                    "evaluation_scripts/eval_anatomy_folder.py",
                    "--pred_dir",
                    "predictions/merged_binary_anatomy",
                    "--gt_dir",
                    "nnUNet/nnUNet_raw/Dataset002_anatomy_network/labelsTs",
                    "--out_csv",
                    "evaluation/merged_binary_anatomy.csv",
                ]
            )

    if not steps:
        print("No steps selected.")
        return

    env = build_env()
    for step in steps:
        run(step, env)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nPipeline failed at: {' '.join(e.cmd)}")
        sys.exit(e.returncode)
