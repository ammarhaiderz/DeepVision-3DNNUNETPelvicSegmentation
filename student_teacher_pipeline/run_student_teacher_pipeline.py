#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


TEACHER_DATASET_IDS = [12, 13, 14]
STUDENT_DATASET_ID = 15


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the erased-bone teacher and distilled-student nnU-Net pipeline."
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Backward-compatible default for both teacher and student GPU counts.",
    )
    parser.add_argument(
        "--teacher_num_gpus",
        type=int,
        default=None,
        help="Number of GPUs per teacher training job. Defaults to --num_gpus.",
    )
    parser.add_argument(
        "--student_num_gpus",
        type=int,
        default=None,
        help="Number of GPUs for student training. Defaults to --num_gpus.",
    )
    parser.add_argument(
        "--parallel_teacher_train",
        action="store_true",
        help="Train LH/RH/S teachers at the same time on separate GPU ranges.",
    )
    parser.add_argument(
        "--parallel_teacher_predict",
        action="store_true",
        help="Run LH/RH/S teacher prior prediction at the same time on separate GPUs.",
    )
    parser.add_argument("--fold", default="0")
    parser.add_argument("--erase_value", type=float, default=0.0)
    parser.add_argument("--distill_weight", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--skip_teacher_datasets", action="store_true")
    parser.add_argument("--skip_teacher_splits", action="store_true")
    parser.add_argument("--skip_teacher_preprocess", action="store_true")
    parser.add_argument("--skip_teacher_train", action="store_true")
    parser.add_argument("--skip_teacher_predict", action="store_true")
    parser.add_argument("--skip_manifest", action="store_true")
    parser.add_argument("--skip_student_dataset", action="store_true")
    parser.add_argument("--skip_student_split", action="store_true")
    parser.add_argument("--skip_student_preprocess", action="store_true")
    parser.add_argument("--skip_student_train", action="store_true")
    parser.add_argument("--install_trainer", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("\n" + "=" * 80)
    print("RUN:", " ".join(cmd))
    print("=" * 80 + "\n")
    subprocess.run(cmd, check=True, env=env)


def run_parallel(commands: list[list[str]], env: dict[str, str]) -> None:
    print("\n" + "=" * 80)
    print("RUNNING IN PARALLEL:")
    for cmd in commands:
        print("  " + " ".join(cmd))
    print("=" * 80 + "\n")

    processes = [subprocess.Popen(cmd, env=env) for cmd in commands]
    failed: list[tuple[list[str], int]] = []
    for cmd, process in zip(commands, processes):
        return_code = process.wait()
        if return_code != 0:
            failed.append((cmd, return_code))

    if failed:
        for cmd, return_code in failed:
            print(f"FAILED ({return_code}): {' '.join(cmd)}")
        raise subprocess.CalledProcessError(failed[0][1], failed[0][0])


def gpu_for_teacher(start_gpu: int, teacher_index: int, teacher_num_gpus: int) -> str:
    return str(start_gpu + teacher_index * teacher_num_gpus)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    teacher_num_gpus = args.teacher_num_gpus if args.teacher_num_gpus is not None else args.num_gpus
    student_num_gpus = args.student_num_gpus if args.student_num_gpus is not None else args.num_gpus
    start_gpu = int(args.gpu)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("nnUNet_n_proc_DA", "2")

    steps: list[list[str]] = []
    if args.install_trainer:
        steps.append(["python", "student_teacher_pipeline/set_up_env/install_student_teacher_trainer.py"])

    if not args.skip_teacher_datasets:
        steps.append(
            [
                "python",
                "student_teacher_pipeline/scripts/generate_erased_bone_teacher_datasets.py",
                "--erase_value",
                str(args.erase_value),
            ]
        )

    if not args.skip_teacher_splits:
        for dataset_id in TEACHER_DATASET_IDS:
            steps.append(
                [
                    "python",
                    "train_val_test_nnunet_scripts/create_train_val_split.py",
                    "--dataset_id",
                    str(dataset_id),
                ]
            )

    if not args.skip_teacher_preprocess:
        for dataset_id in TEACHER_DATASET_IDS:
            steps.append(
                [
                    "python",
                    "train_val_test_nnunet_scripts/preprocess.py",
                    "--dataset_id",
                    str(dataset_id),
                ]
            )

    if not args.skip_teacher_train:
        teacher_train_steps = []
        for teacher_index, dataset_id in enumerate(TEACHER_DATASET_IDS):
            teacher_train_steps.append(
                [
                    "python",
                    "train_val_test_nnunet_scripts/train.py",
                    "--dataset_id",
                    str(dataset_id),
                    "--fold",
                    args.fold,
                    "--gpu",
                    gpu_for_teacher(start_gpu, teacher_index, teacher_num_gpus)
                    if args.parallel_teacher_train
                    else args.gpu,
                    "--num_gpus",
                    str(teacher_num_gpus),
                    "--trainer",
                    "nnUNetTrainer_no_rotation",
                ]
            )
        if args.parallel_teacher_train:
            steps.append(["__parallel__teacher_train__"])
        else:
            steps.extend(teacher_train_steps)
    else:
        teacher_train_steps = []

    if not args.skip_teacher_predict:
        teacher_predict_steps = []
        for teacher_index, dataset_id in enumerate(TEACHER_DATASET_IDS):
            teacher_predict_steps.append(
                [
                    "python",
                    "student_teacher_pipeline/scripts/predict_teacher_priors.py",
                    "--dataset_id",
                    str(dataset_id),
                    "--split",
                    "train",
                    "--fold",
                    args.fold,
                    "--gpu",
                    gpu_for_teacher(start_gpu, teacher_index, 1)
                    if args.parallel_teacher_predict
                    else args.gpu,
                ]
            )
        if args.parallel_teacher_predict:
            steps.append(["__parallel__teacher_predict__"])
        else:
            steps.extend(teacher_predict_steps)
    else:
        teacher_predict_steps = []

    if not args.skip_manifest:
        steps.append(["python", "student_teacher_pipeline/scripts/build_student_teacher_prior_manifest.py"])

    if not args.skip_student_dataset:
        steps.append(["python", "student_teacher_pipeline/scripts/generate_student_dataset.py"])

    if not args.skip_student_split:
        steps.append(
            [
                "python",
                "train_val_test_nnunet_scripts/create_train_val_split.py",
                "--dataset_id",
                str(STUDENT_DATASET_ID),
            ]
        )

    if not args.skip_student_preprocess:
        steps.append(
            [
                "python",
                "train_val_test_nnunet_scripts/preprocess.py",
                "--dataset_id",
                str(STUDENT_DATASET_ID),
            ]
        )

    if not args.skip_student_train:
        steps.append(
            [
                "python",
                "student_teacher_pipeline/scripts/train_student.py",
                "--dataset_id",
                str(STUDENT_DATASET_ID),
                "--fold",
                args.fold,
                "--gpu",
                args.gpu,
                "--num_gpus",
                str(student_num_gpus),
                "--distill_weight",
                str(args.distill_weight),
                "--temperature",
                str(args.temperature),
            ]
        )

    if not steps:
        print("No steps selected.")
        return

    for step in steps:
        if step == ["__parallel__teacher_train__"]:
            run_parallel(teacher_train_steps, env)
        elif step == ["__parallel__teacher_predict__"]:
            run_parallel(teacher_predict_steps, env)
        else:
            run(step, env)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nPipeline failed at: {' '.join(e.cmd)}")
        sys.exit(e.returncode)
