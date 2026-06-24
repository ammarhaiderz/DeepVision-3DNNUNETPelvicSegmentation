#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from common import STUDENT_DATASET_ID, find_project_root


def parse_args():
    parser = argparse.ArgumentParser(description="Train the distilled student anatomy model.")
    parser.add_argument("--dataset_id", type=int, default=STUDENT_DATASET_ID)
    parser.add_argument("--fold", default="0")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs for nnU-Net DDP training. Use 1 for single-GPU training.",
    )
    parser.add_argument("--config", default="3d_fullres")
    parser.add_argument("--distill_weight", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--continue_training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root(Path(__file__).resolve())

    nnunet_raw = project_root / "nnUNet" / "nnUNet_raw"
    nnunet_preprocessed = project_root / "nnUNet" / "nnUNet_preprocessed"
    nnunet_results = project_root / "nnUNet" / "nnUNet_results"

    base_cmd = (
        f"nnUNetv2_train {args.dataset_id} {args.config} {args.fold} "
        f"-tr nnUNetTrainer_student_teacher"
    )
    if args.num_gpus > 1:
        base_cmd = f"{base_cmd} -num_gpus {args.num_gpus}"
    if args.continue_training:
        base_cmd = f"{base_cmd} --c"
    if args.gpu is not None and args.num_gpus == 1:
        base_cmd = f"CUDA_VISIBLE_DEVICES={args.gpu} {base_cmd}"
    elif args.gpu is not None and args.num_gpus > 1:
        visible_devices = ",".join(str(args.gpu + i) for i in range(args.num_gpus))
        base_cmd = f"CUDA_VISIBLE_DEVICES={visible_devices} {base_cmd}"

    env = os.environ.copy()
    env["STUDENT_TEACHER_DISTILL_WEIGHT"] = str(args.distill_weight)
    env["STUDENT_TEACHER_TEMPERATURE"] = str(args.temperature)

    cmd = f"""
    export nnUNet_raw="{nnunet_raw}" && \
    export nnUNet_preprocessed="{nnunet_preprocessed}" && \
    export nnUNet_results="{nnunet_results}" && \
    export STUDENT_TEACHER_DISTILL_WEIGHT="{args.distill_weight}" && \
    export STUDENT_TEACHER_TEMPERATURE="{args.temperature}" && \
    {base_cmd}
    """

    print("\n=== STUDENT TRAINING START ===")
    print(f"Dataset:          {args.dataset_id}")
    print(f"Fold:             {args.fold}")
    print(f"Number of GPUs:   {args.num_gpus}")
    print(f"Distill weight:   {args.distill_weight}")
    print(f"Temperature:      {args.temperature}\n")
    subprocess.run(["bash", "-c", cmd], check=True, env=env)
    print("\n=== STUDENT TRAINING DONE ===\n")


if __name__ == "__main__":
    main()
