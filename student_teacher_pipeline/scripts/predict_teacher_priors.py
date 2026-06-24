#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import dataset_folder_by_id, find_project_root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run teacher prediction and save soft probability maps."
    )
    parser.add_argument("--dataset_id", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--fold", default="0")
    parser.add_argument("--config", default="3d_fullres")
    parser.add_argument("--trainer", default="nnUNetTrainer_no_rotation")
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument("--save_probabilities", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root(Path(__file__).resolve())

    nnunet_root = project_root / "nnUNet"
    nnunet_raw = nnunet_root / "nnUNet_raw"
    nnunet_preprocessed = nnunet_root / "nnUNet_preprocessed"
    nnunet_results = nnunet_root / "nnUNet_results"

    dataset_folder = dataset_folder_by_id(nnunet_raw, args.dataset_id)
    input_folder = dataset_folder / ("imagesTr" if args.split == "train" else "imagesTs")
    output_folder = project_root / "student_teacher_pipeline" / "teacher_priors" / args.split / dataset_folder.name
    output_folder.mkdir(parents=True, exist_ok=True)

    base_cmd = (
        f'nnUNetv2_predict '
        f'-i "{input_folder}" '
        f'-o "{output_folder}" '
        f'-d {args.dataset_id} '
        f'-c {args.config} '
        f'-f {args.fold} '
        f'-tr {args.trainer} '
        f'-chk {args.checkpoint} '
        f'--save_probabilities'
    )
    if args.gpu is not None:
        base_cmd = f"CUDA_VISIBLE_DEVICES={args.gpu} {base_cmd}"

    cmd = f"""
    export nnUNet_raw="{nnunet_raw}" && \
    export nnUNet_preprocessed="{nnunet_preprocessed}" && \
    export nnUNet_results="{nnunet_results}" && \
    {base_cmd}
    """

    print("\n=== TEACHER PRIOR PREDICTION START ===")
    print(f"Dataset: {dataset_folder.name}")
    print(f"Split:   {args.split}")
    print(f"Input:   {input_folder}")
    print(f"Output:  {output_folder}\n")
    subprocess.run(["bash", "-c", cmd], check=True)
    print("\n=== TEACHER PRIOR PREDICTION DONE ===\n")


if __name__ == "__main__":
    main()

