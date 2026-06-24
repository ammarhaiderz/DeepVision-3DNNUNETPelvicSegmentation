#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import (
    FINETUNE_DATASET_ID,
    FINETUNE_DATASET_NAME,
    PLANS_NAME,
    TRAINER_NAME,
    nnunet_paths,
    project_root,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the same fine-tuned LH model on erased and normal PENGWIN "
            "test CTs."
        )
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fold", default="0")
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    return parser.parse_args()


def command_path(name: str) -> str:
    path = Path(sys.executable).resolve().parent / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {name} next to active Python: {path}."
        )
    return str(path)


def main() -> None:
    args = parse_args()
    nnunet_raw, nnunet_preprocessed, nnunet_results = nnunet_paths()
    erased_input = (
        nnunet_raw / FINETUNE_DATASET_NAME / "imagesTs"
    )
    artifacts = project_root() / "lh_pretraining_smoke_test" / "artifacts"
    normal_input = artifacts / "normal_test_images"
    output_root = artifacts / "predictions"

    for path in (erased_input, normal_input):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing test input folder: {path}")

    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_raw)
    env["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    env["nnUNet_results"] = str(nnunet_results)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    for name, input_folder in (
        ("erased_input", erased_input),
        ("normal_input", normal_input),
    ):
        output_folder = output_root / name
        command = [
            command_path("nnUNetv2_predict"),
            "-i",
            str(input_folder),
            "-o",
            str(output_folder),
            "-d",
            str(FINETUNE_DATASET_ID),
            "-c",
            "3d_fullres",
            "-f",
            args.fold,
            "-tr",
            TRAINER_NAME,
            "-p",
            PLANS_NAME,
            "-chk",
            args.checkpoint,
        ]
        print("RUN:", " ".join(command))
        subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()

