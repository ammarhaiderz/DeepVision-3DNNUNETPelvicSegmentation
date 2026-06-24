#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
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


from common import (
    FINETUNE_DATASET_ID,
    FINETUNE_DATASET_NAME,
    PELVIC1K_DATASET_ID,
    PELVIC1K_DATASET_NAME,
    PELVIC1K_TRAINER_NAME,
    PLANS_NAME,
    TRAINER_NAME,
    nnunet_paths,
    project_root,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reuse Pelvic1K plans, load its pretrained feature weights, and "
            "fine-tune the binary erased-LH model."
        )
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--fold", default="0")
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--continue_training", action="store_true")
    return parser.parse_args()


def command_path(name: str) -> str:
    path = PROJECT_ENV / "bin" / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find project command: {path}. "
            "Recreate nnunet_env if this file is missing."
        )
    return str(path)


def checkpoint_path(
    results: Path,
    dataset_name: str,
    plans_name: str,
    fold: str,
) -> Path:
    path = (
        results
        / dataset_name
        / f"{PELVIC1K_TRAINER_NAME}__{plans_name}__3d_fullres"
        / f"fold_{fold}"
        / "checkpoint_best.pth"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Pretraining checkpoint not found: {path}")
    return path


def run(command: list[str], env: dict[str, str]) -> None:
    print("RUN:", " ".join(command))
    subprocess.run(command, check=True, env=env)


def select_visible_devices(
    inherited: str | None,
    first_gpu: int,
    num_gpus: int,
) -> str:
    if first_gpu < 0:
        raise ValueError("gpu must be at least 0.")
    if inherited:
        allocated = [item.strip() for item in inherited.split(",") if item.strip()]
        selected = allocated[first_gpu:first_gpu + num_gpus]
        if len(selected) != num_gpus:
            raise ValueError(
                f"Requested {num_gpus} GPU(s) starting at logical GPU "
                f"{first_gpu}, but CUDA_VISIBLE_DEVICES only contains "
                f"{len(allocated)} allocated GPU(s): {inherited}"
            )
        return ",".join(selected)
    return ",".join(str(first_gpu + offset) for offset in range(num_gpus))


def verify_cuda(env: dict[str, str], num_gpus: int) -> None:
    code = f"""
import torch

required = {num_gpus}
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the project environment.")
available = torch.cuda.device_count()
if available < required:
    raise RuntimeError(
        f"Requested {{required}} GPU(s), but PyTorch sees only {{available}}."
    )
for index in range(required):
    tensor = torch.empty(1, device=f"cuda:{{index}}")
    print(f"GPU {{index}}: {{torch.cuda.get_device_name(index)}}")
del tensor
"""
    run([str(PROJECT_PYTHON), "-c", code], env)


def main() -> None:
    args = parse_args()
    if args.num_gpus < 1:
        raise ValueError("num_gpus must be at least 1.")

    nnunet_raw, nnunet_preprocessed, nnunet_results = nnunet_paths()
    if not (nnunet_raw / FINETUNE_DATASET_NAME).is_dir():
        raise FileNotFoundError(
            f"Prepare {FINETUNE_DATASET_NAME} before fine-tuning."
        )

    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_raw)
    env["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    env["nnUNet_results"] = str(nnunet_results)
    env["CUDA_VISIBLE_DEVICES"] = select_visible_devices(
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        args.gpu,
        args.num_gpus,
    )
    print(f"Selected CUDA devices: {env['CUDA_VISIBLE_DEVICES']}")
    verify_cuda(env, args.num_gpus)

    if not args.skip_preprocess:
        run(
            [
                str(PROJECT_PYTHON),
                str(
                    project_root()
                    / "train_val_test_nnunet_scripts"
                    / "create_train_val_split.py"
                ),
                "--dataset_id",
                str(FINETUNE_DATASET_ID),
            ],
            env,
        )
        run(
            [
                command_path("nnUNetv2_extract_fingerprint"),
                "-d",
                str(FINETUNE_DATASET_ID),
                "--verify_dataset_integrity",
                "-np",
                "1",
            ],
            env,
        )
        run(
            [
                command_path("nnUNetv2_move_plans_between_datasets"),
                "-s",
                str(PELVIC1K_DATASET_ID),
                "-t",
                str(FINETUNE_DATASET_ID),
                "-sp",
                "nnUNetPlans",
                "-tp",
                PLANS_NAME,
            ],
            env,
        )
        shutil.copyfile(
            nnunet_raw / FINETUNE_DATASET_NAME / "dataset.json",
            nnunet_preprocessed / FINETUNE_DATASET_NAME / "dataset.json",
        )
        run(
            [
                command_path("nnUNetv2_preprocess"),
                "-d",
                str(FINETUNE_DATASET_ID),
                "-plans_name",
                PLANS_NAME,
                "-c",
                "3d_fullres",
                "-np",
                "1",
            ],
            env,
        )

    train_command = [
        command_path("nnUNetv2_train"),
        str(FINETUNE_DATASET_ID),
        "3d_fullres",
        args.fold,
        "-tr",
        TRAINER_NAME,
        "-p",
        PLANS_NAME,
        "-num_gpus",
        str(args.num_gpus),
    ]
    if args.continue_training:
        train_command.append("--c")
    else:
        pretrained = checkpoint_path(
            nnunet_results,
            PELVIC1K_DATASET_NAME,
            "nnUNetPlans",
            args.fold,
        )
        train_command.extend(["-pretrained_weights", str(pretrained)])
    run(train_command, env)


if __name__ == "__main__":
    main()
