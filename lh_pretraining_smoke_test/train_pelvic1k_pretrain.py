#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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


from common import (
    PELVIC1K_DATASET_ID,
    PELVIC1K_DATASET_NAME,
    PELVIC1K_TRAINER_NAME,
    nnunet_paths,
    project_root,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess and train the CTPelvic1K multiclass model."
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
print(f"PyTorch: {{torch.__version__}} (CUDA runtime {{torch.version.cuda}})")
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Check the NVIDIA driver and the installed "
        "PyTorch CUDA build before preprocessing or training."
    )
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


def install_trainer() -> Path:
    source = (
        project_root()
        / "set_up_env"
        / f"{PELVIC1K_TRAINER_NAME}.py"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Missing pretraining trainer: {source}")
    import nnunetv2

    target = (
        Path(nnunetv2.__file__).resolve().parent
        / "training"
        / "nnUNetTrainer"
        / source.name
    )
    shutil.copyfile(source, target)
    return target


def save_run_manifest(
    args: argparse.Namespace,
    env: dict[str, str],
    trainer_path: Path,
) -> Path:
    _, _, results = nnunet_paths()
    run_dir = (
        results
        / PELVIC1K_DATASET_NAME
        / "run_manifests"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = run_dir / f"pretraining_{timestamp}.json"
    split_path = (
        Path(env["nnUNet_preprocessed"])
        / PELVIC1K_DATASET_NAME
        / "splits_final.json"
    )
    split_sha256 = hashlib.sha256(split_path.read_bytes()).hexdigest()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root(),
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=project_root(),
        check=False,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    manifest = {
        "created_utc": timestamp,
        "dataset_id": PELVIC1K_DATASET_ID,
        "dataset_name": PELVIC1K_DATASET_NAME,
        "trainer": PELVIC1K_TRAINER_NAME,
        "trainer_source": str(trainer_path),
        "configuration": "3d_fullres",
        "fold": args.fold,
        "gpu": args.gpu,
        "num_gpus": args.num_gpus,
        "continue_training": args.continue_training,
        "skip_preprocess": args.skip_preprocess,
        "python": sys.version,
        "split_file": str(split_path),
        "split_sha256": split_sha256,
        "git_commit": git_commit or None,
        "git_status": git_status,
        "environment": {
            "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
            "nnUNet_raw": env["nnUNet_raw"],
            "nnUNet_preprocessed": env["nnUNet_preprocessed"],
            "nnUNet_results": env["nnUNet_results"],
        },
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return path


def main() -> None:
    args = parse_args()
    if args.num_gpus < 1:
        raise ValueError("num_gpus must be at least 1.")

    nnunet_raw, nnunet_preprocessed, nnunet_results = nnunet_paths()
    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_raw)
    env["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    env["nnUNet_results"] = str(nnunet_results)
    env["CUDA_VISIBLE_DEVICES"] = select_visible_devices(
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        args.gpu,
        args.num_gpus,
    )
    env.setdefault("MPLCONFIGDIR", str(project_root() / ".matplotlib"))
    print(f"Selected CUDA devices: {env['CUDA_VISIBLE_DEVICES']}")
    verify_cuda(env, args.num_gpus)
    trainer_path = install_trainer()

    if not args.skip_preprocess:
        run(
            [
                command_path("nnUNetv2_plan_and_preprocess"),
                "-d",
                str(PELVIC1K_DATASET_ID),
                "-c",
                "3d_fullres",
                "--verify_dataset_integrity",
                "-np",
                "1",
                "-npfp",
                "1",
            ],
            env,
        )

    manifest = save_run_manifest(args, env, trainer_path)
    print(f"Run manifest: {manifest}")
    train_command = [
        command_path("nnUNetv2_train"),
        str(PELVIC1K_DATASET_ID),
        "3d_fullres",
        args.fold,
        "-tr",
        PELVIC1K_TRAINER_NAME,
        "-num_gpus",
        str(args.num_gpus),
    ]
    if args.continue_training:
        train_command.append("--c")
    run(train_command, env)


if __name__ == "__main__":
    main()
