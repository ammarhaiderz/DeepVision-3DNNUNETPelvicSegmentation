#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


from common import (  # noqa: E402
    PELVIC1K_DATASET_ID,
    PELVIC1K_DATASET_NAME,
    PELVIC1K_TRAINER_NAME,
    PLANS_NAME,
    TRAINER_NAME,
    nnunet_paths,
    project_root,
)


SOURCE_DATASET_ID = 2
SOURCE_DATASET_NAME = "Dataset002_anatomy_network"
TARGET_DATASET_ID = 18
TARGET_DATASET_NAME = "Dataset018_PENGWIN_multiclass_anatomy"
TARGET_LABELS = {
    "background": 0,
    "left_hipbone": 1,
    "right_hipbone": 2,
    "sacrum": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a protected multiclass PENGWIN fine-tuning dataset and "
            "fine-tune it from the Pelvic1K pretrained checkpoint."
        )
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--fold", default="0")
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--continue_training", action="store_true")
    parser.add_argument("--fingerprint_workers", type=int, default=8)
    parser.add_argument("--preprocess_workers", type=int, default=8)
    parser.add_argument(
        "--pretrained_checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint. Defaults to Dataset016 checkpoint_best.pth.",
    )
    parser.add_argument(
        "--copy_raw",
        action="store_true",
        help="Copy raw files into Dataset018 instead of symlinking them.",
    )
    parser.add_argument(
        "--overwrite_raw",
        action="store_true",
        help=(
            "Overwrite existing files in Dataset018 only. This never touches "
            "Dataset002, Dataset016, or older result folders."
        ),
    )
    return parser.parse_args()


def command_path(name: str) -> str:
    path = PROJECT_ENV / "bin" / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find project command: {path}. "
            "Recreate nnunet_env if this file is missing."
        )
    return str(path)


def get_dataset_folder_by_id(nnunet_raw: Path, dataset_id: int) -> Path:
    prefix = f"Dataset{dataset_id:03d}_"
    matches = sorted(
        path for path in nnunet_raw.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    )
    if not matches:
        raise FileNotFoundError(f"Could not find {prefix} under {nnunet_raw}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple datasets match {prefix}: {matches}")
    return matches[0]


def resolve_pretrained_checkpoint(results: Path, fold: str, explicit: Path | None) -> Path:
    if explicit is not None:
        checkpoint = explicit
    else:
        checkpoint = (
            results
            / PELVIC1K_DATASET_NAME
            / f"{PELVIC1K_TRAINER_NAME}__nnUNetPlans__3d_fullres"
            / f"fold_{fold}"
            / "checkpoint_best.pth"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint}")
    return checkpoint


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


def safe_link_or_copy(source: Path, target: Path, copy_raw: bool, overwrite: bool) -> str:
    if target.exists() or target.is_symlink():
        if not overwrite:
            return "existing"
        if target.is_dir() and not target.is_symlink():
            raise IsADirectoryError(f"Refusing to overwrite directory: {target}")
        target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)
    if copy_raw:
        shutil.copy2(source, target)
        return "copied"

    relative_source = os.path.relpath(source, start=target.parent)
    target.symlink_to(relative_source)
    return "linked"


def write_dataset_json(target_raw: Path, num_training: int, overwrite: bool) -> str:
    dataset_json = target_raw / "dataset.json"
    if dataset_json.exists() and not overwrite:
        return "existing"

    data = {
        "channel_names": {"0": "CT"},
        "labels": TARGET_LABELS,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }
    target_raw.mkdir(parents=True, exist_ok=True)
    with dataset_json.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
        file.write("\n")
    return "written"


def clone_multiclass_raw(
    source_raw: Path,
    target_raw: Path,
    copy_raw: bool,
    overwrite: bool,
) -> dict[str, int | str]:
    if source_raw.resolve() == target_raw.resolve():
        raise ValueError("Source and target dataset folders must be different.")

    counters = {"linked": 0, "copied": 0, "existing": 0}
    for folder_name in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        source_folder = source_raw / folder_name
        if not source_folder.is_dir():
            raise FileNotFoundError(f"Missing source folder: {source_folder}")
        for source_file in sorted(source_folder.glob("*.nii.gz")):
            target_file = target_raw / folder_name / source_file.name
            status = safe_link_or_copy(source_file, target_file, copy_raw, overwrite)
            counters[status] += 1

    num_training = len(list((target_raw / "labelsTr").glob("*.nii.gz")))
    dataset_json_status = write_dataset_json(target_raw, num_training, overwrite)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source_raw),
        "target_dataset": str(target_raw),
        "mode": "copy" if copy_raw else "symlink",
        "overwrite_raw": overwrite,
        "dataset_json": dataset_json_status,
        "labels": TARGET_LABELS,
        "numTraining": num_training,
        "file_counts": {
            "imagesTr": len(list((target_raw / "imagesTr").glob("*.nii.gz"))),
            "labelsTr": len(list((target_raw / "labelsTr").glob("*.nii.gz"))),
            "imagesTs": len(list((target_raw / "imagesTs").glob("*.nii.gz"))),
            "labelsTs": len(list((target_raw / "labelsTs").glob("*.nii.gz"))),
        },
        **counters,
    }
    manifest_path = target_raw / "dataset018_clone_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    return manifest


def save_run_manifest(
    results: Path,
    target_dataset_name: str,
    args: argparse.Namespace,
    pretrained_checkpoint: Path,
) -> Path:
    manifest_dir = results / target_dataset_name / "run_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = (
        manifest_dir
        / f"multiclass_finetune_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    data = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_dataset_id": TARGET_DATASET_ID,
        "target_dataset_name": target_dataset_name,
        "source_dataset_id": SOURCE_DATASET_ID,
        "source_dataset_name": SOURCE_DATASET_NAME,
        "pretrain_dataset_id": PELVIC1K_DATASET_ID,
        "pretrain_dataset_name": PELVIC1K_DATASET_NAME,
        "pretrained_checkpoint": str(pretrained_checkpoint),
        "plans_name": PLANS_NAME,
        "trainer_name": TRAINER_NAME,
        "fold": args.fold,
        "num_gpus": args.num_gpus,
        "fingerprint_workers": args.fingerprint_workers,
        "preprocess_workers": args.preprocess_workers,
        "skip_preprocess": args.skip_preprocess,
        "continue_training": args.continue_training,
        "labels": TARGET_LABELS,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
    return path


def main() -> None:
    args = parse_args()
    if args.num_gpus < 1:
        raise ValueError("num_gpus must be at least 1.")
    if args.num_gpus > 2:
        raise ValueError(
            "The transferred Pelvic1K 3d_fullres plan has batch size 2. "
            "Use --num_gpus 1 or --num_gpus 2 unless you intentionally "
            "replan with a larger batch size."
        )
    if args.fingerprint_workers < 1 or args.preprocess_workers < 1:
        raise ValueError("worker counts must be at least 1.")

    nnunet_raw, nnunet_preprocessed, nnunet_results = nnunet_paths()
    source_raw = get_dataset_folder_by_id(nnunet_raw, SOURCE_DATASET_ID)
    if source_raw.name != SOURCE_DATASET_NAME:
        print(f"WARNING: expected source {SOURCE_DATASET_NAME}, found {source_raw.name}")
    target_raw = nnunet_raw / TARGET_DATASET_NAME

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

    pretrained_checkpoint = resolve_pretrained_checkpoint(
        nnunet_results,
        args.fold,
        args.pretrained_checkpoint,
    )

    clone_manifest = clone_multiclass_raw(
        source_raw,
        target_raw,
        copy_raw=args.copy_raw,
        overwrite=args.overwrite_raw,
    )
    print(f"Prepared protected raw dataset: {target_raw}")
    print(f"Clone summary: {clone_manifest}")

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
                str(TARGET_DATASET_ID),
            ],
            env,
        )
        run(
            [
                command_path("nnUNetv2_extract_fingerprint"),
                "-d",
                str(TARGET_DATASET_ID),
                "--verify_dataset_integrity",
                "-np",
                str(args.fingerprint_workers),
            ],
            env,
        )
        run(
            [
                command_path("nnUNetv2_move_plans_between_datasets"),
                "-s",
                str(PELVIC1K_DATASET_ID),
                "-t",
                str(TARGET_DATASET_ID),
                "-sp",
                "nnUNetPlans",
                "-tp",
                PLANS_NAME,
            ],
            env,
        )
        target_preprocessed = nnunet_preprocessed / TARGET_DATASET_NAME
        target_preprocessed.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            target_raw / "dataset.json",
            target_preprocessed / "dataset.json",
        )
        run(
            [
                command_path("nnUNetv2_preprocess"),
                "-d",
                str(TARGET_DATASET_ID),
                "-plans_name",
                PLANS_NAME,
                "-c",
                "3d_fullres",
                "-np",
                str(args.preprocess_workers),
            ],
            env,
        )

    verify_cuda(env, args.num_gpus)
    run_manifest = save_run_manifest(
        nnunet_results,
        TARGET_DATASET_NAME,
        args,
        pretrained_checkpoint,
    )
    print(f"Run manifest: {run_manifest}")

    train_command = [
        command_path("nnUNetv2_train"),
        str(TARGET_DATASET_ID),
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
        train_command.extend(["-pretrained_weights", str(pretrained_checkpoint)])
    run(train_command, env)


if __name__ == "__main__":
    main()
