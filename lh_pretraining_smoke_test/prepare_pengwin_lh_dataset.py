#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import (
    FINETUNE_DATASET_NAME,
    binary_lh_mask,
    nnunet_paths,
    non_lh_bone_mask,
    project_root,
    read_array,
    same_geometry,
    save_json,
    strip_nii_suffix,
    write_array,
)


SOURCE_DATASET = "Dataset001_original_data"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create binary LH fine-tuning data. RH and sacrum voxels are set "
            "to the CT background value in the training-style images."
        )
    )
    parser.add_argument(
        "--background_value",
        type=float,
        default=-1024.0,
        help="CT value used to erase non-LH bone voxels.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def label_path_for_image(labels_dir: Path, image_path: Path) -> Path:
    case_id = strip_nii_suffix(image_path.name).removesuffix("_0000")
    path = labels_dir / f"{case_id}.nii.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing label for {image_path}: {path}")
    return path


def process_split(
    src_images: Path,
    src_labels: Path,
    erased_images: Path,
    binary_labels: Path,
    normal_images: Path | None,
    original_labels: Path | None,
    background_value: float,
    overwrite: bool,
) -> int:
    count = 0
    for image_path in tqdm(
        sorted(src_images.glob("*_0000.nii.gz")),
        desc=f"Preparing {src_images.name}",
        unit="case",
    ):
        label_path = label_path_for_image(src_labels, image_path)
        image_arr, image = read_array(image_path)
        label_arr, label = read_array(label_path)
        if image_arr.shape != label_arr.shape or not same_geometry(image, label):
            raise ValueError(
                f"Image/label geometry mismatch for {image_path.name}"
            )

        erased = image_arr.copy()
        erased[non_lh_bone_mask(label_arr)] = background_value
        binary_label = binary_lh_mask(label_arr)
        if not binary_label.any():
            raise ValueError(f"No LH voxels found in {label_path}")

        erased_out = erased_images / image_path.name
        label_out = binary_labels / label_path.name
        if overwrite or not erased_out.exists():
            write_array(erased, image, erased_out)
        if overwrite or not label_out.exists():
            write_array(binary_label, label, label_out)

        if normal_images is not None:
            normal_out = normal_images / image_path.name
            if overwrite or not normal_out.exists():
                normal_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(image_path, normal_out)
        if original_labels is not None:
            original_out = original_labels / label_path.name
            if overwrite or not original_out.exists():
                original_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(label_path, original_out)
        count += 1
    return count


def main() -> None:
    args = parse_args()
    nnunet_raw, _, _ = nnunet_paths()
    source = nnunet_raw / SOURCE_DATASET
    target = nnunet_raw / FINETUNE_DATASET_NAME
    for folder in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        if not (source / folder).is_dir():
            raise FileNotFoundError(f"Missing source folder: {source / folder}")

    images_tr = target / "imagesTr"
    labels_tr = target / "labelsTr"
    images_ts = target / "imagesTs"
    labels_ts = target / "labelsTs"
    for folder in (images_tr, labels_tr, images_ts, labels_ts):
        folder.mkdir(parents=True, exist_ok=True)

    artifacts = project_root() / "lh_pretraining_smoke_test" / "artifacts"
    normal_test_images = artifacts / "normal_test_images"
    original_test_labels = artifacts / "original_multiclass_labelsTs"

    train_count = process_split(
        source / "imagesTr",
        source / "labelsTr",
        images_tr,
        labels_tr,
        None,
        None,
        args.background_value,
        args.overwrite,
    )
    test_count = process_split(
        source / "imagesTs",
        source / "labelsTs",
        images_ts,
        labels_ts,
        normal_test_images,
        original_test_labels,
        args.background_value,
        args.overwrite,
    )

    save_json(
        {
            "channel_names": {"0": "CT"},
            "labels": {"background": 0, "left_hipbone": 1},
            "numTraining": train_count,
            "file_ending": ".nii.gz",
        },
        target / "dataset.json",
    )
    save_json(
        {
            "background_value": args.background_value,
            "train_cases": train_count,
            "test_cases": test_count,
            "erased_test_images": str(images_ts),
            "normal_test_images": str(normal_test_images),
            "binary_test_labels": str(labels_ts),
            "original_multiclass_test_labels": str(original_test_labels),
        },
        artifacts / "dataset_summary.json",
    )
    print(f"Created {target}")
    print(f"Training cases: {train_count}; test cases: {test_count}")
    print(f"Normal test images: {normal_test_images}")


if __name__ == "__main__":
    main()
