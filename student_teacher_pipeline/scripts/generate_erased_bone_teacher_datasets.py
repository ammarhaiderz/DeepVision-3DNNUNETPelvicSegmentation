#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from common import SOURCE_DATASET_NAME, TEACHER_DATASETS, find_project_root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create erased-CT binary teacher datasets for LH, RH, and sacrum."
    )
    parser.add_argument(
        "--erase_value",
        type=float,
        default=0.0,
        help="Intensity written into non-target bone voxels in the CT image.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist in target datasets.",
    )
    return parser.parse_args()


def ensure_structure(dataset_root: Path) -> tuple[Path, Path, Path, Path]:
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    images_ts = dataset_root / "imagesTs"
    labels_ts = dataset_root / "labelsTs"
    for folder in (images_tr, labels_tr, images_ts, labels_ts):
        folder.mkdir(parents=True, exist_ok=True)
    return images_tr, labels_tr, images_ts, labels_ts


def target_mask_from_labels(
    labels: np.ndarray,
    anatomy_label: int,
    pengwin_label_ranges: list[tuple[int, int]],
) -> np.ndarray:
    foreground = np.unique(labels)
    foreground = foreground[foreground > 0]
    if foreground.size > 0 and int(foreground.max()) <= 3:
        return labels == anatomy_label

    mask = np.zeros_like(labels, dtype=bool)
    for start, end in pengwin_label_ranges:
        mask |= (labels >= start) & (labels <= end)
    return mask


def all_bone_mask(labels: np.ndarray) -> np.ndarray:
    return labels > 0


def write_image_like(array: np.ndarray, reference: sitk.Image, path: Path) -> None:
    out = sitk.GetImageFromArray(array)
    out.CopyInformation(reference)
    sitk.WriteImage(out, str(path))


def process_case(
    image_path: Path,
    label_path: Path,
    out_image_path: Path,
    out_label_path: Path,
    anatomy_label: int,
    pengwin_label_ranges: list[tuple[int, int]],
    erase_value: float,
    overwrite: bool,
) -> None:
    if not overwrite and out_image_path.exists() and out_label_path.exists():
        return

    image = sitk.ReadImage(str(image_path))
    label = sitk.ReadImage(str(label_path))
    image_arr = sitk.GetArrayFromImage(image)
    label_arr = sitk.GetArrayFromImage(label)

    if image_arr.shape != label_arr.shape:
        raise ValueError(
            f"Shape mismatch for {image_path.name}: image {image_arr.shape}, label {label_arr.shape}"
        )

    target_mask = target_mask_from_labels(label_arr, anatomy_label, pengwin_label_ranges)
    non_target_bone_mask = all_bone_mask(label_arr) & ~target_mask

    erased_image = image_arr.copy()
    erased_image[non_target_bone_mask] = erase_value

    binary_label = np.zeros_like(label_arr, dtype=np.uint8)
    binary_label[target_mask] = 1

    write_image_like(erased_image, image, out_image_path)
    write_image_like(binary_label, label, out_label_path)


def create_dataset_json(dataset_root: Path, num_training_cases: int, bone_key: str) -> None:
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, f"{bone_key}_target": 1},
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz",
    }
    with (dataset_root / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4)


def process_split(
    src_images: Path,
    src_labels: Path,
    out_images: Path,
    out_labels: Path,
    config: dict,
    erase_value: float,
    overwrite: bool,
    split_name: str,
) -> int:
    image_files = sorted(src_images.glob("*.nii.gz"))
    count = 0
    for image_path in tqdm(image_files, desc=f"{config['dataset_name']} {split_name}", unit="case"):
        label_path = src_labels / image_path.name.replace("_0000.nii.gz", ".nii.gz")
        if not label_path.exists():
            label_path = src_labels / image_path.name
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path}: tried {label_path}")

        process_case(
            image_path,
            label_path,
            out_images / image_path.name,
            out_labels / label_path.name,
            config["anatomy_label"],
            config["pengwin_label_ranges"],
            erase_value,
            overwrite,
        )
        count += 1
    return count


def main() -> None:
    args = parse_args()
    project_root = find_project_root(Path(__file__).resolve())
    nnunet_raw = project_root / "nnUNet" / "nnUNet_raw"
    source = nnunet_raw / SOURCE_DATASET_NAME

    src_images_tr = source / "imagesTr"
    src_labels_tr = source / "labelsTr"
    src_images_ts = source / "imagesTs"
    src_labels_ts = source / "labelsTs"
    for folder in (src_images_tr, src_labels_tr, src_images_ts, src_labels_ts):
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing source folder: {folder}")

    for bone_key, config in TEACHER_DATASETS.items():
        dataset_root = nnunet_raw / config["dataset_name"]
        images_tr, labels_tr, images_ts, labels_ts = ensure_structure(dataset_root)
        train_count = process_split(
            src_images_tr,
            src_labels_tr,
            images_tr,
            labels_tr,
            config,
            args.erase_value,
            args.overwrite,
            "imagesTr",
        )
        test_count = process_split(
            src_images_ts,
            src_labels_ts,
            images_ts,
            labels_ts,
            config,
            args.erase_value,
            args.overwrite,
            "imagesTs",
        )
        create_dataset_json(dataset_root, train_count, bone_key)
        print(f"Created {dataset_root} with {train_count} train and {test_count} test cases.")


if __name__ == "__main__":
    main()

