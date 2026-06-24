#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from common import ANATOMY_DATASET_NAME, STUDENT_DATASET_NAME, find_project_root, strip_nii_gz


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Create the student anatomy dataset. Training images contain CT plus "
            "three teacher-prior channels. Test images contain CT plus zero prior "
            "channels so inference uses only the student."
        )
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Teacher prior manifest. If omitted, expects "
            "student_teacher_pipeline/teacher_prior_manifest.json."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_structure(dataset_root: Path) -> tuple[Path, Path, Path, Path]:
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    images_ts = dataset_root / "imagesTs"
    labels_ts = dataset_root / "labelsTs"
    for folder in (images_tr, labels_tr, images_ts, labels_ts):
        folder.mkdir(parents=True, exist_ok=True)
    return images_tr, labels_tr, images_ts, labels_ts


def convert_labels(arr: np.ndarray) -> np.ndarray:
    foreground = np.unique(arr)
    foreground = foreground[foreground > 0]
    if foreground.size > 0 and int(foreground.max()) <= 3:
        return arr.astype(np.uint8)

    out = np.zeros_like(arr, dtype=np.uint8)
    out[(arr >= 1) & (arr <= 9)] = 3
    out[(arr >= 11) & (arr <= 19)] = 1
    out[(arr >= 21) & (arr <= 29)] = 2
    return out


def load_foreground_probability(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "probabilities" in data:
            probs = data["probabilities"]
        elif "softmax" in data:
            probs = data["softmax"]
        else:
            raise KeyError(f"{path} has no 'probabilities' or 'softmax' array.")

    if probs.ndim != 4:
        raise ValueError(f"Expected (C, Z, Y, X) probability array in {path}, got {probs.shape}")
    if probs.shape[0] == 1:
        return probs[0].astype(np.float32)
    return probs[1].astype(np.float32)


def write_channel(arr: np.ndarray, reference: sitk.Image, path: Path) -> None:
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(reference)
    sitk.WriteImage(out, str(path))


def write_training_images(
    src_dir: Path,
    dst_dir: Path,
    manifest: dict,
    overwrite: bool,
) -> int:
    count = 0
    for path in tqdm(sorted(src_dir.glob("*.nii.gz")), desc="Writing student imagesTr", unit="file"):
        image_name = path.name
        cid = strip_nii_gz(image_name.replace("_0000.nii.gz", ".nii.gz"))
        case_manifest = manifest["cases"].get(cid)
        if case_manifest is None:
            raise FileNotFoundError(f"No teacher priors listed for training case {cid}")

        ct_out = dst_dir / image_name
        if overwrite or not ct_out.exists():
            shutil.copyfile(path, ct_out)

        reference = sitk.ReadImage(str(path))
        reference_shape = sitk.GetArrayFromImage(reference).shape
        for channel_idx, bone in enumerate(("LH", "RH", "S"), start=1):
            prior = load_foreground_probability(Path(case_manifest[bone]))
            if prior.shape != reference_shape:
                raise ValueError(
                    f"Prior shape mismatch for {cid} {bone}: prior {prior.shape}, CT {reference_shape}"
                )
            out = dst_dir / image_name.replace("_0000.nii.gz", f"_{channel_idx:04d}.nii.gz")
            if overwrite or not out.exists():
                write_channel(prior, reference, out)
        count += 1
    return count


def write_test_images_with_zero_priors(src_dir: Path, dst_dir: Path, overwrite: bool) -> int:
    count = 0
    for path in tqdm(sorted(src_dir.glob("*.nii.gz")), desc="Writing student imagesTs", unit="file"):
        image_name = path.name
        ct_out = dst_dir / image_name
        if overwrite or not ct_out.exists():
            shutil.copyfile(path, ct_out)

        reference = sitk.ReadImage(str(path))
        zeros = np.zeros(sitk.GetArrayFromImage(reference).shape, dtype=np.float32)
        for channel_idx in (1, 2, 3):
            out = dst_dir / image_name.replace("_0000.nii.gz", f"_{channel_idx:04d}.nii.gz")
            if overwrite or not out.exists():
                write_channel(zeros, reference, out)
        count += 1
    return count


def convert_label_folder(src_dir: Path, dst_dir: Path, overwrite: bool) -> int:
    count = 0
    for path in tqdm(sorted(src_dir.glob("*.nii.gz")), desc=f"Converting {src_dir.name}", unit="file"):
        out = dst_dir / path.name
        if overwrite or not out.exists():
            img = sitk.ReadImage(str(path))
            arr = convert_labels(sitk.GetArrayFromImage(img))
            out_img = sitk.GetImageFromArray(arr)
            out_img.CopyInformation(img)
            sitk.WriteImage(out_img, str(out))
        count += 1
    return count


def create_dataset_json(dataset_root: Path, num_training: int) -> None:
    dataset_json = {
        "channel_names": {
            "0": "CT",
            "1": "teacher_LH",
            "2": "teacher_RH",
            "3": "teacher_S",
        },
        "labels": {
            "background": 0,
            "left_hipbone": 1,
            "right_hipbone": 2,
            "sacrum": 3,
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }
    with (dataset_root / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4)


def main() -> None:
    args = parse_args()
    project_root = find_project_root(Path(__file__).resolve())
    manifest_path = Path(args.manifest) if args.manifest else project_root / "student_teacher_pipeline" / "teacher_prior_manifest.json"
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing teacher prior manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    nnunet_raw = project_root / "nnUNet" / "nnUNet_raw"
    source = nnunet_raw / ANATOMY_DATASET_NAME
    target = nnunet_raw / STUDENT_DATASET_NAME

    for folder in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        if not (source / folder).is_dir():
            raise FileNotFoundError(f"Missing source folder: {source / folder}")

    images_tr, labels_tr, images_ts, labels_ts = ensure_structure(target)
    num_training = write_training_images(source / "imagesTr", images_tr, manifest, args.overwrite)
    write_test_images_with_zero_priors(source / "imagesTs", images_ts, args.overwrite)
    convert_label_folder(source / "labelsTr", labels_tr, args.overwrite)
    convert_label_folder(source / "labelsTs", labels_ts, args.overwrite)
    create_dataset_json(target, num_training)

    print(f"Created student dataset: {target}")


if __name__ == "__main__":
    main()
