import json
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


NNUNET_ROOT_FOLDER_NAME = "nnUNet"
NNUNET_RAW_FOLDER_NAME = "nnUNet_raw"

SOURCE_DATASET_NAME = "Dataset001_original_data"

BINARY_DATASETS = {
    "LH": {
        "dataset_name": "Dataset009_anatomy_LH",
        "anatomy_label": 1,
        "pengwin_label_ranges": [(11, 19)],
    },
    "RH": {
        "dataset_name": "Dataset010_anatomy_RH",
        "anatomy_label": 2,
        "pengwin_label_ranges": [(21, 29)],
    },
    "S": {
        "dataset_name": "Dataset011_anatomy_S",
        "anatomy_label": 3,
        "pengwin_label_ranges": [(1, 9)],
    },
}


def find_first_folder(start_dir: Path, folder_name: str) -> Path | None:
    for path in start_dir.rglob(folder_name):
        if path.is_dir():
            return path
    return None


def ensure_dataset_structure(dataset_root: Path) -> tuple[Path, Path, Path, Path]:
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    images_ts = dataset_root / "imagesTs"
    labels_ts = dataset_root / "labelsTs"

    for folder in (images_tr, labels_tr, images_ts, labels_ts):
        folder.mkdir(parents=True, exist_ok=True)

    return images_tr, labels_tr, images_ts, labels_ts


def copy_folder_files(in_dir: Path, out_dir: Path, desc: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.nii.gz"))

    with tqdm(files, desc=desc, unit="file") as pbar:
        for file in pbar:
            pbar.set_postfix_str(file.name)
            shutil.copyfile(file, out_dir / file.name)

    return len(files)


def convert_to_binary_bone(
    arr: np.ndarray,
    anatomy_label: int,
    pengwin_label_ranges: list[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.uint8)

    labels = np.unique(arr)
    foreground_labels = labels[labels > 0]

    if foreground_labels.size > 0 and int(foreground_labels.max()) <= 3:
        mask = arr == anatomy_label
    else:
        mask = np.zeros_like(arr, dtype=bool)
        for start, end in pengwin_label_ranges:
            mask |= (arr >= start) & (arr <= end)

    out[mask] = 1
    return out


def convert_label_folder(
    in_dir: Path,
    out_dir: Path,
    anatomy_label: int,
    pengwin_label_ranges: list[tuple[int, int]],
    desc: str,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.nii.gz"))

    with tqdm(files, desc=desc, unit="file") as pbar:
        for file in pbar:
            pbar.set_postfix_str(file.name)

            img = sitk.ReadImage(str(file))
            arr = sitk.GetArrayFromImage(img)
            new_arr = convert_to_binary_bone(
                arr,
                anatomy_label,
                pengwin_label_ranges,
            )

            out_img = sitk.GetImageFromArray(new_arr)
            out_img.CopyInformation(img)
            sitk.WriteImage(out_img, str(out_dir / file.name))

    return len(files)


def create_dataset_json(dataset_root: Path, num_training_cases: int) -> None:
    dataset_json = {
        "channel_names": {
            "0": "CT",
        },
        "labels": {
            "background": 0,
            "target_bone": 1,
        },
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz",
    }

    with (dataset_root / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4)


def validate_source_dataset(source_dataset: Path) -> tuple[Path, Path, Path, Path]:
    if not source_dataset.is_dir():
        print(f"Missing source dataset folder:\n{source_dataset}")
        sys.exit(1)

    src_images_tr = source_dataset / "imagesTr"
    src_labels_tr = source_dataset / "labelsTr"
    src_images_ts = source_dataset / "imagesTs"
    src_labels_ts = source_dataset / "labelsTs"

    for folder in (src_images_tr, src_labels_tr, src_images_ts, src_labels_ts):
        if not folder.is_dir():
            print(f"Missing required source folder:\n{folder}")
            sys.exit(1)

    return src_images_tr, src_labels_tr, src_images_ts, src_labels_ts


def create_binary_dataset(
    source_dirs: tuple[Path, Path, Path, Path],
    nnunet_raw: Path,
    bone_key: str,
    config: dict,
) -> None:
    src_images_tr, src_labels_tr, src_images_ts, src_labels_ts = source_dirs

    target_dataset = nnunet_raw / config["dataset_name"]
    anatomy_label = config["anatomy_label"]
    pengwin_label_ranges = config["pengwin_label_ranges"]

    tgt_images_tr, tgt_labels_tr, tgt_images_ts, tgt_labels_ts = ensure_dataset_structure(
        target_dataset
    )

    copied_images_tr = copy_folder_files(
        src_images_tr,
        tgt_images_tr,
        f"{bone_key}: copying imagesTr",
    )
    copied_images_ts = copy_folder_files(
        src_images_ts,
        tgt_images_ts,
        f"{bone_key}: copying imagesTs",
    )

    converted_labels_tr = convert_label_folder(
        src_labels_tr,
        tgt_labels_tr,
        anatomy_label,
        pengwin_label_ranges,
        f"{bone_key}: converting labelsTr",
    )
    converted_labels_ts = convert_label_folder(
        src_labels_ts,
        tgt_labels_ts,
        anatomy_label,
        pengwin_label_ranges,
        f"{bone_key}: converting labelsTs",
    )

    create_dataset_json(target_dataset, converted_labels_tr)

    print(f"\nCreated {config['dataset_name']}")
    print(f"  Copied imagesTr:    {copied_images_tr}")
    print(f"  Copied imagesTs:    {copied_images_ts}")
    print(f"  Converted labelsTr: {converted_labels_tr}")
    print(f"  Converted labelsTs: {converted_labels_ts}")
    print(f"  dataset.json:       {target_dataset / 'dataset.json'}")


def main() -> None:
    start_dir = Path.cwd()

    nnunet_root = find_first_folder(start_dir, NNUNET_ROOT_FOLDER_NAME)
    if nnunet_root is None:
        print(f"Could not find folder '{NNUNET_ROOT_FOLDER_NAME}' under:\n{start_dir}")
        sys.exit(1)

    nnunet_raw = nnunet_root / NNUNET_RAW_FOLDER_NAME
    source_dataset = nnunet_raw / SOURCE_DATASET_NAME
    source_dirs = validate_source_dataset(source_dataset)

    print(f"Source dataset: {source_dataset}")
    print("Creating binary anatomy datasets: Dataset009, Dataset010, Dataset011")

    for bone_key, config in BINARY_DATASETS.items():
        create_binary_dataset(source_dirs, nnunet_raw, bone_key, config)

    print("\nFinished creating all binary anatomy datasets.")


if __name__ == "__main__":
    main()
