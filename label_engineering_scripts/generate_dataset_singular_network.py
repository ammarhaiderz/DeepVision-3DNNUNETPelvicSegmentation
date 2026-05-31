import json
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation
from tqdm import tqdm


NNUNET_ROOT_FOLDER_NAME = "nnUNet"
NNUNET_RAW_FOLDER_NAME = "nnUNet_raw"

SOURCE_DATASET_NAME = "Dataset001_original_data"
TARGET_DATASET_NAME = "Dataset003_singular_network"

IMAGE_SUFFIX = ".nii.gz"

# =======================
# HARDCODED PARAMETERS
# =======================
DILATE_RADIUS = 1
# =======================


def find_first_folder(start_dir: Path, folder_name: str) -> Path | None:
    """
    Recursively search for the first folder with the given name.

    Parameters
    ----------
    start_dir : Path
        Directory from which the search starts.
    folder_name : str
        Name of the folder to find.

    Returns
    -------
    Path | None
        Path to the first matching folder, or None if not found.
    """
    for path in start_dir.rglob(folder_name):
        if path.is_dir():
            return path
    return None


def read_img(path: Path) -> tuple[np.ndarray, sitk.Image]:
    """
    Read a medical image file and return both the array and the SimpleITK image.

    Parameters
    ----------
    path : Path
        Path to the image file.

    Returns
    -------
    tuple[np.ndarray, sitk.Image]
        Image array and corresponding SimpleITK image object.
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr, img


def write_label(arr: np.ndarray, ref: sitk.Image, path: Path) -> None:
    """
    Write a label array as a NIfTI image while preserving spatial metadata.

    Parameters
    ----------
    arr : np.ndarray
        Output label array.
    ref : sitk.Image
        Reference image providing spacing, origin, and direction.
    path : Path
        Output file path.
    """
    out = sitk.GetImageFromArray(arr.astype(np.uint8))
    out.SetSpacing(ref.GetSpacing())
    out.SetOrigin(ref.GetOrigin())
    out.SetDirection(ref.GetDirection())
    sitk.WriteImage(out, str(path))


def strip_nii_suffix(name: str) -> str:
    """
    Remove .nii.gz or .nii suffix from a filename.

    Parameters
    ----------
    name : str
        Input filename.

    Returns
    -------
    str
        Filename without NIfTI suffix.
    """
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def get_case_stem(path: Path) -> str:
    """
    Get the case name without NIfTI suffix.

    Parameters
    ----------
    path : Path
        Input file path.

    Returns
    -------
    str
        Case stem.
    """
    return strip_nii_suffix(path.name)


def anatomy_group_key(fid: int) -> str | None:
    """
    Map an instance label ID to its anatomy group.

    Mapping
    -------
    - 1..9   -> S
    - 11..19 -> LH
    - 21..29 -> RH

    Parameters
    ----------
    fid : int
        Instance fragment ID.

    Returns
    -------
    str | None
        Anatomy group key ('LH', 'RH', 'S') or None if invalid.
    """
    if fid <= 0:
        return None

    if fid < 10:
        return "S"

    k = fid // 10

    if k == 1:
        return "LH"

    if k == 2:
        return "RH"

    return None


def convert_labels(arr: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """
    Convert instance labels into singular-network labels.

    Output classes
    --------------
    - 0 = background
    - 1 = bone
    - 2 = border between fragments of the same anatomy group only

    Borders are created separately within LH, RH, and S.

    Parameters
    ----------
    arr : np.ndarray
        Input instance label array.
    structure : np.ndarray
        Structuring element used for dilation.

    Returns
    -------
    np.ndarray
        Converted label array with dtype uint8.
    """
    ids = np.unique(arr)
    ids = ids[ids != 0]

    bone_mask = arr > 0
    border_mask = np.zeros_like(arr, dtype=bool)

    for group in ["LH", "RH", "S"]:
        group_ids = [int(fid) for fid in ids if anatomy_group_key(int(fid)) == group]

        # A border is only possible if there are at least two fragments
        # in the same anatomy group.
        if len(group_ids) < 2:
            continue

        dilated_sum = np.zeros_like(arr, dtype=np.uint16)
        any_frag = np.zeros_like(arr, dtype=bool)

        for fid in group_ids:
            mask = arr == fid
            any_frag |= mask

            dilated_mask = binary_dilation(mask, structure=structure)
            dilated_sum += dilated_mask.astype(np.uint16)

        group_border = dilated_sum >= 2

        # Keep the border localized around fragments from this group.
        group_border &= binary_dilation(any_frag, structure=structure)

        border_mask |= group_border

    out_arr = np.zeros_like(arr, dtype=np.uint8)
    out_arr[bone_mask] = 1
    out_arr[border_mask] = 2

    return out_arr


def ensure_dataset_structure(dataset_root: Path) -> tuple[Path, Path, Path, Path]:
    """
    Create the required folder structure for the target nnU-Net dataset.

    Parameters
    ----------
    dataset_root : Path
        Root folder of the target dataset.

    Returns
    -------
    tuple[Path, Path, Path, Path]
        Paths to imagesTr, labelsTr, imagesTs, labelsTs.
    """
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    images_ts = dataset_root / "imagesTs"
    labels_ts = dataset_root / "labelsTs"

    for folder in (images_tr, labels_tr, images_ts, labels_ts):
        folder.mkdir(parents=True, exist_ok=True)

    return images_tr, labels_tr, images_ts, labels_ts


def convert_label_folder(in_dir: Path, out_dir: Path, desc: str, structure: np.ndarray) -> int:
    """
    Convert all label files in a folder into singular-network labels.

    Parameters
    ----------
    in_dir : Path
        Input folder containing .nii.gz label files.
    out_dir : Path
        Output folder where converted labels are saved.
    desc : str
        Description shown in the tqdm progress bar.
    structure : np.ndarray
        Structuring element used for dilation.

    Returns
    -------
    int
        Number of converted files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.nii.gz"))

    with tqdm(files, desc=desc, unit="file") as pbar:
        for file in pbar:
            pbar.set_postfix_str(file.name)

            arr, ref = read_img(file)
            new_arr = convert_labels(arr, structure)

            out_path = out_dir / f"{get_case_stem(file)}.nii.gz"
            write_label(new_arr, ref, out_path)

    return len(files)


def copy_folder_files(in_dir: Path, out_dir: Path, desc: str) -> int:
    """
    Copy all .nii.gz files from one folder to another.

    Parameters
    ----------
    in_dir : Path
        Source folder.
    out_dir : Path
        Target folder.
    desc : str
        Description shown in the tqdm progress bar.

    Returns
    -------
    int
        Number of copied files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.nii.gz"))

    with tqdm(files, desc=desc, unit="file") as pbar:
        for file in pbar:
            pbar.set_postfix_str(file.name)
            shutil.copy2(file, out_dir / file.name)

    return len(files)


def infer_labels_from_first_label(label_path: Path) -> dict[str, int]:
    """
    Infer the label mapping from a single converted label file.

    The classes are named automatically as:
    - background: 0
    - class_1: 1
    - class_2: 2
    - ...

    Parameters
    ----------
    label_path : Path
        Path to one converted label file.

    Returns
    -------
    dict[str, int]
        Label mapping for dataset.json.
    """
    img = sitk.ReadImage(str(label_path))
    arr = sitk.GetArrayFromImage(img)
    unique_values = sorted(int(v) for v in np.unique(arr))

    labels = {"background": 0}

    for value in unique_values:
        if value == 0:
            continue
        labels[f"class_{value}"] = value

    return labels


def create_dataset_json(dataset_root: Path, labels_tr: Path, num_training_cases: int) -> None:
    """
    Create dataset.json automatically by scanning the first training label.

    Parameters
    ----------
    dataset_root : Path
        Root folder of the target dataset.
    labels_tr : Path
        Path to the labelsTr folder.
    num_training_cases : int
        Number of training cases.
    """
    first_label = next(labels_tr.glob("*.nii.gz"), None)
    if first_label is None:
        raise FileNotFoundError("No label files found in labelsTr, cannot create dataset.json.")

    labels_dict = infer_labels_from_first_label(first_label)

    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": labels_dict,
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz"
    }

    out_path = dataset_root / "dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4)


def main() -> None:
    """
    Create Dataset003_singular_network from Dataset001_original_data.

    The script expects the following source structure:
        nnUNet/
            nnUNet_raw/
                Dataset001_original_data/
                    imagesTr/
                    imagesTs/
                    labelsTr/
                    labelsTs/

    It creates:
        nnUNet/
            nnUNet_raw/
                Dataset003_singular_network/
                    imagesTr/
                    imagesTs/
                    labelsTr/
                    labelsTs/
                    dataset.json
    """
    start_dir = Path.cwd()

    nnunet_root = find_first_folder(start_dir, NNUNET_ROOT_FOLDER_NAME)
    if nnunet_root is None:
        print(f"Could not find folder '{NNUNET_ROOT_FOLDER_NAME}' under:\n{start_dir}")
        sys.exit(1)

    nnunet_raw = nnunet_root / NNUNET_RAW_FOLDER_NAME
    source_dataset = nnunet_raw / SOURCE_DATASET_NAME
    target_dataset = nnunet_raw / TARGET_DATASET_NAME

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

    tgt_images_tr, tgt_labels_tr, tgt_images_ts, tgt_labels_ts = ensure_dataset_structure(target_dataset)

    # Create structuring element once and reuse it for all cases.
    k = 2 * DILATE_RADIUS + 1
    structure = np.ones((k, k, k), dtype=bool)

    copied_images_tr = copy_folder_files(src_images_tr, tgt_images_tr, "Copying imagesTr")
    copied_images_ts = copy_folder_files(src_images_ts, tgt_images_ts, "Copying imagesTs")

    converted_labels_tr = convert_label_folder(
        src_labels_tr,
        tgt_labels_tr,
        "Converting labelsTr",
        structure,
    )
    converted_labels_ts = convert_label_folder(
        src_labels_ts,
        tgt_labels_ts,
        "Converting labelsTs",
        structure,
    )

    create_dataset_json(target_dataset, tgt_labels_tr, converted_labels_tr)

    print("\nFinished.")
    print(f"Source dataset: {source_dataset}")
    print(f"Target dataset: {target_dataset}")
    print(f"Copied imagesTr: {copied_images_tr}")
    print(f"Copied imagesTs: {copied_images_ts}")
    print(f"Converted labelsTr: {converted_labels_tr}")
    print(f"Converted labelsTs: {converted_labels_ts}")
    print(f"dataset.json created at: {target_dataset / 'dataset.json'}")


if __name__ == "__main__":
    main()