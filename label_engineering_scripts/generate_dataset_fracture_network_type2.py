import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


NNUNET_ROOT_FOLDER_NAME = "nnUNet"
NNUNET_RAW_FOLDER_NAME = "nnUNet_raw"

SOURCE_DATASET_NAME = "Dataset001_original_data"
TARGET_DATASET_NAME = "Dataset005_fracture_network_type2"

IMAGE_SUFFIX = ".nii.gz"

# =======================
# IMAGE PARAMETERS
# =======================
SUPPRESS_VALUE = -1024

# =======================
# DATASET.JSON LABELS
# =======================
HARDCODED_LABELS = {
    "background": 0,
    "class_1": 1,
    "class_2": 2,
}


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
    Read an image file and return both the array and the SimpleITK image.

    Parameters
    ----------
    path : Path
        Path to the image file.

    Returns
    -------
    tuple[np.ndarray, sitk.Image]
        Image array and reference image.
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr, img


def write_img(arr: np.ndarray, ref: sitk.Image, path: Path) -> None:
    """
    Write an image array while preserving spatial metadata.

    Parameters
    ----------
    arr : np.ndarray
        Output image array.
    ref : sitk.Image
        Reference image providing spacing, origin, and direction.
    path : Path
        Output file path.
    """
    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(ref.GetSpacing())
    out.SetOrigin(ref.GetOrigin())
    out.SetDirection(ref.GetDirection())
    sitk.WriteImage(out, str(path))


def write_label(arr: np.ndarray, ref: sitk.Image, path: Path) -> None:
    """
    Write a label array as uint8 while preserving spatial metadata.

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
    Get the filename stem without NIfTI suffix.

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


def get_image_case_id(path: Path) -> str:
    """
    Extract the base case ID from an nnU-Net image filename.

    Examples
    --------
    001_0000.nii.gz -> 001

    Parameters
    ----------
    path : Path
        Path to an nnU-Net image file.

    Returns
    -------
    str
        Base case ID.
    """
    stem = strip_nii_suffix(path.name)

    if stem.endswith("_0000"):
        return stem[:-5]

    return stem


def anatomy_group_key(fid: int) -> str | None:
    """
    Map an instance fragment ID to its anatomy group.

    Mapping
    -------
    - 1..9   -> S
    - 11..19 -> LH
    - 21..29 -> RH

    Parameters
    ----------
    fid : int
        Fragment ID.

    Returns
    -------
    str | None
        'LH', 'RH', 'S', or None.
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


def build_group_masks(inst: np.ndarray) -> dict[str, np.ndarray]:
    """
    Build binary masks for the three anatomy groups.

    Parameters
    ----------
    inst : np.ndarray
        Instance label array.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys 'LH', 'RH', 'S'.
    """
    ids = np.unique(inst)
    ids = ids[ids != 0]

    group_masks = {
        "LH": np.zeros_like(inst, dtype=bool),
        "RH": np.zeros_like(inst, dtype=bool),
        "S": np.zeros_like(inst, dtype=bool),
    }

    for fid in ids:
        group = anatomy_group_key(int(fid))
        if group is None:
            continue
        group_masks[group] |= (inst == fid)

    return group_masks


def collect_group_fragments(inst: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    """
    Collect fragment IDs and sizes for each anatomy group.

    Returns
    -------
    dict[str, list[tuple[int, int]]]
        Dictionary like:
        {
            "LH": [(fid, size), ...],
            "RH": [(fid, size), ...],
            "S":  [(fid, size), ...],
        }
    """
    ids = np.unique(inst)
    ids = ids[ids != 0]

    groups = {"LH": [], "RH": [], "S": []}

    for fid in ids:
        fid = int(fid)
        group = anatomy_group_key(fid)
        if group is None:
            continue

        size = int(np.count_nonzero(inst == fid))
        if size > 0:
            groups[group].append((fid, size))

    return groups


def build_primary_secondary_label(inst: np.ndarray, group: str) -> np.ndarray:
    """
    Build a primary-secondary label map for one anatomy group.

    Output labels
    -------------
    0 = background
    1 = largest fragment in this bone group
    2 = all smaller fragments in this bone group

    Parameters
    ----------
    inst : np.ndarray
        Instance label array.
    group : str
        Anatomy group name ('LH', 'RH', or 'S').

    Returns
    -------
    np.ndarray
        Output label array.
    """
    out = np.zeros_like(inst, dtype=np.uint8)

    groups = collect_group_fragments(inst)
    frags = groups[group]

    if len(frags) == 0:
        return out

    frags = sorted(frags, key=lambda x: x[1], reverse=True)

    primary_fid = frags[0][0]
    out[inst == primary_fid] = 1

    for fid, _ in frags[1:]:
        out[inst == fid] = 2

    return out


def ensure_dataset_structure(dataset_root: Path) -> tuple[Path, Path, Path, Path]:
    """
    Create the standard target dataset folder structure.

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


def create_grouped_images(
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    desc: str,
) -> int:
    """
    Create suppressed full-size bone-specific CT images.

    For each input case, up to three output images are created:
    - <case>_LH_0000.nii.gz
    - <case>_RH_0000.nii.gz
    - <case>_S_0000.nii.gz

    Parameters
    ----------
    images_dir : Path
        Folder with original CT images.
    labels_dir : Path
        Folder with original instance labels.
    out_dir : Path
        Output folder for grouped images.
    desc : str
        Description for tqdm.

    Returns
    -------
    int
        Number of output images written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(images_dir.glob("*.nii.gz"))

    written = 0

    with tqdm(image_files, desc=desc, unit="file") as pbar:
        for img_path in pbar:
            pbar.set_postfix_str(img_path.name)

            case_id = get_image_case_id(img_path)
            lbl_path = labels_dir / f"{case_id}.nii.gz"

            if not lbl_path.exists():
                raise FileNotFoundError(
                    f"No matching label found for image {img_path.name}"
                )

            img_arr, img_ref = read_img(img_path)
            lbl_arr, _ = read_img(lbl_path)

            if img_arr.shape != lbl_arr.shape:
                raise RuntimeError(
                    f"Shape mismatch for case {img_path.name}: "
                    f"image shape {img_arr.shape}, label shape {lbl_arr.shape}"
                )

            group_masks = build_group_masks(lbl_arr)

            for group in ["LH", "RH", "S"]:
                mask = group_masks[group]

                if not mask.any():
                    continue

                out_arr = np.full_like(img_arr, fill_value=SUPPRESS_VALUE)
                out_arr[mask] = img_arr[mask]

                out_path = out_dir / f"{case_id}_{group}_0000.nii.gz"
                write_img(out_arr, img_ref, out_path)
                written += 1

    return written


def create_grouped_labels(
    in_dir: Path,
    out_dir: Path,
    desc: str,
) -> int:
    """
    Create primary-secondary bone-specific labels.

    For each input case, three output labels are written:
    - <case>_LH.nii.gz
    - <case>_RH.nii.gz
    - <case>_S.nii.gz

    Parameters
    ----------
    in_dir : Path
        Folder with original instance labels.
    out_dir : Path
        Output folder for grouped labels.
    desc : str
        Description for tqdm.

    Returns
    -------
    int
        Number of output labels written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.nii.gz"))

    written = 0

    with tqdm(files, desc=desc, unit="file") as pbar:
        for f in pbar:
            pbar.set_postfix_str(f.name)

            inst, ref = read_img(f)
            case_stem = get_case_stem(f)

            for group in ["LH", "RH", "S"]:
                out_arr = build_primary_secondary_label(inst, group)
                out_path = out_dir / f"{case_stem}_{group}.nii.gz"
                write_label(out_arr, ref, out_path)
                written += 1

    return written


def create_dataset_json(dataset_root: Path, num_training_cases: int) -> None:
    """
    Create dataset.json using hardcoded labels.

    Parameters
    ----------
    dataset_root : Path
        Root folder of the target dataset.
    num_training_cases : int
        Number of training cases.
    """
    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": HARDCODED_LABELS,
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz"
    }

    out_path = dataset_root / "dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4)


def main() -> None:
    """
    Create Dataset005_fracture_network_type2 from Dataset001_original_data.

    Expected source structure:
        nnUNet/
            nnUNet_raw/
                Dataset001_original_data/
                    imagesTr/
                    imagesTs/
                    labelsTr/
                    labelsTs/

    Created target structure:
        nnUNet/
            nnUNet_raw/
                Dataset005_fracture_network_type2/
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

    created_images_tr = create_grouped_images(
        src_images_tr,
        src_labels_tr,
        tgt_images_tr,
        "Creating imagesTr",
    )
    created_images_ts = create_grouped_images(
        src_images_ts,
        src_labels_ts,
        tgt_images_ts,
        "Creating imagesTs",
    )

    created_labels_tr = create_grouped_labels(
        src_labels_tr,
        tgt_labels_tr,
        "Creating labelsTr",
    )
    created_labels_ts = create_grouped_labels(
        src_labels_ts,
        tgt_labels_ts,
        "Creating labelsTs",
    )

    create_dataset_json(target_dataset, created_labels_tr)

    print("\nFinished.")
    print(f"Source dataset: {source_dataset}")
    print(f"Target dataset: {target_dataset}")
    print(f"Created imagesTr: {created_images_tr}")
    print(f"Created imagesTs: {created_images_ts}")
    print(f"Created labelsTr: {created_labels_tr}")
    print(f"Created labelsTs: {created_labels_ts}")
    print(f"dataset.json created at: {target_dataset / 'dataset.json'}")


if __name__ == "__main__":
    main()