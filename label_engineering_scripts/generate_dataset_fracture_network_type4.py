import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation
from tqdm import tqdm


NNUNET_ROOT_FOLDER_NAME = "nnUNet"
NNUNET_RAW_FOLDER_NAME = "nnUNet_raw"

SOURCE_DATASET_NAME = "Dataset001_original_data"
TARGET_DATASET_NAME = "Dataset007_fracture_network_type4"

IMAGE_SUFFIX = ".nii.gz"

# =======================
# IMAGE PARAMETERS
# =======================
SUPPRESS_VALUE = -1024

# =======================
# LABEL PARAMETERS
# =======================
BORDER_RADIUS = 3
BORDER_THICKEN_ITERS = 2
BBOX_PAD = 3
MIN_FRAGMENT_SIZE = 0

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
    """
    return strip_nii_suffix(path.name)


def get_image_case_id(path: Path) -> str:
    """
    Extract the base case ID from an nnU-Net image filename.

    Examples
    --------
    001_0000.nii.gz -> 001
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


def make_structure(radius: int) -> np.ndarray:
    """
    Create a cubic binary structuring element for dilation.

    Parameters
    ----------
    radius : int
        Dilation radius.

    Returns
    -------
    np.ndarray
        Boolean structuring element.
    """
    k = 2 * radius + 1
    return np.ones((k, k, k), dtype=bool)


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


def filter_small_fragments(inst: np.ndarray, group_ids: list[int]) -> list[int]:
    """
    Optionally remove fragments smaller than MIN_FRAGMENT_SIZE.

    Parameters
    ----------
    inst : np.ndarray
        Instance label array.
    group_ids : list[int]
        Fragment IDs in one anatomy group.

    Returns
    -------
    list[int]
        Filtered fragment IDs.
    """
    if MIN_FRAGMENT_SIZE <= 0:
        return group_ids

    kept = []
    for fid in group_ids:
        n = int(np.sum(inst == fid))
        if n >= MIN_FRAGMENT_SIZE:
            kept.append(fid)
    return kept


def get_bbox(mask: np.ndarray, pad: int = 0) -> tuple[slice, slice, slice] | None:
    """
    Compute a padded bounding box around a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask.
    pad : int, optional
        Padding added on all sides.

    Returns
    -------
    tuple[slice, slice, slice] | None
        Bounding box slices, or None if the mask is empty.
    """
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None

    z0, y0, x0 = [int(c.min()) for c in coords]
    z1, y1, x1 = [int(c.max()) + 1 for c in coords]

    shape = mask.shape

    z0 = max(0, z0 - pad)
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)

    z1 = min(shape[0], z1 + pad)
    y1 = min(shape[1], y1 + pad)
    x1 = min(shape[2], x1 + pad)

    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))


def compute_border_mask(
    inst: np.ndarray,
    group_ids: list[int],
    border_structure: np.ndarray,
    thick_structure: np.ndarray,
    case_stem: str,
    group_name: str,
) -> np.ndarray:
    """
    Compute thickened border/contact zones between fragments of the same group.

    Parameters
    ----------
    inst : np.ndarray
        Instance label array.
    group_ids : list[int]
        Fragment IDs belonging to the current group.
    border_structure : np.ndarray
        Structuring element for initial border detection.
    thick_structure : np.ndarray
        Structuring element for final border thickening.
    case_stem : str
        Case identifier for tqdm labeling.
    group_name : str
        Group name ('LH', 'RH', or 'S').

    Returns
    -------
    np.ndarray
        Boolean border mask.
    """
    if len(group_ids) < 2:
        return np.zeros_like(inst, dtype=bool)

    dilated_sum = np.zeros_like(inst, dtype=np.uint16)
    frag_bar = tqdm(group_ids, desc=f"{case_stem} {group_name} border", leave=False)

    for fid in frag_bar:
        frag_bar.set_postfix(fid=int(fid))

        frag = inst == fid
        if not np.any(frag):
            continue

        bbox = get_bbox(frag, pad=BBOX_PAD + BORDER_RADIUS + BORDER_THICKEN_ITERS)
        if bbox is None:
            continue

        frag_crop = frag[bbox]
        dil_crop = binary_dilation(frag_crop, structure=border_structure)
        dilated_sum[bbox] += dil_crop.astype(np.uint16)

    border_mask = dilated_sum >= 2

    if BORDER_THICKEN_ITERS > 0 and np.any(border_mask):
        for _ in range(BORDER_THICKEN_ITERS):
            border_mask = binary_dilation(border_mask, structure=thick_structure)

    group_mask = np.isin(inst, group_ids)
    group_bbox = get_bbox(group_mask, pad=BORDER_RADIUS + BORDER_THICKEN_ITERS)

    if group_bbox is not None:
        group_crop = group_mask[group_bbox]
        near_group_crop = binary_dilation(
            group_crop,
            structure=make_structure(BORDER_RADIUS + BORDER_THICKEN_ITERS),
        )

        near_group_mask = np.zeros_like(group_mask, dtype=bool)
        near_group_mask[group_bbox] = near_group_crop
        border_mask &= near_group_mask

    return border_mask


def make_group_output(
    inst: np.ndarray,
    group_ids: list[int],
    border_structure: np.ndarray,
    thick_structure: np.ndarray,
    case_stem: str,
    group_name: str,
) -> np.ndarray:
    """
    Create the type-4 label output for one anatomy group.

    Output labels
    -------------
    0 = background
    1 = bone
    2 = border between fragments of the same group

    Border has highest priority and overwrites bone.

    Parameters
    ----------
    inst : np.ndarray
        Instance label array.
    group_ids : list[int]
        Fragment IDs belonging to the current group.
    border_structure : np.ndarray
        Structuring element for initial border detection.
    thick_structure : np.ndarray
        Structuring element for border thickening.
    case_stem : str
        Case identifier for tqdm labeling.
    group_name : str
        Group name ('LH', 'RH', or 'S').

    Returns
    -------
    np.ndarray
        Output label array.
    """
    out_arr = np.zeros_like(inst, dtype=np.uint8)

    if len(group_ids) == 0:
        return out_arr

    group_ids = filter_small_fragments(inst, group_ids)
    if len(group_ids) == 0:
        return out_arr

    group_mask = np.isin(inst, group_ids)
    border_mask = compute_border_mask(
        inst=inst,
        group_ids=group_ids,
        border_structure=border_structure,
        thick_structure=thick_structure,
        case_stem=case_stem,
        group_name=group_name,
    )

    out_arr[group_mask] = 1
    out_arr[border_mask] = 2

    return out_arr


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
    border_structure: np.ndarray,
    thick_structure: np.ndarray,
) -> int:
    """
    Create type-4 bone-specific labels.

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
    border_structure : np.ndarray
        Structuring element for border detection.
    thick_structure : np.ndarray
        Structuring element for border thickening.

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
            ids = np.unique(inst)
            ids = ids[ids != 0]

            case_stem = get_case_stem(f)

            for group in ["LH", "RH", "S"]:
                group_ids = [int(fid) for fid in ids if anatomy_group_key(int(fid)) == group]

                out_arr = make_group_output(
                    inst=inst,
                    group_ids=group_ids,
                    border_structure=border_structure,
                    thick_structure=thick_structure,
                    case_stem=case_stem,
                    group_name=group,
                )

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
    Create Dataset007_fracture_network_type4 from Dataset001_original_data.

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
                Dataset007_fracture_network_type4/
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

    border_structure = make_structure(BORDER_RADIUS)
    thick_structure = make_structure(1)

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
        border_structure,
        thick_structure,
    )
    created_labels_ts = create_grouped_labels(
        src_labels_ts,
        tgt_labels_ts,
        "Creating labelsTs",
        border_structure,
        thick_structure,
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