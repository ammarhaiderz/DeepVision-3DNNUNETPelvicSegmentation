import json
import shutil
import sys
from pathlib import Path

from tqdm import tqdm


SOURCE_FOLDER_NAME = "aligned_and_converted_data"
SPLIT_FILE_NAME = "train_val_test.json"

NNUNET_ROOT_FOLDER_NAME = "nnUNet"
NNUNET_RAW_FOLDER_NAME = "nnUNet_raw"
NNUNET_PREPROCESSED_FOLDER_NAME = "nnUNet_preprocessed"
NNUNET_RESULTS_FOLDER_NAME = "nnUNet_results"

DATASET_FOLDER_NAME = "Dataset001_original_data"
IMAGE_SUFFIX = ".nii.gz"
CHANNEL_SUFFIX = "_0000"


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
        Path to the first matching folder, or None if no match is found.
    """
    for path in start_dir.rglob(folder_name):
        if path.is_dir():
            return path
    return None


def find_first_file(start_dir: Path, file_name: str) -> Path | None:
    """
    Recursively search for the first file with the given name.

    Parameters
    ----------
    start_dir : Path
        Directory from which the search starts.
    file_name : str
        Name of the file to find.

    Returns
    -------
    Path | None
        Path to the first matching file, or None if no match is found.
    """
    for path in start_dir.rglob(file_name):
        if path.is_file():
            return path
    return None


def strip_nii_gz(filename: str) -> str:
    """
    Remove the .nii.gz suffix from a filename.

    Parameters
    ----------
    filename : str
        Input filename.

    Returns
    -------
    str
        Filename without the .nii.gz suffix.
    """
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    return Path(filename).stem


def load_split_file(split_path: Path) -> tuple[list[str], list[str], list[str]]:
    """
    Load train, validation, and test case IDs from the split JSON file.

    The file is expected to contain top-level keys:
    - train
    - val
    - test

    Parameters
    ----------
    split_path : Path
        Path to the split JSON file.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Lists of train IDs, val IDs, and test IDs.
    """
    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = {"train", "val", "test"}
    if not required_keys.issubset(data.keys()):
        raise ValueError(
            f"Split file must contain keys {required_keys}, but got {set(data.keys())}"
        )

    train_ids = [str(x) for x in data["train"]]
    val_ids = [str(x) for x in data["val"]]
    test_ids = [str(x) for x in data["test"]]

    return train_ids, val_ids, test_ids


def collect_available_cases(images_dir: Path, labels_dir: Path) -> dict[str, tuple[Path, Path]]:
    """
    Collect all case IDs for which both image and label files exist.

    Parameters
    ----------
    images_dir : Path
        Folder containing image files.
    labels_dir : Path
        Folder containing label files.

    Returns
    -------
    dict[str, tuple[Path, Path]]
        Mapping from case ID to (image_path, label_path).
    """
    image_files = sorted(images_dir.glob("*.nii.gz"))
    label_files = sorted(labels_dir.glob("*.nii.gz"))

    image_map = {strip_nii_gz(path.name): path for path in image_files}
    label_map = {strip_nii_gz(path.name): path for path in label_files}

    common_ids = sorted(set(image_map.keys()) & set(label_map.keys()))
    return {case_id: (image_map[case_id], label_map[case_id]) for case_id in common_ids}


def ensure_nnunet_structure(base_dir: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    """
    Create the required nnUNet folder structure.

    The following folders are created:
        nnUNet/
            nnUNet_raw/
            nnUNet_preprocessed/
            nnUNet_results/

    Inside nnUNet_raw, the dataset folder is created with:
        Dataset001_original_data/
            imagesTr/
            labelsTr/
            imagesTs/
            labelsTs/

    Parameters
    ----------
    base_dir : Path
        Base directory in which the nnUNet folder should be created.

    Returns
    -------
    tuple[Path, Path, Path, Path, Path, Path]
        Paths to:
        - dataset_root
        - imagesTr
        - labelsTr
        - imagesTs
        - labelsTs
        - nnUNet root
    """
    nnunet_root = base_dir / NNUNET_ROOT_FOLDER_NAME
    nnunet_raw = nnunet_root / NNUNET_RAW_FOLDER_NAME
    nnunet_preprocessed = nnunet_root / NNUNET_PREPROCESSED_FOLDER_NAME
    nnunet_results = nnunet_root / NNUNET_RESULTS_FOLDER_NAME

    dataset_root = nnunet_raw / DATASET_FOLDER_NAME
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    images_ts = dataset_root / "imagesTs"
    labels_ts = dataset_root / "labelsTs"

    for folder in (
        nnunet_root,
        nnunet_raw,
        nnunet_preprocessed,
        nnunet_results,
        dataset_root,
        images_tr,
        labels_tr,
        images_ts,
        labels_ts,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    return dataset_root, images_tr, labels_tr, images_ts, labels_ts, nnunet_root


def copy_training_case(
    case_id: str,
    image_path: Path,
    label_path: Path,
    images_tr: Path,
    labels_tr: Path,
) -> None:
    """
    Copy one case into the nnU-Net training folders.

    Images are renamed to nnU-Net channel format:
    case_id.nii.gz -> case_id_0000.nii.gz
    """
    out_image = images_tr / f"{case_id}{CHANNEL_SUFFIX}{IMAGE_SUFFIX}"
    out_label = labels_tr / f"{case_id}{IMAGE_SUFFIX}"

    shutil.copyfile(image_path, out_image)
    shutil.copyfile(label_path, out_label)


def copy_test_case(
    case_id: str,
    image_path: Path,
    label_path: Path,
    images_ts: Path,
    labels_ts: Path,
) -> None:
    """
    Copy one case into the nnU-Net test folders.

    Images are renamed to nnU-Net channel format:
    case_id.nii.gz -> case_id_0000.nii.gz
    """
    out_image = images_ts / f"{case_id}{CHANNEL_SUFFIX}{IMAGE_SUFFIX}"
    out_label = labels_ts / f"{case_id}{IMAGE_SUFFIX}"

    shutil.copyfile(image_path, out_image)
    shutil.copyfile(label_path, out_label)


def main() -> None:
    """
    Build an nnU-Net dataset automatically from aligned data and a split file.

    Expected input structure:
        aligned_and_converted_data/
            images/
            labels/

    The script also expects:
        train_val_test.json

    Output:
        nnUNet/
            nnUNet_raw/
                Dataset001_original_data/
                    imagesTr/
                    labelsTr/
                    imagesTs/
                    labelsTs/
            nnUNet_preprocessed/
            nnUNet_results/

    Split behavior:
    - train + val -> imagesTr / labelsTr
    - test        -> imagesTs / labelsTs
    """
    start_dir = Path.cwd()

    source_root = find_first_folder(start_dir, SOURCE_FOLDER_NAME)
    if source_root is None:
        print(f"Could not find folder '{SOURCE_FOLDER_NAME}' under:\n{start_dir}")
        sys.exit(1)

    split_path = find_first_file(start_dir, SPLIT_FILE_NAME)
    if split_path is None:
        print(f"Could not find file '{SPLIT_FILE_NAME}' under:\n{start_dir}")
        sys.exit(1)

    images_dir = source_root / "images"
    labels_dir = source_root / "labels"

    if not images_dir.is_dir():
        print(f"Missing images folder: {images_dir}")
        sys.exit(1)

    if not labels_dir.is_dir():
        print(f"Missing labels folder: {labels_dir}")
        sys.exit(1)

    try:
        train_ids, val_ids, test_ids = load_split_file(split_path)
    except Exception as e:
        print(f"Failed to load split file: {e}")
        sys.exit(1)

    training_ids = train_ids + val_ids
    available_cases = collect_available_cases(images_dir, labels_dir)

    if not available_cases:
        print("No matching image/label pairs were found.")
        sys.exit(1)

    dataset_root, images_tr, labels_tr, images_ts, labels_ts, nnunet_root = ensure_nnunet_structure(start_dir)

    missing_train_cases = []
    missing_test_cases = []

    with tqdm(training_ids, desc="Copying training cases", unit="case") as pbar:
        for case_id in pbar:
            pbar.set_postfix_str(case_id)

            if case_id not in available_cases:
                missing_train_cases.append(case_id)
                continue

            image_path, label_path = available_cases[case_id]
            copy_training_case(case_id, image_path, label_path, images_tr, labels_tr)

    with tqdm(test_ids, desc="Copying test cases", unit="case") as pbar:
        for case_id in pbar:
            pbar.set_postfix_str(case_id)

            if case_id not in available_cases:
                missing_test_cases.append(case_id)
                continue

            image_path, label_path = available_cases[case_id]
            copy_test_case(case_id, image_path, label_path, images_ts, labels_ts)

    copied_train = len(training_ids) - len(missing_train_cases)
    copied_test = len(test_ids) - len(missing_test_cases)

    print("\nFinished.")
    print(f"nnUNet root created at: {nnunet_root}")
    print(f"Dataset created at:     {dataset_root}")
    print(f"Training cases copied:  {copied_train}")
    print(f"Test cases copied:      {copied_test}")

    if missing_train_cases:
        print(f"Missing training cases: {len(missing_train_cases)}")
        print(", ".join(missing_train_cases))

    if missing_test_cases:
        print(f"Missing test cases:     {len(missing_test_cases)}")
        print(", ".join(missing_test_cases))


if __name__ == "__main__":
    main()
