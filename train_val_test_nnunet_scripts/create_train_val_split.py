import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm


NNUNET_ROOT_FOLDER_NAME = "nnUNet"
NNUNET_RAW_FOLDER_NAME = "nnUNet_raw"
NNUNET_PREPROCESSED_FOLDER_NAME = "nnUNet_preprocessed"
SPLIT_FILE_NAME = "train_val_test.json"


def parse_args():
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing the dataset ID.
    """
    parser = argparse.ArgumentParser(
        description="Create splits_final.json for an nnU-Net dataset from train_val_test.json."
    )
    parser.add_argument(
        "--dataset_id",
        type=int,
        required=True,
        help="Dataset ID, e.g. 1 for Dataset001_..., 2 for Dataset002_...",
    )
    return parser.parse_args()


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
        Matching folder path, or None if not found.
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
        Matching file path, or None if not found.
    """
    for path in start_dir.rglob(file_name):
        if path.is_file():
            return path
    return None


def get_dataset_folder_by_id(nnunet_raw_dir: Path, dataset_id: int) -> Path | None:
    """
    Find the dataset folder inside nnUNet_raw that matches the given dataset ID.

    Parameters
    ----------
    nnunet_raw_dir : Path
        Path to nnUNet_raw.
    dataset_id : int
        Dataset ID, e.g. 1 for Dataset001_*.

    Returns
    -------
    Path | None
        Matching dataset folder, or None if not found.
    """
    prefix = f"Dataset{dataset_id:03d}_"

    for path in sorted(nnunet_raw_dir.iterdir()):
        if path.is_dir() and path.name.startswith(prefix):
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


def load_train_val_split(split_path: Path) -> tuple[list[str], list[str]]:
    """
    Load the train and validation case IDs from train_val_test.json.

    Parameters
    ----------
    split_path : Path
        Path to train_val_test.json.

    Returns
    -------
    tuple[list[str], list[str]]
        Train IDs and validation IDs.

    Raises
    ------
    ValueError
        If the JSON file does not contain the required keys.
    """
    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = {"train", "val"}
    if not required_keys.issubset(data.keys()):
        raise ValueError(
            f"Split file must contain keys {required_keys}, but got {set(data.keys())}"
        )

    train_ids = [str(x) for x in data["train"]]
    val_ids = [str(x) for x in data["val"]]

    return train_ids, val_ids


def collect_training_case_names(dataset_labels_tr: Path) -> list[str]:
    """
    Collect all case names present in labelsTr of the selected dataset.

    Parameters
    ----------
    dataset_labels_tr : Path
        Path to labelsTr inside the selected dataset.

    Returns
    -------
    list[str]
        Sorted case names without .nii.gz.
    """
    return sorted(strip_nii_gz(path.name) for path in dataset_labels_tr.glob("*.nii.gz"))


def build_case_mapping(existing_case_names: list[str]) -> dict[str, list[str]]:
    """
    Build a mapping from base patient IDs to the actual case names present in the dataset.

    Examples
    --------
    "001" -> ["001"]
    "001_LH", "001_RH", "001_S" -> {"001": ["001_LH", "001_RH", "001_S"]}

    Parameters
    ----------
    existing_case_names : list[str]
        Case names found in the dataset.

    Returns
    -------
    dict[str, list[str]]
        Mapping from base ID to matching dataset case names.
    """
    mapping: dict[str, list[str]] = {}

    region_suffixes = {"LH", "RH", "S"}

    for case_name in existing_case_names:
        parts = case_name.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in region_suffixes:
            base_id = parts[0]
        else:
            base_id = case_name
        mapping.setdefault(base_id, []).append(case_name)

    for base_id in mapping:
        mapping[base_id].sort()

    return mapping


def expand_ids_to_dataset_cases(
    base_ids: list[str],
    case_mapping: dict[str, list[str]],
    split_name: str,
) -> tuple[list[str], list[str]]:
    """
    Expand base IDs from train_val_test.json to the actual case names in the dataset.

    Parameters
    ----------
    base_ids : list[str]
        Base IDs from the split file, e.g. ["001", "002"].
    case_mapping : dict[str, list[str]]
        Mapping from base ID to dataset case names.
    split_name : str
        Name of the split for warning messages.

    Returns
    -------
    tuple[list[str], list[str]]
        Expanded case names and missing base IDs.
    """
    expanded = []
    missing = []

    for base_id in tqdm(base_ids, desc=f"Building {split_name} split", unit="case"):
        matches = case_mapping.get(base_id)

        if not matches:
            missing.append(base_id)
            continue

        expanded.extend(matches)

    return expanded, missing


def ensure_preprocessed_dataset_dir(
    nnunet_preprocessed_dir: Path,
    dataset_folder_name: str,
) -> Path:
    """
    Create the dataset folder inside nnUNet_preprocessed if it does not exist.

    Parameters
    ----------
    nnunet_preprocessed_dir : Path
        Path to nnUNet_preprocessed.
    dataset_folder_name : str
        Name of the dataset folder.

    Returns
    -------
    Path
        Path to the dataset folder inside nnUNet_preprocessed.
    """
    dataset_preprocessed_dir = nnunet_preprocessed_dir / dataset_folder_name
    dataset_preprocessed_dir.mkdir(parents=True, exist_ok=True)
    return dataset_preprocessed_dir


def save_splits_final(
    out_path: Path,
    train_cases: list[str],
    val_cases: list[str],
) -> None:
    """
    Save splits_final.json in nnU-Net format.

    Parameters
    ----------
    out_path : Path
        Output path for splits_final.json.
    train_cases : list[str]
        Expanded training case names.
    val_cases : list[str]
        Expanded validation case names.
    """
    splits = [
        {
            "train": train_cases,
            "val": val_cases,
        }
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


def main() -> None:
    """
    Create splits_final.json for the selected dataset.

    The script:
    1. Finds nnUNet, nnUNet_raw, and nnUNet_preprocessed
    2. Finds train_val_test.json
    3. Finds the dataset in nnUNet_raw by dataset ID
    4. Reads existing labelsTr case names
    5. Expands train/val IDs to match the dataset naming
    6. Writes splits_final.json into the matching nnUNet_preprocessed dataset folder
    """
    args = parse_args()
    dataset_id = args.dataset_id

    start_dir = Path.cwd()

    nnunet_root = find_first_folder(start_dir, NNUNET_ROOT_FOLDER_NAME)
    if nnunet_root is None:
        print(f"Could not find folder '{NNUNET_ROOT_FOLDER_NAME}' under:\n{start_dir}")
        sys.exit(1)

    nnunet_raw = nnunet_root / NNUNET_RAW_FOLDER_NAME
    nnunet_preprocessed = nnunet_root / NNUNET_PREPROCESSED_FOLDER_NAME

    if not nnunet_raw.is_dir():
        print(f"Missing folder:\n{nnunet_raw}")
        sys.exit(1)

    if not nnunet_preprocessed.is_dir():
        print(f"Missing folder:\n{nnunet_preprocessed}")
        sys.exit(1)

    split_path = find_first_file(start_dir, SPLIT_FILE_NAME)
    if split_path is None:
        print(f"Could not find file '{SPLIT_FILE_NAME}' under:\n{start_dir}")
        sys.exit(1)

    dataset_raw_dir = get_dataset_folder_by_id(nnunet_raw, dataset_id)
    if dataset_raw_dir is None:
        print(f"Could not find Dataset{dataset_id:03d}_... in:\n{nnunet_raw}")
        sys.exit(1)

    dataset_labels_tr = dataset_raw_dir / "labelsTr"
    dataset_images_tr = dataset_raw_dir / "imagesTr"

    if not dataset_labels_tr.is_dir():
        print(f"Missing labelsTr folder:\n{dataset_labels_tr}")
        sys.exit(1)

    if not dataset_images_tr.is_dir():
        print(f"Missing imagesTr folder:\n{dataset_images_tr}")
        sys.exit(1)

    try:
        train_ids, val_ids = load_train_val_split(split_path)
    except Exception as e:
        print(f"Failed to read split file: {e}")
        sys.exit(1)

    existing_case_names = collect_training_case_names(dataset_labels_tr)
    if not existing_case_names:
        print(f"No training label files found in:\n{dataset_labels_tr}")
        sys.exit(1)

    case_mapping = build_case_mapping(existing_case_names)

    train_cases, missing_train = expand_ids_to_dataset_cases(train_ids, case_mapping, "train")
    val_cases, missing_val = expand_ids_to_dataset_cases(val_ids, case_mapping, "val")

    dataset_preprocessed_dir = ensure_preprocessed_dataset_dir(
        nnunet_preprocessed,
        dataset_raw_dir.name,
    )

    out_path = dataset_preprocessed_dir / "splits_final.json"
    save_splits_final(out_path, train_cases, val_cases)

    print("\nFinished.")
    print(f"Dataset raw folder:         {dataset_raw_dir}")
    print(f"Dataset preprocessed folder:{dataset_preprocessed_dir}")
    print(f"splits_final.json saved to: {out_path}")
    print(f"Train entries written:      {len(train_cases)}")
    print(f"Val entries written:        {len(val_cases)}")

    if missing_train:
        print(f"\nMissing train base IDs: {len(missing_train)}")
        print(", ".join(missing_train))

    if missing_val:
        print(f"\nMissing val base IDs: {len(missing_val)}")
        print(", ".join(missing_val))


if __name__ == "__main__":
    main()
