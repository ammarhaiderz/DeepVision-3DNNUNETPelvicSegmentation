from __future__ import annotations

from pathlib import Path


TEACHER_DATASETS = {
    "LH": {
        "dataset_id": 12,
        "dataset_name": "Dataset012_teacher_erased_LH",
        "anatomy_label": 1,
        "pengwin_label_ranges": [(11, 19)],
    },
    "RH": {
        "dataset_id": 13,
        "dataset_name": "Dataset013_teacher_erased_RH",
        "anatomy_label": 2,
        "pengwin_label_ranges": [(21, 29)],
    },
    "S": {
        "dataset_id": 14,
        "dataset_name": "Dataset014_teacher_erased_S",
        "anatomy_label": 3,
        "pengwin_label_ranges": [(1, 9)],
    },
}

STUDENT_DATASET_ID = 15
STUDENT_DATASET_NAME = "Dataset015_student_anatomy_distilled"
SOURCE_DATASET_NAME = "Dataset001_original_data"
ANATOMY_DATASET_NAME = "Dataset002_anatomy_network"


def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "nnUNet").exists() or (parent / "README.md").exists():
            return parent
    raise FileNotFoundError("Could not find NNUNETPipeline project root.")


def strip_nii_gz(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    return Path(filename).stem


def dataset_folder_by_id(root: Path, dataset_id: int) -> Path:
    prefix = f"Dataset{dataset_id:03d}_"
    for folder in sorted(root.iterdir()):
        if folder.is_dir() and folder.name.startswith(prefix):
            return folder
    raise FileNotFoundError(f"Could not find {prefix}... in {root}")

