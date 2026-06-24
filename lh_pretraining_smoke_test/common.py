from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


PELVIC1K_DATASET_ID = 16
PELVIC1K_DATASET_NAME = "Dataset016_CTPelvic1K_pretrain"
FINETUNE_DATASET_ID = 17
FINETUNE_DATASET_NAME = "Dataset017_PENGWIN_erased_LH"
PLANS_NAME = "Pelvic1KPlans"
TRAINER_NAME = "nnUNetTrainer_no_rotation"
PELVIC1K_TRAINER_NAME = "nnUNetTrainer_PelvicPretrain"


def project_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    if not (root / "nnUNet").is_dir():
        raise FileNotFoundError(f"Missing nnUNet folder under project root: {root}")
    return root


def nnunet_paths() -> tuple[Path, Path, Path]:
    root = project_root() / "nnUNet"
    return (
        root / "nnUNet_raw",
        root / "nnUNet_preprocessed",
        root / "nnUNet_results",
    )


def strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def read_array(path: Path) -> tuple[np.ndarray, sitk.Image]:
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image), image


def write_array(array: np.ndarray, reference: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = sitk.GetImageFromArray(array)
    output.CopyInformation(reference)
    sitk.WriteImage(output, str(path))


def same_geometry(first: sitk.Image, second: sitk.Image) -> bool:
    return (
        first.GetSize() == second.GetSize()
        and np.allclose(first.GetSpacing(), second.GetSpacing())
        and np.allclose(first.GetOrigin(), second.GetOrigin())
        and np.allclose(first.GetDirection(), second.GetDirection())
    )


def save_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def binary_lh_mask(labels: np.ndarray) -> np.ndarray:
    foreground = labels[labels > 0]
    if foreground.size and int(foreground.max()) <= 3:
        mask = labels == 1
    else:
        mask = (labels >= 11) & (labels <= 19)
    return mask.astype(np.uint8)


def non_lh_bone_mask(labels: np.ndarray) -> np.ndarray:
    foreground = labels[labels > 0]
    if foreground.size and int(foreground.max()) <= 3:
        return (labels == 2) | (labels == 3)
    return ((labels >= 1) & (labels <= 9)) | (
        (labels >= 21) & (labels <= 29)
    )
