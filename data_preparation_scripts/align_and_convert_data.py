import sys
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm


# Configuration
TARGET_ORIENTATION = "LPS"
SOURCE_FOLDER_NAME = "original_PENGWIN_data"
OUTPUT_FOLDER_NAME = "aligned_and_converted_data"


def find_source_root(start_dir: Path, target_name: str) -> Path | None:
    """
    Recursively search for a folder with a given name.

    Returns the first match or None if not found.
    """
    for p in start_dir.rglob(target_name):
        if p.is_dir():
            return p
    return None


def collect_files(folder: Path) -> list[Path]:
    """
    Collect supported medical image files from a folder.
    """
    files = []
    files.extend(folder.glob("*.mha"))
    files.extend(folder.glob("*.nii"))
    files.extend(folder.glob("*.nii.gz"))
    return sorted(files)


def make_output_name(input_path: Path) -> str:
    """
    Convert input filename to .nii.gz format.
    """
    name = input_path.name

    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii") or name.endswith(".mha"):
        stem = name[:-4]
    else:
        stem = input_path.stem

    return f"{stem}.nii.gz"


def process_folder(
    in_dir: Path,
    out_dir: Path,
    orienter: sitk.DICOMOrientImageFilter,
    group_name: str,
) -> tuple[int, int, int]:
    """
    Reorient all files in a folder and save them as .nii.gz.

    Returns (total, changed, unchanged).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_files(in_dir)
    if not files:
        print(f"No supported files found in {in_dir}")
        return 0, 0, 0

    print(f"\nProcessing {group_name}: {len(files)} files")

    changed, unchanged = 0, 0

    for f in tqdm(files, desc=group_name):
        img = sitk.ReadImage(str(f))

        old_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            img.GetDirection()
        )

        out_img = orienter.Execute(img)

        new_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            out_img.GetDirection()
        )

        out_path = out_dir / make_output_name(f)
        sitk.WriteImage(out_img, str(out_path))

        if old_orientation == new_orientation:
            unchanged += 1
        else:
            changed += 1

    return len(files), changed, unchanged


def main() -> None:
    """
    Find dataset, reorient images and labels to TARGET_ORIENTATION,
    and save everything as .nii.gz.
    """
    start_dir = Path.cwd()
    source_root = find_source_root(start_dir, SOURCE_FOLDER_NAME)

    if source_root is None:
        print(f"Could not find '{SOURCE_FOLDER_NAME}' under:\n{start_dir}")
        sys.exit(1)

    images_in = source_root / "images"
    labels_in = source_root / "labels"

    if not images_in.is_dir() or not labels_in.is_dir():
        print("Missing 'images' or 'labels' folder in source directory.")
        sys.exit(1)

    output_root = source_root.parent / OUTPUT_FOLDER_NAME
    images_out = output_root / "images"
    labels_out = output_root / "labels"

    print(f"Source: {source_root}")
    print(f"Target orientation: {TARGET_ORIENTATION}")
    print(f"Output: {output_root}")

    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation(TARGET_ORIENTATION)

    img_total, img_changed, img_unchanged = process_folder(
        images_in, images_out, orienter, "images"
    )
    lbl_total, lbl_changed, lbl_unchanged = process_folder(
        labels_in, labels_out, orienter, "labels"
    )

    print("\nFinished")
    print(f"\nImages:  total={img_total}, changed={img_changed}, unchanged={img_unchanged}")
    print(f"Labels:  total={lbl_total}, changed={lbl_changed}, unchanged={lbl_unchanged}")
    print(f"\nSaved to: {output_root}\n")


if __name__ == "__main__":
    main()