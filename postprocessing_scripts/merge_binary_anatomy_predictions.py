#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


BINARY_INPUTS = {
    "LH": {
        "dataset_name": "Dataset009_anatomy_LH",
        "output_label": 1,
    },
    "RH": {
        "dataset_name": "Dataset010_anatomy_RH",
        "output_label": 2,
    },
    "S": {
        "dataset_name": "Dataset011_anatomy_S",
        "output_label": 3,
    },
}

DEFAULT_OUTPUT_FOLDER = "merged_binary_anatomy"


def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "nnUNet").exists() or (parent / "README.md").exists():
            return parent
    raise FileNotFoundError("Could not find project root.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge three binary anatomy predictions into one 3-class anatomy mask."
    )
    parser.add_argument(
        "--lh_dir",
        type=str,
        default=None,
        help="LH prediction folder. Defaults to predictions/Dataset009_anatomy_LH.",
    )
    parser.add_argument(
        "--rh_dir",
        type=str,
        default=None,
        help="RH prediction folder. Defaults to predictions/Dataset010_anatomy_RH.",
    )
    parser.add_argument(
        "--s_dir",
        type=str,
        default=None,
        help="Sacrum prediction folder. Defaults to predictions/Dataset011_anatomy_S.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Merged output folder. Defaults to predictions/merged_binary_anatomy.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Overlap summary CSV. Defaults to evaluation/merged_binary_anatomy_overlap_summary.csv.",
    )
    parser.add_argument(
        "--priority",
        nargs=3,
        choices=["LH", "RH", "S"],
        default=["LH", "RH", "S"],
        help="Merge priority from lowest to highest. Later labels overwrite earlier labels.",
    )
    parser.add_argument("--ignore_missing", action="store_true")
    return parser.parse_args()


def resolve_path(path_str: str | None, default: Path) -> Path:
    if path_str is None:
        return default

    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def list_image_files(folder: Path) -> list[Path]:
    return sorted(
        list(folder.glob("*.nii.gz"))
        + list(folder.glob("*.nii"))
        + list(folder.glob("*.mha"))
    )


def case_id(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii") or name.endswith(".mha"):
        return name[:-4]
    return path.stem


def read_image(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr, img


def write_image(arr: np.ndarray, ref_img: sitk.Image, path: Path) -> None:
    out = sitk.GetImageFromArray(arr.astype(np.uint8))
    out.CopyInformation(ref_img)
    sitk.WriteImage(out, str(path))


def build_prediction_maps(input_dirs: dict[str, Path]) -> dict[str, dict[str, Path]]:
    maps = {}
    for bone, folder in input_dirs.items():
        if not folder.is_dir():
            raise FileNotFoundError(f"{bone} prediction folder not found: {folder}")
        maps[bone] = {case_id(path): path for path in list_image_files(folder)}
    return maps


def collect_case_ids(prediction_maps: dict[str, dict[str, Path]], ignore_missing: bool):
    all_ids = set()
    for case_map in prediction_maps.values():
        all_ids.update(case_map.keys())

    complete_ids = []
    missing_rows = []
    for cid in sorted(all_ids):
        missing = [bone for bone, case_map in prediction_maps.items() if cid not in case_map]
        if missing:
            missing_rows.append({"case": cid, "missing": " ".join(missing)})
            if not ignore_missing:
                raise FileNotFoundError(f"Case {cid} is missing predictions for: {', '.join(missing)}")
            continue
        complete_ids.append(cid)

    return complete_ids, missing_rows


def merge_case(
    cid: str,
    prediction_maps: dict[str, dict[str, Path]],
    priority: list[str],
) -> tuple[np.ndarray, sitk.Image, dict]:
    arrays = {}
    ref_img = None
    ref_shape = None

    for bone in BINARY_INPUTS:
        arr, img = read_image(prediction_maps[bone][cid])
        mask = arr == 1

        if ref_img is None:
            ref_img = img
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            raise ValueError(f"Shape mismatch for {cid}: {bone} has {arr.shape}, expected {ref_shape}")

        arrays[bone] = mask

    stack = np.stack([arrays["LH"], arrays["RH"], arrays["S"]], axis=0)
    overlap_mask = np.sum(stack, axis=0) > 1

    merged = np.zeros(ref_shape, dtype=np.uint8)
    for bone in priority:
        merged[arrays[bone]] = BINARY_INPUTS[bone]["output_label"]

    summary = {
        "case": cid,
        "lh_voxels": int(arrays["LH"].sum()),
        "rh_voxels": int(arrays["RH"].sum()),
        "s_voxels": int(arrays["S"].sum()),
        "overlap_voxels": int(overlap_mask.sum()),
        "merged_foreground_voxels": int((merged > 0).sum()),
    }

    return merged, ref_img, summary


def write_summary_csv(rows: list[dict], missing_rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "lh_voxels",
        "rh_voxels",
        "s_voxels",
        "overlap_voxels",
        "merged_foreground_voxels",
        "missing",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({**row, "missing": ""})

        for row in missing_rows:
            writer.writerow(
                {
                    "case": row["case"],
                    "lh_voxels": "",
                    "rh_voxels": "",
                    "s_voxels": "",
                    "overlap_voxels": "",
                    "merged_foreground_voxels": "",
                    "missing": row["missing"],
                }
            )


def main():
    args = parse_args()

    input_dirs = {
        "LH": resolve_path(
            args.lh_dir,
            PROJECT_ROOT / "predictions" / BINARY_INPUTS["LH"]["dataset_name"],
        ),
        "RH": resolve_path(
            args.rh_dir,
            PROJECT_ROOT / "predictions" / BINARY_INPUTS["RH"]["dataset_name"],
        ),
        "S": resolve_path(
            args.s_dir,
            PROJECT_ROOT / "predictions" / BINARY_INPUTS["S"]["dataset_name"],
        ),
    }
    out_dir = resolve_path(
        args.out_dir,
        PROJECT_ROOT / "predictions" / DEFAULT_OUTPUT_FOLDER,
    )
    summary_csv = resolve_path(
        args.summary_csv,
        PROJECT_ROOT / "evaluation" / "merged_binary_anatomy_overlap_summary.csv",
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    prediction_maps = build_prediction_maps(input_dirs)
    case_ids, missing_rows = collect_case_ids(prediction_maps, args.ignore_missing)

    if not case_ids:
        raise RuntimeError("No complete cases were found to merge.")

    rows = []
    for cid in tqdm(case_ids, desc="Merging binary anatomy predictions"):
        merged, ref_img, summary = merge_case(cid, prediction_maps, args.priority)
        write_image(merged, ref_img, out_dir / f"{cid}.nii.gz")
        rows.append(summary)

    write_summary_csv(rows, missing_rows, summary_csv)

    print(f"Saved merged predictions to: {out_dir}")
    print(f"Saved overlap summary CSV to: {summary_csv}")
    print(f"Priority, lowest to highest: {' -> '.join(args.priority)}")


if __name__ == "__main__":
    main()
