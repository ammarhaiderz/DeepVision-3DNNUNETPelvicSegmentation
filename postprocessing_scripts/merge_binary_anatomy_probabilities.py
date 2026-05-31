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

DEFAULT_OUTPUT_FOLDER = "merged_binary_anatomy_prob"


def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "nnUNet").exists() or (parent / "README.md").exists():
            return parent
    raise FileNotFoundError("Could not find project root.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge binary anatomy probability maps by voxel-wise maximum over background/LH/RH/S."
    )
    parser.add_argument("--lh_dir", default=None)
    parser.add_argument("--rh_dir", default=None)
    parser.add_argument("--s_dir", default=None)
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Merged output folder. Defaults to predictions/merged_binary_anatomy_prob.",
    )
    parser.add_argument(
        "--summary_csv",
        default=None,
        help="Summary CSV. Defaults to evaluation/merged_binary_anatomy_prob_summary.csv.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Foreground probability threshold used by --background_mode threshold. "
            "Voxels below this stay background."
        ),
    )
    parser.add_argument(
        "--background_mode",
        choices=["threshold", "product"],
        default="threshold",
        help=(
            "How to score background. 'threshold' uses a fixed background threshold, "
            "while 'product' uses (1-p_LH)*(1-p_RH)*(1-p_S)."
        ),
    )
    parser.add_argument(
        "--ignore_missing",
        action="store_true",
        help="Skip cases missing one of the three probability files.",
    )
    parser.add_argument(
        "--min_component_volume_mm3",
        type=float,
        default=1000.0,
        help=(
            "Remove connected components smaller than this physical volume per anatomy class. "
            "Default is 1000 mm^3, equal to 1 cm^3 as used in the paper. "
            "Set to 0 to disable small-component cleanup."
        ),
    )
    parser.add_argument(
        "--keep_largest_component",
        action="store_true",
        help=(
            "Keep only the largest connected component for each anatomy class. "
            "This is stricter and may remove true fracture fragments."
        ),
    )
    return parser.parse_args()


def resolve(path_str: str | None, default: Path) -> Path:
    if path_str is None:
        return default
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def case_id(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def list_npz(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Prediction folder not found: {folder}")
    return {path.stem: path for path in sorted(folder.glob("*.npz"))}


def image_path(folder: Path, cid: str) -> Path:
    for suffix in (".nii.gz", ".nii", ".mha"):
        path = folder / f"{cid}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing reference prediction image for {cid} in {folder}")


def load_foreground_probability(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "probabilities" in data:
            probs = data["probabilities"]
        elif "softmax" in data:
            probs = data["softmax"]
        else:
            keys = ", ".join(data.files)
            raise KeyError(f"{path} does not contain 'probabilities' or 'softmax'. Keys: {keys}")

    if probs.ndim != 4:
        raise ValueError(f"Expected probability array with shape (C, Z, Y, X), got {probs.shape} in {path}")

    if probs.shape[0] == 1:
        return probs[0].astype(np.float32)
    return probs[1].astype(np.float32)


def read_reference(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def write_label(arr: np.ndarray, ref: sitk.Image, path: Path) -> None:
    out = sitk.GetImageFromArray(arr.astype(np.uint8))
    out.CopyInformation(ref)
    sitk.WriteImage(out, str(path))


def connected_component_cleanup(
    merged: np.ndarray,
    min_component_volume_mm3: float,
    voxel_volume_mm3: float,
    keep_largest_component: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    cleaned = np.zeros_like(merged, dtype=np.uint8)
    stats: dict[str, int] = {}

    for bone, config in BINARY_INPUTS.items():
        label = config["output_label"]
        mask = merged == label
        before_voxels = int(mask.sum())

        if before_voxels == 0:
            stats[f"{bone.lower()}_removed_voxels"] = 0
            stats[f"{bone.lower()}_kept_components"] = 0
            continue

        cc = sitk.ConnectedComponent(sitk.GetImageFromArray(mask.astype(np.uint8)))
        relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
        cc_arr = sitk.GetArrayFromImage(relabeled)
        component_ids, component_sizes = np.unique(cc_arr[cc_arr > 0], return_counts=True)

        keep_mask = np.zeros_like(mask, dtype=bool)
        if keep_largest_component:
            keep_mask = cc_arr == 1
        elif min_component_volume_mm3 > 0:
            component_volumes_mm3 = component_sizes.astype(np.float64) * voxel_volume_mm3
            kept_ids = component_ids[component_volumes_mm3 >= min_component_volume_mm3]
            keep_mask = np.isin(cc_arr, kept_ids)
        else:
            keep_mask = mask

        cleaned[keep_mask] = label
        after_voxels = int(keep_mask.sum())
        stats[f"{bone.lower()}_removed_voxels"] = before_voxels - after_voxels
        stats[f"{bone.lower()}_kept_components"] = int(len(np.unique(cc_arr[keep_mask])))

    return cleaned, stats


def build_maps(input_dirs: dict[str, Path]) -> dict[str, dict[str, Path]]:
    maps = {}
    for bone, folder in input_dirs.items():
        maps[bone] = list_npz(folder)
        if not maps[bone]:
            raise FileNotFoundError(
                f"No .npz probability files found in {folder}. "
                "Rerun prediction with --save_probabilities."
            )
    return maps


def collect_cases(prob_maps: dict[str, dict[str, Path]], ignore_missing: bool):
    all_cases = set()
    for m in prob_maps.values():
        all_cases.update(m.keys())

    complete = []
    missing_rows = []
    for cid in sorted(all_cases):
        missing = [bone for bone, m in prob_maps.items() if cid not in m]
        if missing:
            missing_rows.append({"case": cid, "missing": " ".join(missing)})
            if not ignore_missing:
                raise FileNotFoundError(f"Case {cid} is missing probability files for: {', '.join(missing)}")
            continue
        complete.append(cid)
    return complete, missing_rows


def merge_case(
    cid: str,
    prob_maps: dict[str, dict[str, Path]],
    threshold: float,
    background_mode: str,
    min_component_volume_mm3: float,
    voxel_volume_mm3: float,
    keep_largest_component: bool,
):
    probs = []
    bones = ["LH", "RH", "S"]
    for bone in bones:
        probs.append(load_foreground_probability(prob_maps[bone][cid]))

    bone_stack = np.stack(probs, axis=0)

    if background_mode == "threshold":
        background = np.full_like(bone_stack[0], threshold, dtype=np.float32)
    elif background_mode == "product":
        background = np.prod(1.0 - bone_stack, axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unknown background_mode: {background_mode}")

    score_stack = np.concatenate([background[None], bone_stack], axis=0)
    best_idx = np.argmax(score_stack, axis=0)
    best_score = np.max(score_stack, axis=0)

    merged = np.zeros(best_score.shape, dtype=np.uint8)
    for idx, bone in enumerate(bones):
        merged[best_idx == idx + 1] = BINARY_INPUTS[bone]["output_label"]

    second_best_score = np.partition(score_stack, kth=-2, axis=0)[-2]
    foreground = merged > 0
    ambiguous = foreground & ((best_score - second_best_score) < 0.05)

    cleaned, cleanup_stats = connected_component_cleanup(
        merged,
        min_component_volume_mm3,
        voxel_volume_mm3,
        keep_largest_component,
    )
    foreground = cleaned > 0

    summary = {
        "case": cid,
        "lh_voxels": int((cleaned == 1).sum()),
        "rh_voxels": int((cleaned == 2).sum()),
        "s_voxels": int((cleaned == 3).sum()),
        "foreground_voxels": int(foreground.sum()),
        "pre_cleanup_foreground_voxels": int((merged > 0).sum()),
        "cleanup_removed_voxels": int((merged > 0).sum() - foreground.sum()),
        "ambiguous_voxels_margin_lt_0_05": int((ambiguous & foreground).sum()),
        "mean_best_score": float(best_score[foreground].mean()) if np.any(foreground) else 0.0,
        "background_mode": background_mode,
        "min_component_volume_mm3": min_component_volume_mm3,
        "voxel_volume_mm3": voxel_volume_mm3,
        "keep_largest_component": int(keep_largest_component),
    }
    summary.update(cleanup_stats)
    return cleaned, summary


def write_summary(rows: list[dict], missing_rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "lh_voxels",
        "rh_voxels",
        "s_voxels",
        "foreground_voxels",
        "pre_cleanup_foreground_voxels",
        "cleanup_removed_voxels",
        "ambiguous_voxels_margin_lt_0_05",
        "mean_best_score",
        "background_mode",
        "min_component_volume_mm3",
        "voxel_volume_mm3",
        "keep_largest_component",
        "lh_removed_voxels",
        "rh_removed_voxels",
        "s_removed_voxels",
        "lh_kept_components",
        "rh_kept_components",
        "s_kept_components",
        "missing",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
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
                    "foreground_voxels": "",
                    "pre_cleanup_foreground_voxels": "",
                    "cleanup_removed_voxels": "",
                    "ambiguous_voxels_margin_lt_0_05": "",
                    "mean_best_score": "",
                    "background_mode": "",
                    "min_component_volume_mm3": "",
                    "voxel_volume_mm3": "",
                    "keep_largest_component": "",
                    "lh_removed_voxels": "",
                    "rh_removed_voxels": "",
                    "s_removed_voxels": "",
                    "lh_kept_components": "",
                    "rh_kept_components": "",
                    "s_kept_components": "",
                    "missing": row["missing"],
                }
            )


def main():
    args = parse_args()
    input_dirs = {
        "LH": resolve(args.lh_dir, PROJECT_ROOT / "predictions" / BINARY_INPUTS["LH"]["dataset_name"]),
        "RH": resolve(args.rh_dir, PROJECT_ROOT / "predictions" / BINARY_INPUTS["RH"]["dataset_name"]),
        "S": resolve(args.s_dir, PROJECT_ROOT / "predictions" / BINARY_INPUTS["S"]["dataset_name"]),
    }
    out_dir = resolve(args.out_dir, PROJECT_ROOT / "predictions" / DEFAULT_OUTPUT_FOLDER)
    summary_csv = resolve(
        args.summary_csv,
        PROJECT_ROOT / "evaluation" / "merged_binary_anatomy_prob_summary.csv",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    prob_maps = build_maps(input_dirs)
    cases, missing_rows = collect_cases(prob_maps, args.ignore_missing)
    if not cases:
        raise RuntimeError("No complete cases found.")

    rows = []
    for cid in tqdm(cases, desc="Merging probability maps"):
        ref = read_reference(image_path(input_dirs["LH"], cid))
        voxel_volume_mm3 = float(np.prod(ref.GetSpacing()))
        merged, summary = merge_case(
            cid,
            prob_maps,
            args.threshold,
            args.background_mode,
            args.min_component_volume_mm3,
            voxel_volume_mm3,
            args.keep_largest_component,
        )
        write_label(merged, ref, out_dir / f"{cid}.nii.gz")
        rows.append(summary)

    write_summary(rows, missing_rows, summary_csv)
    print(f"Saved probability-merged predictions to: {out_dir}")
    print(f"Saved summary CSV to: {summary_csv}")
    print(f"Background mode: {args.background_mode}")
    if args.background_mode == "threshold":
        print(f"Background threshold score: {args.threshold}")
    print(f"Minimum connected component volume: {args.min_component_volume_mm3} mm^3")
    print(f"Keep largest component only: {args.keep_largest_component}")


if __name__ == "__main__":
    main()
