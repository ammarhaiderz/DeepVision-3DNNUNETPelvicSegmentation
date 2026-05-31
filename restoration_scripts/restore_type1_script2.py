#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm

def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "nnUNet").exists():
            return parent
    raise FileNotFoundError(
        "Could not find project root. Expected a parent folder containing 'nnUNet'."
    )

PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
ANATOMY_PREDICTIONS_DATASET = "Dataset002_anatomy_network"
FRACTURE_PREDICTIONS_DATASET = "Dataset004_fracture_network_type1"
DEFAULT_RESTORATION_SUBDIR = "restore_type1_script2"


def resolve_runtime_dirs(args):
    anatomy_dir = Path(args.anatomy_dir) if args.anatomy_dir else PROJECT_ROOT / "predictions" / ANATOMY_PREDICTIONS_DATASET
    fracture_dir = Path(args.fracture_dir) if args.fracture_dir else PROJECT_ROOT / "predictions" / FRACTURE_PREDICTIONS_DATASET
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "restorations" / DEFAULT_RESTORATION_SUBDIR
    return anatomy_dir, fracture_dir, out_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split anatomy into fragments using fracture predictions "
            "(core split + border cutting, faster ROI-based version).\n\n"
            "Inputs:\n"
            "- anatomy: case.nii.gz (0=bg,1=LH,2=RH,3=S)\n"
            "- fracture: case_LH.nii.gz, case_RH.nii.gz, case_S.nii.gz\n"
            "            (0=bg,1=core,2=boundary,3=border)\n\n"
            "Logic:\n"
            "- Connected components are computed on CORE only\n"
            "- BORDER is dilated and used to CUT the core before CC\n"
            "- Everything else is reassigned afterward\n"
        )
    )

    parser.add_argument(
        "--anatomy_dir",
        type=str,
        default=None,
        help="Optional anatomy prediction folder. Default: predictions/Dataset002_anatomy_network",
    )
    parser.add_argument(
        "--fracture_dir",
        type=str,
        default=None,
        help="Optional fracture prediction folder. Default: predictions/Dataset004_fracture_network_type1",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional output folder. Default: restorations/restore_type1_script2_auto",
    )

    parser.add_argument("--min_fragment_voxels", type=int, default=1000)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--roi_margin", type=int, default=8)

    parser.add_argument(
        "--border_dilate_percent",
        type=float,
        default=1.0,
        help="Percent of bbox diagonal used to dilate border",
    )

    return parser.parse_args()


def read_image(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return img, arr


def write_image(arr, ref_img, path):
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref_img)
    sitk.WriteImage(out, str(path))


def get_bbox_slices(mask, margin=0):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    shape = np.array(mask.shape)

    mins = np.maximum(0, mins - margin)
    maxs = np.minimum(shape, maxs + margin)

    return tuple(slice(int(a), int(b)) for a, b in zip(mins, maxs))


def compute_bbox_diagonal(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0.0

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extents = (maxs - mins + 1).astype(float)
    return float(np.sqrt(np.sum(extents ** 2)))


def percent_to_dilate_iters(mask, percent):
    if percent <= 0 or not np.any(mask):
        return 0

    diag = compute_bbox_diagonal(mask)
    if diag <= 0:
        return 0

    iters = int(math.ceil((percent / 100.0) * diag))
    return max(1, min(iters, 4))


def remove_small_cc_fast(mask, min_size):
    if not np.any(mask):
        return mask.astype(bool)

    cc, _ = ndimage.label(mask)
    sizes = np.bincount(cc.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[cc]


def assign_to_nearest(labels, target_mask):
    if not np.any(target_mask):
        return labels
    if not np.any(labels > 0):
        return labels

    # nearest nonzero label for all zero locations
    _, indices = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    nearest = labels[tuple(indices)]

    out = labels.copy()
    out[target_mask] = nearest[target_mask]
    return out


def relabel_consecutively_fast(arr):
    labels = np.unique(arr)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return np.zeros_like(arr, dtype=np.int32)

    mapping = np.zeros(labels.max() + 1, dtype=np.int32)
    mapping[labels] = np.arange(1, len(labels) + 1, dtype=np.int32)
    return mapping[arr]


def clean_fragments_fast(instances, min_size):
    if not np.any(instances > 0):
        return instances.astype(np.int32)

    sizes = np.bincount(instances.ravel())
    keep = sizes >= min_size
    keep[0] = False

    kept = keep[instances]
    out = np.where(kept, instances, 0).astype(np.int32)
    removed_mask = (instances > 0) & (~kept)

    if np.any(out > 0) and np.any(removed_mask):
        out = assign_to_nearest(out, removed_mask)

    return relabel_consecutively_fast(out)


def split_class_core_with_border_cut_fast(class_mask, fracture_arr, border_dilate_percent, roi_margin):
    if not np.any(class_mask):
        return np.zeros_like(class_mask, dtype=np.int32)

    roi = get_bbox_slices(class_mask, margin=roi_margin)
    if roi is None:
        return np.zeros_like(class_mask, dtype=np.int32)

    cm = class_mask[roi]
    fa = fracture_arr[roi]

    core = (fa == 1) & cm
    border = (fa == 3) & cm

    core = remove_small_cc_fast(core, 20)
    border = remove_small_cc_fast(border, 20)

    iters = percent_to_dilate_iters(cm, border_dilate_percent)

    if iters > 0 and np.any(border):
        thick_border = ndimage.binary_dilation(border, iterations=iters)
        thick_border &= cm
    else:
        thick_border = border

    cut_core = core & (~thick_border)

    cc, n = ndimage.label(cut_core)

    if n == 0:
        out_roi = np.zeros_like(cm, dtype=np.int32)
        out_roi[cm] = 1
    else:
        missing_mask = cm & (cc == 0)
        out_roi = assign_to_nearest(cc.astype(np.int32), missing_mask).astype(np.int32)

    out_full = np.zeros_like(class_mask, dtype=np.int32)
    out_full[roi] = out_roi
    return out_full


def apply_pengwin_ids(final, inst, cls):
    labels = np.unique(inst)
    labels = labels[labels > 0]

    if cls == 3:
        start = 1
    elif cls == 1:
        start = 11
    elif cls == 2:
        start = 21
    else:
        raise ValueError(f"Unexpected class {cls}")

    for i, lbl in enumerate(labels):
        final[inst == lbl] = start + i


def collect_cases(fracture_dir):
    cases = {}

    for p in fracture_dir.glob("*.nii.gz"):
        name = p.name

        if name.endswith("_LH.nii.gz"):
            cid = name[:-10]
            part = "LH"
        elif name.endswith("_RH.nii.gz"):
            cid = name[:-10]
            part = "RH"
        elif name.endswith("_S.nii.gz"):
            cid = name[:-9]
            part = "S"
        else:
            continue

        cases.setdefault(cid, {})[part] = p

    return cases


def main():
    args = parse_args()

    anatomy_dir, fracture_dir, out_dir = resolve_runtime_dirs(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Anatomy predictions: {anatomy_dir}")
    print(f"Fracture predictions: {fracture_dir}")
    print(f"Restorations output: {out_dir}")

    if not anatomy_dir.is_dir():
        raise FileNotFoundError(f"Missing anatomy prediction folder: {anatomy_dir}")
    if not fracture_dir.is_dir():
        raise FileNotFoundError(f"Missing fracture prediction folder: {fracture_dir}")

    anatomy_files = sorted(anatomy_dir.glob("*.nii.gz"))
    if args.max_cases:
        anatomy_files = anatomy_files[:args.max_cases]

    fracture_cases = collect_cases(fracture_dir)

    print(f"\nProcessing {len(anatomy_files)} cases")
    print(f"Border dilation: {args.border_dilate_percent}%")
    print(f"ROI margin: {args.roi_margin}\n")

    for ap in tqdm(anatomy_files):
        cid = ap.stem.replace(".nii", "")

        if cid not in fracture_cases:
            print(f"Missing fracture for {cid}")
            continue

        if not all(k in fracture_cases[cid] for k in ["LH", "RH", "S"]):
            print(f"Incomplete fracture for {cid}")
            continue

        anatomy_img, anatomy = read_image(ap)

        _, f_LH = read_image(fracture_cases[cid]["LH"])
        _, f_RH = read_image(fracture_cases[cid]["RH"])
        _, f_S = read_image(fracture_cases[cid]["S"])

        final = np.zeros_like(anatomy, dtype=np.int32)

        for cls, f_arr in zip([1, 2, 3], [f_LH, f_RH, f_S]):
            class_mask = anatomy == cls
            if not np.any(class_mask):
                continue

            inst = split_class_core_with_border_cut_fast(
                class_mask,
                f_arr,
                args.border_dilate_percent,
                args.roi_margin,
            )

            inst = clean_fragments_fast(inst, args.min_fragment_voxels)
            apply_pengwin_ids(final, inst, cls)

        write_image(final.astype(np.uint16), anatomy_img, out_dir / f"{cid}.nii.gz")

    print("\nDone\n")


if __name__ == "__main__":
    main()