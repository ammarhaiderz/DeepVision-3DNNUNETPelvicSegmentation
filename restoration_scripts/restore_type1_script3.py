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
DEFAULT_RESTORATION_SUBDIR = "restore_type1_script3"


def resolve_runtime_dirs(args):
    anatomy_dir = Path(args.anatomy_dir) if args.anatomy_dir else PROJECT_ROOT / "predictions" / ANATOMY_PREDICTIONS_DATASET
    fracture_dir = Path(args.fracture_dir) if args.fracture_dir else PROJECT_ROOT / "predictions" / FRACTURE_PREDICTIONS_DATASET
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "restorations" / DEFAULT_RESTORATION_SUBDIR
    return anatomy_dir, fracture_dir, out_dir


# =========================
# INTERNAL HYPERPARAMETERS
# =========================
MIN_BORDER_CC_VOXELS = 30

# normal dilation of predicted border
BASE_BORDER_DILATE_PERCENT = 1.8
MAX_BASE_DILATE_ITERS = 6

# if a border connected component is large enough, extend it through the bone
ENABLE_BORDER_EXTENSION = True
EXTEND_ONLY_LARGE_BORDERS = True
LARGE_BORDER_THRESHOLD = 120  # voxels

# thickness of the synthetic extended cut
EXTENSION_HALF_THICKNESS = 2.5  # in voxels around the estimated border line

# optional extra dilation after extension
POST_EXTENSION_DILATE_ITERS = 1

# if the extended cut removes too much, it can destroy the bone;
# so require at least this much remaining core before accepting it
MIN_REMAINING_CORE_RATIO = 0.02

# cleanup
MIN_CORE_CC_VOXELS = 20
MIN_FRAGMENT_VOXELS = 1000

# ROI
ROI_MARGIN = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split anatomy into fragments using fracture predictions.\n\n"
            "Inputs:\n"
            "- anatomy: case.nii.gz (0=bg,1=LH,2=RH,3=S)\n"
            "- fracture: case_LH.nii.gz, case_RH.nii.gz, case_S.nii.gz\n"
            "            (0=bg,1=core,2=boundary,3=border)\n\n"
            "Logic:\n"
            "- Connected components are computed on CORE only\n"
            "- BORDER is thickened and optionally extended through the bone ROI\n"
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
        help="Optional output folder. Default: restorations/restore_type1_script3_auto",
    )

    parser.add_argument("--min_fragment_voxels", type=int, default=MIN_FRAGMENT_VOXELS)
    parser.add_argument("--max_cases", type=int, default=None)
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


def percent_to_dilate_iters(mask, percent, max_iters=6):
    if percent <= 0 or not np.any(mask):
        return 0

    diag = compute_bbox_diagonal(mask)
    if diag <= 0:
        return 0

    iters = int(math.ceil((percent / 100.0) * diag))
    return max(1, min(iters, max_iters))


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


def get_component_masks(mask, min_size=1):
    if not np.any(mask):
        return []

    cc, n = ndimage.label(mask)
    if n == 0:
        return []

    sizes = np.bincount(cc.ravel())
    comps = []
    for lbl in range(1, n + 1):
        if sizes[lbl] >= min_size:
            comps.append(cc == lbl)
    return comps


def principal_direction_from_points(coords):
    """
    coords: (N, 3) array in z,y,x order
    returns center and dominant direction unit vector
    """
    center = coords.mean(axis=0)
    centered = coords - center

    if coords.shape[0] < 2:
        return center, None

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, np.argmax(eigvals)]

    norm = np.linalg.norm(direction)
    if norm == 0:
        return center, None

    direction = direction / norm
    return center, direction


def make_extended_cut_from_component(component_mask, class_mask, half_thickness=2.5):
    """
    Extend a border component through the whole class ROI.

    We approximate the border component by its principal axis in 3D.
    Then we create a thick plane orthogonal to the shortest axis?
    No:
    For a line-like fracture border, the first PCA axis runs ALONG the border.
    The separating surface should contain that line and extend across the bone.
    In voxel space, a simple practical approximation is:
      - find dominant line direction v
      - create all voxels whose perpendicular distance to that line is small
      - restrict to class_mask
    This often creates a stronger cut through the bone.
    """
    coords = np.argwhere(component_mask)
    if coords.shape[0] < 2:
        return np.zeros_like(component_mask, dtype=bool)

    center, direction = principal_direction_from_points(coords)
    if direction is None:
        return np.zeros_like(component_mask, dtype=bool)

    bone_coords = np.argwhere(class_mask)
    if bone_coords.size == 0:
        return np.zeros_like(component_mask, dtype=bool)

    # distance from each bone voxel to the fitted line
    rel = bone_coords - center[None, :]
    proj_len = rel @ direction
    proj = np.outer(proj_len, direction)
    perp = rel - proj
    dist = np.linalg.norm(perp, axis=1)

    cut_coords = bone_coords[dist <= half_thickness]

    out = np.zeros_like(component_mask, dtype=bool)
    if cut_coords.size > 0:
        out[tuple(cut_coords.T)] = True

    return out


def build_strong_cut_mask(class_mask, border_mask):
    """
    Build stronger cut mask:
    1) keep real predicted border
    2) dilate it
    3) for sufficiently large border CCs, extend through whole bone ROI
    4) optional post dilation
    """
    border_mask = remove_small_cc_fast(border_mask, MIN_BORDER_CC_VOXELS)

    if not np.any(border_mask):
        return np.zeros_like(border_mask, dtype=bool)

    iters = percent_to_dilate_iters(
        class_mask,
        BASE_BORDER_DILATE_PERCENT,
        max_iters=MAX_BASE_DILATE_ITERS,
    )

    if iters > 0:
        thick = ndimage.binary_dilation(border_mask, iterations=iters)
        thick &= class_mask
    else:
        thick = border_mask.copy()

    strong_cut = thick.copy()

    if ENABLE_BORDER_EXTENSION:
        comps = get_component_masks(border_mask, min_size=MIN_BORDER_CC_VOXELS)
        for comp in comps:
            comp_size = int(comp.sum())

            if EXTEND_ONLY_LARGE_BORDERS and comp_size < LARGE_BORDER_THRESHOLD:
                continue

            ext = make_extended_cut_from_component(
                comp,
                class_mask,
                half_thickness=EXTENSION_HALF_THICKNESS,
            )
            strong_cut |= ext

    strong_cut &= class_mask

    if POST_EXTENSION_DILATE_ITERS > 0 and np.any(strong_cut):
        strong_cut = ndimage.binary_dilation(
            strong_cut,
            iterations=POST_EXTENSION_DILATE_ITERS
        )
        strong_cut &= class_mask

    return strong_cut


def split_class_core_with_strong_border_cut(class_mask, fracture_arr):
    if not np.any(class_mask):
        return np.zeros_like(class_mask, dtype=np.int32)

    roi = get_bbox_slices(class_mask, margin=ROI_MARGIN)
    if roi is None:
        return np.zeros_like(class_mask, dtype=np.int32)

    cm = class_mask[roi]
    fa = fracture_arr[roi]

    core = (fa == 1) & cm
    border = (fa == 3) & cm

    core = remove_small_cc_fast(core, MIN_CORE_CC_VOXELS)
    border = remove_small_cc_fast(border, MIN_BORDER_CC_VOXELS)

    strong_cut = build_strong_cut_mask(cm, border)

    cut_core = core & (~strong_cut)

    # safety: if the cut destroys almost everything, fall back to simpler cut
    if np.any(core):
        remaining_ratio = cut_core.sum() / max(1, core.sum())
    else:
        remaining_ratio = 0.0

    if remaining_ratio < MIN_REMAINING_CORE_RATIO:
        # fallback: only use base thick border, not aggressive extension
        iters = percent_to_dilate_iters(
            cm,
            BASE_BORDER_DILATE_PERCENT,
            max_iters=MAX_BASE_DILATE_ITERS,
        )
        if iters > 0 and np.any(border):
            fallback_cut = ndimage.binary_dilation(border, iterations=iters)
            fallback_cut &= cm
        else:
            fallback_cut = border.copy()

        cut_core = core & (~fallback_cut)

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
    print(f"Base border dilate percent: {BASE_BORDER_DILATE_PERCENT}%")
    print(f"Large border threshold: {LARGE_BORDER_THRESHOLD} voxels")
    print(f"Extension enabled: {ENABLE_BORDER_EXTENSION}")
    print(f"Extension half thickness: {EXTENSION_HALF_THICKNESS}")
    print(f"Post extension dilation: {POST_EXTENSION_DILATE_ITERS}")
    print(f"ROI margin: {ROI_MARGIN}\n")

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

            inst = split_class_core_with_strong_border_cut(class_mask, f_arr)
            inst = clean_fragments_fast(inst, args.min_fragment_voxels)
            apply_pengwin_ids(final, inst, cls)

        write_image(final.astype(np.uint16), anatomy_img, out_dir / f"{cid}.nii.gz")

    print("\nDone\n")


if __name__ == "__main__":
    main()