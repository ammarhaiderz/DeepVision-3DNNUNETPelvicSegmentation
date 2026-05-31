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
FRACTURE_PREDICTIONS_DATASET = "Dataset006_fracture_network_type3"
DEFAULT_RESTORATION_SUBDIR = "restore_type3_script1"


def resolve_runtime_dirs(args):
    anatomy_dir = Path(args.anatomy_dir) if args.anatomy_dir else PROJECT_ROOT / "predictions" / ANATOMY_PREDICTIONS_DATASET
    fracture_dir = Path(args.fracture_dir) if args.fracture_dir else PROJECT_ROOT / "predictions" / FRACTURE_PREDICTIONS_DATASET
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "restorations" / DEFAULT_RESTORATION_SUBDIR
    return anatomy_dir, fracture_dir, out_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Post-process combined 3-class fracture predictions into instance labels.\n\n"
            "Inputs:\n"
            "- anatomy: case.nii.gz (0=bg,1=LH,2=RH,3=S)\n"
            "- fracture: case_LH.nii.gz, case_RH.nii.gz, case_S.nii.gz\n"
            "            with labels:\n"
            "            0=background, 1=largest fragment, 2=small fragments, 3=border\n\n"
            "Output:\n"
            "0 = background\n"
            "Sacrum fragments: 1,2,3,...\n"
            "LH fragments:      11,12,13,...\n"
            "RH fragments:      21,22,23,..."
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
        help="Optional fracture prediction folder. Default: predictions/Dataset006_fracture_network_type3",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional output folder. Default: restorations/restore_type3_script1_auto",
    )

    parser.add_argument("--min_fragment_voxels", type=int, default=1000)
    parser.add_argument("--max_cases", type=int, default=None)

    parser.add_argument(
        "--border_dilate_percent",
        type=float,
        default=1.0,
        help=(
            "Artificial border thickening as percent of class-mask bbox diagonal. "
            "Example: 1.0 means about 1%% of the diagonal."
        ),
    )

    parser.add_argument(
        "--min_border_cc",
        type=int,
        default=20,
        help="Remove tiny border connected components smaller than this."
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


def relabel_consecutively(arr):
    out = np.zeros_like(arr, dtype=np.int32)
    labels = np.unique(arr)
    labels = labels[labels > 0]

    for new_id, old_id in enumerate(labels, start=1):
        out[arr == old_id] = new_id

    return out


def remove_small_cc(mask, min_size, structure=None):
    if not np.any(mask):
        return mask.astype(bool)

    if structure is None:
        structure = ndimage.generate_binary_structure(3, 3)

    cc, n = ndimage.label(mask, structure=structure)
    out = np.zeros_like(mask, dtype=bool)

    for i in range(1, n + 1):
        comp = cc == i
        if int(comp.sum()) >= min_size:
            out[comp] = True

    return out


def assign_to_nearest(labels, target_mask):
    """
    Assign target_mask voxels to nearest existing nonzero label in labels.
    """
    out = labels.copy()

    if not np.any(target_mask):
        return out

    if not np.any(out > 0):
        return out

    _, indices = ndimage.distance_transform_edt(out == 0, return_indices=True)
    nearest = out[tuple(indices)]
    out[target_mask] = nearest[target_mask]
    return out


def compute_bbox_diagonal(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0.0

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extents = (maxs - mins + 1).astype(float)

    return float(np.sqrt(np.sum(extents ** 2)))


def percent_to_dilate_iters(class_mask, percent):
    if percent <= 0 or not np.any(class_mask):
        return 0

    diag = compute_bbox_diagonal(class_mask)
    if diag <= 0:
        return 0

    iters = int(math.ceil((percent / 100.0) * diag))
    return max(0, iters)


def keep_only_largest_cc(mask, structure=None):
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    if structure is None:
        structure = ndimage.generate_binary_structure(3, 3)

    cc, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)

    if n == 1:
        return mask.astype(bool)

    sizes = ndimage.sum(mask, cc, index=np.arange(1, n + 1))
    largest_idx = int(np.argmax(sizes)) + 1
    return cc == largest_idx


def split_class(class_mask, fracture_arr, border_dilate_percent=1.0, min_border_cc=20):
    """
    fracture labels:
      0 = background
      1 = biggest fragment
      2 = all smaller fragments together
      3 = border

    Strategy:
    - restrict to anatomy class
    - keep only the largest connected component of class 1 as the true main fragment
    - move other disconnected class-1 parts into class-2 pool
    - use class-3 border (optionally thickened) to cut/split the small-fragment region
    - assign border voxels back to nearest final fragment
    """
    out = np.zeros_like(fracture_arr, dtype=np.int32)

    if not np.any(class_mask):
        return out

    structure = ndimage.generate_binary_structure(3, 3)

    pred = fracture_arr.copy()
    pred[~class_mask] = 0

    raw_main = pred == 1
    raw_small = pred == 2
    raw_border = pred == 3

    # clean tiny border islands
    border = remove_small_cc(raw_border, min_border_cc, structure=structure)

    # true main = only largest connected component of class 1
    main_mask = keep_only_largest_cc(raw_main, structure=structure)

    # disconnected class-1 islands should belong to small fragments instead
    extra_main_parts = raw_main & (~main_mask)
    small_mask = raw_small | extra_main_parts

    # thicken border a bit
    border_dilate_iters = percent_to_dilate_iters(class_mask, border_dilate_percent)
    if border_dilate_iters > 0 and np.any(border):
        thick_border = ndimage.binary_dilation(border, structure=structure, iterations=border_dilate_iters)
        thick_border = thick_border & class_mask
    else:
        thick_border = border

    next_id = 1

    # keep main fragment as one instance
    if np.any(main_mask):
        out[main_mask] = next_id
        next_id += 1

    # split small fragments after removing the thick border
    small_interior = small_mask & (~thick_border)

    if np.any(small_interior):
        cc_small, n_small = ndimage.label(small_interior, structure=structure)
        for i in range(1, n_small + 1):
            comp = cc_small == i
            if np.any(comp):
                out[comp] = next_id
                next_id += 1

    # If class 2 disappeared completely but anatomy exists, keep fallback
    if not np.any(out > 0) and np.any(class_mask):
        out[class_mask] = 1
        return out.astype(np.int32)

    # reassign remaining predicted fracture voxels that were removed by borders
    predicted_region = (raw_main | raw_small | raw_border) & class_mask
    missing_predicted = predicted_region & (out == 0)
    if np.any(missing_predicted):
        out = assign_to_nearest(out, missing_predicted)

    # optional final fallback: if still nothing exists, keep anatomy as one fragment
    if not np.any(out > 0) and np.any(class_mask):
        out[class_mask] = 1

    return out.astype(np.int32)


def remove_small_labels_and_reassign(instance_map, min_size):
    """
    Keep large instances.
    Small ones are reassigned to nearest kept label if possible.
    If nothing survives, keep the largest original instance.
    """
    if not np.any(instance_map > 0):
        return instance_map.astype(np.int32)

    kept = np.zeros_like(instance_map, dtype=np.int32)
    removed_mask = np.zeros_like(instance_map, dtype=bool)

    labels = np.unique(instance_map)
    labels = labels[labels > 0]

    next_id = 1
    for lbl in labels:
        mask = instance_map == lbl
        if int(mask.sum()) >= min_size:
            kept[mask] = next_id
            next_id += 1
        else:
            removed_mask[mask] = True

    if np.any(kept > 0) and np.any(removed_mask):
        kept = assign_to_nearest(kept, removed_mask)

    if not np.any(kept > 0):
        largest_lbl = max(labels, key=lambda x: int((instance_map == x).sum()))
        kept = np.zeros_like(instance_map, dtype=np.int32)
        kept[instance_map == largest_lbl] = 1

    return relabel_consecutively(kept)


def collect_fracture_cases(fracture_dir):
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
    if len(anatomy_files) == 0:
        print(f"No anatomy files found in {anatomy_dir}")
        return

    if args.max_cases is not None:
        anatomy_files = anatomy_files[:args.max_cases]

    fracture_cases = collect_fracture_cases(fracture_dir)

    print(f"\nProcessing {len(anatomy_files)} cases\n")
    print(f"Border thickening: {args.border_dilate_percent:.3f}% of class-mask bbox diagonal")
    print(f"Min fragment voxels: {args.min_fragment_voxels}")
    print(f"Min border CC: {args.min_border_cc}\n")

    for anatomy_path in tqdm(anatomy_files):
        cid = anatomy_path.stem
        if cid.endswith(".nii"):
            cid = cid[:-4]

        if cid not in fracture_cases:
            print(f"Skipping {cid}: missing fracture files")
            continue

        if not all(k in fracture_cases[cid] for k in ["LH", "RH", "S"]):
            print(f"Skipping {cid}: incomplete fracture files")
            continue

        anatomy_img, anatomy = read_image(anatomy_path)

        _, frac_lh = read_image(fracture_cases[cid]["LH"])
        _, frac_rh = read_image(fracture_cases[cid]["RH"])
        _, frac_s = read_image(fracture_cases[cid]["S"])

        if not (anatomy.shape == frac_lh.shape == frac_rh.shape == frac_s.shape):
            print(
                f"Skipping {cid}: shape mismatch "
                f"(anatomy {anatomy.shape}, LH {frac_lh.shape}, RH {frac_rh.shape}, S {frac_s.shape})"
            )
            continue

        final = np.zeros_like(anatomy, dtype=np.int32)

        # Output ID convention:
        # Sacrum -> 1,2,3,...
        # LH     -> 11,12,13,...
        # RH     -> 21,22,23,...
        class_info = [
            (3, frac_s, 0),    # S
            (1, frac_lh, 10),  # LH
            (2, frac_rh, 20),  # RH
        ]

        for cls, frac_arr, offset in class_info:
            class_mask = anatomy == cls
            if not np.any(class_mask):
                continue

            inst = split_class(
                class_mask=class_mask,
                fracture_arr=frac_arr,
                border_dilate_percent=args.border_dilate_percent,
                min_border_cc=args.min_border_cc,
            )

            inst = remove_small_labels_and_reassign(inst, args.min_fragment_voxels)

            labels = np.unique(inst)
            labels = labels[labels > 0]

            for i, lbl in enumerate(labels, start=1):
                final[inst == lbl] = offset + i

        write_image(final.astype(np.uint16), anatomy_img, out_dir / f"{cid}.nii.gz")

    print(f"\nDone. Saved results to: {out_dir}\n")


if __name__ == "__main__":
    main()