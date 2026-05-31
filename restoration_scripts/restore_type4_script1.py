#!/usr/bin/env python3

import argparse
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
FRACTURE_PREDICTIONS_DATASET = "Dataset007_fracture_network_type4"
DEFAULT_RESTORATION_SUBDIR = "restore_type4_script1"


def resolve_runtime_dirs(args):
    anatomy_dir = Path(args.anatomy_dir) if args.anatomy_dir else PROJECT_ROOT / "predictions" / ANATOMY_PREDICTIONS_DATASET
    fracture_dir = Path(args.fracture_dir) if args.fracture_dir else PROJECT_ROOT / "predictions" / FRACTURE_PREDICTIONS_DATASET
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "restorations" / DEFAULT_RESTORATION_SUBDIR
    return anatomy_dir, fracture_dir, out_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split anatomy into fragments using per-region fracture predictions.\n\n"
            "Inputs:\n"
            "- anatomy:  case.nii.gz (0=bg,1=LH,2=RH,3=S)\n"
            "- fracture: case_LH.nii.gz, case_RH.nii.gz, case_S.nii.gz\n"
            "            (0=bg,1=bone,2=border)\n\n"
            "Assumption:\n"
            "- border class (2) is already slightly dilated and is used directly\n"
            "  as split region.\n\n"
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
        help="Optional fracture prediction folder. Default: predictions/Dataset007_fracture_network_type4",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional output folder. Default: restorations/restore_type4_script1_auto",
    )

    parser.add_argument("--min_border_voxels", type=int, default=20)
    parser.add_argument("--min_fragment_voxels", type=int, default=1000)
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


def relabel_consecutively(arr):
    out = np.zeros_like(arr, dtype=np.int32)
    labels = np.unique(arr)
    labels = labels[labels > 0]

    for i, lbl in enumerate(labels, start=1):
        out[arr == lbl] = i

    return out


def remove_small_cc(mask, min_size):
    if not np.any(mask):
        return mask

    cc, n = ndimage.label(mask)
    out = np.zeros_like(mask, dtype=bool)

    for i in range(1, n + 1):
        comp = cc == i
        if comp.sum() >= min_size:
            out[comp] = True

    return out


def assign_to_nearest(labels, target_mask):
    if not np.any(target_mask):
        return labels

    if not np.any(labels > 0):
        return labels

    _, indices = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    nearest = labels[tuple(indices)]
    labels[target_mask] = nearest[target_mask]

    return labels


def split_class_with_border(class_mask, fracture_arr, min_border_voxels):
    """
    class_mask: binary mask for one anatomy class
    fracture_arr: per-region prediction with labels
                  0=bg, 1=bone, 2=border
    """
    if not np.any(class_mask):
        return np.zeros_like(class_mask, dtype=np.int32)

    border_mask = (fracture_arr == 2) & class_mask

    # remove tiny noisy border islands
    border_mask = remove_small_cc(border_mask, min_border_voxels)

    # cut class by border
    interior = class_mask & (~border_mask)

    cc, n = ndimage.label(interior)

    # fallback: if everything got removed, keep whole class as one fragment
    if n == 0:
        out = np.zeros_like(class_mask, dtype=np.int32)
        out[class_mask] = 1
        return out

    # fill removed border voxels back to nearest fragment
    full = assign_to_nearest(cc, class_mask & (cc == 0))
    return full.astype(np.int32)


def clean_fragments(instances, min_size):
    if not np.any(instances > 0):
        return instances

    out = np.zeros_like(instances, dtype=np.int32)
    removed_mask = np.zeros_like(instances, dtype=bool)

    labels = np.unique(instances)
    labels = labels[labels > 0]

    next_id = 1
    for lbl in labels:
        mask = instances == lbl
        if mask.sum() >= min_size:
            out[mask] = next_id
            next_id += 1
        else:
            removed_mask[mask] = True

    # if all were removed, keep largest
    if not np.any(out > 0):
        largest_lbl = None
        largest_size = -1
        for lbl in labels:
            sz = np.sum(instances == lbl)
            if sz > largest_size:
                largest_size = sz
                largest_lbl = lbl

        if largest_lbl is not None:
            out[instances == largest_lbl] = 1
            removed_mask = (instances > 0) & (instances != largest_lbl)

    if np.any(out > 0) and np.any(removed_mask):
        out = assign_to_nearest(out, removed_mask)

    return relabel_consecutively(out)


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
    if args.max_cases is not None:
        anatomy_files = anatomy_files[:args.max_cases]

    fracture_cases = collect_fracture_cases(fracture_dir)

    print(f"\nProcessing {len(anatomy_files)} cases\n")

    final_id_starts = {
        3: 1,   # Sacrum -> 1,2,3,...
        1: 11,  # LH     -> 11,12,13,...
        2: 21   # RH     -> 21,22,23,...
    }

    for anatomy_path in tqdm(anatomy_files, desc="Cases"):
        cid = anatomy_path.stem.replace(".nii", "")

        if cid not in fracture_cases:
            print(f"Missing fracture prediction for {cid}")
            continue

        if not all(k in fracture_cases[cid] for k in ["LH", "RH", "S"]):
            print(f"Incomplete fracture prediction for {cid}")
            continue

        anatomy_img, anatomy = read_image(anatomy_path)

        _, fracture_LH = read_image(fracture_cases[cid]["LH"])
        _, fracture_RH = read_image(fracture_cases[cid]["RH"])
        _, fracture_S = read_image(fracture_cases[cid]["S"])

        if anatomy.shape != fracture_LH.shape or anatomy.shape != fracture_RH.shape or anatomy.shape != fracture_S.shape:
            print(f"Shape mismatch for {cid}")
            continue

        final = np.zeros_like(anatomy, dtype=np.int32)

        for cls, fracture_arr in zip([1, 2, 3], [fracture_LH, fracture_RH, fracture_S]):
            class_mask = anatomy == cls
            if not np.any(class_mask):
                continue

            inst = split_class_with_border(
                class_mask=class_mask,
                fracture_arr=fracture_arr,
                min_border_voxels=args.min_border_voxels
            )

            inst = clean_fragments(
                instances=inst,
                min_size=args.min_fragment_voxels
            )

            labels = np.unique(inst)
            labels = labels[labels > 0]

            start_id = final_id_starts[cls]

            for i, lbl in enumerate(labels):
                final[inst == lbl] = start_id + i

        write_image(final.astype(np.uint16), anatomy_img, out_dir / f"{cid}.nii.gz")

    print("\nDone\n")


if __name__ == "__main__":
    main()