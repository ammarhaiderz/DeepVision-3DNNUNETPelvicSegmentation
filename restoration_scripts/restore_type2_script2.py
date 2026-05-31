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
FRACTURE_PREDICTIONS_DATASET = "Dataset005_fracture_network_type2"
DEFAULT_RESTORATION_SUBDIR = "restore_type2_script2"


def resolve_runtime_dirs(args):
    anatomy_dir = Path(args.anatomy_dir) if args.anatomy_dir else PROJECT_ROOT / "predictions" / ANATOMY_PREDICTIONS_DATASET
    fracture_dir = Path(args.fracture_dir) if args.fracture_dir else PROJECT_ROOT / "predictions" / FRACTURE_PREDICTIONS_DATASET
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "restorations" / DEFAULT_RESTORATION_SUBDIR
    return anatomy_dir, fracture_dir, out_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create fragment instance labels from anatomy + split fracture labels.\n\n"
            "Inputs:\n"
            "- anatomy: case.nii.gz (0=bg,1=LH,2=RH,3=S)\n"
            "- fracture: case_LH.nii.gz, case_RH.nii.gz, case_S.nii.gz\n"
            "            with labels:\n"
            "            0=background, 1=main component, 2=all smaller components\n\n"
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
        help="Optional fracture prediction folder. Default: predictions/Dataset005_fracture_network_type2",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional output folder. Default: restorations/restore_type2_script2_auto",
    )
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

    for new_id, old_id in enumerate(labels, start=1):
        out[arr == old_id] = new_id

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
        # fallback: keep largest original instance
        largest_lbl = max(labels, key=lambda x: int((instance_map == x).sum()))
        kept = np.zeros_like(instance_map, dtype=np.int32)
        kept[instance_map == largest_lbl] = 1

    return relabel_consecutively(kept)


def split_fracture_class(class_mask, fracture_arr):
    """
    fracture labels:
      0 = background
      1 = main component
      2 = all smaller components together

    Improved strategy:
    - restrict everything to the anatomy class
    - inside label 1, keep only the largest connected component as the true main fragment
    - any other disconnected label-1 islands are moved to the small-fragment region
    - split the final small-fragment region into connected components
    """
    out = np.zeros_like(fracture_arr, dtype=np.int32)

    if not np.any(class_mask):
        return out

    fracture_inside = fracture_arr.copy()
    fracture_inside[~class_mask] = 0

    raw_main_mask = fracture_inside == 1
    raw_small_mask = fracture_inside == 2

    main_mask = np.zeros_like(raw_main_mask, dtype=bool)
    small_mask = raw_small_mask.copy()

    # Only largest CC of label 1 remains the true main fragment
    if np.any(raw_main_mask):
        cc_main, n_main = ndimage.label(
            raw_main_mask,
            structure=ndimage.generate_binary_structure(3, 3)
        )

        if n_main == 1:
            main_mask = raw_main_mask
        else:
            sizes = ndimage.sum(raw_main_mask, cc_main, index=np.arange(1, n_main + 1))
            largest_idx = int(np.argmax(sizes)) + 1

            main_mask = cc_main == largest_idx

            # move all other disconnected label-1 parts to the small-fragment pool
            extra_main_parts = (cc_main > 0) & (~main_mask)
            small_mask |= extra_main_parts

    next_id = 1

    # Main component stays one instance
    if np.any(main_mask):
        out[main_mask] = next_id
        next_id += 1

    # Split smaller components by connected components
    if np.any(small_mask):
        cc_small, n_small = ndimage.label(
            small_mask,
            structure=ndimage.generate_binary_structure(3, 3)
        )
        for i in range(1, n_small + 1):
            out[cc_small == i] = next_id
            next_id += 1

    # Safety fallback:
    # if neither label 1 nor 2 exists but anatomy class exists, keep anatomy as one instance
    if not np.any(out > 0) and np.any(class_mask):
        out[class_mask] = 1

    return out.astype(np.int32)


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


def map_local_to_pengwin_ids(inst, base_id):
    """
    Convert local consecutive labels 1,2,3,... into:
    base_id, base_id+1, base_id+2, ...

    Examples:
      base_id=1  -> 1,2,3,...    (sacrum)
      base_id=11 -> 11,12,13,... (LH)
      base_id=21 -> 21,22,23,... (RH)
    """
    out = np.zeros_like(inst, dtype=np.int32)

    labels = np.unique(inst)
    labels = labels[labels > 0]

    for i, lbl in enumerate(labels):
        out[inst == lbl] = base_id + i

    return out


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

        # anatomy labels:
        # 1 = LH
        # 2 = RH
        # 3 = S
        #
        # desired output ranges:
        # S  -> 1,2,3,...
        # LH -> 11,12,13,...
        # RH -> 21,22,23,...

        region_info = [
            ("LH", 1, frac_lh, 11),
            ("RH", 2, frac_rh, 21),
            ("S",  3, frac_s, 1),
        ]

        for region_name, anatomy_label, frac_arr, base_id in region_info:
            class_mask = anatomy == anatomy_label
            if not np.any(class_mask):
                continue

            inst = split_fracture_class(class_mask, frac_arr)
            inst = remove_small_labels_and_reassign(inst, args.min_fragment_voxels)
            inst = map_local_to_pengwin_ids(inst, base_id)

            final[inst > 0] = inst[inst > 0]

        write_image(final.astype(np.uint16), anatomy_img, out_dir / f"{cid}.nii.gz")

    print(f"\nDone. Saved results to: {out_dir}\n")


if __name__ == "__main__":
    main()