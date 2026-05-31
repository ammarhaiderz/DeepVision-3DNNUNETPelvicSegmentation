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
FRACTURE_PREDICTIONS_DATASET = "Dataset008_fracture_network_type5"
DEFAULT_RESTORATION_SUBDIR = "restore_type5_script2"


def resolve_runtime_dirs(args):
    anatomy_dir = Path(args.anatomy_dir) if args.anatomy_dir else PROJECT_ROOT / "predictions" / ANATOMY_PREDICTIONS_DATASET
    fracture_dir = Path(args.fracture_dir) if args.fracture_dir else PROJECT_ROOT / "predictions" / FRACTURE_PREDICTIONS_DATASET
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "restorations" / DEFAULT_RESTORATION_SUBDIR
    return anatomy_dir, fracture_dir, out_dir


# =========================
# Fixed internal hyperparameters
# =========================

MIN_SEED_VOXELS = 20
MIN_FINAL_FRAGMENT_VOXELS = 1000

ADAPTIVE_EROSION = True

EROSION_SIZE_THR_1 = 2000
EROSION_SIZE_THR_2 = 8000
EROSION_SIZE_THR_3 = 20000
MAX_EROSION_ITERS = 3

MIN_ERODED_PIECE_VOXELS = 200
MIN_SPLIT_PIECES = 2
MAX_PIECE_SIZE_RATIO = 0.98
MIN_SPLIT_FRACTION = 0.05


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Restore fragment instances from anatomy + per-region split-bone predictions.\n\n"
            "Inputs:\n"
            "- anatomy:  case.nii.gz (0=bg,1=LH,2=RH,3=S)\n"
            "- fracture: case_LH.nii.gz, case_RH.nii.gz, case_S.nii.gz\n"
            "            (0=background/removed separation, 1=bone)\n\n"
            "Pipeline:\n"
            "1) Build seed components from predicted bone inside each anatomy region\n"
            "2) Use adaptive per-component erosion refinement (fixed internal hyperparameters)\n"
            "3) Restore full anatomy region by nearest-seed assignment\n"
            "4) Remove tiny final fragments\n"
            "5) Restore removed voxels again to nearest surviving fragment\n\n"
            "Output IDs:\n"
            "- Sacrum fragments: 1,2,3,...\n"
            "- LH fragments:     11,12,13,...\n"
            "- RH fragments:     21,22,23,..."
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
        help="Optional fracture prediction folder. Default: predictions/Dataset008_fracture_network_type5",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional output folder. Default: restorations/restore_type5_script2_auto",
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
        if int(comp.sum()) >= min_size:
            out[comp] = True

    return out


def assign_to_nearest(labels, target_mask):
    """
    labels: int array, 0=unassigned, >0 fragment ids
    target_mask: voxels that should be filled from nearest existing label
    """
    if not np.any(target_mask):
        return labels

    if not np.any(labels > 0):
        return labels

    _, indices = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    nearest = labels[tuple(indices)]
    labels[target_mask] = nearest[target_mask]
    return labels


def get_bbox(mask, pad=0):
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None

    z0, y0, x0 = [int(c.min()) for c in coords]
    z1, y1, x1 = [int(c.max()) + 1 for c in coords]

    shape = mask.shape
    z0 = max(0, z0 - pad)
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)

    z1 = min(shape[0], z1 + pad)
    y1 = min(shape[1], y1 + pad)
    x1 = min(shape[2], x1 + pad)

    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))


def choose_max_erosion_iters(comp_size):
    """
    Adaptive policy:
    - smaller than EROSION_SIZE_THR_1 -> 0
    - >= EROSION_SIZE_THR_1 -> 1
    - >= EROSION_SIZE_THR_2 -> 2
    - >= EROSION_SIZE_THR_3 -> 3
    capped by MAX_EROSION_ITERS
    """
    allowed = 0
    if comp_size >= EROSION_SIZE_THR_1:
        allowed = 1
    if comp_size >= EROSION_SIZE_THR_2:
        allowed = 2
    if comp_size >= EROSION_SIZE_THR_3:
        allowed = 3

    return min(allowed, MAX_EROSION_ITERS)


def evaluate_eroded_split(comp_crop, erosion_iters):
    """
    Try a specific erosion level on one cropped connected component.
    Return (list_of_piece_masks, did_split).
    """
    if not np.any(comp_crop):
        return None, False

    structure = ndimage.generate_binary_structure(3, 1)

    eroded = ndimage.binary_erosion(
        comp_crop,
        structure=structure,
        iterations=erosion_iters,
        border_value=0
    )

    if not np.any(eroded):
        return None, False

    cc, n = ndimage.label(eroded)
    if n < MIN_SPLIT_PIECES:
        return None, False

    kept_masks = []
    sizes = []

    for i in range(1, n + 1):
        piece = cc == i
        sz = int(piece.sum())
        if sz >= MIN_ERODED_PIECE_VOXELS:
            kept_masks.append(piece)
            sizes.append(sz)

    if len(kept_masks) < MIN_SPLIT_PIECES:
        return None, False

    original_size = int(comp_crop.sum())
    largest_ratio = max(sizes) / max(original_size, 1)
    kept_fraction = sum(sizes) / max(original_size, 1)

    if largest_ratio > MAX_PIECE_SIZE_RATIO:
        return None, False

    if kept_fraction < MIN_SPLIT_FRACTION:
        return None, False

    return kept_masks, True


def refine_seed_components_adaptive(seed_mask):
    """
    Refine seed mask:
    - each connected component is treated separately
    - a larger component is allowed more erosion attempts
    - for a given component we try erosion 1..max_allowed and stop at the first valid split
    - if no valid split is found, keep the original component as one seed
    """
    if not np.any(seed_mask):
        return np.zeros_like(seed_mask, dtype=np.int32)

    cc, n = ndimage.label(seed_mask)
    refined = np.zeros_like(seed_mask, dtype=np.int32)
    next_id = 1

    for lbl in range(1, n + 1):
        comp = cc == lbl
        comp_size = int(comp.sum())

        if not ADAPTIVE_EROSION:
            refined[comp] = next_id
            next_id += 1
            continue

        max_allowed = choose_max_erosion_iters(comp_size)

        if max_allowed <= 0:
            refined[comp] = next_id
            next_id += 1
            continue

        bbox = get_bbox(comp, pad=max_allowed + 2)
        comp_crop = comp[bbox]

        accepted_split = None

        for erosion_iters in range(1, max_allowed + 1):
            split_masks, did_split = evaluate_eroded_split(
                comp_crop=comp_crop,
                erosion_iters=erosion_iters
            )

            if did_split:
                accepted_split = split_masks
                break

        if accepted_split is None:
            refined[comp] = next_id
            next_id += 1
            continue

        for piece_crop in accepted_split:
            piece_full = np.zeros_like(seed_mask, dtype=bool)
            piece_full[bbox] = piece_crop
            refined[piece_full] = next_id
            next_id += 1

    return refined


def split_class_with_prediction(class_mask, fracture_arr):
    """
    class_mask: binary mask for one anatomy class
    fracture_arr: per-region binary prediction
                  0=background/removed separation
                  1=bone
    """
    if not np.any(class_mask):
        return np.zeros_like(class_mask, dtype=np.int32)

    seed_mask = (fracture_arr == 1) & class_mask
    seed_mask = remove_small_cc(seed_mask, MIN_SEED_VOXELS)

    seed_labels = refine_seed_components_adaptive(seed_mask)

    if not np.any(seed_labels > 0):
        out = np.zeros_like(class_mask, dtype=np.int32)
        out[class_mask] = 1
        return out

    missing_mask = class_mask & (seed_labels == 0)
    restored = assign_to_nearest(seed_labels.copy(), missing_mask)

    return restored.astype(np.int32)


def filter_and_restore_final_fragments(instances, class_mask):
    """
    After full restoration:
    1) remove tiny final fragments
    2) if all were removed, keep largest one
    3) restore removed voxels to nearest surviving fragment
    """
    if not np.any(instances > 0):
        return instances

    out = np.zeros_like(instances, dtype=np.int32)
    removed_mask = np.zeros_like(instances, dtype=bool)

    labels = np.unique(instances)
    labels = labels[labels > 0]

    next_id = 1
    kept_any = False

    for lbl in labels:
        mask = instances == lbl
        if int(mask.sum()) >= MIN_FINAL_FRAGMENT_VOXELS:
            out[mask] = next_id
            next_id += 1
            kept_any = True
        else:
            removed_mask[mask] = True

    if not kept_any:
        largest_lbl = None
        largest_size = -1

        for lbl in labels:
            sz = int(np.sum(instances == lbl))
            if sz > largest_size:
                largest_size = sz
                largest_lbl = lbl

        if largest_lbl is not None:
            out[instances == largest_lbl] = 1
            removed_mask = (instances > 0) & (instances != largest_lbl)

    unassigned_inside_class = class_mask & (out == 0)
    restore_mask = removed_mask | unassigned_inside_class

    if np.any(out > 0) and np.any(restore_mask):
        out = assign_to_nearest(out, restore_mask)

    out[~class_mask] = 0
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


def get_region_base_id(cls):
    """
    anatomy labels:
    1 = LH -> 11,12,13,...
    2 = RH -> 21,22,23,...
    3 = S  -> 1,2,3,...
    """
    if cls == 3:
        return 1
    if cls == 1:
        return 11
    if cls == 2:
        return 21
    raise ValueError(f"Unexpected class: {cls}")


def write_region_coded_instances(final, inst, cls):
    labels = np.unique(inst)
    labels = labels[labels > 0]

    base_id = get_region_base_id(cls)

    for offset, lbl in enumerate(labels):
        final[inst == lbl] = base_id + offset

    return final


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
    fracture_cases = collect_fracture_cases(fracture_dir)

    print(f"\nProcessing {len(anatomy_files)} cases\n")
    print("Using fixed internal hyperparameters:")
    print(f"  MIN_SEED_VOXELS = {MIN_SEED_VOXELS}")
    print(f"  MIN_FINAL_FRAGMENT_VOXELS = {MIN_FINAL_FRAGMENT_VOXELS}")
    print(f"  ADAPTIVE_EROSION = {ADAPTIVE_EROSION}")
    print(f"  EROSION_SIZE_THR_1 = {EROSION_SIZE_THR_1}")
    print(f"  EROSION_SIZE_THR_2 = {EROSION_SIZE_THR_2}")
    print(f"  EROSION_SIZE_THR_3 = {EROSION_SIZE_THR_3}")
    print(f"  MAX_EROSION_ITERS = {MAX_EROSION_ITERS}")
    print(f"  MIN_ERODED_PIECE_VOXELS = {MIN_ERODED_PIECE_VOXELS}")
    print(f"  MIN_SPLIT_PIECES = {MIN_SPLIT_PIECES}")
    print(f"  MAX_PIECE_SIZE_RATIO = {MAX_PIECE_SIZE_RATIO}")
    print(f"  MIN_SPLIT_FRACTION = {MIN_SPLIT_FRACTION}\n")

    for anatomy_path in tqdm(anatomy_files, desc="Cases"):
        cid = anatomy_path.stem.replace(".nii", "")

        if cid not in fracture_cases:
            print(f"Missing predictions for {cid}")
            continue

        if not all(k in fracture_cases[cid] for k in ["LH", "RH", "S"]):
            print(f"Incomplete predictions for {cid}")
            continue

        anatomy_img, anatomy = read_image(anatomy_path)

        _, fracture_LH = read_image(fracture_cases[cid]["LH"])
        _, fracture_RH = read_image(fracture_cases[cid]["RH"])
        _, fracture_S = read_image(fracture_cases[cid]["S"])

        if (
            anatomy.shape != fracture_LH.shape or
            anatomy.shape != fracture_RH.shape or
            anatomy.shape != fracture_S.shape
        ):
            print(f"Shape mismatch for {cid}")
            continue

        final = np.zeros_like(anatomy, dtype=np.int32)

        # Process in order: sacrum, LH, RH
        for cls, fracture_arr in [(3, fracture_S), (1, fracture_LH), (2, fracture_RH)]:
            class_mask = anatomy == cls
            if not np.any(class_mask):
                continue

            inst = split_class_with_prediction(
                class_mask=class_mask,
                fracture_arr=fracture_arr
            )

            inst = filter_and_restore_final_fragments(
                instances=inst,
                class_mask=class_mask
            )

            final = write_region_coded_instances(final, inst, cls)

        write_image(final.astype(np.uint16), anatomy_img, out_dir / f"{cid}.nii.gz")

    print("\nDone\n")


if __name__ == "__main__":
    main()