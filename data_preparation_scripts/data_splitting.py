#!/usr/bin/env python3

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

# Folder that contains the label files
INPUT_DIR = Path("aligned_and_converted_data") / "labels"

# Output files
OUTPUT_CSV = Path("fracture_analysis.csv")
OUTPUT_JSON = Path("train_val_test.json")

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
SEED = 42


def strip_nii_suffix(name: str) -> str:
    """Remove .nii or .nii.gz from a filename."""
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def read_label(path: Path) -> np.ndarray:
    """Read a medical label file and return it as a NumPy array."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr


def get_group_ids(arr: np.ndarray):
    """
    Extract fragment IDs by anatomical group.

    Expected ID convention:
    - Sacrum: 1-9
    - Left hipbone: 11-19
    - Right hipbone: 21-29
    """
    ids = np.unique(arr)
    ids = ids[ids > 0]

    s_ids = sorted(int(x) for x in ids if 1 <= x <= 9)
    lh_ids = sorted(int(x) for x in ids if 11 <= x <= 19)
    rh_ids = sorted(int(x) for x in ids if 21 <= x <= 29)

    return s_ids, lh_ids, rh_ids


def make_pattern(has_lh: bool, has_rh: bool, has_s: bool) -> str:
    """Create a pattern string such as LH+RH, LH+S, RH, etc."""
    parts = []
    if has_lh:
        parts.append("LH")
    if has_rh:
        parts.append("RH")
    if has_s:
        parts.append("S")
    return "+".join(parts) if parts else "none"


def analyze_case(path: Path) -> dict:
    """Analyze one label file and extract fracture statistics."""
    arr = read_label(path)
    s_ids, lh_ids, rh_ids = get_group_ids(arr)

    n_frag_s = len(s_ids)
    n_frag_lh = len(lh_ids)
    n_frag_rh = len(rh_ids)

    # A region is treated as fractured if it contains more than one fragment ID
    frac_s = int(n_frag_s > 1)
    frac_lh = int(n_frag_lh > 1)
    frac_rh = int(n_frag_rh > 1)

    has_s_region = int(n_frag_s > 0)
    has_lh_region = int(n_frag_lh > 0)
    has_rh_region = int(n_frag_rh > 0)

    total_fragments = n_frag_s + n_frag_lh + n_frag_rh
    fractured_regions = frac_s + frac_lh + frac_rh
    pattern = make_pattern(bool(frac_lh), bool(frac_rh), bool(frac_s))

    # Simple complexity grouping based on total number of fragments
    if total_fragments <= 3:
        complexity_bucket = "low"
    elif total_fragments <= 6:
        complexity_bucket = "medium"
    else:
        complexity_bucket = "high"

    return {
        "case": strip_nii_suffix(path.name),
        "path": str(path),
        "n_frag_s": n_frag_s,
        "n_frag_lh": n_frag_lh,
        "n_frag_rh": n_frag_rh,
        "frac_s": frac_s,
        "frac_lh": frac_lh,
        "frac_rh": frac_rh,
        "has_s_region": has_s_region,
        "has_lh_region": has_lh_region,
        "has_rh_region": has_rh_region,
        "total_fragments": total_fragments,
        "fractured_regions": fractured_regions,
        "pattern": pattern,
        "complexity_bucket": complexity_bucket,
        "s_ids": s_ids,
        "lh_ids": lh_ids,
        "rh_ids": rh_ids,
    }


def compute_group_sizes(n: int, train_ratio: float, val_ratio: float, test_ratio: float):
    """Convert ratios into concrete case counts for one group."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    # Safety correction in case rounding overshoots
    while n_test < 0:
        if n_train >= n_val and n_train > 0:
            n_train -= 1
        elif n_val > 0:
            n_val -= 1
        n_test = n - n_train - n_val

    return n_train, n_val, n_test


def stratified_pattern_split(case_infos, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Split cases by fracture pattern first, then distribute cases roughly
    evenly across train/val/test while mixing easier and harder cases.
    """
    rng = random.Random(seed)

    by_pattern = defaultdict(list)
    for info in case_infos:
        by_pattern[info["pattern"]].append(info)

    split = {"train": [], "val": [], "test": []}

    for pattern in tqdm(sorted(by_pattern.keys()), desc="Splitting patterns"):
        group = by_pattern[pattern][:]

        # Prevent filename-order bias
        rng.shuffle(group)

        # Sort by difficulty-like properties so round-robin distribution is more balanced
        group = sorted(
            group,
            key=lambda x: (
                x["total_fragments"],
                x["fractured_regions"],
                x["n_frag_lh"],
                x["n_frag_rh"],
                x["n_frag_s"],
            ),
            reverse=True,
        )

        n = len(group)
        n_train, n_val, n_test = compute_group_sizes(n, train_ratio, val_ratio, test_ratio)

        train_count = 0
        val_count = 0
        test_count = 0

        for info in group:
            candidates = []

            if train_count < n_train:
                candidates.append(("train", train_count / max(n_train, 1)))
            if val_count < n_val:
                candidates.append(("val", val_count / max(n_val, 1)))
            if test_count < n_test:
                candidates.append(("test", test_count / max(n_test, 1)))

            # Choose the currently least-filled split
            candidates.sort(key=lambda x: x[1])
            chosen = candidates[0][0]

            split[chosen].append(info)

            if chosen == "train":
                train_count += 1
            elif chosen == "val":
                val_count += 1
            else:
                test_count += 1

    # Final shuffle for cleaner ordering
    for split_name in ["train", "val", "test"]:
        rng.shuffle(split[split_name])

    return split


def write_analysis_csv(case_infos, out_csv: Path):
    """Write per-case fracture statistics to a CSV file."""
    fieldnames = [
        "case",
        "path",
        "n_frag_s",
        "n_frag_lh",
        "n_frag_rh",
        "frac_s",
        "frac_lh",
        "frac_rh",
        "has_s_region",
        "has_lh_region",
        "has_rh_region",
        "total_fragments",
        "fractured_regions",
        "pattern",
        "complexity_bucket",
        "s_ids",
        "lh_ids",
        "rh_ids",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for info in tqdm(sorted(case_infos, key=lambda x: x["case"]), desc="Writing CSV"):
            row = info.copy()
            row["s_ids"] = " ".join(map(str, row["s_ids"]))
            row["lh_ids"] = " ".join(map(str, row["lh_ids"]))
            row["rh_ids"] = " ".join(map(str, row["rh_ids"]))
            writer.writerow(row)


def summarize_split(split_dict):
    """Create summary statistics for train/val/test."""
    summary = {}

    for split_name in ["train", "val", "test"]:
        cases = split_dict[split_name]

        pattern_counts = {}
        complexity_counts = {}

        for c in cases:
            pattern_counts[c["pattern"]] = pattern_counts.get(c["pattern"], 0) + 1
            complexity_counts[c["complexity_bucket"]] = complexity_counts.get(c["complexity_bucket"], 0) + 1

        summary[split_name] = {
            "n_cases": len(cases),
            "cases": sorted(c["case"] for c in cases),
            "frac_lh_cases": int(sum(c["frac_lh"] for c in cases)),
            "frac_rh_cases": int(sum(c["frac_rh"] for c in cases)),
            "frac_s_cases": int(sum(c["frac_s"] for c in cases)),
            "multi_region_fracture_cases": int(sum(c["fractured_regions"] >= 2 for c in cases)),
            "avg_total_fragments": float(np.mean([c["total_fragments"] for c in cases])) if cases else 0.0,
            "patterns": dict(sorted(pattern_counts.items())),
            "complexity_buckets": dict(sorted(complexity_counts.items())),
        }

    return summary


def write_split_json(split_dict, out_json: Path):
    """Write the final split and summary to JSON."""
    summary = summarize_split(split_dict)

    out = {
        "train": summary["train"]["cases"],
        "val": summary["val"]["cases"],
        "test": summary["test"]["cases"],
        "summary": summary,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def print_summary(split_dict):
    """Print a short summary to the terminal."""
    summary = summarize_split(split_dict)

    print("\n===== SPLIT SUMMARY =====\n")
    for split_name in ["train", "val", "test"]:
        s = summary[split_name]
        print(f"{split_name.upper()}:")
        print(f"  n_cases: {s['n_cases']}")
        print(f"  frac_lh_cases: {s['frac_lh_cases']}")
        print(f"  frac_rh_cases: {s['frac_rh_cases']}")
        print(f"  frac_s_cases: {s['frac_s_cases']}")
        print(f"  multi_region_fracture_cases: {s['multi_region_fracture_cases']}")
        print(f"  avg_total_fragments: {s['avg_total_fragments']:.2f}")
        print(f"  complexity_buckets: {s['complexity_buckets']}")
        print(f"  patterns: {s['patterns']}")
        print("")


def main():
    # Check input folder
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder does not exist: {INPUT_DIR}")

    # Collect label files
    files = sorted(list(INPUT_DIR.glob("*.nii")) + list(INPUT_DIR.glob("*.nii.gz")))
    if len(files) == 0:
        raise RuntimeError(f"No .nii or .nii.gz files found in: {INPUT_DIR}")

    print(f"\nFound {len(files)} files in {INPUT_DIR}\n")

    # Analyze all cases
    case_infos = []
    for path in tqdm(files, desc="Analyzing cases"):
        case_infos.append(analyze_case(path))

    case_infos = sorted(case_infos, key=lambda x: x["case"])

    # Create split
    split_dict = stratified_pattern_split(
        case_infos=case_infos,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    # Save outputs
    write_analysis_csv(case_infos, OUTPUT_CSV)
    write_split_json(split_dict, OUTPUT_JSON)

    # Show terminal summary
    print_summary(split_dict)

    print(f"Saved analysis CSV: {OUTPUT_CSV}")
    print(f"Saved split JSON:   {OUTPUT_JSON}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()