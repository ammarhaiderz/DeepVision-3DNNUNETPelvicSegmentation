#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import STUDENT_DATASET_NAME, TEACHER_DATASETS, find_project_root, strip_nii_gz


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a manifest mapping student cases to LH/RH/S teacher probability maps."
    )
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--ignore_missing", action="store_true")
    return parser.parse_args()


def case_id_from_npz(path: Path) -> str:
    return path.stem


def main() -> None:
    args = parse_args()
    project_root = find_project_root(Path(__file__).resolve())
    priors_root = project_root / "student_teacher_pipeline" / "teacher_priors" / args.split
    student_labels = project_root / "nnUNet" / "nnUNet_raw" / STUDENT_DATASET_NAME / "labelsTr"
    if not student_labels.is_dir():
        raise FileNotFoundError(f"Missing student labelsTr: {student_labels}")

    case_ids = [strip_nii_gz(p.name) for p in sorted(student_labels.glob("*.nii.gz"))]
    teacher_maps: dict[str, dict[str, Path]] = {}
    for bone, config in TEACHER_DATASETS.items():
        folder = priors_root / config["dataset_name"]
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing teacher prior folder for {bone}: {folder}")
        teacher_maps[bone] = {case_id_from_npz(path): path for path in sorted(folder.glob("*.npz"))}

    manifest = {
        "description": "Soft teacher probability maps for student distillation.",
        "student_dataset": STUDENT_DATASET_NAME,
        "classes": {"LH": 1, "RH": 2, "S": 3},
        "cases": {},
    }
    missing_rows = []
    for cid in case_ids:
        row = {}
        missing = []
        for bone in ("LH", "RH", "S"):
            path = teacher_maps[bone].get(cid)
            if path is None:
                missing.append(bone)
                continue
            row[bone] = str(path)
        if missing:
            missing_rows.append((cid, missing))
            if not args.ignore_missing:
                raise FileNotFoundError(f"Missing priors for {cid}: {', '.join(missing)}")
            continue
        manifest["cases"][cid] = row

    out_path = project_root / "student_teacher_pipeline" / "teacher_prior_manifest.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest: {out_path}")
    print(f"Complete cases: {len(manifest['cases'])}")
    if missing_rows:
        print(f"Cases with missing priors: {len(missing_rows)}")


if __name__ == "__main__":
    main()

