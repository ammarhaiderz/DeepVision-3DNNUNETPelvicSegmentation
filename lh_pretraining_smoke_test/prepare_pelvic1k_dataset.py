#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import re
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from common import (
    PELVIC1K_DATASET_NAME,
    nnunet_paths,
    same_geometry,
    save_json,
)


IMAGE_SUFFIX = "_data.nii.gz"
LABEL_SUFFIX = "_mask_4label.nii.gz"
EXPECTED_LABELS = {0, 1, 2, 3, 4}


@dataclass(frozen=True)
class PelvicCase:
    source: str
    source_case_id: str
    image_path: Path
    label_path: Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert downloaded CTPelvic1K image/annotation pairs into an "
            "nnU-Net v2 multiclass pretraining dataset."
        )
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help=(
            "Folder containing CTPelvic1K files. The search is recursive and "
            "expects *_data.nii.gz and *_mask_4label.nii.gz files. Dataset 7 "
            "prefix differences and unlabeled images are handled explicitly."
        ),
    )
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Optional small subset for a smoke test.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Remove the generated Dataset016 folders before recreating them.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Parallel workers for validation/copying into the nnU-Net dataset.",
    )
    return parser.parse_args()


def source_and_case_id(path: Path, suffix: str) -> tuple[str, str]:
    stem = path.name[: -len(suffix)]
    lowered = str(path).lower()
    match = re.match(r"dataset([0-9]+)_(.+)", stem, flags=re.IGNORECASE)
    if match:
        source = f"dataset{match.group(1)}"
        case_id = match.group(2)
    elif "dataset7" in lowered or "clinic_metal" in stem.lower():
        source = "dataset7"
        case_id = stem
    elif "dataset6" in lowered:
        source = "dataset6"
        case_id = stem
    else:
        source = "unknown"
        case_id = stem
    return source.lower(), case_id.lower()


def discover_pairs(data_root: Path) -> tuple[list[PelvicCase], list[Path]]:
    def is_helper_path(path: Path) -> bool:
        return any(part.startswith("_") for part in path.relative_to(data_root).parts[:-1])

    def index_files(suffix: str) -> dict[tuple[str, str], Path]:
        indexed: dict[tuple[str, str], Path] = {}
        duplicates: list[tuple[str, str]] = []
        for path in sorted(data_root.rglob(f"*{suffix}")):
            if is_helper_path(path):
                continue
            key = source_and_case_id(path, suffix)
            if key in indexed:
                duplicates.append(key)
            indexed[key] = path
        if duplicates:
            raise ValueError(
                f"Duplicate Pelvic1K case names found: {sorted(set(duplicates))[:5]}"
            )
        return indexed

    images = index_files(IMAGE_SUFFIX)
    labels = index_files(LABEL_SUFFIX)
    missing_images = sorted(set(labels) - set(images))
    if missing_images:
        raise FileNotFoundError(
            "Pelvic1K labels without matching images: "
            f"{missing_images[:5]}"
        )
    if not images:
        raise FileNotFoundError(
            f"No *{IMAGE_SUFFIX} / *{LABEL_SUFFIX} pairs found under {data_root}"
        )
    paired_keys = sorted(set(images) & set(labels))
    cases = [
        PelvicCase(
            source=source,
            source_case_id=case_id,
            image_path=images[(source, case_id)],
            label_path=labels[(source, case_id)],
        )
        for source, case_id in paired_keys
    ]
    unlabeled_images = [
        images[key] for key in sorted(set(images) - set(labels))
    ]
    return cases, unlabeled_images


def split_cases(
    cases: list[PelvicCase],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[PelvicCase], list[PelvicCase], list[PelvicCase]]:
    by_source: dict[str, list[PelvicCase]] = defaultdict(list)
    for case in cases:
        by_source[case.source].append(case)

    train_cases = []
    val_cases = []
    test_cases = []
    for source in sorted(by_source):
        source_cases = sorted(
            by_source[source], key=lambda case: case.source_case_id
        )
        random.Random(f"{seed}:{source}").shuffle(source_cases)
        test_count = max(1, round(len(source_cases) * test_fraction))
        val_count = max(1, round(len(source_cases) * validation_fraction))
        if test_count + val_count >= len(source_cases):
            raise ValueError(
                f"Not enough {source} cases for train/validation/test split."
            )
        test_cases.extend(source_cases[:test_count])
        val_cases.extend(source_cases[test_count : test_count + val_count])
        train_cases.extend(source_cases[test_count + val_count :])
    return train_cases, val_cases, test_cases


def validate_pair(image_path: Path, label_path: Path) -> tuple[set[int], bool]:
    image = sitk.ReadImage(str(image_path))
    label = sitk.ReadImage(str(label_path))
    if image.GetSize() != label.GetSize():
        raise ValueError(
            "Image/label size mismatch:\n"
            f"image={image_path} size={image.GetSize()}\n"
            f"label={label_path} size={label.GetSize()}"
        )
    needs_label_geometry_rewrite = not same_geometry(image, label)
    labels = set(int(value) for value in np.unique(sitk.GetArrayFromImage(label)))
    unexpected = labels - EXPECTED_LABELS
    if unexpected:
        raise ValueError(
            f"Unexpected labels {sorted(unexpected)} in {label_path}. "
            "Expected CTPelvic1K labels 0-4."
        )
    return labels, needs_label_geometry_rewrite


def write_label_with_image_geometry(label_path: Path, image_path: Path, out_path: Path) -> None:
    image = sitk.ReadImage(str(image_path))
    label = sitk.ReadImage(str(label_path))
    output = sitk.GetImageFromArray(sitk.GetArrayFromImage(label))
    output.CopyInformation(image)
    sitk.WriteImage(output, str(out_path))


def prepare_case(job: tuple) -> dict:
    (
        index,
        split_name,
        case,
        image_out,
        label_out,
        overwrite,
    ) = job
    labels, needs_label_geometry_rewrite = validate_pair(
        case.image_path, case.label_path
    )
    if overwrite or not image_out.exists():
        shutil.copyfile(case.image_path, image_out)
    if overwrite or not label_out.exists():
        if needs_label_geometry_rewrite:
            write_label_with_image_geometry(case.label_path, case.image_path, label_out)
        else:
            shutil.copyfile(case.label_path, label_out)

    return {
        "index": index,
        "labels": sorted(labels),
        "case_id": image_out.name.removesuffix("_0000.nii.gz"),
        "manifest": {
            "case_id": image_out.name.removesuffix("_0000.nii.gz"),
            "split": split_name,
            "source": case.source,
            "source_case_id": case.source_case_id,
            "image": str(case.image_path.resolve()),
            "label": str(case.label_path.resolve()),
            "label_geometry_rewritten": needs_label_geometry_rewrite,
        },
    }


def main() -> None:
    args = parse_args()
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Pelvic1K data root not found: {args.data_root}")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if not 0 < args.test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if args.validation_fraction + args.test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below 1.")

    cases, unlabeled_images = discover_pairs(args.data_root)
    rng = random.Random(args.seed)
    rng.shuffle(cases)
    if args.max_cases is not None:
        if args.max_cases < 2:
            raise ValueError("max_cases must be at least 2.")
        cases = cases[: args.max_cases]

    train_cases, val_cases, test_cases = split_cases(
        cases, args.validation_fraction, args.test_fraction, args.seed
    )

    nnunet_raw, nnunet_preprocessed, _ = nnunet_paths()
    dataset = nnunet_raw / PELVIC1K_DATASET_NAME
    preprocessed_dataset = nnunet_preprocessed / PELVIC1K_DATASET_NAME
    if args.rebuild:
        shutil.rmtree(dataset, ignore_errors=True)
        shutil.rmtree(preprocessed_dataset, ignore_errors=True)
    images_tr = dataset / "imagesTr"
    labels_tr = dataset / "labelsTr"
    images_ts = dataset / "imagesTs"
    labels_ts = dataset / "labelsTs"
    for folder in (images_tr, labels_tr, images_ts, labels_ts):
        folder.mkdir(parents=True, exist_ok=True)

    split_case_ids: dict[str, list[str]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    source_manifest = []
    observed_labels = set()
    split_cases_map = {
        "train": train_cases,
        "validation": val_cases,
        "test": test_cases,
    }
    workers = max(1, args.num_workers)
    for split_name, split_members in split_cases_map.items():
        jobs = []
        for index, case in enumerate(split_members):
            safe_source_id = re.sub(r"[^a-z0-9]+", "_", case.source_case_id)
            case_id = f"pelvic1k_{case.source}_{safe_source_id}"
            if split_name == "test":
                image_out = images_ts / f"{case_id}_0000.nii.gz"
                label_out = labels_ts / f"{case_id}.nii.gz"
            else:
                image_out = images_tr / f"{case_id}_0000.nii.gz"
                label_out = labels_tr / f"{case_id}.nii.gz"
            jobs.append((index, split_name, case, image_out, label_out, args.overwrite))

        results = []
        desc = f"Preparing CTPelvic1K {split_name}"
        if workers == 1:
            iterator = (prepare_case(job) for job in jobs)
            results = list(tqdm(iterator, total=len(jobs), desc=desc, unit="case"))
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(prepare_case, job) for job in jobs]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"{desc} ({workers} workers)",
                    unit="case",
                ):
                    results.append(future.result())

        for result in sorted(results, key=lambda item: item["index"]):
            observed_labels |= set(result["labels"])
            split_case_ids[split_name].append(result["case_id"])
            source_manifest.append(result["manifest"])

    train_ids = split_case_ids["train"]
    val_ids = split_case_ids["validation"]
    test_ids = split_case_ids["test"]
    split = [{"train": train_ids, "val": val_ids}]

    save_json(
        {
            "channel_names": {"0": "CT"},
            "labels": {
                "background": 0,
                "sacrum": 1,
                "right_hip": 2,
                "left_hip": 3,
                "lumbar_vertebra": 4,
            },
            "numTraining": len(train_ids) + len(val_ids),
            "numTest": len(test_ids),
            "file_ending": ".nii.gz",
        },
        dataset / "dataset.json",
    )
    save_json(
        split,
        nnunet_preprocessed / PELVIC1K_DATASET_NAME / "splits_final.json",
    )
    save_json(
        {
            "source_root": str(args.data_root.resolve()),
            "num_cases": len(cases),
            "cases_by_source": dict(
                sorted(Counter(case.source for case in cases).items())
            ),
            "unlabeled_images_ignored": [
                str(path.resolve()) for path in unlabeled_images
            ],
            "observed_labels": sorted(observed_labels),
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "test_fraction": args.test_fraction,
            "train_cases": len(train_ids),
            "validation_cases": len(val_ids),
            "test_cases": len(test_ids),
            "case_manifest": source_manifest,
        },
        dataset / "preparation_summary.json",
    )

    print(f"Created {dataset}")
    print(
        f"Cases: {len(cases)}; train: {len(train_ids)}; "
        f"val: {len(val_ids)}; test: {len(test_ids)}"
    )
    print(
        "Cases by source:",
        dict(sorted(Counter(case.source for case in cases).items())),
    )
    print(f"Unlabeled images ignored: {len(unlabeled_images)}")
    print(f"Observed labels: {sorted(observed_labels)}")


if __name__ == "__main__":
    main()
