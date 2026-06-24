#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm


IMAGE_SUFFIX = "_data.nii.gz"
LABEL_SUFFIX = "_mask_4label.nii.gz"
EXPECTED_LABELS = {0, 1, 2, 3, 4}


@dataclass(frozen=True)
class Dataset2Row:
    mask: str
    series_instance_uid: str
    patient_id: str
    series_number: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standardize the expanded CTPelvic1K sources into one folder of "
            "*_data.nii.gz / *_mask_4label.nii.gz pairs for nnU-Net pretraining."
        )
    )
    parser.add_argument(
        "--pelvic1k_root",
        type=Path,
        default=Path("/data/home/ue704/Pelvic1k"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/data/home/ue704/Pelvic1k/expanded_pretrain_standardized"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["2", "3", "4", "6", "7"],
        choices=["2", "3", "4", "6", "7"],
        help="CTPelvic1K source datasets to standardize.",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Use symlinks for existing NIfTI files when possible, or copy data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing standardized output files.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Read every completed pair and check image/label geometry and labels.",
    )
    parser.add_argument(
        "--max_dataset2",
        type=int,
        default=None,
        help="Optional small dataset2 subset for a conversion smoke test.",
    )
    parser.add_argument(
        "--dataset2_workers",
        type=int,
        default=1,
        help="Parallel workers for TCIA dataset2 DICOM zip to NIfTI conversion.",
    )
    return parser.parse_args()


def safe_case_id(name: str) -> str:
    name = name.removesuffix(IMAGE_SUFFIX).removesuffix(LABEL_SUFFIX)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def replace_existing(path: Path, overwrite: bool) -> None:
    if overwrite and (path.exists() or path.is_symlink()):
        path.unlink()


def link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> bool:
    replace_existing(dst, overwrite)
    if dst.exists() or dst.is_symlink():
        return False
    ensure_parent(dst)
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)
    return True


def copy_stream(src_file, dst: Path, overwrite: bool) -> bool:
    replace_existing(dst, overwrite)
    if dst.exists():
        return False
    ensure_parent(dst)
    with dst.open("wb") as out_file:
        shutil.copyfileobj(src_file, out_file)
    return True


def tar_members_by_basename(archive_path: Path, suffix: str) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name.startswith("._") or not name.endswith(suffix):
                continue
            if name in members:
                raise ValueError(f"Duplicate member {name} in {archive_path}")
            members[name] = member
    return members


def extract_tar_member(
    archive_path: Path,
    member: tarfile.TarInfo,
    dst: Path,
    overwrite: bool,
) -> bool:
    replace_existing(dst, overwrite)
    if dst.exists():
        return False
    ensure_parent(dst)
    with tarfile.open(archive_path) as archive:
        src = archive.extractfile(member)
        if src is None:
            raise ValueError(f"Could not read {member.name} from {archive_path}")
        with src:
            return copy_stream(src, dst, overwrite=False)


def write_image(image: sitk.Image, dst: Path, overwrite: bool) -> bool:
    replace_existing(dst, overwrite)
    if dst.exists():
        return False
    ensure_parent(dst)
    sitk.WriteImage(image, str(dst))
    return True


def validate_pair(image_path: Path, label_path: Path) -> list[int]:
    image = sitk.ReadImage(str(image_path))
    label = sitk.ReadImage(str(label_path))
    if image.GetSize() != label.GetSize():
        raise ValueError(
            f"Size mismatch for {image_path.name}: "
            f"image={image.GetSize()} label={label.GetSize()}"
        )
    if any(abs(a - b) > 1e-4 for a, b in zip(image.GetSpacing(), label.GetSpacing())):
        raise ValueError(
            f"Spacing mismatch for {image_path.name}: "
            f"image={image.GetSpacing()} label={label.GetSpacing()}"
        )
    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(label, label)
    labels = sorted(int(value) for value in stats.GetLabels())
    unexpected = set(labels) - EXPECTED_LABELS
    if unexpected:
        raise ValueError(f"Unexpected labels {sorted(unexpected)} in {label_path}")
    return labels


def standardize_existing_pairs(
    image_paths: list[Path],
    label_paths: list[Path],
    out_dir: Path,
    mode: str,
    overwrite: bool,
    validate: bool,
    source_name: str,
) -> dict:
    labels_by_id = {safe_case_id(path.name): path for path in label_paths}
    images_by_id = {safe_case_id(path.name): path for path in image_paths}
    common_ids = sorted(set(images_by_id) & set(labels_by_id))
    missing_images = sorted(set(labels_by_id) - set(images_by_id))
    missing_labels = sorted(set(images_by_id) - set(labels_by_id))
    written = 0
    reused = 0
    for case_id in tqdm(common_ids, desc=f"Standardizing {source_name}", unit="case"):
        image_dst = out_dir / f"{case_id}{IMAGE_SUFFIX}"
        label_dst = out_dir / f"{case_id}{LABEL_SUFFIX}"
        image_new = link_or_copy(images_by_id[case_id], image_dst, mode, overwrite)
        label_new = link_or_copy(labels_by_id[case_id], label_dst, mode, overwrite)
        if validate:
            validate_pair(image_dst, label_dst)
        if image_new or label_new:
            written += 1
        else:
            reused += 1
    return {
        "source": source_name,
        "cases": len(common_ids),
        "written_or_replaced_cases": written,
        "reused_cases": reused,
        "missing_images": missing_images,
        "missing_labels": missing_labels,
    }


def standardize_dataset6(root: Path, out_dir: Path, mode: str, overwrite: bool, validate: bool) -> dict:
    images = sorted((root / "extracted" / "CTPelvic1K_dataset6_data").rglob(f"*{IMAGE_SUFFIX}"))
    labels = sorted((root / "extracted" / "ipcai2021_dataset6_Anonymized").rglob(f"*{LABEL_SUFFIX}"))
    return standardize_existing_pairs(images, labels, out_dir, mode, overwrite, validate, "dataset6")


def standardize_dataset7(root: Path, out_dir: Path, mode: str, overwrite: bool, validate: bool) -> dict:
    image_root = root / "extracted" / "CTPelvic1K_dataset7_data"
    label_root = root / "extracted" / "dataset7_masks"
    if not label_root.exists():
        label_root = root / "extracted"
    image_paths = sorted(image_root.rglob(f"*{IMAGE_SUFFIX}"))
    raw_labels = sorted(label_root.rglob(f"*{LABEL_SUFFIX}"))
    label_paths: list[Path] = []
    temp_dir = out_dir / "_dataset7_label_aliases"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for label in raw_labels:
        name = label.name
        if name.startswith("dataset7_"):
            label_paths.append(label)
            continue
        alias = temp_dir / f"dataset7_{name}"
        link_or_copy(label, alias, mode, overwrite)
        label_paths.append(alias)
    return standardize_existing_pairs(image_paths, label_paths, out_dir, mode, overwrite, validate, "dataset7")


def task10_image_members(task10_tar: Path) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(task10_tar) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            path = Path(member.name)
            name = path.name
            if name.startswith("._") or not name.endswith(".nii.gz"):
                continue
            if "imagesTr" not in path.parts and "imagesTs" not in path.parts:
                continue
            case_id = name.removesuffix(".nii.gz")
            members[case_id] = member
    return members


def standardize_dataset3(root: Path, out_dir: Path, overwrite: bool, validate: bool) -> dict:
    task10_tar = root / "Task10_Colon.tar"
    mask_tar = root / "CTPelvic1K_dataset3_mask_mappingback.tar.gz"
    image_members = task10_image_members(task10_tar)
    mask_members = tar_members_by_basename(mask_tar, LABEL_SUFFIX)
    written = 0
    reused = 0
    missing_images = []
    with tarfile.open(mask_tar) as mask_archive, tarfile.open(task10_tar) as image_archive:
        for mask_name, mask_member in tqdm(sorted(mask_members.items()), desc="Standardizing dataset3", unit="case"):
            case_core = mask_name.removeprefix("dataset3_").removesuffix(LABEL_SUFFIX)
            image_member = image_members.get(case_core)
            if image_member is None:
                missing_images.append(mask_name)
                continue
            case_id = f"dataset3_{case_core}"
            image_dst = out_dir / f"{case_id}{IMAGE_SUFFIX}"
            label_dst = out_dir / f"{case_id}{LABEL_SUFFIX}"
            image_new = False
            label_new = False
            replace_existing(image_dst, overwrite)
            if not image_dst.exists():
                src = image_archive.extractfile(image_member)
                if src is None:
                    raise ValueError(f"Could not read {image_member.name} from {task10_tar}")
                with src:
                    image_new = copy_stream(src, image_dst, overwrite=False)
            replace_existing(label_dst, overwrite)
            if not label_dst.exists():
                src = mask_archive.extractfile(mask_member)
                if src is None:
                    raise ValueError(f"Could not read {mask_member.name} from {mask_tar}")
                with src:
                    label_new = copy_stream(src, label_dst, overwrite=False)
            if validate:
                validate_pair(image_dst, label_dst)
            if image_new or label_new:
                written += 1
            else:
                reused += 1
    return {
        "source": "dataset3",
        "cases": len(mask_members) - len(missing_images),
        "written_or_replaced_cases": written,
        "reused_cases": reused,
        "missing_images": missing_images,
        "missing_labels": [],
    }


def standardize_dataset4(root: Path, out_dir: Path, overwrite: bool, validate: bool) -> dict:
    mask_tar = root / "CTPelvic1K_dataset4_mask_mappingback.tar.gz"
    mask_members = tar_members_by_basename(mask_tar, LABEL_SUFFIX)
    written = 0
    reused = 0
    missing_images = []
    with tarfile.open(mask_tar) as mask_archive:
        for mask_name, mask_member in tqdm(sorted(mask_members.items()), desc="Standardizing dataset4", unit="case"):
            case_core = mask_name.removeprefix("dataset4_").removesuffix(LABEL_SUFFIX)
            image_src = root / "kits19" / "data" / case_core / "imaging.nii.gz"
            if not image_src.exists():
                missing_images.append(mask_name)
                continue
            case_id = f"dataset4_{case_core}"
            image_dst = out_dir / f"{case_id}{IMAGE_SUFFIX}"
            label_dst = out_dir / f"{case_id}{LABEL_SUFFIX}"
            image_new = link_or_copy(image_src, image_dst, "symlink", overwrite)
            replace_existing(label_dst, overwrite)
            label_new = False
            if not label_dst.exists():
                src = mask_archive.extractfile(mask_member)
                if src is None:
                    raise ValueError(f"Could not read {mask_member.name} from {mask_tar}")
                with src:
                    label_new = copy_stream(src, label_dst, overwrite=False)
            if validate:
                validate_pair(image_dst, label_dst)
            if image_new or label_new:
                written += 1
            else:
                reused += 1
    return {
        "source": "dataset4",
        "cases": len(mask_members) - len(missing_images),
        "written_or_replaced_cases": written,
        "reused_cases": reused,
        "missing_images": missing_images,
        "missing_labels": [],
    }


def dataset2_rows(mapping_csv: Path, max_rows: int | None) -> list[Dataset2Row]:
    rows = []
    with mapping_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            uid = row["series_instance_uid"].strip()
            if not uid:
                continue
            if row["match_status"] not in {"unique", "ambiguous"}:
                continue
            rows.append(
                Dataset2Row(
                    mask=row["mask"],
                    series_instance_uid=uid,
                    patient_id=row["patient_id"],
                    series_number=row["series_number"],
                )
            )
    rows = sorted(rows, key=lambda item: item.series_instance_uid)
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def read_dicom_zip(zip_path: Path, temp_root: Path) -> sitk.Image:
    extract_dir = temp_root / zip_path.stem
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(extract_dir))
    if not series_ids:
        raise ValueError(f"No DICOM series found in {zip_path}")
    if len(series_ids) > 1:
        series_ids = sorted(series_ids)
    files = reader.GetGDCMSeriesFileNames(str(extract_dir), series_ids[0])
    reader.SetFileNames(files)
    return reader.Execute()


def standardize_dataset2(
    root: Path,
    out_dir: Path,
    overwrite: bool,
    validate: bool,
    max_rows: int | None,
    workers: int,
) -> dict:
    mapping_csv = root / "TCIA_COLONOG" / "dataset2_mask_to_tcia_series_mapping.csv"
    series_zip_dir = root / "TCIA_COLONOG" / "series_zips"
    mask_tar = root / "CTPelvic1K_dataset2_mask_mappingback.tar.gz"
    mask_members = {
        name: member.name
        for name, member in tar_members_by_basename(mask_tar, LABEL_SUFFIX).items()
    }
    rows = dataset2_rows(mapping_csv, max_rows)
    written = 0
    reused = 0
    missing_zips = []
    missing_masks = []
    failed = []
    jobs = []
    for row in rows:
        zip_path = series_zip_dir / f"{row.series_instance_uid}.zip"
        if not zip_path.exists():
            missing_zips.append(row.series_instance_uid)
            continue
        mask_member_name = mask_members.get(row.mask)
        if mask_member_name is None:
            missing_masks.append(row.mask)
            continue
        jobs.append(
            (
                row,
                zip_path,
                mask_tar,
                mask_member_name,
                out_dir,
                overwrite,
                validate,
            )
        )

    workers = max(1, workers)
    if workers == 1:
        iterator = (
            standardize_dataset2_case(job)
            for job in jobs
        )
        for result in tqdm(iterator, total=len(jobs), desc="Standardizing dataset2", unit="case"):
            if result["status"] == "written":
                written += 1
            elif result["status"] == "reused":
                reused += 1
            else:
                failed.append(result)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(standardize_dataset2_case, job) for job in jobs]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Standardizing dataset2 ({workers} workers)",
                unit="case",
            ):
                result = future.result()
                if result["status"] == "written":
                    written += 1
                elif result["status"] == "reused":
                    reused += 1
                else:
                    failed.append(result)
    return {
        "source": "dataset2",
        "expected_mapped_cases": len(rows),
        "cases": len(rows) - len(missing_zips) - len(missing_masks) - len(failed),
        "written_or_replaced_cases": written,
        "reused_cases": reused,
        "missing_zips": missing_zips,
        "missing_masks": missing_masks,
        "failed": failed,
    }


def standardize_dataset2_case(job: tuple) -> dict:
    row, zip_path, mask_tar, mask_member_name, out_dir, overwrite, validate = job
    case_id = safe_case_id(row.mask)
    image_dst = out_dir / f"{case_id}{IMAGE_SUFFIX}"
    label_dst = out_dir / f"{case_id}{LABEL_SUFFIX}"
    image_new = False
    label_new = False
    try:
        replace_existing(image_dst, overwrite)
        if not image_dst.exists():
            with tempfile.TemporaryDirectory(prefix=f"ctpelvic1k_{case_id}_") as tmp:
                image = read_dicom_zip(zip_path, Path(tmp))
                image_new = write_image(image, image_dst, overwrite=False)

        replace_existing(label_dst, overwrite)
        if not label_dst.exists():
            with tarfile.open(mask_tar) as mask_archive:
                member = mask_archive.getmember(mask_member_name)
                src = mask_archive.extractfile(member)
                if src is None:
                    raise ValueError(f"Could not read {mask_member_name} from {mask_tar}")
                with src:
                    label_new = copy_stream(src, label_dst, overwrite=False)

        if validate:
            validate_pair(image_dst, label_dst)
        status = "written" if image_new or label_new else "reused"
        return {
            "status": status,
            "series_instance_uid": row.series_instance_uid,
            "mask": row.mask,
            "case_id": case_id,
        }
    except Exception as exc:
        replace_existing(image_dst, True)
        replace_existing(label_dst, True)
        return {
            "status": "failed",
            "series_instance_uid": row.series_instance_uid,
            "mask": row.mask,
            "case_id": case_id,
            "error": str(exc),
        }


def count_standardized_pairs(out_dir: Path) -> dict:
    images = {safe_case_id(path.name): path for path in out_dir.glob(f"*{IMAGE_SUFFIX}")}
    labels = {safe_case_id(path.name): path for path in out_dir.glob(f"*{LABEL_SUFFIX}")}
    paired = sorted(set(images) & set(labels))
    by_source: dict[str, int] = {}
    for case_id in paired:
        source = case_id.split("_", 1)[0]
        by_source[source] = by_source.get(source, 0) + 1
    return {
        "total_pairs": len(paired),
        "images_without_labels": len(set(images) - set(labels)),
        "labels_without_images": len(set(labels) - set(images)),
        "pairs_by_source": dict(sorted(by_source.items())),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    if "6" in args.datasets:
        summaries.append(standardize_dataset6(args.pelvic1k_root, args.out_dir, args.mode, args.overwrite, args.validate))
    if "7" in args.datasets:
        summaries.append(standardize_dataset7(args.pelvic1k_root, args.out_dir, args.mode, args.overwrite, args.validate))
    if "3" in args.datasets:
        summaries.append(standardize_dataset3(args.pelvic1k_root, args.out_dir, args.overwrite, args.validate))
    if "4" in args.datasets:
        summaries.append(standardize_dataset4(args.pelvic1k_root, args.out_dir, args.overwrite, args.validate))
    if "2" in args.datasets:
        summaries.append(
            standardize_dataset2(
                args.pelvic1k_root,
                args.out_dir,
                args.overwrite,
                args.validate,
                args.max_dataset2,
                args.dataset2_workers,
            )
        )

    summary = {
        "pelvic1k_root": str(args.pelvic1k_root.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "datasets_requested": args.datasets,
        "mode": args.mode,
        "validated": args.validate,
        "sources": summaries,
        "standardized": count_standardized_pairs(args.out_dir),
    }
    summary_path = args.out_dir / "standardization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Standardized folder: {args.out_dir}")
    print(f"Total paired cases: {summary['standardized']['total_pairs']}")
    print(f"Pairs by source: {summary['standardized']['pairs_by_source']}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
