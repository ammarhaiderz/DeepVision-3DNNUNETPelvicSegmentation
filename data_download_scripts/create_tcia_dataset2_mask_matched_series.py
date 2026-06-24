#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import tarfile
from pathlib import Path

import requests


TCIA_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
MASK_PATTERN = re.compile(
    r"dataset2_(1\.3\.6\.1\.4\.1\.9328\.50\.4\.\d+)_(\d+)_"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a TCIA SeriesInstanceUID list for only the CT COLONOGRAPHY "
            "series that have CTPelvic1K dataset2 masks."
        )
    )
    parser.add_argument(
        "--mask_archive",
        type=Path,
        default=Path("/data/home/ue704/Pelvic1k/CTPelvic1K_dataset2_mask_mappingback.tar.gz"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/data/home/ue704/Pelvic1k/TCIA_COLONOG"),
    )
    parser.add_argument(
        "--include_ambiguous",
        action="store_true",
        help="Include all candidate TCIA series for masks with ambiguous matches.",
    )
    return parser.parse_args()


def masked_patient_series(mask_archive: Path) -> list[tuple[str, int, str]]:
    entries = []
    with tarfile.open(mask_archive) as archive:
        for member in archive.getmembers():
            match = MASK_PATTERN.search(Path(member.name).name)
            if match:
                entries.append(
                    (match.group(1), int(match.group(2)), Path(member.name).name)
                )
    if not entries:
        raise ValueError(f"No dataset2 mask filenames found in {mask_archive}")
    return entries


def tcia_series_metadata() -> list[dict]:
    response = requests.get(
        f"{TCIA_API}/getSeries",
        params={"Collection": "CT COLONOGRAPHY", "Modality": "CT"},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    masks = masked_patient_series(args.mask_archive)
    series = tcia_series_metadata()

    by_patient_series: dict[tuple[str, int], list[dict]] = {}
    for item in series:
        key = (
            item.get("PatientID"),
            int(item.get("SeriesNumber", -999999)),
        )
        by_patient_series.setdefault(key, []).append(item)

    rows = []
    selected_uids = []
    missing = []
    ambiguous = []
    for patient_id, series_number, mask_name in masks:
        matches = by_patient_series.get((patient_id, series_number), [])
        if not matches:
            missing.append(mask_name)
            rows.append(
                {
                    "mask": mask_name,
                    "patient_id": patient_id,
                    "series_number": series_number,
                    "series_instance_uid": "",
                    "match_status": "missing",
                    "series_description": "",
                    "image_count": "",
                }
            )
            continue
        if len(matches) > 1:
            ambiguous.append(mask_name)
            if not args.include_ambiguous:
                rows.append(
                    {
                        "mask": mask_name,
                        "patient_id": patient_id,
                        "series_number": series_number,
                        "series_instance_uid": "",
                        "match_status": "ambiguous_excluded",
                        "series_description": "",
                        "image_count": "",
                    }
                )
                continue
        for match in matches:
            uid = match["SeriesInstanceUID"]
            selected_uids.append(uid)
            rows.append(
                {
                    "mask": mask_name,
                    "patient_id": patient_id,
                    "series_number": series_number,
                    "series_instance_uid": uid,
                    "match_status": "ambiguous" if len(matches) > 1 else "unique",
                    "series_description": match.get("SeriesDescription", ""),
                    "image_count": match.get("ImageCount", ""),
                }
            )

    unique_uids = sorted(set(selected_uids))
    series_path = args.out_dir / "dataset2_mask_matched_series_uids.txt"
    mapping_path = args.out_dir / "dataset2_mask_to_tcia_series_mapping.csv"
    series_path.write_text("\n".join(unique_uids) + "\n", encoding="utf-8")
    with mapping_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Masks: {len(masks)}")
    print(f"Matched TCIA series: {len(unique_uids)}")
    print(f"Missing masks: {len(missing)}")
    if args.include_ambiguous:
        print(f"Ambiguous masks included with all candidates: {len(ambiguous)}")
    else:
        print(f"Ambiguous masks excluded: {len(ambiguous)}")
    print(f"Series UID list: {series_path}")
    print(f"Mapping CSV: {mapping_path}")


if __name__ == "__main__":
    main()
