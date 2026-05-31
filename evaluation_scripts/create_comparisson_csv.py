#!/usr/bin/env python3

import csv
from pathlib import Path


OUTPUT_CSV = "comparison_of_averages.csv"


def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "nnUNet").exists():
            return parent
    raise FileNotFoundError(
        "Could not find project root. Expected a parent folder containing 'nnUNet'."
    )


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)


def find_csv_files(folder: Path):
    return sorted([f for f in folder.glob("*.csv") if f.name != OUTPUT_CSV])


def read_last_average_row(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return None, None

    header = rows[0]
    data_rows = rows[1:]

    last_average_row = None
    for row in data_rows:
        if len(row) > 0 and row[0].strip().upper() == "AVERAGE":
            last_average_row = row

    return header, last_average_row


def main():
    evaluation_dir = PROJECT_ROOT / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    csv_files = find_csv_files(evaluation_dir)

    if not csv_files:
        print(f"No CSV files found in {evaluation_dir}")
        return

    output_rows = []
    output_header = None

    for csv_file in csv_files:
        header, avg_row = read_last_average_row(csv_file)

        if header is None:
            print(f"Skipping empty file: {csv_file.name}")
            continue

        if avg_row is None:
            print(f"Skipping (no AVERAGE row): {csv_file.name}")
            continue

        model_name = csv_file.stem
        new_row = [model_name] + avg_row[1:]

        if output_header is None:
            output_header = ["name"] + header[1:]

        output_rows.append(new_row)

    if not output_rows:
        print("No valid AVERAGE rows found.")
        return

    output_path = evaluation_dir / OUTPUT_CSV

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(output_header)
        writer.writerows(output_rows)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()