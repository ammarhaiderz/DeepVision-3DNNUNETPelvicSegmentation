#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


TCIA_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download TCIA series listed in a .tcia manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/data/home/ue704/TCIA_CT_COLONOGRAPHY_06-22-2015.tcia"),
    )
    parser.add_argument(
        "--series_list",
        type=Path,
        default=None,
        help="Optional text file with one SeriesInstanceUID per line.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/data/home/ue704/Pelvic1k/TCIA_COLONOG/series_zips"),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--redownload_bad_zip",
        action="store_true",
        help="Delete and redownload existing files that are not valid ZIP files.",
    )
    return parser.parse_args()


def read_series_uids(manifest: Path) -> list[str]:
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    series = []
    in_list = False
    for line in manifest.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "ListOfSeriesToDownload=":
            in_list = True
            continue
        if in_list and stripped:
            series.append(stripped)
    if not series:
        raise ValueError(f"No SeriesInstanceUID entries found in {manifest}")
    return series


def read_series_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    series = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not series:
        raise ValueError(f"No SeriesInstanceUID entries found in {path}")
    return series


def is_valid_zip(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def download_series(uid: str, output: Path, timeout: int) -> dict:
    url = f"{TCIA_API}/getImage"
    params = {"SeriesInstanceUID": uid}
    partial = output.with_suffix(output.suffix + ".part")
    partial.unlink(missing_ok=True)
    start = time.time()
    bytes_written = 0
    with requests.get(url, params=params, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with partial.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
                    bytes_written += len(chunk)
    partial.rename(output)
    return {
        "uid": uid,
        "path": str(output),
        "bytes": bytes_written,
        "seconds": round(time.time() - start, 3),
    }


def handle_series(
    offset: int,
    uid: str,
    total: int,
    out_dir: Path,
    timeout: int,
    redownload_bad_zip: bool,
    log_path: Path,
    log_lock: threading.Lock,
) -> str:
    output = out_dir / f"{uid}.zip"
    status_prefix = f"[{offset + 1}/{total}]"
    if output.exists():
        if is_valid_zip(output):
            return f"{status_prefix} skip valid {uid}"
        if redownload_bad_zip:
            output.unlink()
        else:
            record = {
                "uid": uid,
                "index": offset,
                "error": f"bad zip exists: {output}",
            }
            with log_lock:
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(json.dumps(record) + "\n")
            return f"{status_prefix} bad zip exists {output}"

    try:
        record = download_series(uid, output, timeout)
        record["index"] = offset
        record["valid_zip"] = is_valid_zip(output)
        message = (
            f"{status_prefix} wrote {record['bytes'] / (1024 ** 2):.1f} MiB "
            f"in {record['seconds']:.1f}s {uid}"
        )
    except Exception as error:
        record = {
            "uid": uid,
            "index": offset,
            "error": repr(error),
        }
        message = f"{status_prefix} ERROR {uid}: {error}"

    with log_lock:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")
    time.sleep(0.01)
    return message


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be at least 1.")
    series = read_series_list(args.series_list) if args.series_list else read_series_uids(args.manifest)
    selected = series[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir.parent / "download_log.jsonl"
    (args.out_dir.parent / "series_uids.txt").write_text(
        "\n".join(series) + "\n", encoding="utf-8"
    )

    print(f"Manifest: {args.manifest}")
    print(f"Total series in manifest: {len(series)}")
    print(f"Downloading {len(selected)} series starting at index {args.start}")
    print(f"Workers: {args.workers}")
    print(f"Output: {args.out_dir}")
    print(f"Log: {log_path}")

    log_lock = threading.Lock()
    jobs = [
        (offset, uid)
        for offset, uid in enumerate(selected, start=args.start)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                handle_series,
                offset,
                uid,
                len(series),
                args.out_dir,
                args.timeout,
                args.redownload_bad_zip,
                log_path,
                log_lock,
            )
            for offset, uid in jobs
        ]
        for future in as_completed(futures):
            print(future.result(), flush=True)
            time.sleep(args.sleep)
    print("Finished. Rerun the same command to skip valid ZIPs and retry failures.")


if __name__ == "__main__":
    main()
