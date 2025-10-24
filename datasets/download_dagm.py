"""Download and unpack the DAGM2007 dataset.

This script downloads the official DAGM2007 archive (or a user provided URL)
into a local directory whose structure is immediately compatible with
``prepare_dagm.py``.

Example:
    python datasets/download_dagm.py --out data_raw/dagm

If the output directory already exists, pass ``--force`` to remove it first.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable
import zipfile
import tarfile

DEFAULT_URL = "https://download.visinf.tu-darmstadt.de/data/dagm2007/DAGM2007.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and unpack DAGM2007")
    parser.add_argument("--out", type=Path, required=True, help="Destination directory")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help="Archive URL. Defaults to the official DAGM2007 mirror.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory if it already exists.",
    )
    return parser.parse_args()


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dst, "wb") as f:
        total = resp.getheader("Content-Length")
        total_size = int(total) if total is not None else None
        downloaded = 0
        chunk_size = 8192
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = downloaded * 100 // total_size
                sys.stdout.write(f"\rDownloading... {percent}%")
                sys.stdout.flush()
        sys.stdout.write("\rDownload complete.\n")
        sys.stdout.flush()


def extract_archive(archive_path: Path, dst: Path) -> None:
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dst)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            tf.extractall(dst)
    else:
        raise RuntimeError(f"Unsupported archive format: {archive_path}")


def find_class_root(root: Path) -> Path:
    """Locate the directory that directly contains the Class* folders."""
    candidates: Iterable[Path] = [root] + list(root.rglob("Class*"))
    for cand in candidates:
        base = cand if cand.is_dir() else cand.parent
        classes = [p for p in base.iterdir() if p.is_dir() and p.name.lower().startswith("class")]
        if len(classes) >= 6:
            return base
    return root


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out
    if out_dir.exists():
        if args.force:
            shutil.rmtree(out_dir)
        else:
            raise SystemExit(f"Output directory {out_dir} already exists. Use --force to overwrite.")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        archive_path = tmp_path / "dagm_archive"
        print(f"Downloading DAGM2007 from {args.url} ...")
        download(args.url, archive_path)
        print("Extracting archive ...")
        extract_archive(archive_path, tmp_path)
        class_root = find_class_root(tmp_path)
        if class_root == tmp_path:
            print("Warning: could not find explicit Class* folders. Using extracted root.")
        shutil.copytree(class_root, out_dir)
    print(f"DAGM2007 extracted to: {out_dir.resolve()}")
    print("You can now run prepare_dagm.py with --root pointing to this directory.")


if __name__ == "__main__":
    main()
