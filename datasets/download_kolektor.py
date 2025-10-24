"""Download and unpack the KolektorSDD / KolektorSDD2 datasets.

The script downloads the official Kolektor archives (or a user supplied URL)
and arranges them into the ``images/`` and ``gt/`` folders expected by
``prepare_kolektor.py``.

Example (download KolektorSDD):
    python datasets/download_kolektor.py --variant sdd --out data_raw/kolektor

To download KolektorSDD2 instead:
    python datasets/download_kolektor.py --variant sdd2 --out data_raw/kolektor2

If the output directory already exists, pass ``--force`` to remove it first.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, Optional
import zipfile
import tarfile

DEFAULT_URLS = {
    "sdd": "https://www.vicos.si/resources/kolektorsdd/KolektorSDD.zip",
    "sdd2": "https://www.vicos.si/resources/kolektorsdd/KolektorSDD2.zip",
}
MASK_KEYWORDS = ["mask", "label", "ground", "gt", "defect"]
MASK_SUFFIXES = ["_label", "-label", "_mask", "-mask", "_gt", "-gt", "_labels", "-labels", "_defect"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download KolektorSDD/SDD2")
    parser.add_argument("--out", type=Path, required=True, help="Destination directory")
    parser.add_argument(
        "--variant",
        choices=sorted(DEFAULT_URLS.keys()),
        default="sdd",
        help="Which dataset variant to download (default: sdd).",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Override download URL (defaults to the official mirror for the variant).",
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


def is_mask_path(path: Path) -> bool:
    lower_path = str(path).lower()
    if any(keyword in lower_path for keyword in MASK_KEYWORDS):
        return True
    return False


def sanitize_stem(stem: str) -> str:
    lower = stem.lower()
    for suf in MASK_SUFFIXES:
        if lower.endswith(suf):
            return stem[: -len(suf)]
    return stem


def collect_files(root: Path) -> Dict[str, Dict[str, Optional[Path]]]:
    pairs: Dict[str, Dict[str, Optional[Path]]] = {}
    for file in root.rglob("*"):
        if not file.is_file():
            continue
        if file.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = file.stem
        cleaned = sanitize_stem(stem)
        entry = pairs.setdefault(cleaned, {"image": None, "mask": None})
        if is_mask_path(file):
            if entry["mask"] is None:
                entry["mask"] = file
        else:
            if entry["image"] is None:
                entry["image"] = file
    return pairs


def normalise_structure(src_root: Path, dst_root: Path) -> None:
    pairs = collect_files(src_root)
    images_dir = dst_root / "images"
    masks_dir = dst_root / "gt"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing_masks = 0
    for key, data in pairs.items():
        img = data["image"]
        mask = data["mask"]
        if img is None:
            continue
        dst_img = images_dir / img.name
        shutil.copy2(img, dst_img)
        if mask is not None:
            dst_mask = masks_dir / (dst_img.stem + ".png")
            shutil.copy2(mask, dst_mask)
        else:
            missing_masks += 1
        copied += 1
    if copied == 0:
        raise RuntimeError("No images were discovered in the extracted archive.")
    print(f"Copied {copied} images. {missing_masks} images did not have an associated mask.")


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out
    if out_dir.exists():
        if args.force:
            shutil.rmtree(out_dir)
        else:
            raise SystemExit(f"Output directory {out_dir} already exists. Use --force to overwrite.")
    url = args.url or DEFAULT_URLS[args.variant]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        archive_path = tmp_path / "kolektor_archive"
        print(f"Downloading Kolektor {args.variant.upper()} from {url} ...")
        download(url, archive_path)
        print("Extracting archive ...")
        extract_archive(archive_path, tmp_path)
        print("Normalising structure ...")
        normalise_structure(tmp_path, out_dir)
    print(f"Kolektor {args.variant.upper()} prepared at: {out_dir.resolve()}")
    print("You can now run prepare_kolektor.py with --root pointing to this directory.")


if __name__ == "__main__":
    main()
