"""Utility script to download the DAGM2007 and KolektorSDD datasets.

The script downloads the requested dataset(s), extracts the archives and
normalises their folder layout so that the existing ``prepare_*.py`` helpers can
be run without any manual tweaks.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile


CHUNK_SIZE = 1 << 20  # 1 MiB
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class DatasetConfig:
    key: str
    display_name: str
    urls: List[str]
    target_subdir: str
    post_process: Callable[[Path, Path], None]


def download_with_progress(url: str, dst: Path) -> None:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp, open(dst, "wb") as f:  # nosec: B310 - trusted URLs
        total = resp.length or 0
        downloaded = 0
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded * 100 // total
                print(f"\r  -> {percent:3d}% ({downloaded / (1<<20):.1f} / {total / (1<<20):.1f} MiB)", end="", flush=True)
            else:
                print(f"\r  -> {downloaded / (1<<20):.1f} MiB", end="", flush=True)
    print()


def safe_extract_zip(zip_path: Path, dst: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            extracted = (dst / member.filename).resolve()
            if not str(extracted).startswith(str(dst.resolve())):
                raise RuntimeError(f"Unsafe path detected in archive: {member.filename}")
        zf.extractall(dst)


def find_class_root(extracted_dir: Path) -> Path:
    candidates: List[Path] = []
    for path in [extracted_dir] + [p for p in extracted_dir.rglob("*") if p.is_dir()]:
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        class_dirs = [d for d in subdirs if d.name.lower().startswith("class")]
        if class_dirs:
            candidates.append((path, len(class_dirs)))
    if not candidates:
        raise RuntimeError("Could not locate DAGM class directories in the downloaded archive")
    # Prefer the folder that exposes the largest number of class directories;
    # break ties by choosing the deepest (most specific) path.
    return max(candidates, key=lambda item: (item[1], len(item[0].parts)))[0]


def post_process_dagm(extracted_dir: Path, target_root: Path) -> None:
    class_root = find_class_root(extracted_dir)
    target_root.mkdir(parents=True, exist_ok=True)
    for item in target_root.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    for cdir in sorted(class_root.iterdir()):
        if not cdir.is_dir() or not cdir.name.lower().startswith("class"):
            continue
        dst = target_root / cdir.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(cdir, dst)
    print(f"  -> Extracted class folders under {target_root}")


def is_mask_path(path: Path) -> bool:
    lowered = "/".join(part.lower() for part in path.parts)
    mask_tokens = ["mask", "ground_truth", "ground-truth", "gt", "defect", "annotation", "label"]
    return any(token in lowered for token in mask_tokens)


def collect_image_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def post_process_kolektor(extracted_dir: Path, target_root: Path) -> None:
    images_dir = target_root / "images"
    masks_dir = target_root / "gt"
    shutil.rmtree(target_root, ignore_errors=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    files = collect_image_files(extracted_dir)
    mask_map: Dict[str, List[Path]] = {}
    for path in files:
        if is_mask_path(path):
            mask_map.setdefault(path.stem, []).append(path)
    used_names: Dict[str, int] = {}
    copied = 0
    for path in files:
        if is_mask_path(path):
            continue
        stem = path.stem
        used_names[stem] = used_names.get(stem, 0) + 1
        unique_stem = stem if used_names[stem] == 1 else f"{stem}_{used_names[stem]}"
        dst_img = images_dir / f"{unique_stem}{path.suffix.lower()}"
        shutil.copy2(path, dst_img)
        mask_choice: Optional[Path] = None
        for cand in mask_map.get(stem, []):
            mask_choice = cand
            break
        if mask_choice is not None:
            dst_mask = masks_dir / f"{unique_stem}{mask_choice.suffix.lower()}"
            shutil.copy2(mask_choice, dst_mask)
        copied += 1
    if not copied:
        raise RuntimeError("No images were found while processing KolektorSDD")
    print(f"  -> Normalised {copied} images into {target_root}")


DATASETS: Dict[str, DatasetConfig] = {
    "dagm2007": DatasetConfig(
        key="dagm2007",
        display_name="DAGM2007",
        urls=[
            "https://huggingface.co/datasets/eising/dagm2007/resolve/main/DAGM2007.zip",
            "https://huggingface.co/datasets/zhangliangza/Surface-Defect-Detection/resolve/main/DAGM2007.zip",
        ],
        target_subdir="dagm2007",
        post_process=post_process_dagm,
    ),
    "kolektorsdd": DatasetConfig(
        key="kolektorsdd",
        display_name="KolektorSDD",
        urls=[
            "https://box.vicos.si/vicoslab/kolektorsdd.zip",
            "https://huggingface.co/datasets/zhangliangza/Surface-Defect-Detection/resolve/main/KolektorSDD.zip",
        ],
        target_subdir="kolektorsdd",
        post_process=post_process_kolektor,
    ),
    "kolektorsdd2": DatasetConfig(
        key="kolektorsdd2",
        display_name="KolektorSDD2",
        urls=[
            "https://box.vicos.si/vicoslab/kolektorsdd2.zip",
            "https://huggingface.co/datasets/zhangliangza/Surface-Defect-Detection/resolve/main/KolektorSDD2.zip",
        ],
        target_subdir="kolektorsdd2",
        post_process=post_process_kolektor,
    ),
}


def run_download(cfg: DatasetConfig, output_root: Path) -> Path:
    target_root = output_root / cfg.target_subdir
    if target_root.exists() and any(target_root.iterdir()):
        print(f"[skip] {cfg.display_name} already present at {target_root}")
        return target_root

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        archive_path = tmp_dir_path / "archive.zip"
        last_error: Optional[Exception] = None
        for url in cfg.urls:
            print(f"[download] {cfg.display_name} from {url}")
            try:
                download_with_progress(url, archive_path)
                if archive_path.stat().st_size == 0:
                    raise RuntimeError("Downloaded archive is empty")
                break
            except (HTTPError, URLError, RuntimeError) as exc:
                last_error = exc
                print(f"  -> Failed: {exc}")
        else:
            raise RuntimeError(f"Failed to download {cfg.display_name}: {last_error}")

        extracted_dir = tmp_dir_path / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        safe_extract_zip(archive_path, extracted_dir)
        cfg.post_process(extracted_dir, target_root)

    print(f"[done] {cfg.display_name} available at {target_root}")
    return target_root


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download DAGM2007 / KolektorSDD datasets")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", *DATASETS.keys()],
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--output",
        default="data/raw",
        type=Path,
        help="Destination directory for the raw datasets",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    if args.dataset == "all":
        keys = DATASETS.keys()
    else:
        keys = [args.dataset]

    for key in keys:
        cfg = DATASETS[key]
        try:
            run_download(cfg, args.output)
        except Exception as exc:  # pragma: no cover - CLI feedback
            print(f"[error] {cfg.display_name}: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
