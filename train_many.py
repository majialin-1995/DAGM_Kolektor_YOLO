"""Utility helpers for running multiple YOLO experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from ultralytics import YOLO
from ultralytics.nn import tasks as yolo_tasks

from yolo_dagm.selite_registry import register_selite_modules

register_selite_modules()

from callbacks.npr_miner import UltralyticsNPRCallback
from callbacks.preproc_srts import UltralyticsSRTSCallback
from modules.se_layers import C2f_SELite, SELite

# Register custom DAGM layers with Ultralytics so YAML configs can reference them
yolo_tasks.C2f_SELite = C2f_SELite
yolo_tasks.SELite = SELite

THIS = Path(__file__).parent.resolve()


def ensure_model_variants() -> None:
    """Ensure that derivative YOLO configuration files exist on disk."""

    model_dir = THIS / "models"
    model_dir.mkdir(exist_ok=True)
    model_p2_se = model_dir / "yolov8n_p2_se.yaml"
    model_p2_only = model_dir / "yolov8n_p2_only.yaml"
    if model_p2_se.exists() and not model_p2_only.exists():
        text = model_p2_se.read_text()
        model_p2_only.write_text(text.replace("C2f_SELite", "C2f"))


def _resolve_run_dir(project: Path, name: str, before: Sequence[Path]) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    before_set = {b.resolve() for b in before}
    candidates = [p for p in project.glob(f"{name}*") if p.is_dir()]
    new_dirs = [p for p in candidates if p.resolve() not in before_set]
    if new_dirs:
        return sorted(new_dirs, key=lambda p: p.stat().st_mtime)[-1]
    if candidates:
        return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
    return project / name


def train_once(
    model_def,
    data,
    project,
    name,
    imgsz: int = 640,
    epochs: int = 150,
    batch: int = 16,
    device=0,
    seed: int = 42,
    callbacks: Iterable | None = None,
):
    """Run a single YOLO training experiment and return its output directory."""

    project_path = Path(project)
    before = list(project_path.glob(f"{name}*"))
    model = YOLO(model_def)
    if callbacks:
        for cb in callbacks:
            if hasattr(cb, "on_pretrain_routine_end"):
                model.add_callback("on_pretrain_routine_end", cb.on_pretrain_routine_end)
    model.train(
        data=data,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        project=project,
        name=name,
        seed=seed,
        workers=4,
        verbose=True,
    )
    return _resolve_run_dir(project_path, name, before)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", default="runs_defects")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=0)
    args = ap.parse_args()

    ensure_model_variants()
    model_p2_se = THIS / "models" / "yolov8n_p2_se.yaml"

    train_once(
        "yolov8n.pt",
        args.data,
        args.project,
        "B0_yolov8n",
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
    )
    train_once(
        str(THIS / "models/yolov8n_p2_only.yaml"),
        args.data,
        args.project,
        "A1_P2only",
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
    )
    train_once(
        str(model_p2_se),
        args.data,
        args.project,
        "A1A2_P2_SE",
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
    )
    cb_srts = UltralyticsSRTSCallback(prob=0.5, sigma=2.5, alpha=0.65)
    train_once(
        str(model_p2_se),
        args.data,
        args.project,
        "A1A2B1_P2_SE_SRTS",
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        callbacks=[cb_srts],
    )
    cb_npr = UltralyticsNPRCallback(prob=0.5, patch_size=128)
    train_once(
        str(model_p2_se),
        args.data,
        args.project,
        "A1A2B1B2_P2_SE_SRTS_NPR",
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        callbacks=[cb_srts, cb_npr],
    )


if __name__ == "__main__":
    main()
