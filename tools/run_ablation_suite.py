"""Run the full 8-experiment ablation suite and export artifacts."""

from __future__ import annotations

import argparse
import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from callbacks.npr_miner import UltralyticsNPRCallback
from callbacks.preproc_srts import UltralyticsSRTSCallback
from train_many import THIS, ensure_model_variants, train_once

LOGGER = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    name: str
    model: str
    callback_factories: Sequence[Callable[[], object]] = field(default_factory=tuple)
    description: str | None = None

    def instantiate_callbacks(self) -> list[object]:
        return [factory() for factory in self.callback_factories]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all ablation experiments sequentially.")
    parser.add_argument("--data", required=True, help="Path to the Ultralytics data YAML file.")
    parser.add_argument("--project", default="runs_ablation", help="Directory where experiments are stored.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default=0, help="Device specification passed to Ultralytics.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments that already have a completed results.csv file.",
    )
    parser.add_argument("--srts-prob", type=float, default=0.5, help="Probability for SRTS augmentation.")
    parser.add_argument("--srts-sigma", type=float, default=2.5, help="Sigma for SRTS Gaussian blur.")
    parser.add_argument("--srts-alpha", type=float, default=0.65, help="Alpha for SRTS residual fusion.")
    parser.add_argument("--npr-prob", type=float, default=0.5, help="Probability of applying NPR.")
    parser.add_argument("--npr-patch", type=int, default=128, help="Patch size used by NPR.")
    parser.add_argument(
        "--summary-name",
        default="experiment_summary",
        help="Base filename (without extension) for exported summary tables and plots.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Optional subset of experiment names to run instead of the full suite.",
    )
    return parser.parse_args()


def build_experiments(args: argparse.Namespace) -> list[ExperimentConfig]:
    model_dir = THIS / "models"
    srts_factory = lambda prob=args.srts_prob, sigma=args.srts_sigma, alpha=args.srts_alpha: UltralyticsSRTSCallback(  # noqa: E731
        prob=prob,
        sigma=sigma,
        alpha=alpha,
    )
    npr_factory = lambda prob=args.npr_prob, patch=args.npr_patch: UltralyticsNPRCallback(  # noqa: E731
        prob=prob,
        patch_size=patch,
    )
    return [
        ExperimentConfig("A0_Baseline", "yolov8n.pt", description="Official YOLOv8n baseline"),
        ExperimentConfig(
            "A1_P2only",
            str(model_dir / "yolov8n_p2_only.yaml"),
            description="Baseline + P2 detection head",
        ),
        ExperimentConfig(
            "A2_SELite",
            str(model_dir / "yolov8n_se_only.yaml"),
            description="Baseline + SELite attention",
        ),
        ExperimentConfig(
            "A12_P2_SELite",
            str(model_dir / "yolov8n_p2_se.yaml"),
            description="P2 head with SELite attention",
        ),
        ExperimentConfig(
            "B0_A12_Baseline",
            str(model_dir / "yolov8n_p2_se.yaml"),
            description="A12 architecture baseline for group B",
        ),
        ExperimentConfig(
            "B1_SRTS",
            str(model_dir / "yolov8n_p2_se.yaml"),
            callback_factories=[srts_factory],
            description="A12 + SRTS augmentation",
        ),
        ExperimentConfig(
            "B2_NPR",
            str(model_dir / "yolov8n_p2_se.yaml"),
            callback_factories=[npr_factory],
            description="A12 + NPR mining",
        ),
        ExperimentConfig(
            "B12_SRTS_NPR",
            str(model_dir / "yolov8n_p2_se.yaml"),
            callback_factories=[srts_factory, npr_factory],
            description="A12 + SRTS + NPR",
        ),
    ]


def find_existing_run(project: Path, name: str) -> Path | None:
    candidates = [p for p in project.glob(f"{name}*") if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def read_results_csv(path: Path) -> tuple[list[str], list[dict[str, float | str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows: list[dict[str, float | str]] = []
        for row in reader:
            parsed: dict[str, float | str] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if value == "":
                    parsed[key] = math.nan
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return fieldnames, rows


def numeric_columns(columns: Iterable[str], rows: Sequence[dict[str, float | str]]) -> list[str]:
    numeric_keys: list[str] = []
    for key in columns:
        values: list[float] = []
        valid = True
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
            elif isinstance(value, str):
                valid = False
                break
        if valid and values:
            numeric_keys.append(key)
    return numeric_keys


def plot_training_curves(results_csv: Path, columns: list[str], rows: list[dict[str, float | str]]) -> None:
    if not rows:
        LOGGER.warning("No rows found in %s; skipping curve export.", results_csv)
        return

    x_key = columns[0]
    metrics = numeric_columns(columns[1:], rows)
    if not metrics:
        LOGGER.warning("No numeric metrics found in %s; skipping curve export.", results_csv)
        return

    x = np.array([rows[i].get(x_key, i) for i in range(len(rows))], dtype=float)
    n_plots = len(metrics)
    ncols = min(4, n_plots)
    nrows = int(math.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        y = np.array([rows[i].get(metric, np.nan) for i in range(len(rows))], dtype=float)
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.8, label=metric)
        ax.set_title(metric)
        ax.set_xlabel(x_key)
        ax.grid(True, alpha=0.3)
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].axis("off")

    base = results_csv.with_name("training_curves")
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=200)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    plt.close(fig)


def summarise_experiment(
    experiment: ExperimentConfig,
    run_dir: Path,
    columns: list[str],
    rows: list[dict[str, float | str]],
) -> dict[str, float | str]:
    summary: dict[str, float | str] = {
        "experiment": experiment.name,
        "description": experiment.description or "",
        "run_dir": str(run_dir.resolve()),
    }
    if not rows:
        return summary

    last_row = rows[-1]
    for key in columns[1:]:
        value = last_row.get(key)
        if isinstance(value, (int, float)) and not math.isnan(value):
            summary[key] = value
    weights_dir = run_dir / "weights"
    best_weight = weights_dir / "best.pt"
    last_weight = weights_dir / "last.pt"
    if best_weight.exists():
        summary["best_weight"] = str(best_weight.resolve())
    if last_weight.exists():
        summary["last_weight"] = str(last_weight.resolve())
    return summary


def export_summary_artifacts(project: Path, summary_rows: list[dict[str, float | str]], base_name: str) -> None:
    if not summary_rows:
        return

    output_csv = project / f"{base_name}.csv"
    fieldnames: list[str] = sorted({key for row in summary_rows for key in row.keys()})
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    key_metrics = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    available_metrics = [metric for metric in key_metrics if any(metric in row for row in summary_rows)]
    if not available_metrics:
        LOGGER.warning("No key metrics available for summary plotting.")
        return

    width = 0.18
    indices = np.arange(len(summary_rows))
    fig, ax = plt.subplots(figsize=(max(8, len(summary_rows) * 1.2), 5))
    for idx, metric in enumerate(available_metrics):
        offset = (idx - (len(available_metrics) - 1) / 2.0) * width
        values = [row.get(metric, np.nan) for row in summary_rows]
        ax.bar(indices + offset, values, width=width, label=metric)
    ax.set_xticks(indices)
    ax.set_xticklabels([row.get("experiment", "?") for row in summary_rows], rotation=35, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Final validation metrics per experiment")
    ax.legend()
    fig.tight_layout()

    base_path = project / base_name
    fig.savefig(base_path.with_suffix(".png"), dpi=200)
    fig.savefig(base_path.with_suffix(".pdf"))
    fig.savefig(base_path.with_suffix(".svg"))
    plt.close(fig)


def run_suite(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    ensure_model_variants()

    project_path = Path(args.project)
    project_path.mkdir(parents=True, exist_ok=True)
    experiments = build_experiments(args)
    if args.experiments:
        selected = set(args.experiments)
        known = {exp.name for exp in experiments}
        missing = sorted(selected - known)
        if missing:
            raise SystemExit(f"Unknown experiments requested: {', '.join(missing)}")
        experiments = [exp for exp in experiments if exp.name in selected]

    summary_rows: list[dict[str, float | str]] = []
    for exp in experiments:
        LOGGER.info("Starting experiment %s", exp.name)
        existing = find_existing_run(project_path, exp.name) if args.skip_existing else None
        if existing and (existing / "results.csv").exists():
            LOGGER.info("Skipping %s because results already exist at %s", exp.name, existing)
            run_dir = existing
        else:
            run_dir = train_once(
                exp.model,
                args.data,
                str(project_path),
                exp.name,
                imgsz=args.imgsz,
                epochs=args.epochs,
                batch=args.batch,
                device=args.device,
                seed=args.seed,
                callbacks=exp.instantiate_callbacks(),
            )
        results_csv = run_dir / "results.csv"
        if not results_csv.exists():
            LOGGER.warning("results.csv not found for %s at %s", exp.name, results_csv)
            summary_rows.append({
                "experiment": exp.name,
                "description": exp.description or "",
                "run_dir": str(run_dir.resolve()),
            })
            continue

        columns, rows = read_results_csv(results_csv)
        plot_training_curves(results_csv, columns, rows)
        summary_rows.append(summarise_experiment(exp, run_dir, columns, rows))

    export_summary_artifacts(project_path, summary_rows, args.summary_name)


def main() -> None:
    args = parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
