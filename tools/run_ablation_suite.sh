#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: tools/run_ablation_suite.sh [options]

Options:
  --data PATH           Path to the Ultralytics dataset YAML (default: yolo_dagm/data.yaml)
  --project DIR         Directory for experiment outputs (default: runs_ablation)
  --imgsz N             Training image size (default: 640)
  --epochs N            Number of training epochs (default: 150)
  --batch N             Training batch size (default: 4)
  --device SPEC         Device passed to Ultralytics (default: 0)
  --seed N              Random seed (default: 42)
  --srts-prob FLOAT     Probability for SRTS augmentation (default: 0.5)
  --srts-sigma FLOAT    Sigma for SRTS Gaussian blur (default: 2.5)
  --srts-alpha FLOAT    Alpha for SRTS residual fusion (default: 0.65)
  --npr-prob FLOAT      Probability for NPR mining (default: 0.5)
  --npr-patch N         Patch size for NPR (default: 128)
  --summary-name NAME   Base name for the aggregated summary artifacts (default: experiment_summary)
  --python PATH         Python interpreter to use (default: python)
  --skip-existing       Skip training when a completed run already exists
  -h, --help            Show this help message and exit
USAGE
}

DATA="yolo_dagm/data.yaml"
PROJECT="runs_ablation"
IMG_SIZE=640
EPOCHS=150
BATCH=4
DEVICE=0
SEED=42
SRTS_PROB=0.5
SRTS_SIGMA=2.5
SRTS_ALPHA=0.65
NPR_PROB=0.5
NPR_PATCH=128
SUMMARY_NAME="experiment_summary"
PYTHON_BIN=${PYTHON:-python}
SKIP_EXISTING=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)
            DATA="$2"
            shift 2
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --imgsz)
            IMG_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --srts-prob)
            SRTS_PROB="$2"
            shift 2
            ;;
        --srts-sigma)
            SRTS_SIGMA="$2"
            shift 2
            ;;
        --srts-alpha)
            SRTS_ALPHA="$2"
            shift 2
            ;;
        --npr-prob)
            NPR_PROB="$2"
            shift 2
            ;;
        --npr-patch)
            NPR_PATCH="$2"
            shift 2
            ;;
        --summary-name)
            SUMMARY_NAME="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=1
            shift 1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

EXPERIMENTS=(
    "A0_Baseline"
    "A1_P2only"
    "A2_SELite"
    "A12_P2_SELite"
    "B0_A12_Baseline"
    "B1_SRTS"
    "B2_NPR"
    "B12_SRTS_NPR"
)

common_args=(
    --data "$DATA"
    --project "$PROJECT"
    --imgsz "$IMG_SIZE"
    --epochs "$EPOCHS"
    --batch "$BATCH"
    --device "$DEVICE"
    --seed "$SEED"
    --srts-prob "$SRTS_PROB"
    --srts-sigma "$SRTS_SIGMA"
    --srts-alpha "$SRTS_ALPHA"
    --npr-prob "$NPR_PROB"
    --npr-patch "$NPR_PATCH"
)

if [[ "$SKIP_EXISTING" -eq 1 ]]; then
    common_args+=(--skip-existing)
fi

echo "Running ablation suite experiments sequentially..."
for exp in "${EXPERIMENTS[@]}"; do
    echo "[INFO] Starting $exp"
    "$PYTHON_BIN" tools/run_ablation_suite.py \
        "${common_args[@]}" \
        --experiments "$exp" \
        --summary-name "${exp}_summary"
    echo "[INFO] Completed $exp"
    echo
done

echo "[INFO] Generating aggregated summary artifacts"
final_args=("${common_args[@]}" --summary-name "$SUMMARY_NAME")
if [[ "$SKIP_EXISTING" -eq 0 ]]; then
    final_args+=(--skip-existing)
fi
"$PYTHON_BIN" tools/run_ablation_suite.py "${final_args[@]}"

echo "All experiments finished. Artifacts stored under $PROJECT"
