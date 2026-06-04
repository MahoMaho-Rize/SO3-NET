#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
One-step pipeline for partial ModelNet15 uprightness-classifier training.

This script:
  1. downloads official ModelNet40 OFF meshes,
  2. extracts the 15 UprightNet categories,
  3. generates camera-style partial point clouds with Blender,
  4. packs partial OFF files into train/test NPZ files,
  5. starts candidate-conditioned uprightness binary classification training.

Usage:
  ./scripts/run_modelnet15_uprightness_pipeline.sh

Common overrides:
  MODELNET40_URL=http://modelnet.cs.princeton.edu/ModelNet40.zip
  PYTHON=python
  BLENDER=blender
  DATA_ROOT=datasets
  UPRIGHTNET15_ROOT=datasets/uprightnet15
  PARTIAL_ROOT=datasets/uprightnet15_partial_camera
  NPZ_ROOT=datasets/uprightness_partial_npz
  OUT_DIR=models/uprightness_classifier
  VIEWS_PER_MODEL=8
  EPOCHS=80
  BATCH_SIZE=128
  SAMPLES_PER_CLOUD=20
  DEVICE=cuda
  NUM_WORKERS=8

Example:
  DEVICE=cuda EPOCHS=80 BATCH_SIZE=128 ./scripts/run_modelnet15_uprightness_pipeline.sh

Skip steps:
  SKIP_DOWNLOAD=1  # assume ModelNet40 is already extracted
  SKIP_PARTIAL=1   # assume partial OFF files already exist
  SKIP_NPZ=1       # assume train/test NPZ files already exist
  SKIP_TRAIN=1     # prepare data only
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODELNET40_URL="${MODELNET40_URL:-http://modelnet.cs.princeton.edu/ModelNet40.zip}"
PYTHON="${PYTHON:-python}"
BLENDER="${BLENDER:-blender}"
DATA_ROOT="${DATA_ROOT:-datasets}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$DATA_ROOT/downloads}"
MODELNET40_ZIP="${MODELNET40_ZIP:-$DOWNLOAD_DIR/ModelNet40.zip}"
MODELNET40_ROOT="${MODELNET40_ROOT:-$DATA_ROOT/ModelNet40}"
UPRIGHTNET15_ROOT="${UPRIGHTNET15_ROOT:-$DATA_ROOT/uprightnet15}"
PARTIAL_ROOT="${PARTIAL_ROOT:-$DATA_ROOT/uprightnet15_partial_camera}"
NPZ_ROOT="${NPZ_ROOT:-$DATA_ROOT/uprightness_partial_npz}"
OUT_DIR="${OUT_DIR:-models/uprightness_classifier}"

VIEWS_PER_MODEL="${VIEWS_PER_MODEL:-8}"
OUTPUT_COUNT="${OUTPUT_COUNT:-2048}"
DEPTH_WIDTH="${DEPTH_WIDTH:-128}"
DEPTH_HEIGHT="${DEPTH_HEIGHT:-128}"
MAX_DEPTH_SIZE="${MAX_DEPTH_SIZE:-512}"

EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SAMPLES_PER_CLOUD="${SAMPLES_PER_CLOUD:-20}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-auto}"
NUM_WORKERS="${NUM_WORKERS:-8}"

SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_PARTIAL="${SKIP_PARTIAL:-0}"
SKIP_NPZ="${SKIP_NPZ:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

CATEGORIES=(
  bed bench bottle bowl car chair cone cup lamp monitor sofa stool table toilet vase
)

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

download_modelnet40() {
  mkdir -p "$DOWNLOAD_DIR" "$DATA_ROOT"
  if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    log "SKIP_DOWNLOAD=1, not downloading ModelNet40"
    return
  fi

  if [[ ! -f "$MODELNET40_ZIP" ]]; then
    require_cmd curl
    log "Downloading ModelNet40 from $MODELNET40_URL"
    curl -L "$MODELNET40_URL" -o "$MODELNET40_ZIP"
  else
    log "Using existing archive $MODELNET40_ZIP"
  fi

  if [[ ! -d "$MODELNET40_ROOT" ]]; then
    log "Extracting $MODELNET40_ZIP into $DATA_ROOT"
    "$PYTHON" -m zipfile -e "$MODELNET40_ZIP" "$DATA_ROOT"
  else
    log "Using existing extracted ModelNet40 root $MODELNET40_ROOT"
  fi
}

extract_uprightnet15_categories() {
  if [[ ! -d "$MODELNET40_ROOT" ]]; then
    echo "ModelNet40 root not found: $MODELNET40_ROOT" >&2
    echo "Set MODELNET40_ROOT or disable SKIP_DOWNLOAD." >&2
    exit 1
  fi

  mkdir -p "$UPRIGHTNET15_ROOT"
  log "Extracting 15 selected categories into $UPRIGHTNET15_ROOT"
  for category in "${CATEGORIES[@]}"; do
    src="$MODELNET40_ROOT/$category"
    dst="$UPRIGHTNET15_ROOT/$category"
    if [[ ! -d "$src" ]]; then
      echo "Missing ModelNet40 category directory: $src" >&2
      exit 1
    fi
    if [[ -d "$dst" ]]; then
      echo "  [skip] $category already exists"
    else
      cp -a "$src" "$dst"
      echo "  [copy] $category"
    fi
  done

  log "Repairing inline OFF headers for viewer/tool compatibility"
  UPRIGHTNET15_ROOT_ABS="$(cd "$UPRIGHTNET15_ROOT" && pwd)" "$PYTHON" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["UPRIGHTNET15_ROOT_ABS"])
count = 0
for path in root.rglob("*.off"):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        continue
    if not text:
        continue
    first = text[0].strip()
    if first.startswith("OFF") and first != "OFF":
        text[0] = "OFF"
        text.insert(1, first[3:].strip())
        path.write_text("\n".join(text) + "\n", encoding="utf-8")
        count += 1
print(f"  repaired_inline_off_headers={count}")
PY
}

generate_partial_off() {
  if [[ "$SKIP_PARTIAL" == "1" ]]; then
    log "SKIP_PARTIAL=1, not generating partial OFF files"
    return
  fi
  require_cmd "$BLENDER"
  log "Generating camera-style partial point clouds"
  "$BLENDER" --background --python scripts/blender_partial_uprightnet15.py -- \
    --input-root "$UPRIGHTNET15_ROOT" \
    --output-root "$PARTIAL_ROOT" \
    --views-per-model "$VIEWS_PER_MODEL" \
    --output-count "$OUTPUT_COUNT" \
    --depth-width "$DEPTH_WIDTH" \
    --depth-height "$DEPTH_HEIGHT" \
    --max-depth-size "$MAX_DEPTH_SIZE"
}

build_npz() {
  if [[ "$SKIP_NPZ" == "1" ]]; then
    log "SKIP_NPZ=1, not rebuilding NPZ files"
    return
  fi
  log "Packing partial OFF files into NPZ"
  "$PYTHON" scripts/build_uprightness_npz.py \
    --input-root "$PARTIAL_ROOT" \
    --out-dir "$NPZ_ROOT" \
    --num-points "$OUTPUT_COUNT" \
    --source-up-axis z
}

train_classifier() {
  if [[ "$SKIP_TRAIN" == "1" ]]; then
    log "SKIP_TRAIN=1, data preparation complete"
    return
  fi
  log "Starting uprightness binary-classifier training"
  "$PYTHON" scripts/train_uprightness_classifier.py \
    --train-npz "$NPZ_ROOT/train.npz" \
    --test-npz "$NPZ_ROOT/test.npz" \
    --out-dir "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --samples-per-cloud "$SAMPLES_PER_CLOUD" \
    --lr "$LR" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE"
}

log "Pipeline root: $ROOT_DIR"
download_modelnet40
extract_uprightnet15_categories
generate_partial_off
build_npz
train_classifier
log "Done"
