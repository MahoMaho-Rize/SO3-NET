#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
One-step pipeline for partial ModelNet15 hierarchical uprightness training.

This script:
  1. downloads/extracts ModelNet40 if needed,
  2. copies the 15 UprightNet categories,
  3. generates camera-style partial OFF files with Blender,
  4. builds point-wise bottom-to-top hierarchy labels,
  5. trains a hierarchy segmentation model and evaluates upright direction.

Common overrides:
  PYTHON=python3
  BLENDER=blender
  MODELNET40_ROOT=datasets/ModelNet40
  UPRIGHTNET15_ROOT=datasets/uprightnet15
  PARTIAL_ROOT=datasets/uprightnet15_partial_camera
  NPZ_ROOT=datasets/upright_hierarchy_npz
  OUT_DIR=models/hierarchical_uprightnet
  VIEWS_PER_MODEL=8
  NUM_LEVELS=5
  EPOCHS=80
  BATCH_SIZE=128
  DEVICE=cuda
  DATA_PARALLEL=1

Skip steps:
  SKIP_DOWNLOAD=1
  SKIP_PARTIAL=1
  SKIP_NPZ=1
  SKIP_TRAIN=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODELNET40_URL="${MODELNET40_URL:-http://modelnet.cs.princeton.edu/ModelNet40.zip}"
PYTHON="${PYTHON:-python3}"
BLENDER="${BLENDER:-blender}"
DATA_ROOT="${DATA_ROOT:-datasets}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$DATA_ROOT/downloads}"
MODELNET40_ZIP="${MODELNET40_ZIP:-$DOWNLOAD_DIR/ModelNet40.zip}"
MODELNET40_ROOT="${MODELNET40_ROOT:-$DATA_ROOT/ModelNet40}"
UPRIGHTNET15_ROOT="${UPRIGHTNET15_ROOT:-$DATA_ROOT/uprightnet15}"
PARTIAL_ROOT="${PARTIAL_ROOT:-$DATA_ROOT/uprightnet15_partial_camera}"
NPZ_ROOT="${NPZ_ROOT:-$DATA_ROOT/upright_hierarchy_npz}"
OUT_DIR="${OUT_DIR:-models/hierarchical_uprightnet}"

VIEWS_PER_MODEL="${VIEWS_PER_MODEL:-8}"
OUTPUT_COUNT="${OUTPUT_COUNT:-2048}"
DEPTH_WIDTH="${DEPTH_WIDTH:-128}"
DEPTH_HEIGHT="${DEPTH_HEIGHT:-128}"
MAX_DEPTH_SIZE="${MAX_DEPTH_SIZE:-512}"
NUM_LEVELS="${NUM_LEVELS:-5}"

EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-auto}"
DATA_PARALLEL="${DATA_PARALLEL:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CLASS_BALANCE="${CLASS_BALANCE:-1}"

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
    log "SKIP_NPZ=1, not rebuilding hierarchy NPZ files"
    return
  fi
  log "Building point-wise hierarchy NPZ"
  "$PYTHON" scripts/build_hierarchy_npz.py \
    --input-root "$PARTIAL_ROOT" \
    --full-root "$UPRIGHTNET15_ROOT" \
    --out-dir "$NPZ_ROOT" \
    --num-points "$OUTPUT_COUNT" \
    --num-levels "$NUM_LEVELS" \
    --source-up-axis z
}

train_hierarchy() {
  if [[ "$SKIP_TRAIN" == "1" ]]; then
    log "SKIP_TRAIN=1, data preparation complete"
    return
  fi
  log "Starting hierarchical uprightness training"
  extra_train_args=()
  if [[ "$DATA_PARALLEL" == "1" ]]; then
    extra_train_args+=(--data-parallel)
  fi
  if [[ "$CLASS_BALANCE" == "1" ]]; then
    extra_train_args+=(--class-balance)
  fi
  "$PYTHON" scripts/train_hierarchical_uprightnet.py \
    --train-npz "$NPZ_ROOT/train.npz" \
    --test-npz "$NPZ_ROOT/test.npz" \
    --out-dir "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    "${extra_train_args[@]}"
}

log "Pipeline root: $ROOT_DIR"
download_modelnet40
extract_uprightnet15_categories
generate_partial_off
build_npz
train_hierarchy
log "Done"
