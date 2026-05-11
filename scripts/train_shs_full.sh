#!/bin/bash
# Full-training recipe for SHS-style decomposed-head upright network.
#
# Based on the 160-epoch staircase run that gave best = 15.5° mean / 83% acc@10
# / 7.4% flip on the fixed 10-rotation test set. The recipe uses:
#
#   - Single-stage end-to-end (the "freeze trunk then unfreeze" phase fails
#     because the Pang 2022 trunk is NOT rotation-equivariant — frozen
#     features can't tell direction apart, see 2026-05-05 overfit run).
#   - Warm-restart cosine: high LR → low LR across 3 segments, each loading
#     the previous segment's final checkpoint.
#   - λ_stab=0.4 (doubled from 0.2) because the 10rot diagnostic showed
#     the remaining failures are systematic sign-flips on 5 lamp + 3 bed
#     objects that L_stab should be able to resolve (they are all mass-
#     above-support-gap violations).
#
# Usage:
#   bash scripts/train_shs_full.sh              # run all 3 segments end-to-end
#   bash scripts/train_shs_full.sh eval         # only run final eval
#   SKIP_EVAL=1 bash scripts/train_shs_full.sh  # train, skip eval at end
#
# Total wall time: ~22 min on a single RTX5880-Ada.

set -euo pipefail

cd "$(dirname "$0")/.."

# ---- Config --------------------------------------------------------------
STAMP=$(date +%Y%m%d_%H%M)
LOG_DIR=logs
MODEL_DIR=models
LOG_PREFIX="$LOG_DIR/shs_full_${STAMP}"
SEG1_CKPT="$MODEL_DIR/shs_full_seg1_${STAMP}.pth"
SEG2_CKPT="$MODEL_DIR/shs_full_seg2_${STAMP}.pth"
SEG3_CKPT="$MODEL_DIR/shs_full_seg3_${STAMP}.pth"

BATCH_SIZE=48
ROTS_PER_OBJ=1        # each object is re-rotated every epoch (effectively ∞ aug)
EVAL_EVERY=5
# Decomposed loss weights — λ_stab doubled vs default (0.2 → 0.4) to pull
# the long-tail sign flips back.
LAM_AXIS=1.0
LAM_SIGN=0.5
LAM_SUP=0.1
LAM_STAB=0.4

mkdir -p "$LOG_DIR" "$MODEL_DIR"

# ---- Eval-only shortcut --------------------------------------------------
if [[ "${1:-}" == "eval" ]]; then
    CKPT="${2:-$MODEL_DIR/shs_best.pth}"
    echo "=== Fixed 10-rotation evaluation on $CKPT ==="
    python3 scripts/eval_shs_10rot.py --ckpt "$CKPT"
    exit 0
fi

# ---- Segment 1: warm-start from pretrained trunk (40 epoch, LR 1e-3) -----
echo
echo "========================================================================"
echo "  SEGMENT 1/3 — warm-start end-to-end (40 ep, lr=1e-3, lr_trunk=1e-4)"
echo "========================================================================"
python3 -u scripts/train_shs.py \
    --epochs 40 --batch_size $BATCH_SIZE \
    --lr 1e-3 --lr_trunk 1e-4 \
    --head_type decomposed --loss decomposed \
    --lambda_axis $LAM_AXIS --lambda_sign $LAM_SIGN \
    --lambda_sup $LAM_SUP --lambda_stab $LAM_STAB \
    --eval_every $EVAL_EVERY --rotations_per_object $ROTS_PER_OBJ \
    --seed 42 \
    2>&1 | tee "${LOG_PREFIX}_seg1.log"

# Copy the segment's final checkpoint to a timestamped name so we can
# resume the next segment from a known file (train_shs.py writes
# shs_final_<timestamp>.pth — grab the most recent one).
cp "$(ls -t $MODEL_DIR/shs_final_*.pth | head -1)" "$SEG1_CKPT"
echo ">> Segment 1 final saved to $SEG1_CKPT"

# ---- Segment 2: mid-LR refinement (60 epoch, LR 3e-4) --------------------
echo
echo "========================================================================"
echo "  SEGMENT 2/3 — refinement (60 ep, lr=3e-4, lr_trunk=3e-5)"
echo "========================================================================"
python3 -u scripts/train_shs.py \
    --epochs 60 --batch_size $BATCH_SIZE \
    --lr 3e-4 --lr_trunk 3e-5 \
    --head_type decomposed --loss decomposed \
    --lambda_axis $LAM_AXIS --lambda_sign $LAM_SIGN \
    --lambda_sup $LAM_SUP --lambda_stab $LAM_STAB \
    --eval_every $EVAL_EVERY --rotations_per_object $ROTS_PER_OBJ \
    --seed 43 \
    --resume "$SEG1_CKPT" \
    2>&1 | tee "${LOG_PREFIX}_seg2.log"
cp "$(ls -t $MODEL_DIR/shs_final_*.pth | head -1)" "$SEG2_CKPT"
echo ">> Segment 2 final saved to $SEG2_CKPT"

# ---- Segment 3: fine-tune (60 epoch, LR 1e-4) ----------------------------
echo
echo "========================================================================"
echo "  SEGMENT 3/3 — fine-tune (60 ep, lr=1e-4, lr_trunk=1e-5)"
echo "========================================================================"
python3 -u scripts/train_shs.py \
    --epochs 60 --batch_size $BATCH_SIZE \
    --lr 1e-4 --lr_trunk 1e-5 \
    --head_type decomposed --loss decomposed \
    --lambda_axis $LAM_AXIS --lambda_sign $LAM_SIGN \
    --lambda_sup $LAM_SUP --lambda_stab $LAM_STAB \
    --eval_every $EVAL_EVERY --rotations_per_object $ROTS_PER_OBJ \
    --seed 44 \
    --resume "$SEG2_CKPT" \
    2>&1 | tee "${LOG_PREFIX}_seg3.log"
cp "$(ls -t $MODEL_DIR/shs_final_*.pth | head -1)" "$SEG3_CKPT"
echo ">> Segment 3 final saved to $SEG3_CKPT"

echo
echo "========================================================================"
echo "  Training complete."
echo "    segment checkpoints: $SEG1_CKPT  $SEG2_CKPT  $SEG3_CKPT"
echo "    best (by signed mean): $MODEL_DIR/shs_best.pth"
echo "    best (by flip rate):   $MODEL_DIR/shs_best_flip.pth"
echo "    per-segment logs:      ${LOG_PREFIX}_seg{1,2,3}.log"
echo "========================================================================"

# ---- Final fixed 10-rotation evaluation ---------------------------------
if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
    echo
    echo "=== Final evaluation on test_10rot (3700 samples) ==="
    python3 scripts/eval_shs_10rot.py --ckpt "$MODEL_DIR/shs_best.pth" \
        2>&1 | tee "${LOG_PREFIX}_eval.log"
fi
