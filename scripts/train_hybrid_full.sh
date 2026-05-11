#!/bin/bash
# Full-volume training for HybridUprightNet.
#
# Resumes from the working ft80 checkpoint (10.38° mean / 4.59% flip) instead
# of restarting from the Pang trunk, then does 5x more gradient steps per
# epoch (rpo=5) across three decreasing-LR segments. Expected wall time:
# ~70 min.  Segment LRs are one step lower than the initial ft40+ft80 schedule
# since the model has already been trained.
#
# Usage:
#   bash scripts/train_hybrid_full.sh               # run all 3 segments + eval
#   SKIP_EVAL=1 bash scripts/train_hybrid_full.sh   # train only
#   bash scripts/train_hybrid_full.sh eval          # evaluate hybrid_best.pth

set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date +%Y%m%d_%H%M)
LOG_PREFIX="logs/hybrid_full_${STAMP}"
SEG1_CKPT="models/hybrid_full_seg1_${STAMP}.pth"
SEG2_CKPT="models/hybrid_full_seg2_${STAMP}.pth"
SEG3_CKPT="models/hybrid_full_seg3_${STAMP}.pth"
# Warm-start from the best ft80 checkpoint; falls back to letting
# train_hybrid.py init from Pang trunk if missing.
RESUME_FROM="${RESUME_FROM:-models/hybrid_ft80_best_backup.pth}"

BATCH_SIZE=48
ROTS_PER_OBJ=5     # 5x more gradient steps per epoch than the previous runs
EVAL_EVERY=5
# Loss weights: same as the working ft80 run.
LAM_AXIS=1.0
LAM_SIGN=0.5
LAM_SUP=0.5
LAM_STAB=0.0

mkdir -p logs models

if [[ "${1:-}" == "eval" ]]; then
    CKPT="${2:-models/hybrid_best.pth}"
    echo "=== Fixed 10-rotation evaluation on $CKPT ==="
    python3 scripts/eval_shs_10rot.py --arch hybrid --ckpt "$CKPT" --split full
    exit 0
fi

if [[ ! -f "$RESUME_FROM" ]]; then
    echo "ERROR: warm-start ckpt $RESUME_FROM not found."
    echo "Set RESUME_FROM=<path> or bootstrap with train_hybrid.py first."
    exit 1
fi
echo "Warm-starting from $RESUME_FROM"

# ---- Segment 1: larger-volume refinement (60 ep, LR 3e-4) ---------------
echo
echo "========================================================================"
echo "  SEG 1/3 — refinement (60 ep, rpo=$ROTS_PER_OBJ, lr=3e-4, lr_trunk=3e-5)"
echo "========================================================================"
python3 -u scripts/train_hybrid.py \
    --epochs 60 --batch_size $BATCH_SIZE \
    --lr 3e-4 --lr_trunk 3e-5 \
    --lambda_axis $LAM_AXIS --lambda_sign $LAM_SIGN \
    --lambda_sup $LAM_SUP --lambda_stab $LAM_STAB \
    --eval_every $EVAL_EVERY --rotations_per_object $ROTS_PER_OBJ \
    --seed 42 --resume "$RESUME_FROM" \
    2>&1 | tee "${LOG_PREFIX}_seg1.log"
cp "$(ls -t models/hybrid_final_*.pth | head -1)" "$SEG1_CKPT"
echo ">> Seg1 final saved to $SEG1_CKPT"

# ---- Segment 2: mid-LR refinement (60 ep, LR 1e-4) ----------------------
echo
echo "========================================================================"
echo "  SEG 2/3 — refinement (60 ep, rpo=$ROTS_PER_OBJ, lr=1e-4, lr_trunk=1e-5)"
echo "========================================================================"
python3 -u scripts/train_hybrid.py \
    --epochs 60 --batch_size $BATCH_SIZE \
    --lr 1e-4 --lr_trunk 1e-5 \
    --lambda_axis $LAM_AXIS --lambda_sign $LAM_SIGN \
    --lambda_sup $LAM_SUP --lambda_stab $LAM_STAB \
    --eval_every $EVAL_EVERY --rotations_per_object $ROTS_PER_OBJ \
    --seed 43 --resume "$SEG1_CKPT" \
    2>&1 | tee "${LOG_PREFIX}_seg2.log"
cp "$(ls -t models/hybrid_final_*.pth | head -1)" "$SEG2_CKPT"
echo ">> Seg2 final saved to $SEG2_CKPT"

# ---- Segment 3: fine-tune (40 ep, LR 3e-5) ------------------------------
echo
echo "========================================================================"
echo "  SEG 3/3 — fine-tune (40 ep, rpo=$ROTS_PER_OBJ, lr=3e-5, lr_trunk=3e-6)"
echo "========================================================================"
python3 -u scripts/train_hybrid.py \
    --epochs 40 --batch_size $BATCH_SIZE \
    --lr 3e-5 --lr_trunk 3e-6 \
    --lambda_axis $LAM_AXIS --lambda_sign $LAM_SIGN \
    --lambda_sup $LAM_SUP --lambda_stab $LAM_STAB \
    --eval_every $EVAL_EVERY --rotations_per_object $ROTS_PER_OBJ \
    --seed 44 --resume "$SEG2_CKPT" \
    2>&1 | tee "${LOG_PREFIX}_seg3.log"
cp "$(ls -t models/hybrid_final_*.pth | head -1)" "$SEG3_CKPT"
echo ">> Seg3 final saved to $SEG3_CKPT"

echo
echo "========================================================================"
echo "  Training complete."
echo "    segment ckpts:  $SEG1_CKPT  $SEG2_CKPT  $SEG3_CKPT"
echo "    best by mean:   models/hybrid_best.pth"
echo "    best by flip:   models/hybrid_best_flip.pth"
echo "    per-seg logs:   ${LOG_PREFIX}_seg{1,2,3}.log"
echo "========================================================================"

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
    echo
    echo "=== Final evaluation on test_10rot full set (37000 samples) ==="
    python3 scripts/eval_shs_10rot.py --arch hybrid \
        --ckpt models/hybrid_best.pth --split full \
        2>&1 | tee "${LOG_PREFIX}_eval.log"
fi
