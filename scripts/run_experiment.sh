#!/usr/bin/env bash
set -euo pipefail

# Install the package and authenticate with Hugging Face before running.
DATA_ROOT=${DATA_ROOT:-data}
RUN_ROOT=${RUN_ROOT:-runs/esen}
DEVICE=${DEVICE:-cuda}
PLATFORM=${PLATFORM:-CPU}
FORCE_STEPS=${FORCE_STEPS:-5000}
STUDENT_STEPS=${STUDENT_STEPS:-10000}
FORCE_BATCH=${FORCE_BATCH:-16}
STUDENT_BATCH=${STUDENT_BATCH:-8}
SAMPLES=${SAMPLES:-32}

mkdir -p "$RUN_ROOT"
pdd-md download-data --data-root "$DATA_ROOT"
pdd-md train-force --backend esen --data-root "$DATA_ROOT" \
  --output "$RUN_ROOT/force.pt" --steps "$FORCE_STEPS" \
  --batch-size "$FORCE_BATCH" --device "$DEVICE" --platform "$PLATFORM"
pdd-md evaluate-force --data-root "$DATA_ROOT" \
  --force-checkpoint "$RUN_ROOT/force.pt" --output "$RUN_ROOT/force_eval.json" \
  --samples 128 --device "$DEVICE" --platform "$PLATFORM"
pdd-md train-pdd --data-root "$DATA_ROOT" \
  --force-checkpoint "$RUN_ROOT/force.pt" --output "$RUN_ROOT/pdd.pt" \
  --steps "$STUDENT_STEPS" --batch-size "$STUDENT_BATCH" \
  --max-block 8 --block-sizes 1 2 4 8 --prefix-blocks 2 \
  --device "$DEVICE" --platform "$PLATFORM"
pdd-md train-direct --data-root "$DATA_ROOT" \
  --force-checkpoint "$RUN_ROOT/force.pt" --output "$RUN_ROOT/direct_L8.pt" \
  --coarse-factor 8 --steps "$STUDENT_STEPS" --batch-size "$STUDENT_BATCH" \
  --device "$DEVICE" --platform "$PLATFORM"
pdd-md evaluate --data-root "$DATA_ROOT" \
  --pdd-checkpoint "$RUN_ROOT/pdd.pt" --direct-checkpoint "$RUN_ROOT/direct_L8.pt" \
  --output "$RUN_ROOT/eval_40fs.json" --blocks 1 2 4 8 \
  --fine-steps 80 --samples "$SAMPLES" --device "$DEVICE" --platform "$PLATFORM"
