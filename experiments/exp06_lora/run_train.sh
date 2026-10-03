#!/bin/bash
# LoRA training launcher (GPU 0).
set -e
cd /workspace-SR008.fs2/mikheev-kandy/TD
export TD_PREF=vlm
source env.sh
export CUDA_VISIBLE_DEVICES=0
# Tokenizers fork warning silencing
export TOKENIZERS_PARALLELISM=false

OUT_DIR="${1:-experiments/exp06_lora/checkpoints/run1}"
SMOKE="${SMOKE_TEST:-0}"
EPOCHS="${EPOCHS:-3}"

python3 experiments/exp06_lora/train_lora.py \
    --output-dir "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --per-device-batch 1 \
    --grad-accum 8 \
    --lr 2e-5 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --max-seq-len 6144 \
    --min-pixels $((256*28*28)) \
    --max-pixels $((1024*28*28)) \
    --log-every 5 \
    --smoke-test "$SMOKE"
