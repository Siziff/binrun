#!/bin/bash
cd /workspace-SR008.fs2/mikheev-kandy/TD
export TD_PREF=vlm
source env.sh
export CUDA_VISIBLE_DEVICES=1
python3 experiments/exp05_vlm_normalizer/run_ocr_vlm.py \
  --ocr logics-parsing-v2 --no-formatter \
  --inputs experiments/exp07_text_gold/picked_50_inputs.jsonl \
  --out-suffix __exp07_50 \
  --max-new-tokens 8192
