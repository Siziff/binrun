#!/usr/bin/env bash
set -e
cd /workspace-SR008.fs2/mikheev-kandy/TD
export TD_PREF=ocr
source env.sh
export CUDA_VISIBLE_DEVICES=2
python -m src.preprocessing.preprocess_v3 \
  --src-dir Smuts/Anthracoidea \
  --dst-dir data/preprocessed_v3/Anthracoidea \
  --index-path data/preprocessed_v3/preproc_index_anthracoidea.json
