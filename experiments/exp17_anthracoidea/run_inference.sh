#!/usr/bin/env bash
set -e
cd /workspace-SR008.fs2/mikheev-kandy/TD

# Ждём завершения препроцессинга (534 файла в preprocessed_v3/Anthracoidea)
echo "[infer] waiting for preprocessing to finish (target 534)..."
while true; do
    n=$(ls data/preprocessed_v3/Anthracoidea/ 2>/dev/null | wc -l)
    if pgrep -f "preprocessing.preprocess_v3" >/dev/null; then
        echo "[infer] preprocessing still running: $n/534 done"
        sleep 30
    else
        echo "[infer] preprocessing finished: $n/534 done"
        break
    fi
done

export TD_PREF=vlm
source env.sh
export CUDA_VISIBLE_DEVICES=3

python experiments/exp06_lora/eval_full394.py \
  --adapter experiments/exp06_lora/sweep/checkpoints/p5j_r32_lr7e5_ep5_a256/final \
  --inputs experiments/exp17_anthracoidea/anthracoidea_inputs.jsonl \
  --out experiments/exp17_anthracoidea/outputs/anthracoidea_lora_p5j.jsonl
