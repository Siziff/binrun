#!/bin/bash
# Sweep runner: train + eval pairs for a list of LoRA configs.
#
# Usage:
#   ./sweep_runner.sh <phase_name> <gpu_id> <config1.json> [config2.json ...]
#
# Each config.json is a small dict like:
#   {
#     "run_id": "p1_r8_lr5e6",
#     "rank": 8, "lr": 5e-6, "epochs": 3, "dropout": 0.05,
#     "target_modules": "all-linear", "grad_accum": 8
#   }

set -uo pipefail  # removed -e to prevent stop on individual run errors

PHASE="$1"; shift
GPU="$1"; shift

TD=/workspace-SR008.fs2/mikheev-kandy/TD
SWEEP=$TD/experiments/exp06_lora/sweep
mkdir -p $SWEEP/{configs,checkpoints,eval_test79,logs}

export HF_HOME=$TD/data/models/hf_cache
export HF_HUB_CACHE=$TD/data/models/hf_cache
export TMPDIR=$TD/data/tmp
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=$GPU

PY=$TD/data/envs/td-vlm/bin/python
PY_VLLM=$TD/data/envs/td-vllm/bin/python  # for eval if needed; train always td-vlm

cd $TD

for CFG_PATH in "$@"; do
    CFG_FILE=$(basename "$CFG_PATH")
    RUN_ID=$(${PY} -c "import json; print(json.load(open('$CFG_PATH'))['run_id'])")
    OUT_DIR=$SWEEP/checkpoints/$RUN_ID
    TRAIN_LOG=$SWEEP/logs/${RUN_ID}_train.log
    EVAL_LOG=$SWEEP/logs/${RUN_ID}_eval.log
    EVAL_OUT=$SWEEP/eval_test79/${RUN_ID}.jsonl

    if [[ -f "$OUT_DIR/final/adapter_model.safetensors" ]]; then
        echo "[$PHASE/$GPU] $RUN_ID adapter already exists — skipping train"
    else
        echo "[$PHASE/$GPU] $RUN_ID TRAIN start $(date)"
        # Read config
        RANK=$($PY -c "import json; print(json.load(open('$CFG_PATH'))['rank'])")
        LR=$($PY -c "import json; print(json.load(open('$CFG_PATH'))['lr'])")
        EP=$($PY -c "import json; print(json.load(open('$CFG_PATH'))['epochs'])")
        DROP=$($PY -c "import json; print(json.load(open('$CFG_PATH'))['dropout'])")
        TM=$($PY -c "import json; print(json.load(open('$CFG_PATH'))['target_modules'])")
        GA=$($PY -c "import json; print(json.load(open('$CFG_PATH'))['grad_accum'])")
        WUP=$($PY -c "import json,sys; d=json.load(open('$CFG_PATH')); print(d.get('warmup_ratio',0.1))")
        ALPHA=$($PY -c "import json,sys; d=json.load(open('$CFG_PATH')); print(d.get('lora_alpha',-1))")

        $PY -u experiments/exp06_lora/sweep_train.py \
            --output-dir "$OUT_DIR" \
            --lora-rank $RANK --lr $LR --epochs $EP \
            --lora-dropout $DROP --target-modules "$TM" \
            --grad-accum $GA --warmup-ratio $WUP \
            --lora-alpha $ALPHA \
            > "$TRAIN_LOG" 2>&1
        TRAIN_RC=$?
        echo "[$PHASE/$GPU] $RUN_ID TRAIN done rc=$TRAIN_RC $(date)"
        if [[ $TRAIN_RC -ne 0 ]]; then
            echo "[$PHASE/$GPU] $RUN_ID TRAIN FAILED, skipping eval"
            continue
        fi
    fi

    if [[ -f "$EVAL_OUT" ]] && [[ $(wc -l < "$EVAL_OUT") -ge 79 ]]; then
        echo "[$PHASE/$GPU] $RUN_ID eval already exists with $(wc -l < "$EVAL_OUT") records — skipping"
    else
        echo "[$PHASE/$GPU] $RUN_ID EVAL start $(date)"
        # Hack: use sweep_eval which writes to outputs dir, then move
        $PY -u experiments/exp06_lora/sweep_eval.py \
            --adapter "$OUT_DIR/final" \
            --out-suffix "_${RUN_ID}_sweep79" \
            --max-new-tokens 2048 \
            > "$EVAL_LOG" 2>&1
        EVAL_RC=$?
        echo "[$PHASE/$GPU] $RUN_ID EVAL done rc=$EVAL_RC $(date)"
        # Move output to canonical location
        DEFAULT_OUT=$TD/experiments/exp06_lora/outputs/lora_${RUN_ID}_sweep79.jsonl
        if [[ -f "$DEFAULT_OUT" ]]; then
            mv "$DEFAULT_OUT" "$EVAL_OUT"
        fi
    fi
done

echo "[$PHASE/$GPU] DONE all runs $(date)"
