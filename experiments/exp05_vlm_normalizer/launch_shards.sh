#!/usr/bin/env bash
# Запускает run_qwen.py параллельно на нескольких GPU.
#
# Usage:
#   bash launch_shards.sh <model> <mode> <gpus_csv> [extra args...]
# Example:
#   bash launch_shards.sh Qwen/Qwen3-VL-8B-Thinking A 1,2,3,4,5,6,7
#
# Каждый шард обрабатывает свою долю input.jsonl. Все шарды дописывают в
# один общий outputs/<model>__<mode>.jsonl (с резюмированием).
set -euo pipefail

MODEL="$1"
MODE="$2"
GPUS="$3"
shift 3
EXTRA="$@"

PROJ="/workspace-SR008.fs2/mikheev-kandy/TD"
PY="${PROJ}/data/envs/td-vlm/bin/python"
EXP_DIR="${PROJ}/experiments/exp05_vlm_normalizer"
INPUTS="${EXP_DIR}/merged_inputs.jsonl"
LOG_DIR="${EXP_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Сплит входов на шарды равными частями.
TOTAL=$(wc -l < "${INPUTS}")
IFS=',' read -ra GPU_ARR <<< "${GPUS}"
N_SHARDS=${#GPU_ARR[@]}
echo "Total inputs: ${TOTAL}; sharding into ${N_SHARDS} (GPUs: ${GPUS})"

SHARD_DIR="${EXP_DIR}/shards"
mkdir -p "${SHARD_DIR}"
SLUG=$(echo "${MODEL}" | tr '/' '__')

# Делим: индексы строк [start, end) для шарда i.
PER=$(( (TOTAL + N_SHARDS - 1) / N_SHARDS ))

PIDS=()
for i in "${!GPU_ARR[@]}"; do
    GPU="${GPU_ARR[i]}"
    START=$(( i * PER ))
    END=$(( START + PER ))
    if [ "$END" -gt "$TOTAL" ]; then END="$TOTAL"; fi
    SHARD_PATH="${SHARD_DIR}/shard_${SLUG}_mode${MODE}_${i}.jsonl"
    sed -n "$((START+1)),${END}p" "${INPUTS}" > "${SHARD_PATH}"
    LOG="${LOG_DIR}/${SLUG}_mode${MODE}_gpu${GPU}.log"
    echo "  shard ${i}: lines ${START}..${END}  GPU=${GPU}  log=${LOG}"
    (
        export CUDA_VISIBLE_DEVICES="${GPU}"
        export HF_HOME="${PROJ}/data/models/hf_cache"
        export TRANSFORMERS_CACHE="${HF_HOME}"
        export HF_HUB_DISABLE_TELEMETRY=1
        export TOKENIZERS_PARALLELISM=false
        unset PYTHONPATH PYTHONSTARTUP
        export PYTHONNOUSERSITE=1
        cd "${PROJ}"
        "${PY}" -m experiments.exp05_vlm_normalizer.run_qwen \
            --model "${MODEL}" --mode "${MODE}" \
            --inputs "${SHARD_PATH}" \
            --out-suffix "__shard${i}" \
            ${EXTRA} \
            > "${LOG}" 2>&1
    ) &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} shards: ${PIDS[*]}"
echo "Tail logs with: tail -f ${LOG_DIR}/${SLUG}_mode${MODE}_gpu*.log"
wait
echo "All shards done."

# Объединение результатов шардов в один файл.
MERGED="${EXP_DIR}/outputs/${SLUG}__mode${MODE}.jsonl"
> "${MERGED}"
for i in "${!GPU_ARR[@]}"; do
    SHARD_OUT="${EXP_DIR}/outputs/${SLUG}__mode${MODE}__shard${i}.jsonl"
    [ -f "${SHARD_OUT}" ] && cat "${SHARD_OUT}" >> "${MERGED}"
done
echo "Merged → ${MERGED}"
wc -l "${MERGED}" 2>/dev/null || true
