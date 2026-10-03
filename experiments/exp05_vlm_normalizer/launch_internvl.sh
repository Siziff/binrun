#!/usr/bin/env bash
# Запускает run_internvl.py параллельно на нескольких GPU.
set -euo pipefail

MODE="$1"
GPUS="$2"

PROJ="/workspace-SR008.fs2/mikheev-kandy/TD"
PY="${PROJ}/data/envs/td-vlm/bin/python"
EXP_DIR="${PROJ}/experiments/exp05_vlm_normalizer"
INPUTS="${EXP_DIR}/merged_inputs.jsonl"
LOG_DIR="${EXP_DIR}/logs"
mkdir -p "${LOG_DIR}"

TOTAL=$(wc -l < "${INPUTS}")
IFS=',' read -ra GPU_ARR <<< "${GPUS}"
N_SHARDS=${#GPU_ARR[@]}
SLUG="OpenGVLab__InternVL3-8B-hf"
PER=$(( (TOTAL + N_SHARDS - 1) / N_SHARDS ))

PIDS=()
for i in "${!GPU_ARR[@]}"; do
    GPU="${GPU_ARR[i]}"
    START=$(( i * PER ))
    END=$(( START + PER ))
    if [ "$END" -gt "$TOTAL" ]; then END="$TOTAL"; fi
    SHARD_PATH="${EXP_DIR}/shards/shard_${SLUG}_mode${MODE}_${i}.jsonl"
    sed -n "$((START+1)),${END}p" "${INPUTS}" > "${SHARD_PATH}"
    LOG="${LOG_DIR}/${SLUG}_mode${MODE}_gpu${GPU}.log"
    echo "  shard ${i}: ${START}..${END}  GPU=${GPU}  log=${LOG}"
    (
        export CUDA_VISIBLE_DEVICES="${GPU}"
        export HF_HOME="${PROJ}/data/models/hf_cache"
        export TRANSFORMERS_CACHE="${HF_HOME}"
        export HF_HUB_DISABLE_TELEMETRY=1
        export TOKENIZERS_PARALLELISM=false
        unset PYTHONPATH PYTHONSTARTUP
        export PYTHONNOUSERSITE=1
        cd "${PROJ}"
        "${PY}" -m experiments.exp05_vlm_normalizer.run_internvl \
            --mode "${MODE}" --inputs "${SHARD_PATH}" \
            --out-suffix "__shard${i}" \
            > "${LOG}" 2>&1
    ) &
    PIDS+=($!)
done
echo "Launched ${#PIDS[@]} shards: ${PIDS[*]}"
wait
echo "All shards done."

MERGED="${EXP_DIR}/outputs/${SLUG}__mode${MODE}.jsonl"
> "${MERGED}"
for i in "${!GPU_ARR[@]}"; do
    SHARD_OUT="${EXP_DIR}/outputs/${SLUG}__mode${MODE}__shard${i}.jsonl"
    [ -f "${SHARD_OUT}" ] && cat "${SHARD_OUT}" >> "${MERGED}"
done
echo "Merged → ${MERGED}"
wc -l "${MERGED}" 2>/dev/null || true
