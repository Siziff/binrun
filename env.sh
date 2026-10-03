#!/usr/bin/env bash
# Единый источник истины для всех путей кэша/моделей/tmp.
# Sourcing: `source env.sh` ИЛИ `. ./env.sh` перед запуском любых OCR-инструментов.
#
# Если что-то меняется — синхронизировать с `src/utils/env_setup.py`.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT

# --- Активируем проектный conda env (изолирован от base) ---
# Создан 2026-06: base /home/user/conda потерял OCR-зависимости из-за чужой
# активности на shared-узле. С тех пор всё ставим только сюда.
#
# Два env:
#   td-ocr — для Эксп.1–4 (Tesseract / PaddleOCR / EasyOCR / docTR / Surya / TrOCR),
#            требует transformers 4.49 (Surya 0.14.6).
#   td-vlm — для Эксп.5+ (Qwen3-VL / InternVL / PaddleOCR-VL / GOT-OCR2.0),
#            требует transformers >= 4.57 (git-главу для Qwen3-VL).
#
# По умолчанию активируем td-ocr. Чтобы перейти на td-vlm:
#   export TD_PREF=vlm && source env.sh
TD_PREF="${TD_PREF:-ocr}"
TD_ENV="${PROJECT_ROOT}/data/envs/td-${TD_PREF}"
if [ -d "${TD_ENV}/bin" ]; then
    # Удаляем любые наши env-пути из PATH (на случай повторного source).
    PATH_CLEAN=$(echo "${PATH}" | tr ':' '\n' \
        | grep -v "${PROJECT_ROOT}/data/envs/td-" \
        | paste -sd ':' -)
    export PATH="${TD_ENV}/bin:${PATH_CLEAN}"
    export CONDA_PREFIX="${TD_ENV}"
    # Очищаем кэш bash, иначе `python` остаётся резолвиться в старый путь.
    hash -r 2>/dev/null || true
fi

# VSCode выставляет PYTHONSTARTUP=…/pythonrc.py, который перенаправляет
# `python` на base-интерпретатор и ломает изоляцию env. Сбрасываем.
unset PYTHONSTARTUP

# Полностью изолируем td-* python от base /home/user/conda:
#   - убираем PYTHONPATH (может содержать base site-packages);
#   - выключаем user-сайт-пакеты (`/home/jovyan/zakharov/.local`).
unset PYTHONPATH
export PYTHONNOUSERSITE=1

MODELS_DIR="${PROJECT_ROOT}/data/models"
TMP_DIR="${PROJECT_ROOT}/data/tmp"
CACHE_DIR="${PROJECT_ROOT}/data/cache"

mkdir -p \
    "${MODELS_DIR}/hf_cache" \
    "${MODELS_DIR}/easyocr" \
    "${MODELS_DIR}/doctr" \
    "${MODELS_DIR}/paddleocr/paddle_home" \
    "${MODELS_DIR}/surya" \
    "${MODELS_DIR}/torch_hub" \
    "${TMP_DIR}" \
    "${CACHE_DIR}/pip" \
    "${CACHE_DIR}/matplotlib"

# --- Hugging Face ---
export HF_HOME="${MODELS_DIR}/hf_cache"
export HF_HUB_CACHE="${MODELS_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${MODELS_DIR}/hf_cache"
export HF_DATASETS_CACHE="${MODELS_DIR}/hf_cache/datasets"
export SENTENCE_TRANSFORMERS_HOME="${MODELS_DIR}/hf_cache/sentence_transformers"

# --- PyTorch ---
export TORCH_HOME="${MODELS_DIR}/torch_hub"
export TORCH_HUB="${MODELS_DIR}/torch_hub"
export XDG_CACHE_HOME="${CACHE_DIR}"

# --- EasyOCR / docTR / Paddle / Surya ---
export EASYOCR_MODULE_PATH="${MODELS_DIR}/easyocr"
export DOCTR_CACHE_DIR="${MODELS_DIR}/doctr"
export PADDLE_PDX_CACHE_HOME="${MODELS_DIR}/paddleocr"
export PADDLE_HOME="${MODELS_DIR}/paddleocr/paddle_home"
export MODEL_CACHE_DIR="${MODELS_DIR}/surya"
export DATALAB_CACHE_DIR="${MODELS_DIR}/surya"

# --- TMP / системные ---
export TMPDIR="${TMP_DIR}"
export TMP="${TMP_DIR}"
export TEMP="${TMP_DIR}"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export MPLCONFIGDIR="${CACHE_DIR}/matplotlib"

# --- Телеметрия отключена ---
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export ANONYMIZED_TELEMETRY=False
export TOKENIZERS_PARALLELISM=false

# --- GPU pinning ---
# GPU 1 по умолчанию (GPU 0 часто занят на shared-узле).
# Переопределяется внешним CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

echo "[env.sh] project root: ${PROJECT_ROOT}"
echo "[env.sh] models       : ${MODELS_DIR}"
echo "[env.sh] tmp          : ${TMP_DIR}"
echo "[env.sh] env          : td-${TD_PREF} (${TD_ENV})"
if [ -x "${TD_ENV}/bin/python" ]; then
    echo "[env.sh] python       : ${TD_ENV}/bin/python"
fi
echo "[env.sh] python       : $(which python 2>/dev/null || echo not-found)"
