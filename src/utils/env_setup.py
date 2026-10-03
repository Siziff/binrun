"""Централизованная настройка путей для всех кэшей и временных файлов.

Импортируется ПЕРВОЙ строкой во всех скриптах проекта (`import src.utils.env_setup`).
Гарантирует, что ни одна модель, ни один кэш, ни один tmp-файл не уходит вне
корня проекта `/workspace-SR008.fs2/mikheev-kandy/TD`.

Покрытые библиотеки/инструменты:
    - Hugging Face (transformers, datasets, hub)
    - PyTorch (torch.hub, torchvision)
    - EasyOCR
    - docTR (mindee)
    - PaddleOCR / PaddleX
    - Surya (datalab)
    - системные TMPDIR / pip-cache / matplotlib

Изменять только этот файл, чтобы не размножать пути.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Корень проекта определяем относительно расположения этого файла:
#   src/utils/env_setup.py  →  parents[2] = project root.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "data" / "models"
TMP_DIR = PROJECT_ROOT / "data" / "tmp"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Создаём все каталоги заранее, чтобы библиотеки не падали при первом вызове.
for d in (
    MODELS_DIR,
    TMP_DIR,
    CACHE_DIR,
    MODELS_DIR / "hf_cache",
    MODELS_DIR / "easyocr",
    MODELS_DIR / "doctr",
    MODELS_DIR / "paddleocr",
    MODELS_DIR / "surya",
    MODELS_DIR / "torch_hub",
    CACHE_DIR / "matplotlib",
    CACHE_DIR / "pip",
):
    d.mkdir(parents=True, exist_ok=True)


def _set(key: str, value: str | Path) -> None:
    """Устанавливает env-переменную только если она ещё не задана извне."""
    os.environ.setdefault(key, str(value))


# --- Hugging Face ---------------------------------------------------------
_set("HF_HOME", MODELS_DIR / "hf_cache")
_set("HF_HUB_CACHE", MODELS_DIR / "hf_cache")
_set("TRANSFORMERS_CACHE", MODELS_DIR / "hf_cache")
_set("SENTENCE_TRANSFORMERS_HOME", MODELS_DIR / "hf_cache" / "sentence_transformers")
_set("HF_DATASETS_CACHE", MODELS_DIR / "hf_cache" / "datasets")

# --- PyTorch / torchvision ------------------------------------------------
_set("TORCH_HOME", MODELS_DIR / "torch_hub")
_set("TORCH_HUB", MODELS_DIR / "torch_hub")
_set("XDG_CACHE_HOME", CACHE_DIR)  # многие либы наследуют XDG

# --- EasyOCR --------------------------------------------------------------
_set("EASYOCR_MODULE_PATH", MODELS_DIR / "easyocr")

# --- docTR (mindee) -------------------------------------------------------
_set("DOCTR_CACHE_DIR", MODELS_DIR / "doctr")

# --- PaddleOCR / PaddleX --------------------------------------------------
_set("PADDLE_PDX_CACHE_HOME", MODELS_DIR / "paddleocr")
# Старый paddleocr-2.x опирался на ~/.paddleocr; новый 3.x — на PADDLE_PDX_CACHE_HOME.
# Дополнительно изолируем PADDLE_HOME для C++-ядра:
_set("PADDLE_HOME", MODELS_DIR / "paddleocr" / "paddle_home")

# --- Surya (datalab) ------------------------------------------------------
# Surya читает settings через pydantic BaseSettings; ключ MODEL_CACHE_DIR.
_set("MODEL_CACHE_DIR", MODELS_DIR / "surya")
# Доп. перехватываем XDG-cache на случай datalab-utils.
_set("DATALAB_CACHE_DIR", MODELS_DIR / "surya")

# --- TMP / системные ------------------------------------------------------
_set("TMPDIR", TMP_DIR)
_set("TMP", TMP_DIR)
_set("TEMP", TMP_DIR)
_set("PIP_CACHE_DIR", CACHE_DIR / "pip")
_set("MPLCONFIGDIR", CACHE_DIR / "matplotlib")

# --- Отключаем телеметрию и опросы ----------------------------------------
_set("HF_HUB_DISABLE_TELEMETRY", "1")
_set("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
_set("DO_NOT_TRACK", "1")
_set("ANONYMIZED_TELEMETRY", "False")
_set("TOKENIZERS_PARALLELISM", "false")  # избегаем шумные warnings

# --- GPU pinning ----------------------------------------------------------
# По умолчанию используем GPU 1, т.к. GPU 0 часто занят другими процессами
# на shared-узле. Внешний `CUDA_VISIBLE_DEVICES` имеет приоритет.
_set("CUDA_VISIBLE_DEVICES", "1")


# --------------------------------------------------------------------------
# Жёстко зашитые пути в сторонних библиотеках, которые нельзя переопределить
# переменной окружения. Подменяем симлинком на каталог внутри проекта.
# --------------------------------------------------------------------------
_HARDCODED_LINKS: list[tuple[Path, Path]] = [
    # paddle.dataset.common: HOME = os.path.expanduser('~'); → ~/.cache/paddle/dataset
    (
        Path.home() / ".cache" / "paddle" / "dataset",
        MODELS_DIR / "paddleocr" / "paddle_dataset",
    ),
]


def _ensure_symlinks() -> None:
    for link, target in _HARDCODED_LINKS:
        target.mkdir(parents=True, exist_ok=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            if link.is_symlink():
                if link.resolve() == target.resolve():
                    continue
                link.unlink()
            elif link.exists():
                # Реальная папка: удаляем, только если она пустая.
                if not any(link.iterdir()):
                    link.rmdir()
                else:
                    # Не трогаем непустые директории, чтобы ничего не потерять.
                    continue
            link.symlink_to(target, target_is_directory=True)
        except (OSError, PermissionError):
            # На read-only ФС/без прав — молча игнорируем, кэш не критичен.
            pass


_ensure_symlinks()


def report() -> None:
    """Печатает текущую конфигурацию путей."""
    keys = [
        "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
        "TORCH_HOME", "EASYOCR_MODULE_PATH",
        "DOCTR_CACHE_DIR", "PADDLE_PDX_CACHE_HOME", "PADDLE_HOME",
        "MODEL_CACHE_DIR", "DATALAB_CACHE_DIR",
        "TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR",
    ]
    width = max(len(k) for k in keys)
    print(f"Project root: {PROJECT_ROOT}")
    for k in keys:
        print(f"  {k:<{width}} = {os.environ.get(k)}")


if __name__ == "__main__":
    report()
