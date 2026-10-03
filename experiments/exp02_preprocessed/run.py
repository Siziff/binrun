"""Эксперимент 2: тот же набор движков, но на препроцессед-изображениях.

Препроцессинг: EXIF transpose → page-level orientation (PaddleX
PP-LCNet_x1_0_doc_ori) → resize до max_side=3000.

Картинки лежат в `data/preprocessed/Smuts/`, индекс — в
`data/preprocessed/preproc_index.json`.

Под капотом — тот же runner, что и в Эксп. 1, с другими путями
к изображениям и выгрузкам.
"""
from __future__ import annotations

# env_setup первой строкой.
from src.utils import env_setup  # noqa: F401

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PREP_DIR = ROOT / "data" / "preprocessed" / "Smuts"
OUT_DIR = ROOT / "experiments" / "exp02_preprocessed" / "outputs"

# Прокидываем CLI-аргументы в runner Эксп. 1, заменяя пути по умолчанию.
extra = sys.argv[1:]
sys.argv = [
    "run",
    "--image-dir", str(PREP_DIR),
    "--out", str(OUT_DIR),
    *extra,
]

from experiments.exp01_ocr_baseline.run import main  # noqa: E402

if __name__ == "__main__":
    main()
