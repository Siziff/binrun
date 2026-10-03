"""Эксперимент 3: все 7 движков на гибридно ориентированном корпусе.

Препроцессинг: EXIF → Grounding DINO label crop → PaddleX-on-crop / docTR×4
fallback → resize до max_side=3000. См. `src/preprocessing/preprocess_v2.py`.

Картинки: `data/preprocessed_v2/Smuts/`.
По умолчанию запускает все движки из `ALL_ENGINES`.
"""
from __future__ import annotations

from src.utils import env_setup  # noqa: F401

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PREP_DIR = ROOT / "data" / "preprocessed_v2" / "Smuts"
OUT_DIR = ROOT / "experiments" / "exp03_hybrid_orient" / "outputs"


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
