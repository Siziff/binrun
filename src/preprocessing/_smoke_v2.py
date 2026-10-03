"""Smoke-тест гибридной ориентации на нескольких файлах."""
from __future__ import annotations

from src.utils import env_setup  # noqa: F401

import json
from pathlib import Path

from src.preprocessing.preprocess_v2 import main


ROOT = Path(__file__).resolve().parents[2]
SMOKE_SRC = ROOT / "data" / "preprocessed_v2_smoke" / "_src"
SMOKE_DST = ROOT / "data" / "preprocessed_v2_smoke" / "Smuts"
SMOKE_IDX = ROOT / "data" / "preprocessed_v2_smoke" / "preproc_index_v2.json"

if __name__ == "__main__":
    SMOKE_DST.mkdir(parents=True, exist_ok=True)
    main(src_dir=SMOKE_SRC, dst_dir=SMOKE_DST, index_path=SMOKE_IDX, force=True)
    print(json.dumps(json.loads(SMOKE_IDX.read_text()), ensure_ascii=False, indent=2))
