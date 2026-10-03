"""Smoke-тест консервативного гибрида."""
from __future__ import annotations

from src.utils import env_setup  # noqa: F401

import json
from pathlib import Path

from src.preprocessing.preprocess_v3 import main


ROOT = Path(__file__).resolve().parents[2]
SMOKE_SRC = ROOT / "data" / "preprocessed_v3_smoke" / "_src"
SMOKE_DST = ROOT / "data" / "preprocessed_v3_smoke" / "Smuts"
SMOKE_IDX = ROOT / "data" / "preprocessed_v3_smoke" / "preproc_index_v3.json"


if __name__ == "__main__":
    SMOKE_DST.mkdir(parents=True, exist_ok=True)
    SMOKE_SRC.mkdir(parents=True, exist_ok=True)
    # подготовка симлинков
    real_src = ROOT / "Smuts"
    needed = ["LE F-43577.JPG", "LE F-198016.JPG", "LE F-113259.JPG",
              "LE F-43558.JPG", "LE F-43576.JPG"]
    for f in SMOKE_SRC.iterdir():
        if f.is_symlink() or f.is_file():
            f.unlink()
    for name in needed:
        target = real_src / name
        if target.exists():
            (SMOKE_SRC / name).symlink_to(target)
    print("Smoke files:", sorted(p.name for p in SMOKE_SRC.iterdir()))

    main(src_dir=SMOKE_SRC, dst_dir=SMOKE_DST, index_path=SMOKE_IDX, force=True)
    print(json.dumps(json.loads(SMOKE_IDX.read_text()), ensure_ascii=False, indent=2))
