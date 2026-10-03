"""Препроцессинг гербарных сканов перед OCR.

Реализованные этапы:
    1. EXIF transpose                — нормализация по EXIF Orientation tag.
    2. Page-level orientation        — классификатор `PP-LCNet_x1_0_doc_ori`
                                       (PaddleX), 4 класса {0, 90, 180, 270}.
                                       Применяется только при confidence >= τ
                                       (по умолчанию τ=0.70).
    3. Resize до max_side=3000       — ускоряет все движки в 2–4× почти без
                                       потери качества (этикетки занимают
                                       много меньше всего листа).

Результат сохраняется в `data/preprocessed/Smuts/<имя.jpg>` + индекс
`data/preprocessed/preproc_index.json` с записанными для каждого файла
{orientation, rotation_applied, confidence, original_size, new_size}.
"""
from __future__ import annotations

from src.utils import env_setup  # noqa: F401

import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           TextColumn, TimeRemainingColumn)

# paddle PIR-executor выключаем (есть баг на PP-OCRv5 CPU, не влияет на ori-classifier)
os.environ.setdefault("FLAGS_enable_pir_in_executor", "false")
os.environ.setdefault("FLAGS_use_mkldnn", "false")


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "Smuts"
DST_DIR = ROOT / "data" / "preprocessed" / "Smuts"
INDEX_PATH = ROOT / "data" / "preprocessed" / "preproc_index.json"


class Orientator:
    """Лёгкий wrapper над PaddleX PP-LCNet_x1_0_doc_ori."""

    def __init__(self) -> None:
        from paddlex import create_model
        self._model = create_model("PP-LCNet_x1_0_doc_ori")

    def predict_batch(self, paths: list[Path]) -> list[tuple[int, float]]:
        """Возвращает [(angle, confidence), ...]. angle ∈ {0,90,180,270}."""
        out: list[tuple[int, float]] = []
        for r in self._model.predict([str(p) for p in paths]):
            angle = int(r["label_names"][0])
            conf = float(r["scores"][0])
            out.append((angle, conf))
        return out


def rotate_pil(img: Image.Image, angle: int) -> Image.Image:
    """Поворачивает картинку на N° по часовой стрелке (стандарт PIL — против)."""
    if angle == 0:
        return img
    # PIL.rotate против часовой, нам нужен «по часовой».
    # Если classifier говорит «270», значит «текст лежит на левом боку,
    # надо повернуть на 270° по часовой» = 90° против часовой.
    pil_angle = -angle  # против часовой → по часовой
    return img.rotate(pil_angle, expand=True, resample=Image.BICUBIC)


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    ratio = max_side / max(img.size)
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    return img.resize(new_size, Image.LANCZOS)


def preprocess_one(
    img_path: Path,
    out_path: Path,
    angle: int,
    conf: float,
    *,
    rotation_conf_threshold: float = 0.70,
    max_side: int = 3000,
    jpeg_quality: int = 92,
) -> dict:
    """Применяет к одной картинке EXIF/rotate/resize и сохраняет JPEG.

    Возвращает dict с метаданными.
    """
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)  # учли EXIF Orientation
    img = img.convert("RGB")
    original_size = img.size  # (W, H)

    rotation_applied = 0
    if angle != 0 and conf >= rotation_conf_threshold:
        img = rotate_pil(img, angle)
        rotation_applied = angle

    img = resize_max_side(img, max_side)
    new_size = img.size

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)
    return {
        "original_size": original_size,
        "new_size": new_size,
        "orientation_predicted": angle,
        "orientation_confidence": conf,
        "rotation_applied": rotation_applied,
    }


def main(
    src_dir: Path = SRC_DIR,
    dst_dir: Path = DST_DIR,
    index_path: Path = INDEX_PATH,
    rotation_conf_threshold: float = 0.70,
    max_side: int = 3000,
    batch_size: int = 16,
    force: bool = False,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(list(src_dir.glob("*.JPG")) + list(src_dir.glob("*.jpg")))
    print(f"Found {len(images)} source images.", flush=True)

    todo = []
    for p in images:
        out_p = dst_dir / p.name
        if not force and out_p.exists():
            continue
        todo.append((p, out_p))
    print(f"Need to process: {len(todo)} (skipping {len(images)-len(todo)} already done).",
          flush=True)

    index: dict[str, dict] = {}
    if index_path.exists() and not force:
        index = json.loads(index_path.read_text(encoding="utf-8"))

    if not todo:
        print("Nothing to do.")
        return

    print("Loading PP-LCNet_x1_0_doc_ori …", flush=True)
    orientator = Orientator()
    t0 = time.perf_counter()

    with Progress(
        TextColumn("[bold blue]preprocess"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[note]}"),
    ) as progress:
        task = progress.add_task("preprocess", total=len(todo), note="")
        for batch_start in range(0, len(todo), batch_size):
            batch = todo[batch_start: batch_start + batch_size]
            paths = [p for p, _ in batch]
            try:
                preds = orientator.predict_batch(paths)
            except Exception as e:
                # На случай отдельных битых файлов — обходим по одному.
                preds = []
                for p in paths:
                    try:
                        preds.extend(orientator.predict_batch([p]))
                    except Exception as e2:
                        preds.append((0, 0.0))
                        print(f"  orient FAILED on {p.name}: {e2}", flush=True)

            n_rot = 0
            for (src_p, out_p), (angle, conf) in zip(batch, preds):
                meta = preprocess_one(
                    src_p, out_p, angle, conf,
                    rotation_conf_threshold=rotation_conf_threshold,
                    max_side=max_side,
                )
                index[src_p.name] = meta
                if meta["rotation_applied"]:
                    n_rot += 1
                progress.update(task, advance=1, note=f"rot+={n_rot}")

            # Периодически сохраняем индекс.
            index_path.write_text(
                json.dumps(index, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s. Index → {index_path}")
    n_rot_total = sum(1 for m in index.values() if m["rotation_applied"])
    print(f"Total rotated: {n_rot_total} of {len(index)}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-side", type=int, default=3000)
    p.add_argument("--rotation-threshold", type=float, default=0.70)
    args = p.parse_args()
    main(force=args.force, max_side=args.max_side,
         rotation_conf_threshold=args.rotation_threshold)
