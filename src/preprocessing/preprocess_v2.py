"""Препроцессинг v2 — гибридная ориентация (Grounding DINO + PaddleX + docTR).

Отличия от `preprocess.py`:
    * этап 2 заменён на `HybridOrientator` из `orient_hybrid.py`;
    * каждое решение по ориентации логируется с указанием источника
      (paddlex_crop / paddlex_page / doctr_fallback / no_label);
    * пишем в `data/preprocessed_v2/Smuts/` и `preproc_index_v2.json`.

Используется как drop-in замена для последующих экспериментов.
"""
from __future__ import annotations

from src.utils import env_setup  # noqa: F401

import json
import time
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageOps
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           TextColumn, TimeRemainingColumn)

from src.preprocessing.orient_hybrid import HybridOrientator
from src.preprocessing.preprocess import resize_max_side, rotate_pil


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "Smuts"
DST_DIR = ROOT / "data" / "preprocessed_v2" / "Smuts"
INDEX_PATH = ROOT / "data" / "preprocessed_v2" / "preproc_index_v2.json"


def main(
    src_dir: Path = SRC_DIR,
    dst_dir: Path = DST_DIR,
    index_path: Path = INDEX_PATH,
    paddle_conf_threshold: float = 0.85,
    max_side: int = 3000,
    force: bool = False,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(list(src_dir.glob("*.JPG")) + list(src_dir.glob("*.jpg")))
    print(f"Found {len(images)} source images.", flush=True)

    index: dict[str, dict] = {}
    if index_path.exists() and not force:
        index = json.loads(index_path.read_text(encoding="utf-8"))

    todo: list[Path] = []
    for p in images:
        out_p = dst_dir / p.name
        if not force and out_p.exists() and p.name in index:
            continue
        todo.append(p)
    print(f"Need to process: {len(todo)} "
          f"(skipping {len(images) - len(todo)} already done).", flush=True)
    if not todo:
        return

    orientator = HybridOrientator(paddle_conf_threshold=paddle_conf_threshold)

    t0 = time.perf_counter()
    src_counts = {"paddlex_crop": 0, "paddlex_page": 0, "doctr_fallback": 0}
    n_rot = 0

    with Progress(
        TextColumn("[bold blue]preproc-v2"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[note]}"),
    ) as progress:
        task = progress.add_task("preproc-v2", total=len(todo), note="")
        for src_p in todo:
            try:
                decision = orientator.decide(src_p)
            except Exception as e:
                print(f"  hybrid orient FAILED on {src_p.name}: {e}", flush=True)
                from src.preprocessing.orient_hybrid import OrientDecision
                decision = OrientDecision(
                    angle=0, confidence=0.0, source="error",
                    paddlex_angle=0, paddlex_conf=0.0, doctr_scores={},
                    crop_bbox=None, elapsed_ms=0.0,
                )

            try:
                img = ImageOps.exif_transpose(Image.open(src_p)).convert("RGB")
            except Exception as e:
                print(f"  open FAILED on {src_p.name}: {e}", flush=True)
                progress.update(task, advance=1, note=f"skipped {src_p.name}")
                continue
            original_size = img.size
            if decision.angle != 0:
                img = rotate_pil(img, decision.angle)
                n_rot += 1
            img = resize_max_side(img, max_side)
            new_size = img.size

            out_p = dst_dir / src_p.name
            img.save(out_p, format="JPEG", quality=92, optimize=True)

            src_counts[decision.source] = src_counts.get(decision.source, 0) + 1
            index[src_p.name] = {
                "original_size": original_size,
                "new_size": new_size,
                "rotation_applied": decision.angle,
                "source": decision.source,
                "paddlex_angle": decision.paddlex_angle,
                "paddlex_conf": round(decision.paddlex_conf, 4),
                "doctr_scores": {str(k): round(v, 4)
                                 for k, v in decision.doctr_scores.items()},
                "crop_bbox": decision.crop_bbox,
                "elapsed_ms": round(decision.elapsed_ms, 1),
            }
            progress.update(task, advance=1,
                            note=f"rot={n_rot} fb={src_counts.get('doctr_fallback', 0)}")

            # Периодически сохраняем индекс.
            if (progress.tasks[0].completed % 10) == 0:
                index_path.write_text(
                    json.dumps(index, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/len(todo):.2f} s/img).")
    print(f"Sources: {src_counts}")
    print(f"Total rotated: {n_rot} of {len(todo)}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-side", type=int, default=3000)
    p.add_argument("--paddle-conf-threshold", type=float, default=0.85)
    p.add_argument("--limit", type=int, default=None,
                   help="ограничить обработку первыми N файлами (для smoke-теста)")
    args = p.parse_args()

    if args.limit is not None:
        # запустим вручную с обрезкой
        images = sorted(list(SRC_DIR.glob("*.JPG")) + list(SRC_DIR.glob("*.jpg")))[: args.limit]
        SMOKE_DIR = ROOT / "data" / "preprocessed_v2_smoke" / "Smuts"
        SMOKE_IDX = ROOT / "data" / "preprocessed_v2_smoke" / "preproc_index_v2.json"
        SMOKE_DIR.mkdir(parents=True, exist_ok=True)
        # Импровизированный вариант с подмененным src
        TMP_SRC = ROOT / "data" / "preprocessed_v2_smoke" / "_src"
        TMP_SRC.mkdir(parents=True, exist_ok=True)
        # очистка симлинков от прошлых запусков
        for f in TMP_SRC.iterdir():
            if f.is_symlink() or f.is_file():
                f.unlink()
        for src in images:
            (TMP_SRC / src.name).symlink_to(src)
        main(src_dir=TMP_SRC, dst_dir=SMOKE_DIR, index_path=SMOKE_IDX,
             paddle_conf_threshold=args.paddle_conf_threshold,
             max_side=args.max_side, force=args.force)
    else:
        main(force=args.force, max_side=args.max_side,
             paddle_conf_threshold=args.paddle_conf_threshold)
