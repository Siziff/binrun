"""Препроцессинг v3 — консервативный гибрид (PaddleX-on-page first).

Использует `ConservativeOrientator` из `orient_conservative.py`. Если
PaddleX-on-page уверен (conf ≥ 0.85), берём его решение (как в Эксп. 2).
Иначе включаем гибрид Эксп. 3.

Выход: `data/preprocessed_v3/Smuts/` + `preproc_index_v3.json`.
"""
from __future__ import annotations

from src.utils import env_setup  # noqa: F401

import json
import time
from pathlib import Path

from PIL import Image, ImageOps
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           TextColumn, TimeRemainingColumn)

from src.preprocessing.orient_conservative import ConservativeOrientator
from src.preprocessing.preprocess import resize_max_side, rotate_pil


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "Smuts"
DST_DIR = ROOT / "data" / "preprocessed_v3" / "Smuts"
INDEX_PATH = ROOT / "data" / "preprocessed_v3" / "preproc_index_v3.json"


def main(
    src_dir: Path = SRC_DIR,
    dst_dir: Path = DST_DIR,
    index_path: Path = INDEX_PATH,
    page_conf_threshold: float = 0.85,
    crop_conf_threshold: float = 0.85,
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

    orientator = ConservativeOrientator(
        page_conf_threshold=page_conf_threshold,
        crop_conf_threshold=crop_conf_threshold,
    )

    t0 = time.perf_counter()
    src_counts = {"paddlex_page": 0, "paddlex_crop": 0,
                  "doctr_fallback": 0, "doctr_nolabel": 0, "error": 0}
    n_rot = 0

    with Progress(
        TextColumn("[bold blue]preproc-v3"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[note]}"),
    ) as progress:
        task = progress.add_task("preproc-v3", total=len(todo), note="")
        for src_p in todo:
            try:
                decision = orientator.decide(src_p)
            except Exception as e:
                print(f"  conservative orient FAILED on {src_p.name}: {e}", flush=True)
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
                            note=f"rot={n_rot} crop={src_counts['paddlex_crop']} "
                                 f"fb={src_counts['doctr_fallback']+src_counts['doctr_nolabel']}")

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
    p.add_argument("--page-conf-threshold", type=float, default=0.85)
    p.add_argument("--crop-conf-threshold", type=float, default=0.85)
    p.add_argument("--src-dir", type=str, default=None,
                   help="Источник изображений (по умолчанию Smuts/).")
    p.add_argument("--dst-dir", type=str, default=None,
                   help="Куда писать препроцессед JPG.")
    p.add_argument("--index-path", type=str, default=None,
                   help="Путь к preproc_index JSON.")
    args = p.parse_args()
    kwargs = dict(
        force=args.force, max_side=args.max_side,
        page_conf_threshold=args.page_conf_threshold,
        crop_conf_threshold=args.crop_conf_threshold,
    )
    if args.src_dir:
        kwargs["src_dir"] = Path(args.src_dir)
    if args.dst_dir:
        kwargs["dst_dir"] = Path(args.dst_dir)
    if args.index_path:
        kwargs["index_path"] = Path(args.index_path)
    main(**kwargs)
