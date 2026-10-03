"""Сохраняет визуализацию всех боксов Grounding DINO для отладки."""
from __future__ import annotations

from src.utils import env_setup  # noqa: F401

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "preprocessed_v2_smoke" / "dino_debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main(paths: list[Path], prompt: str, box_thr: float, text_thr: float) -> None:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base").to(device).eval()

    for p in paths:
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        inputs = proc(images=img, text=prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        res = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=box_thr, text_threshold=text_thr,
            target_sizes=[img.size[::-1]])[0]
        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        labels = res.get("labels", res.get("text_labels", [""] * len(boxes)))

        print(f"\n{p.name}: {len(boxes)} boxes")
        for b, s, lab in zip(boxes, scores, labels):
            x0, y0, x1, y1 = b.tolist()
            W, H = img.size
            print(f"  score={s:.3f} label={lab!r} "
                  f"bbox=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}) "
                  f"frac={((x1-x0)*(y1-y0))/(W*H):.3f}")

        # Визуализация.
        vis = img.copy().resize((img.size[0] // 4, img.size[1] // 4))
        draw = ImageDraw.Draw(vis)
        for b, s in zip(boxes, scores):
            x0, y0, x1, y1 = (b / 4).tolist()
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            draw.text((x0 + 2, y0 + 2), f"{s:.2f}", fill="red")
        vis.save(OUT_DIR / f"{p.stem}_dino.jpg", quality=85)
    print(f"\nVis saved → {OUT_DIR}")


if __name__ == "__main__":
    SMOKE = ROOT / "Smuts"
    files = [SMOKE / "LE F-43577.JPG", SMOKE / "LE F-198016.JPG", SMOKE / "LE F-113259.JPG"]
    # Разные настройки, чтобы посмотреть.
    prompt = (sys.argv[1] if len(sys.argv) > 1
              else "a paper label. a herbarium label. a printed text label.")
    box_thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    text_thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.20
    print(f"prompt={prompt!r} box_thr={box_thr} text_thr={text_thr}")
    main(files, prompt, box_thr, text_thr)
