"""Label detection через Grounding DINO (zero-shot).

Идея: задаём несколько концептов того, что считаем "этикеткой",
и Grounding DINO выдаёт bbox-ы. Мы выбираем лучший по эвристике:
    1. фильтр по площади (1% ≤ area ≤ 90% — слишком мелкие шумят,
       слишком большие — это сам лист);
    2. предпочитаем bbox с явно "label-подобными" пропорциями
       (aspect ratio не сильно меньше 1.5);
    3. при tie — берём с наибольшим score.

Возвращаем **один** crop (плюс fallback к полному изображению).
"""
from __future__ import annotations

from src.utils import env_setup  # noqa: F401

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image


# Несколько концептов через ". "-separator (так требует Grounding DINO).
DEFAULT_PROMPT = "specimen label. printed text label. paper label. handwritten label."

MIN_AREA_PCT = 0.5   # отсекаем совсем мелкие срабатывания (шум)
MAX_AREA_PCT = 92.0  # отсекаем "весь лист"


@dataclass
class LabelBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    text_label: str

    @property
    def w(self) -> float: return self.x2 - self.x1
    @property
    def h(self) -> float: return self.y2 - self.y1
    @property
    def area(self) -> float: return self.w * self.h
    @property
    def aspect(self) -> float:
        # ≥ 1, безразлично к ориентации
        return max(self.w, self.h) / max(min(self.w, self.h), 1e-6)

    def pad(self, px: int, img_w: int, img_h: int) -> "LabelBox":
        return LabelBox(
            x1=max(0, self.x1 - px),
            y1=max(0, self.y1 - px),
            x2=min(img_w, self.x2 + px),
            y2=min(img_h, self.y2 + px),
            score=self.score,
            text_label=self.text_label,
        )

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


class LabelDetector:
    """Grounding DINO как zero-shot детектор этикеток."""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base",
                 device: Optional[str] = None,
                 prompt: str = DEFAULT_PROMPT,
                 box_threshold: float = 0.25):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.prompt = prompt
        self.box_threshold = box_threshold
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = (AutoModelForZeroShotObjectDetection
                       .from_pretrained(model_id).to(device).eval())

    @torch.inference_mode()
    def detect(self, image: Image.Image) -> list[LabelBox]:
        """Возвращает список найденных bbox-ов после фильтрации по площади."""
        inputs = self._processor(images=image, text=self.prompt,
                                 return_tensors="pt").to(self.device)
        outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            threshold=self.box_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        img_area = image.size[0] * image.size[1]
        boxes: list[LabelBox] = []
        for box, score, label in zip(
            results["boxes"], results["scores"], results["text_labels"],
        ):
            x1, y1, x2, y2 = box.tolist()
            b = LabelBox(x1=x1, y1=y1, x2=x2, y2=y2,
                         score=float(score), text_label=str(label))
            area_pct = 100 * b.area / img_area
            if not (MIN_AREA_PCT <= area_pct <= MAX_AREA_PCT):
                continue
            boxes.append(b)
        return boxes

    def pick_best(self, boxes: list[LabelBox], image_size: tuple[int, int]) -> Optional[LabelBox]:
        """Эвристика выбора главного label-bbox.

        - предпочитаем bbox с явно label-подобным aspect (>1.5),
        - и большой площадью (главный сценарий),
        - а среди равных — с большим score.
        """
        if not boxes:
            return None
        # сортируем: сначала "label-like" + большая площадь, потом всё остальное
        def score_box(b: LabelBox) -> float:
            label_like = 1.0 if b.aspect >= 1.5 else 0.5
            area_score = b.area / (image_size[0] * image_size[1])
            return label_like * 0.5 + area_score * 0.5 + b.score * 0.2
        return max(boxes, key=score_box)

    def detect_and_crop(self, image: Image.Image, pad_pct: float = 3.0
                        ) -> tuple[Image.Image, Optional[LabelBox]]:
        """Высокоуровневый API: вернуть crop этикетки + bbox, либо целую картинку.

        Padding `pad_pct`% от меньшей стороны для перестраховки от обрезки.
        """
        boxes = self.detect(image)
        best = self.pick_best(boxes, image.size)
        if best is None:
            return image, None
        pad_px = int(min(image.size) * pad_pct / 100.0)
        padded = best.pad(pad_px, image.size[0], image.size[1])
        crop = image.crop(padded.to_xyxy())
        return crop, best


def main():
    """CLI-проверка на нескольких файлах."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    args = ap.parse_args()
    det = LabelDetector()
    for p in args.images:
        img = Image.open(p).convert("RGB")
        t0 = time.perf_counter()
        boxes = det.detect(img)
        best = det.pick_best(boxes, img.size)
        dt = time.perf_counter() - t0
        print(f"\n{p}  ({img.size[0]}×{img.size[1]})  {dt:.2f}s  found {len(boxes)} candidates")
        for b in boxes:
            area_pct = 100 * b.area / (img.size[0] * img.size[1])
            marker = " ⭐" if b is best else ""
            print(f"  bbox=({b.x1:.0f},{b.y1:.0f})-({b.x2:.0f},{b.y2:.0f})  "
                  f"area={area_pct:.1f}% aspect={b.aspect:.2f}  "
                  f"score={b.score:.2f} '{b.text_label}'{marker}")


if __name__ == "__main__":
    main()
