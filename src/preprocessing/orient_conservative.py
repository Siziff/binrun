"""Консервативный гибрид: PaddleX-on-page first, гибрид только при неуверенности.

Идея (вариант B из развилки Эксп. 3):
  1. Запустить PaddleX `PP-LCNet_x1_0_doc_ori` на ПОЛНОМ листе (как в Эксп. 2).
  2. Если confidence ≥ `page_conf_threshold` (по умолчанию 0.85) — взять
     это решение, выходим. Большинство файлов с очевидной ориентацией
     уйдут по этой ветке (в Эксп. 2 порог 0.70 пропускал 389/394 файла,
     порог 0.85 пропускает примерно столько же).
  3. Иначе — запускаем гибрид Эксп. 3: Grounding DINO → PaddleX-on-crop →
     docTR×4 fallback. Точечно правим только «трудные» случаи.

Возвращаемое решение применяется к полному листу. Кроп этикетки используется
исключительно как оракул ориентации.
"""
from __future__ import annotations

from src.utils import env_setup  # noqa: F401

import time
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from src.preprocessing.orient_hybrid import (
    DocTRRotationFallback,
    GroundingDinoLabelLocator,
    OrientDecision,
    PaddleXCropOrient,
)


class ConservativeOrientator:
    """PaddleX-on-page first; гибрид только при низкой уверенности."""

    def __init__(
        self,
        page_conf_threshold: float = 0.85,
        crop_conf_threshold: float = 0.85,
        gdino_model_id: str = "IDEA-Research/grounding-dino-base",
    ) -> None:
        self.page_conf_threshold = page_conf_threshold
        self.crop_conf_threshold = crop_conf_threshold

        print("[Conservative] loading PaddleX page+crop orient …", flush=True)
        self.paddle = PaddleXCropOrient()  # один и тот же классификатор

        print("[Conservative] loading Grounding DINO …", flush=True)
        self.locator = GroundingDinoLabelLocator(model_id=gdino_model_id)

        print("[Conservative] loading docTR fallback …", flush=True)
        self.doctr = DocTRRotationFallback()

    def decide(self, img_path: Path) -> OrientDecision:
        t0 = time.perf_counter()
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img).convert("RGB")

        # 1) PaddleX-on-page (как в Эксп. 2).
        try:
            page_angle, page_conf = self.paddle.predict(img)
        except Exception as e:
            print(f"  paddle-on-page failed on {img_path.name}: {e}", flush=True)
            page_angle, page_conf = 0, 0.0

        if page_conf >= self.page_conf_threshold:
            return OrientDecision(
                angle=page_angle,
                confidence=page_conf,
                source="paddlex_page",
                paddlex_angle=page_angle,
                paddlex_conf=page_conf,
                doctr_scores={},
                crop_bbox=None,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        # 2) Неуверенность → гибрид: DINO crop → PaddleX-on-crop → docTR×4.
        bbox = self.locator.locate(img)
        have_label = bbox is not None
        crop = img.crop(bbox) if have_label else img

        crop_angle, crop_conf = 0, 0.0
        if have_label:
            try:
                crop_angle, crop_conf = self.paddle.predict(crop)
            except Exception as e:
                print(f"  paddle-on-crop failed on {img_path.name}: {e}", flush=True)

            if crop_conf >= self.crop_conf_threshold:
                return OrientDecision(
                    angle=crop_angle,
                    confidence=crop_conf,
                    source="paddlex_crop",
                    paddlex_angle=crop_angle,
                    paddlex_conf=crop_conf,
                    doctr_scores={},
                    crop_bbox=bbox,
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                )

        # 3) docTR×4 на кропе (или странице, если бокса нет).
        scores = self.doctr.predict(crop)
        best_angle = max(
            scores.keys(),
            key=lambda a: (scores[a], a == 0),
        )
        return OrientDecision(
            angle=best_angle,
            confidence=scores[best_angle],
            source="doctr_fallback" if have_label else "doctr_nolabel",
            paddlex_angle=crop_angle if have_label else page_angle,
            paddlex_conf=crop_conf if have_label else page_conf,
            doctr_scores=scores,
            crop_bbox=bbox,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )
