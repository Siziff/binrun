"""Гибридный детектор ориентации (схема C+F).

Идея:
    1) Grounding DINO (`IDEA-Research/grounding-dino-base`) находит на снимке
       прямоугольник с этикеткой по zero-shot текстовым запросам
       («a paper label», «herbarium label»). Если боксов нет — fallback на
       весь лист.
    2) PaddleX `PP-LCNet_x1_0_doc_ori` предсказывает один из 4 углов
       {0, 90, 180, 270} на КРОПЕ этикетки и возвращает confidence.
    3) Если PaddleX confidence < `tau` (по умолчанию 0.85), запускается
       fallback по docTR: детектируем текстовые боксы на ВСЕХ четырёх
       ротациях кропа и выбираем ту, где средняя «прямоугольность» текстовых
       боксов (отношение площади горизонтальных строк к вертикальным)
       максимальна, а число боксов выше всего. Тай-брейкеры:
         - больше детекций → выше скор;
         - 0° предпочтительнее остальных при равенстве (минимизирует риск
           ложного поворота).

Возвращаемое решение применяется к ИСХОДНОМУ листу целиком (Grounding DINO
используется только как «оракул» по ориентации; ничего не вырезаем).

Используется как замена PaddleX-on-page в `preprocess.py`.
"""
from __future__ import annotations

from src.utils import env_setup  # noqa: F401

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageOps

os.environ.setdefault("FLAGS_enable_pir_in_executor", "false")
os.environ.setdefault("FLAGS_use_mkldnn", "false")


@dataclass
class OrientDecision:
    angle: int                       # выбранный угол {0,90,180,270}
    confidence: float                # confidence выбранного источника
    source: str                      # 'paddlex_crop' | 'doctr_fallback' | 'no_label'
    paddlex_angle: int               # «сырой» вывод PaddleX
    paddlex_conf: float
    doctr_scores: dict[int, float]   # доступно, если был fallback
    crop_bbox: Optional[tuple[int, int, int, int]]   # (x0, y0, x1, y1) на исходном RGB после EXIF
    elapsed_ms: float


class GroundingDinoLabelLocator:
    """Находит этикетку: zero-shot bbox через `IDEA-Research/grounding-dino-base`."""

    PROMPT = "a paper label. a herbarium label. a printed text label."

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        min_frac: float = 0.03,
        max_frac: float = 0.45,
    ) -> None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.min_frac = min_frac
        self.max_frac = max_frac
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()

    @torch.inference_mode()
    def locate(self, img: Image.Image) -> Optional[tuple[int, int, int, int]]:
        """Возвращает bbox этикетки (x0,y0,x1,y1) или None.

        Стратегия: фильтр по площади [min_frac, max_frac] от листа, затем
        выбираем бокс с максимальным score. Если ни одного — None.
        """
        inputs = self.processor(images=img, text=self.PROMPT, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[img.size[::-1]],   # (H, W)
        )[0]
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        if len(boxes) == 0:
            return None

        W, H = img.size
        page_area = float(W * H)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        fracs = areas / page_area

        # Фильтр по диапазону площади.
        mask = (fracs >= self.min_frac) & (fracs <= self.max_frac)
        if not mask.any():
            return None

        cand_boxes = boxes[mask]
        cand_scores = scores[mask]
        i = int(np.argmax(cand_scores))
        x0, y0, x1, y1 = cand_boxes[i].tolist()

        # Расширим бокс на 5% по каждой стороне.
        pad_x = 0.05 * (x1 - x0)
        pad_y = 0.05 * (y1 - y0)
        x0 = max(0, int(x0 - pad_x))
        y0 = max(0, int(y0 - pad_y))
        x1 = min(W, int(x1 + pad_x))
        y1 = min(H, int(y1 + pad_y))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return None
        return (x0, y0, x1, y1)


class PaddleXCropOrient:
    """`PP-LCNet_x1_0_doc_ori` поверх кропа."""

    def __init__(self) -> None:
        from paddlex import create_model
        self._model = create_model("PP-LCNet_x1_0_doc_ori")

    def predict(self, img: Image.Image) -> tuple[int, float]:
        # paddlex предпочитает путь к файлу; пишем во временный буфер.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f, format="JPEG", quality=92)
            tmp = f.name
        try:
            res = next(iter(self._model.predict([tmp])))
            angle = int(res["label_names"][0])
            conf = float(res["scores"][0])
            return angle, conf
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass


class DocTRRotationFallback:
    """4-кратная ротация → docTR detection → выбор лучшей."""

    def __init__(self, max_side: int = 1280) -> None:
        from doctr.models import detection_predictor
        self.det = detection_predictor(pretrained=True, assume_straight_pages=True)
        self.max_side = max_side

    @staticmethod
    def _resize(img: Image.Image, max_side: int) -> Image.Image:
        if max(img.size) <= max_side:
            return img
        ratio = max_side / max(img.size)
        return img.resize(
            (int(img.size[0] * ratio), int(img.size[1] * ratio)),
            Image.LANCZOS,
        )

    def _score(self, img: Image.Image) -> float:
        small = self._resize(img, self.max_side)
        arr = np.asarray(small.convert("RGB"))
        out = self.det([arr])
        # docTR ≥1.0 возвращает list[dict{'words': ndarray Nx5}].
        try:
            words = out[0]["words"]
        except (KeyError, TypeError):
            return 0.0
        if words is None or len(words) == 0:
            return 0.0
        # words shape: (N, 5) → (x0,y0,x1,y1,score) в нормализованных координатах.
        boxes = np.asarray(words)
        scores = boxes[:, 4] if boxes.shape[1] >= 5 else np.ones(len(boxes))
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        # Доля «горизонтальных» строк (W > H) среди всех боксов; для книжной
        # ориентации текста это ≈ 1.0, для боковой — ≈ 0.
        horiz = (widths > heights).astype(np.float32)
        horiz_share = float(horiz.mean()) if len(horiz) else 0.0
        # Скор: средняя уверенность × доля горизонтальных × log(N+1).
        return float(scores.mean()) * (0.2 + 0.8 * horiz_share) * np.log1p(len(boxes))

    def predict(self, img: Image.Image) -> dict[int, float]:
        scores = {}
        for ang in (0, 90, 180, 270):
            rot = img if ang == 0 else img.rotate(-ang, expand=True, resample=Image.BICUBIC)
            scores[ang] = self._score(rot)
        return scores


class HybridOrientator:
    """Полный пайплайн C+F."""

    def __init__(
        self,
        paddle_conf_threshold: float = 0.85,
        gdino_model_id: str = "IDEA-Research/grounding-dino-base",
    ) -> None:
        self.paddle_conf_threshold = paddle_conf_threshold
        print("[HybridOrient] loading Grounding DINO …", flush=True)
        self.locator = GroundingDinoLabelLocator(model_id=gdino_model_id)
        print("[HybridOrient] loading PaddleX crop-orient …", flush=True)
        self.paddle = PaddleXCropOrient()
        print("[HybridOrient] loading docTR fallback …", flush=True)
        self.doctr = DocTRRotationFallback()

    def decide(self, img_path: Path) -> OrientDecision:
        t0 = time.perf_counter()
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img).convert("RGB")

        # 1) Grounding DINO → label crop.
        bbox = self.locator.locate(img)
        have_label = bbox is not None
        crop = img.crop(bbox) if have_label else img

        # 2) PaddleX (только если бокс этикетки нашли — это его сильная сторона;
        #    на полном листе он же даёт ошибки, ради чего и был задуман гибрид).
        paddle_angle, paddle_conf = 0, 0.0
        if have_label:
            try:
                paddle_angle, paddle_conf = self.paddle.predict(crop)
            except Exception as e:
                print(f"  paddle failed on {img_path.name}: {e}", flush=True)

            if paddle_conf >= self.paddle_conf_threshold:
                return OrientDecision(
                    angle=paddle_angle,
                    confidence=paddle_conf,
                    source="paddlex_crop",
                    paddlex_angle=paddle_angle,
                    paddlex_conf=paddle_conf,
                    doctr_scores={},
                    crop_bbox=bbox,
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                )

        # 3) Fallback: docTR×4 на кропе или (если бокса нет) на странице.
        scores = self.doctr.predict(crop)
        best_angle = max(
            scores.keys(),
            key=lambda a: (scores[a], a == 0),  # ties → prefer 0°
        )
        return OrientDecision(
            angle=best_angle,
            confidence=scores[best_angle],
            source="doctr_fallback" if have_label else "doctr_nolabel",
            paddlex_angle=paddle_angle,
            paddlex_conf=paddle_conf,
            doctr_scores=scores,
            crop_bbox=bbox,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )
