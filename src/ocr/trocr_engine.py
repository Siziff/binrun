"""TrOCR (Microsoft) — VisionEncoderDecoder для печати и рукописи.

TrOCR работает на CROP-ах одной строки текста, поэтому требует детектор.
Используем docTR DBNet (легковесный) как text-line detector.

Доступны две модели:
    * `microsoft/trocr-large-printed` — печатный текст
    * `microsoft/trocr-large-handwritten` — рукопись

В Эксп. 1 запускаем обе как отдельные движки, чтобы сравнить.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from src.ocr.base import OCREngine
from src.ocr.schema import BBox, OCRDocument, OCRLine

Variant = Literal["printed", "handwritten"]


class TrOCREngine(OCREngine):
    """TrOCR + docTR-DBNet для line-level OCR."""

    def __init__(self,
                 variant: Variant = "printed",
                 device: str | None = None,
                 batch_size: int = 8,
                 max_lines: int = 256,
                 **kwargs):
        super().__init__(**kwargs)
        self.variant = variant
        self.name = f"trocr-{variant}"
        self.device = device
        self.batch_size = batch_size
        self.max_lines = max_lines
        self._processor = None
        self._model = None
        self._detector = None

    def _load(self) -> None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from doctr.models import detection_predictor

        repo = f"microsoft/trocr-large-{self.variant}"
        self._processor = TrOCRProcessor.from_pretrained(repo)
        self._model = VisionEncoderDecoderModel.from_pretrained(repo)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self.device).eval()
        # детектор строк (только det из docTR)
        self._detector = detection_predictor(arch="db_resnet50", pretrained=True)
        self._version = "trocr-large-" + self.variant

    def _detect_lines(self, image_path: Path) -> list[BBox]:
        """Возвращает bbox строк через docTR DBNet."""
        from doctr.io import DocumentFile

        doc = DocumentFile.from_images(str(image_path))
        # detection_predictor вернёт list[ {'words': ndarray[N, 5]} per page ]
        # values: [xmin, ymin, xmax, ymax, score] in [0, 1]
        det_out = self._detector(doc)
        if not det_out:
            return []
        # размер картинки для денормализации
        from PIL import Image
        img = Image.open(image_path)
        w, h = img.size
        page0 = det_out[0]
        if isinstance(page0, dict):
            words = page0.get("words")
        else:
            words = page0
        bboxes: list[BBox] = []
        if words is None:
            return []
        for row in words:
            x1, y1, x2, y2 = row[0] * w, row[1] * h, row[2] * w, row[3] * h
            bboxes.append(BBox.from_xyxy(x1, y1, x2, y2))
        return bboxes

    def _group_into_lines(self, bboxes: list[BBox]) -> list[list[BBox]]:
        """Группирует word-bbox в строки по вертикальному перекрытию."""
        if not bboxes:
            return []
        sorted_b = sorted(bboxes, key=lambda b: (b.y1, b.x1))
        rows: list[list[BBox]] = []
        for b in sorted_b:
            placed = False
            cy = (b.y1 + b.y2) / 2
            for row in rows:
                # средняя высота уже размещённых
                ry1 = min(x.y1 for x in row)
                ry2 = max(x.y2 for x in row)
                if ry1 - 5 <= cy <= ry2 + 5:
                    row.append(b)
                    placed = True
                    break
            if not placed:
                rows.append([b])
        # сортируем слова в каждой строке слева направо
        for row in rows:
            row.sort(key=lambda b: b.x1)
        return rows

    def _recognize(self, image_path: Path) -> OCRDocument:
        import torch
        from PIL import Image

        word_boxes = self._detect_lines(image_path)
        if not word_boxes:
            return OCRDocument.from_lines(
                image=image_path.name, engine=self.name,
                engine_version=self._version, lines=[],
            )

        rows = self._group_into_lines(word_boxes)
        # Соберём один bbox на строку из всех слов в этой строке.
        line_bboxes = []
        for row in rows[: self.max_lines]:
            x1 = min(b.x1 for b in row)
            y1 = min(b.y1 for b in row)
            x2 = max(b.x2 for b in row)
            y2 = max(b.y2 for b in row)
            # лёгкий padding по высоте
            ph = (y2 - y1) * 0.1
            line_bboxes.append(BBox.from_xyxy(x1, max(0, y1 - ph), x2, y2 + ph))

        img = Image.open(image_path).convert("RGB")
        crops = []
        for b in line_bboxes:
            crop = img.crop((int(b.x1), int(b.y1), int(b.x2), int(b.y2)))
            if crop.width < 5 or crop.height < 5:
                continue
            crops.append(crop)

        if not crops:
            return OCRDocument.from_lines(
                image=image_path.name, engine=self.name,
                engine_version=self._version, lines=[],
            )

        # Бэтч-инференс.
        lines: list[OCRLine] = []
        with torch.inference_mode():
            for i in range(0, len(crops), self.batch_size):
                batch = crops[i: i + self.batch_size]
                pixel_values = self._processor(
                    images=batch, return_tensors="pt"
                ).pixel_values.to(self.device)
                generated = self._model.generate(
                    pixel_values,
                    max_new_tokens=64,
                    num_beams=1,
                )
                texts = self._processor.batch_decode(generated, skip_special_tokens=True)
                for j, text in enumerate(texts):
                    bbox = line_bboxes[i + j] if (i + j) < len(line_bboxes) else None
                    if text and text.strip():
                        lines.append(OCRLine(
                            text=text.strip(),
                            bbox=bbox,
                            confidence=None,  # TrOCR не отдаёт надёжный score
                        ))

        return OCRDocument.from_lines(
            image=image_path.name, engine=self.name,
            engine_version=self._version, lines=lines,
        )
