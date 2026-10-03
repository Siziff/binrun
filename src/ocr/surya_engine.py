"""Surya OCR (datalab) — современный multilingual."""
from __future__ import annotations

from pathlib import Path

from src.ocr.base import OCREngine
from src.ocr.schema import BBox, OCRDocument, OCRLine


class SuryaEngine(OCREngine):
    """Surya: DetectionPredictor + RecognitionPredictor.

    Surya 0.20+: новый API без `langs` — модель сама определяет язык.
    На GPU использует bf16/fp16 автоматически.
    """

    name = "surya"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._det = None
        self._rec = None

    def _load(self) -> None:
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        self._det = DetectionPredictor()
        self._rec = RecognitionPredictor()
        try:
            import surya
            self._version = getattr(surya, "__version__", "?")
        except Exception:
            self._version = "?"

    def _recognize(self, image_path: Path) -> OCRDocument:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        # Surya 0.15: det_predictor + recognition, без VLM.
        ocr_results = self._rec(
            [img],
            det_predictor=self._det,
            task_names=["ocr_with_boxes"] * 1,
        )
        if not ocr_results:
            return OCRDocument.from_lines(
                image=image_path.name, engine=self.name,
                engine_version=self._version, lines=[],
            )
        page = ocr_results[0]
        lines: list[OCRLine] = []
        # PageOCRResult: text_lines: list[TextLine]
        for text_line in getattr(page, "text_lines", []):
            text = getattr(text_line, "text", "") or ""
            if not text:
                continue
            conf = getattr(text_line, "confidence", None)
            bbox_attr = getattr(text_line, "bbox", None)
            bbox = None
            if bbox_attr is not None:
                try:
                    if len(bbox_attr) == 4:
                        bbox = BBox.from_xyxy(*[float(v) for v in bbox_attr])
                except (TypeError, ValueError):
                    bbox = None
            lines.append(OCRLine(
                text=text.strip(),
                bbox=bbox,
                confidence=float(conf) if conf is not None else None,
            ))
        return OCRDocument.from_lines(
            image=image_path.name, engine=self.name,
            engine_version=self._version, lines=lines,
        )
