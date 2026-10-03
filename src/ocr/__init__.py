"""Реестр OCR-движков."""
from __future__ import annotations

from src.ocr.base import OCREngine
from src.ocr.schema import BBox, OCRDocument, OCRLine


def get_engine(name: str, **kwargs) -> OCREngine:
    """Создаёт OCR-обёртку по имени.

    Доступные движки:
        * 'tesseract'
        * 'paddleocr'
        * 'easyocr'
        * 'doctr'
        * 'surya'
        * 'trocr-printed'
        * 'trocr-handwritten'
    """
    name = name.lower().strip()
    if name == "tesseract":
        from src.ocr.tesseract_engine import TesseractEngine
        return TesseractEngine(**kwargs)
    if name == "paddleocr":
        from src.ocr.paddle_engine import PaddleEngine
        return PaddleEngine(**kwargs)
    if name == "easyocr":
        from src.ocr.easyocr_engine import EasyOCREngine
        return EasyOCREngine(**kwargs)
    if name == "doctr":
        from src.ocr.doctr_engine import DocTREngine
        return DocTREngine(**kwargs)
    if name == "surya":
        from src.ocr.surya_engine import SuryaEngine
        return SuryaEngine(**kwargs)
    if name in ("trocr-printed", "trocr_printed"):
        from src.ocr.trocr_engine import TrOCREngine
        return TrOCREngine(variant="printed", **kwargs)
    if name in ("trocr-handwritten", "trocr_handwritten"):
        from src.ocr.trocr_engine import TrOCREngine
        return TrOCREngine(variant="handwritten", **kwargs)
    raise ValueError(f"Unknown OCR engine: {name!r}")


ALL_ENGINES = (
    "tesseract", "paddleocr", "easyocr", "doctr", "surya",
    "trocr-printed", "trocr-handwritten",
)

__all__ = ["OCREngine", "OCRDocument", "OCRLine", "BBox", "get_engine", "ALL_ENGINES"]
