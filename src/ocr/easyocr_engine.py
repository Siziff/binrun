"""EasyOCR — простой multilingual detector+recognizer."""
from __future__ import annotations

from pathlib import Path

from src.ocr.base import OCREngine
from src.ocr.schema import BBox, OCRDocument, OCRLine


class EasyOCREngine(OCREngine):
    """EasyOCR с GPU (rus+en).

    EasyOCR 1.7 не позволяет смешивать языки разных алфавитных групп
    в одном Reader; для исторических этикеток с латынью можно подключить
    отдельный Reader (`en`) и объединить. Пока стартуем с ru+en.
    """

    name = "easyocr"

    def __init__(self, langs: tuple[str, ...] = ("ru", "en"), gpu: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.langs = list(langs)
        self.gpu = gpu
        self._reader = None

    def _load(self) -> None:
        import easyocr
        self._reader = easyocr.Reader(
            self.langs,
            gpu=self.gpu,
            download_enabled=False,
            verbose=False,
        )
        self._version = getattr(easyocr, "__version__", "?")

    def _recognize(self, image_path: Path) -> OCRDocument:
        # readtext возвращает list of (bbox4points, text, confidence)
        results = self._reader.readtext(str(image_path), detail=1, paragraph=False)
        lines: list[OCRLine] = []
        for item in results:
            if len(item) < 3:
                continue
            poly, text, conf = item[0], item[1], item[2]
            if not text:
                continue
            try:
                coords = [[float(p[0]), float(p[1])] for p in poly]
                bbox = BBox.from_quad(coords)
            except (TypeError, IndexError):
                bbox = None
            lines.append(OCRLine(
                text=str(text),
                bbox=bbox,
                confidence=float(conf) if conf is not None else None,
            ))
        return OCRDocument.from_lines(
            image=image_path.name, engine=self.name,
            engine_version=self._version, lines=lines,
        )
