"""Tesseract 4/5 как baseline.

Используем `pytesseract.image_to_data` чтобы получить bbox + confidence,
а не только plain text.
"""
from __future__ import annotations

from pathlib import Path

from src.ocr.base import OCREngine
from src.ocr.schema import BBox, OCRDocument, OCRLine


class TesseractEngine(OCREngine):
    """Tesseract с многоязычной моделью.

    По умолчанию используем комбинацию языков rus+eng+deu+lat+fra
    (все имеются в `tesseract --list-langs`).
    """

    name = "tesseract"

    def __init__(self, languages: str = "rus+eng+deu+lat+fra",
                 psm: int = 6, max_side: int = 3000, **kwargs):
        super().__init__(**kwargs)
        self.languages = languages
        self.psm = psm
        self.max_side = max_side
        self._pytesseract = None

    def _load(self) -> None:
        import pytesseract  # noqa
        self._pytesseract = pytesseract
        self._version = str(pytesseract.get_tesseract_version())

    def _recognize(self, image_path: Path) -> OCRDocument:
        import pytesseract
        from PIL import Image

        img = Image.open(image_path)
        # Уменьшаем большие сканы (по умолчанию 7296×5472) до max_side=3000
        # для разумного времени; на качество это почти не влияет, т.к. этикетки
        # обычно много меньше всего листа.
        if max(img.size) > self.max_side:
            ratio = self.max_side / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        # PSM 6 — Assume a single uniform block of text (хорошо для этикетки).
        config = f"--oem 3 --psm {self.psm}"
        data = pytesseract.image_to_data(
            img,
            lang=self.languages,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        n = len(data["text"])
        # Группируем токены в строки по (block_num, par_num, line_num).
        lines_buf: dict[tuple, dict] = {}
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            conf_raw = data["conf"][i]
            try:
                conf = float(conf_raw) / 100.0 if float(conf_raw) >= 0 else None
            except (TypeError, ValueError):
                conf = None
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            if key not in lines_buf:
                lines_buf[key] = {
                    "tokens": [],
                    "confs": [],
                    "x1": data["left"][i],
                    "y1": data["top"][i],
                    "x2": data["left"][i] + data["width"][i],
                    "y2": data["top"][i] + data["height"][i],
                }
            line = lines_buf[key]
            line["tokens"].append(text)
            if conf is not None:
                line["confs"].append(conf)
            # расширяем bbox
            line["x1"] = min(line["x1"], data["left"][i])
            line["y1"] = min(line["y1"], data["top"][i])
            line["x2"] = max(line["x2"], data["left"][i] + data["width"][i])
            line["y2"] = max(line["y2"], data["top"][i] + data["height"][i])

        lines: list[OCRLine] = []
        for key in sorted(lines_buf.keys()):
            ln = lines_buf[key]
            lines.append(OCRLine(
                text=" ".join(ln["tokens"]),
                bbox=BBox.from_xyxy(ln["x1"], ln["y1"], ln["x2"], ln["y2"]),
                confidence=(sum(ln["confs"]) / len(ln["confs"])) if ln["confs"] else None,
            ))

        return OCRDocument.from_lines(
            image=image_path.name, engine=self.name,
            engine_version=self._version, lines=lines,
        )
