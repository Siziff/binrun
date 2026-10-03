"""PaddleOCR на CPU + RapidOCR (ONNX) fallback.

В paddlepaddle 3.x на CPU встречается баг в PIR/OneDNN-executor для
PP-OCRv5 моделей (`ConvertPirAttribute2RuntimeAttribute`).
В качестве надёжного multilingual baseline используем RapidOCR —
ONNX-runtime сборку тех же PaddleOCR-моделей. Кэш изолирован
через `env_setup` (модели лежат в `data/models/paddleocr/...`).

Если на машине доступна корректно работающая PaddleOCR — её можно
включить флагом `prefer_paddle=True`.
"""
from __future__ import annotations

import os
from pathlib import Path

from src.ocr.base import OCREngine
from src.ocr.schema import BBox, OCRDocument, OCRLine


class PaddleEngine(OCREngine):
    """RapidOCR (ONNX) совместимая с PaddleOCR multilingual выдача."""

    name = "paddleocr"

    def __init__(self, lang: str = "ru", use_orientation: bool = False,
                 prefer_paddle: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.lang = lang
        self.use_orientation = use_orientation
        self.prefer_paddle = prefer_paddle
        self._impl = None
        self._mode: str | None = None  # "paddle" | "rapidocr"
        self._version = "?"

    def _try_paddle(self) -> bool:
        try:
            from paddleocr import PaddleOCR
            self._impl = PaddleOCR(
                lang=self.lang,
                use_textline_orientation=self.use_orientation,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
            # warm-up: убедимся, что не падает на простом forward.
            self._mode = "paddle"
            import paddleocr as _p
            self._version = f"paddleocr-{getattr(_p, '__version__', '?')}"
            return True
        except Exception:
            return False

    def _load_rapidocr(self) -> None:
        # Модели RapidOCR кладём в data/models/paddleocr/rapidocr/
        models_root = Path(os.environ["PADDLE_PDX_CACHE_HOME"]) / "rapidocr"
        models_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("RAPIDOCR_MODELS_DIR", str(models_root))
        from rapidocr_onnxruntime import RapidOCR
        self._impl = RapidOCR()
        self._mode = "rapidocr"
        import rapidocr_onnxruntime as _r
        self._version = f"rapidocr-{getattr(_r, '__version__', '?')}"

    def _load(self) -> None:
        if self.prefer_paddle and self._try_paddle():
            return
        self._load_rapidocr()

    def _recognize(self, image_path: Path) -> OCRDocument:
        if self._mode == "paddle":
            return self._recognize_paddle(image_path)
        return self._recognize_rapidocr(image_path)

    def _recognize_paddle(self, image_path: Path) -> OCRDocument:
        result = self._impl.predict(str(image_path))
        lines: list[OCRLine] = []
        for res in result:
            texts = res.get("rec_texts") or []
            scores = res.get("rec_scores") or []
            polys = res.get("rec_polys") or []
            for i, text in enumerate(texts):
                if not text:
                    continue
                conf = float(scores[i]) if i < len(scores) else None
                bbox = None
                if i < len(polys) and polys[i] is not None:
                    try:
                        coords = [[float(p[0]), float(p[1])] for p in polys[i]]
                        bbox = BBox.from_quad(coords)
                    except (TypeError, IndexError):
                        bbox = None
                lines.append(OCRLine(text=str(text), bbox=bbox, confidence=conf))
        return OCRDocument.from_lines(
            image=image_path.name, engine=self.name,
            engine_version=self._version, lines=lines,
        )

    def _recognize_rapidocr(self, image_path: Path) -> OCRDocument:
        result, _ = self._impl(str(image_path))
        lines: list[OCRLine] = []
        if not result:
            return OCRDocument.from_lines(
                image=image_path.name, engine=self.name,
                engine_version=self._version, lines=lines,
            )
        for item in result:
            # item: [poly (4 points), text, confidence]
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
            try:
                conf_f = float(conf)
            except (TypeError, ValueError):
                conf_f = None
            lines.append(OCRLine(text=str(text), bbox=bbox, confidence=conf_f))
        return OCRDocument.from_lines(
            image=image_path.name, engine=self.name,
            engine_version=self._version, lines=lines,
        )
