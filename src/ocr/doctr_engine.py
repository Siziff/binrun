"""docTR (Mindee) — DBNet detector + CRNN recognizer."""
from __future__ import annotations

from pathlib import Path

from src.ocr.base import OCREngine
from src.ocr.schema import BBox, OCRDocument, OCRLine


class DocTREngine(OCREngine):
    """docTR с предобученными весами."""

    name = "doctr"

    def __init__(self, det_arch: str = "db_resnet50",
                 reco_arch: str = "crnn_vgg16_bn", **kwargs):
        super().__init__(**kwargs)
        self.det_arch = det_arch
        self.reco_arch = reco_arch
        self._predictor = None

    def _load(self) -> None:
        from doctr.models import ocr_predictor
        import doctr
        self._predictor = ocr_predictor(
            det_arch=self.det_arch,
            reco_arch=self.reco_arch,
            pretrained=True,
            assume_straight_pages=True,
        )
        self._version = getattr(doctr, "__version__", "?")

    def _recognize(self, image_path: Path) -> OCRDocument:
        from doctr.io import DocumentFile

        doc = DocumentFile.from_images(str(image_path))
        result = self._predictor(doc)
        # result.pages[0].blocks[i].lines[j].words[k]
        # relative bbox: ((x1,y1),(x2,y2)) в [0,1] → нужно умножить на размер картинки.
        page = result.pages[0]
        h, w = page.dimensions  # (H, W)
        lines: list[OCRLine] = []
        for block in page.blocks:
            for line in block.lines:
                if not line.words:
                    continue
                text = " ".join(word.value for word in line.words if word.value)
                if not text:
                    continue
                # средняя confidence по словам
                confs = [w_.confidence for w_ in line.words
                         if getattr(w_, "confidence", None) is not None]
                conf = (sum(confs) / len(confs)) if confs else None
                # объединённый bbox
                (rx1, ry1), (rx2, ry2) = line.geometry
                bbox = BBox.from_xyxy(rx1 * w, ry1 * h, rx2 * w, ry2 * h)
                lines.append(OCRLine(text=text, bbox=bbox, confidence=conf))
        return OCRDocument.from_lines(
            image=image_path.name, engine=self.name,
            engine_version=self._version, lines=lines,
        )
