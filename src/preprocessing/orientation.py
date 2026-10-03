"""Каскадное определение ориентации (вариант F + усиление через docTR rotation-score).

Шаги:
    1. PaddleX classifier (~50 ms) → (angle, conf).
    2. Если conf < threshold (по умолчанию 0.85) → проверка docTR-детектором
       на 4 поворотах. Угол, на котором детектор нашёл больше всего строк
       с высокой суммарной confidence, считаем верным.
    3. Если docTR не нашёл текста ни на одном повороте → возвращаем 0°
       (картинка без текста / совсем пустая).

Логика поворотов: классификатор возвращает «текст лежит на боку X°»,
для PIL мы поворачиваем «по часовой стрелке» = `img.rotate(-X, expand=True)`.
"""
from __future__ import annotations

from src.utils import env_setup  # noqa: F401

import io
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

os.environ.setdefault("FLAGS_enable_pir_in_executor", "false")
os.environ.setdefault("FLAGS_use_mkldnn", "false")

ANGLES = (0, 90, 180, 270)


@dataclass
class OrientationResult:
    angle: int                # 0/90/180/270
    confidence: float
    source: str               # "paddle" | "doctr_cascade" | "fallback"
    paddle_angle: Optional[int] = None
    paddle_conf: Optional[float] = None
    doctr_scores: Optional[dict] = None  # {angle: total_score}


class OrientationCascade:
    """Каскад: PaddleX → docTR rotation score."""

    def __init__(self,
                 paddle_threshold: float = 0.85,
                 doctr_max_side: int = 1500):
        from paddlex import create_model
        self.paddle_threshold = paddle_threshold
        self.doctr_max_side = doctr_max_side
        self._paddle = create_model("PP-LCNet_x1_0_doc_ori")
        self._doctr_det = None  # lazy

    def _load_doctr(self) -> None:
        if self._doctr_det is None:
            from doctr.models import detection_predictor
            self._doctr_det = detection_predictor(
                arch="db_resnet50", pretrained=True,
                assume_straight_pages=True,
            )

    def _paddle_predict(self, img: Image.Image) -> tuple[int, float]:
        arr = np.array(img.convert("RGB"))
        results = list(self._paddle.predict([arr]))
        r = results[0]
        return int(r["label_names"][0]), float(r["scores"][0])

    def _doctr_score_rotation(self, img: Image.Image) -> dict:
        """{angle: total_score} — суммарная confidence слов после поворота."""
        self._load_doctr()

        # Downscale для скорости (детектор всё равно работает на ~1024).
        side = max(img.size)
        if side > self.doctr_max_side:
            ratio = self.doctr_max_side / side
            small = img.resize(
                (int(img.size[0] * ratio), int(img.size[1] * ratio)),
                Image.LANCZOS,
            )
        else:
            small = img

        from doctr.io import DocumentFile
        scores: dict = {}
        for angle in ANGLES:
            if angle == 0:
                rotated = small
            else:
                rotated = small.rotate(-angle, expand=True, resample=Image.BICUBIC)
            buf = io.BytesIO()
            rotated.convert("RGB").save(buf, format="JPEG", quality=88)
            doc = DocumentFile.from_images(buf.getvalue())
            det_out = self._doctr_det(doc)
            total = 0.0
            n_words = 0
            if det_out:
                page = det_out[0]
                # det_out shape: list of dicts or list of arrays depending on doctr version
                if isinstance(page, dict):
                    words = page.get("words")
                else:
                    words = page
                if words is not None:
                    arr = np.asarray(words)
                    if arr.size and arr.ndim == 2 and arr.shape[1] >= 5:
                        n_words = arr.shape[0]
                        total = float(arr[:, 4].sum())
            scores[angle] = total
        return scores

    def predict(self, img: Image.Image) -> OrientationResult:
        # Шаг 1: дешёвый PaddleX classifier.
        pa, pc = self._paddle_predict(img)
        if pc >= self.paddle_threshold:
            return OrientationResult(
                angle=pa, confidence=pc, source="paddle",
                paddle_angle=pa, paddle_conf=pc,
            )

        # Шаг 2: сомнительные случаи — docTR на 4 поворотах.
        scores = self._doctr_score_rotation(img)
        # Берём угол с максимальным total score (но требуем минимальный сигнал).
        best_angle, best_score = max(scores.items(), key=lambda x: x[1])
        if best_score < 0.5:
            return OrientationResult(
                angle=0, confidence=0.0, source="fallback",
                paddle_angle=pa, paddle_conf=pc,
                doctr_scores=scores,
            )
        # Нормируем confidence как отношение score лучшего к сумме.
        total = sum(scores.values()) or 1.0
        conf = float(best_score / total)
        return OrientationResult(
            angle=int(best_angle), confidence=conf,
            source="doctr_cascade",
            paddle_angle=pa, paddle_conf=pc,
            doctr_scores=scores,
        )


def main():
    """CLI-проверка на нескольких файлах."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=0.85)
    args = ap.parse_args()
    oc = OrientationCascade(paddle_threshold=args.threshold)
    import time
    for p in args.images:
        img = Image.open(p).convert("RGB")
        t0 = time.perf_counter()
        r = oc.predict(img)
        dt = time.perf_counter() - t0
        line = (f"{p.split('/')[-1]:<25} {dt:5.2f}s  "
                f"angle={r.angle:>3} conf={r.confidence:.2f} src={r.source}")
        if r.doctr_scores:
            ds = "  ".join(f"{a:>3}:{s:.1f}" for a, s in r.doctr_scores.items())
            line += f"  paddle={r.paddle_angle}/{r.paddle_conf:.2f}  doctr=[{ds}]"
        print(line)


if __name__ == "__main__":
    main()
