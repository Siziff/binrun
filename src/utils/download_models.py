"""Прокачка весов всех OCR-моделей в локальный кэш проекта.

Все пути изолированы в `data/models/` через `env_setup`.
"""
from __future__ import annotations

# ВАЖНО: env_setup должен импортироваться ДО любых ML-библиотек.
from src.utils import env_setup  # noqa: F401


def step(name: str) -> None:
    print(f"\n=== {name} ===", flush=True)


def download_tesseract_models() -> None:
    step("Tesseract")
    import subprocess
    r = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True)
    needed = ("rus", "eng", "deu", "lat", "fra", "ita", "spa")
    available = [l for l in r.stdout.split("\n") if l in needed]
    print(f"Verified langs: {available}")


def download_easyocr() -> None:
    step("EasyOCR (ru+en latin)")
    import easyocr
    reader = easyocr.Reader(["ru", "en"], gpu=True, download_enabled=True, verbose=False)
    del reader
    print("  done")


def download_paddleocr() -> None:
    step("PaddleOCR PP-OCRv5")
    from paddleocr import PaddleOCR
    for lang in ("ru", "german"):
        try:
            _ = PaddleOCR(lang=lang, use_textline_orientation=False)
            print(f"  loaded lang={lang}")
        except Exception as e:
            print(f"  skip lang={lang}: {type(e).__name__}: {e}")


def download_doctr() -> None:
    step("docTR (db_resnet50 + crnn_vgg16_bn)")
    from doctr.models import ocr_predictor
    _ = ocr_predictor(
        det_arch="db_resnet50",
        reco_arch="crnn_vgg16_bn",
        pretrained=True,
    )
    print("  done")


def download_surya() -> None:
    step("Surya OCR (det + rec)")
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    _det = DetectionPredictor()
    _rec = RecognitionPredictor()
    print("  surya predictors initialized")


def download_trocr() -> None:
    step("TrOCR (printed + handwritten)")
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    for repo in ("microsoft/trocr-large-printed", "microsoft/trocr-large-handwritten"):
        print(f"  → {repo}")
        TrOCRProcessor.from_pretrained(repo)
        VisionEncoderDecoderModel.from_pretrained(repo)


def main() -> None:
    env_setup.report()
    download_tesseract_models()
    download_easyocr()
    download_paddleocr()
    download_doctr()
    download_surya()
    download_trocr()
    print("\nAll model weights are cached locally in data/models/.")


if __name__ == "__main__":
    main()
