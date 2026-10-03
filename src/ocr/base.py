"""Абстрактный базовый класс OCR-обёртки."""
from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path

from src.ocr.schema import OCRDocument


class OCREngine(ABC):
    """Базовая обёртка над OCR-движком.

    Контракт:
        * lazy-init: тяжёлая загрузка моделей происходит при первом вызове `run`,
          либо явно через `load()`.
        * детерминированный вывод через `OCRDocument`.
        * исключения внутри `_recognize` ловятся и оборачиваются в
          `OCRDocument.error_result`, чтобы один битый файл не валил весь прогон.
    """

    name: str = "base"

    def __init__(self, **kwargs):
        self._loaded = False
        self._kwargs = kwargs

    def load(self) -> None:
        """Явная загрузка моделей. По умолчанию ничего не делает."""
        if not self._loaded:
            self._load()
            self._loaded = True

    @abstractmethod
    def _load(self) -> None:
        """Подгружает модель/процессор. Должен быть идемпотентным."""

    @abstractmethod
    def _recognize(self, image_path: Path) -> OCRDocument:
        """Сырая логика распознавания одного изображения."""

    def run(self, image_path: str | Path) -> OCRDocument:
        """Безопасный публичный API: гарантирует возврат `OCRDocument`."""
        image_path = Path(image_path)
        if not image_path.exists():
            return OCRDocument.error_result(
                image=image_path.name, engine=self.name,
                error=f"Image not found: {image_path}",
            )
        try:
            self.load()
        except Exception as e:
            return OCRDocument.error_result(
                image=image_path.name, engine=self.name,
                error=f"Load failed: {type(e).__name__}: {e}",
            )

        t0 = time.perf_counter()
        try:
            doc = self._recognize(image_path)
        except Exception:
            tb = traceback.format_exc(limit=3)
            doc = OCRDocument.error_result(
                image=image_path.name, engine=self.name,
                error=tb, runtime_seconds=time.perf_counter() - t0,
            )
        if not doc.runtime_seconds:
            doc.runtime_seconds = time.perf_counter() - t0
        return doc
