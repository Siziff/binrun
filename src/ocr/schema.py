"""Единый формат OCR-выдачи для всех движков.

Каждая обёртка возвращает `OCRDocument` с одинаковой структурой,
что позволяет сравнивать движки и считать общие метрики.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Bounding box в пиксельных координатах (xyxy)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "BBox":
        return cls(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))

    @classmethod
    def from_quad(cls, quad: list[list[float]]) -> "BBox":
        """Преобразует quadrilateral [[x,y],...] в axis-aligned bbox."""
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        return cls(x1=float(min(xs)), y1=float(min(ys)),
                   x2=float(max(xs)), y2=float(max(ys)))


class OCRLine(BaseModel):
    """Одна распознанная строка/токен."""
    text: str
    bbox: Optional[BBox] = None
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Уверенность в [0, 1]; None если движок не возвращает.",
    )


class OCRDocument(BaseModel):
    """Полная выдача одного движка по одному изображению."""
    image: str = Field(..., description="Имя файла изображения.")
    engine: str = Field(..., description="Идентификатор движка (например, 'tesseract', 'paddleocr').")
    engine_version: Optional[str] = None
    lines: list[OCRLine] = Field(default_factory=list)
    full_text: str = Field("", description="Конкатенация всех строк через '\\n'.")
    runtime_seconds: float = 0.0
    error: Optional[str] = None

    @classmethod
    def from_lines(cls, image: str, engine: str, lines: list[OCRLine],
                   runtime_seconds: float = 0.0,
                   engine_version: Optional[str] = None) -> "OCRDocument":
        """Конструктор с автоматической сборкой `full_text`."""
        full_text = "\n".join(line.text for line in lines if line.text)
        return cls(
            image=image,
            engine=engine,
            engine_version=engine_version,
            lines=lines,
            full_text=full_text,
            runtime_seconds=runtime_seconds,
        )

    @classmethod
    def error_result(cls, image: str, engine: str, error: str,
                     runtime_seconds: float = 0.0) -> "OCRDocument":
        return cls(
            image=image,
            engine=engine,
            lines=[],
            full_text="",
            runtime_seconds=runtime_seconds,
            error=error,
        )
