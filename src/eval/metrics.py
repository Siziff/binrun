"""Метрики для Эксперимента 1.

Уровни оценки:
    A. Document-level — CER/WER по конкатенированному тексту этикетки.
       Сравниваем `OCRDocument.full_text` с эталоном (склейка всех verbatim+normalized полей).
    B. Field-level — для каждого ожидаемого поля считаем максимальную
       сходство (RapidFuzz token_set_ratio) среди подстрок OCR-выдачи и эталона.
       Не строгий "exact match", т.к. baseline OCR не структурирует выдачу по полям —
       это будет в Эксп. 2.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import jiwer
from rapidfuzz import fuzz


# Поля ground-truth, которые имеют смысл для OCR-baseline.
# Verbatim → текст «как на этикетке», именно его OCR должен прочитать.
GT_VERBATIM_FIELDS = (
    "verbatim_identification",
    "verbatim_collector",
    "verbatim_locality",
    "verbatim_collection_date",
    "verbatim_substrate",
    "verbatim_habitat",
    "verbatim_identification_date",
    "field_number",
    "verbatim_latitude",
    "verbatim_longitude",
    "verbatim_elevation",
    "exsiccata_title",
    "exsiccata_number",
    "type_status",
    "catalog_number",
)

# Normalized (то, что задаётся куратором) — для широкой метрики «попал ли куда-то».
GT_NORMALIZED_FIELDS = (
    "scientific_name",
    "determiner",
    "collector",
    "country",
    "organism_remarks",
    "identification_remarks",
)


# --------------------------------------------------------------------------
# Нормализация текста для сравнения.
# --------------------------------------------------------------------------
_WS = re.compile(r"\s+")
# Убираем control-chars и многие punctuation для устойчивого CER.
_PUNCT = re.compile(r"[^\w\s\u00C0-\u024F\u0400-\u04FF°'\".,:;\-/+]")


def norm_text(s: str | None) -> str:
    """Унификация: нижний регистр, NFC, схлопывание пробелов."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFC", s)
    s = s.lower()
    s = _WS.sub(" ", s).strip()
    return s


def norm_text_strict(s: str | None) -> str:
    """Жёсткая нормализация для CER: убирает большинство пунктуации."""
    s = norm_text(s)
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


# --------------------------------------------------------------------------
# Document-level метрики.
# --------------------------------------------------------------------------
def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate в [0, ∞); типично [0, 1]."""
    r = norm_text_strict(reference)
    h = norm_text_strict(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    return float(jiwer.cer(r, h))


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate."""
    r = norm_text_strict(reference)
    h = norm_text_strict(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    return float(jiwer.wer(r, h))


def similarity(reference: str, hypothesis: str) -> float:
    """Gestalt/RapidFuzz token_set_ratio в [0, 1]."""
    r = norm_text(reference)
    h = norm_text(hypothesis)
    if not r and not h:
        return 1.0
    return float(fuzz.token_set_ratio(r, h)) / 100.0


# --------------------------------------------------------------------------
# Field-level: для baseline OCR.
# --------------------------------------------------------------------------
def field_recall(field_value: str, ocr_text: str,
                 threshold: float = 0.85) -> tuple[float, bool]:
    """Насколько ожидаемое значение поля «нашлось» в OCR-выдаче.

    Возвращает:
        sim_score: float ∈ [0, 1] — максимальное частичное совпадение
                   (partial_ratio) ожидаемого значения с OCR-выдачей.
        found: bool — sim_score ≥ threshold.
    """
    fv = norm_text(field_value)
    ot = norm_text(ocr_text)
    if not fv:
        return 1.0, True  # пустое поле — нечего искать
    if not ot:
        return 0.0, False
    score = float(fuzz.partial_ratio(fv, ot)) / 100.0
    return score, score >= threshold


def build_reference_text(gold_record: dict,
                         fields: Iterable[str] = GT_VERBATIM_FIELDS) -> str:
    """Склеивает значения указанных полей в один эталонный текст."""
    parts = []
    for f in fields:
        v = gold_record.get(f)
        if v is None:
            continue
        v = str(v).strip()
        if v:
            parts.append(v)
    return "\n".join(parts)
