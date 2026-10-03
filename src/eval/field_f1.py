"""F1-метрики на уровне полей для Эксп. 1 (baseline OCR).

Определения:
    - Per-image, per-field положительность определяется наличием значения
      в ground-truth (gt_has = bool(value)).
    - Per-image, per-field предсказание «найдено» определяется через
      partial_ratio(gt_value, ocr_full_text) ≥ threshold (по умолчанию 0.85).
    - TP: gt_has=True  ∧ found=True
      FN: gt_has=True  ∧ found=False
      FP: gt_has=False ∧ found=True   — для baseline-OCR этот случай определить
          корректно нельзя (OCR читает весь текст, его нельзя «обвинить»
          в FP по структуре). Поэтому в baseline-режиме мы считаем
          ТОЛЬКО Recall (он же = TPR), а Precision/F1 — отдельно через
          альтернативу: predicted_has = (sim_score > 0.0). При таком
          определении на baseline-OCR Precision стремится к Recall,
          но шкала всё равно интерпретируема и позволяет ранжировать движки.
    - Также считаем similarity-weighted F1: score_TP = max(0, sim - threshold)
      нормируется в [0, 1] и используется как «soft hit».
"""
from __future__ import annotations

from dataclasses import dataclass

from src.eval.metrics import (
    GT_NORMALIZED_FIELDS,
    GT_VERBATIM_FIELDS,
    field_recall,
    norm_text,
)


# Какие поля считать «структурно проверяемыми» для baseline-OCR.
DEFAULT_FIELDS = (*GT_VERBATIM_FIELDS, *GT_NORMALIZED_FIELDS)


@dataclass
class FieldCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    # для soft-F1
    sim_sum_when_gt: float = 0.0
    n_when_gt: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    @property
    def soft_recall(self) -> float:
        """Средний sim-score по картинкам, где поле было в GT.

        Это threshold-free прокси для recall: ≥ 0.0, до 1.0.
        Хорошо отражает «насколько хорошо OCR прочитал поле».
        """
        return self.sim_sum_when_gt / self.n_when_gt if self.n_when_gt else 0.0


def compute_field_f1(
    per_image_metrics: list[dict],
    fields: tuple[str, ...] = DEFAULT_FIELDS,
    threshold: float = 0.85,
) -> dict:
    """Считает per-field и aggregate F1 на основе результатов evaluate().

    Args:
        per_image_metrics: список словарей из `evaluate()` с ключами
            `fields_score`, `fields_found`, плюс полные GT-значения нужны;
            мы их берём из переданного `gold_by_image`.
        fields: поля для оценки.
        threshold: порог partial_ratio для «найдено».

    Returns:
        {
            "per_field": {field: {precision, recall, f1, support, soft_recall}},
            "macro": {precision, recall, f1},
            "micro": {precision, recall, f1},
            "soft_macro_recall": float,
        }
    """
    counts: dict[str, FieldCounts] = {f: FieldCounts() for f in fields}

    for m in per_image_metrics:
        scores = m.get("fields_score") or {}
        founds = m.get("fields_found") or {}
        gt_has = m.get("gt_has") or {}
        # gt_has должен быть заполнен в evaluate(); если нет — упасть в derive:
        for f in fields:
            score = float(scores.get(f, 0.0))
            found = bool(founds.get(f, False))
            had = bool(gt_has.get(f, False))

            if had:
                counts[f].n_when_gt += 1
                counts[f].sim_sum_when_gt += score
                if found:
                    counts[f].tp += 1
                else:
                    counts[f].fn += 1
            else:
                if found:
                    counts[f].fp += 1
                else:
                    counts[f].tn += 1

    per_field = {}
    for f, c in counts.items():
        per_field[f] = {
            "precision": c.precision,
            "recall": c.recall,
            "f1": c.f1,
            "tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn,
            "support": c.support,
            "soft_recall": c.soft_recall,
        }

    # macro: простое среднее по полям (только тем, у которых support > 0).
    informative = [f for f in fields if counts[f].support > 0]
    if informative:
        macro_p = sum(per_field[f]["precision"] for f in informative) / len(informative)
        macro_r = sum(per_field[f]["recall"] for f in informative) / len(informative)
        macro_f = sum(per_field[f]["f1"] for f in informative) / len(informative)
        soft_macro = sum(per_field[f]["soft_recall"] for f in informative) / len(informative)
    else:
        macro_p = macro_r = macro_f = soft_macro = 0.0

    # micro: суммируем все TP/FP/FN.
    sum_tp = sum(c.tp for c in counts.values())
    sum_fp = sum(c.fp for c in counts.values())
    sum_fn = sum(c.fn for c in counts.values())
    micro_p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) else 0.0
    micro_r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) else 0.0
    micro_f = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    return {
        "per_field": per_field,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f},
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f},
        "soft_macro_recall": soft_macro,
        "threshold": threshold,
        "n_fields_with_support": len(informative),
    }


def derive_gt_has(gt_record: dict, fields: tuple[str, ...] = DEFAULT_FIELDS) -> dict[str, bool]:
    """Маркер «есть ли значение этого поля в gold для данной картинки»."""
    out = {}
    for f in fields:
        v = gt_record.get(f)
        out[f] = bool(v and str(v).strip())
    return out
