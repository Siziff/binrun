"""Оценка VLM-выводов против GT.

В отличие от Эксп. 1-4 (где OCR читал «всё подряд» и precision определить
было трудно), VLM выдаёт структурированный JSON: для каждого поля у нас
есть либо строка (предсказание), либо null. Можно честно считать TP/FP/FN.

Поле:
    gt_value: GT (None или строка)
    pred_value: VLM (None или строка)

Решение (per field, per image):
    gt_has  = gt_value is not None and str(gt_value).strip() != ""
    pred_has = pred_value is not None and str(pred_value).strip() != ""

    sim = similarity(gt_value, pred_value)  # rapidfuzz.token_set_ratio / 100
    match = sim >= TAU   (по умолчанию 0.85)

    TP: gt_has & pred_has & match
    FP: pred_has & not (gt_has & match)
        — то есть либо GT нет и мы что-то выдали, либо GT есть, но похоже не на него
    FN: gt_has & not (pred_has & match)
    TN: not gt_has & not pred_has

Метрики:
    Per-field: P, R, F1, soft_recall (среднее sim среди gt_has).
    Micro: суммируем TP/FP/FN по всем полям и считаем общий F1.
    Macro: усредняем per-field F1.

Скрипт читает все outputs/*.jsonl и для каждого пишет
    outputs/<basename>_metrics.json.

Также формирует сводную таблицу (CSV + Markdown).
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.pop("PYTHONSTARTUP", None)
os.environ.pop("PYTHONPATH", None)

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz


HERE = Path(__file__).resolve().parent
INPUTS = HERE / "merged_inputs.jsonl"
OUT_DIR = HERE / "outputs"
SUMMARY_DIR = HERE / "summary"
SUMMARY_DIR.mkdir(exist_ok=True)

TAU = 0.85   # порог matching (на шкале [0, 1])

FIELDS = [
    "verbatim_identification", "verbatim_collector", "verbatim_locality",
    "verbatim_collection_date", "verbatim_substrate", "verbatim_habitat",
    "verbatim_identification_date", "field_number",
    "verbatim_latitude", "verbatim_longitude", "verbatim_elevation",
    "exsiccata_title", "exsiccata_number", "type_status",
    "catalog_number", "scientific_name", "determiner", "collector",
    "country", "organism_remarks", "identification_remarks",
]


def has_value(v: Any) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    return bool(s) and s.lower() not in ("null", "none", "n/a", "na")


def similarity(a: Any, b: Any) -> float:
    """RapidFuzz token_set_ratio normalized в [0, 1]."""
    s1 = "" if a is None else str(a).strip()
    s2 = "" if b is None else str(b).strip()
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return fuzz.token_set_ratio(s1.lower(), s2.lower()) / 100.0


def read_jsonl(p: Path):
    """Tolerant JSONL reader: некоторые модели вставляют literal newlines
    в свои raw_response, ломая JSONL. Накапливаем буфер, начиная новый
    объект ТОЛЬКО когда строка начинается с `{`."""
    buf = ""
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            # начало нового объекта?
            if raw.lstrip().startswith('{"') and buf.strip():
                # сначала попробуем yield накопленный буфер
                try:
                    yield json.loads(buf)
                except Exception:
                    pass
                buf = ""
            buf += raw
    if buf.strip():
        try:
            yield json.loads(buf)
        except Exception:
            pass


def load_gold() -> dict[str, dict]:
    """{image: gt_dict}."""
    out = {}
    for rec in read_jsonl(INPUTS):
        out[rec["image"]] = rec.get("gt") or {}
    return out


def evaluate_file(jsonl_path: Path, gold: dict[str, dict]) -> dict:
    """Возвращает сводный dict с per-field и aggregate."""
    per_field = {f: {"tp": 0, "fp": 0, "fn": 0, "tn": 0,
                      "sim_sum_when_gt": 0.0, "n_gt": 0,
                      "sims_at_gt": []} for f in FIELDS}
    n_records = 0
    n_parse_ok = 0
    n_parse_fail = 0
    runtimes = []

    for rec in read_jsonl(jsonl_path):
        n_records += 1
        if rec.get("parse_ok"):
            n_parse_ok += 1
        else:
            n_parse_fail += 1
        runtimes.append(rec.get("runtime_seconds") or 0)

        img = rec["image"]
        gt = gold.get(img, {})
        pred = rec.get("parsed_json") or {}

        for f in FIELDS:
            gt_v = gt.get(f)
            pred_v = pred.get(f)
            gt_h = has_value(gt_v)
            pred_h = has_value(pred_v)

            sim = similarity(gt_v, pred_v) if (gt_h and pred_h) else 0.0
            match = sim >= TAU

            if gt_h:
                per_field[f]["n_gt"] += 1
                per_field[f]["sim_sum_when_gt"] += sim if pred_h else 0.0
                per_field[f]["sims_at_gt"].append(sim if pred_h else 0.0)

            if gt_h and pred_h and match:
                per_field[f]["tp"] += 1
            elif pred_h and not (gt_h and match):
                per_field[f]["fp"] += 1
            elif gt_h and not (pred_h and match):
                per_field[f]["fn"] += 1
            else:
                per_field[f]["tn"] += 1

    # Финализация
    pf_out = {}
    sum_tp, sum_fp, sum_fn = 0, 0, 0
    soft_recall_per_field = []
    f1s = []
    precisions = []
    recalls = []

    for f, c in per_field.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        soft_r = c["sim_sum_when_gt"] / c["n_gt"] if c["n_gt"] else 0.0
        pf_out[f] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": c["tn"],
            "n_gt": c["n_gt"], "support": c["n_gt"],
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "soft_recall": round(soft_r, 4),
        }
        sum_tp += tp; sum_fp += fp; sum_fn += fn
        if c["n_gt"] > 0:   # включаем в макро только поля с support
            f1s.append(f1)
            precisions.append(p)
            recalls.append(r)
            soft_recall_per_field.append(soft_r)

    micro_p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) else 0.0
    micro_r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    return {
        "n_records": n_records,
        "n_parse_ok": n_parse_ok,
        "n_parse_fail": n_parse_fail,
        "parse_rate": round(n_parse_ok / max(1, n_records), 4),
        "mean_runtime_s": round(st.mean(runtimes), 2) if runtimes else 0.0,
        "median_runtime_s": round(st.median(runtimes), 2) if runtimes else 0.0,
        "aggregate": {
            "micro_precision": round(micro_p, 4),
            "micro_recall":    round(micro_r, 4),
            "micro_f1":        round(micro_f1, 4),
            "macro_precision": round(st.mean(precisions), 4) if precisions else 0.0,
            "macro_recall":    round(st.mean(recalls), 4) if recalls else 0.0,
            "macro_f1":        round(st.mean(f1s), 4) if f1s else 0.0,
            "soft_macro_recall": round(st.mean(soft_recall_per_field), 4) if soft_recall_per_field else 0.0,
            "n_fields_with_support": len(f1s),
        },
        "per_field": pf_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="*.jsonl",
                    help="glob внутри outputs/")
    args = ap.parse_args()

    gold = load_gold()
    rows = []
    for p in sorted(OUT_DIR.glob(args.pattern)):
        if "shard" in p.name or "metrics" in p.name:
            continue
        if p.name.endswith("_metrics.jsonl"):
            continue
        if p.stat().st_size == 0:
            continue
        print(f"\n=== {p.name} ===")
        metrics = evaluate_file(p, gold)
        agg = metrics["aggregate"]
        print(f"  records={metrics['n_records']}  parse_ok={metrics['n_parse_ok']}  "
              f"parse_rate={metrics['parse_rate']}  mean_runtime={metrics['mean_runtime_s']}s")
        print(f"  micro F1={agg['micro_f1']:.3f}  macro F1={agg['macro_f1']:.3f}  "
              f"soft-R={agg['soft_macro_recall']:.3f}")
        # сохраняем JSON
        out_metrics = p.with_name(p.stem + "__metrics.json")
        out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

        # парсим модель и режим из имени файла
        name = p.stem
        # формат: <model_slug>__mode<A|B>
        if "__mode" in name:
            slug, mode_part = name.rsplit("__mode", 1)
            mode = mode_part.split("__")[0]
        else:
            slug, mode = name, "?"

        rows.append({
            "model": slug.replace("__", "/"),
            "mode": mode,
            "records": metrics["n_records"],
            "parse_rate": metrics["parse_rate"],
            "micro_F1": agg["micro_f1"],
            "macro_F1": agg["macro_f1"],
            "soft_macro_R": agg["soft_macro_recall"],
            "mean_runtime_s": metrics["mean_runtime_s"],
        })

    if rows:
        df = pd.DataFrame(rows).sort_values(["mode", "micro_F1"], ascending=[True, False])
        df.to_csv(SUMMARY_DIR / "summary.csv", index=False)
        md = df.to_markdown(index=False, floatfmt=".4f")
        (SUMMARY_DIR / "summary.md").write_text("# Exp.5 VLM summary\n\n" + md + "\n")
        print("\n=== Sweep ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
