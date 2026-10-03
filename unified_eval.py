"""Унифицированный evaluator для всех экспериментов проекта.

Считает единый набор метрик для каждого эксперимента (1-5):
    - CER, WER       (char/word error rate, RapidFuzz character-level edit distance)
    - sim            (mean similarity between predicted text and GT text)
    - hard-R         (recall с порогом sim >= 0.85)
    - soft-R         (mean similarity per field, где gt_has=True)
    - F1             (== F1-micro, default)
    - F1-micro       (TP/FP/FN суммируем по всем полям)
    - F1-macro       (mean F1 по полям с support>0)
    - sec/img        (mean runtime per image)

Для OCR-экспериментов (1-4): берёт `full_text` каждой OCR-выдачи и сравнивает
с конкатенацией всех verbatim+normalized полей GT. Per-field F1 считается
через partial_ratio: поле «найдено», если max similarity между значением GT
и подстрокой `full_text` >= 0.85.

Для VLM-эксперимента (5): берёт `parsed_json` и сравнивает поле-в-поле с GT.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.pop("PYTHONSTARTUP", None)
os.environ.pop("PYTHONPATH", None)

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any, Iterable

import jiwer
import pandas as pd
from rapidfuzz import fuzz


ROOT = Path(__file__).resolve().parent

# Поля 21-полевой схемы. Используются И для агрегации текста (склейка GT),
# И для per-field F1 (VLM).
FIELDS = [
    "verbatim_identification", "verbatim_collector", "verbatim_locality",
    "verbatim_collection_date", "verbatim_substrate", "verbatim_habitat",
    "verbatim_identification_date", "field_number",
    "verbatim_latitude", "verbatim_longitude", "verbatim_elevation",
    "exsiccata_title", "exsiccata_number", "type_status",
    "catalog_number", "scientific_name", "determiner", "collector",
    "country", "organism_remarks", "identification_remarks",
]

# Порог для hard-recall.
TAU_HARD = 0.85


# -------------------- helpers --------------------

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


def partial_sim(needle: Any, haystack: Any) -> float:
    """Похоже на partial_ratio: ищем `needle` как подстроку в `haystack`."""
    s1 = "" if needle is None else str(needle).strip()
    s2 = "" if haystack is None else str(haystack).strip()
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return fuzz.partial_ratio(s1.lower(), s2.lower()) / 100.0


def gt_concat_text(gt: dict) -> str:
    """Склейка всех GT-полей в один текст (для CER/WER/sim)."""
    parts = []
    for f in FIELDS:
        v = gt.get(f)
        if has_value(v):
            parts.append(str(v).strip())
    return " | ".join(parts)


def cer_metric(ref: str, hyp: str) -> float:
    """Character-Error-Rate, через jiwer."""
    ref = (ref or "").strip()
    hyp = (hyp or "").strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    try:
        return float(jiwer.cer(ref, hyp))
    except Exception:
        return float("nan")


def wer_metric(ref: str, hyp: str) -> float:
    ref = (ref or "").strip()
    hyp = (hyp or "").strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    try:
        return float(jiwer.wer(ref, hyp))
    except Exception:
        return float("nan")


def read_jsonl(p: Path) -> Iterable[dict]:
    """Tolerant JSONL reader (поскольку VLM иногда вставляет \n в raw)."""
    buf = ""
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            if raw.lstrip().startswith('{"') and buf.strip():
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


# -------------------- GT loader --------------------

def load_gold() -> dict[str, dict]:
    """Из labels.jsonl: {image_name: {field: value}}."""
    drop = {"image_files", "has_image", "is_image_flag",
            "catalog_number_raw", "specimen_remarks", "specimen_group",
            "verbatim_determiner"}
    out: dict[str, dict] = {}
    gold_path = ROOT / "data" / "gold" / "labels.jsonl"
    for rec in read_jsonl(gold_path):
        files = rec.get("image_files") or []
        if not files:
            continue
        fields = {k: v for k, v in rec.items() if k not in drop}
        for fn in files:
            if fn not in out:
                out[fn] = fields
    return out


# -------------------- evaluators --------------------

def eval_ocr_jsonl(jsonl_path: Path, gold: dict[str, dict]) -> dict:
    """Эксп.1-4: OCR-выдача (одна строка full_text)."""
    n = 0
    cers, wers, sims = [], [], []
    runtimes = []

    # per-field stats
    pf_tp = {f: 0 for f in FIELDS}
    pf_fp = {f: 0 for f in FIELDS}
    pf_fn = {f: 0 for f in FIELDS}
    pf_n_gt = {f: 0 for f in FIELDS}
    pf_sim_sum = {f: 0.0 for f in FIELDS}

    for rec in read_jsonl(jsonl_path):
        n += 1
        img = rec.get("image")
        gt = gold.get(img, {})
        gt_text = gt_concat_text(gt)
        hyp_text = (rec.get("full_text") or "")
        cers.append(cer_metric(gt_text, hyp_text))
        wers.append(wer_metric(gt_text, hyp_text))
        sims.append(similarity(gt_text, hyp_text))
        rt = rec.get("runtime_seconds")
        if rt is not None:
            runtimes.append(rt)

        for f in FIELDS:
            gt_v = gt.get(f)
            if not has_value(gt_v):
                continue
            pf_n_gt[f] += 1
            sim = partial_sim(gt_v, hyp_text)
            pf_sim_sum[f] += sim
            if sim >= TAU_HARD:
                pf_tp[f] += 1
            else:
                pf_fn[f] += 1
        # FP для OCR-режима плохо определены (full_text — это весь текст).
        # Для совместимости с VLM-метриками: FP = 0 (OCR не «утверждает» полей).

    # aggregate
    micro_tp = sum(pf_tp.values())
    micro_fp = sum(pf_fp.values())
    micro_fn = sum(pf_fn.values())
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    macro_f1s, macro_rs, soft_rs = [], [], []
    for f in FIELDS:
        tp, fp, fn = pf_tp[f], pf_fp[f], pf_fn[f]
        if pf_n_gt[f] == 0:
            continue
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        sr = pf_sim_sum[f] / pf_n_gt[f]
        macro_f1s.append(f1)
        macro_rs.append(r)
        soft_rs.append(sr)

    return {
        "n_records": n,
        "CER": round(st.mean(cers), 4) if cers else None,
        "WER": round(st.mean(wers), 4) if wers else None,
        "sim": round(st.mean(sims), 4) if sims else None,
        "hard_R": round(st.mean(macro_rs), 4) if macro_rs else None,
        "soft_R": round(st.mean(soft_rs), 4) if soft_rs else None,
        "F1": round(micro_f1, 4),
        "F1_micro": round(micro_f1, 4),
        "F1_macro": round(st.mean(macro_f1s), 4) if macro_f1s else None,
        "sec_per_img": round(st.mean(runtimes), 2) if runtimes else None,
    }


def eval_vlm_jsonl(jsonl_path: Path, gold: dict[str, dict]) -> dict:
    """Эксп.5: VLM-выдача (parsed_json по 21 полю)."""
    n = 0
    cers, wers, sims = [], [], []
    runtimes = []

    pf_tp = {f: 0 for f in FIELDS}
    pf_fp = {f: 0 for f in FIELDS}
    pf_fn = {f: 0 for f in FIELDS}
    pf_n_gt = {f: 0 for f in FIELDS}
    pf_sim_sum = {f: 0.0 for f in FIELDS}

    n_parse_ok = 0
    for rec in read_jsonl(jsonl_path):
        n += 1
        img = rec.get("image")
        gt = gold.get(img, {})
        gt_text = gt_concat_text(gt)
        pred = rec.get("parsed_json") or {}
        if rec.get("parse_ok"):
            n_parse_ok += 1

        # склеиваем pred-поля для CER/WER/sim
        pred_text_parts = []
        for f in FIELDS:
            v = pred.get(f)
            if has_value(v):
                pred_text_parts.append(str(v).strip())
        hyp_text = " | ".join(pred_text_parts)

        cers.append(cer_metric(gt_text, hyp_text))
        wers.append(wer_metric(gt_text, hyp_text))
        sims.append(similarity(gt_text, hyp_text))
        rt = rec.get("runtime_seconds")
        if rt is not None:
            runtimes.append(rt)

        # per-field
        for f in FIELDS:
            gt_v = gt.get(f)
            pred_v = pred.get(f)
            gt_h = has_value(gt_v)
            pred_h = has_value(pred_v)
            sim = similarity(gt_v, pred_v) if (gt_h and pred_h) else 0.0
            match = sim >= TAU_HARD

            if gt_h:
                pf_n_gt[f] += 1
                pf_sim_sum[f] += sim if pred_h else 0.0

            if gt_h and pred_h and match:
                pf_tp[f] += 1
            elif pred_h and not (gt_h and match):
                pf_fp[f] += 1
            elif gt_h and not (pred_h and match):
                pf_fn[f] += 1

    micro_tp = sum(pf_tp.values())
    micro_fp = sum(pf_fp.values())
    micro_fn = sum(pf_fn.values())
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    macro_f1s, macro_rs, soft_rs = [], [], []
    for f in FIELDS:
        if pf_n_gt[f] == 0:
            continue
        tp, fp, fn = pf_tp[f], pf_fp[f], pf_fn[f]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        sr = pf_sim_sum[f] / pf_n_gt[f]
        macro_f1s.append(f1)
        macro_rs.append(r)
        soft_rs.append(sr)

    return {
        "n_records": n,
        "parse_rate": round(n_parse_ok / max(1, n), 4),
        "CER": round(st.mean(cers), 4) if cers else None,
        "WER": round(st.mean(wers), 4) if wers else None,
        "sim": round(st.mean(sims), 4) if sims else None,
        "hard_R": round(st.mean(macro_rs), 4) if macro_rs else None,
        "soft_R": round(st.mean(soft_rs), 4) if soft_rs else None,
        "F1": round(micro_f1, 4),
        "F1_micro": round(micro_f1, 4),
        "F1_macro": round(st.mean(macro_f1s), 4) if macro_f1s else None,
        "sec_per_img": round(st.mean(runtimes), 2) if runtimes else None,
    }


# -------------------- discovery --------------------

OCR_EXPS = {
    "exp01": "experiments/exp01_ocr_baseline/outputs",
    "exp02": "experiments/exp02_preprocessed/outputs",
    "exp03": "experiments/exp03_hybrid_orient/outputs",
    "exp04": "experiments/exp04_conservative_orient/outputs",
}
VLM_EXP = ("exp05", "experiments/exp05_vlm_normalizer/outputs")


def discover(exp_id: str, exp_dir: Path) -> list[tuple[str, str, Path]]:
    """Возвращает [(exp_id, model_label, path), ...]."""
    out = []
    for p in sorted(exp_dir.glob("*.jsonl")):
        if "shard" in p.name or "_smoke" in p.name:
            continue
        if p.stat().st_size == 0:
            continue
        # выкинем индексные/служебные:
        if p.name.startswith("_") or p.name == "preproc_index.json":
            continue
        # Для OCR-экспериментов имена: tesseract.jsonl, surya.jsonl и т.п.
        # Для VLM: Qwen__Qwen3-VL-30B-A3B-Instruct__modeA.jsonl
        out.append((exp_id, p.stem, p))
    return out


# -------------------- main --------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "summary_all.csv"))
    ap.add_argument("--md", default=str(ROOT / "summary_all.md"))
    args = ap.parse_args()

    gold = load_gold()
    print(f"GT records: {len(gold)}", flush=True)

    rows = []

    for exp_id, sub in OCR_EXPS.items():
        exp_dir = ROOT / sub
        if not exp_dir.exists():
            continue
        for eid, model, path in discover(exp_id, exp_dir):
            try:
                m = eval_ocr_jsonl(path, gold)
            except Exception as e:
                print(f"  [SKIP] {path}: {e}")
                continue
            row = {"exp": eid, "model": model, "mode": "ocr"}
            row.update(m)
            rows.append(row)
            print(f"  {eid}/{model:<22}: n={m['n_records']:>3}  "
                  f"sim={m['sim']:.3f}  micro_F1={m['F1_micro']:.3f}", flush=True)

    # Эксп.5 — VLM
    exp_dir = ROOT / VLM_EXP[1]
    for eid, model, path in discover(VLM_EXP[0], exp_dir):
        # mode из имени: …__modeA.jsonl или …__modeB.jsonl
        mode_part = ""
        if "__mode" in path.stem:
            mode_part = "vlm-A" if path.stem.endswith("modeA") else (
                "vlm-B" if path.stem.endswith("modeB") else "vlm")
            slug = path.stem.rsplit("__mode", 1)[0]
        else:
            slug = path.stem
            mode_part = "vlm"
        try:
            m = eval_vlm_jsonl(path, gold)
        except Exception as e:
            print(f"  [SKIP] {path}: {e}")
            continue
        row = {"exp": eid, "model": slug.replace("__", "/"), "mode": mode_part}
        row.update(m)
        rows.append(row)
        print(f"  {eid}/{slug:<35} {mode_part}: n={m['n_records']:>3}  "
              f"sim={m['sim']:.3f}  micro_F1={m['F1_micro']:.3f}", flush=True)

    if not rows:
        print("No experiments found.")
        return

    df = pd.DataFrame(rows)
    # Удобный порядок столбцов
    col_order = ["exp", "model", "mode", "n_records", "parse_rate",
                 "CER", "WER", "sim", "hard_R", "soft_R",
                 "F1", "F1_micro", "F1_macro", "sec_per_img"]
    cols = [c for c in col_order if c in df.columns]
    df = df[cols]
    df = df.sort_values(["exp", "F1_micro"], ascending=[True, False])

    df.to_csv(args.out, index=False)
    print(f"\nCSV → {args.out}")

    # Markdown-таблица: одна на все эксперименты
    md_lines = ["# Сводная таблица метрик по всем экспериментам\n",
                "Метрики: WER, CER, sim (RapidFuzz token_set_ratio), hard-R (recall @ τ=0.85),",
                "soft-R (mean similarity per field), F1 (= F1_micro), F1_micro, F1_macro,",
                "sec/img (runtime per image).\n",
                df.to_markdown(index=False, floatfmt=".3f")]
    Path(args.md).write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"MD  → {args.md}")

    # Печать в консоль
    print("\n=== ALL EXPERIMENTS ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
