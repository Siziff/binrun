#!/usr/bin/env python3
"""Exp 7: Pure text-quality benchmark against gold_text.jsonl (50 files).

For each model output JSONL, extracts plain-text per image (different fields
per model family), normalises it, then computes:
  - CER (character error rate, Levenshtein / len(gold_chars))
  - WER (word error rate, token-level Levenshtein / len(gold_words))
  - sim (1 - char_edit_ratio, the SequenceMatcher ratio)
  - rec_chars (chars(prediction)/chars(gold)) -- coverage proxy
  - sec_per_img (mean ocr_runtime_s if present)

Outputs:
  exp07_text_gold/results.csv         -- all models
  exp07_text_gold/results.md          -- ranking table
  exp07_text_gold/per_image/<m>.csv   -- per-file breakdown (optional)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path("/workspace-SR008.fs2/mikheev-kandy/TD")
GOLD_PATH = ROOT / "experiments/exp07_text_gold/gold_text.jsonl"
OUTDIR = ROOT / "experiments/exp07_text_gold"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---- text extraction per model family ----------------------------------------

VERBATIM_FIELDS = [
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
]


def _from_json(d: dict) -> str:
    """Concatenate verbatim_* fields + exsiccata_* + catalog_number for json-output models."""
    parts = []
    for k in VERBATIM_FIELDS:
        v = d.get(k)
        if v is None or v == "":
            continue
        parts.append(str(v))
    return "\n".join(parts)


def _strip_html(s: str) -> str:
    # remove <tags ...> markup; keep inner text
    s = re.sub(r"<[^>]+>", " ", s)
    # remove markdown bold/italic markers
    s = re.sub(r"\*+", "", s)
    s = re.sub(r"_+", "", s)
    return s


def extract_pred(rec: dict, model_tag: str) -> str:
    # Priority 1: explicit raw ocr_text (paddle, deepseek, got, unlimited, glm-ocr, logics)
    txt = rec.get("ocr_text")
    if txt:
        # logics-parsing-v2 returns html-style markdown in ocr_text; strip markup
        if "<" in txt and ">" in txt:
            return _strip_html(txt)
        return txt
    # chandra raw markdown
    md = rec.get("ocr_markdown")
    if md:
        return _strip_html(md)
    # Qwen/InternVL JSON modes
    pj = rec.get("parsed_json")
    if isinstance(pj, dict):
        return _from_json(pj)
    raw = rec.get("raw_response")
    if raw:
        # try to parse JSON; if fails, return raw text
        try:
            j = json.loads(raw)
            if isinstance(j, dict):
                return _from_json(j)
        except Exception:
            pass
        return raw
    return ""


# ---- normalisation -----------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def normalise(text: str) -> str:
    if not text:
        return ""
    # NFKC + casefold + collapse whitespace + strip punctuation noise
    t = unicodedata.normalize("NFKC", text)
    t = t.casefold()
    # replace dashes/quotes variants
    t = t.replace("—", "-").replace("–", "-").replace("‐", "-")
    t = t.replace("„", '"').replace(""", '"').replace(""", '"')
    t = t.replace("'", "'").replace("'", "'")
    # drop pure punctuation noise (keep letters/digits/whitespace/dot/comma/dash/slash)
    t = re.sub(r"[^\w\s.,\-\/°'\"μäöüßéèêàçñ]", " ", t, flags=re.UNICODE)
    t = _WS_RE.sub(" ", t).strip()
    return t


# ---- metrics ----------------------------------------------------------------


def _levenshtein(a, b) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr[j] = min(ins, dele, sub)
        prev = curr
    return prev[-1]


def cer(gold: str, pred: str) -> float:
    if not gold:
        return 0.0 if not pred else 1.0
    return _levenshtein(gold, pred) / len(gold)


def wer(gold: str, pred: str) -> float:
    g = gold.split()
    p = pred.split()
    if not g:
        return 0.0 if not p else 1.0
    return _levenshtein(g, p) / len(g)


def sim(gold: str, pred: str) -> float:
    if not gold and not pred:
        return 1.0
    return SequenceMatcher(None, gold, pred).ratio()


# ---- main loop -------------------------------------------------------------


@dataclass
class ModelSpec:
    tag: str  # short label for table
    path: str  # relative to ROOT
    family: str  # ocr-raw, chandra-md, vlm-json, ocr+fmt-mixed


# (tag, jsonl, family). Sorted into groups.
MODELS = [
    # --- direct OCR (raw text) ---
    ModelSpec("paddleocr-vl",     "experiments/exp05_vlm_normalizer/outputs/paddleocr-vl-1.6__raw.jsonl",          "ocr-raw"),
    ModelSpec("deepseek-ocr",     "experiments/exp05_vlm_normalizer/outputs/deepseek-ocr__raw.jsonl",              "ocr-raw"),
    ModelSpec("got-ocr2",         "experiments/exp05_vlm_normalizer/outputs/got-ocr2__raw.jsonl",                  "ocr-raw"),
    ModelSpec("unlimited-ocr",    "experiments/exp05_vlm_normalizer/outputs/unlimited-ocr__raw.jsonl",             "ocr-raw"),
    ModelSpec("glm-ocr",          "experiments/exp05_vlm_normalizer/outputs/glm-ocr__formatted_fs.jsonl",          "ocr-raw"),
    ModelSpec("glm-ocr-RAW50",    "experiments/exp05_vlm_normalizer/outputs/glm-ocr__raw__exp07_50.jsonl",         "ocr-raw"),
    ModelSpec("logics-parsing-v2","experiments/exp05_vlm_normalizer/outputs/logics-parsing-v2__formatted_fs.jsonl", "ocr-raw"),
    ModelSpec("logics-parsing-v2-RAW50", "experiments/exp05_vlm_normalizer/outputs/logics-parsing-v2__raw__exp07_50.jsonl", "ocr-raw"),
    # --- chandra markdown ---
    ModelSpec("chandra-md",       "experiments/exp05_vlm_normalizer/outputs/datalab-to__chandra-ocr-2__modeA__raw_md.jsonl", "chandra-md"),
    # --- VLM JSON (verbatim_* concat) ---
    ModelSpec("Qwen3-VL-8B-Instruct-A",      "experiments/exp05_vlm_normalizer/outputs/Qwen__Qwen3-VL-8B-Instruct__modeA.jsonl",         "vlm-json"),
    ModelSpec("Qwen3-VL-8B-Instruct-A_fs",   "experiments/exp05_vlm_normalizer/outputs/Qwen__Qwen3-VL-8B-Instruct__modeA_fs.jsonl",      "vlm-json"),
    ModelSpec("Qwen3-VL-30B-A3B-Instruct-A", "experiments/exp05_vlm_normalizer/outputs/Qwen__Qwen3-VL-30B-A3B-Instruct__modeA.jsonl",    "vlm-json"),
    ModelSpec("Qwen3-VL-30B-A3B-Thinking-A", "experiments/exp05_vlm_normalizer/outputs/Qwen__Qwen3-VL-30B-A3B-Thinking__modeA.jsonl",    "vlm-json"),
    ModelSpec("InternVL3-8B-A",              "experiments/exp05_vlm_normalizer/outputs/OpenGVLab__InternVL3-8B-hf__modeA.jsonl",         "vlm-json"),
]


def _tolerant_jsonl(path: Path):
    """Same idea as evaluate.py read_jsonl: accumulate until next '{"' line."""
    buf = ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith('{"') and buf:
                try:
                    yield json.loads(buf)
                except Exception:
                    pass
                buf = ""
            buf += line
    if buf.strip():
        try:
            yield json.loads(buf)
        except Exception:
            pass


def load_gold() -> dict[str, str]:
    out = {}
    for rec in _tolerant_jsonl(GOLD_PATH):
        img = rec.get("image")
        txt = rec.get("full_text") or rec.get("text") or ""
        if img and txt:
            out[img] = txt
    return out


def load_preds(path: Path) -> dict[str, dict]:
    out = {}
    for rec in _tolerant_jsonl(path):
        img = rec.get("image")
        if img:
            out[img] = rec
    return out


def evaluate_model(spec: ModelSpec, gold: dict[str, str]) -> dict:
    path = ROOT / spec.path
    if not path.exists():
        return {"tag": spec.tag, "family": spec.family, "n": 0, "missing": True}
    preds_all = load_preds(path)
    n = 0
    n_missing = 0
    cer_acc = wer_acc = sim_acc = rec_acc = 0.0
    runtime_acc = 0.0
    runtime_n = 0
    per_image = []
    for img, gtxt in gold.items():
        rec = preds_all.get(img)
        if rec is None:
            n_missing += 1
            continue
        n += 1
        pred_raw = extract_pred(rec, spec.tag)
        g = normalise(gtxt)
        p = normalise(pred_raw)
        c = cer(g, p)
        w = wer(g, p)
        s = sim(g, p)
        r = (len(p) / max(1, len(g))) if g else 0.0
        cer_acc += c
        wer_acc += w
        sim_acc += s
        rec_acc += r
        per_image.append({"image": img, "cer": c, "wer": w, "sim": s, "rec": r,
                          "pred_chars": len(p), "gold_chars": len(g)})
        rt = rec.get("ocr_runtime_s") or rec.get("runtime_seconds") or rec.get("total_runtime_s")
        if isinstance(rt, (int, float)):
            runtime_acc += float(rt)
            runtime_n += 1
    if n == 0:
        return {"tag": spec.tag, "family": spec.family, "n": 0, "missing": True}
    return {
        "tag": spec.tag,
        "family": spec.family,
        "n": n,
        "n_missing": n_missing,
        "cer": cer_acc / n,
        "wer": wer_acc / n,
        "sim": sim_acc / n,
        "rec": rec_acc / n,
        "sec_per_img": (runtime_acc / runtime_n) if runtime_n else None,
        "per_image": per_image,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-per-image", action="store_true")
    args = ap.parse_args()

    gold = load_gold()
    if len(gold) != 50:
        print(f"WARN: gold has {len(gold)} entries (expected 50)", file=sys.stderr)
    print(f"Loaded {len(gold)} gold entries")

    rows = []
    for spec in MODELS:
        print(f"  evaluating {spec.tag}...", end=" ", flush=True)
        res = evaluate_model(spec, gold)
        if res.get("missing"):
            print("MISSING")
            continue
        rows.append(res)
        print(f"n={res['n']} cer={res['cer']:.3f} wer={res['wer']:.3f} sim={res['sim']:.3f}")
        if args.write_per_image:
            pidir = OUTDIR / "per_image"
            pidir.mkdir(exist_ok=True)
            with open(pidir / f"{spec.tag}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["image", "cer", "wer", "sim", "rec", "pred_chars", "gold_chars"])
                w.writeheader()
                for r in res["per_image"]:
                    w.writerow(r)

    # rank by sim desc
    rows.sort(key=lambda r: -r["sim"])

    csv_out = OUTDIR / "results.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "tag", "family", "n", "n_missing", "CER", "WER", "sim", "rec", "sec_per_img"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["tag"], r["family"], r["n"], r.get("n_missing", 0),
                        f"{r['cer']:.4f}", f"{r['wer']:.4f}", f"{r['sim']:.4f}", f"{r['rec']:.3f}",
                        (f"{r['sec_per_img']:.2f}" if r.get("sec_per_img") is not None else "")])
    print(f"\nWrote {csv_out}")

    md_out = OUTDIR / "results.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Эксперимент 7 — text-gold benchmark\n\n")
        f.write(f"50 файлов, gold = `gold_text.jsonl` (полный verbatim transcript).\n\n")
        f.write("Метрики (после NFKC + casefold + dash/quote normalisation):\n")
        f.write("- **CER** = Levenshtein(chars) / len(gold_chars) (lower is better)\n")
        f.write("- **WER** = Levenshtein(words) / len(gold_words) (lower is better)\n")
        f.write("- **sim** = SequenceMatcher ratio (higher is better)\n")
        f.write("- **rec** = chars(pred) / chars(gold) — purity/coverage proxy\n\n")
        f.write("| # | модель | family | n | CER ↓ | WER ↓ | sim ↑ | rec | sec/img |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(rows, 1):
            f.write(f"| {i} | `{r['tag']}` | {r['family']} | {r['n']} "
                    f"| {r['cer']:.3f} | {r['wer']:.3f} | **{r['sim']:.3f}** "
                    f"| {r['rec']:.2f} | {('%.1f' % r['sec_per_img']) if r.get('sec_per_img') is not None else '—'} |\n")
        f.write("\n")
    print(f"Wrote {md_out}")


if __name__ == "__main__":
    main()
