"""Собирает per-image вход для VLM-нормализатора.

Для каждого изображения корпуса (data/preprocessed_v3/Smuts/) ищет соответствие
во всех трёх OCR-jsonl (PaddleOCR / Surya / TrOCR-printed), плюс GT из
data/gold/labels.jsonl. Пишет один объединённый файл
`merged_inputs.jsonl` со структурой:

    {
      "image": "LE F-113259.JPG",
      "image_path": "/abs/path/.../preprocessed_v3/Smuts/LE F-113259.JPG",
      "gt": { <21 поле> },
      "ocr": {
          "paddleocr":     {"full_text": "...", "lines": [...], "runtime_s": 3.2},
          "surya":         {"full_text": "...", "lines": [...], "runtime_s": 4.3},
          "trocr-printed": {"full_text": "...", "lines": [...], "runtime_s": 1.5}
      }
    }

Также пишет компактную JSON-схему наших 21 полей в `field_schema.json`
для использования в промпте VLM.
"""
from __future__ import annotations

from src.utils import env_setup  # noqa: F401

import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "data" / "preprocessed_v3" / "Smuts"
TRIO_DIR = Path(__file__).resolve().parent / "trio_inputs"
GOLD_PATH = ROOT / "data" / "gold" / "labels.jsonl"
OUT_PATH = Path(__file__).resolve().parent / "merged_inputs.jsonl"
SCHEMA_PATH = Path(__file__).resolve().parent / "field_schema.json"


def read_jsonl(p: Path) -> Iterable[dict]:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_ocr(engine: str) -> dict[str, dict]:
    """{image_name: {full_text, lines, runtime}}"""
    out: dict[str, dict] = {}
    for rec in read_jsonl(TRIO_DIR / f"{engine}.jsonl"):
        out[rec["image"]] = {
            "full_text": rec.get("full_text", "") or "",
            "lines": [
                # компактный вид: только text + bbox + score
                {
                    "text": ln.get("text", ""),
                    "bbox": ln.get("bbox"),
                    "score": ln.get("confidence", ln.get("score")),
                }
                for ln in (rec.get("lines") or [])
            ],
            "runtime_s": rec.get("runtime_seconds"),
        }
    return out


def load_gold() -> dict[str, dict]:
    """{image_name: {field: value, ...}}.

    GT хранит список `image_files` — индексируем по каждому файлу. Поля
    `image_files`/`has_image`/`is_image_flag`/`*_raw` отбрасываем.
    """
    drop = {"image_files", "has_image", "is_image_flag",
            "catalog_number_raw", "specimen_remarks", "specimen_group",
            "verbatim_determiner"}
    out: dict[str, dict] = {}
    for rec in read_jsonl(GOLD_PATH):
        files = rec.get("image_files") or []
        if not files:
            continue
        fields = {k: v for k, v in rec.items() if k not in drop}
        for fn in files:
            if fn not in out:
                out[fn] = fields
    return out


# Полная схема 21 поля (используем для constrained decoding во время инференса).
FIELD_SCHEMA: dict[str, dict] = {
    "verbatim_identification":     {"type": "string", "verbatim": True,
                                    "desc": "Полное verbatim название таксона, как написано на этикетке (включая авторов, var./f./n. sp.)."},
    "verbatim_collector":          {"type": "string", "verbatim": True,
                                    "desc": "Сборщик, verbatim запись на этикетке."},
    "verbatim_locality":           {"type": "string", "verbatim": True,
                                    "desc": "Полная локация сбора, verbatim."},
    "verbatim_collection_date":    {"type": "string", "verbatim": True,
                                    "desc": "Дата сбора, как написано на этикетке."},
    "verbatim_substrate":          {"type": "string", "verbatim": True,
                                    "desc": "Растение-хозяин или субстрат, verbatim."},
    "verbatim_habitat":            {"type": "string", "verbatim": True,
                                    "desc": "Местообитание, verbatim."},
    "verbatim_identification_date":{"type": "string", "verbatim": True,
                                    "desc": "Дата определения вида, verbatim."},
    "field_number":                {"type": "string",
                                    "desc": "Полевой номер сборщика."},
    "verbatim_latitude":           {"type": "string", "verbatim": True,
                                    "desc": "Широта, verbatim (DMS или decimal)."},
    "verbatim_longitude":          {"type": "string", "verbatim": True,
                                    "desc": "Долгота, verbatim."},
    "verbatim_elevation":          {"type": "string", "verbatim": True,
                                    "desc": "Высота над уровнем моря, verbatim."},
    "exsiccata_title":             {"type": "string",
                                    "desc": "Название серии эксикат (e.g. 'Sydow, Ustilagineen', 'Rabenhorst, Fungi europaei')."},
    "exsiccata_number":            {"type": "string",
                                    "desc": "Номер в серии эксикат."},
    "type_status":                 {"type": "string|null",
                                    "enum": [None, "HOLOTYPE", "ISOTYPE", "LECTOTYPE", "PARATYPE", "SYNTYPE", "TYPUS", "OTHER"],
                                    "desc": "Тип статуса, если есть стикер TYPUS / HOLOTYPE / и т.д."},
    "catalog_number":              {"type": "string",
                                    "desc": "Каталожный номер LE F-XXXXX."},
    "scientific_name":             {"type": "string", "normalized": True,
                                    "desc": "Нормализованное научное название (текущая комбинация, без авторов)."},
    "determiner":                  {"type": "string|null", "normalized": True,
                                    "desc": "Кто определил вид, нормализовано 'Фамилия И. О.'."},
    "collector":                   {"type": "string|null", "normalized": True,
                                    "desc": "Сборщик, нормализовано 'Фамилия И. О.'."},
    "country":                     {"type": "string|null", "normalized": True,
                                    "desc": "Страна сбора, нормализованное название."},
    "organism_remarks":            {"type": "string|null", "verbatim": True,
                                    "desc": "Замечания о биологии организма."},
    "identification_remarks":      {"type": "string|null", "verbatim": True,
                                    "desc": "Замечания об определении."},
}


def main() -> None:
    paddle = load_ocr("paddleocr")
    surya = load_ocr("surya")
    trocr = load_ocr("trocr-printed")
    gold = load_gold()

    images = sorted(p.name for p in IMG_DIR.glob("*.JPG"))
    images += sorted(p.name for p in IMG_DIR.glob("*.jpg"))
    images = sorted(set(images))

    n_written = 0
    n_missing_gt = 0
    n_missing_ocr = 0

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for img in images:
            ocr = {}
            ok = True
            for name, src in [("paddleocr", paddle), ("surya", surya),
                              ("trocr-printed", trocr)]:
                if img not in src:
                    ok = False
                    break
                ocr[name] = src[img]
            if not ok:
                n_missing_ocr += 1
                continue

            gt = gold.get(img)
            if gt is None:
                n_missing_gt += 1

            rec = {
                "image": img,
                "image_path": str(IMG_DIR / img),
                "gt": gt,
                "ocr": ocr,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    SCHEMA_PATH.write_text(json.dumps(FIELD_SCHEMA, ensure_ascii=False, indent=2))

    print(f"Wrote {n_written} records → {OUT_PATH}")
    print(f"  missing OCR for: {n_missing_ocr}")
    print(f"  missing GT for : {n_missing_gt}")
    print(f"Wrote schema     → {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
