"""Готовит gold-набор: xlsx → JSONL с маппингом на нашу схему.

Связь с изображениями: catalogNumber → файлы вида `LE F-<N>(_<k>)?.JPG`.
Один catalogNumber может иметь несколько ракурсов (_1, _2).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
XLSX = ROOT / "LE_Fungarium_Ustilaginomycotina.xlsx"
IMG_DIR = ROOT / "Smuts"
OUT_DIR = ROOT / "data" / "gold"
OUT_JSONL = OUT_DIR / "labels.jsonl"
OUT_INDEX = OUT_DIR / "image_index.json"


# Маппинг колонок xlsx → поля нашей схемы.
COLUMN_MAP: dict[str, str] = {
    "catalogNumber": "catalog_number",
    "scientificName": "scientific_name",
    "verbatimIdentification": "verbatim_identification",
    "typeStatus": "type_status",
    "identifiedBy": "determiner",
    "verbatimIdentifiedBy": "verbatim_determiner",
    "verbatimIdentificationDate": "verbatim_identification_date",
    "collectedBy": "collector",
    "verbatimCollectedBy": "verbatim_collector",
    "fieldNumber": "field_number",
    "verbatimCollectionDate": "verbatim_collection_date",
    "country": "country",
    "verbatimLocality": "verbatim_locality",
    "verbatimLatitude": "verbatim_latitude",
    "verbatimLongitude": "verbatim_longitude",
    "verbatimElevation": "verbatim_elevation",
    "verbatimHabitat": "verbatim_habitat",
    "verbatimSubstrate": "verbatim_substrate",
    "organismRemarks": "organism_remarks",
    "identificationRemarks": "identification_remarks",
    "specimenRemarks": "specimen_remarks",
    "specimenGroup": "specimen_group",
    "exsiccataTitle": "exsiccata_title",
    "exsiccataNumber": "exsiccata_number",
}


def _stem_num(p: Path) -> str | None:
    m = re.search(r"(\d+)", p.stem)
    return m.group(1) if m else None


def _coerce(value):
    """Приводит значения к JSON-сериализуемому виду; NaN → None."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Изображения ---
    images = sorted(list(IMG_DIR.glob("*.JPG")) + list(IMG_DIR.glob("*.jpg")))
    img_by_num: dict[str, list[str]] = {}
    for p in images:
        n = _stem_num(p)
        if n is None:
            continue
        img_by_num.setdefault(n, []).append(p.name)

    # --- Разметка ---
    df = pd.read_excel(XLSX, sheet_name="Data")
    df["catalogNumber"] = df["catalogNumber"].astype(str).str.strip()

    records: list[dict] = []
    n_with_img = 0
    n_no_img = 0
    for _, row in df.iterrows():
        cat = str(row["catalogNumber"]).strip()
        rec: dict = {}
        for src_col, dst_col in COLUMN_MAP.items():
            if src_col in df.columns:
                rec[dst_col] = _coerce(row[src_col])

        # catalog_number в форме "LE F-<N>" — как на штрих-коде.
        rec["catalog_number_raw"] = cat
        rec["catalog_number"] = f"LE F-{cat}"

        # Привязка к изображениям.
        files = img_by_num.get(cat, [])
        rec["image_files"] = files
        rec["has_image"] = bool(files)
        # is_image: явный флаг из xlsx; NaN = есть, 0 = нет (по результатам анализа)
        is_img_raw = _coerce(row["isImage"])
        rec["is_image_flag"] = (is_img_raw is None)

        if files:
            n_with_img += 1
        else:
            n_no_img += 1

        records.append(rec)

    # Записываем JSONL.
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Индекс изображений: для каждой картинки — соответствующая запись.
    image_index: dict[str, dict] = {}
    for r in records:
        for fname in r["image_files"]:
            image_index[fname] = {
                "catalog_number": r["catalog_number"],
                "catalog_number_raw": r["catalog_number_raw"],
                "scientific_name": r.get("scientific_name"),
                "type_status": r.get("type_status"),
            }
    OUT_INDEX.write_text(
        json.dumps(image_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Gold records:   {len(records)}")
    print(f"  with image:   {n_with_img}")
    print(f"  no image:     {n_no_img}")
    print(f"Image files:    {len(images)}")
    print(f"Indexed images: {len(image_index)}")
    print(f"Wrote: {OUT_JSONL}")
    print(f"Wrote: {OUT_INDEX}")


if __name__ == "__main__":
    build()
