"""Сводная статистика по инференсу Anthracoidea (без GT).

Считает: parse_rate, sec/img, per-field fill-rate (доля непустых значений),
распределение по видам, длины raw_response. Пишет CSV + markdown.

Usage:
    python experiments/exp17_anthracoidea/summarize.py \
        --in experiments/exp17_anthracoidea/outputs/anthracoidea_lora_p5j.jsonl \
        --out-prefix experiments/exp17_anthracoidea/summary
"""
from __future__ import annotations
import argparse, json, statistics
from collections import Counter, defaultdict
from pathlib import Path

# 21 поле GT (порядок как в RESULTS.md)
FIELDS = [
    "catalog_number", "scientific_name", "verbatim_identification", "type_status",
    "determiner", "verbatim_identification_date", "collector", "verbatim_collector",
    "field_number", "verbatim_collection_date", "country", "verbatim_locality",
    "verbatim_latitude", "verbatim_longitude", "verbatim_elevation",
    "verbatim_habitat", "verbatim_substrate", "organism_remarks",
    "identification_remarks", "exsiccata_title", "exsiccata_number",
]


def read_jsonl(p: Path):
    for line in p.open("r", encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def is_filled(v) -> bool:
    return v not in (None, "", [], {})


def species_of(name: str) -> str:
    # Anthracoidea_arenaria_001.JPG -> Anthracoidea_arenaria
    return "_".join(name.split("_")[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    recs = list(read_jsonl(Path(args.inp)))
    n = len(recs)
    if n == 0:
        print("no records")
        return

    n_ok = sum(1 for r in recs if r.get("parse_ok"))
    runtimes = [r.get("runtime_seconds", 0) or 0 for r in recs]
    out_tokens = [r.get("n_output_tokens", 0) or 0 for r in recs]
    raw_lens = [len(r.get("raw_response") or "") for r in recs]

    # per-field fill rate (только по успешно распарсенным)
    parsed = [r["parsed_json"] for r in recs if r.get("parse_ok") and r.get("parsed_json")]
    field_fill = {f: sum(1 for j in parsed if is_filled(j.get(f))) for f in FIELDS}

    # per-species
    sp_counter = Counter(species_of(r["image"]) for r in recs)
    sp_ok = defaultdict(int)
    for r in recs:
        if r.get("parse_ok"):
            sp_ok[species_of(r["image"])] += 1

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # --- CSV: per-field fill rate ---
    csv_path = prefix.with_name(prefix.name + "_fields.csv")
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("field,filled,total_parsed,fill_rate\n")
        for fld in FIELDS:
            fr = field_fill[fld] / max(1, len(parsed))
            f.write(f"{fld},{field_fill[fld]},{len(parsed)},{fr:.4f}\n")

    # --- CSV: per-species ---
    csv_sp = prefix.with_name(prefix.name + "_species.csv")
    with csv_sp.open("w", encoding="utf-8") as f:
        f.write("species,n_images,n_parsed,parse_rate\n")
        for sp, cnt in sorted(sp_counter.items()):
            pr = sp_ok[sp] / cnt
            f.write(f"{sp},{cnt},{sp_ok[sp]},{pr:.4f}\n")

    # --- Markdown ---
    md = prefix.with_name(prefix.name + ".md")
    with md.open("w", encoding="utf-8") as f:
        f.write("# Anthracoidea inference summary\n\n")
        f.write(f"- Файл: `{args.inp}`\n")
        f.write(f"- Записей: **{n}**\n")
        f.write(f"- parse_ok: **{n_ok}/{n} ({n_ok/n*100:.1f}%)**\n")
        f.write(f"- sec/img: mean **{statistics.mean(runtimes):.2f}**, "
                f"median {statistics.median(runtimes):.2f}, "
                f"max {max(runtimes):.2f}\n")
        f.write(f"- output tokens: mean {statistics.mean(out_tokens):.0f}, "
                f"max {max(out_tokens)}\n")
        f.write(f"- raw_response length (chars): mean {statistics.mean(raw_lens):.0f}, "
                f"max {max(raw_lens)}\n\n")

        f.write("## Per-field fill rate (доля непустых значений среди распарсенных)\n\n")
        f.write("| Поле | filled | fill_rate |\n|---|---:|---:|\n")
        for fld in sorted(FIELDS, key=lambda x: -field_fill[x]):
            fr = field_fill[fld] / max(1, len(parsed))
            f.write(f"| {fld} | {field_fill[fld]} | {fr*100:.1f}% |\n")

        f.write("\n## Per-species (parse rate)\n\n")
        f.write("| Вид | картинок | распарсено | parse_rate |\n|---|---:|---:|---:|\n")
        for sp, cnt in sorted(sp_counter.items(), key=lambda kv: -kv[1]):
            pr = sp_ok[sp] / cnt
            f.write(f"| {sp} | {cnt} | {sp_ok[sp]} | {pr*100:.1f}% |\n")

    print(f"[summary] records={n} parse_ok={n_ok} ({n_ok/n*100:.1f}%)")
    print(f"[summary] wrote:\n  {md}\n  {csv_path}\n  {csv_sp}")


if __name__ == "__main__":
    main()
