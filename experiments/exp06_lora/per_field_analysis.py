"""Per-field error analysis for top-N LoRA models on full 394 corpus.

Анализирует каждое из 21 поля Darwin Core отдельно:
  - precision, recall, F1, soft_R (similarity-based)
  - error categories: hallucination, omission, partial, wrong-format
  - top-N worst cases per field

Outputs:
  experiments/exp06_lora/sweep/per_field/<run_id>.{csv,md}
  experiments/exp06_lora/sweep/per_field/error_categories.md
  experiments/exp06_lora/sweep/per_field/worst_cases.md
  experiments/exp06_lora/sweep/per_field/comparison_top_models.md

Usage:
    python experiments/exp06_lora/per_field_analysis.py \\
        --evals path1.jsonl path2.jsonl path3.jsonl path4.jsonl
"""
from __future__ import annotations
import argparse, json, sys, csv, statistics
from pathlib import Path
from difflib import SequenceMatcher

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
from src.utils import env_setup  # noqa

from unified_eval import load_gold, FIELDS, similarity, has_value, TAU_HARD  # noqa

OUT_DIR = REPO_ROOT / "experiments/exp06_lora/sweep/per_field"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_jsonl(p: Path):
    for line in p.open("r", encoding="utf-8"):
        line = line.strip()
        if not line: continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def classify_error(gt_v, pred_v) -> str:
    """Категория ошибки на конкретном поле."""
    gt_h = has_value(gt_v)
    pred_h = has_value(pred_v)
    if not gt_h and not pred_h:
        return "true_null"
    if not gt_h and pred_h:
        return "hallucination"   # gt пусто, predicted нечто
    if gt_h and not pred_h:
        return "omission"        # gt есть, pred None
    sim = similarity(gt_v, pred_v)
    if sim >= TAU_HARD:
        return "correct"
    elif sim >= 0.3:
        return "partial"
    else:
        return "wrong"


def analyze_one(eval_path: Path, gold: dict) -> dict:
    """Анализ одного eval.jsonl на 394 файлах."""
    records = list(read_jsonl(eval_path))
    n_total = len(records)
    print(f"  loaded {n_total} records from {eval_path.name}")

    # per-field stats
    field_stats = {}
    for f in FIELDS:
        field_stats[f] = {
            "tp": 0, "fp": 0, "fn": 0,
            "n_gt": 0, "n_pred": 0,
            "sim_sum": 0.0,
            "errors": {
                "true_null": 0, "hallucination": 0, "omission": 0,
                "partial": 0, "wrong": 0, "correct": 0,
            },
            "worst_cases": [],   # (sim, image, gt_val, pred_val) topN
        }

    n_parse_ok = 0
    runtimes = []
    for rec in records:
        img = rec.get("image")
        if img not in gold:
            continue
        if rec.get("parse_ok"):
            n_parse_ok += 1
        rt = rec.get("runtime_seconds")
        if rt: runtimes.append(rt)
        gt = gold[img]
        pred = rec.get("parsed_json") or {}
        for f in FIELDS:
            gt_v = gt.get(f)
            pred_v = pred.get(f) if pred else None
            cat = classify_error(gt_v, pred_v)
            gt_h = has_value(gt_v)
            pred_h = has_value(pred_v)
            field_stats[f]["errors"][cat] += 1
            if gt_h:
                field_stats[f]["n_gt"] += 1
            if pred_h:
                field_stats[f]["n_pred"] += 1
            sim = similarity(gt_v, pred_v) if (gt_h and pred_h) else 0.0
            if gt_h:
                field_stats[f]["sim_sum"] += sim if pred_h else 0.0
            # tp/fp/fn
            if gt_h and pred_h and sim >= TAU_HARD:
                field_stats[f]["tp"] += 1
            elif pred_h and (not gt_h or sim < TAU_HARD):
                field_stats[f]["fp"] += 1
            elif gt_h and (not pred_h or sim < TAU_HARD):
                field_stats[f]["fn"] += 1
            # worst cases
            if gt_h and (not pred_h or sim < TAU_HARD):
                field_stats[f]["worst_cases"].append((
                    sim, img, str(gt_v)[:120], str(pred_v)[:120] if pred_h else "<NULL>"
                ))

    # finalize
    for f in FIELDS:
        s = field_stats[f]
        # sort worst cases by sim asc
        s["worst_cases"].sort(key=lambda x: x[0])
        s["worst_cases"] = s["worst_cases"][:10]  # top 10 worst
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        s["precision"] = p
        s["recall"] = r
        s["f1"] = f1
        s["soft_R"] = s["sim_sum"] / s["n_gt"] if s["n_gt"] else 0.0

    return {
        "n_records": n_total,
        "n_parse_ok": n_parse_ok,
        "parse_rate": n_parse_ok / max(1, n_total),
        "mean_sec_per_img": statistics.mean(runtimes) if runtimes else 0.0,
        "fields": field_stats,
    }


def write_per_field_md(run_id: str, analysis: dict, out: Path):
    lines = [f"# Per-field error analysis: {run_id}\n"]
    lines.append(f"- n_records: {analysis['n_records']}")
    lines.append(f"- parse_rate: {analysis['parse_rate']:.4f}")
    lines.append(f"- mean sec/img: {analysis['mean_sec_per_img']:.2f}\n")
    lines.append("## Per-field metrics\n")
    lines.append("| Field | n_gt | precision | recall | F1 | soft_R | TP | FP | FN |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    # sort fields by F1 desc
    fields_sorted = sorted(analysis["fields"].items(),
                           key=lambda x: -x[1]["f1"])
    for f, s in fields_sorted:
        lines.append(
            f"| {f} | {s['n_gt']} | {s['precision']:.3f} | {s['recall']:.3f} | "
            f"{s['f1']:.3f} | {s['soft_R']:.3f} | {s['tp']} | {s['fp']} | {s['fn']} |"
        )
    lines.append("\n## Error categories per field\n")
    lines.append("| Field | correct | partial | wrong | omission | halluc | true_null |")
    lines.append("|---|---|---|---|---|---|---|")
    for f, s in fields_sorted:
        e = s["errors"]
        lines.append(
            f"| {f} | {e['correct']} | {e['partial']} | {e['wrong']} | "
            f"{e['omission']} | {e['hallucination']} | {e['true_null']} |"
        )
    lines.append("\n## Top-10 worst cases per field (sorted by similarity asc)\n")
    for f, s in fields_sorted:
        if not s["worst_cases"]: continue
        lines.append(f"\n### `{f}` (F1={s['f1']:.3f}, n_gt={s['n_gt']})\n")
        lines.append("| sim | image | gt | pred |")
        lines.append("|---|---|---|---|")
        for sim, img, gt_v, pred_v in s["worst_cases"]:
            gt_e = (gt_v or "")[:80].replace("|", "\\|").replace("\n", " ")
            pred_e = (pred_v or "")[:80].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {sim:.2f} | {img} | {gt_e} | {pred_e} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(run_id: str, analysis: dict, out: Path):
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "n_gt", "precision", "recall", "f1", "soft_R",
                    "tp", "fp", "fn", "correct", "partial", "wrong",
                    "omission", "hallucination", "true_null"])
        for f_name, s in analysis["fields"].items():
            e = s["errors"]
            w.writerow([
                f_name, s["n_gt"], s["precision"], s["recall"], s["f1"], s["soft_R"],
                s["tp"], s["fp"], s["fn"],
                e["correct"], e["partial"], e["wrong"],
                e["omission"], e["hallucination"], e["true_null"],
            ])


def write_comparison(analyses: dict[str, dict], out: Path):
    """Сравнительная таблица топ моделей по полям."""
    lines = ["# Per-field comparison — top models on 394 full corpus\n"]
    lines.append(f"Models: {', '.join(analyses.keys())}\n")
    lines.append("## F1 per field — все модели\n")
    header = ["Field", "n_gt"]
    for run_id in analyses:
        header.append(f"F1 {run_id[:30]}")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    # Возьмём поля из первой модели
    first_run = next(iter(analyses.values()))
    fields = list(first_run["fields"].keys())
    # Sort fields by overall difficulty (mean F1 across models, ascending)
    field_mean_f1 = {}
    for f in fields:
        f1s = [a["fields"][f]["f1"] for a in analyses.values()]
        field_mean_f1[f] = sum(f1s) / len(f1s) if f1s else 0
    fields.sort(key=lambda f: field_mean_f1[f])

    for f in fields:
        row = [f, str(first_run["fields"][f]["n_gt"])]
        for run_id, a in analyses.items():
            row.append(f"{a['fields'][f]['f1']:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## CER, parse_rate, sec/img summary\n")
    lines.append("| Run | parse_rate | mean sec/img | n_records |")
    lines.append("|---|---|---|---|")
    for run_id, a in analyses.items():
        lines.append(f"| {run_id} | {a['parse_rate']:.4f} | "
                     f"{a['mean_sec_per_img']:.2f} | {a['n_records']} |")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", required=True,
                    help="eval JSONL paths (one per LoRA model)")
    args = ap.parse_args()

    gold = load_gold()
    print(f"[ana] gold images: {len(gold)}")

    analyses = {}
    for ep in args.evals:
        ep = Path(ep)
        run_id = ep.stem
        print(f"\n[ana] analyzing {run_id}")
        analyses[run_id] = analyze_one(ep, gold)

        # Per-model output
        write_csv(run_id, analyses[run_id], OUT_DIR / f"{run_id}.csv")
        write_per_field_md(run_id, analyses[run_id], OUT_DIR / f"{run_id}.md")
        print(f"  → {OUT_DIR}/{run_id}.{{csv,md}}")

    # Comparison
    if len(analyses) > 1:
        write_comparison(analyses, OUT_DIR / "comparison.md")
        print(f"\n→ comparison: {OUT_DIR}/comparison.md")


if __name__ == "__main__":
    main()
