"""Cross-validation hallucinations через top-4 LoRA модели.

Идея: если несколько LoRA-моделей независимо выдают одно и то же значение
поля, которого нет в нашей gt — скорее всего это РЕАЛЬНЫЙ текст на этикетке,
а не галлюцинация. Это указывает на ошибки/пропуски в gt-разметке.

Анализ:
1. Для каждого поля, каждого image — собрать предсказания всех 4 LoRA моделей.
2. Найти cases где gt[field] is null, но ≥2 модели вывели одинаковое значение.
3. Эти случаи — кандидаты на исправление gt.

Outputs:
  experiments/exp16_gt_audit/hallucination_consensus.csv
  experiments/exp16_gt_audit/hallucination_consensus.md
"""
from __future__ import annotations
import argparse, json, re, sys, csv
from pathlib import Path
from collections import defaultdict, Counter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.utils import env_setup  # noqa
from unified_eval import load_gold, FIELDS, similarity, has_value, TAU_HARD


def read_jsonl_tolerant(path: Path):
    text = path.read_text(encoding='utf-8', errors='replace')
    starts = [m.start() for m in re.finditer(r'^\{"image":\s*"', text, re.M)]
    for i, s in enumerate(starts):
        e = starts[i+1] if i+1 < len(starts) else len(text)
        blob = text[s:e].rstrip()
        try:
            yield json.loads(blob)
        except Exception:
            continue


def normalize_for_consensus(val):
    """Нормализация для сравнения значений между моделями.
    Сохраняем содержание, но игнорируем регистр, лишние пробелы, knочи в начале."""
    if not has_value(val):
        return None
    s = str(val).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = s.rstrip('.,')
    return s if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", required=True,
                    help="Multiple LoRA eval JSONLs to cross-compare")
    ap.add_argument("--output-dir", default="experiments/exp16_gt_audit",
                    help="Output directory for consensus reports")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    gold = load_gold()
    print(f"[audit] gold images: {len(gold)}")
    
    # Загрузить все evals
    models = {}
    for eval_path in args.evals:
        ep = Path(eval_path)
        model_name = ep.stem  # без .jsonl
        records = {}
        for rec in read_jsonl_tolerant(ep):
            records[rec['image']] = rec
        models[model_name] = records
        print(f"[audit] {model_name}: {len(records)} records")
    
    # Image union
    all_images = set()
    for m in models.values():
        all_images.update(m.keys())
    all_images &= set(gold.keys())  # только те которые в gold
    print(f"[audit] images in gold + at least 1 model: {len(all_images)}")
    
    # Cross-comparison
    # Для каждого (image, field) собираем все предсказания
    # Найти cases:
    #   1. gt = null, ≥2 модели согласны = HALLUC_CONSENSUS (потенциальная ошибка gt — модели правы)
    #   2. gt ≠ null, все модели одинаково расходятся с gt = MODEL_BIAS (модели согласны но расходятся с gt)
    
    halluc_consensus = []  # gt null, models agree
    model_bias = []         # gt ≠ null, models agree but differ from gt
    
    for img in sorted(all_images):
        gt = gold[img]
        for field in FIELDS:
            gt_v = gt.get(field)
            gt_h = has_value(gt_v)
            
            # Collect non-null predictions from all models
            preds = []
            for model_name, recs in models.items():
                if img not in recs:
                    continue
                pj = recs[img].get('parsed_json') or {}
                pv = pj.get(field) if pj else None
                if has_value(pv):
                    preds.append((model_name, pv, normalize_for_consensus(pv)))
            
            if len(preds) < 2:
                continue  # нужно ≥2 модели
            
            # Counter по нормализованным значениям
            norms = [p[2] for p in preds]
            counter = Counter(norms)
            most_common, count = counter.most_common(1)[0]
            
            if count < 2:
                continue  # consensus = только 1 модель, не считается
            
            # Получим оригинальное значение (не норм)
            consensus_val = next(p[1] for p in preds if p[2] == most_common)
            agreeing_models = [p[0] for p in preds if p[2] == most_common]
            
            if not gt_h:
                # gt пусто, но ≥2 модели согласны — потенциальная ошибка gt
                halluc_consensus.append({
                    'image': img,
                    'field': field,
                    'gt': '<NULL>',
                    'consensus_value': consensus_val,
                    'n_agreeing': count,
                    'agreeing_models': ','.join(agreeing_models),
                    'all_n_models_predicted': len(preds),
                })
            else:
                # gt не пусто, но модели согласны с другим значением
                sim_gt = similarity(gt_v, consensus_val)
                if sim_gt < TAU_HARD:
                    model_bias.append({
                        'image': img,
                        'field': field,
                        'gt': str(gt_v)[:100],
                        'consensus_value': consensus_val[:100],
                        'sim_gt_consensus': round(sim_gt, 3),
                        'n_agreeing': count,
                        'agreeing_models': ','.join(agreeing_models),
                        'all_n_models_predicted': len(preds),
                    })
    
    print(f"\n=== РЕЗУЛЬТАТЫ ===")
    print(f"HALLUC_CONSENSUS (gt null, ≥2 модели согласны): {len(halluc_consensus)}")
    print(f"MODEL_BIAS (gt ≠ null, ≥2 модели согласны на другом значении): {len(model_bias)}")
    
    # Группировка по полю
    halluc_by_field = Counter(r['field'] for r in halluc_consensus)
    bias_by_field = Counter(r['field'] for r in model_bias)
    
    print(f"\n--- HALLUC_CONSENSUS по полям ---")
    for f, count in halluc_by_field.most_common():
        print(f"  {f:<30}: {count}")
    
    print(f"\n--- MODEL_BIAS по полям ---")
    for f, count in bias_by_field.most_common():
        print(f"  {f:<30}: {count}")
    
    # Сохранить CSV
    if halluc_consensus:
        with (out_dir / "hallucination_consensus.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=halluc_consensus[0].keys())
            w.writeheader()
            w.writerows(halluc_consensus)
        print(f"\n[write] {out_dir}/hallucination_consensus.csv ({len(halluc_consensus)} rows)")
    
    if model_bias:
        with (out_dir / "model_bias.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=model_bias[0].keys())
            w.writeheader()
            w.writerows(model_bias)
        print(f"[write] {out_dir}/model_bias.csv ({len(model_bias)} rows)")
    
    # Markdown отчёт
    md_lines = [f"# GT Audit — Cross-validation hallucinations\n"]
    md_lines.append(f"Models analyzed: {list(models.keys())}\n")
    md_lines.append(f"Images compared: {len(all_images)}\n")
    md_lines.append(f"\n## HALLUC_CONSENSUS ({len(halluc_consensus)} cases)\n")
    md_lines.append(f"gt = null, но ≥2 модели независимо вывели одинаковое значение → возможно gt неполное.\n")
    md_lines.append(f"\n### По полям:\n")
    for f, c in halluc_by_field.most_common():
        md_lines.append(f"- **{f}**: {c}")
    
    # Show top-30 examples sorted by n_agreeing desc
    halluc_consensus.sort(key=lambda r: (-r['n_agreeing'], r['field']))
    md_lines.append(f"\n### Топ-30 кандидатов на исправление gt (сортировка по n_agreeing):\n")
    md_lines.append("| Image | Field | n_models | Consensus value |")
    md_lines.append("|---|---|---|---|")
    for r in halluc_consensus[:30]:
        v = r['consensus_value'][:80].replace('|', '\\|')
        md_lines.append(f"| {r['image']} | {r['field']} | {r['n_agreeing']} | {v!r} |")
    
    md_lines.append(f"\n## MODEL_BIAS ({len(model_bias)} cases)\n")
    md_lines.append(f"gt ≠ null, но ≥2 модели согласны на ДРУГОМ значении.\n")
    if model_bias:
        model_bias.sort(key=lambda r: (-r['n_agreeing'], r['sim_gt_consensus']))
        md_lines.append(f"\n### Топ-30 (потенциальные ошибки модели или gt):\n")
        md_lines.append("| Image | Field | sim | n_models | GT | Consensus |")
        md_lines.append("|---|---|---|---|---|---|")
        for r in model_bias[:30]:
            gt = r['gt'][:50].replace('|', '\\|')
            cv = r['consensus_value'][:50].replace('|', '\\|')
            md_lines.append(f"| {r['image']} | {r['field']} | {r['sim_gt_consensus']:.2f} | {r['n_agreeing']} | {gt!r} | {cv!r} |")
    
    (out_dir / "audit_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[write] {out_dir}/audit_report.md")


if __name__ == "__main__":
    main()
