"""Эксперимент 1: baseline OCR на корпусе Smuts.

Запускает выбранные движки на изображениях из `data/gold/image_index.json`,
сохраняет per-engine OCR-выгрузки и считает метрики (CER/WER/similarity
+ field-level recall).
"""
from __future__ import annotations

# ВАЖНО: env-shim первой строкой.
from src.utils import env_setup  # noqa: F401

import argparse
import json
import time
from pathlib import Path

from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn

from src.eval.metrics import (
    GT_NORMALIZED_FIELDS,
    GT_VERBATIM_FIELDS,
    build_reference_text,
    cer,
    field_recall,
    similarity,
    wer,
)
from src.eval.field_f1 import DEFAULT_FIELDS, compute_field_f1, derive_gt_has
from src.ocr import ALL_ENGINES, get_engine

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "gold"
DEFAULT_IMG_DIR = ROOT / "Smuts"
DEFAULT_OUT = ROOT / "experiments" / "exp01_ocr_baseline" / "outputs"


def load_gold() -> tuple[dict[str, dict], dict[str, dict]]:
    """Возвращает (image_index, by_image_record) — record на каждое изображение."""
    index = json.loads((GOLD / "image_index.json").read_text(encoding="utf-8"))
    # Загружаем полные записи разметки (labels.jsonl) и индексируем по catalog_number.
    by_cat: dict[str, dict] = {}
    for line in (GOLD / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        by_cat[rec["catalog_number"]] = rec

    # Для каждого изображения возьмём полную запись.
    by_image: dict[str, dict] = {}
    for fname, short in index.items():
        rec = by_cat.get(short["catalog_number"]) or short
        by_image[fname] = rec
    return index, by_image


def _read_done(jsonl_path: Path) -> set[str]:
    """Возвращает множество имён изображений, уже сохранённых в jsonl."""
    if not jsonl_path.exists():
        return set()
    done = set()
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
            img = doc.get("image")
            if img:
                done.add(img)
        except json.JSONDecodeError:
            continue
    return done


def run_engine(engine_name: str, images: list[Path], out_dir: Path,
               sample: int | None = None, resume: bool = True) -> tuple[Path, dict]:
    """Запускает один OCR-движок на списке изображений.

    Если `resume=True` и `{engine}.jsonl` уже существует, пропускает изображения,
    уже представленные в jsonl, и дописывает только недостающие.

    Сохраняет:
        - `outputs/{engine}.jsonl` — построчно OCRDocument (json).
        - `outputs/{engine}_summary.json` — суммарная статистика.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{engine_name}.jsonl"
    out_summary = out_dir / f"{engine_name}_summary.json"

    if sample is not None:
        images = images[:sample]

    done = _read_done(out_jsonl) if resume else set()
    todo = [p for p in images if p.name not in done]
    mode = "a" if (done and resume) else "w"

    print(f"\n[{engine_name}] total={len(images)} already_done={len(done)} "
          f"todo={len(todo)} (resume={resume}) …", flush=True)

    if not todo:
        print(f"[{engine_name}] nothing to do.")
        return out_jsonl, {
            "engine": engine_name, "n_images": len(images),
            "n_ok": len(done), "n_err": 0,
            "total_seconds": 0.0, "mean_runtime": 0.0,
        }

    eng = get_engine(engine_name)
    t0 = time.perf_counter()
    n_ok = n_err = 0
    runtimes: list[float] = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[note]}"),
    ) as progress:
        task = progress.add_task(engine_name, total=len(todo), note="")
        with out_jsonl.open(mode, encoding="utf-8") as f:
            for img in todo:
                doc = eng.run(img)
                f.write(doc.model_dump_json() + "\n")
                f.flush()
                runtimes.append(doc.runtime_seconds)
                if doc.error:
                    n_err += 1
                else:
                    n_ok += 1
                progress.update(
                    task, advance=1,
                    note=f"ok={n_ok} err={n_err} last={doc.runtime_seconds:.2f}s",
                )

    summary = {
        "engine": engine_name,
        "n_images": len(images),
        "n_ok": n_ok,
        "n_err": n_err,
        "total_seconds": time.perf_counter() - t0,
        "mean_runtime": sum(runtimes) / max(len(runtimes), 1),
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"[{engine_name}] done in {summary['total_seconds']:.1f}s "
          f"(mean {summary['mean_runtime']:.2f}s/img); errors: {n_err}",
          flush=True)
    return out_jsonl, summary


def evaluate(engine_name: str, out_dir: Path,
             by_image: dict[str, dict]) -> dict:
    """Считает метрики по уже сохранённой выгрузке `{engine}.jsonl`."""
    in_jsonl = out_dir / f"{engine_name}.jsonl"
    out_metrics = out_dir / f"{engine_name}_metrics.json"

    per_image: list[dict] = []
    for line in in_jsonl.read_text(encoding="utf-8").splitlines():
        doc = json.loads(line)
        fname = doc["image"]
        gt = by_image.get(fname)
        if gt is None:
            continue
        ref = build_reference_text(gt, GT_VERBATIM_FIELDS)
        hyp = doc.get("full_text", "")

        metrics = {
            "image": fname,
            "catalog_number": gt.get("catalog_number"),
            "cer": cer(ref, hyp),
            "wer": wer(ref, hyp),
            "similarity": similarity(ref, hyp),
            "runtime": doc.get("runtime_seconds", 0.0),
            "error": doc.get("error"),
            "n_lines": len(doc.get("lines") or []),
        }
        fields_score: dict[str, float] = {}
        fields_found: dict[str, bool] = {}
        for f in DEFAULT_FIELDS:
            score, found = field_recall(gt.get(f), hyp)
            fields_score[f] = score
            fields_found[f] = found
        metrics["fields_score"] = fields_score
        metrics["fields_found"] = fields_found
        metrics["gt_has"] = derive_gt_has(gt, DEFAULT_FIELDS)
        per_image.append(metrics)

    # Агрегаты.
    n = len(per_image)
    n_with_gt = sum(1 for m in per_image if m.get("cer", 0) is not None)
    avg = lambda key: (sum(m[key] for m in per_image) / n) if n else 0.0
    median = lambda key: (sorted(m[key] for m in per_image)[n // 2]) if n else 0.0

    field_means = {}
    field_found_rates = {}
    for f in (*GT_VERBATIM_FIELDS, *GT_NORMALIZED_FIELDS):
        vals = [m["fields_score"].get(f, 0.0) for m in per_image
                if m["fields_score"].get(f) is not None]
        founds = [int(m["fields_found"].get(f, False)) for m in per_image]
        field_means[f] = (sum(vals) / max(len(vals), 1)) if vals else 0.0
        field_found_rates[f] = (sum(founds) / max(len(founds), 1)) if founds else 0.0

    # F1-метрики на уровне полей.
    f1_report = compute_field_f1(per_image, fields=DEFAULT_FIELDS, threshold=0.85)

    aggregate = {
        "engine": engine_name,
        "n_images": n,
        "mean_cer": avg("cer"),
        "median_cer": median("cer"),
        "mean_wer": avg("wer"),
        "median_wer": median("wer"),
        "mean_similarity": avg("similarity"),
        "median_similarity": median("similarity"),
        "mean_runtime": avg("runtime"),
        "field_mean_similarity": field_means,
        "field_found_rate": field_found_rates,
        "f1": f1_report,
    }
    out_metrics.write_text(
        json.dumps({"aggregate": aggregate, "per_image": per_image},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[{engine_name}] eval: CER={aggregate['mean_cer']:.3f} "
          f"WER={aggregate['mean_wer']:.3f} "
          f"sim={aggregate['mean_similarity']:.3f} "
          f"micro_F1={f1_report['micro']['f1']:.3f} "
          f"macro_F1={f1_report['macro']['f1']:.3f}", flush=True)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", nargs="+", default=list(ALL_ENGINES))
    parser.add_argument("--sample", type=int, default=None,
                        help="Запустить только на первых N изображениях.")
    parser.add_argument("--image-dir", default=str(DEFAULT_IMG_DIR),
                        help="Где брать изображения (можно подменить на preprocessed/).")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="Папка для сохранения выгрузок.")
    parser.add_argument("--skip-recognition", action="store_true",
                        help="Только метрики на уже сохранённых .jsonl.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Игнорировать уже сохранённые результаты и переписывать заново.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    img_dir = Path(args.image_dir)
    print(f"Image source: {img_dir}", flush=True)
    image_index, by_image = load_gold()
    images = sorted(img_dir / fname for fname in image_index.keys()
                    if (img_dir / fname).exists())
    if args.sample:
        images = images[: args.sample]

    summaries = []
    metrics = []
    for eng in args.engines:
        if not args.skip_recognition:
            try:
                _, summary = run_engine(
                    eng, images, out_dir,
                    sample=args.sample,
                    resume=not args.no_resume,
                )
                summaries.append(summary)
            except KeyboardInterrupt:
                print(f"\n[!] Interrupted at {eng}")
                raise
            except Exception as e:
                print(f"[{eng}] FAILED to run: {type(e).__name__}: {e}")
                continue
        try:
            agg = evaluate(eng, out_dir, by_image)
            metrics.append(agg)
        except FileNotFoundError as e:
            print(f"[{eng}] eval skipped (no jsonl): {e}")

    if metrics:
        report_path = out_dir / "report.json"
        report_path.write_text(
            json.dumps({"summaries": summaries, "metrics": metrics},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nReport: {report_path}")
        # Сводная таблица в stdout.
        header = (f"\n{'engine':<22} | {'CER':>6} | {'WER':>6} | {'sim':>6} | "
                  f"{'mP':>5} | {'mR':>5} | {'mF1':>5} | {'MF1':>5} | {'sec':>5}")
        print(header)
        print("-" * len(header))
        for m in metrics:
            f1 = m["f1"]
            print(f"{m['engine']:<22} | {m['mean_cer']:>6.3f} | "
                  f"{m['mean_wer']:>6.3f} | {m['mean_similarity']:>6.3f} | "
                  f"{f1['micro']['precision']:>5.2f} | "
                  f"{f1['micro']['recall']:>5.2f} | "
                  f"{f1['micro']['f1']:>5.2f} | "
                  f"{f1['macro']['f1']:>5.2f} | "
                  f"{m['mean_runtime']:>5.2f}")


if __name__ == "__main__":
    main()
