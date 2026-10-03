"""Stage-1 раннер: OCR-модель в одном env → markdown/plain text на диск.
Форматтер (Qwen3-VL-8B) запускается отдельно в td-vlm.

Подходит для моделей, требующих старые/несовместимые с td-vlm env-ы:
    - deepseek-ocr (td-legacy)
    - got-ocr2     (td-legacy)
    - unlimited-ocr (отдельный env)
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.pop("PYTHONSTARTUP", None)
os.environ.pop("PYTHONPATH", None)

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.exp05_vlm_normalizer.run_qwen import read_jsonl, already_done
from experiments.exp05_vlm_normalizer.run_ocr_vlm import OCR_LOADERS


ROOT = Path(__file__).resolve().parents[2]
INPUTS_PATH = Path(__file__).resolve().parent / "merged_inputs.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", required=True, choices=list(OCR_LOADERS.keys()))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--inputs", default=str(INPUTS_PATH))
    ap.add_argument("--out-suffix", default="__raw")
    args = ap.parse_args()

    slug = args.ocr.replace("/", "__")
    out_path = OUT_DIR / f"{slug}{args.out_suffix}.jsonl"
    print(f"[run] ocr={args.ocr} (stage-1 only) out={out_path}", flush=True)

    skip = already_done(out_path)
    if skip:
        print(f"[run] resume: already done = {len(skip)}", flush=True)
    records = list(read_jsonl(Path(args.inputs)))
    if args.limit:
        records = records[: args.limit]
    todo = [r for r in records if r["image"] not in skip]
    print(f"[run] todo = {len(todo)} / {len(records)}", flush=True)
    if not todo:
        return

    read_one, _ = OCR_LOADERS[args.ocr]()

    t_all = time.perf_counter()
    n_done = 0
    n_err = 0

    with out_path.open("a", encoding="utf-8") as f_out:
        for i, rec in enumerate(todo, 1):
            t0 = time.perf_counter()
            try:
                ocr_text = read_one(rec["image_path"]) or ""
                err = None
            except Exception as e:
                ocr_text = ""
                err = str(e)
                n_err += 1
            ocr_dt = time.perf_counter() - t0

            f_out.write(json.dumps({
                "image": rec["image"],
                "model": args.ocr,
                "ocr_text": ocr_text,
                "ocr_runtime_s": round(ocr_dt, 3),
                "runtime_seconds": round(ocr_dt, 3),
                "error": err,
            }, ensure_ascii=False) + "\n")
            f_out.flush()
            n_done += 1

            if i % 10 == 0 or i == len(todo):
                elapsed = time.perf_counter() - t_all
                rate = elapsed / i
                eta = rate * (len(todo) - i)
                print(f"[run] {i}/{len(todo)}  ok={n_done-n_err}  err={n_err}  "
                      f"avg={rate:.1f}s/img  ETA {eta/60:.1f}min", flush=True)

    print(f"\n[run] done. {n_done-n_err}/{n_done}", flush=True)


if __name__ == "__main__":
    main()
