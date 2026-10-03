"""Stage-2: форматтер. Читает stage-1 jsonl (ocr_text) и форматит в JSON-21.

Используется после run_ocr_only.py.

CLI:
    python -m experiments.exp05_vlm_normalizer.run_format_stage \
        --input outputs/deepseek-ocr__raw.jsonl \
        --output outputs/deepseek-ocr__formatted.jsonl
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


ROOT = Path(__file__).resolve().parents[2]
INPUTS_PATH = Path(__file__).resolve().parent / "merged_inputs.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

sys.path.insert(0, str(ROOT))
from experiments.exp05_vlm_normalizer.formatter import load_formatter, format_one
from experiments.exp05_vlm_normalizer.run_qwen import already_done
from experiments.exp05_vlm_normalizer.evaluate import read_jsonl  # tolerant version


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="stage-1 jsonl")
    ap.add_argument("--output", required=True, help="formatted jsonl")
    ap.add_argument("--ocr-name", default=None, help="display name in prompt")
    ap.add_argument("--formatter-model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-memory-gib", type=int, default=None)
    ap.add_argument("--few-shot", action="store_true",
                    help="Use 5 few-shot examples in formatter prompt")
    ap.add_argument("--skip-fewshot-images", action="store_true",
                    help="Exclude few-shot example images from evaluation (to avoid leakage)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    ocr_name = args.ocr_name or in_path.stem.split("__")[0]
    print(f"[run] stage-2 format {in_path.name} → {out_path.name}  ocr={ocr_name}",
          flush=True)

    # Read stage-1
    stage1_records = list(read_jsonl(in_path))
    if args.limit:
        stage1_records = stage1_records[: args.limit]

    # Index input images for image_path lookup
    img_path_idx = {}
    for r in read_jsonl(INPUTS_PATH):
        img_path_idx[r["image"]] = r["image_path"]

    skip = already_done(out_path)
    if args.skip_fewshot_images:
        from experiments.exp05_vlm_normalizer.prompts.builder import FEW_SHOT_IMAGE_NAMES
        skip = skip | FEW_SHOT_IMAGE_NAMES
        print(f"[run] excluding {len(FEW_SHOT_IMAGE_NAMES)} few-shot images", flush=True)
    todo = [r for r in stage1_records if r["image"] not in skip]
    print(f"[run] todo = {len(todo)} / {len(stage1_records)}", flush=True)
    if not todo:
        return

    fmt_model, fmt_proc = load_formatter(args.formatter_model,
                                          max_memory_gib=args.max_memory_gib)

    t_all = time.perf_counter()
    n_ok = 0
    n_fail = 0

    with out_path.open("a", encoding="utf-8") as f_out:
        for i, rec in enumerate(todo, 1):
            image_path = img_path_idx.get(rec["image"])
            if not image_path:
                continue
            ocr_text = rec.get("ocr_text", "") or ""
            ocr_dt = rec.get("ocr_runtime_s", 0)

            try:
                result = format_one(fmt_model, fmt_proc, image_path, ocr_text, ocr_name,
                                    max_new_tokens=args.max_new_tokens,
                                    few_shot=args.few_shot)
                err = None
            except Exception as e:
                result = {"raw_response": None, "parsed_json": None,
                          "parse_ok": False, "parse_error": f"fmt: {e}",
                          "runtime_seconds": 0, "n_input_tokens": 0, "n_output_tokens": 0}
                err = str(e)

            out_rec = {
                "image": rec["image"],
                "model": ocr_name,
                "ocr_text": ocr_text,
                "ocr_runtime_s": ocr_dt,
                "fmt_runtime_s": round(result["runtime_seconds"], 3),
                "total_runtime_s": round(ocr_dt + result["runtime_seconds"], 3),
                "runtime_seconds": round(ocr_dt + result["runtime_seconds"], 3),
                "parse_ok": result["parse_ok"],
                "parse_error": result["parse_error"],
                "parsed_json": result["parsed_json"],
                "raw_response": result["raw_response"],
                "n_input_tokens": result["n_input_tokens"],
                "n_output_tokens": result["n_output_tokens"],
                "error": err,
            }
            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            f_out.flush()

            if result["parse_ok"]:
                n_ok += 1
            else:
                n_fail += 1

            if i % 10 == 0 or i == len(todo):
                elapsed = time.perf_counter() - t_all
                rate = elapsed / i
                eta = rate * (len(todo) - i)
                print(f"[run] {i}/{len(todo)}  ok={n_ok}  fail={n_fail}  "
                      f"avg={rate:.1f}s/img  ETA {eta/60:.1f}min", flush=True)

    print(f"\n[run] done. ok={n_ok}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
