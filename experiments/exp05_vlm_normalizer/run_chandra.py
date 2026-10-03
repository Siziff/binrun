"""Запуск datalab-to/chandra-ocr-2 через transformers + chandra-ocr пакет.

chandra-ocr-2 нативно выдаёт markdown + JSON через свой пакет.
Используем `chandra.model.hf.generate_hf` для batched inference.

Затем форматтер (Qwen3-VL-8B-Instruct) превращает markdown → 21-полевой JSON.

CLI:
    python -m experiments.exp05_vlm_normalizer.run_chandra --mode A [--limit N]
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

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.exp05_vlm_normalizer.run_qwen import read_jsonl, already_done, parse_response


ROOT = Path(__file__).resolve().parents[2]
INPUTS_PATH = Path(__file__).resolve().parent / "merged_inputs.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_chandra():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model_id = "datalab-to/chandra-ocr-2"
    print(f"[ocr] loading {model_id}", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    model.processor = processor
    print(f"[ocr] chandra-ocr-2 loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    return model, processor


def chandra_ocr_one(model, processor, image_path: str, prompt_type: str = "ocr_layout") -> str:
    from chandra.model.hf import generate_hf
    from chandra.model.schema import BatchInputItem
    image = Image.open(image_path).convert("RGB")
    batch = [BatchInputItem(image=image, prompt_type=prompt_type)]
    result = generate_hf(batch, model)[0]
    # raw содержит markdown с layout-тэгами
    return result.raw if hasattr(result, "raw") else str(result)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B"], default="A")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--inputs", default=str(INPUTS_PATH))
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--prompt-type", default="ocr_layout")
    args = ap.parse_args()

    slug = "datalab-to__chandra-ocr-2"
    suffix = f"__mode{args.mode}{args.out_suffix}__raw_md"
    out_path = OUT_DIR / f"{slug}{suffix}.jsonl"

    print(f"[run] ocr={slug} mode={args.mode} out={out_path}", flush=True)

    skip = set() if args.no_resume else already_done(out_path)
    if skip:
        print(f"[run] resume: already done = {len(skip)}", flush=True)
    records = list(read_jsonl(Path(args.inputs)))
    if args.limit:
        records = records[: args.limit]
    todo = [r for r in records if r["image"] not in skip]
    print(f"[run] todo = {len(todo)} / {len(records)}", flush=True)
    if not todo:
        return

    model, processor = load_chandra()

    t_all = time.perf_counter()
    n_done = 0
    n_err = 0

    with out_path.open("a", encoding="utf-8") as f_out:
        for i, rec in enumerate(todo, 1):
            t0 = time.perf_counter()
            try:
                ocr_md = chandra_ocr_one(model, processor, rec["image_path"],
                                          prompt_type=args.prompt_type)
                err = None
            except Exception as e:
                ocr_md = ""
                err = str(e)
                n_err += 1
            ocr_dt = time.perf_counter() - t0

            out_rec = {
                "image": rec["image"],
                "model": "datalab-to/chandra-ocr-2",
                "mode": args.mode,
                "prompt_type": args.prompt_type,
                "ocr_markdown": ocr_md,
                "ocr_runtime_s": round(ocr_dt, 3),
                "runtime_seconds": round(ocr_dt, 3),
                "error": err,
            }
            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            f_out.flush()
            n_done += 1

            if i % 5 == 0 or i == len(todo):
                elapsed = time.perf_counter() - t_all
                rate = elapsed / i
                eta = rate * (len(todo) - i)
                print(f"[run] {i}/{len(todo)}  ok={n_done-n_err}  err={n_err}  "
                      f"avg={rate:.1f}s/img  ETA {eta/60:.1f}min", flush=True)

    print(f"\n[run] done. {n_done-n_err}/{n_done}", flush=True)


if __name__ == "__main__":
    main()
