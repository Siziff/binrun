"""Запуск InternVL3-8B-hf на наших данных.

InternVL3-8B-hf — официальный HF-порт от OpenGVLab, использует встроенную
архитектуру `InternVLForConditionalGeneration` в transformers 5.x.
Работает через тот же chat-template API, что и Qwen3-VL.

CLI идентичный run_qwen.py.
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
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.exp05_vlm_normalizer.prompts.builder import build_prompt
from experiments.exp05_vlm_normalizer.run_qwen import (
    parse_response, read_jsonl, already_done,
)


ROOT = Path(__file__).resolve().parents[2]
INPUTS_PATH = Path(__file__).resolve().parent / "merged_inputs.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(model_id: str):
    print(f"[load] {model_id}", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    elapsed = time.perf_counter() - t0
    print(f"[load] done in {elapsed:.1f}s", flush=True)
    return model, processor


@torch.inference_mode()
def generate_one(model, processor, image_path: str, mode: str, ocr: dict | None,
                 max_new_tokens: int = 12288) -> dict:
    prompt = build_prompt(mode, ocr=ocr)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt["system"]}]},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt["user_text"]},
        ]},
    ]
    t0 = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    n_input = int(inputs["input_ids"].shape[-1])
    gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_trim = gen_ids[:, inputs["input_ids"].shape[-1]:]
    n_output = int(gen_trim.shape[-1])
    text = processor.batch_decode(gen_trim, skip_special_tokens=True)[0]
    runtime = time.perf_counter() - t0
    parsed, perr = parse_response(text)
    return {
        "raw_response": text,
        "parsed_json": parsed,
        "parse_ok": parsed is not None,
        "parse_error": perr,
        "runtime_seconds": runtime,
        "n_input_tokens": n_input,
        "n_output_tokens": n_output,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OpenGVLab/InternVL3-8B-hf")
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=12288)
    ap.add_argument("--inputs", default=str(INPUTS_PATH))
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    model_slug = args.model.replace("/", "__")
    out_path = OUT_DIR / f"{model_slug}__mode{args.mode}{args.out_suffix}.jsonl"

    print(f"[run] model={args.model} mode={args.mode}", flush=True)
    print(f"[run] inputs={args.inputs}", flush=True)
    print(f"[run] out={out_path}", flush=True)

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

    model, processor = load_model(args.model)
    t_all = time.perf_counter()
    n_ok = 0
    n_fail = 0

    with out_path.open("a", encoding="utf-8") as f_out:
        for i, rec in enumerate(todo, 1):
            ocr = rec["ocr"] if args.mode == "B" else None
            try:
                result = generate_one(
                    model, processor, rec["image_path"], args.mode, ocr,
                    max_new_tokens=args.max_new_tokens,
                )
                err = None
            except Exception as e:
                result = {
                    "raw_response": None, "parsed_json": None, "parse_ok": False,
                    "parse_error": f"runtime: {type(e).__name__}: {e}",
                    "runtime_seconds": 0, "n_input_tokens": 0, "n_output_tokens": 0,
                }
                err = str(e)

            out_rec = {
                "image": rec["image"],
                "model": args.model,
                "mode": args.mode,
                **result,
                "error": err,
            }
            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            f_out.flush()
            if result["parse_ok"]:
                n_ok += 1
            else:
                n_fail += 1
            if i % 5 == 0 or i == len(todo):
                elapsed = time.perf_counter() - t_all
                rate = elapsed / i
                eta = rate * (len(todo) - i)
                print(f"[run] {i}/{len(todo)}  ok={n_ok}  fail={n_fail}  "
                      f"{rate:.1f}s/img  ETA {eta/60:.1f}min", flush=True)

    print(f"\n[run] done. ok={n_ok}/{len(todo)} ({n_ok/len(todo)*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
