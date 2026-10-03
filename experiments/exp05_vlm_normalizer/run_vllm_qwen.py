"""Qwen3-VL через vLLM offline-batch (для измерения prod-скорости).

Параметры по умолчанию подобраны для H100 80GB:
    --batch-size 8 — параллельные изображения за раз
    --tensor-parallel 1 — одна GPU
    --max-model-len 16384

CLI:
    python -m experiments.exp05_vlm_normalizer.run_vllm_qwen \
        --model Qwen/Qwen3-VL-8B-Instruct --mode A
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.pop("PYTHONSTARTUP", None)
os.environ.pop("PYTHONPATH", None)
os.environ.setdefault("VLLM_USE_V1", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.exp05_vlm_normalizer.prompts.builder import build_prompt
from experiments.exp05_vlm_normalizer.run_qwen import (
    parse_response, read_jsonl, already_done,
)


ROOT = Path(__file__).resolve().parents[2]
INPUTS_PATH = Path(__file__).resolve().parent / "merged_inputs.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["A", "B"], default="A")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--inputs", default=str(INPUTS_PATH))
    ap.add_argument("--out-suffix", default="__vllm")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    slug = args.model.replace("/", "__")
    out_path = OUT_DIR / f"{slug}__mode{args.mode}{args.out_suffix}.jsonl"
    print(f"[run] model={args.model} mode={args.mode}  out={out_path}", flush=True)

    skip = already_done(out_path)
    records = list(read_jsonl(Path(args.inputs)))
    if args.limit:
        records = records[: args.limit]
    todo = [r for r in records if r["image"] not in skip]
    print(f"[run] todo = {len(todo)} / {len(records)}  (skip={len(skip)})", flush=True)
    if not todo:
        return

    # vLLM offline-инференс
    from vllm import LLM, SamplingParams

    print(f"[load] {args.model}  tp={args.tensor_parallel}  ml={args.max_model_len}", flush=True)
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )
    print(f"[load] done in {time.perf_counter()-t0:.1f}s", flush=True)

    t_all = time.perf_counter()
    n_ok = 0
    n_fail = 0

    # Используем vLLM chat API (для VLM)
    with out_path.open("a", encoding="utf-8") as f_out:
        for batch_start in range(0, len(todo), args.batch_size):
            batch = todo[batch_start: batch_start + args.batch_size]

            # build chat-messages для каждого
            conversations = []
            for rec in batch:
                ocr = rec["ocr"] if args.mode == "B" else None
                prompt_parts = build_prompt(args.mode, ocr=ocr)
                conversations.append([
                    {"role": "system", "content": prompt_parts["system"]},
                    {"role": "user", "content": [
                        {"type": "image_pil",
                         "image_pil": Image.open(rec["image_path"]).convert("RGB")},
                        {"type": "text", "text": prompt_parts["user_text"]},
                    ]},
                ])

            t_batch = time.perf_counter()
            outputs = llm.chat(conversations, sampling, use_tqdm=False)
            dt_batch = time.perf_counter() - t_batch
            per_img = dt_batch / max(1, len(batch))

            for rec, out in zip(batch, outputs):
                text = out.outputs[0].text
                parsed, perr = parse_response(text)
                f_out.write(json.dumps({
                    "image": rec["image"],
                    "model": args.model,
                    "mode": args.mode,
                    "raw_response": text,
                    "parsed_json": parsed,
                    "parse_ok": parsed is not None,
                    "parse_error": perr,
                    "runtime_seconds": per_img,
                    "n_output_tokens": len(out.outputs[0].token_ids),
                    "n_input_tokens": len(out.prompt_token_ids) if out.prompt_token_ids else -1,
                    "error": None,
                }, ensure_ascii=False) + "\n")
                f_out.flush()
                if parsed is not None:
                    n_ok += 1
                else:
                    n_fail += 1

            done = batch_start + len(batch)
            elapsed = time.perf_counter() - t_all
            avg = elapsed / done
            eta = avg * (len(todo) - done)
            print(f"[run] {done}/{len(todo)}  ok={n_ok}  fail={n_fail}  "
                  f"batch={dt_batch:.1f}s ({per_img:.1f}s/img)  "
                  f"avg={avg:.1f}s/img  ETA {eta/60:.1f}min", flush=True)

    print(f"\n[run] done. ok={n_ok}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
