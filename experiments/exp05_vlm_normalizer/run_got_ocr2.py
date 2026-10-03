"""GOT-OCR2.0 — узкоспециальная OCR-only модель (~0.58B).

Аналогично PaddleOCR-VL: даёт сырой текст, затем тот же Qwen3-VL-8B
post-processor превращает его в JSON.
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
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.exp05_vlm_normalizer.run_qwen import (
    parse_response, read_jsonl, already_done,
)
from experiments.exp05_vlm_normalizer.run_paddleocr_vl import (
    load_qwen_post, qwen_post_generate,
)

INPUTS_PATH = Path(__file__).resolve().parent / "merged_inputs.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_got():
    model_id = "ucaslcl/GOT-OCR2_0"
    print(f"[load] {model_id}", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, low_cpu_mem_usage=True,
        device_map="cuda", use_safetensors=True, pad_token_id=tok.eos_token_id,
    ).eval()
    print(f"[load] GOT-OCR2.0 done in {time.perf_counter()-t0:.1f}s", flush=True)
    return model, tok


def got_read(model, tok, image_path: str) -> str:
    # Используем 'multi-crop ocr' для документов.
    try:
        text = model.chat_crop(tok, image_path, ocr_type="ocr")
    except Exception:
        text = model.chat(tok, image_path, ocr_type="ocr")
    return text or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=12288)
    ap.add_argument("--inputs", default=str(INPUTS_PATH))
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    model_name = "stepfun-ai__GOT-OCR2.0"
    out_path = OUT_DIR / f"{model_name}__mode{args.mode}{args.out_suffix}.jsonl"
    print(f"[run] model={model_name} mode={args.mode}  out={out_path}", flush=True)

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

    got_model, got_tok = load_got()
    qwen_model, qwen_proc = load_qwen_post()

    t_all = time.perf_counter()
    n_ok = 0; n_fail = 0
    with out_path.open("a", encoding="utf-8") as f_out:
        for i, rec in enumerate(todo, 1):
            t0 = time.perf_counter()
            try:
                got_text = got_read(got_model, got_tok, rec["image_path"])
                got_time = time.perf_counter() - t0
            except Exception as e:
                got_text, got_time = "", 0.0
                f_out.write(json.dumps({
                    "image": rec["image"], "model": model_name, "mode": args.mode,
                    "parse_ok": False, "parse_error": f"got: {e}",
                    "raw_response": None, "parsed_json": None,
                    "got_text": "", "got_runtime_s": 0,
                    "post_runtime_s": 0, "error": str(e),
                }, ensure_ascii=False) + "\n")
                f_out.flush(); n_fail += 1; continue

            try:
                ocr_b = rec["ocr"] if args.mode == "B" else None
                # Для post-processor подменим "paddleocr-vl" → "got-ocr2"
                # но качество в JSON одинаковое (это primary OCR text).
                result = qwen_post_generate(
                    qwen_model, qwen_proc, rec["image_path"],
                    got_text, args.mode, ocr_b, max_new_tokens=args.max_new_tokens,
                )
                err = None
            except Exception as e:
                result = {"raw_response": None, "parsed_json": None,
                          "parse_ok": False, "parse_error": f"post: {e}",
                          "runtime_seconds": 0, "n_input_tokens": 0, "n_output_tokens": 0}
                err = str(e)

            out_rec = {
                "image": rec["image"], "model": model_name, "mode": args.mode,
                **result,
                "got_text": got_text,
                "got_runtime_s": round(got_time, 2),
                "post_runtime_s": round(result["runtime_seconds"], 2),
                "error": err,
            }
            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            f_out.flush()
            if result["parse_ok"]: n_ok += 1
            else: n_fail += 1
            if i % 5 == 0 or i == len(todo):
                elapsed = time.perf_counter() - t_all
                rate = elapsed / i
                eta = rate * (len(todo) - i)
                print(f"[run] {i}/{len(todo)}  ok={n_ok}  fail={n_fail}  "
                      f"{rate:.1f}s/img  ETA {eta/60:.1f}min", flush=True)

    print(f"\n[run] done. ok={n_ok}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
