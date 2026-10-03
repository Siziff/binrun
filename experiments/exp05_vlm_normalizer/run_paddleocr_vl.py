"""PaddleOCR-VL (0.9B) — element-level OCR.

Эта модель НЕ агент: на вход — картинка + промпт `"OCR:"`, на выход — сырой
markdown-текст распознанной страницы. Чтобы получить JSON-схему,
дополнительно прогоняем сырой текст через text-only LLM (Qwen3-VL-8B —
он же в режиме text-only, без vision).

В mode A: VLM смотрит только на картинку → текст → post-LLM → JSON.
В mode B: VLM смотрит на картинку → текст; post-LLM получает картинку +
          OCR-выводы PaddleOCR (этот же текст) + Surya + TrOCR.
          Это даёт честное сравнение «PaddleOCR-VL as primary OCR».
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
from transformers import AutoModelForCausalLM, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.exp05_vlm_normalizer.prompts.builder import build_prompt
from experiments.exp05_vlm_normalizer.run_qwen import (
    parse_response, read_jsonl, already_done,
)


ROOT = Path(__file__).resolve().parents[2]
INPUTS_PATH = Path(__file__).resolve().parent / "merged_inputs.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_paddle_vl(model_id: str = "PaddlePaddle/PaddleOCR-VL"):
    print(f"[load] {model_id}", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print(f"[load] PaddleOCR-VL done in {time.perf_counter()-t0:.1f}s", flush=True)
    return model, processor


@torch.inference_mode()
def paddle_vl_read(model, processor, image_path: str, max_new_tokens: int = 2048) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "OCR:"},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    # Срезаем начало промпта.
    if "OCR:" in text:
        text = text.split("OCR:", 1)[-1].strip()
    return text


# --- Postprocessor: Qwen3-VL-8B-Thinking. Уже умеет JSON.
def load_qwen_post(model_id: str = "Qwen/Qwen3-VL-8B-Thinking"):
    from transformers import AutoModelForImageTextToText
    print(f"[load] post-LLM {model_id}", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"[load] post-LLM done in {time.perf_counter()-t0:.1f}s", flush=True)
    return model, processor


@torch.inference_mode()
def qwen_post_generate(model, processor, image_path: str, paddle_text: str,
                       mode: str, other_ocr: dict | None,
                       max_new_tokens: int = 12288) -> dict:
    """Финальный шаг: даём картинку + PaddleOCR-VL output (+ другие OCR если mode=B)."""
    # Конструируем ocr-словарь: PaddleOCR-VL играет роль primary OCR.
    if mode == "B":
        merged_ocr = dict(other_ocr or {})
        merged_ocr["paddleocr-vl"] = {"full_text": paddle_text}
    else:
        merged_ocr = {"paddleocr-vl": {"full_text": paddle_text}}

    # Сборка промпта: mode='B' с одним «движком» в mode='A' и с тремя в mode='B'.
    # Используем тот же builder, но подменим список движков напрямую.
    from experiments.exp05_vlm_normalizer.prompts.builder import _user_intro, SYSTEM_MD

    user = _user_intro()
    if mode == "A":
        user += ("\n\nИзучи изображение этикетки. Дополнительно у тебя есть OCR-"
                 "вывод от PaddleOCR-VL (document VLM). Используй его как подсказку, "
                 "проверяй визуально по картинке.\n\n"
                 f"### OCR (PaddleOCR-VL)\n```\n{paddle_text[:4000]}\n```\n\n"
                 "Теперь выдай JSON.")
    else:
        ocr_blocks = [f"### OCR (PaddleOCR-VL)\n```\n{paddle_text[:4000]}\n```"]
        for eng in ("paddleocr", "surya", "trocr-printed"):
            t = (other_ocr.get(eng) or {}).get("full_text") or ""
            ocr_blocks.append(f"### OCR ({eng})\n```\n{t[:4000]}\n```")
        user += ("\n\nИзучи изображение этикетки и сверь его с OCR-выводами. "
                 "Доверяй прежде всего изображению; OCR — подсказки.\n\n"
                 + "\n\n".join(ocr_blocks)
                 + "\n\nТеперь выдай JSON.")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MD}]},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": user},
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
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=12288)
    ap.add_argument("--inputs", default=str(INPUTS_PATH))
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    model_name = "PaddlePaddle__PaddleOCR-VL"
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

    paddle_model, paddle_proc = load_paddle_vl()
    qwen_model, qwen_proc = load_qwen_post()

    t_all = time.perf_counter()
    n_ok = 0
    n_fail = 0
    with out_path.open("a", encoding="utf-8") as f_out:
        for i, rec in enumerate(todo, 1):
            t0 = time.perf_counter()
            try:
                paddle_text = paddle_vl_read(paddle_model, paddle_proc, rec["image_path"])
                paddle_time = time.perf_counter() - t0
            except Exception as e:
                paddle_text, paddle_time = "", 0.0
                f_out.write(json.dumps({
                    "image": rec["image"], "model": model_name, "mode": args.mode,
                    "parse_ok": False, "parse_error": f"paddle: {e}",
                    "raw_response": None, "parsed_json": None,
                    "paddle_text": "", "paddle_runtime_s": 0,
                    "post_runtime_s": 0, "error": str(e),
                }, ensure_ascii=False) + "\n")
                f_out.flush()
                n_fail += 1
                continue

            try:
                ocr_b = rec["ocr"] if args.mode == "B" else None
                result = qwen_post_generate(
                    qwen_model, qwen_proc, rec["image_path"],
                    paddle_text, args.mode, ocr_b, max_new_tokens=args.max_new_tokens,
                )
                err = None
            except Exception as e:
                result = {
                    "raw_response": None, "parsed_json": None, "parse_ok": False,
                    "parse_error": f"post: {e}",
                    "runtime_seconds": 0, "n_input_tokens": 0, "n_output_tokens": 0,
                }
                err = str(e)

            out_rec = {
                "image": rec["image"],
                "model": model_name,
                "mode": args.mode,
                **result,
                "paddle_text": paddle_text,
                "paddle_runtime_s": round(paddle_time, 2),
                "post_runtime_s": round(result["runtime_seconds"], 2),
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

    print(f"\n[run] done. ok={n_ok}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
