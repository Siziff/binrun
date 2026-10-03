"""Запуск Qwen3-VL (8B-Thinking / 30B-A3B-Thinking MoE).

Пайплайн:
  1. Загружаем модель через AutoModelForImageTextToText.from_pretrained(
       …, dtype=torch.bfloat16, device_map="auto") — distribuируется по GPU.
  2. По merged_inputs.jsonl: для каждой записи строим chat-messages с
     картинкой и текстовым prompt (mode A / mode B).
  3. processor.apply_chat_template → inputs → model.generate(max_new_tokens=2048,
     do_sample=False).
  4. Парсим JSON-ответ (с устойчивостью к thinking-блокам и markdown-обёрткам).
  5. Пишем в outputs/<model>_<mode>.jsonl: {image, mode, model, raw_response,
     parsed_json, parse_ok, runtime_seconds, n_input_tokens, n_output_tokens}.

CLI:
    python -m experiments.exp05_vlm_normalizer.run_qwen \
        --model Qwen/Qwen3-VL-8B-Thinking \
        --mode A \
        [--limit N]
"""
from __future__ import annotations

# === isolation: только td-vlm site-packages, без base/user сайтов ===
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.pop("PYTHONSTARTUP", None)
os.environ.pop("PYTHONPATH", None)

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image

# Будем импортировать класс модели позже, чтобы быстрее упасть при
# неправильном env.
from transformers import AutoProcessor

# Промпт-билдер
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.exp05_vlm_normalizer.prompts.builder import build_prompt, SCHEMA


ROOT = Path(__file__).resolve().parents[2]
INPUTS_PATH = Path(__file__).resolve().parent / "merged_inputs.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------- утилиты ----------------------------

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
MD_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def strip_thinking(text: str) -> str:
    """Срезает <think>...</think> блоки (Qwen3-VL Thinking)."""
    return THINK_BLOCK_RE.sub("", text).strip()


def parse_response(raw: str) -> tuple[dict | None, str]:
    """Возвращает (parsed, error). parsed=None если не распарсилось.

    Алгоритм:
      1. убираем <think>...</think>;
      2. убираем markdown-fence ```json ... ``` ;
      3. ищем самый большой JSON-объект regex'ом, парсим;
      4. при ошибке — пробуем .replace на ASCII-кавычки.
    """
    text = strip_thinking(raw)
    text = MD_FENCE_RE.sub("", text).strip()
    candidates = JSON_BLOCK_RE.findall(text)
    if not candidates:
        return None, "no JSON object found"
    candidate = max(candidates, key=len)
    try:
        obj = json.loads(candidate)
        return obj, ""
    except Exception as e:
        # Иногда модель ставит «умные» кавычки
        cleaned = (candidate.replace("\u201c", '"').replace("\u201d", '"')
                            .replace("\u2018", "'").replace("\u2019", "'"))
        try:
            obj = json.loads(cleaned)
            return obj, ""
        except Exception as e2:
            return None, f"json parse error: {e2}"


def read_jsonl(p: Path) -> Iterable[dict]:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def already_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    out = set()
    for rec in read_jsonl(out_path):
        out.add(rec.get("image"))
    return out


# ---------------------------- основная логика ----------------------------

def load_model(model_id: str, load_4bit: bool = False,
               max_memory_per_gpu_gib: int | None = None,
               use_flash_attn: bool = True):
    """Загружает VLM с device_map='auto'.

    Опционально:
        load_4bit=True              — bitsandbytes nf4 4-bit (для больших MoE).
        max_memory_per_gpu_gib=N    — ограничить VRAM на каждой видимой GPU.
        use_flash_attn=True         — попытаться использовать `flash_attention_2`,
                                      при ошибке откатимся к `sdpa`.
    """
    print(f"[load] {model_id}  4bit={load_4bit}  "
          f"max_mem={max_memory_per_gpu_gib}  flash_attn={use_flash_attn}", flush=True)
    from transformers import AutoModelForImageTextToText
    t0 = time.perf_counter()
    kwargs = {"device_map": "auto"}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["dtype"] = torch.bfloat16
    if max_memory_per_gpu_gib is not None:
        n = torch.cuda.device_count()
        kwargs["max_memory"] = {i: f"{max_memory_per_gpu_gib}GiB" for i in range(n)}
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except (ImportError, ValueError) as e:
        if use_flash_attn:
            print(f"[load] flash-attn unavailable ({e}); fallback to SDPA", flush=True)
            kwargs.pop("attn_implementation", None)
            model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        else:
            raise
    processor = AutoProcessor.from_pretrained(model_id)
    elapsed = time.perf_counter() - t0
    print(f"[load] done in {elapsed:.1f}s. devices: {set(p.device for p in model.parameters())}", flush=True)
    return model, processor


def build_messages(image_path: str, mode: str, ocr: dict | None) -> tuple[list, str]:
    prompt = build_prompt(mode, ocr=ocr)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt["system"]}]},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt["user_text"]},
        ]},
    ]
    return messages, prompt["system"] + "\n\n" + prompt["user_text"]


@torch.inference_mode()
def generate_one(model, processor, image_path: str, mode: str, ocr: dict | None,
                 max_new_tokens: int = 2048) -> dict:
    messages, _ = build_messages(image_path, mode, ocr)
    t0 = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    n_input = int(inputs["input_ids"].shape[-1])
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
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
    ap.add_argument("--model", required=True, help="HF id, e.g. Qwen/Qwen3-VL-8B-Thinking")
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=12288)
    ap.add_argument("--inputs", default=str(INPUTS_PATH))
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--out-suffix", default="",
                    help="Дополнительный суффикс к имени output (для шардов).")
    ap.add_argument("--load-4bit", action="store_true",
                    help="Загружать модель в 4-bit квантизации (для 30B на ограниченной VRAM).")
    ap.add_argument("--max-memory-gib", type=int, default=None,
                    help="Limit per-GPU memory for device_map='auto' (например 30).")
    ap.add_argument("--no-flash-attn", action="store_true",
                    help="Не использовать flash-attention 2 (fallback на SDPA).")
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

    model, processor = load_model(
        args.model, load_4bit=args.load_4bit,
        max_memory_per_gpu_gib=args.max_memory_gib,
        use_flash_attn=not args.no_flash_attn,
    )
    t_all = time.perf_counter()
    n_ok = 0
    n_parse_fail = 0

    with out_path.open("a", encoding="utf-8") as f_out:
        for i, rec in enumerate(todo, 1):
            image_path = rec["image_path"]
            ocr = rec["ocr"] if args.mode == "B" else None
            try:
                result = generate_one(
                    model, processor, image_path, args.mode, ocr,
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
                n_parse_fail += 1

            if i % 5 == 0 or i == len(todo):
                elapsed = time.perf_counter() - t_all
                rate = elapsed / i
                eta = rate * (len(todo) - i)
                print(f"[run] {i}/{len(todo)}  parse_ok={n_ok}  fail={n_parse_fail}  "
                      f"{rate:.1f}s/img  ETA {eta/60:.1f}min", flush=True)

    print(f"\n[run] done. parse_ok={n_ok}/{len(todo)} ({n_ok/len(todo)*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
