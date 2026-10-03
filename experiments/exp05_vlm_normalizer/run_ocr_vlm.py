"""Универсальный runner для OCR-VLM моделей второй волны Эксп. 5.

Подходящие модели:
    - deepseek-ai/DeepSeek-OCR
    - PaddlePaddle/PaddleOCR-VL-1.6
    - stepfun-ai/GOT-OCR2_0
    - baidu/Unlimited-OCR
    - datalab-to/chandra-ocr-2

Пайплайн (две стадии на одной GPU):
    1) OCR-VLM читает картинку → markdown/plain-text
    2) Форматтер (Qwen3-VL-8B-Instruct) превращает (image + text) → JSON-21

Если OCR-VLM умеет JSON сама (chandra-ocr-2) — флаг `--native-json` пропускает
форматтер.

Файл результата: outputs/<ocr_slug>__formatted.jsonl
Поля: image, model, ocr_text, ocr_runtime_s, fmt_runtime_s, total_runtime_s,
      parse_ok, parsed_json, raw_response, n_input_tokens, n_output_tokens, error.

Каждая OCR-модель имеет свою функцию загрузки (см. `OCR_LOADERS` ниже).
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
from typing import Callable

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.exp05_vlm_normalizer.run_qwen import read_jsonl, already_done
from experiments.exp05_vlm_normalizer.formatter import load_formatter, format_one


ROOT = Path(__file__).resolve().parents[2]
INPUTS_PATH = Path(__file__).resolve().parent / "merged_inputs.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OCR loaders: каждая возвращает (read_one_fn, model_objects)
# read_one_fn(image_path: str) -> str  # markdown/plain-text
# ============================================================

def _load_deepseek_ocr() -> tuple[Callable, dict]:
    """deepseek-ai/DeepSeek-OCR (3B)."""
    from transformers import AutoModel, AutoTokenizer
    model_id = "deepseek-ai/DeepSeek-OCR"
    print(f"[ocr] loading {model_id}", flush=True)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, use_safetensors=True,
        _attn_implementation="flash_attention_2",
    ).eval().cuda().to(torch.bfloat16)
    print(f"[ocr] DeepSeek-OCR loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    def read_one(image_path: str) -> str:
        # Gundam config: base_size=1024, image_size=640, crop_mode=True
        out_dir = ROOT / "data" / "tmp" / "deepseek_ocr_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt = "<image>\nFree OCR."

        # DeepSeek-OCR пишет результат прямо в stdout через print() внутри
        # model.infer(). Захватим его через io.StringIO.
        import contextlib, io
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                res = model.infer(
                    tokenizer, prompt=prompt, image_file=image_path,
                    output_path=str(out_dir), base_size=1024, image_size=640,
                    crop_mode=True, save_results=False, test_compress=False,
                )
            captured = buf.getvalue()
        except Exception as e:
            print(f"[ocr] DeepSeek inference error: {e}", flush=True)
            return ""

        # Если infer вернул строку — берём её. Иначе — выбираем последнюю
        # «содержательную» секцию stdout (после `==========` блока).
        if isinstance(res, str) and res.strip():
            return res
        if isinstance(res, dict):
            for k in ("text", "markdown", "output"):
                if res.get(k):
                    return res[k]
        # фоллбэк: парсим stdout
        # ищем «=====================» как разделитель, после него идёт текст
        lines = captured.splitlines()
        start = 0
        for i in range(len(lines) - 1, -1, -1):
            if "=====" in lines[i]:
                start = i + 1
                break
        text = "\n".join(lines[start:]).strip()
        return text

    return read_one, {"model": model, "tokenizer": tokenizer}


def _load_paddle_vl_16() -> tuple[Callable, dict]:
    """PaddlePaddle/PaddleOCR-VL-1.6 (0.9B). Требует transformers>=5.0."""
    from transformers import AutoProcessor, AutoModelForImageTextToText
    model_id = "PaddlePaddle/PaddleOCR-VL-1.6"
    print(f"[ocr] loading {model_id}", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"[ocr] PaddleOCR-VL-1.6 loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    @torch.inference_mode()
    def read_one(image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "OCR:"},
        ]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=2048)
        return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])

    return read_one, {"model": model, "processor": processor}


def _load_got_ocr2() -> tuple[Callable, dict]:
    """stepfun-ai/GOT-OCR2_0 (0.58B). Требует transformers<=4.42."""
    from transformers import AutoModel, AutoTokenizer
    model_id = "stepfun-ai/GOT-OCR2_0"
    print(f"[ocr] loading {model_id}", flush=True)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, low_cpu_mem_usage=True,
        device_map="cuda", use_safetensors=True, pad_token_id=tokenizer.eos_token_id,
    ).eval()
    print(f"[ocr] GOT-OCR2_0 loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    def read_one(image_path: str) -> str:
        try:
            return model.chat_crop(tokenizer, image_path, ocr_type="ocr") or ""
        except Exception:
            try:
                return model.chat(tokenizer, image_path, ocr_type="ocr") or ""
            except Exception:
                return ""

    return read_one, {"model": model, "tokenizer": tokenizer}


def _load_unlimited_ocr() -> tuple[Callable, dict]:
    """baidu/Unlimited-OCR (3B). Требует transformers>=4.57 + custom code."""
    from transformers import AutoModel, AutoTokenizer
    model_id = "baidu/Unlimited-OCR"
    print(f"[ocr] loading {model_id}", flush=True)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, use_safetensors=True,
    ).eval().cuda().to(torch.bfloat16)
    print(f"[ocr] Unlimited-OCR loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    out_dir = ROOT / "data" / "tmp" / "unlimited_ocr_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    def read_one(image_path: str) -> str:
        # Захват stdout (Unlimited.infer печатает результат через print)
        import contextlib, io
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                res = model.infer(
                    tokenizer, prompt="<image>document parsing.",
                    image_file=image_path, output_path=str(out_dir),
                    base_size=1024, image_size=640, crop_mode=True,
                    max_length=8192, no_repeat_ngram_size=35, ngram_window=128,
                    save_results=False,
                )
            captured = buf.getvalue()
        except Exception as e:
            print(f"[ocr] Unlimited error: {e}", flush=True)
            return ""

        if isinstance(res, str) and res.strip():
            return res
        if isinstance(res, dict):
            for k in ("text", "markdown", "output"):
                if res.get(k):
                    return res[k]
        # Fallback: парсим stdout. Unlimited печатает блоки `<|det|>...<|/det|>text\n`
        # без явного разделителя — берём всё что после загрузочных сообщений.
        # Эвристика: ищем первую строку с <|det|>.
        lines = captured.splitlines()
        start = 0
        for i, ln in enumerate(lines):
            if "<|det|>" in ln or "Sydow" in ln or "title" in ln.lower():
                start = i
                break
        return "\n".join(lines[start:]).strip()

    return read_one, {"model": model, "tokenizer": tokenizer}


def _load_chandra_ocr2() -> tuple[Callable, dict]:
    """datalab-to/chandra-ocr-2 (5B). Через HF transformers, нативный JSON."""
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
    print(f"[ocr] chandra-ocr-2 loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    @torch.inference_mode()
    def read_one(image_path: str) -> str:
        # chandra может через chandra-ocr пакет. Здесь — простой transformers-вызов.
        from chandra.model.hf import generate_hf
        from chandra.model.schema import BatchInputItem
        image = Image.open(image_path).convert("RGB")
        batch = [BatchInputItem(image=image, prompt_type="ocr_layout")]
        result = generate_hf(batch, model)[0]
        return result.raw if hasattr(result, "raw") else str(result)

    return read_one, {"model": model, "processor": processor}


def _load_glm_ocr() -> tuple[Callable, dict]:
    """zai-org/GLM-OCR (0.9B). Использует абсолютный путь снэпшота — обходит HF_HUB_OFFLINE-баг."""
    from transformers import AutoProcessor, AutoModelForImageTextToText
    snap = ROOT / "data/models/hf_cache/models--zai-org--GLM-OCR/snapshots/ca5d8b3e287e52589e37c28385d9655ee4372f9d"
    print(f"[ocr] loading GLM-OCR from {snap}", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        str(snap), dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    ).eval()
    processor = AutoProcessor.from_pretrained(str(snap))
    print(f"[ocr] GLM-OCR loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    PROMPT = ("Read all text on this herbarium label verbatim. "
              "Include every visible character: printed text, handwriting, "
              "barcodes, numbers, dates. Preserve line breaks where natural.")

    @torch.inference_mode()
    def read_one(image_path: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": PROMPT},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        gen = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        gen_trim = gen[:, inputs["input_ids"].shape[-1]:]
        return processor.batch_decode(gen_trim, skip_special_tokens=True)[0]

    return read_one, {"model": model, "processor": processor}


def _load_logics_parsing_v2() -> tuple[Callable, dict]:
    """Logics-MLLM/Logics-Parsing-v2 (4B, Qwen3VL backbone). Абсолютный путь снэпшота."""
    from transformers import AutoProcessor, AutoModelForImageTextToText
    snap = ROOT / "data/models/hf_cache/models--Logics-MLLM--Logics-Parsing-v2/snapshots/6b6c506897949fda4f1391b155e68bf3f99810f1"
    print(f"[ocr] loading Logics-Parsing-v2 from {snap}", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        str(snap), dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    ).eval()
    processor = AutoProcessor.from_pretrained(str(snap))
    print(f"[ocr] Logics-Parsing-v2 loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    PROMPT = ("Extract all text from this herbarium label. Output as markdown, "
              "preserving structure (headings, lines, handwritten vs printed). "
              "Read every visible character verbatim: scientific names, collector, "
              "locality, date, barcodes, numbers, handwriting.")

    @torch.inference_mode()
    def read_one(image_path: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": PROMPT},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        gen = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        gen_trim = gen[:, inputs["input_ids"].shape[-1]:]
        return processor.batch_decode(gen_trim, skip_special_tokens=True)[0]

    return read_one, {"model": model, "processor": processor}


def _load_qwen3_5_vlm(model_id: str, prompt: str = "Convert the document to markdown.",
                       max_new_tokens: int = 1024, no_repeat_ngram_size: int = 0):
    """Universal loader for qwen3_5-arch OCR-VLMs (chandra-ocr-2, lift, surya-ocr-2)."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[ocr] loading {model_id}", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"[ocr] {model_id} loaded in {time.perf_counter()-t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)", flush=True)

    @torch.inference_mode()
    def read_one(image_path: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
        if no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        gen = model.generate(**inputs, **gen_kwargs)
        gen_trim = gen[:, inputs["input_ids"].shape[-1]:]
        return processor.batch_decode(gen_trim, skip_special_tokens=True)[0]

    return read_one, {"model": model, "processor": processor}


def _load_chandra_ocr2_simple() -> tuple[Callable, dict]:
    """chandra-ocr-2 via universal qwen3_5 loader (no chandra-ocr package needed)."""
    return _load_qwen3_5_vlm("datalab-to/chandra-ocr-2",
                              prompt="Convert the document to markdown.",
                              max_new_tokens=1024)


def _load_lift() -> tuple[Callable, dict]:
    """datalab-to/lift (10B, qwen3_5)."""
    return _load_qwen3_5_vlm("datalab-to/lift",
                              prompt="Convert the document to markdown.",
                              max_new_tokens=1024)


def _load_surya_ocr2() -> tuple[Callable, dict]:
    """datalab-to/surya-ocr-2 (0.7B, qwen3_5)."""
    return _load_qwen3_5_vlm("datalab-to/surya-ocr-2",
                              prompt="Convert the document to markdown.",
                              max_new_tokens=1024)


OCR_LOADERS = {
    "deepseek-ocr": _load_deepseek_ocr,
    "paddleocr-vl-1.6": _load_paddle_vl_16,
    "got-ocr2": _load_got_ocr2,
    "unlimited-ocr": _load_unlimited_ocr,
    "chandra-ocr-2": _load_chandra_ocr2,
    "chandra-ocr-2-simple": _load_chandra_ocr2_simple,
    "lift": _load_lift,
    "surya-ocr-2": _load_surya_ocr2,
    "glm-ocr": _load_glm_ocr,
    "logics-parsing-v2": _load_logics_parsing_v2,
}


# ============================================================
# Main loop
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", required=True, choices=list(OCR_LOADERS.keys()))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--inputs", default=str(INPUTS_PATH))
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--no-formatter", action="store_true",
                    help="OCR-VLM already produces JSON — skip formatter step.")
    ap.add_argument("--formatter-model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--few-shot", action="store_true",
                    help="Use few-shot prompt (mode F_fs) in formatter.")
    args = ap.parse_args()

    slug = args.ocr.replace("/", "__")
    fs_marker = "_fs" if args.few_shot and not args.no_formatter else ""
    suffix = ("__raw" if args.no_formatter else f"__formatted{fs_marker}") + args.out_suffix
    out_path = OUT_DIR / f"{slug}{suffix}.jsonl"
    print(f"[run] ocr={args.ocr} out={out_path}  formatter={not args.no_formatter}", flush=True)

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

    # 1) Загрузить OCR-VLM
    ocr_loader = OCR_LOADERS[args.ocr]
    read_one, ocr_state = ocr_loader()

    # 2) Загрузить форматтер (если нужен)
    fmt_model = fmt_proc = None
    if not args.no_formatter:
        fmt_model, fmt_proc = load_formatter(args.formatter_model)

    t_all = time.perf_counter()
    n_ok = 0
    n_fail = 0

    with out_path.open("a", encoding="utf-8") as f_out:
        for i, rec in enumerate(todo, 1):
            image_path = rec["image_path"]

            # --- OCR ---
            t0 = time.perf_counter()
            try:
                ocr_text = read_one(image_path) or ""
            except Exception as e:
                ocr_text = ""
                print(f"  ocr FAILED on {rec['image']}: {e}", flush=True)
            ocr_dt = time.perf_counter() - t0

            # --- formatter (или сразу пишем raw OCR) ---
            if args.no_formatter:
                # chandra-ocr-2 уже выдаёт JSON — пробуем распарсить.
                from experiments.exp05_vlm_normalizer.run_qwen import parse_response
                parsed, perr = parse_response(ocr_text)
                rec_out = {
                    "image": rec["image"],
                    "model": args.ocr,
                    "ocr_text": ocr_text,
                    "ocr_runtime_s": round(ocr_dt, 3),
                    "fmt_runtime_s": 0,
                    "total_runtime_s": round(ocr_dt, 3),
                    "parse_ok": parsed is not None,
                    "parse_error": perr,
                    "parsed_json": parsed,
                    "raw_response": ocr_text,
                    "n_input_tokens": -1, "n_output_tokens": -1,
                    "error": None,
                }
            else:
                try:
                    fmt_result = format_one(
                        fmt_model, fmt_proc, image_path, ocr_text, args.ocr,
                        max_new_tokens=args.max_new_tokens,
                        few_shot=args.few_shot,
                    )
                    err = None
                except Exception as e:
                    fmt_result = {
                        "raw_response": None, "parsed_json": None,
                        "parse_ok": False, "parse_error": f"fmt: {e}",
                        "runtime_seconds": 0, "n_input_tokens": 0, "n_output_tokens": 0,
                    }
                    err = str(e)

                rec_out = {
                    "image": rec["image"],
                    "model": args.ocr,
                    "ocr_text": ocr_text,
                    "ocr_runtime_s": round(ocr_dt, 3),
                    "fmt_runtime_s": round(fmt_result["runtime_seconds"], 3),
                    "total_runtime_s": round(ocr_dt + fmt_result["runtime_seconds"], 3),
                    "parse_ok": fmt_result["parse_ok"],
                    "parse_error": fmt_result["parse_error"],
                    "parsed_json": fmt_result["parsed_json"],
                    "raw_response": fmt_result["raw_response"],
                    "n_input_tokens": fmt_result["n_input_tokens"],
                    "n_output_tokens": fmt_result["n_output_tokens"],
                    "error": err,
                }

            # Для совместимости с evaluate.py: дублируем total_runtime под runtime_seconds.
            rec_out["runtime_seconds"] = rec_out["total_runtime_s"]

            f_out.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            f_out.flush()

            if rec_out["parse_ok"]:
                n_ok += 1
            else:
                n_fail += 1

            if i % 5 == 0 or i == len(todo):
                elapsed = time.perf_counter() - t_all
                rate = elapsed / i
                eta = rate * (len(todo) - i)
                print(f"[run] {i}/{len(todo)}  ok={n_ok}  fail={n_fail}  "
                      f"ocr={ocr_dt:.1f}s  total={rec_out['total_runtime_s']:.1f}s  "
                      f"avg={rate:.1f}s/img  ETA {eta/60:.1f}min", flush=True)

    print(f"\n[run] done. ok={n_ok}/{len(todo)} ({n_ok/len(todo)*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
