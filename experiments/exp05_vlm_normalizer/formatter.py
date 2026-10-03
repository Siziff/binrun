"""LLM-форматтер: преобразует выход OCR-VLM в наш JSON по 21-полевой схеме.

Используется в Эксп. 5b для моделей, выдающих markdown/plain text (DeepSeek-OCR,
PaddleOCR-VL, GOT-OCR2_0, Unlimited-OCR). Те, что нативно выдают JSON
(chandra-ocr-2), могут обойти этот шаг.

Архитектура:
    картинка + ocr_text → Qwen3-VL-8B-Instruct (text-fusion mode) → JSON

Поддерживает few-shot (5 примеров) через mode='F_fs'.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.pop("PYTHONSTARTUP", None)
os.environ.pop("PYTHONPATH", None)

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.exp05_vlm_normalizer.prompts.builder import (
    SYSTEM_MD, _user_intro, build_prompt, FEW_SHOT_IMAGE_NAMES,
)
from experiments.exp05_vlm_normalizer.run_qwen import parse_response


DEFAULT_FORMATTER_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def load_formatter(model_id: str = DEFAULT_FORMATTER_MODEL,
                   device_map: str = "auto",
                   use_flash_attn: bool = True,
                   max_memory_gib: int | None = None):
    from transformers import AutoModelForImageTextToText
    print(f"[formatter] loading {model_id} (flash_attn={use_flash_attn}, max_mem={max_memory_gib})", flush=True)
    t0 = time.perf_counter()
    kwargs = {"dtype": torch.bfloat16, "device_map": device_map}
    if max_memory_gib is not None:
        n = torch.cuda.device_count()
        kwargs["max_memory"] = {i: f"{max_memory_gib}GiB" for i in range(n)}
        kwargs["max_memory"]["cpu"] = "32GiB"
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except (ImportError, ValueError) as e:
        if use_flash_attn:
            print(f"[formatter] flash-attn fallback: {e}", flush=True)
            kwargs.pop("attn_implementation", None)
            model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        else:
            raise
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"[formatter] loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    return model, processor


def build_format_messages(image_path: str, ocr_text: str, ocr_name: str,
                          few_shot: bool = False) -> list:
    """Сборка сообщений для форматтера.

    Если `few_shot=True`, используем mode 'F_fs' с 5 примерами.
    """
    mode = "F_fs" if few_shot else "F"
    prompt = build_prompt(mode, single_ocr_text=ocr_text, single_ocr_name=ocr_name)

    # Конвертируем user_messages с placeholder в HF chat format
    content = []
    for p in prompt["user_messages"]:
        if p["type"] == "text":
            content.append({"type": "text", "text": p["text"]})
        elif p["type"] == "image":
            content.append({"type": "image", "image": p["image"]})
        elif p["type"] == "_target_image_placeholder":
            content.append({"type": "image", "image": image_path})

    return [
        {"role": "system", "content": [{"type": "text", "text": prompt["system"]}]},
        {"role": "user", "content": content},
    ]


@torch.inference_mode()
def format_one(model, processor, image_path: str, ocr_text: str, ocr_name: str,
               max_new_tokens: int = 4096, few_shot: bool = False) -> dict:
    """Возвращает dict с raw_response, parsed_json, parse_ok, runtime."""
    messages = build_format_messages(image_path, ocr_text, ocr_name, few_shot=few_shot)
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

