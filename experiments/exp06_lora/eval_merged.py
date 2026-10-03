"""Eval merged Qwen3-VL (LoRA уже merged в base weights) на 79 test split.

Не использует peft — модель загружается как обычная Qwen3-VL-8B-Instruct.
Эталон для сравнения с adapter-based eval (sweep_eval.py).

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/exp06_lora/eval_merged.py \\
        --model-path data/models/production/qwen3vl_lora_p5j_merged \\
        --out outputs/lora_p5j_merged_eval.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("HF_HOME", str(REPO_ROOT / "data" / "models" / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(REPO_ROOT / "data" / "models" / "hf_cache"))

import torch
from src.utils import env_setup  # noqa
from experiments.exp05_vlm_normalizer.prompts.builder import build_prompt  # noqa
from experiments.exp05_vlm_normalizer.run_qwen import parse_response  # noqa

EXP_DIR = Path(__file__).resolve().parent
IMG_DIR = REPO_ROOT / "data" / "preprocessed_v3" / "Smuts"
SPLIT_TEST = EXP_DIR / "splits" / "test.jsonl"


def messages_for_image(image_path: str) -> list[dict]:
    prompt = build_prompt(mode="A")
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


def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def already_done(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    return {rec["image"] for rec in read_jsonl(out_path)}


def load_merged(model_path: Path, use_flash_attn: bool = True):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[load] merged model from {model_path}", flush=True)
    t0 = time.perf_counter()
    kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    try:
        model = AutoModelForImageTextToText.from_pretrained(str(model_path), **kwargs)
    except Exception as e:
        if use_flash_attn:
            print(f"[load] flash_attn failed: {e}, fallback to sdpa", flush=True)
            kwargs.pop("attn_implementation", None)
            model = AutoModelForImageTextToText.from_pretrained(str(model_path), **kwargs)
        else:
            raise
    model.eval()
    print(f"[load] model done in {time.perf_counter()-t0:.1f}s "
          f"(NO peft overhead — merged)", flush=True)

    processor = AutoProcessor.from_pretrained(str(model_path))
    return model, processor


@torch.inference_mode()
def generate_one(model, processor, image_path: str, max_new_tokens: int = 2048) -> dict:
    messages = messages_for_image(image_path)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="Path to merged model dir")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--no-flash-attn", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    model_path = Path(args.model_path).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = list(read_jsonl(SPLIT_TEST))
    if args.limit:
        records = records[: args.limit]
    print(f"[run] model={model_path}", flush=True)
    print(f"[run] out={out_path}", flush=True)

    skip = set() if args.no_resume else already_done(out_path)
    todo = [r for r in records if r["image"] not in skip]
    print(f"[run] todo = {len(todo)} / {len(records)} (resume skip={len(skip)})", flush=True)
    if not todo:
        return

    model, processor = load_merged(model_path, use_flash_attn=not args.no_flash_attn)

    t_all = time.perf_counter()
    n_ok = n_fail = 0
    with out_path.open("a", encoding="utf-8") as f_out:
        for i, rec in enumerate(todo, 1):
            image_path = rec.get("image_path") or str(IMG_DIR / rec["image"])
            try:
                result = generate_one(model, processor, image_path, args.max_new_tokens)
                err = None
            except Exception as e:
                result = {
                    "raw_response": None, "parsed_json": None,
                    "parse_ok": False, "parse_error": f"runtime: {type(e).__name__}: {e}",
                    "runtime_seconds": 0, "n_input_tokens": 0, "n_output_tokens": 0,
                }
                err = str(e)
            out_rec = {
                "image": rec["image"],
                "model": f"merged@{model_path.name}",
                "mode": "A",
                **result, "error": err,
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

    print(f"\n[run] done. ok={n_ok}/{len(todo)} "
          f"({n_ok/max(1,len(todo))*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
