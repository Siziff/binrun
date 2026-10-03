"""Eval LoRA adapter on full 394-image corpus (vs 79 test split).

Generates JSONL output with same format as sweep_eval.py — predictions for all 394 images.
Used for final production numbers + per-field analysis.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/exp06_lora/eval_full394.py \\
        --adapter <path/to/adapter> --out <path/to/output.jsonl>
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
GOLD_PATH = EXP_DIR / "splits" / "all394_inputs.jsonl"
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


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


def load_base_with_adapter(adapter_path: Path, use_flash_attn: bool = True):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    print(f"[load] base={BASE_MODEL}", flush=True)
    t0 = time.perf_counter()
    kwargs = {"device_map": "auto", "dtype": torch.bfloat16}
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    try:
        base = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, **kwargs)
    except Exception as e:
        if use_flash_attn:
            print(f"[load] flash-attn failed ({e}); fallback to SDPA", flush=True)
            kwargs.pop("attn_implementation", None)
            base = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, **kwargs)
        else:
            raise
    print(f"[load] base done in {time.perf_counter()-t0:.1f}s", flush=True)

    t1 = time.perf_counter()
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    print(f"[load] adapter done in {time.perf_counter()-t1:.1f}s", flush=True)

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
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
    ap.add_argument("--adapter", required=True, help="Path to LoRA adapter dir")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--inputs", default=None,
                    help="Path to inputs JSONL (records with 'image' + optional 'image_path'). "
                         "Default: merged_inputs.jsonl (394 corpus).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--no-flash-attn", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    adapter = Path(args.adapter).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use merged_inputs (394 imgs, has image_path + gt) unless --inputs given
    inputs_path = Path(args.inputs).resolve() if args.inputs else \
        REPO_ROOT / "experiments/exp05_vlm_normalizer/merged_inputs.jsonl"
    records = list(read_jsonl(inputs_path))
    if args.limit:
        records = records[:args.limit]
    print(f"[run] adapter={adapter}", flush=True)
    print(f"[run] out={out_path}", flush=True)
    print(f"[run] records={len(records)}", flush=True)

    skip = set() if args.no_resume else already_done(out_path)
    todo = [r for r in records if r["image"] not in skip]
    print(f"[run] todo = {len(todo)} / {len(records)} (resume skip={len(skip)})", flush=True)
    if not todo:
        return

    model, processor = load_base_with_adapter(
        adapter, use_flash_attn=not args.no_flash_attn,
    )

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
                "model": f"{BASE_MODEL} + LoRA[{adapter.name}]",
                "mode": "A",
                **result, "error": err,
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
                      f"{rate:.1f}s/img  ETA {eta/60:.1f}min", flush=True)

    print(f"\n[run] done. ok={n_ok}/{len(todo)} "
          f"({n_ok/max(1,len(todo))*100:.1f}%)  out={out_path}", flush=True)


if __name__ == "__main__":
    main()
