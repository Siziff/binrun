"""Manual batching eval для merged Qwen3-VL модели.

Параллельно обрабатывает N изображений в одном forward pass через model.generate().
Padding на input_ids + attention_mask + concat pixel_values.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/exp06_lora/eval_merged_batched.py \\
        --model-path data/models/production/qwen3vl_lora_p5j_merged \\
        --out outputs/lora_p5j_merged_batched_test79.jsonl \\
        --batch-size 4
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
    print(f"[load] model done in {time.perf_counter()-t0:.1f}s", flush=True)

    processor = AutoProcessor.from_pretrained(str(model_path))
    # Tokenizer должен padding выровнить слева для generate()
    processor.tokenizer.padding_side = "left"
    return model, processor


@torch.inference_mode()
def generate_batch(model, processor, image_paths: list[str], max_new_tokens: int = 2048) -> list[dict]:
    """Inference на batch изображений. Возвращает list of dict (по одному на каждое изображение)."""
    # Подготовим все messages
    batch_messages = [messages_for_image(p) for p in image_paths]

    t0 = time.perf_counter()
    # Process all images вместе
    inputs = processor.apply_chat_template(
        batch_messages,
        tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
        padding=True,
    ).to(model.device)
    n_input_per = inputs["input_ids"].shape[-1]

    # Run batched generate
    gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=processor.tokenizer.pad_token_id)

    # Декодируем каждый sample
    batch_size = inputs["input_ids"].shape[0]
    gen_trim = gen_ids[:, n_input_per:]
    texts = processor.batch_decode(gen_trim, skip_special_tokens=True)
    runtime = time.perf_counter() - t0
    per_img_runtime = runtime / batch_size

    results = []
    for i, text in enumerate(texts):
        n_output = (gen_trim[i] != processor.tokenizer.pad_token_id).sum().item()
        parsed, perr = parse_response(text)
        results.append({
            "raw_response": text,
            "parsed_json": parsed,
            "parse_ok": parsed is not None,
            "parse_error": perr,
            "runtime_seconds": per_img_runtime,
            "n_input_tokens": int(n_input_per),
            "n_output_tokens": int(n_output),
        })
    return results, runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4)
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
    print(f"[run] batch_size={args.batch_size}", flush=True)

    skip = set() if args.no_resume else already_done(out_path)
    todo = [r for r in records if r["image"] not in skip]
    print(f"[run] todo = {len(todo)} / {len(records)} (resume skip={len(skip)})", flush=True)
    if not todo:
        return

    model, processor = load_merged(model_path, use_flash_attn=not args.no_flash_attn)

    bs = args.batch_size
    t_all = time.perf_counter()
    n_ok = n_fail = 0
    total_batches = (len(todo) + bs - 1) // bs

    with out_path.open("a", encoding="utf-8") as f_out:
        for batch_idx in range(total_batches):
            batch_recs = todo[batch_idx*bs : (batch_idx+1)*bs]
            image_paths = [rec.get("image_path") or str(IMG_DIR / rec["image"]) for rec in batch_recs]

            try:
                results, batch_time = generate_batch(model, processor, image_paths, args.max_new_tokens)
                for rec, res in zip(batch_recs, results):
                    out_rec = {
                        "image": rec["image"],
                        "model": f"merged@{model_path.name}_batched{bs}",
                        "mode": "A",
                        **res, "error": None,
                    }
                    f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    if res["parse_ok"]:
                        n_ok += 1
                    else:
                        n_fail += 1
                f_out.flush()
            except Exception as e:
                for rec in batch_recs:
                    out_rec = {
                        "image": rec["image"],
                        "model": f"merged@{model_path.name}_batched{bs}",
                        "mode": "A",
                        "raw_response": None, "parsed_json": None,
                        "parse_ok": False, "parse_error": f"batch runtime: {type(e).__name__}: {e}",
                        "runtime_seconds": 0, "n_input_tokens": 0, "n_output_tokens": 0,
                        "error": str(e),
                    }
                    f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    n_fail += 1
                f_out.flush()

            done = (batch_idx + 1) * bs
            done = min(done, len(todo))
            elapsed = time.perf_counter() - t_all
            rate = elapsed / done
            eta = rate * (len(todo) - done)
            print(f"[run] batch={batch_idx+1}/{total_batches}  done={done}/{len(todo)}  "
                  f"ok={n_ok}  fail={n_fail}  {rate:.2f}s/img  ETA {eta/60:.1f}min", flush=True)

    print(f"\n[run] done. ok={n_ok}/{len(todo)} ({n_ok/max(1,len(todo))*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
