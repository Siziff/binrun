"""Universal runner for image-to-JSON VLM (mode='A'). Uses model's OWN processor.

Works with: Gemma-4-E4B-it, Gemma-4-12b-it, Qwen3.5-9B-Base, etc.
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
from experiments.exp05_vlm_normalizer.prompts.builder import build_prompt
from experiments.exp05_vlm_normalizer.run_qwen import parse_response

EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "outputs"; OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = REPO_ROOT / "data" / "preprocessed_v3" / "Smuts"
SPLIT_TEST = EXP_DIR / "splits" / "test.jsonl"


def messages_for_image(image_path):
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


def read_jsonl(p):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def already_done(out_path):
    if not out_path.exists():
        return set()
    return {rec["image"] for rec in read_jsonl(out_path)}


def load_model_processor(model_id, custom_processor_id=None, dtype=torch.bfloat16):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc_id = custom_processor_id or model_id
    print(f"[load] processor from {proc_id}", flush=True)
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(proc_id, trust_remote_code=True)
    print(f"[load] processor: {type(processor).__name__} ({time.perf_counter()-t0:.1f}s)", flush=True)

    print(f"[load] model {model_id}", flush=True)
    t1 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=dtype, device_map={"": 0}, trust_remote_code=True,
    )
    print(f"[load] model done ({time.perf_counter()-t1:.1f}s, "
          f"vram={torch.cuda.memory_allocated(0)/1e9:.1f} GB, "
          f"params={sum(p.numel() for p in model.parameters())/1e9:.2f}B)", flush=True)
    return model, processor


@torch.inference_mode()
def generate_one(model, processor, image_path, max_new_tokens, sampling):
    messages = messages_for_image(image_path)
    t0 = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    n_input = int(inputs["input_ids"].shape[-1])
    gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling)
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--processor", default=None,
                    help="Override processor model id (defaults to --model)")
    ap.add_argument("--out-suffix", default="_run1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    slug = args.model.replace("/", "__")
    out_path = OUT_DIR / f"{slug}{args.out_suffix}.jsonl"
    print(f"[run] model={args.model}  processor={args.processor or args.model}", flush=True)
    print(f"[run] out={out_path}", flush=True)

    records = list(read_jsonl(SPLIT_TEST))
    if args.limit:
        records = records[: args.limit]

    skip = already_done(out_path)
    todo = [r for r in records if r["image"] not in skip]
    print(f"[run] todo = {len(todo)} / {len(records)} (resume skip={len(skip)})", flush=True)
    if not todo:
        return

    sampling = ({"do_sample": False} if args.greedy
                else {"do_sample": True, "temperature": 0.6,
                      "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05})
    print(f"[run] sampling={sampling} max_new_tokens={args.max_new_tokens}", flush=True)

    model, processor = load_model_processor(args.model, args.processor)

    t_all = time.perf_counter()
    n_ok = n_fail = 0
    with out_path.open("a", encoding="utf-8") as f_out:
        for i, rec in enumerate(todo, 1):
            image_path = rec.get("image_path") or str(IMG_DIR / rec["image"])
            try:
                result = generate_one(model, processor, image_path,
                                       args.max_new_tokens, sampling)
                err = None
            except Exception as e:
                result = {"raw_response": None, "parsed_json": None,
                          "parse_ok": False,
                          "parse_error": f"runtime: {type(e).__name__}: {e}",
                          "runtime_seconds": 0, "n_input_tokens": 0, "n_output_tokens": 0}
                err = str(e)
            out_rec = {"image": rec["image"], "model": args.model, "mode": "A",
                       **result, "error": err}
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
                print(f"[run] {i}/{len(todo)} ok={n_ok} fail={n_fail} "
                      f"{rate:.1f}s/img ETA {eta/60:.1f}min", flush=True)

    print(f"[run] done ok={n_ok}/{len(todo)} ({n_ok/max(1,len(todo))*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
