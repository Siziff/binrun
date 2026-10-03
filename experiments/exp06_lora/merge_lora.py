"""Merge LoRA adapter into base Qwen3-VL-8B-Instruct weights.

Loads base + adapter via peft, merges, saves as a standard HF model.
Output directory contains full model weights (no peft dependency required for inference).

Usage:
    python experiments/exp06_lora/merge_lora.py \\
        --adapter experiments/exp06_lora/sweep/checkpoints/p5j_r32_lr7e5_ep5_a256/final \\
        --output data/models/production/qwen3vl_lora_p5j_merged
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("HF_HOME", str(REPO_ROOT / "data" / "models" / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(REPO_ROOT / "data" / "models" / "hf_cache"))

import torch
from src.utils import env_setup  # noqa

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Path to LoRA adapter dir")
    ap.add_argument("--output", required=True, help="Output dir for merged model")
    ap.add_argument("--no-flash-attn", action="store_true")
    args = ap.parse_args()

    adapter = Path(args.adapter).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"[merge] adapter: {adapter}", flush=True)
    print(f"[merge] output:  {output}", flush=True)

    if not adapter.exists():
        raise FileNotFoundError(adapter)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    print(f"[merge] loading base {BASE_MODEL}", flush=True)
    t0 = time.perf_counter()
    kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    if not args.no_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    try:
        base = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, **kwargs)
    except Exception as e:
        if "attn_implementation" in kwargs:
            print(f"[merge] flash_attn failed: {e}, fallback to sdpa", flush=True)
            kwargs.pop("attn_implementation", None)
            base = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, **kwargs)
        else:
            raise
    print(f"[merge] base loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"[merge] loading adapter", flush=True)
    t1 = time.perf_counter()
    model = PeftModel.from_pretrained(base, str(adapter))
    print(f"[merge] adapter loaded in {time.perf_counter()-t1:.1f}s", flush=True)

    print(f"[merge] merging adapter into base weights (merge_and_unload)", flush=True)
    t2 = time.perf_counter()
    merged = model.merge_and_unload()
    print(f"[merge] merged in {time.perf_counter()-t2:.1f}s", flush=True)

    print(f"[merge] saving merged model to {output}", flush=True)
    t3 = time.perf_counter()
    merged.save_pretrained(str(output), safe_serialization=True)
    # Also save processor for convenience
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    processor.save_pretrained(str(output))
    print(f"[merge] saved in {time.perf_counter()-t3:.1f}s", flush=True)

    # Report sizes
    total_size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    print(f"[merge] total output size: {total_size/1e9:.2f} GB", flush=True)
    print(f"[merge] DONE", flush=True)


if __name__ == "__main__":
    main()
