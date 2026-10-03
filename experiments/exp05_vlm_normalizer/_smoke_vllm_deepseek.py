"""Quick smoke: можем ли загрузить deepseek-ai/DeepSeek-OCR в vLLM?"""
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.pop("PYTHONSTARTUP", None)

from vllm import LLM, SamplingParams
from PIL import Image

try:
    llm = LLM(
        model="deepseek-ai/DeepSeek-OCR",
        trust_remote_code=True,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        max_model_len=8192,
        dtype="bfloat16",
    )
    print("[load] DeepSeek-OCR vLLM OK")

    image = Image.open("Smuts/LE F-113259.JPG").convert("RGB")
    prompt = "<image>\nFree OCR."
    out = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": image}}],
        SamplingParams(temperature=0, max_tokens=2048, skip_special_tokens=False),
    )
    print("=== OUTPUT ===")
    print(out[0].outputs[0].text[:1500])
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
