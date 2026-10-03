"""Quick sanity check of collate_fn pieces."""
import os, sys
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.pop("PYTHONSTARTUP", None); os.environ.pop("PYTHONPATH", None)
from pathlib import Path
sys.path.insert(0, "/workspace-SR008.fs2/mikheev-kandy/TD")

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
from experiments.exp05_vlm_normalizer.prompts.builder import build_prompt

img = "/workspace-SR008.fs2/mikheev-kandy/TD/data/preprocessed_v3/Smuts/LE F-341596.JPG"
prompt = build_prompt(mode="A")
content = []
for p in prompt["user_messages"]:
    if p["type"] == "text":
        content.append({"type": "text", "text": p["text"]})
    elif p["type"] == "image":
        content.append({"type": "image", "image": p["image"]})
    elif p["type"] == "_target_image_placeholder":
        content.append({"type": "image", "image": img})

messages = [
    {"role": "system", "content": [{"type": "text", "text": prompt["system"]}]},
    {"role": "user", "content": content},
]

prompt_enc = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
)
print("prompt_enc keys:", list(prompt_enc.keys()))
for k, v in prompt_enc.items():
    try:
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    except Exception:
        print(f"  {k}: {type(v).__name__}")

target = '{"verbatim_identification": "test"}'
messages_full = messages + [
    {"role": "assistant", "content": [{"type": "text", "text": target}]},
]
full_enc = processor.apply_chat_template(
    messages_full, tokenize=True, add_generation_prompt=False,
    return_dict=True, return_tensors="pt",
)
print("\nfull_enc keys:", list(full_enc.keys()))
for k, v in full_enc.items():
    try:
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    except Exception:
        print(f"  {k}: {type(v).__name__}")

# Count image tokens
import torch
ids = full_enc["input_ids"][0]
# Qwen vision image token id?
img_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
img_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
print(f"\n<|image_pad|> id={img_pad_id}, count in full_enc={int((ids == img_pad_id).sum())}")
print(f"<|vision_start|> id={img_start_id}, count={int((ids == img_start_id).sum())}")
print(f"prompt input_ids length: {prompt_enc['input_ids'].shape[-1]}")
print(f"full   input_ids length: {full_enc['input_ids'].shape[-1]}")

# image_grid_thw → product (multiplied through merge size)
if "image_grid_thw" in full_enc:
    print(f"image_grid_thw: {full_enc['image_grid_thw']}")
    thw = full_enc["image_grid_thw"]
    total_patches = int(thw.prod(dim=-1).sum())
    print(f"  total patches: {total_patches}")
    # Qwen3-VL uses merge_size = 2 typically → divide by 4
    print(f"  /4 (merge_size**2) = {total_patches//4}")
