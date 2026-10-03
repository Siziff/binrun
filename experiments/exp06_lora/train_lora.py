"""LoRA fine-tune Qwen3-VL-8B-Instruct on Smuts labels (mode A: vision-only).

Train: experiments/exp06_lora/splits/train.jsonl (310 image+gt pairs)
Test:  experiments/exp06_lora/splits/test.jsonl  (79 pairs)

Setup:
  - base: Qwen/Qwen3-VL-8B-Instruct (bfloat16)
  - LoRA: rank=16, alpha=32, dropout=0.05, target=all linear (q,k,v,o,gate,up,down)
  - 3 epochs, lr=2e-5, per_device_batch=1, grad_accum=8 (effective=8 per GPU)
  - Single H100 80GB. Mode A (vision-only, no few-shot in train).

Loss: causal LM on response tokens only (labels = -100 on system+user prompt).

Usage:
    cd /workspace-SR008.fs2/mikheev-kandy/TD
    export TD_PREF=vlm
    source env.sh
    export CUDA_VISIBLE_DEVICES=0
    python3 experiments/exp06_lora/train_lora.py \\
        --epochs 3 \\
        --output-dir experiments/exp06_lora/checkpoints/run1
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.pop("PYTHONSTARTUP", None)
os.environ.pop("PYTHONPATH", None)

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp05_vlm_normalizer.prompts.builder import build_prompt  # noqa: E402


SPLITS_DIR = ROOT / "experiments/exp06_lora/splits"
IMG_DIR = ROOT / "data/preprocessed_v3/Smuts"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _gt_to_target_json(gt: dict) -> str:
    """Detserialize gt to canonical JSON string used as completion target.

    Order of keys is fixed (matches field_schema.json). null stays null.
    """
    # Use deterministic ordering by importing the schema (already loaded by builder).
    from experiments.exp05_vlm_normalizer.prompts.builder import SCHEMA
    ordered = {k: gt.get(k) for k in SCHEMA.keys()}
    return json.dumps(ordered, ensure_ascii=False, indent=2)


class SmutsLoRADataset(Dataset):
    """Returns dicts with raw chat-format messages + image path + target string.

    Tokenization happens in collate_fn (because tokenizer/processor depends on
    model loading).
    """

    def __init__(self, jsonl_path: Path):
        self.records: list[dict] = []
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image_path = str(IMG_DIR / rec["image"])
        target = _gt_to_target_json(rec["gt"])
        return {
            "image": rec["image"],
            "image_path": image_path,
            "target_json": target,
        }


def make_collate_fn(processor, max_seq_len: int = 4096):
    """Build a collate function that converts dataset items to a model batch.

    Each example: system + user (with image + intro+schema text) → assistant (JSON).
    Labels masked everywhere except the assistant response.
    """
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        all_input_ids = []
        all_labels = []
        all_attention_mask = []
        all_pixel_values = []
        all_image_grid_thw = []

        for ex in batch:
            prompt = build_prompt(mode="A")
            system = prompt["system"]
            # Reproduce chat messages exactly as run_fewshot.messages_to_chat does
            # (but mode='A' has no few-shot examples, just intro + target_image).
            content = []
            for p in prompt["user_messages"]:
                if p["type"] == "text":
                    content.append({"type": "text", "text": p["text"]})
                elif p["type"] == "image":
                    content.append({"type": "image", "image": p["image"]})
                elif p["type"] == "_target_image_placeholder":
                    content.append({"type": "image", "image": ex["image_path"]})

            messages = [
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content},
            ]
            # Tokenize the prompt portion only (with generation prompt added).
            prompt_enc = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            # Tokenize the full conversation including assistant response.
            messages_full = messages + [
                {"role": "assistant", "content": [{"type": "text", "text": ex["target_json"]}]},
            ]
            full_enc = processor.apply_chat_template(
                messages_full,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )

            input_ids = full_enc["input_ids"][0]
            # Build labels: copy of input_ids, but mask everything up to
            # prompt_enc length (= response tokens are supervised).
            prompt_len = prompt_enc["input_ids"].shape[-1]
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            # Safety: never truncate past prompt_len (would desync image tokens
            # from pixel_values). If it would happen, raise loud.
            if input_ids.shape[-1] > max_seq_len:
                print(f"[collate] WARN sample {ex['image']}: "
                      f"full_len={input_ids.shape[-1]} > max_seq_len={max_seq_len}; "
                      f"truncating to {max_seq_len}", flush=True)
                if prompt_len >= max_seq_len:
                    raise ValueError(
                        f"Prompt alone is {prompt_len} tokens, exceeds "
                        f"max_seq_len={max_seq_len}. Increase --max-seq-len.")
                input_ids = input_ids[:max_seq_len]
                labels = labels[:max_seq_len]

            attention_mask = torch.ones_like(input_ids)

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_mask.append(attention_mask)

            # Pixel values shape varies per image; keep per-example.
            if "pixel_values" in full_enc:
                all_pixel_values.append(full_enc["pixel_values"])
            if "image_grid_thw" in full_enc:
                all_image_grid_thw.append(full_enc["image_grid_thw"])

        # Pad to the longest in batch.
        max_len = max(x.shape[-1] for x in all_input_ids)
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = processor.tokenizer.eos_token_id

        def _pad(x, val):
            if x.shape[-1] == max_len:
                return x
            pad = torch.full((max_len - x.shape[-1],), val, dtype=x.dtype)
            return torch.cat([x, pad], dim=-1)

        input_ids = torch.stack([_pad(x, pad_id) for x in all_input_ids])
        labels = torch.stack([_pad(x, -100) for x in all_labels])
        attention_mask = torch.stack([_pad(x, 0) for x in all_attention_mask])

        out: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        if all_pixel_values:
            # Qwen3-VL stacks pixel_values along the "image" axis. Each element
            # is already a tensor; concat along dim=0.
            out["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        if all_image_grid_thw:
            out["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)

        return out

    return collate


# ---------------------------------------------------------------------------
# Training loop (no accelerate, single GPU, manual)
# ---------------------------------------------------------------------------

def train(args):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    def log(msg: str):
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"[train] starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"[train] args={vars(args)}")
    log(f"[train] cuda devices: {torch.cuda.device_count()}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # --- model ---
    t0 = time.perf_counter()
    log(f"[load] {args.model_id}  dtype=bfloat16  flash_attn=True")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map={"": 0},  # single GPU
        attn_implementation="flash_attention_2",
    )
    # Enable gradient checkpointing for big seq_len.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False  # required by gradient checkpointing
    log("[load] gradient_checkpointing enabled, use_cache=False")

    # Configure processor with reduced image resolution to fit in VRAM.
    # Defaults to 28*28=784 patches per side base; using min_pixels/max_pixels
    # caps the total image-token budget.
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    log(f"[load] processor: min_pixels={args.min_pixels} max_pixels={args.max_pixels}")
    log(f"[load] done in {time.perf_counter()-t0:.1f}s")

    # --- LoRA ---
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    # Required for gradient checkpointing with PEFT: ensure input embeddings
    # require grad so the gradient can flow back through them.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    # Save print to log
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"[lora] trainable: {n_train/1e6:.1f}M / {n_total/1e9:.2f}B "
        f"({100*n_train/n_total:.3f}%)")

    # --- data ---
    train_ds = SmutsLoRADataset(SPLITS_DIR / "train.jsonl")
    log(f"[data] train_n={len(train_ds)}")

    collate = make_collate_fn(processor, max_seq_len=args.max_seq_len)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.per_device_batch,
        shuffle=True,
        num_workers=2,
        collate_fn=collate,
        drop_last=False,
    )

    # --- optimizer ---
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.0,
        betas=(0.9, 0.999),
    )

    steps_per_epoch = math.ceil(len(train_dl) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.03 * total_steps))
    log(f"[sched] total_steps={total_steps}  warmup={warmup_steps}  "
        f"per_device_batch={args.per_device_batch}  grad_accum={args.grad_accum}  "
        f"effective_batch={args.per_device_batch*args.grad_accum}")

    def get_lr(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        # cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    # --- loop ---
    model.train()
    global_step = 0
    seen_examples = 0
    t_train = time.perf_counter()
    loss_acc = 0.0
    loss_n = 0
    smoke_done = False
    for epoch in range(args.epochs):
        if smoke_done:
            break
        for batch_idx, batch in enumerate(train_dl):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            loss_acc += outputs.loss.item()
            loss_n += 1
            seen_examples += batch["input_ids"].shape[0]
            log(f"[batch] e={epoch+1} b={batch_idx+1}/{len(train_dl)} "
                f"loss={outputs.loss.item():.4f} "
                f"input_ids.shape={tuple(batch['input_ids'].shape)} "
                f"pixel_values.shape={tuple(batch.get('pixel_values', torch.zeros(0)).shape)}")

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_dl):
                # update
                lr = get_lr(global_step)
                for pg in optim.param_groups:
                    pg["lr"] = lr
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0 or global_step == 1:
                    mean_loss = loss_acc / max(1, loss_n)
                    elapsed = time.perf_counter() - t_train
                    log(f"[train] epoch={epoch+1}/{args.epochs} step={global_step}/{total_steps} "
                        f"loss={mean_loss:.4f} lr={lr:.2e} "
                        f"seen={seen_examples} elapsed={elapsed:.1f}s")
                    loss_acc = 0.0
                    loss_n = 0

            if args.smoke_test and (batch_idx + 1) >= args.smoke_test:
                log(f"[smoke] done {batch_idx+1} batches, exiting (no save).")
                smoke_done = True
                break

        if smoke_done:
            break

        # Save checkpoint per epoch.
        ckpt = out_dir / f"epoch{epoch+1}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt))
        log(f"[ckpt] saved {ckpt} after epoch {epoch+1}")

    if smoke_done:
        return

    # Save final adapter + tokenizer.
    final = out_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final))
    processor.save_pretrained(str(final))
    log(f"[done] total elapsed={time.perf_counter()-t_train:.1f}s; final adapter at {final}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-test", type=int, default=0,
                    help="If >0, run only this many batches and exit (no save).")
    # Image resolution caps (Qwen3-VL processor).
    ap.add_argument("--min-pixels", type=int, default=256*28*28,
                    help="Minimum image area in pixels (after resize).")
    ap.add_argument("--max-pixels", type=int, default=1024*28*28,
                    help="Maximum image area in pixels (after resize). "
                         "Smaller = less VRAM/faster. 1024*784=802816 pixels ~= 900x900.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)
