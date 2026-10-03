#!/usr/bin/env python
"""Pack LoRA adapters for git: fp32 -> bf16, split adapter_model.safetensors into <95 MB parts.

Usage (on the training host, from project root):
    python experiments/exp06_lora/adapters/pack_adapters.py RUN_ID [RUN_ID ...]

Reads  experiments/exp06_lora/sweep/checkpoints/<RUN_ID>/final/
Writes experiments/exp06_lora/adapters/<RUN_ID>/  (adapter_config.json, README.md,
       adapter_model.safetensors.part-00, part-01, ..., SHA256SUMS)

Reassemble with experiments/exp06_lora/adapters/unpack_adapters.sh
"""
import hashlib
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "experiments/exp06_lora/sweep/checkpoints"
DST = Path(__file__).resolve().parent
PART = 90 * 1024 * 1024  # 90 MiB, under GitHub's 100 MB hard limit
KEEP = ["adapter_config.json", "README.md"]  # tokenizer/processor files == base model, not copied


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(run_id: str) -> None:
    src = SRC / run_id / "final"
    dst = DST / run_id
    dst.mkdir(parents=True, exist_ok=True)
    for name in KEEP:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)

    tensors = load_file(str(src / "adapter_model.safetensors"))
    n_params = sum(t.numel() for t in tensors.values())
    tensors = {k: v.to(torch.bfloat16) for k, v in tensors.items()}
    tmp = dst / "adapter_model.safetensors"
    save_file(tensors, str(tmp), metadata={"format": "pt", "note": "fp32->bf16 for git"})

    full_sha = sha256(tmp)
    size = tmp.stat().st_size
    parts = []
    with open(tmp, "rb") as f:
        i = 0
        while True:
            chunk = f.read(PART)
            if not chunk:
                break
            p = dst / f"adapter_model.safetensors.part-{i:02d}"
            p.write_bytes(chunk)
            parts.append(p)
            i += 1
    tmp.unlink()

    with open(dst / "SHA256SUMS", "w") as f:
        f.write(f"{full_sha}  adapter_model.safetensors\n")
        for p in parts:
            f.write(f"{sha256(p)}  {p.name}\n")
    print(f"{run_id}: {n_params/1e6:.1f}M params, bf16 {size/2**20:.1f} MiB -> {len(parts)} parts")


if __name__ == "__main__":
    for rid in sys.argv[1:]:
        pack(rid)
