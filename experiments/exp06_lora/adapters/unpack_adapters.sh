#!/usr/bin/env bash
# Reassemble LoRA adapters split by pack_adapters.py.
# Usage: bash experiments/exp06_lora/adapters/unpack_adapters.sh [RUN_ID ...]
# Without args — unpacks every adapter directory here.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
runs=("$@")
if [ ${#runs[@]} -eq 0 ]; then
    mapfile -t runs < <(find . -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
fi
for r in "${runs[@]}"; do
    [ -f "$r/adapter_config.json" ] || continue
    if [ -f "$r/adapter_model.safetensors" ]; then
        echo "[skip] $r already assembled"; continue
    fi
    cat "$r"/adapter_model.safetensors.part-* > "$r/adapter_model.safetensors"
    (cd "$r" && sha256sum -c --quiet SHA256SUMS) && echo "[ok]   $r -> $r/adapter_model.safetensors"
done
echo "Load with: PeftModel.from_pretrained(base, 'experiments/exp06_lora/adapters/<RUN_ID>')"
