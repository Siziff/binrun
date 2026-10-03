"""Batched eval merged Qwen3-VL на полных 394 файлах."""
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
from experiments.exp06_lora.eval_merged_batched import (  # noqa
    messages_for_image, load_merged, generate_batch
)
from experiments.exp05_vlm_normalizer.run_qwen import parse_response  # noqa

EXP_DIR = Path(__file__).resolve().parent
IMG_DIR = REPO_ROOT / "data" / "preprocessed_v3" / "Smuts"


def read_jsonl(p: Path):
    for line in p.open("r", encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def already_done(out_path: Path) -> set:
    """Tolerant reader для resume — пропускает broken JSON lines."""
    if not out_path.exists(): return set()
    done = set()
    import re
    text = out_path.read_text(encoding='utf-8', errors='replace')
    # Regex: всё что начинается с {"image":"..."} даже если запись broken
    for m in re.finditer(r'"image":\s*"([^"]+)"', text):
        done.add(m.group(1))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--no-flash-attn", action="store_true")
    args = ap.parse_args()

    model_path = Path(args.model_path).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use merged_inputs.jsonl (394 imgs, has image_path + gt)
    inputs_path = REPO_ROOT / "experiments/exp05_vlm_normalizer/merged_inputs.jsonl"
    records = list(read_jsonl(inputs_path))
    print(f"[run] model={model_path}", flush=True)
    print(f"[run] out={out_path}", flush=True)
    print(f"[run] batch_size={args.batch_size}", flush=True)
    print(f"[run] records={len(records)}", flush=True)

    skip = already_done(out_path)
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
                results, _ = generate_batch(model, processor, image_paths, args.max_new_tokens)
                for rec, res in zip(batch_recs, results):
                    out_rec = {
                        "image": rec["image"],
                        "model": f"merged@{model_path.name}_batched{bs}",
                        "mode": "A",
                        **res, "error": None,
                    }
                    # Защита от broken JSONL: заменим control chars в raw_response
                    if out_rec.get("raw_response"):
                        out_rec["raw_response"] = out_rec["raw_response"].replace('\r', '\\r').replace('\x00', '')
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

            done = min((batch_idx + 1) * bs, len(todo))
            elapsed = time.perf_counter() - t_all
            rate = elapsed / done
            eta = rate * (len(todo) - done)
            print(f"[run] batch={batch_idx+1}/{total_batches}  done={done}/{len(todo)}  "
                  f"ok={n_ok}  fail={n_fail}  {rate:.2f}s/img  ETA {eta/60:.1f}min", flush=True)

    print(f"\n[run] done. ok={n_ok}/{len(todo)} ({n_ok/max(1,len(todo))*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
