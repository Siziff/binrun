"""Compare LoRA-finetuned model vs baseline A_fs on the 79-image test split.

For each model JSONL:
  1. Read all records.
  2. Keep only records whose `image` is in `splits/test.jsonl`.
  3. Run unified_eval.eval_vlm_jsonl on the filtered slice.

Writes:
  experiments/exp06_lora/comparison_test.{csv,md}
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.utils import env_setup  # noqa: F401

from unified_eval import eval_vlm_jsonl, load_gold, read_jsonl  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
SPLIT_TEST = EXP_DIR / "splits" / "test.jsonl"


def load_test_images() -> set[str]:
    imgs = set()
    with SPLIT_TEST.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                imgs.add(json.loads(line)["image"])
    return imgs


def slice_jsonl_to_test(src: Path, test_imgs: set[str]) -> Path:
    """Filter src JSONL to test images only, return path to a temp file."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".jsonl", delete=False,
    )
    n_keep = 0
    n_skip = 0
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("image") in test_imgs:
                tmp.write(line + "\n")
                n_keep += 1
            else:
                n_skip += 1
    tmp.close()
    print(f"  [slice] kept {n_keep}/{n_keep+n_skip} records from {src.name}")
    return Path(tmp.name)


def main():
    test_imgs = load_test_images()
    print(f"[run] test split: {len(test_imgs)} images")

    gold = load_gold()
    print(f"[run] gold: {len(gold)} images")

    OUTDIR = REPO_ROOT / "experiments/exp05_vlm_normalizer/outputs"
    runs = [
        ("baseline_A_fs",
         OUTDIR / "Qwen__Qwen3-VL-8B-Instruct__modeA_fs.jsonl"),
        ("baseline_A_zero",
         OUTDIR / "Qwen__Qwen3-VL-8B-Instruct__modeA.jsonl"),
        ("lora_run1_final",
         EXP_DIR / "outputs/lora_run1_final.jsonl"),
        ("qwythos_9b_run1",
         EXP_DIR / "outputs/empero-ai__Qwythos-9B-Claude-Mythos-5-1M_qwythos_run1.jsonl"),
        ("qwen35_9b_base_run1",
         EXP_DIR / "outputs/Qwen__Qwen3.5-9B-Base_qwen35base_run1_fixed.jsonl"),
        ("gemma4_E4B_it_modeA",
         EXP_DIR / "outputs/google__gemma-4-E4B-it_gemma4e_run2.jsonl"),
        ("gemma4_12b_it_modeA",
         EXP_DIR / "outputs/google__gemma-4-12b-it_gemma412_run1.jsonl"),
        ("gemma4_E4B_it_A_fs",
         OUTDIR / "google__gemma-4-E4B-it__modeA_fs_gemma4e_fs_test79.jsonl"),
        # OCR-VLM → Qwen3-VL A_fs formatter
        ("chandra-ocr-2_fmt_fs",
         OUTDIR / "chandra-ocr-2-simple__formatted_fs_test79.jsonl"),
        ("lift_fmt_fs",
         OUTDIR / "lift__formatted_fs_test79.jsonl"),
        ("unlimited-ocr_fmt_fs",
         OUTDIR / "unlimited-ocr__formatted_fs_test79.jsonl"),
        ("surya-ocr-2_fmt_fs",
         OUTDIR / "surya-ocr-2__formatted_fs_test79.jsonl"),
        ("paddleocr-vl-1.6_fmt_fs",
         OUTDIR / "paddleocr-vl-1.6__formatted_fs_test79.jsonl"),
    ]

    rows = []
    for name, path in runs:
        if not path.exists():
            print(f"[skip] {name}: {path} missing")
            continue
        print(f"\n[eval] {name}: {path.name}")
        tmp = slice_jsonl_to_test(path, test_imgs)
        try:
            metrics = eval_vlm_jsonl(tmp, gold)
        finally:
            tmp.unlink(missing_ok=True)
        row = {"run": name, **metrics}
        rows.append(row)
        print(f"  n={metrics['n_records']}  parse_rate={metrics['parse_rate']}  "
              f"F1={metrics['F1']}  F1_macro={metrics['F1_macro']}  "
              f"sim={metrics['sim']}  CER={metrics['CER']}  "
              f"sec/img={metrics['sec_per_img']}")

    if not rows:
        print("[run] no rows to write")
        return

    csv_out = EXP_DIR / "comparison_test.csv"
    md_out = EXP_DIR / "comparison_test.md"

    cols = ["run", "n_records", "parse_rate",
            "CER", "WER", "sim",
            "hard_R", "soft_R",
            "F1_micro", "F1_macro",
            "sec_per_img"]

    with csv_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[write] {csv_out}")

    md = ["# LoRA vs baseline on 79-image test split\n"]
    md.append("| run | n | parse | CER | sim | F1_micro | F1_macro | sec/img |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['run']} | {r['n_records']} | "
            f"{r['parse_rate']} | {r['CER']} | {r['sim']} | "
            f"{r['F1_micro']} | {r['F1_macro']} | {r['sec_per_img']} |")
    md_out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[write] {md_out}")


if __name__ == "__main__":
    main()
