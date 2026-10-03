"""Aggregate sweep results across all runs into a single summary table.

Scans `experiments/exp06_lora/sweep/eval_test79/*.jsonl`,
runs `unified_eval.eval_vlm_jsonl` on each (restricted to 79 test split),
and writes:
    experiments/exp06_lora/sweep/sweep_summary.csv
    experiments/exp06_lora/sweep/sweep_summary.md
"""
from __future__ import annotations
import csv, json, sys, tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.utils import env_setup  # noqa
from unified_eval import eval_vlm_jsonl, load_gold  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
SWEEP_DIR = EXP_DIR / "sweep"
EVAL_DIR = SWEEP_DIR / "eval_test79"
CFG_DIR = SWEEP_DIR / "configs"
SPLIT_TEST = EXP_DIR / "splits" / "test.jsonl"


def load_test_imgs():
    return {json.loads(l)["image"] for l in SPLIT_TEST.open()}


def slice_to_test(src: Path, test_imgs: set):
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".jsonl", delete=False)
    kept = 0
    for line in src.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("image") in test_imgs:
            tmp.write(line + "\n")
            kept += 1
    tmp.close()
    return Path(tmp.name), kept


def load_config(run_id: str) -> dict:
    cfg_path = CFG_DIR / f"{run_id}.json"
    if cfg_path.exists():
        return json.load(cfg_path.open())
    return {}


def main():
    if not EVAL_DIR.exists():
        print(f"No eval dir: {EVAL_DIR}")
        return

    test_imgs = load_test_imgs()
    gold = load_gold()
    print(f"[summary] test={len(test_imgs)} images, gold={len(gold)} images")

    eval_files = sorted(EVAL_DIR.glob("*.jsonl"))
    if not eval_files:
        print(f"[summary] no eval files in {EVAL_DIR}")
        return

    # Include baseline run1 if present
    baseline_path = EXP_DIR / "outputs" / "lora_run1_final.jsonl"
    if baseline_path.exists():
        # virtual entry
        cfg_run1 = {"run_id": "baseline_run1", "rank": 16, "lr": 2e-5,
                    "epochs": 3, "dropout": 0.05, "target_modules": "all-linear",
                    "grad_accum": 8, "warmup_ratio": 0.03}
    else:
        cfg_run1 = None

    rows = []

    def evaluate(path: Path, run_id: str, cfg: dict):
        tmp, n = slice_to_test(path, test_imgs)
        try:
            metrics = eval_vlm_jsonl(tmp, gold)
        finally:
            tmp.unlink(missing_ok=True)
        print(f"  {run_id}: n={metrics['n_records']} F1={metrics['F1']} "
              f"parse={metrics['parse_rate']} CER={metrics['CER']} "
              f"sec/img={metrics['sec_per_img']}")
        return {"run_id": run_id, **cfg, **metrics}

    if cfg_run1 is not None:
        print(f"[eval] baseline_run1")
        rows.append(evaluate(baseline_path, "baseline_run1", cfg_run1))

    for ef in eval_files:
        run_id = ef.stem
        cfg = load_config(run_id)
        print(f"[eval] {run_id}")
        try:
            rows.append(evaluate(ef, run_id, cfg))
        except Exception as e:
            print(f"  FAILED: {e}")
            rows.append({"run_id": run_id, "error": str(e)})

    if not rows:
        return

    rows.sort(key=lambda r: -float(r.get("F1") or 0))

    cols = ["run_id", "rank", "lr", "epochs", "dropout", "target_modules",
            "grad_accum", "warmup_ratio",
            "n_records", "parse_rate", "CER", "WER", "sim",
            "hard_R", "soft_R", "F1_micro", "F1_macro", "sec_per_img"]

    csv_out = SWEEP_DIR / "sweep_summary.csv"
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[write] {csv_out}")

    md = ["# LoRA Sweep Results — ranked by F1\n"]
    md.append("| rank | run_id | r | lr | ep | drop | TM | ga | F1 | F1m | CER | sim | sec/img |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        f1 = r.get("F1") or r.get("F1_micro") or "—"
        md.append(
            f"| {i} | {r['run_id']} | {r.get('rank','—')} | {r.get('lr','—')} | "
            f"{r.get('epochs','—')} | {r.get('dropout','—')} | "
            f"{r.get('target_modules','—')} | {r.get('grad_accum','—')} | "
            f"{f1} | {r.get('F1_macro','—')} | {r.get('CER','—')} | "
            f"{r.get('sim','—')} | {r.get('sec_per_img','—')} |")
    md_out = SWEEP_DIR / "sweep_summary.md"
    md_out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[write] {md_out}")


if __name__ == "__main__":
    main()
