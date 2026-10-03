"""Top-5 best / worst для выбранной тройки движков (Эксп. 2).

Тройка: PaddleOCR + Surya + TrOCR-printed.
Для каждого движка строим отдельные PNG: top-5 best и top-5 worst
по per-image similarity (среди single-файлов).
"""
from __future__ import annotations

from src.utils import env_setup  # noqa: F401

import json
import re
from pathlib import Path
from textwrap import shorten

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "exp02_preprocessed" / "outputs"
FIG = ROOT / "experiments" / "exp02_preprocessed" / "figures"
IMG_DIR = ROOT / "data" / "preprocessed" / "Smuts"
GOLD_PATH = ROOT / "data" / "gold" / "labels.jsonl"

TRIO = ("paddleocr", "surya", "trocr-printed")
SHORT = {"paddleocr": "PaddleOCR", "surya": "Surya", "trocr-printed": "TrOCR-pr"}

GT_FIELDS = (
    "scientific_name", "verbatim_identification", "type_status",
    "verbatim_collector", "verbatim_locality", "country",
    "verbatim_collection_date", "verbatim_substrate",
    "exsiccata_title", "exsiccata_number",
)


def load_gold_by_image() -> dict[str, dict]:
    out = {}
    for line in GOLD_PATH.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        for fname in rec.get("image_files") or []:
            out[fname] = rec
    return out


def load_engine_outputs(eng: str) -> dict[str, dict]:
    """image → {full_text, lines, runtime_seconds, …}"""
    by_img: dict[str, dict] = {}
    p = OUT / f"{eng}.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        doc = json.loads(line)
        by_img[doc["image"]] = doc
    return by_img


def load_engine_metrics(eng: str) -> dict[str, dict]:
    """image → {similarity, fields_score, …}"""
    data = json.load((OUT / f"{eng}_metrics.json").open(encoding="utf-8"))
    return {m["image"]: m for m in data["per_image"]}


def short_text(s: str, n: int = 250) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return shorten(s, width=n, placeholder=" …")


def safe_mpl_text(s: str) -> str:
    """Экранируем символы для matplotlib mathtext."""
    return (s or "").replace("$", "＄").replace("\\", "\\\\")


def render_grid(picks: list[dict], engine: str,
                gold: dict, outputs: dict,
                title: str, out_path: Path) -> None:
    n_rows = len(picks)
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, 4.2 * n_rows),
                             gridspec_kw={"width_ratios": [1.0, 2.4]})
    if n_rows == 1:
        axes = np.array([axes])
    for row_i, pick in enumerate(picks):
        img_name = pick["image"]
        img_path = IMG_DIR / img_name
        ax_img = axes[row_i, 0]
        ax_txt = axes[row_i, 1]

        # Картинка (уже препроцессед-версия).
        try:
            im = Image.open(img_path)
            ax_img.imshow(im)
        except Exception as e:
            ax_img.text(0.5, 0.5, f"(image not loaded: {e})", ha="center", va="center")
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        sim = pick["similarity"]
        runtime = pick.get("runtime", 0.0)
        n_lines = pick.get("n_lines", 0)
        ax_img.set_title(
            f"{img_name}\n"
            f"sim={sim:.3f}  CER={pick['cer']:.2f}  "
            f"runtime={runtime:.2f}s  lines={n_lines}",
            fontsize=10,
        )

        # Правая колонка: GT + OCR-выдача нашего движка.
        rec = gold.get(img_name, {})
        gt_parts = []
        for k in GT_FIELDS:
            v = rec.get(k)
            if v:
                gt_parts.append(f"  {k}: {short_text(str(v), 120)}")
        gt_block = "GROUND TRUTH:\n" + "\n".join(gt_parts) if gt_parts else "GROUND TRUTH: (empty)"

        doc = outputs.get(img_name, {})
        ocr_text = short_text(doc.get("full_text", ""), 700)
        ocr_block = f"\nOCR — {SHORT[engine]}:\n  {ocr_text}"

        # Per-field similarity по нашему движку (топ-5 + bottom-3 для контекста).
        fs = pick.get("fields_score", {})
        fs_present = [(k, v) for k, v in fs.items()
                      if rec.get(k) is not None and str(rec.get(k, "")).strip()]
        fs_present.sort(key=lambda x: -x[1])
        if fs_present:
            top = fs_present[:5]
            bot = fs_present[-3:] if len(fs_present) > 8 else []
            lines = ["\nPER-FIELD SIMILARITY:"]
            for k, v in top:
                lines.append(f"  ✓ {k:<32s} {v:.2f}")
            if bot:
                lines.append("  ...")
                for k, v in bot:
                    lines.append(f"  ✗ {k:<32s} {v:.2f}")
            field_block = "\n".join(lines)
        else:
            field_block = ""

        text_block = gt_block + ocr_block + field_block
        ax_txt.text(0, 1, safe_mpl_text(text_block),
                    family="monospace", fontsize=8.5,
                    va="top", ha="left", wrap=True)
        ax_txt.set_xticks([])
        ax_txt.set_yticks([])
        ax_txt.set_xlim(0, 1)
        ax_txt.set_ylim(0, 1)
        for spine in ax_txt.spines.values():
            spine.set_visible(False)

    fig.suptitle(title, fontsize=14, weight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def pick_best_worst(metrics_by_img: dict[str, dict], gold: dict,
                    n: int = 5) -> tuple[list[dict], list[dict]]:
    """Выбирает best и worst по similarity, исключая _1/_2 (развороты)."""
    candidates = []
    for img, m in metrics_by_img.items():
        if "_1" in img or "_2" in img:
            continue  # развороты — отдельная история, для метрики некорректны
        rec = gold.get(img, {})
        # Нужна хоть какая-то информация в GT.
        if not (rec.get("scientific_name") or rec.get("verbatim_locality") or
                rec.get("verbatim_collector")):
            continue
        candidates.append(m)
    candidates.sort(key=lambda r: -float(r.get("similarity", 0)))
    best = candidates[:n]
    worst = list(reversed(candidates[-n:]))
    return best, worst


def dump_markdown(per_engine: dict[str, dict], gold: dict,
                  outputs_by_eng: dict[str, dict],
                  out_path: Path) -> None:
    lines = ["# Эксп. 2 — top-5 best / worst для тройки PaddleOCR + Surya + TrOCR-pr\n"]
    lines.append(
        "Картинки — препроцессед версии (после orientation + resize).\n"
        "Worst — среди single-файлов (без `_1`/`_2`-разворотов).\n",
    )
    for eng in TRIO:
        best = per_engine[eng]["best"]
        worst = per_engine[eng]["worst"]
        lines.append(f"\n## {SHORT[eng]}\n")
        for title, items in (("Top-5 best", best), ("Top-5 worst", worst)):
            lines.append(f"\n### {title}\n")
            for i, pick in enumerate(items, 1):
                img = pick["image"]
                rec = gold.get(img, {})
                lines.append(f"**{i}. `{img}`**  sim={pick['similarity']:.3f}  "
                             f"CER={pick['cer']:.2f}  "
                             f"runtime={pick.get('runtime', 0):.2f}s\n")
                lines.append("GT:")
                for k in GT_FIELDS:
                    v = rec.get(k)
                    if v:
                        lines.append(f"- `{k}` = {short_text(str(v), 180)}")
                doc = outputs_by_eng[eng].get(img, {})
                lines.append("\nOCR:")
                lines.append(f"  `{short_text(doc.get('full_text', ''), 500)}`\n")
                lines.append("---")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {out_path}")


def main() -> None:
    print("Loading…")
    gold = load_gold_by_image()
    per_engine = {}
    outputs_by_eng = {}
    for eng in TRIO:
        m = load_engine_metrics(eng)
        outs = load_engine_outputs(eng)
        best, worst = pick_best_worst(m, gold, n=5)
        per_engine[eng] = {"best": best, "worst": worst}
        outputs_by_eng[eng] = outs

    FIG.mkdir(parents=True, exist_ok=True)
    for eng in TRIO:
        render_grid(per_engine[eng]["best"], eng, gold, outputs_by_eng[eng],
                    f"{SHORT[eng]} — Top-5 best (Эксп. 2)",
                    FIG / f"top5_best_{eng}.png")
        render_grid(per_engine[eng]["worst"], eng, gold, outputs_by_eng[eng],
                    f"{SHORT[eng]} — Top-5 worst (Эксп. 2, single-файлы)",
                    FIG / f"top5_worst_{eng}.png")

    dump_markdown(per_engine, gold, outputs_by_eng,
                  FIG / "best_worst_trio_summary.md")


if __name__ == "__main__":
    main()
