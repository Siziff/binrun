"""Top-5 best / worst примеры из Эксп. 1.

Стратегия:
    * для каждого изображения берём **среднюю similarity по всем 7 движкам**
      как интегральный показатель «лёгкости» образца;
    * top-5 best — где средняя similarity максимальна;
    * top-5 worst — где минимальна (но gt_has хоть что-то, чтобы это были не
      пустые записи).

Артефакты:
    * `figures/best_worst_grid.png` — большая фигура с 10 примерами:
      картинка слева, текст GT/OCR разных движков справа.
    * `figures/best_worst_summary.md` — текстовый дамп для чтения.
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
OUT_DIR = ROOT / "experiments" / "exp01_ocr_baseline" / "outputs"
FIG_DIR = ROOT / "experiments" / "exp01_ocr_baseline" / "figures"
IMG_DIR = ROOT / "Smuts"
GOLD_PATH = ROOT / "data" / "gold" / "labels.jsonl"

ENGINES = ("tesseract", "paddleocr", "easyocr", "doctr", "surya",
           "trocr-printed", "trocr-handwritten")
SHORT = {
    "tesseract": "Tesseract", "paddleocr": "PaddleOCR", "easyocr": "EasyOCR",
    "doctr": "docTR", "surya": "Surya",
    "trocr-printed": "TrOCR-pr", "trocr-handwritten": "TrOCR-hw",
}


def load_gold_by_image() -> dict[str, dict]:
    out = {}
    for line in GOLD_PATH.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        for fname in rec.get("image_files") or []:
            out[fname] = rec
    return out


def load_engine_outputs() -> dict[str, dict[str, dict]]:
    """engine → image → {full_text, lines, …}"""
    by_eng = {}
    for eng in ENGINES:
        p = OUT_DIR / f"{eng}.jsonl"
        if not p.exists():
            continue
        by_img = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            doc = json.loads(line)
            by_img[doc["image"]] = doc
        by_eng[eng] = by_img
    return by_eng


def load_per_image_metrics() -> dict[str, dict[str, dict]]:
    """engine → image → {similarity, cer, fields_score, …}"""
    by_eng = {}
    for eng in ENGINES:
        p = OUT_DIR / f"{eng}_metrics.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        by_img = {m["image"]: m for m in data["per_image"]}
        by_eng[eng] = by_img
    return by_eng


def build_ranking(metrics_by_eng: dict[str, dict[str, dict]]) -> list[dict]:
    """Считает для каждой картинки средние метрики по 7 движкам."""
    all_images = sorted({
        img for by_img in metrics_by_eng.values() for img in by_img.keys()
    })
    rank = []
    for img in all_images:
        sims, cers = [], []
        per_eng_sim = {}
        for eng, by_img in metrics_by_eng.items():
            m = by_img.get(img)
            if not m or m.get("similarity") is None:
                continue
            sims.append(float(m["similarity"]))
            cers.append(float(m["cer"]))
            per_eng_sim[eng] = float(m["similarity"])
        if not sims:
            continue
        rank.append({
            "image": img,
            "mean_sim": float(np.mean(sims)),
            "std_sim": float(np.std(sims)),
            "mean_cer": float(np.mean(cers)),
            "min_sim": min(sims),
            "max_sim": max(sims),
            "n_engines": len(sims),
            "per_eng_sim": per_eng_sim,
        })
    return rank


def short_text(s: str, n: int = 200) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return shorten(s, width=n, placeholder=" …")


def safe_mpl_text(s: str) -> str:
    """Экранирует символы, которые matplotlib интерпретирует как mathtext."""
    return (s or "").replace("$", "＄").replace("\\", "\\\\")


def render_grid(picks: list[dict],
                gold: dict, outputs: dict,
                title: str, out_path: Path) -> None:
    """Рисует фигуру: 5 строк × (картинка | текстовая колонка)."""
    n_rows = len(picks)
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, 4.0 * n_rows),
                             gridspec_kw={"width_ratios": [1.0, 2.4]})
    if n_rows == 1:
        axes = np.array([axes])
    for row_i, pick in enumerate(picks):
        img_name = pick["image"]
        img_path = IMG_DIR / img_name
        ax_img = axes[row_i, 0]
        ax_txt = axes[row_i, 1]

        # картинка
        try:
            im = Image.open(img_path)
            ax_img.imshow(im)
        except Exception:
            ax_img.text(0.5, 0.5, "(image not loaded)", ha="center", va="center")
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title(
            f"{img_name}\n"
            f"mean sim={pick['mean_sim']:.2f}  "
            f"std={pick['std_sim']:.2f}  "
            f"min={pick['min_sim']:.2f}  max={pick['max_sim']:.2f}",
            fontsize=10,
        )

        # текстовая колонка: GT verbatim+normalized + 3 лучших движка
        rec = gold.get(img_name, {})
        gt_parts = []
        for k in ("scientific_name", "verbatim_identification",
                  "type_status", "verbatim_collector",
                  "verbatim_locality", "country",
                  "verbatim_collection_date", "verbatim_substrate",
                  "exsiccata_title", "exsiccata_number"):
            v = rec.get(k)
            if v:
                gt_parts.append(f"  {k}: {short_text(str(v), 110)}")
        gt_block = "GROUND TRUTH:\n" + "\n".join(gt_parts) if gt_parts else "GROUND TRUTH: (empty)"

        # сортируем движки по similarity для данной картинки (по убыванию)
        sorted_engines = sorted(
            pick["per_eng_sim"].items(), key=lambda kv: -kv[1]
        )
        # покажем top-2 и worst-1
        show_engines = sorted_engines[:2] + [sorted_engines[-1]]
        ocr_blocks = []
        for eng, sim in show_engines:
            doc = outputs.get(eng, {}).get(img_name, {})
            txt = short_text(doc.get("full_text", ""), 380)
            ocr_blocks.append(f"[{SHORT[eng]:<10}] sim={sim:.2f}  "
                              f"runtime={doc.get('runtime_seconds', 0):.2f}s\n  {txt}")
        text_block = gt_block + "\n\n" + "\n".join(ocr_blocks)

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


def dump_markdown(best: list[dict], worst: list[dict],
                  gold: dict, outputs: dict, out_path: Path) -> None:
    lines = ["# Эксперимент 1 — top-5 best / worst примеры\n"]

    def section(title, picks):
        lines.append(f"\n## {title}\n")
        for i, pick in enumerate(picks, 1):
            img = pick["image"]
            rec = gold.get(img, {})
            lines.append(f"### {i}. `{img}`  (mean sim = {pick['mean_sim']:.3f}, "
                         f"std = {pick['std_sim']:.3f}, "
                         f"min = {pick['min_sim']:.3f}, max = {pick['max_sim']:.3f})\n")
            lines.append("**Ground truth (выборка полей):**")
            for k in ("scientific_name", "verbatim_identification",
                      "type_status", "verbatim_collector",
                      "verbatim_locality", "country",
                      "verbatim_collection_date", "verbatim_substrate",
                      "exsiccata_title", "exsiccata_number"):
                v = rec.get(k)
                if v:
                    lines.append(f"- `{k}` = {short_text(str(v), 200)}")
            lines.append("\n**Per-engine similarity:**\n")
            lines.append("| engine | sim |")
            lines.append("|---|---:|")
            for eng in ENGINES:
                if eng in pick["per_eng_sim"]:
                    lines.append(f"| {SHORT[eng]} | {pick['per_eng_sim'][eng]:.3f} |")
            lines.append("")
            lines.append("**OCR-выдача (top + worst движков):**\n")
            sorted_engines = sorted(pick["per_eng_sim"].items(), key=lambda kv: -kv[1])
            shown = sorted_engines[:2] + [sorted_engines[-1]]
            for eng, sim in shown:
                doc = outputs.get(eng, {}).get(img, {})
                t = short_text(doc.get("full_text", ""), 400)
                lines.append(f"- **{SHORT[eng]}** (sim={sim:.2f}): `{t}`")
            lines.append("\n---")
    section("Top-5 best (самые «лёгкие» — все движки справились)", best)
    section("Top-5 worst (самые «трудные» — никто не справился)", worst)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {out_path}")


def main() -> None:
    print("Loading data…")
    gold = load_gold_by_image()
    outputs = load_engine_outputs()
    metrics = load_per_image_metrics()
    rank = build_ranking(metrics)
    print(f"  ranked {len(rank)} images")

    # Top-5 best: mean_sim максимальная.
    best = sorted(rank, key=lambda r: -r["mean_sim"])[:5]
    # Top-5 worst среди single-файлов (без _1/_2):
    # развороты гербарных листов имеют картинку без этикетки → метрика
    # сравнивает OCR полной страницы с этикеточным GT (некорректное сравнение).
    # Worst среди single-файлов реально показывает, какие этикетки тяжело
    # распознавать всем 7 движкам.
    single_only = [
        r for r in rank
        if "_1" not in r["image"] and "_2" not in r["image"] and
           (gold.get(r["image"], {}).get("verbatim_locality") or
            gold.get(r["image"], {}).get("verbatim_collector") or
            gold.get(r["image"], {}).get("scientific_name"))
    ]
    worst = sorted(single_only, key=lambda r: r["mean_sim"])[:5]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    render_grid(best, gold, outputs,
                "Top-5 best — самые «лёгкие» этикетки (все движки сработали)",
                FIG_DIR / "top5_best.png")
    render_grid(worst, gold, outputs,
                "Top-5 worst — самые «трудные» этикетки",
                FIG_DIR / "top5_worst.png")
    dump_markdown(best, worst, gold, outputs,
                  FIG_DIR / "best_worst_summary.md")


if __name__ == "__main__":
    main()
