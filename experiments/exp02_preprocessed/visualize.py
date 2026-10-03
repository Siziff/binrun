"""Визуализации для Эксп. 2 + сравнительный график Эксп. 1 vs 2."""
from __future__ import annotations

from src.utils import env_setup  # noqa: F401

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
OUT1 = ROOT / "experiments" / "exp01_ocr_baseline" / "outputs"
OUT2 = ROOT / "experiments" / "exp02_preprocessed" / "outputs"
FIG = ROOT / "experiments" / "exp02_preprocessed" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ENGINES = ("tesseract", "paddleocr", "easyocr", "doctr", "surya",
           "trocr-printed", "trocr-handwritten")
SHORT = {"tesseract":"Tesseract","paddleocr":"PaddleOCR","easyocr":"EasyOCR",
         "doctr":"docTR","surya":"Surya",
         "trocr-printed":"TrOCR-pr","trocr-handwritten":"TrOCR-hw"}
DEVICE = {"tesseract":"CPU","paddleocr":"CPU","easyocr":"GPU","doctr":"GPU",
          "surya":"GPU","trocr-printed":"GPU","trocr-handwritten":"GPU"}


def load_aggregate(out_dir: Path, eng: str) -> dict:
    return json.load((out_dir / f"{eng}_metrics.json").open(encoding="utf-8"))["aggregate"]


def plot_exp1_vs_exp2():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    metrics = [("f1.micro.f1", "micro-F1 ↑", 0.0),
               ("mean_similarity", "mean similarity ↑", 0.0),
               ("mean_runtime", "sec / image  ↓ (log scale)", None)]

    def get(d, dotted):
        for p in dotted.split("."):
            d = d[p]
        return d

    palette = sns.color_palette("Set2", n_colors=len(ENGINES))
    x = np.arange(len(ENGINES))
    w = 0.35

    for ax, (m, title, ymin) in zip(axes, metrics):
        v1 = [get(load_aggregate(OUT1, e), m) for e in ENGINES]
        v2 = [get(load_aggregate(OUT2, e), m) for e in ENGINES]
        b1 = ax.bar(x - w/2, v1, w, color=palette, alpha=0.55, edgecolor="#333",
                    linewidth=0.6, label="Эксп. 1 (raw)")
        b2 = ax.bar(x + w/2, v2, w, color=palette, alpha=1.0, edgecolor="#000",
                    linewidth=0.8, label="Эксп. 2 (preprocessed)")
        if "runtime" in m:
            ax.set_yscale("log")
        else:
            ax.set_ylim(ymin, max(max(v1), max(v2)) * 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[e] for e in ENGINES], rotation=25)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(loc="best", fontsize=9)
        # значения над столбцами
        for bars, vals in [(b1, v1), (b2, v2)]:
            for b, val in zip(bars, vals):
                if "runtime" in m:
                    label = f"{val:.1f}"
                else:
                    label = f"{val:.2f}"
                ax.text(b.get_x() + b.get_width()/2, val,
                        label, ha="center", va="bottom",
                        fontsize=7.5)

    fig.suptitle("Эксп. 1 vs Эксп. 2 — эффект препроцессинга (orientation + resize)",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "exp1_vs_exp2.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {FIG / 'exp1_vs_exp2.png'}")


def plot_per_engine_delta():
    """Бар: Δ F1 от препроцессинга, упорядоченный."""
    deltas = []
    for e in ENGINES:
        a1 = load_aggregate(OUT1, e)["f1"]["micro"]["f1"]
        a2 = load_aggregate(OUT2, e)["f1"]["micro"]["f1"]
        deltas.append((e, a2 - a1, a1, a2))
    deltas.sort(key=lambda r: -r[1])
    palette = sns.color_palette("RdYlGn", n_colors=11)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (e, d, a1, a2) in enumerate(deltas):
        c = palette[5 + int(round(d * 100))] if abs(d) < 0.05 else (palette[-1] if d > 0 else palette[0])
        ax.barh(i, d, color=c, edgecolor="#000", linewidth=0.6)
        # подпись справа
        ax.text(d + 0.001 if d >= 0 else d - 0.001, i,
                f"{a1:.3f} → {a2:.3f}  Δ {d:+.3f}",
                va="center", ha="left" if d >= 0 else "right",
                fontsize=10)
    ax.set_yticks(range(len(deltas)))
    ax.set_yticklabels([SHORT[e] + f" ({DEVICE[e]})" for e, _, _, _ in deltas])
    ax.axvline(0, color="#000", lw=0.7)
    ax.set_xlabel("Δ micro-F1 (Эксп. 2 − Эксп. 1)")
    ax.set_title("Эффект препроцессинга на каждый движок", fontsize=12, weight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(-0.02, max(d for _, d, _, _ in deltas) + 0.04)
    fig.tight_layout()
    fig.savefig(FIG / "delta_f1.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {FIG / 'delta_f1.png'}")


def plot_speed_quality_2():
    fig, ax = plt.subplots(figsize=(7.8, 5.5))
    palette = sns.color_palette("Set2", n_colors=len(ENGINES))
    for e, c in zip(ENGINES, palette):
        a1 = load_aggregate(OUT1, e)
        a2 = load_aggregate(OUT2, e)
        x1, y1 = a1["mean_runtime"], a1["f1"]["micro"]["f1"]
        x2, y2 = a2["mean_runtime"], a2["f1"]["micro"]["f1"]
        # стрелка
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=c, alpha=0.6, lw=1.5))
        # точки
        ax.scatter(x1, y1, color=c, s=140, edgecolor="#888",
                   linewidth=0.6, alpha=0.45, zorder=3)
        ax.scatter(x2, y2, color=c, s=220, edgecolor="#000",
                   linewidth=0.9, alpha=1.0, zorder=4, label=SHORT[e])
        ax.annotate(f"  {SHORT[e]}", (x2, y2), xytext=(8, -3),
                    textcoords="offset points", fontsize=9, weight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("sec / image  (log, ниже — лучше)")
    ax.set_ylabel("micro-F1 (выше — лучше)")
    ax.set_title("Эксп. 1 → Эксп. 2: движение по Pareto", fontsize=12, weight="bold")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "pareto_movement.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {FIG / 'pareto_movement.png'}")


def main():
    plot_exp1_vs_exp2()
    plot_per_engine_delta()
    plot_speed_quality_2()


if __name__ == "__main__":
    main()
