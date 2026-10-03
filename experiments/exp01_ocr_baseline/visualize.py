"""Визуализация результатов Эксперимента 1.

Генерирует в `figures/`:
    1. summary_bar.png       — сводная гистограмма CER/sim/F1/runtime
    2. per_field_f1.png      — heatmap F1 по полям × движкам
    3. per_field_soft.png    — heatmap soft_recall (sim) по полям × движкам
    4. speed_quality.png     — scatter sec/img × micro-F1
    5. cer_distribution.png  — boxplot per-image CER по движкам
"""
from __future__ import annotations

from src.utils import env_setup  # noqa: F401

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments" / "exp01_ocr_baseline" / "outputs"
FIG_DIR = ROOT / "experiments" / "exp01_ocr_baseline" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ENGINES = ("tesseract", "paddleocr", "easyocr", "doctr", "surya",
           "trocr-printed", "trocr-handwritten")

# Удобные короткие имена для подписей.
SHORT = {
    "tesseract": "Tesseract",
    "paddleocr": "PaddleOCR",
    "easyocr": "EasyOCR",
    "doctr": "docTR",
    "surya": "Surya",
    "trocr-printed": "TrOCR-pr",
    "trocr-handwritten": "TrOCR-hw",
}

# Поля для heatmap — упорядочены логически.
FIELDS_ORDER = (
    "scientific_name",
    "verbatim_identification",
    "type_status",
    "collector",
    "verbatim_collector",
    "determiner",
    "field_number",
    "verbatim_collection_date",
    "country",
    "verbatim_locality",
    "verbatim_latitude",
    "verbatim_longitude",
    "verbatim_elevation",
    "verbatim_habitat",
    "verbatim_substrate",
    "exsiccata_title",
    "exsiccata_number",
    "catalog_number",
    "organism_remarks",
    "identification_remarks",
    "verbatim_identification_date",
)


def load_all() -> dict[str, dict]:
    data = {}
    for eng in ENGINES:
        p = OUT_DIR / f"{eng}_metrics.json"
        if p.exists():
            data[eng] = json.loads(p.read_text(encoding="utf-8"))
    return data


def plot_summary_bars(data: dict[str, dict]) -> None:
    """Сводная гистограмма: 4 метрики × движки."""
    engines = list(data.keys())
    labels = [SHORT[e] for e in engines]
    metrics = ["mean_similarity", "f1.micro.f1", "f1.macro.f1", "mean_runtime"]
    titles = ["mean similarity ↑", "micro-F1 ↑", "macro-F1 ↑", "sec / img ↓"]

    def get(metrics_dict, dotted):
        parts = dotted.split(".")
        v = metrics_dict
        for p in parts:
            v = v[p] if isinstance(v, dict) else v
        return v

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    palette = sns.color_palette("Set2", n_colors=len(engines))
    for ax, m, title in zip(axes.flat, metrics, titles):
        vals = [get(data[e]["aggregate"], m) for e in engines]
        # сортируем по убыванию (для runtime — по возрастанию).
        idx = np.argsort(vals)
        if "runtime" not in m:
            idx = idx[::-1]
        x = np.array(labels)[idx]
        y = np.array(vals)[idx]
        c = [palette[engines.index(e)] for e in np.array(engines)[idx]]
        bars = ax.bar(x, y, color=c, edgecolor="#333", linewidth=0.6)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_ylim(0, max(y) * 1.20)
        ax.tick_params(axis="x", rotation=25)
        for b, v in zip(bars, y):
            ax.text(b.get_x() + b.get_width() / 2, v + max(y) * 0.02,
                    f"{v:.3f}", ha="center", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("Baseline OCR (Эксперимент 1) — сводка по 7 движкам, корпус Smuts (n=394)",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "summary_bar.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {FIG_DIR / 'summary_bar.png'}")


def plot_field_heatmap(data: dict[str, dict], key: str = "f1",
                       fname: str = "per_field_f1.png",
                       title: str = "F1 по полям × движкам") -> None:
    """Heatmap: рядки — поля, столбцы — движки, цвет — метрика."""
    engines = list(data.keys())
    labels = [SHORT[e] for e in engines]
    rows = []
    field_used = []
    for f in FIELDS_ORDER:
        row = []
        supports = []
        for e in engines:
            pf = data[e]["aggregate"]["f1"]["per_field"].get(f, {})
            val = pf.get(key, 0.0)
            row.append(val)
            supports.append(pf.get("support", 0))
        # включаем поле, только если хоть у одного движка support > 0
        if max(supports) > 0:
            rows.append(row)
            field_used.append(f)

    arr = np.array(rows)
    fig, ax = plt.subplots(figsize=(8, max(6, 0.45 * len(field_used))))
    sns.heatmap(
        arr, ax=ax, cmap="YlGnBu", vmin=0, vmax=1,
        annot=True, fmt=".2f", cbar_kws={"label": key},
        xticklabels=labels, yticklabels=field_used,
        linewidths=0.4, linecolor="#fff",
    )
    ax.set_title(title, fontsize=12, weight="bold")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {FIG_DIR / fname}")


def plot_field_softrecall(data: dict[str, dict]) -> None:
    """Heatmap по soft_recall (среднее sim где gt_has=True)."""
    engines = list(data.keys())
    labels = [SHORT[e] for e in engines]
    rows = []
    field_used = []
    for f in FIELDS_ORDER:
        row = []
        supports = []
        for e in engines:
            pf = data[e]["aggregate"]["f1"]["per_field"].get(f, {})
            row.append(pf.get("soft_recall", 0.0))
            supports.append(pf.get("support", 0))
        if max(supports) > 0:
            rows.append(row)
            field_used.append(f)
    arr = np.array(rows)
    fig, ax = plt.subplots(figsize=(8, max(6, 0.45 * len(field_used))))
    sns.heatmap(arr, ax=ax, cmap="rocket_r", vmin=0, vmax=1,
                annot=True, fmt=".2f",
                cbar_kws={"label": "soft recall (avg sim)"},
                xticklabels=labels, yticklabels=field_used,
                linewidths=0.4, linecolor="#fff")
    ax.set_title("Soft recall по полям (avg sim при gt_has=True)",
                 fontsize=12, weight="bold")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_field_soft.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {FIG_DIR / 'per_field_soft.png'}")


def plot_speed_quality(data: dict[str, dict]) -> None:
    """Scatter: время × F1."""
    engines = list(data.keys())
    palette = sns.color_palette("Set2", n_colors=len(engines))
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for e, c in zip(engines, palette):
        a = data[e]["aggregate"]
        x = a["mean_runtime"]
        y = a["f1"]["micro"]["f1"]
        ax.scatter(x, y, color=c, s=220, edgecolor="#222", linewidth=0.8,
                   label=SHORT[e], zorder=3)
        ax.annotate(SHORT[e], (x, y), xytext=(8, -3),
                    textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("sec / image  (log scale, ниже — лучше)")
    ax.set_ylabel("micro-F1  (выше — лучше)")
    ax.set_title("Скорость vs качество: baseline OCR", fontsize=12, weight="bold")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "speed_quality.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {FIG_DIR / 'speed_quality.png'}")


def plot_cer_box(data: dict[str, dict]) -> None:
    """Распределение per-image similarity по движкам (CER ушёл бы за шкалу)."""
    engines = list(data.keys())
    palette = sns.color_palette("Set2", n_colors=len(engines))
    series = []
    for e in engines:
        per = data[e]["per_image"]
        sims = [m["similarity"] for m in per if m.get("similarity") is not None]
        series.append(sims)

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(series, patch_artist=True, showfliers=True,
                    tick_labels=[SHORT[e] for e in engines], widths=0.55)
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("#000")
    ax.set_ylabel("per-image similarity")
    ax.set_ylim(0, 1.0)
    ax.set_title("Распределение per-image similarity по движкам", fontsize=12, weight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cer_distribution.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {FIG_DIR / 'cer_distribution.png'}")


def main() -> None:
    print("Loading metrics…")
    data = load_all()
    print(f"  loaded engines: {list(data.keys())}")
    plot_summary_bars(data)
    plot_field_heatmap(data, key="f1", fname="per_field_f1.png",
                       title="F1 по полям × движкам")
    plot_field_softrecall(data)
    plot_speed_quality(data)
    plot_cer_box(data)


if __name__ == "__main__":
    main()
