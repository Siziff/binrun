"""Сводная таблица Эксп. 3 + сравнение с Эксп. 2 + выбор лучшего trio.

Читает `<engine>_metrics.json` (структура: aggregate{mean_cer, mean_wer,
mean_similarity, f1{macro, micro, soft_macro_recall}, …}, per_image[]) для
каждого движка и считает:
  • сводную таблицу Эксп. 3 (CER/WER/sim/F1/soft_R/sec_per_img);
  • Δ-таблицу Эксп. 3 vs Эксп. 2;
  • Spearman ρ между движками по per-image similarity;
  • перебор всех C(7,3) trio по mean(max_sim_over_trio) — критерий
    комплементарности.
"""
from __future__ import annotations

from src.utils import env_setup  # noqa: F401

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
EXP3 = ROOT / "experiments" / "exp03_hybrid_orient" / "outputs"
EXP2 = ROOT / "experiments" / "exp02_preprocessed" / "outputs"

ENGINES = [
    "tesseract", "paddleocr", "easyocr", "doctr",
    "surya", "trocr-printed", "trocr-handwritten",
]


def load_metrics(out_dir: Path, engine: str) -> dict:
    p = out_dir / f"{engine}_metrics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def load_summary(out_dir: Path, engine: str) -> dict:
    p = out_dir / f"{engine}_summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def to_row(metrics: dict, summary: dict) -> dict:
    agg = metrics.get("aggregate", {}) if metrics else {}
    f1 = agg.get("f1", {}) or {}
    macro = f1.get("macro", {}) or {}
    micro = f1.get("micro", {}) or {}
    return {
        "CER":       agg.get("mean_cer"),
        "WER":       agg.get("mean_wer"),
        "sim":       agg.get("mean_similarity"),
        "micro_P":   micro.get("precision"),
        "micro_R":   micro.get("recall"),
        "micro_F1":  micro.get("f1"),
        "macro_F1":  macro.get("f1"),
        "soft_R":    f1.get("soft_macro_recall"),
        "sec/img":   summary.get("mean_runtime"),
    }


def build_table(out_dir: Path) -> pd.DataFrame:
    rows = {}
    for e in ENGINES:
        rows[e] = to_row(load_metrics(out_dir, e), load_summary(out_dir, e))
    return pd.DataFrame(rows).T


def load_per_image_sim(out_dir: Path) -> dict[str, dict[str, float]]:
    """Возвращает {engine: {image_name: doc-level similarity}}."""
    out = {}
    for e in ENGINES:
        m = load_metrics(out_dir, e)
        if not m:
            continue
        sub = {}
        for rec in m.get("per_image", []):
            sub[rec["image"]] = float(rec.get("similarity", 0.0))
        out[e] = sub
    return out


def compute_correlations(per_eng: dict[str, dict[str, float]]) -> pd.DataFrame:
    engines = list(per_eng.keys())
    common = sorted(set.intersection(*[set(per_eng[e].keys()) for e in engines]))
    M = np.array([[per_eng[e][img] for img in common] for e in engines])
    n = len(engines)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(M[i], M[j])
            corr[i, j] = corr[j, i] = float(rho)
    return pd.DataFrame(corr, index=engines, columns=engines)


def best_trios(per_eng: dict[str, dict[str, float]], top_k: int = 10) -> pd.DataFrame:
    engines = list(per_eng.keys())
    common = sorted(set.intersection(*[set(per_eng[e].keys()) for e in engines]))
    M = {e: np.array([per_eng[e][img] for img in common]) for e in engines}
    all_stack = np.stack([M[e] for e in engines], axis=0)
    oracle = float(all_stack.max(axis=0).mean())

    rows = []
    for trio in combinations(engines, 3):
        stack = np.stack([M[e] for e in trio], axis=0)
        mean_max = float(stack.max(axis=0).mean())
        rhos = [spearmanr(M[a], M[b]).statistic for a, b in combinations(trio, 2)]
        rows.append({
            "trio": " + ".join(trio),
            "mean_max_sim": mean_max,
            "frac_of_oracle": mean_max / oracle if oracle else 0.0,
            "mean_pair_rho": float(np.mean(rhos)),
        })
    df = pd.DataFrame(rows).sort_values("mean_max_sim", ascending=False)
    df.attrs["oracle_7"] = oracle
    return df.head(top_k)


def main() -> None:
    t2 = build_table(EXP2)
    t3 = build_table(EXP3)

    # Δ‑таблица.
    delta = (t3 - t2).round(3)
    delta.columns = [f"Δ{c}" for c in delta.columns]
    t2r = t2.round(3); t2r.columns = [f"{c}_exp2" for c in t2.columns]
    t3r = t3.round(3); t3r.columns = [f"{c}_exp3" for c in t3.columns]

    out_dir = EXP3.parent / "summary"
    out_dir.mkdir(exist_ok=True)
    t3r.to_csv(out_dir / "table_exp3.csv", index_label="engine")
    t2r.to_csv(out_dir / "table_exp2.csv", index_label="engine")
    delta.to_csv(out_dir / "delta_exp3_vs_exp2.csv", index_label="engine")

    # «Широкая» табличка для печати: ключевые поля.
    wide = pd.concat([t3r[["sim_exp3", "micro_F1_exp3", "macro_F1_exp3",
                            "soft_R_exp3", "sec/img_exp3"]],
                      delta[["Δsim", "Δmicro_F1", "Δmacro_F1", "Δsoft_R"]]], axis=1)
    wide = wide.sort_values("micro_F1_exp3", ascending=False)
    wide.to_csv(out_dir / "table_wide.csv", index_label="engine")

    print("\n=== Эксп. 3 (hybrid orientation): все 7 движков ===")
    print(t3.round(3).sort_values("micro_F1", ascending=False).to_string())

    print("\n=== Δ Эксп. 3 − Эксп. 2 (положительное = улучшение) ===")
    print(delta.sort_values("Δmicro_F1", ascending=False).to_string())

    print("\n=== Сводно (Exp3 + Δ vs Exp2) ===")
    print(wide.to_string())

    # Корреляции.
    per_eng = load_per_image_sim(EXP3)
    if len(per_eng) == len(ENGINES):
        corr = compute_correlations(per_eng).round(3)
        corr.to_csv(out_dir / "spearman_corr.csv")
        print("\n=== Spearman ρ (per-image similarity) ===")
        print(corr.to_string())

        trios = best_trios(per_eng).round(4)
        trios.to_csv(out_dir / "best_trios.csv", index=False)
        oracle = trios.attrs["oracle_7"]
        print(f"\n=== Top trios by mean(max sim over trio) ===")
        print(f"    oracle(7 движков) = {oracle:.4f}")
        print(trios.to_string(index=False))


if __name__ == "__main__":
    main()
