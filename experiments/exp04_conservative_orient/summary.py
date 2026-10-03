"""Сводная таблица Эксп. 4 + сравнение со всеми предыдущими + выбор trio.

Делает то же, что `exp03_hybrid_orient/summary.py`, но для exp04.
Сравнения: Exp2 (baseline-preprocessed), Exp3 (aggressive hybrid), Exp4 (conservative).
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
EXP4 = ROOT / "experiments" / "exp04_conservative_orient" / "outputs"
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
        "CER":      agg.get("mean_cer"),
        "WER":      agg.get("mean_wer"),
        "sim":      agg.get("mean_similarity"),
        "micro_P":  micro.get("precision"),
        "micro_R":  micro.get("recall"),
        "micro_F1": micro.get("f1"),
        "macro_F1": macro.get("f1"),
        "soft_R":   f1.get("soft_macro_recall"),
        "sec/img":  summary.get("mean_runtime"),
    }


def build_table(out_dir: Path) -> pd.DataFrame:
    rows = {e: to_row(load_metrics(out_dir, e), load_summary(out_dir, e)) for e in ENGINES}
    return pd.DataFrame(rows).T


def load_per_image_sim(out_dir: Path) -> dict[str, dict[str, float]]:
    out = {}
    for e in ENGINES:
        m = load_metrics(out_dir, e)
        if not m:
            continue
        out[e] = {rec["image"]: float(rec.get("similarity", 0.0))
                  for rec in m.get("per_image", [])}
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
    oracle = float(np.stack([M[e] for e in engines], axis=0).max(axis=0).mean())
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
    t4 = build_table(EXP4)

    delta_42 = (t4 - t2).round(3); delta_42.columns = [f"Δ{c}" for c in delta_42.columns]
    delta_43 = (t4 - t3).round(3); delta_43.columns = [f"Δ{c}" for c in delta_43.columns]

    out_dir = EXP4.parent / "summary"
    out_dir.mkdir(exist_ok=True)
    t4.round(3).to_csv(out_dir / "table_exp4.csv", index_label="engine")
    delta_42.to_csv(out_dir / "delta_exp4_vs_exp2.csv", index_label="engine")
    delta_43.to_csv(out_dir / "delta_exp4_vs_exp3.csv", index_label="engine")

    print("\n=== Эксп. 4 (consrvative hybrid, threshold 0.92): все 7 движков ===")
    print(t4.round(3).sort_values("micro_F1", ascending=False).to_string())

    print("\n=== Δ Эксп. 4 − Эксп. 2 (baseline-preprocessed) ===")
    print(delta_42.sort_values("Δmicro_F1", ascending=False).to_string())

    print("\n=== Δ Эксп. 4 − Эксп. 3 (aggressive hybrid) ===")
    print(delta_43.sort_values("Δmicro_F1", ascending=False).to_string())

    print("\n=== Сравнение micro_F1 по 3 экспериментам ===")
    cmp = pd.DataFrame({"Exp2": t2["micro_F1"], "Exp3": t3["micro_F1"], "Exp4": t4["micro_F1"]}).round(3)
    cmp["best"] = cmp.idxmax(axis=1)
    print(cmp.sort_values("Exp4", ascending=False).to_string())

    per_eng = load_per_image_sim(EXP4)
    if len(per_eng) == len(ENGINES):
        corr = compute_correlations(per_eng).round(3)
        corr.to_csv(out_dir / "spearman_corr.csv")
        print("\n=== Spearman ρ (per-image similarity, Exp4) ===")
        print(corr.to_string())

        trios = best_trios(per_eng).round(4)
        trios.to_csv(out_dir / "best_trios.csv", index=False)
        print(f"\n=== Top trios (Exp4); oracle(7) = {trios.attrs['oracle_7']:.4f} ===")
        print(trios.to_string(index=False))


if __name__ == "__main__":
    main()
