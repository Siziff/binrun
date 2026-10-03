"""Taxonomy validator для scientific_name через GBIF API.

Использует GBIF Species Match API для каждого scientific_name из LoRA-предсказаний:
  https://api.gbif.org/v1/species/match?name=<name>&kingdom=Fungi

Логика:
  - exact match → leave as is (or normalize to canonical)
  - fuzzy match → suggest correction (с пометкой confidence)
  - no match → flag "unknown"

Output:
  - {input}_validated.jsonl — copy of input + added 'validation' field per record
  - validation_report.csv — статистика по полю

Usage:
  python experiments/exp15_validator/validate_taxonomy.py \\
    --input experiments/exp06_lora/sweep/eval_full394/p5j_merged_batched_full394.jsonl \\
    --output experiments/exp15_validator/p5j_merged_batched_validated.jsonl \\
    --field scientific_name
"""
from __future__ import annotations
import argparse, json, re, sys, time, urllib.parse, urllib.request, urllib.error
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl_tolerant(p: Path):
    """Tolerant reader — пропускает broken JSON lines."""
    import re as re_local
    text = p.read_text(encoding='utf-8', errors='replace')
    starts = [m.start() for m in re_local.finditer(r'^\{"image":\s*"', text, re_local.M)]
    for i, s in enumerate(starts):
        e = starts[i+1] if i+1 < len(starts) else len(text)
        blob = text[s:e].rstrip()
        try:
            yield json.loads(blob)
        except Exception:
            continue


def gbif_match(name: str, kingdom: str = "Fungi", strict: bool = False, timeout: int = 10) -> dict:
    """Прямой match endpoint GBIF Species API. Возвращает dict с matchType, canonicalName и др."""
    if not name or not isinstance(name, str):
        return {"matchType": "NONE", "reason": "empty_name"}

    # Очистка имени — уберём цитаты, экстра-spaces, "n. sp.", "sp. nova" и т.п.
    cleaned = name.strip()
    cleaned = re.sub(r'\bn\.\s*sp\.|sp\.\s*nov\.?|nov\.\s*spec\.?|new\s+species\b', '', cleaned, flags=re.I).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(',. ')
    if not cleaned:
        return {"matchType": "NONE", "reason": "empty_after_clean", "original": name}

    params = {
        "name": cleaned,
        "kingdom": kingdom,
        "verbose": "true",
    }
    if strict:
        params["strict"] = "true"

    url = "https://api.gbif.org/v1/species/match?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
            data["_query"] = cleaned
            data["_original"] = name
            return data
    except urllib.error.URLError as e:
        return {"matchType": "ERROR", "reason": f"http_error: {e}", "_query": cleaned, "_original": name}
    except Exception as e:
        return {"matchType": "ERROR", "reason": f"{type(e).__name__}: {e}", "_query": cleaned, "_original": name}


def decide_action(match_result: dict) -> dict:
    """Решает что делать с результатом GBIF match.

    Returns:
      {
        action: 'keep' | 'correct' | 'flag',
        confidence: high | medium | low | none,
        corrected_name: str | None,
        reason: str,
      }
    """
    mt = match_result.get("matchType")
    confidence = match_result.get("confidence", 0)

    if mt == "EXACT":
        return {
            "action": "keep",
            "confidence": "high",
            "corrected_name": match_result.get("canonicalName"),
            "matched_name": match_result.get("scientificName"),
            "reason": f"exact match in GBIF",
            "gbif_key": match_result.get("usageKey"),
            "rank": match_result.get("rank"),
            "family": match_result.get("family"),
        }
    elif mt == "FUZZY":
        # Высокий confidence для fuzzy
        if confidence >= 95:
            level = "high"
            action = "correct"
        elif confidence >= 80:
            level = "medium"
            action = "correct"
        else:
            level = "low"
            action = "flag"
        return {
            "action": action,
            "confidence": level,
            "corrected_name": match_result.get("canonicalName"),
            "matched_name": match_result.get("scientificName"),
            "reason": f"fuzzy match (conf={confidence})",
            "gbif_key": match_result.get("usageKey"),
            "rank": match_result.get("rank"),
            "family": match_result.get("family"),
        }
    elif mt == "HIGHERRANK":
        return {
            "action": "flag",
            "confidence": "low",
            "corrected_name": match_result.get("canonicalName"),
            "matched_name": match_result.get("scientificName"),
            "reason": f"only higher rank match: {match_result.get('rank')}",
            "rank": match_result.get("rank"),
        }
    elif mt == "NONE":
        return {
            "action": "flag",
            "confidence": "none",
            "corrected_name": None,
            "reason": match_result.get("reason", "no match in GBIF"),
        }
    elif mt == "ERROR":
        return {
            "action": "flag",
            "confidence": "none",
            "corrected_name": None,
            "reason": match_result.get("reason"),
        }
    else:
        return {
            "action": "flag",
            "confidence": "none",
            "corrected_name": None,
            "reason": f"unknown matchType: {mt}",
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="LoRA predictions JSONL")
    ap.add_argument("--output", required=True, help="Output JSONL with validation")
    ap.add_argument("--field", default="scientific_name", help="Field to validate")
    ap.add_argument("--kingdom", default="Fungi", help="GBIF kingdom filter")
    ap.add_argument("--apply-corrections", action="store_true",
                    help="If set, replace original value with corrected_name when action=correct")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = list(read_jsonl_tolerant(input_path))
    print(f"[validate] loaded {len(records)} records from {input_path.name}", flush=True)
    print(f"[validate] field={args.field}, kingdom={args.kingdom}, apply={args.apply_corrections}", flush=True)

    stats = Counter()
    out_records = []
    t0 = time.perf_counter()
    for i, rec in enumerate(records, 1):
        img = rec.get("image")
        pj = rec.get("parsed_json") or {}
        original = pj.get(args.field) if pj else None

        if not original or not isinstance(original, str):
            stats["skipped_no_value"] += 1
            rec["validation"] = {
                args.field: {"action": "skip", "reason": "no original value"}
            }
            out_records.append(rec)
            continue

        match = gbif_match(original, kingdom=args.kingdom)
        decision = decide_action(match)
        stats[f"action_{decision['action']}"] += 1
        stats[f"confidence_{decision['confidence']}"] += 1

        # Save raw GBIF match for debugging (without large redundant fields)
        rec["validation"] = {
            args.field: {
                "original": original,
                **decision,
            },
        }

        # Apply correction if requested
        if args.apply_corrections and decision["action"] == "correct":
            new_val = decision["corrected_name"]
            if new_val and new_val != original:
                pj[args.field] = new_val
                stats["applied_corrections"] += 1

        out_records.append(rec)

        if i % 20 == 0 or i == len(records):
            elapsed = time.perf_counter() - t0
            rate = elapsed / i
            eta = rate * (len(records) - i)
            print(f"[validate] {i}/{len(records)}  "
                  f"keep={stats['action_keep']} correct={stats['action_correct']} "
                  f"flag={stats['action_flag']}  {rate:.2f}s/rec  ETA {eta:.0f}s", flush=True)

    # Write output
    with output_path.open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[validate] written: {output_path}", flush=True)

    print(f"\n=== Stats ({args.field}) ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
