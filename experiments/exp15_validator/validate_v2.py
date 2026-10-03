"""Расширенный валидатор v2 — все поля через несколько API.

Validates:
  - scientific_name: GBIF (Fungi) → Index Fungorum fallback
  - taxon_authority: проверяется параллельно с scientific_name через Index Fungorum
  - country: REST Countries API
  - verbatim_substrate: GBIF Plantae (для host plants)

Сохраняет авторов при applying corrections (не теряет таксономическую информацию).

Usage:
  python experiments/exp15_validator/validate_v2.py \\
    --input experiments/exp06_lora/sweep/eval_full394/p5j_merged_batched_full394.jsonl \\
    --output experiments/exp15_validator/p5j_validated_v2.jsonl \\
    --apply-corrections
"""
from __future__ import annotations
import argparse, json, re, sys, time, urllib.parse, urllib.request, urllib.error, xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from functools import lru_cache

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl_tolerant(p: Path):
    text = p.read_text(encoding='utf-8', errors='replace')
    starts = [m.start() for m in re.finditer(r'^\{"image":\s*"', text, re.M)]
    for i, s in enumerate(starts):
        e = starts[i+1] if i+1 < len(starts) else len(text)
        blob = text[s:e].rstrip()
        try:
            yield json.loads(blob)
        except Exception:
            continue


def http_get(url: str, timeout: int = 10) -> tuple[int, str]:
    """Простой HTTP GET с обработкой ошибок."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'td-validator/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        return e.code, ''
    except Exception as e:
        return 0, ''


def split_taxon_and_authors(scientific_name: str) -> tuple[str, str]:
    """Разделяет 'Genus species authors' → ('Genus species', 'authors')."""
    if not scientific_name:
        return '', ''
    # Простая эвристика: первые 2 слова = Genus + species (или Genus + sp.)
    parts = scientific_name.strip().split()
    if len(parts) < 2:
        return scientific_name.strip(), ''
    # Если 2-е слово начинается с большой буквы — это может быть род подвида или авторы
    # Берём первые 2 как taxon, остальное — authors
    taxon = ' '.join(parts[:2])
    authors = ' '.join(parts[2:]).strip()
    return taxon, authors


def clean_name(name: str) -> str:
    """Очищает имя от 'n. sp.', 'sp. nov.' и подобных пометок."""
    cleaned = name.strip()
    cleaned = re.sub(r'\bn\.\s*sp\.|sp\.\s*nov\.?|nov\.\s*spec\.?|new\s+species\b', '', cleaned, flags=re.I).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(',. ')
    return cleaned


# ============================================================
# GBIF Species Match
# ============================================================

@lru_cache(maxsize=1024)
def gbif_match(name: str, kingdom: str = "Fungi") -> dict:
    """GBIF Species Match API."""
    if not name:
        return {"matchType": "NONE", "reason": "empty"}
    cleaned = clean_name(name)
    if not cleaned:
        return {"matchType": "NONE", "reason": "empty_after_clean"}
    params = {"name": cleaned, "kingdom": kingdom, "verbose": "true"}
    url = "https://api.gbif.org/v1/species/match?" + urllib.parse.urlencode(params)
    status, body = http_get(url)
    if status != 200:
        return {"matchType": "ERROR", "reason": f"http_{status}", "_query": cleaned}
    try:
        data = json.loads(body)
        data["_query"] = cleaned
        return data
    except Exception as e:
        return {"matchType": "ERROR", "reason": f"json: {e}", "_query": cleaned}


# ============================================================
# Index Fungorum REST API (fallback для редких видов)
# ============================================================

@lru_cache(maxsize=512)
def indexfungorum_search(name: str) -> dict:
    """Index Fungorum NameSearch API. Возвращает первый match."""
    if not name:
        return {"matchType": "NONE"}
    cleaned = clean_name(name)
    url = f"http://www.indexfungorum.org/ixfwebservice/fungus.asmx/NameSearch?SearchText={urllib.parse.quote(cleaned)}&AnywhereInText=false&MaxNumber=5"
    status, body = http_get(url, timeout=15)
    if status != 200:
        return {"matchType": "ERROR", "reason": f"http_{status}", "_query": cleaned}
    # Парсим XML
    try:
        # Index Fungorum возвращает namespaced XML, упростим: regex поиск
        names = re.findall(r'<NAME[^>]*>([^<]+)</NAME[^>]*>', body)
        authors = re.findall(r'<AUTHORS>([^<]*)</AUTHORS>', body)
        if not names:
            return {"matchType": "NONE", "_query": cleaned}
        # Прямое exact match — name == cleaned
        for i, n in enumerate(names):
            if n.strip().lower() == cleaned.lower():
                return {
                    "matchType": "EXACT",
                    "canonicalName": n.strip(),
                    "authors": authors[i] if i < len(authors) else '',
                    "_query": cleaned,
                }
        # Иначе fuzzy
        return {
            "matchType": "FUZZY",
            "canonicalName": names[0].strip(),
            "authors": authors[0] if authors else '',
            "_query": cleaned,
        }
    except Exception as e:
        return {"matchType": "ERROR", "reason": f"parse: {e}", "_query": cleaned}


# ============================================================
# REST Countries API (для country)
# ============================================================

# Local mapping для частых русских/английских стран
COUNTRY_NORMALIZATION = {
    'россия': 'Russia', 'рoссия': 'Russia',
    'ussr': 'Russia', 'cccp': 'Russia', 'ссср': 'Russia',
    'china': 'China', 'китай': 'China',
    'india': 'India', 'индия': 'India',
    'germany': 'Germany', 'deutschland': 'Germany',
    'usa': 'United States', 'us': 'United States',
    'uk': 'United Kingdom',
}

@lru_cache(maxsize=256)
def restcountries_lookup(name: str) -> dict:
    """Country lookup через pycountry (offline ISO 3166). Возвращает canonical English name + ISO."""
    if not name:
        return {"matchType": "NONE"}
    cleaned = name.strip()
    
    # Local quick map для частых alias
    if cleaned.lower() in COUNTRY_NORMALIZATION:
        canonical = COUNTRY_NORMALIZATION[cleaned.lower()]
    else:
        canonical = cleaned
    
    # Используем pycountry
    try:
        import pycountry
    except ImportError:
        return {"matchType": "NONE", "reason": "pycountry not installed", "_query": cleaned}
    
    # Try exact name match
    try:
        c = pycountry.countries.lookup(canonical)
    except LookupError:
        # Try search by partial
        results = pycountry.countries.search_fuzzy(canonical)
        if not results:
            return {"matchType": "NONE", "reason": "not found in pycountry", "_query": cleaned}
        c = results[0]
    
    # Если canonical из мапы — strict exact, иначе FUZZY если name отличается
    out_name = c.name
    is_exact = out_name.lower() == cleaned.lower() or canonical == cleaned
    return {
        "matchType": "EXACT" if is_exact else "FUZZY",
        "canonicalName": out_name,
        "iso2": c.alpha_2,
        "iso3": c.alpha_3,
        "_query": cleaned,
    }


# ============================================================
# Validators per field
# ============================================================

def validate_scientific_name(value: str) -> dict:
    """Валидирует scientific_name через GBIF + Index Fungorum fallback."""
    if not value:
        return {"action": "skip", "reason": "empty"}
    
    taxon, authors = split_taxon_and_authors(value)
    
    # Сначала GBIF
    gbif = gbif_match(value, kingdom="Fungi")
    mt = gbif.get("matchType")
    conf = gbif.get("confidence", 0)
    
    if mt == "EXACT":
        return {
            "source": "gbif",
            "action": "keep",
            "confidence": "high",
            "corrected_name": gbif.get("canonicalName"),
            "matched_name": gbif.get("scientificName"),
            "authors": authors,
            "rank": gbif.get("rank"),
            "family": gbif.get("family"),
        }
    elif mt == "FUZZY" and conf >= 80:
        # Корректируем — но сохраняем оригинальных авторов
        canonical = gbif.get("canonicalName")
        corrected_full = f"{canonical} {authors}".strip() if authors else canonical
        return {
            "source": "gbif",
            "action": "correct",
            "confidence": "high" if conf >= 95 else "medium",
            "corrected_name": canonical,
            "corrected_full": corrected_full,
            "matched_name": gbif.get("scientificName"),
            "authors": authors,
            "rank": gbif.get("rank"),
            "family": gbif.get("family"),
            "gbif_confidence": conf,
        }
    
    # Fallback на Index Fungorum для не-EXACT
    if_result = indexfungorum_search(value)
    if_mt = if_result.get("matchType")
    
    if if_mt == "EXACT":
        return {
            "source": "indexfungorum",
            "action": "keep",
            "confidence": "high",
            "corrected_name": if_result.get("canonicalName"),
            "if_authors": if_result.get("authors", ""),
            "authors": authors,
            "reason": "GBIF unknown, Index Fungorum exact",
        }
    elif if_mt == "FUZZY":
        canonical = if_result.get("canonicalName")
        corrected_full = f"{canonical} {authors}".strip() if authors else canonical
        return {
            "source": "indexfungorum",
            "action": "correct",
            "confidence": "medium",
            "corrected_name": canonical,
            "corrected_full": corrected_full,
            "if_authors": if_result.get("authors", ""),
            "authors": authors,
            "reason": "GBIF unknown, Index Fungorum fuzzy",
        }
    
    # Ни GBIF, ни Index Fungorum не помогли
    return {
        "source": "none",
        "action": "flag",
        "confidence": "none",
        "corrected_name": None,
        "authors": authors,
        "reason": f"GBIF={mt}, IF={if_mt}",
        "gbif_higher_rank": gbif.get("canonicalName") if mt == "HIGHERRANK" else None,
    }


def validate_country(value: str) -> dict:
    """Validates country name.
    
    ВАЖНО: возвращает info/keep а не correct — country names в gt используют
    различные конвенции (USA, US, Russia, etc.), и слепая нормализация
    в "United States" может УХУДШИТЬ F1.
    """
    if not value:
        return {"action": "skip", "reason": "empty"}
    result = restcountries_lookup(value)
    mt = result.get("matchType")
    if mt == "EXACT":
        return {"action": "keep", "confidence": "high",
                "canonical_name": result.get("canonicalName"),
                "iso2": result.get("iso2"), "iso3": result.get("iso3"),
                "source": result.get("_method", "pycountry")}
    elif mt == "FUZZY":
        # NO auto-correction — только info, оставляем original
        return {"action": "info", "confidence": "medium",
                "canonical_name": result.get("canonicalName"),
                "iso2": result.get("iso2"), "iso3": result.get("iso3"),
                "source": "pycountry",
                "note": "fuzzy match found but not applied (gt may use abbreviated form)"}
    return {"action": "flag", "confidence": "none",
            "reason": result.get("reason", "no match"),
            "canonical_name": None}


def validate_substrate(value: str) -> dict:
    """Validates verbatim_substrate as host plant via GBIF Plantae kingdom.
    
    Извлекает латинское бинарное имя (Genus species) из substrate.
    Игнорирует общие латинские слова (In, Matrix, foliis, spicis и т.п.).
    """
    if not value or len(value) < 5:
        return {"action": "skip", "reason": "empty or too short"}
    
    # Стоп-список общих латинских слов в substrate
    STOPWORDS = {'In', 'Matrix', 'foliis', 'spicis', 'caulibus', 'radicibus', 'frondibus',
                 'pedunculis', 'inflorescentiae', 'culmis', 'on', 'spica', 'spicae',
                 'culmo', 'radice', 'folio', 'vivo', 'siccis', 'sicca', 'culta'}
    
    # Найдём ВСЕ кандидаты "Genus species" (Genus с большой, species со маленькой)
    # Шаблон: <Cap><lower>+\s<lower>+
    candidates = re.findall(r'\b([A-Z][a-z]{2,})\s+([a-z]{3,})', value)
    
    # Фильтруем по stop-list
    valid_candidates = []
    for genus, species in candidates:
        if genus in STOPWORDS: continue
        # species тоже может быть stopword (например 'spec.', 'sp.')
        if species in {'spec', 'sp', 'spp'}: continue
        valid_candidates.append(f"{genus} {species}")
    
    if not valid_candidates:
        return {"action": "skip", "reason": "no Latin binomen found"}
    
    # Попробуем первого кандидата
    extracted = valid_candidates[0]
    gbif = gbif_match(extracted, kingdom="Plantae")
    mt = gbif.get("matchType")
    
    if mt == "EXACT":
        return {"action": "keep", "confidence": "high",
                "extracted_host": extracted,
                "canonical_host": gbif.get("canonicalName"),
                "family": gbif.get("family"),
                "source": "gbif_plantae"}
    elif mt == "FUZZY" and gbif.get("confidence", 0) >= 80:
        return {"action": "info", "confidence": "medium",
                "extracted_host": extracted,
                "canonical_host": gbif.get("canonicalName"),
                "family": gbif.get("family"),
                "source": "gbif_plantae",
                "gbif_confidence": gbif.get("confidence", 0)}
    
    # Попробуем другие кандидаты если есть
    for cand in valid_candidates[1:3]:
        g2 = gbif_match(cand, kingdom="Plantae")
        if g2.get("matchType") == "EXACT":
            return {"action": "keep", "confidence": "high",
                    "extracted_host": cand,
                    "all_candidates": valid_candidates,
                    "canonical_host": g2.get("canonicalName"),
                    "family": g2.get("family"),
                    "source": "gbif_plantae"}
    
    return {"action": "flag", "confidence": "none",
            "extracted_host": extracted,
            "all_candidates": valid_candidates,
            "reason": "no GBIF Plantae match for any candidate"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--apply-corrections", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = list(read_jsonl_tolerant(input_path))
    print(f"[validate-v2] loaded {len(records)} records", flush=True)
    print(f"[validate-v2] apply_corrections={args.apply_corrections}", flush=True)

    stats = Counter()
    out_records = []
    t0 = time.perf_counter()
    
    FIELDS = [
        ('scientific_name', validate_scientific_name),
        ('country', validate_country),
        ('verbatim_substrate', validate_substrate),
    ]
    
    for i, rec in enumerate(records, 1):
        pj = rec.get("parsed_json") or {}
        rec["validation"] = {}
        
        for field, validator in FIELDS:
            original = pj.get(field) if pj else None
            if not original:
                rec["validation"][field] = {"action": "skip", "reason": "no value"}
                stats[f"{field}_skipped"] += 1
                continue
            
            result = validator(original)
            result["original"] = original
            rec["validation"][field] = result
            stats[f"{field}_{result.get('action', 'unknown')}"] += 1
            
            # Apply corrections (используем corrected_full если есть, чтобы сохранить authors)
            if args.apply_corrections and result.get("action") == "correct":
                new_val = result.get("corrected_full") or result.get("corrected_name")
                if new_val and new_val != original:
                    pj[field] = new_val
                    stats[f"{field}_applied"] += 1
        
        out_records.append(rec)
        
        if i % 20 == 0 or i == len(records):
            elapsed = time.perf_counter() - t0
            rate = elapsed / i
            eta = rate * (len(records) - i)
            print(f"[validate-v2] {i}/{len(records)}  {rate:.2f}s/rec  ETA {eta:.0f}s", flush=True)

    with output_path.open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[validate-v2] written: {output_path}")

    print(f"\n=== Stats ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
