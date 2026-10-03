"""LLM-agent для пере-проверки flagged видов через merged Qwen3-VL модель.

Для каждого случая где GBIF/Index Fungorum не нашли таксон, мы повторно даём
ту же картинку LLM с прицельным промптом:
  - "Внимательно прочитай scientific name на этикетке"
  - Подсказываем "GBIF не знает '<original>'; ближайший род — '<genus>'"
  - Спрашиваем: "Возможно ли это другое название? Может opcs в gt?"

Output: для каждого flagged record → llm_suggested_name + reasoning

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiments/exp16_gt_audit/llm_agent_for_flagged.py \\
    --flagged /tmp/flagged_species.jsonl \\
    --model-path data/models/production/qwen3vl_lora_p5j_merged \\
    --output experiments/exp16_gt_audit/llm_agent_results.jsonl
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("HF_HOME", str(REPO_ROOT / "data" / "models" / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(REPO_ROOT / "data" / "models" / "hf_cache"))

import torch
from src.utils import env_setup  # noqa

IMG_DIR = REPO_ROOT / "data" / "preprocessed_v3" / "Smuts"


PROMPT_TEMPLATE = """Ты эксперт по таксономии грибов. На этикетке гербарного образца записано научное название.

Ранее извлечённое значение: "{original}"

Этот вариант названия НЕ найден в базе данных GBIF/Index Fungorum. Ближайший известный таксон в базе: "{gbif_hint}".

Внимательно посмотри на изображение этикетки и ответь:

1. Точное написание научного названия с авторами (в одну строку, как на этикетке)
2. Возможные причины почему GBIF не знает (опечатка в нашей предыдущей экстракции / редкий вид / синоним современного / устаревшее имя)

Ответ строго в JSON-формате:
{{
  "scientific_name_verified": "<точное название с авторами>",
  "confidence": "high|medium|low",
  "reasoning": "<короткое объяснение>"
}}

Только JSON, без markdown-блоков."""


def build_messages(image_path: str, original: str, gbif_hint: str):
    return [
        {"role": "system", "content": [{"type": "text",
            "text": "Ты помощник для таксономической верификации гербарных этикеток."}]},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": PROMPT_TEMPLATE.format(
                original=original, gbif_hint=gbif_hint or "не определён",
            )},
        ]},
    ]


def parse_response(text: str) -> dict | None:
    """Парсим JSON из ответа."""
    text = text.strip()
    # Уберём markdown fence
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.M).strip()
    # Найдём JSON блок
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def load_model(model_path: Path):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[load] model from {model_path}", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path), dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    print(f"[load] done in {time.perf_counter()-t0:.1f}s", flush=True)
    processor = AutoProcessor.from_pretrained(str(model_path))
    return model, processor


@torch.inference_mode()
def query_one(model, processor, image_path: str, original: str, gbif_hint: str,
              max_new_tokens: int = 512) -> dict:
    messages = build_messages(image_path, original, gbif_hint)
    t0 = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_trim = gen_ids[:, inputs["input_ids"].shape[-1]:]
    text = processor.batch_decode(gen_trim, skip_special_tokens=True)[0]
    runtime = time.perf_counter() - t0
    parsed = parse_response(text)
    return {
        "raw_response": text,
        "parsed": parsed,
        "runtime_seconds": runtime,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagged", required=True, help="JSONL list of flagged records")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    model_path = Path(args.model_path).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flagged = []
    for line in open(args.flagged, encoding='utf-8'):
        line = line.strip()
        if line:
            flagged.append(json.loads(line))
    if args.limit:
        flagged = flagged[:args.limit]
    print(f"[agent] flagged records: {len(flagged)}", flush=True)

    model, processor = load_model(model_path)

    out_records = []
    t_all = time.perf_counter()
    for i, r in enumerate(flagged, 1):
        img = r['image']
        image_path = str(IMG_DIR / img)
        original = r['original']
        gbif_hint = r.get('gbif_higher_rank', 'не определён')

        try:
            result = query_one(model, processor, image_path, original, gbif_hint,
                               max_new_tokens=args.max_new_tokens)
            err = None
        except Exception as e:
            result = {"raw_response": None, "parsed": None, "runtime_seconds": 0}
            err = str(e)

        out_rec = {
            "image": img,
            "original": original,
            "gbif_hint": gbif_hint,
            "agent_raw": result.get("raw_response"),
            "agent_parsed": result.get("parsed"),
            "runtime_sec": result.get("runtime_seconds"),
            "error": err,
        }
        out_records.append(out_rec)
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

        elapsed = time.perf_counter() - t_all
        rate = elapsed / i
        eta = rate * (len(flagged) - i)
        if result.get("parsed"):
            verified = result['parsed'].get('scientific_name_verified', '')[:60]
            conf = result['parsed'].get('confidence', '?')
            status = f"[{conf}] {verified!r}"
        else:
            status = "PARSE_FAIL" if result.get("raw_response") else "ERROR"
        print(f"[agent] {i}/{len(flagged)}  {rate:.1f}s/req  ETA {eta/60:.1f}min  "
              f"{img}: {status}", flush=True)

    print(f"\n[agent] done. {len(out_records)} records → {output_path}")


if __name__ == "__main__":
    main()
