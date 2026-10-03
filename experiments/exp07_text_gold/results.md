# Эксперимент 7 — text-gold benchmark

50 файлов, gold = `gold_text.jsonl` (полный verbatim transcript).

Метрики (после NFKC + casefold + dash/quote normalisation):
- **CER** = Levenshtein(chars) / len(gold_chars) (lower is better)
- **WER** = Levenshtein(words) / len(gold_words) (lower is better)
- **sim** = SequenceMatcher ratio (higher is better)
- **rec** = chars(pred) / chars(gold) — purity/coverage proxy

| # | модель | family | n | CER ↓ | WER ↓ | sim ↑ | rec | sec/img |
|---|---|---|---|---|---|---|---|---|
| 1 | `glm-ocr-RAW50` | ocr-raw | 45 | 0.410 | 0.568 | **0.633** | 1.04 | 6.7 |
| 2 | `glm-ocr` | ocr-raw | 50 | 0.399 | 0.558 | **0.630** | 1.03 | 6.6 |
| 3 | `paddleocr-vl` | ocr-raw | 50 | 0.426 | 0.605 | **0.525** | 1.06 | 26.0 |
| 4 | `chandra-md` | chandra-md | 50 | 0.790 | 0.992 | **0.412** | 1.51 | 53.4 |
| 5 | `logics-parsing-v2-RAW50` | ocr-raw | 49 | 0.636 | 0.855 | **0.329** | 1.08 | 18.8 |
| 6 | `logics-parsing-v2` | ocr-raw | 48 | 0.655 | 0.885 | **0.312** | 1.07 | 20.4 |
| 7 | `deepseek-ocr` | ocr-raw | 50 | 2.851 | 3.339 | **0.284** | 3.05 | 31.9 |
| 8 | `Qwen3-VL-30B-A3B-Thinking-A` | vlm-json | 2 | 0.705 | 0.804 | **0.239** | 0.42 | 194.6 |
| 9 | `unlimited-ocr` | ocr-raw | 49 | 2.701 | 4.367 | **0.207** | 3.14 | 74.0 |
| 10 | `got-ocr2` | ocr-raw | 50 | 2.917 | 4.251 | **0.187** | 2.67 | 21.8 |
| 11 | `Qwen3-VL-8B-Instruct-A_fs` | vlm-json | 50 | 1.812 | 1.668 | **0.170** | 1.55 | 21.7 |
| 12 | `InternVL3-8B-A` | vlm-json | 49 | 1.825 | 2.429 | **0.166** | 1.47 | 27.0 |
| 13 | `Qwen3-VL-30B-A3B-Instruct-A` | vlm-json | 48 | 3.941 | 4.280 | **0.148** | 4.14 | 189.5 |
| 14 | `Qwen3-VL-8B-Instruct-A` | vlm-json | 50 | 8.361 | 13.894 | **0.132** | 8.20 | 61.0 |

