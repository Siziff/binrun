# TD — распознавание и нормализация метаданных гербарных этикеток (LE Fungarium, Smuts)

Гибридная система извлечения структурированных метаданных (Darwin Core + микологические поля)
из фотографий этикеток микологического гербария БИН РАН (LE): OCR/HTR -> VLM-нормализатор ->
LoRA fine-tune Qwen3-VL-8B -> tool-using валидатор (GBIF/Index Fungorum).

Лучший результат: **LoRA Qwen3-VL-8B-Instruct `p5j_r32_lr7e5_ep5_a256` — F1-micro 0.972, CER 0.061
на полном корпусе 394 изображений**; merged + batch=8 даёт 2.77 с/изображение на 1x H100.

Главные документы:

| Файл | Что внутри |
|---|---|
| `RESULTS.md` / `RESULTS.pdf` | Полный журнал всех экспериментов (1-17), метрики, выводы, гайдлайны |
| `README_start.md` / `README.pdf` | Научная постановка, обзор SOTA, библиография |
| `summary_all.{csv,md}` | Сводная таблица метрик экспериментов 1-7 |
| `paper/` | LaTeX-черновик статьи (CVPR template) |

> В репозитории нет сырых изображений, весов базовых моделей и conda-окружений (см. разделы ниже
> — всё восстанавливается). Абсолютные пути в скриптах указывают на исходный хост
> `/workspace-SR008.fs2/mikheev-kandy/TD/`; при переносе замените на свой `PROJECT_ROOT`.

---

## 1. Структура репозитория

```
.
├── RESULTS.md, RESULTS.pdf          журнал экспериментов (главный документ)
├── README_start.md, README.pdf      постановка задачи и обзор литературы
├── LE_Fungarium_Ustilaginomycotina.xlsx   исходная таблица кураторов (ground truth)
├── env.sh                           активация окружения + все пути кэшей (TD_PREF=ocr|vlm|vllm|legacy|sglang)
├── unified_eval.py                  единый evaluator (CER/WER/sim/F1 против 21 поля GT)
├── build_pdf.py                     MD -> PDF
├── envs/requirements-td-*.txt       pip freeze всех пяти окружений
├── src/
│   ├── preprocessing/               preprocess.py (v1), preprocess_v2.py, preprocess_v3.py, ориентация, DINO-кроп
│   ├── ocr/                         обёртки движков: tesseract, paddle, easyocr, doctr, surya, trocr
│   ├── eval/                        метрики, per-field F1
│   └── utils/                       build_gold.py (xlsx -> labels.jsonl), download_models.py, env_setup.py
├── data/
│   └── gold/                        labels.jsonl (429 записей GT, 21 поле) + image_index.json
└── experiments/
    ├── exp01_ocr_baseline/          6 OCR-движков на сырых изображениях (outputs, figures, best_worst.py)
    ├── exp02_preprocessed/          те же движки после препроцессинга v1
    ├── exp03_hybrid_orient/         агрессивная гибридная ориентация (v2)
    ├── exp04_conservative_orient/   консервативная ориентация (v3, канонический препроцессинг)
    ├── exp05_vlm_normalizer/        VLM-нормализаторы (Qwen3-VL, InternVL, chandra, GLM-OCR, lift, ...), few-shot, форматтер
    ├── exp06_lora/                  LoRA fine-tune Qwen3-VL-8B: трейнер, sweep (83 конфига), eval, merge, адаптеры
    │   ├── adapters/                топ-4 LoRA-адаптера в bf16, разбитые на части <95 MB (см. раздел 4)
    │   ├── splits/                  train 310 / test 79
    │   ├── sweep/configs/           83 JSON-конфига sweep'а
    │   ├── sweep/eval_test79/       предсказания каждого run на 79 test
    │   ├── sweep/eval_full394/      предсказания топ-4 на полных 394
    │   ├── sweep/per_field/         per-field отчёты
    │   ├── sweep/logs/*_train.log   кривые loss всех sweep-run
    │   └── sweep/sweep_summary.{csv,md}  рейтинг всех 83 run
    ├── exp07_text_gold/             verbatim text-gold benchmark (50 файлов)
    ├── exp15_validator/             tool-using валидатор через GBIF/Index Fungorum API
    ├── exp16_gt_audit/              аудит GT через cross-validation топ-4 LoRA
    └── exp17_anthracoidea/          inference production-модели на новой подколлекции Anthracoidea (534 изобр.)
```

Удалено при публикации (не нужно для воспроизведения): raw/orchestrator/debug/download/install
логи, `__pycache__`, временные файлы `data/tmp`, промежуточные epoch-чекпоинты, LaTeX-артефакты сборки.

---

## 2. Данные

### 2.1 Сырые изображения

Источник: Яндекс.Диск — https://disk.yandex.ru/d/pnH8tH4ScOwcQQ (архив `Smuts_new.zip`, ~16.6 GB).

```
Smuts/
├── Types/          394 JPEG  "LE F-*.JPG"      — подколлекция Ustilaginomycotina, основной корпус (есть GT)
└── Anthracoidea/   534 JPEG  "Anthracoidea_<species>_NNN.JPG" — новая подколлекция (без GT, Эксп. 17)
```

Скачайте и распакуйте так, чтобы рядом с `env.sh` появились `Smuts/Types/` и `Smuts/Anthracoidea/`.

### 2.2 Ground truth

`LE_Fungarium_Ustilaginomycotina.xlsx` (лист `Data`, 429 строк, 385 с изображениями) ->
`data/gold/labels.jsonl` + `data/gold/image_index.json` через:

```bash
export TD_PREF=ocr && source env.sh
python -m src.utils.build_gold
```

Схема 21 поля: `catalog_number, scientific_name, verbatim_identification, type_status, determiner,
verbatim_identification_date, collector, verbatim_collector, field_number, verbatim_collection_date,
country, verbatim_locality, verbatim_latitude, verbatim_longitude, verbatim_elevation,
verbatim_habitat, verbatim_substrate, organism_remarks, identification_remarks,
exsiccata_title, exsiccata_number`.

### 2.3 Препроцессинг (как получался канонический корпус)

Сырые JPEG (6.7 GB для Types) никогда не подавались в модели напрямую. Все эксперименты 2+ и весь
LoRA fine-tune/inference используют `data/preprocessed_v3/`. Три версии пайплайна:

| Версия | Скрипт | Шаги | Использовалась в |
|---|---|---|---|
| v1 | `src/preprocessing/preprocess.py` | EXIF transpose -> page-level orientation PaddleX `PP-LCNet_x1_0_doc_ori` (conf >= 0.70) -> resize max_side=3000 (LANCZOS) | Эксп. 2 (47/394 повёрнуто) |
| v2 | `preprocess_v2.py` + `orient_hybrid.py` | EXIF -> Grounding DINO кроп этикетки (prompt "a paper label. a herbarium label. a printed text label.", площадь 3-45 %) -> PaddleX on crop (conf >= 0.85) -> docTR x4 fallback -> resize 3000 | Эксп. 3 (116/394 повёрнуто; регрессия в среднем) |
| **v3** | `preprocess_v3.py` + `orient_conservative.py` | EXIF -> PaddleX on page; если conf >= `--page-conf-threshold` (0.92) принять, иначе ветка v2 (DINO crop -> PaddleX crop -> docTR fallback) -> resize 3000 | **Эксп. 4-17, LoRA, production** (59/394 повёрнуто; 85.5 % решений page-level) |

Для DINO-кропа используется только оракул ориентации; в OCR/VLM подаётся полный лист.
Объём корпуса после препроцессинга: 6.7 GB -> ~440 MB (Types), ~660 MB (Anthracoidea).

Воспроизведение v3:

```bash
export TD_PREF=ocr && source env.sh            # PaddleX + docTR + Grounding DINO живут в td-ocr
python -m src.preprocessing.preprocess_v3 \
    --src-dir Smuts/Types \
    --dst-dir data/preprocessed_v3/Smuts \
    --index-path data/preprocessed_v3/preproc_index_v3.json \
    --page-conf-threshold 0.92

# Anthracoidea (Эксп. 17) — см. experiments/exp17_anthracoidea/preprocess.sh
python -m src.preprocessing.preprocess_v3 \
    --src-dir Smuts/Anthracoidea \
    --dst-dir data/preprocessed_v3/Anthracoidea \
    --index-path data/preprocessed_v3/preproc_index_anthracoidea.json
```

Индекс `preproc_index_v3.json` хранит для каждого файла угол поворота, источник решения
(`paddlex_page` / `paddlex_crop` / `doctr_fallback`) и confidence.

---

## 3. Окружения

Пять изолированных conda/venv-окружений под `data/envs/td-*`; `env.sh` переключает их через
`TD_PREF` и выставляет все кэши (HF_HOME, TORCH_HOME, PADDLE, EasyOCR, docTR, Surya, TMPDIR) внутрь
`data/`. Точные `pip freeze` — в `envs/`.

| Env | Python | Ключевое | Для чего |
|---|---|---|---|
| `td-ocr` | 3.11 | torch 2.5.1+cu121, transformers 4.49, surya-ocr 0.14, paddleocr 3.7 / paddlex 3.7, easyocr 1.7.2, python-doctr 1.0.1 | Эксп. 1-4, препроцессинг |
| `td-vlm` | 3.11 | torch 2.5.1+cu121, transformers 4.57.6, peft 0.19.1, flash-attn 2.7.4.post1 | Эксп. 5-14, 17: Qwen3-VL, LoRA train/eval/merge |
| `td-vllm` | 3.11 | torch 2.10+cu128, transformers 5.12.1, vllm 0.22.1, flash-attn 2.8.3 | GLM-OCR, PaddleOCR-VL-1.6, Logics-Parsing-v2, Gemma-4 |
| `td-legacy` | 3.11 | torch 2.5.1, transformers 4.46.3 | DeepSeek-OCR, Unlimited-OCR, GOT-OCR2.0 |
| `td-sglang` | 3.11 | torch 2.11, transformers 5.8.1, sglang 0.5.14 | эксперименты со sglang (Эксп. 5e, blocked) |

```bash
python3.11 -m venv data/envs/td-vlm && data/envs/td-vlm/bin/pip install -r envs/requirements-td-vlm.txt
# flash-attn ставился из prebuilt wheel (см. комментарий в requirements), не из sdist
export TD_PREF=vlm && source env.sh
```

Также нужен системный Tesseract с языками `rus eng deu lat fra ita spa`.

---

## 4. Модели

### 4.1 LoRA-адаптеры (в репозитории)

`experiments/exp06_lora/adapters/` содержит топ-4 адаптера sweep'а. Для git они переведены из fp32
в **bf16** (max относительная ошибка ~3.6e-3, при bf16-инференсе эквивалентно) и разбиты на части по
90 MiB. Сборка:

```bash
bash experiments/exp06_lora/adapters/unpack_adapters.sh          # все
bash experiments/exp06_lora/adapters/unpack_adapters.sh p5j_r32_lr7e5_ep5_a256   # один
```

| Run | rank | alpha | lr | ep | F1 (79 test) | F1 (394 full) | bf16 |
|---|---|---|---|---|---|---|---|
| **p5j_r32_lr7e5_ep5_a256** (production) | 32 | 256 | 7e-5 | 5 | 0.9564 | **0.9720** | 167 MB |
| p4h_r32_lr2e4_ep5 | 32 | 64 | 2e-4 | 5 | **0.9615** | 0.9710 | 167 MB |
| p4g_r16_lr1e4_ep8 | 16 | 32 | 1e-4 | 8 | 0.9548 | 0.9666 | 83 MB |

(`p4g_r48_lr1e4_ep5`, F1 0.9694, в репозиторий не включён из-за размера — 250 MB; воспроизводится
конфигом `sweep/configs/p4g_r48_lr1e4_ep5.json`.)

Загрузка:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, device_map="auto",
    attn_implementation="flash_attention_2")
model = PeftModel.from_pretrained(base, "experiments/exp06_lora/adapters/p5j_r32_lr7e5_ep5_a256").eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
# merged = model.merge_and_unload(); merged.save_pretrained("data/models/production/qwen3vl_lora_p5j_merged")
```

Tokenizer/processor-файлы в адаптерах не дублируются — они идентичны базовой модели.
Скрипт упаковки: `adapters/pack_adapters.py`.

### 4.2 Базовые модели (скачать отдельно)

Все HF-модели кладутся в `data/models/hf_cache` (его выставляет `env.sh` как `HF_HOME`):

```bash
export TD_PREF=vlm && source env.sh
pip install -U huggingface_hub
for m in Qwen/Qwen3-VL-8B-Instruct IDEA-Research/grounding-dino-base; do hf download "$m"; done
```

| Модель | Размер | Нужна для |
|---|---|---|
| `Qwen/Qwen3-VL-8B-Instruct` | 17 GB | **база LoRA, production**, формatter в Эксп. 5/11, few-shot |
| `IDEA-Research/grounding-dino-base` (и `-tiny`) | 0.9 GB | препроцессинг v2/v3 (кроп этикетки) |
| `Qwen/Qwen3-VL-8B-Thinking`, `Qwen/Qwen3-VL-30B-A3B-Instruct` | — | Эксп. 5 (сравнение) |
| `OpenGVLab/InternVL3-8B-hf` | — | Эксп. 5 |
| `datalab-to/chandra-ocr-2`, `datalab-to/surya-ocr-2`, `datalab-to/lift` | 10 / 1.4 / 18 GB | Эксп. 5, 11 |
| `PaddlePaddle/PaddleOCR-VL-1.6`, `zai-org/GLM-OCR`, `Logics-MLLM/Logics-Parsing-v2` | 1.8 / 2.5 GB | Эксп. 5b-5d (td-vllm) |
| `baidu/Unlimited-OCR`, `deepseek-ai/DeepSeek-OCR`, `stepfun-ai/GOT-OCR2_0` | 6.4 GB / — | Эксп. 5b (td-legacy) |
| `google/gemma-4-12b-it`, `google/gemma-4-E4B-it` | 23 / 15 GB | Эксп. 11 |
| `Qwen/Qwen3.5-9B-Base`, `empero-ai/Qwythos-9B-Claude-Mythos-5-1M` | 18 GB / — | Эксп. 9-10 (negative results) |
| `opendatalab/MinerU2.5-Pro-2604-1.2B` | — | Эксп. 5c (не заработала через vanilla transformers) |
| `microsoft/trocr-large-printed`, `microsoft/trocr-large-handwritten` | — | Эксп. 1-4 |

Классические OCR-веса (Surya 0.14 det/rec, PaddleOCR PP-OCRv5/v6 + `PP-LCNet_x1_0_doc_ori`,
EasyOCR craft+cyrillic_g2, docTR db_resnet50+crnn_vgg16_bn) качаются автоматически:

```bash
export TD_PREF=ocr && source env.sh
python -m src.utils.download_models
```

Merged production-модель (`data/models/production/qwen3vl_lora_p5j_merged`, 17 GB) не хранится —
пересобирается из адаптера: `experiments/exp06_lora/merge_lora.py`.

---

## 5. Восстановление production-пайплайна распознавания (с нуля до JSON)

Production-пайплайн = **препроцессинг v3 -> Qwen3-VL-8B-Instruct + LoRA `p5j` (prompt mode A) ->
parse_response -> (опционально) валидатор v2**. Ниже — минимальный путь на чистой машине с одним
GPU (>= 24 GB VRAM для peft-варианта, ~21 GB; merged + batch=8 — тоже укладывается в 80 GB H100).

### Шаг 0. Клонировать и собрать адаптер

```bash
git clone https://github.com/Siziff/binrun.git TD && cd TD
bash experiments/exp06_lora/adapters/unpack_adapters.sh p5j_r32_lr7e5_ep5_a256
# -> experiments/exp06_lora/adapters/p5j_r32_lr7e5_ep5_a256/adapter_model.safetensors (166 MB, sha256 проверен)
```

### Шаг 1. Окружение `td-vlm` (inference + LoRA)

```bash
python3.11 -m venv data/envs/td-vlm
data/envs/td-vlm/bin/pip install -r envs/requirements-td-vlm.txt    # torch 2.5.1+cu121, transformers 4.57.6, peft 0.19.1
# flash-attn 2.7.4.post1: поставить prebuilt wheel под cu12/torch2.5/cp311, или пропустить и запускать с --no-flash-attn (SDPA, медленнее ~1.5x)
export TD_PREF=vlm && source env.sh      # выставит HF_HOME=data/models/hf_cache, TMPDIR, PYTHONNOUSERSITE и т.д.
```

Если нужен ещё и препроцессинг на этой же машине — отдельно `td-ocr`
(`envs/requirements-td-ocr.txt`: paddlex/paddleocr, python-doctr, transformers 4.49 для Grounding DINO).
Окружения конфликтуют по версии transformers, поэтому держатся раздельно.

### Шаг 2. Базовые веса

```bash
hf download Qwen/Qwen3-VL-8B-Instruct          # 17 GB — база для LoRA
hf download IDEA-Research/grounding-dino-base  # 0.9 GB — только для препроцессинга v3
# PaddleX PP-LCNet_x1_0_doc_ori и docTR db_resnet50 подтянутся сами при первом запуске preprocess_v3 (кэш в data/models/*)
```

### Шаг 3. Изображения -> препроцессинг v3

Положить фотографии этикеток в любую папку (например `Smuts/Types/`, см. раздел 2.1) и прогнать:

```bash
export TD_PREF=ocr && source env.sh
python -m src.preprocessing.preprocess_v3 \
    --src-dir Smuts/Types --dst-dir data/preprocessed_v3/Smuts \
    --index-path data/preprocessed_v3/preproc_index_v3.json --page-conf-threshold 0.92
```

На выходе — JPEG с исправленной ориентацией и `max_side=3000`. Модель обучалась именно на таких
изображениях; подавать сырые кадры без поворота можно, но F1 будет ниже (см. Эксп. 1 vs 2 в RESULTS.md).
При обучении processor Qwen3-VL ограничивал картинку `min_pixels=200704 … max_pixels=802816`
(~900x900 px); в inference-скриптах используется дефолтный processor базовой модели — так были получены
все production-метрики.

### Шаг 4. Файл входов (JSONL)

Одна строка на изображение; `image_path` — абсолютный или относительный путь к препроцессированному файлу.
Пример — `experiments/exp17_anthracoidea/anthracoidea_inputs.jsonl`:

```bash
python - <<'EOF'
import json, pathlib
d = pathlib.Path("data/preprocessed_v3/Smuts")
with open("my_inputs.jsonl", "w") as f:
    for p in sorted(d.glob("*.JPG")):
        f.write(json.dumps({"image": p.name, "image_path": str(p.resolve())}, ensure_ascii=False) + "\n")
EOF
```

(Поле `gt` не обязательно; если его нет, скрипты просто не считают метрики.)

### Шаг 5. Inference

Вариант A — peft-адаптер поверх базы (F1 0.972, ~14 с/img на H100):

```bash
export TD_PREF=vlm && source env.sh
python experiments/exp06_lora/eval_full394.py \
    --adapter experiments/exp06_lora/adapters/p5j_r32_lr7e5_ep5_a256 \
    --inputs my_inputs.jsonl \
    --out outputs/my_preds.jsonl            # [--no-flash-attn] [--max-new-tokens 2048] [--limit N]
```

Вариант B — merged-модель + батчинг (F1 0.9716, ~2.8 с/img при batch=8, Эксп. 14):

```bash
python experiments/exp06_lora/merge_lora.py \
    --adapter experiments/exp06_lora/adapters/p5j_r32_lr7e5_ep5_a256 \
    --output  data/models/production/qwen3vl_lora_p5j_merged          # один раз, 17 GB
python experiments/exp06_lora/eval_full394_batched.py \
    --model-path data/models/production/qwen3vl_lora_p5j_merged \
    --inputs my_inputs.jsonl --out outputs/my_preds_batched.jsonl --batch-size 8
```

Оба скрипта идемпотентны (`--no-resume`, чтобы пересчитать) и дописывают результат построчно.

Что происходит внутри (если нужно встроить в свой код):

```python
# промпт: experiments/exp05_vlm_normalizer/prompts/{system.md,builder.py}, mode="A" (vision-only, zero-shot);
# схема 21 поля: experiments/exp05_vlm_normalizer/field_schema.json
from experiments.exp05_vlm_normalizer.prompts.builder import build_prompt
from experiments.exp05_vlm_normalizer.run_qwen import parse_response   # <think>-strip, ```json-fence, самый большой {...}, умные кавычки

prompt = build_prompt(mode="A")
messages = [{"role": "system", "content": [{"type": "text", "text": prompt["system"]}]},
            {"role": "user", "content": [... prompt["user_messages"] с подстановкой image_path ...]}]
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=2048, do_sample=False)     # greedy
parsed, err = parse_response(processor.batch_decode(out[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0])
```

### Шаг 6. Формат результата

Каждая строка выходного JSONL:

```json
{"image": "LE F-198013.JPG", "model": "...", "mode": "A",
 "raw_response": "```json\n{...}\n```",
 "parsed_json": {"catalog_number": "LE F-198013", "scientific_name": "...", "...": "...", "exsiccata_number": null},
 "parse_ok": true, "parse_error": "", "runtime_seconds": 13.9, "n_input_tokens": 1312, "n_output_tokens": 280}
```

`parsed_json` — словарь ровно с 21 ключом схемы (раздел 2.2), отсутствующие значения = `null`.
Сводка по заполняемости полей (fill rate, parse rate, тайминги) для коллекции без GT:
`python experiments/exp17_anthracoidea/summarize.py --in outputs/my_preds.jsonl --out-prefix outputs/my_summary`
(пример — `experiments/exp17_anthracoidea/summary.md`).

### Шаг 7 (опционально). Валидатор справочниками

```bash
python experiments/exp15_validator/validate_v2.py \
    --input outputs/my_preds.jsonl --output outputs/my_preds_validated.jsonl --apply-corrections
```

Проверяет `scientific_name` через GBIF (Fungi) с fallback на Index Fungorum, `country` через REST
Countries, `verbatim_substrate` (растение-хозяин) через GBIF Plantae; добавляет флаги/кандидаты
исправлений, с `--apply-corrections` — переписывает поля. Нужен доступ в интернет; без ключей.

### Шаг 8 (опционально). Метрики против GT

Если во входах есть `gt` (как в `experiments/exp05_vlm_normalizer/merged_inputs.jsonl` для 394 Types):

```bash
python experiments/exp06_lora/per_field_analysis.py --evals outputs/my_preds.jsonl   # per-field F1/CER против data/gold
python unified_eval.py                                                                 # сводная таблица по всем experiments/*/outputs
```

### Переобучение адаптера (если нужно)

```bash
export TD_PREF=vlm && source env.sh
python experiments/exp06_lora/sweep_train.py \
    --output-dir experiments/exp06_lora/sweep/checkpoints/p5j_retrain \
    --lora-rank 32 --lora-alpha 256 --lr 7e-5 --epochs 5 --lora-dropout 0.05 \
    --target-modules all-linear --grad-accum 8 --warmup-ratio 0.1 --max-pixels 802816
# train split: experiments/exp06_lora/splits/train.jsonl (310 изобр. из data/preprocessed_v3/Smuts), ~24 мин на 1x H100
```

Известные грабли при восстановлении (подробнее — «Гайдлайны» в RESULTS.md):
- при обучении `--max-pixels 802816` обязателен, иначе native-разрешение даёт ~6.5k image-токенов и OOM на batch=1;
- при обучении после `get_peft_model()` нужен `model.enable_input_require_grads()` (уже в `sweep_train.py`);
- `td-vllm`/`td-sglang` **не** нужны для production — только для бенчмарков сторонних OCR-VLM;
- если flash-attn не собирается — все скрипты имеют `--no-flash-attn` и автоматический fallback на SDPA.

---

## 6. Как воспроизвести остальные эксперименты

```bash
# 0. данные + препроцессинг v3 (раздел 2)

# 1. OCR baseline (Эксп. 1-4); unified_eval.py пересчитывает summary_all.{csv,md} по всем outputs
export TD_PREF=ocr && source env.sh
python experiments/exp01_ocr_baseline/run.py --engines tesseract,paddleocr,easyocr,doctr,surya
python unified_eval.py

# 2. VLM zero/few-shot (Эксп. 5-6)
export TD_PREF=vlm && source env.sh
python experiments/exp05_vlm_normalizer/run_qwen.py    --model Qwen/Qwen3-VL-8B-Instruct --mode A
python experiments/exp05_vlm_normalizer/run_fewshot.py --model Qwen/Qwen3-VL-8B-Instruct --mode A_fs

# 3. LoRA: sweep_runner читает JSON-конфиг и вызывает sweep_train.py + sweep_eval.py
bash   experiments/exp06_lora/sweep_runner.sh phase5 0 experiments/exp06_lora/sweep/configs/p5j_r32_lr7e5_ep5_a256.json
# или вручную (аргументы = поля конфига):
python experiments/exp06_lora/sweep_train.py --output-dir experiments/exp06_lora/sweep/checkpoints/p5j \
       --lora-rank 32 --lora-alpha 256 --lr 7e-5 --epochs 5 --lora-dropout 0.05 --target-modules all-linear --grad-accum 8
python experiments/exp06_lora/sweep_eval.py   --adapter experiments/exp06_lora/adapters/p5j_r32_lr7e5_ep5_a256
python experiments/exp06_lora/eval_full394.py --adapter experiments/exp06_lora/adapters/p5j_r32_lr7e5_ep5_a256
python experiments/exp06_lora/sweep_summary.py

# 4. Merge + batched inference (Эксп. 14), ~2.8 с/img
python experiments/exp06_lora/merge_lora.py --adapter experiments/exp06_lora/adapters/p5j_r32_lr7e5_ep5_a256 \
       --output data/models/production/qwen3vl_lora_p5j_merged
python experiments/exp06_lora/eval_merged_batched.py --model-path data/models/production/qwen3vl_lora_p5j_merged --batch-size 8

# 5. Валидатор (Эксп. 15) и GT-аудит (Эксп. 16)
python experiments/exp15_validator/validate_v2.py --input <preds.jsonl> --output <validated.jsonl>
python experiments/exp16_gt_audit/cross_validate_halluc.py --evals experiments/exp06_lora/sweep/eval_full394/*.jsonl --output-dir experiments/exp16_gt_audit

# 6. Новая подколлекция (Эксп. 17)
bash experiments/exp17_anthracoidea/preprocess.sh && bash experiments/exp17_anthracoidea/run_inference.sh
```

Точные аргументы, тайминги и известные грабли (flash-attn, `max_pixels`, `enable_input_require_grads`,
несовместимости transformers между окружениями) — в разделах "Гайдлайны" и в описаниях
экспериментов `RESULTS.md`.
