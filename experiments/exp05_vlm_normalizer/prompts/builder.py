"""Сборка промптов для VLM-нормализатора.

Режимы:
  A    — vision-only (zero-shot)
  B    — vision + 3 OCR-fusion (zero-shot)
  A_fs — vision-only + 5 few-shot examples (с картинками)
  B_fs — vision + OCR-fusion + 5 few-shot examples
  F    — formatter (vision + 1 OCR text → JSON), zero-shot
  F_fs — formatter + 5 few-shot examples

GT никогда не передаётся в промпт целевого изображения.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Literal


HERE = Path(__file__).resolve().parent
SYSTEM_MD = (HERE / "system.md").read_text(encoding="utf-8")
SCHEMA_PATH = HERE.parent / "field_schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
FEW_SHOT_PATH = HERE.parent / "few_shot_examples.json"


# Skeleton JSON: 21 ключ → null. Демонстрирует требуемую структуру.
_SKELETON = {k: None for k in SCHEMA.keys()}

# Краткое описание схемы для модели.
def _schema_block() -> str:
    lines = []
    for k, meta in SCHEMA.items():
        kind = "verbatim" if meta.get("verbatim") else ("normalized" if meta.get("normalized") else "—")
        lines.append(f"- `{k}` ({kind}): {meta['desc']}")
    return "\n".join(lines)


SCHEMA_BLOCK = _schema_block()
SKELETON_JSON = json.dumps(_SKELETON, ensure_ascii=False, indent=2)


# Загружаем few-shot примеры (5 эталонных пар image+JSON).
def _load_few_shot() -> list[dict]:
    if not FEW_SHOT_PATH.exists():
        return []
    return json.loads(FEW_SHOT_PATH.read_text(encoding="utf-8"))


FEW_SHOT = _load_few_shot()
FEW_SHOT_IMAGE_NAMES = {ex["image"] for ex in FEW_SHOT}


def _user_intro() -> str:
    return dedent(f"""\
        Перед тобой скан гербарного листа из микологической коллекции LE.

        Извлеки информацию по схеме (21 поле). Ключи и порядок ровно такие:

        {SCHEMA_BLOCK}

        Формат ответа — ровно один JSON-объект ниже, с заполненными значениями.
        Если поле не читается — оставь null. Не добавляй ничего вне JSON.

        Заготовка структуры (заменишь null на значения, ничего не удалишь):
        ```
        {SKELETON_JSON}
        ```
    """)


def _ocr_block(ocr: dict, engines: tuple = ("paddleocr", "surya", "trocr-printed")) -> str:
    """Форматирует выводы OCR-движков в виде маркированных блоков."""
    parts = []
    for engine in engines:
        text = (ocr.get(engine) or {}).get("full_text") or ""
        text = text.strip()
        if len(text) > 4000:
            text = text[:4000] + "\n…[truncated]"
        parts.append(f"### OCR ({engine})\n```\n{text}\n```")
    return "\n\n".join(parts)


def _example_block_text(ex: dict) -> str:
    """Текстовая часть few-shot примера: JSON-ответ."""
    expected = json.dumps(ex["expected_json"], ensure_ascii=False, indent=2)
    return f"### Пример (правильный ответ):\n```json\n{expected}\n```"


def build_prompt(mode: Literal["A", "B", "A_fs", "B_fs", "F", "F_fs"],
                 ocr: dict | None = None,
                 single_ocr_text: str | None = None,
                 single_ocr_name: str = "ocr") -> dict:
    """Возвращает dict {"system", "user_messages", "mode"}.

    `user_messages` — список сегментов; каждый — либо
        {"type": "image", "image": <path>}
        либо {"type": "text", "text": "..."}.
    Это даёт возможность вставлять few-shot картинки с их ответами.

    mode:
        'A'    — vision-only zero-shot
        'B'    — vision + 3 OCR-fusion zero-shot
        'A_fs' — A + 5 few-shot (картинка + JSON × 5)
        'B_fs' — B + 5 few-shot
        'F'    — formatter (image + 1 ocr_text → JSON), zero-shot
        'F_fs' — F + 5 few-shot
    """
    is_fs = mode.endswith("_fs")
    base_mode = mode.replace("_fs", "")

    msgs: list[dict] = []

    # 1) Intro c описанием схемы.
    msgs.append({"type": "text", "text": _user_intro()})

    # 2) Если few-shot — кидаем все 5 примеров.
    if is_fs:
        msgs.append({"type": "text", "text":
            "\nНиже — пять примеров правильно заполненных этикеток разного типа. "
            "Используй их как образец стиля и формата (не копируй значения, читай "
            "свою картинку):\n"})
        for i, ex in enumerate(FEW_SHOT, 1):
            msgs.append({"type": "text", "text": f"\n--- Пример {i}/{len(FEW_SHOT)} ---"})
            msgs.append({"type": "image", "image": ex["image_path"]})
            msgs.append({"type": "text", "text": _example_block_text(ex)})

    # 3) Целевая картинка (всегда) + специфичные подсказки.
    msgs.append({"type": "text", "text":
        "\n\n=== Теперь твоя задача ===\n"
        "Изучи следующее изображение этикетки и заполни JSON."})

    if base_mode == "A":
        # картинка добавится в runner'е (передаётся image_path)
        msgs.append({"type": "_target_image_placeholder"})

    elif base_mode == "B":
        if ocr is None:
            raise ValueError("mode='B' requires ocr dict from 3 engines")
        msgs.append({"type": "_target_image_placeholder"})
        msgs.append({"type": "text", "text":
            "\nДополнительно — выводы трёх OCR-движков. Доверяй прежде всего "
            "ИЗОБРАЖЕНИЮ; OCR — подсказки.\n\n" + _ocr_block(ocr)})

    elif base_mode == "F":
        if single_ocr_text is None:
            raise ValueError("mode='F' requires single_ocr_text")
        msgs.append({"type": "_target_image_placeholder"})
        text = (single_ocr_text or "")[:6000]
        msgs.append({"type": "text", "text":
            f"\nДополнительно — вывод OCR-модели {single_ocr_name!r}. "
            f"Доверяй прежде всего изображению.\n\n"
            f"### OCR ({single_ocr_name})\n```\n{text}\n```"})
    else:
        raise ValueError(f"unknown base_mode {base_mode!r}")

    msgs.append({"type": "text", "text": "\nТеперь выдай JSON."})

    return {"system": SYSTEM_MD, "user_messages": msgs, "mode": mode}


# Обратная совместимость со старым API.
def build_legacy_prompt(mode: Literal["A", "B"], ocr: dict | None = None) -> dict:
    """Старая сигнатура: возвращает {system, user_text, mode} (без few-shot)."""
    if mode == "A":
        user = _user_intro() + "\n\nИзучи изображение этикетки и заполни JSON."
        return {"system": SYSTEM_MD, "user_text": user, "mode": "A"}

    if mode == "B":
        if ocr is None:
            raise ValueError("mode='B' requires ocr dict from 3 engines")
        user = (
            _user_intro()
            + "\n\nИзучи изображение этикетки и сверь его с выводами трёх независимых OCR-движков. "
              "Доверяй прежде всего ИЗОБРАЖЕНИЮ; OCR-выводы используй как подсказки / проверку, "
              "поскольку они могут содержать ошибки распознавания, перепутанные буквы и пропуски.\n\n"
            + _ocr_block(ocr)
            + "\n\nТеперь выдай JSON."
        )
        return {"system": SYSTEM_MD, "user_text": user, "mode": "B"}

    raise ValueError(f"unknown mode {mode!r}")
