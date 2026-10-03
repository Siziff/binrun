#!/usr/bin/env python3
"""Конвертер MD → PDF с разрывами страниц по разделам уровня H2.

Использование:
    python build_pdf.py                      # README.md → README.pdf (по умолчанию)
    python build_pdf.py FILE.md              # FILE.md → FILE.pdf
    python build_pdf.py FILE.md OUT.pdf      # явное имя выхода
"""
import re
import sys
from pathlib import Path
from markdown_pdf import MarkdownPdf, Section


def main() -> int:
    root = Path(__file__).parent
    args = sys.argv[1:]
    if not args:
        md_path = root / "README.md"
        pdf_path = root / "README.pdf"
    elif len(args) == 1:
        md_path = Path(args[0]).resolve()
        pdf_path = md_path.with_suffix(".pdf")
    else:
        md_path = Path(args[0]).resolve()
        pdf_path = Path(args[1]).resolve()

    text = md_path.read_text(encoding="utf-8")

    # CSS: A4-страница, читаемый шрифт, аккуратные таблицы, code-блоки без переноса.
    css = """
        @page { size: A4; margin: 2cm 2cm 2cm 2cm; }
        body  { font-family: "DejaVu Sans", "Liberation Sans", "Arial", sans-serif;
                font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; }
        h1    { font-size: 20pt; margin-top: 0; color: #11244e; }
        h2    { font-size: 15pt; margin-top: 1.4em; color: #11244e;
                border-bottom: 1px solid #cfd8dc; padding-bottom: 4px; }
        h3    { font-size: 12.5pt; margin-top: 1.2em; color: #1f3a68; }
        h4    { font-size: 11.5pt; color: #1f3a68; }
        p, li { text-align: justify; }
        ul, ol { margin-top: 0.25em; margin-bottom: 0.5em; }
        code  { font-family: "DejaVu Sans Mono", "Courier New", monospace;
                font-size: 9pt; background: #f3f4f6; padding: 1px 3px; border-radius: 3px; }
        pre   { font-family: "DejaVu Sans Mono", "Courier New", monospace;
                font-size: 8.5pt; line-height: 1.25; background: #f6f8fa;
                padding: 8px 10px; border-radius: 4px;
                white-space: pre; overflow: hidden; page-break-inside: avoid; }
        table { border-collapse: collapse; width: 100%; font-size: 9.5pt;
                page-break-inside: avoid; margin: 0.5em 0 1em 0; }
        th, td { border: 1px solid #cfd8dc; padding: 5px 7px; vertical-align: top; }
        th    { background: #eef2f7; }
        blockquote { border-left: 3px solid #cfd8dc; padding-left: 10px; color: #455a64; }
        a     { color: #11244e; text-decoration: none; }
        hr    { border: none; border-top: 1px solid #cfd8dc; margin: 1.2em 0; }
    """

    # Метаданные документа. Заголовок берём из первой H1, если есть.
    first_h1 = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
    doc_title = first_h1.group(1).strip() if first_h1 else md_path.stem

    pdf = MarkdownPdf(toc_level=2, optimize=True)
    pdf.meta["title"] = doc_title
    pdf.meta["author"] = "БИН РАН (LE) — исследовательский проект"
    pdf.meta["subject"] = "OCR + VLM + Tool-Using Agent for herbarium label metadata"

    # Разбиваем документ на секции по H2: каждая секция — новая страница,
    # что даёт чистые разрывы между логическими блоками.
    # `---` (горизонтальные линии в markdown) удаляем, т.к. они уже служили
    # разделителями секций, а в PDF их роль играет page-break.
    text = re.sub(r"^---\s*$", "", text, flags=re.MULTILINE)

    parts = re.split(r"(?m)^(?=## )", text)
    parts = [p.strip() for p in parts if p.strip()]

    for i, part in enumerate(parts):
        # Первая часть — title + первый H2; остальные — отдельные секции с разрывом.
        paginate = i > 0
        pdf.add_section(Section(part + "\n", paper_size="A4"), user_css=css)

    pdf.save(str(pdf_path))

    size_kb = pdf_path.stat().st_size / 1024
    print(f"PDF written: {pdf_path} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
