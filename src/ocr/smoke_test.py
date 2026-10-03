"""Smoke-test: запускает все движки на одной картинке и печатает сводку."""
from __future__ import annotations

# ВАЖНО: env-shim первой строкой.
from src.utils import env_setup  # noqa: F401

import argparse
from pathlib import Path

from src.ocr import ALL_ENGINES, get_engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default=str(Path(__file__).resolve().parents[2] / "Smuts" / "LE F-113259.JPG"),
    )
    parser.add_argument("--engines", nargs="*", default=list(ALL_ENGINES))
    args = parser.parse_args()

    img = Path(args.image)
    print(f"Image: {img}\n")

    for name in args.engines:
        print(f"=== {name} ===", flush=True)
        try:
            eng = get_engine(name)
            doc = eng.run(img)
        except Exception as e:
            print(f"  CONSTRUCTOR FAILED: {type(e).__name__}: {e}\n")
            continue
        if doc.error:
            print(f"  ERROR: {doc.error.strip().splitlines()[-1]}")
        print(f"  runtime: {doc.runtime_seconds:.2f}s  lines: {len(doc.lines)}")
        print(f"  --- text ---")
        print(doc.full_text[:600])
        print()


if __name__ == "__main__":
    main()
