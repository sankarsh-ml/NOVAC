import sys
import json
from pathlib import Path

from ocr_service import extract_text


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python backend/test_ocr.py path\\to\\image.png")
        print("  python backend/test_ocr.py backend\\uploads\\sample.png")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    print("=" * 70)
    print("NOVAC OCR TEST")
    print("=" * 70)
    print(f"Input file: {image_path}")
    print()

    result = extract_text(str(image_path))

    print("OCR SUMMARY")
    print("-" * 70)
    print(f"Engine: {result.get('ocr_engine')}")
    print(f"Best variant: {result.get('ocr_variant')}")
    print(f"Candidates tested: {result.get('ocr_candidates_tested')}")
    print(f"Average confidence: {result.get('avg_confidence')}")
    print(f"Line count: {len(result.get('lines', []))}")
    print(f"Warning: {result.get('ocr_warning')}")
    print()

    print("EXTRACTED TEXT")
    print("-" * 70)
    text = result.get("text") or ""
    print(text if text else "[No text extracted]")
    print()

    print("OCR LINES")
    print("-" * 70)

    lines = result.get("lines", [])

    if not lines:
        print("[No OCR lines found]")
    else:
        for index, line in enumerate(lines, start=1):
            print(f"{index}. {line.get('text')}")
            print(f"   confidence: {line.get('confidence')}")
            if line.get("region"):
                print(f"   region: {line.get('region')}")
            print()

    output_path = image_path.with_name(f"{image_path.stem}_ocr_result.json")

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(f"Full OCR JSON saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()