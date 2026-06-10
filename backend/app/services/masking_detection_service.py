import re


MASKING_PATTERNS = [
    r"X{4,}",
    r"x{4,}",
    r"\*{4,}",
    r"#{4,}",
    r"\.{4,}",
    r"-{4,}",
]


def is_masked_text(text: str) -> bool:

    if not text:
        return False

    cleaned = text.strip()

    for pattern in MASKING_PATTERNS:
        if re.search(pattern, cleaned):
            return True

    compact = cleaned.replace(" ", "").upper()

    if len(compact) >= 4 and set(compact) == {"X"}:
        return True

    return False


def detect_masking(ocr_result):

    masked_regions = []

    for line in ocr_result.get("lines", []):

        text = line.get("text", "")

        if is_masked_text(text):

            masked_regions.append({
                "text": text,
                "bbox": line.get("bbox"),
                "confidence": line.get(
                    "confidence",
                    0
                )
            })

    return {
        "masking_detected": len(masked_regions) > 0,
        "masking_score": min(
            len(masked_regions) * 25,
            50
        ),
        "masked_field_count": len(masked_regions),
        "masked_regions": masked_regions,
        "reasons": (
            ["Masked fields detected in OCR"]
            if masked_regions
            else []
        )
    }
