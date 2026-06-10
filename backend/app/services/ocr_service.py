import os
import re
from pathlib import Path

import cv2
from paddleocr import PaddleOCR


ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)


OCR_VARIANT_DIR = Path("uploads") / "ocr_variants"
OCR_VARIANT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def _normalize_text(text):

    text = str(text or "")
    replacements = [
        (r"\bD0B\b", "DOB"),
        (r"\bD O B\b", "DOB"),
        (r"\bAadhar\b", "Aadhaar"),
        (r"\bAdhaar\b", "Aadhaar"),
        (r"\bYear\s+0f\s+Birth\b", "Year of Birth"),
        (r"\bV1D\b", "VID"),
        (r"\bMaIe\b", "Male"),
        (r"\bFemaIe\b", "Female")
    ]

    for pattern, replacement in replacements:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE
        )

    return re.sub(
        r"[ \t]+",
        " ",
        text
    ).strip()


def _parse_ocr_result(result):

    texts = []
    confidences = []
    line_results = []

    if not result or not result[0]:
        return {
            "text": "",
            "avg_confidence": 0,
            "lines": []
        }

    for line in result[0]:

        bbox = line[0]
        text = _normalize_text(
            line[1][0]
        )
        confidence = float(
            line[1][1]
        )

        if not text:
            continue

        texts.append(text)
        confidences.append(confidence)

        line_results.append({
            "text": text,
            "confidence": round(confidence, 3),
            "bbox": bbox,
        })

    avg_confidence = (
        sum(confidences) / len(confidences)
        if confidences else 0
    )

    return {
        "text": "\n".join(texts),
        "avg_confidence": round(avg_confidence, 3),
        "lines": line_results
    }


def _quality_score(result):

    text = result.get(
        "text",
        ""
    )
    normalized = text.lower()
    keyword_hits = sum(
        1
        for keyword in [
            "aadhaar",
            "dob",
            "date of birth",
            "year of birth",
            "vid",
            "male",
            "female",
            "government"
        ]
        if keyword in normalized
    )

    return (
        result.get("avg_confidence", 0) * 100
        + min(len(result.get("lines", [])), 20) * 2
        + min(len(text), 500) * 0.02
        + keyword_hits * 4
    )


def _write_variant(image_path, suffix, image):

    source = Path(image_path)
    output = OCR_VARIANT_DIR / f"{source.stem}_{suffix}.png"
    cv2.imwrite(
        str(output),
        image
    )
    return str(output)


def _variant_paths(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return []

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )
    variants = []

    variants.append(
        _write_variant(
            image_path,
            "gray",
            gray
        )
    )

    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        10,
        7,
        21
    )
    variants.append(
        _write_variant(
            image_path,
            "denoised",
            denoised
        )
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(
        gray
    )
    variants.append(
        _write_variant(
            image_path,
            "clahe",
            enhanced
        )
    )

    sharpened = cv2.addWeighted(
        gray,
        1.6,
        cv2.GaussianBlur(gray, (0, 0), 1.2),
        -0.6,
        0
    )
    variants.append(
        _write_variant(
            image_path,
            "sharpened",
            sharpened
        )
    )

    thresholded = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8
    )
    variants.append(
        _write_variant(
            image_path,
            "adaptive",
            thresholded
        )
    )

    return variants


def _run_ocr(image_path):

    result = ocr.ocr(
        image_path,
        cls=True
    )

    return _parse_ocr_result(
        result
    )


def extract_text(image_path):

    original_result = _run_ocr(
        image_path
    )
    best_result = {
        **original_result,
        "ocr_variant": "original"
    }

    should_try_variants = (
        best_result["avg_confidence"] < 0.86
        or len(best_result["lines"]) < 4
        or len(best_result["text"]) < 40
    )

    if should_try_variants:
        for variant_path in _variant_paths(image_path):
            try:
                variant_result = _run_ocr(
                    variant_path
                )
                variant_result["ocr_variant"] = os.path.basename(
                    variant_path
                )

                if _quality_score(variant_result) > _quality_score(best_result):
                    best_result = variant_result

            except Exception:
                continue

    best_result["text"] = "\n".join(
        _normalize_text(line)
        for line in best_result.get("text", "").splitlines()
        if _normalize_text(line)
    )

    return best_result
