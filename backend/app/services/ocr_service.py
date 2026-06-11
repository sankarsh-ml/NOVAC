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
    """
    Light cleanup only.
    Keep this generic because uploaded documents may be Aadhaar, PAN,
    passport, certificates, invoices, IDs, forms, etc.
    """

    text = str(text or "")

    replacements = [
        (r"\bD0B\b", "DOB"),
        (r"\bD O B\b", "DOB"),
        (r"\bD\.O\.B\b", "DOB"),
        (r"\bAadhar\b", "Aadhaar"),
        (r"\bAdhaar\b", "Aadhaar"),
        (r"\bYear\s+0f\s+Birth\b", "Year of Birth"),
        (r"\bV1D\b", "VID"),
        (r"\bMaIe\b", "Male"),
        (r"\bFemaIe\b", "Female"),
        (r"\bGovt\b", "Government"),
    ]

    for pattern, replacement in replacements:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE
        )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()


def _bbox_to_xywh(bbox):
    """
    Converts PaddleOCR 4-point bbox to simple x/y/w/h.
    Keeps original bbox too, but this helps downstream services if needed.
    """

    try:
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]

        x = int(min(xs))
        y = int(min(ys))
        w = int(max(xs) - min(xs))
        h = int(max(ys) - min(ys))

        return {
            "x": x,
            "y": y,
            "w": w,
            "h": h
        }

    except Exception:
        return None


def _parse_ocr_result(result):
    """
    Normalizes PaddleOCR output into NOVAC's expected schema:
    {
        text,
        avg_confidence,
        lines
    }
    """

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

        try:
            bbox = line[0]
            raw_text = line[1][0]
            confidence = float(line[1][1])
        except Exception:
            continue

        text = _normalize_text(raw_text)

        if not text:
            continue

        texts.append(text)
        confidences.append(confidence)

        line_item = {
            "text": text,
            "confidence": round(confidence, 3),
            "bbox": bbox,
        }

        xywh = _bbox_to_xywh(bbox)
        if xywh:
            line_item["region"] = xywh

        line_results.append(line_item)

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
    """
    Scores OCR result quality in a general way.

    Priority:
    - confidence
    - useful line count
    - useful text length
    - document-like keywords
    - dates, ID numbers, amounts
    - low garbage characters

    Aadhaar is only a small optional bonus, not the main decision factor.
    """

    text = result.get(
        "text",
        ""
    )
    normalized = text.lower()
    lines = result.get(
        "lines",
        []
    )

    general_document_keywords = [
        "name",
        "date",
        "dob",
        "birth",
        "address",
        "id",
        "number",
        "certificate",
        "registration",
        "license",
        "licence",
        "passport",
        "invoice",
        "receipt",
        "total",
        "amount",
        "government",
        "authority",
        "department",
        "issued",
        "valid",
        "expiry",
        "signature",
        "father",
        "mother",
        "gender",
        "male",
        "female",
        "nationality",
        "pan",
        "tax",
        "account",
        "bank",
        "student",
        "roll",
        "university",
        "college",
        "school"
    ]

    keyword_hits = sum(
        1
        for keyword in general_document_keywords
        if keyword in normalized
    )

    id_like_pattern_hits = len(
        re.findall(
            r"\b[A-Z0-9]{2,}[-/ ]?[A-Z0-9]{2,}[-/ ]?[A-Z0-9]{2,}\b",
            text,
            flags=re.IGNORECASE
        )
    )

    date_pattern_hits = len(
        re.findall(
            r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b|\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b",
            text
        )
    )

    money_pattern_hits = len(
        re.findall(
            r"(?:₹|rs\.?|inr|\$|usd)?\s?\d+(?:,\d{3})*(?:\.\d{1,2})?",
            text,
            flags=re.IGNORECASE
        )
    )

    aadhaar_optional_hits = len(
        re.findall(
            r"\b(?:\d{4}[\s-]?){2}\d{4}\b|\b(?:x{4}|X{4})[\s-]?(?:x{4}|X{4})[\s-]?(?:\d{4}|x{4}|X{4})\b",
            text
        )
    )

    garbage_chars = len(
        re.findall(
            r"[^A-Za-z0-9\s:/.,()\-₹$#@&]",
            text
        )
    )

    garbage_penalty = min(
        garbage_chars * 2,
        30
    )

    useful_text_length = len(
        re.sub(
            r"\s+",
            "",
            text
        )
    )

    return (
        result.get("avg_confidence", 0) * 100
        + min(len(lines), 30) * 2
        + min(useful_text_length, 800) * 0.025
        + keyword_hits * 3
        + id_like_pattern_hits * 4
        + date_pattern_hits * 5
        + money_pattern_hits * 2
        + aadhaar_optional_hits * 4
        - garbage_penalty
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
    """
    Creates OCR preprocessing variants.

    Original image is handled separately in extract_text().
    These variants are only used when original OCR is weak.
    """

    image = cv2.imread(image_path)

    if image is None:
        return []

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    variants = []

    # 2x upscale usually helps small document text.
    upscaled = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )
    variants.append(
        _write_variant(
            image_path,
            "upscaled_2x",
            upscaled
        )
    )

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

    clahe_upscaled = cv2.resize(
        enhanced,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )
    variants.append(
        _write_variant(
            image_path,
            "clahe_upscaled_2x",
            clahe_upscaled
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

    sharpened_upscaled = cv2.resize(
        sharpened,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )
    variants.append(
        _write_variant(
            image_path,
            "sharpened_upscaled_2x",
            sharpened_upscaled
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


def _is_weak_ocr(result):
    """
    Decides whether to try extra OCR variants.
    """

    return (
        result.get("avg_confidence", 0) < 0.86
        or len(result.get("lines", [])) < 4
        or len(result.get("text", "")) < 40
    )


def extract_text(image_path):
    """
    Main OCR entrypoint used by NOVAC.

    Keeps required output keys:
    - text
    - lines
    - avg_confidence

    Adds safe optional debug keys:
    - ocr_engine
    - ocr_variant
    - ocr_candidates_tested
    - ocr_warning
    """

    candidates_tested = 1

    try:
        original_result = _run_ocr(
            image_path
        )
    except Exception as exc:
        return {
            "text": "",
            "avg_confidence": 0,
            "lines": [],
            "ocr_engine": "paddleocr",
            "ocr_variant": "original",
            "ocr_candidates_tested": 0,
            "ocr_warning": f"OCR failed: {str(exc)}"
        }

    best_result = {
        **original_result,
        "ocr_variant": "original"
    }

    if _is_weak_ocr(best_result):
        for variant_path in _variant_paths(image_path):
            try:
                candidates_tested += 1

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

    best_result["ocr_engine"] = "paddleocr"
    best_result["ocr_candidates_tested"] = candidates_tested

    best_result["ocr_warning"] = (
        "OCR result may be weak"
        if best_result.get("avg_confidence", 0) < 0.70
        or len(best_result.get("text", "")) < 25
        else None
    )

    return best_result