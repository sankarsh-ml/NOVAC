import os
import re
import time
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

OCR_MAX_SIDE = 1800
OCR_MAX_CANDIDATES = 4
OCR_UPSCALE_SMALL_SIDE_THRESHOLD = 900


def _ocr_candidate_name(image_path):
    name = Path(image_path).stem

    for suffix in [
        "clahe_upscaled_2x",
        "sharpened_upscaled_2x",
        "upscaled_2x",
        "ocr_input",
        "sharpened",
        "denoised",
        "adaptive",
        "clahe",
        "gray"
    ]:
        if name.endswith(f"_{suffix}"):
            return suffix

    return name


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


def _resize_to_max_side(image, max_side=OCR_MAX_SIDE):
    height, width = image.shape[:2]
    largest_side = max(width, height)

    if largest_side <= max_side:
        return image

    scale = max_side / float(largest_side)

    return cv2.resize(
        image,
        (
            int(width * scale),
            int(height * scale)
        ),
        interpolation=cv2.INTER_AREA
    )


def _prepare_ocr_input(image_path):
    """
    Caps very large uploads before OCR to avoid repeated expensive inference.
    """

    image = cv2.imread(image_path)

    if image is None:
        return image_path

    resized = _resize_to_max_side(
        image
    )

    if resized.shape[:2] == image.shape[:2]:
        return image_path

    return _write_variant(
        image_path,
        "ocr_input",
        resized
    )


def _is_very_small_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return False

    height, width = image.shape[:2]

    return max(width, height) < OCR_UPSCALE_SMALL_SIDE_THRESHOLD


def _should_try_upscaled_variants(original_result, image_path):
    return (
        _is_very_small_image(image_path)
        or len(original_result.get("text", "")) < 15
        or original_result.get("avg_confidence", 0) < 0.55
    )


def _variant_paths(image_path, include_upscaled=False, limit=None):
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

    def has_capacity():
        return limit is None or len(variants) < limit

    def add_variant(suffix, variant_image):
        if not has_capacity():
            return False

        variants.append(
            _write_variant(
                image_path,
                suffix,
                variant_image
            )
        )

        return True

    if not add_variant(
        "gray",
        gray
    ):
        return variants

    if not has_capacity():
        return variants

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(
        gray
    )

    if not add_variant(
        "clahe",
        enhanced
    ):
        return variants

    if not has_capacity():
        return variants

    sharpened = cv2.addWeighted(
        gray,
        1.6,
        cv2.GaussianBlur(gray, (0, 0), 1.2),
        -0.6,
        0
    )

    if not add_variant(
        "sharpened",
        sharpened
    ):
        return variants

    if not has_capacity():
        return variants

    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        10,
        7,
        21
    )

    if not add_variant(
        "denoised",
        denoised
    ):
        return variants

    if not has_capacity():
        return variants

    thresholded = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8
    )

    if not add_variant(
        "adaptive",
        thresholded
    ):
        return variants

    if include_upscaled and has_capacity():
        upscaled = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        if not add_variant(
            "upscaled_2x",
            upscaled
        ):
            return variants

        if not has_capacity():
            return variants

        clahe_upscaled = cv2.resize(
            enhanced,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        if not add_variant(
            "clahe_upscaled_2x",
            clahe_upscaled
        ):
            return variants

        if not has_capacity():
            return variants

        sharpened_upscaled = cv2.resize(
            sharpened,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        add_variant(
            "sharpened_upscaled_2x",
            sharpened_upscaled
        )

    return variants


def _run_ocr(image_path, use_cls=False):

    result = ocr.ocr(
        image_path,
        cls=use_cls
    )

    return _parse_ocr_result(
        result
    )


def _run_timed_ocr(image_path, candidate_name, use_cls=False):
    started_at = time.perf_counter()

    try:
        return _run_ocr(
            image_path,
            use_cls=use_cls
        )

    finally:
        elapsed = time.perf_counter() - started_at
        print(f"OCR candidate {candidate_name} took {elapsed:.2f} seconds")


def _is_weak_ocr(result):
    """
    Decides whether to try extra OCR variants.
    """

    return not _is_good_enough_ocr(result)


def _is_good_enough_ocr(result):
    return (
        result.get("avg_confidence", 0) >= 0.75
        and len(result.get("lines", [])) >= 3
        and len(result.get("text", "")) >= 25
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
        ocr_input_path = _prepare_ocr_input(
            image_path
        )

        original_result = _run_timed_ocr(
            ocr_input_path,
            "original",
            use_cls=True
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

    if _is_good_enough_ocr(best_result):
        best_result["text"] = "\n".join(
            _normalize_text(line)
            for line in best_result.get("text", "").splitlines()
            if _normalize_text(line)
        )

        best_result["ocr_engine"] = "paddleocr"
        best_result["ocr_candidates_tested"] = candidates_tested
        best_result["ocr_warning"] = None

        return best_result

    if _is_weak_ocr(best_result):
        include_upscaled = _should_try_upscaled_variants(
            best_result,
            ocr_input_path
        )

        for variant_path in _variant_paths(
            ocr_input_path,
            include_upscaled=include_upscaled,
            limit=OCR_MAX_CANDIDATES - candidates_tested
        ):
            if candidates_tested >= OCR_MAX_CANDIDATES:
                break

            try:
                candidates_tested += 1
                candidate_name = _ocr_candidate_name(
                    variant_path
                )

                variant_result = _run_timed_ocr(
                    variant_path,
                    candidate_name,
                    use_cls=False
                )

                variant_result["ocr_variant"] = os.path.basename(
                    variant_path
                )

                if _quality_score(variant_result) > _quality_score(best_result):
                    best_result = variant_result

                if _is_good_enough_ocr(variant_result):
                    if not _is_good_enough_ocr(best_result):
                        best_result = variant_result

                    break

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
