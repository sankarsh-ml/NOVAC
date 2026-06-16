import re
import logging


logger = logging.getLogger(__name__)


MASKED_IDENTIFIER_REGEXES = [
    re.compile(r"\b[xX*]{4}\s?[xX*]{4}\s?[xX*]{4}\b"),
    re.compile(r"\b[xX*]{8,12}\b"),
    re.compile(r"\b(?:X{4}|x{4})[-\s]?(?:X{4}|x{4})[-\s]?(?:X{4}|x{4})\b"),
]

MASKED_PHRASE_REGEX = re.compile(
    r"\b(?:masked|hidden)\s+(?:aadhaar|aadhar|identifier|id)\s+(?:number|field)\b",
    re.IGNORECASE
)

MASKING_PATTERNS = [
    r"X{4,}",
    r"x{4,}",
    r"\*{4,}",
    r"#{4,}",
    r"\.{4,}",
    r"-{4,}",
]


def is_masked_text(text: str) -> bool:

    return _masked_candidate(text) is not None


def detect_masking(ocr_result):

    candidates = []

    for line in ocr_result.get("lines", []):

        text = line.get("text", "")
        candidate = _masked_candidate(text)

        if candidate:
            region = line.get("region") or _region_from_bbox(
                line.get("bbox")
            )
            original_region = dict(region) if region else None
            source_image_width = (
                line.get("source_image_width")
                or ocr_result.get("source_image_width")
            )
            source_image_height = (
                line.get("source_image_height")
                or ocr_result.get("source_image_height")
            )

            candidates.append({
                "text": text,
                "masked_text": text,
                "bbox": line.get("bbox"),
                "region": region,
                "original_region": original_region,
                "x": region.get("x") if region else None,
                "y": region.get("y") if region else None,
                "w": region.get("w") if region else None,
                "h": region.get("h") if region else None,
                "type": "masking",
                "source": "ocr_mask_pattern",
                "source_detector": "masking",
                "label": "Masked field",
                "annotation_eligible": bool(region),
                "confidence": line.get(
                    "confidence",
                    0
                ),
                "reason": "Masked identifier pattern detected in OCR text",
                "mask_priority": candidate["priority"],
                "mask_score": candidate["score"],
                "mask_source": candidate["source"],
                "source_image_width": source_image_width,
                "source_image_height": source_image_height,
                "original_image_width": (
                    line.get("original_image_width")
                    or ocr_result.get("original_image_width")
                ),
                "original_image_height": (
                    line.get("original_image_height")
                    or ocr_result.get("original_image_height")
                ),
            })

    masked_regions = _select_primary_masked_regions(candidates)

    for region in masked_regions:
        if region.get("original_region"):
            logger.info(
                "Selected masked OCR region: %s",
                {
                    "masked_text": region.get("masked_text") or region.get("text"),
                    "ocr_region": region.get("original_region"),
                    "source_image_width": region.get("source_image_width"),
                    "source_image_height": region.get("source_image_height"),
                    "original_image_width": region.get("original_image_width"),
                    "original_image_height": region.get("original_image_height"),
                    "confidence": region.get("confidence"),
                    "source": "ocr_mask_pattern",
                }
            )

    return {
        "masking_detected": len(masked_regions) > 0,
        "masking_score": min(
            len(masked_regions) * 25,
            50
        ),
        "masked_field_count": len(masked_regions),
        "masked_regions": masked_regions,
        "reasons": (
            ["Masked identifier pattern detected in OCR text"]
            if masked_regions
            else []
        )
    }


def _masked_candidate(text):
    if not text:
        return None

    cleaned = re.sub(r"\s+", " ", str(text).strip())
    if not cleaned:
        return None

    compact = re.sub(r"[\s-]+", "", cleaned)
    mask_chars = len(re.findall(r"[xX*]", cleaned))
    alnum_chars = len(re.findall(r"[A-Za-z0-9]", cleaned))
    masked_ratio = mask_chars / max(alnum_chars, 1)

    if re.fullmatch(r"[xX*]{8,16}", compact):
        return {
            "priority": 0,
            "score": 100 + mask_chars,
            "source": "ocr_mask_pattern",
        }

    for pattern in MASKED_IDENTIFIER_REGEXES:
        if pattern.search(cleaned) and masked_ratio >= 0.7:
            return {
                "priority": 0,
                "score": 90 + mask_chars,
                "source": "ocr_mask_pattern",
            }

    if MASKED_PHRASE_REGEX.search(cleaned):
        return {
            "priority": 1,
            "score": 50,
            "source": "ocr_mask_phrase",
        }

    if re.fullmatch(r"[-–—]{3,16}", cleaned):
        return {
            "priority": 2,
            "score": 20,
            "source": "ocr_mask_dash",
        }

    return None


def _select_primary_masked_regions(candidates):
    if not candidates:
        return []

    with_region = [
        candidate
        for candidate in candidates
        if candidate.get("original_region")
    ]
    pool = with_region or candidates

    def sort_key(candidate):
        region = candidate.get("original_region") or {}
        return (
            candidate.get("mask_priority", 99),
            -candidate.get("mask_score", 0),
            -int(region.get("y", 0) or 0),
        )

    selected = sorted(pool, key=sort_key)[0]
    selected.pop("mask_priority", None)
    selected.pop("mask_score", None)
    return [selected]


def _region_from_bbox(bbox):
    try:
        if isinstance(bbox, dict):
            x = bbox.get("x", bbox.get("left", 0))
            y = bbox.get("y", bbox.get("top", 0))
            w = bbox.get("w", bbox.get("width", 0))
            h = bbox.get("h", bbox.get("height", 0))
            return {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h)
            }

        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            if all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in bbox):
                xs = [float(point[0]) for point in bbox]
                ys = [float(point[1]) for point in bbox]
                return {
                    "x": int(min(xs)),
                    "y": int(min(ys)),
                    "w": int(max(xs) - min(xs)),
                    "h": int(max(ys) - min(ys))
                }

            x, y, w, h = bbox
            return {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h)
            }

    except Exception:
        logger.debug("Unable to normalize masked field bbox", exc_info=True)

    return None
