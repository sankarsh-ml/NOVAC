import re

import cv2
import numpy as np

from app.services.visual_region_utils import any_region_near


KEY_FIELD_PATTERNS = [
    ("dob", r"\b(dob|date of birth|year of birth|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"),
    ("aadhaar_number", r"\b\d{4}\s+\d{4}\s+\d{4}\b"),
    ("vid", r"\b\d{4}\s+\d{4}\s+\d{4}\s+\d{4}\b|\bvid\b"),
    ("gender", r"\b(male|female|other|gender|sex)\b"),
    ("name", r"\b(name|s/o|d/o|father|husband)\b")
]


def _rect_from_bbox(bbox):

    if not bbox or len(bbox) != 4:
        return None

    xs = [point[0] for point in bbox]
    ys = [point[1] for point in bbox]

    x = int(min(xs))
    y = int(min(ys))
    w = int(max(xs) - min(xs))
    h = int(max(ys) - min(ys))

    if w <= 0 or h <= 0:
        return None

    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h
    }


def _bounded(rect, width, height):

    x = max(
        0,
        min(rect["x"], width - 1)
    )
    y = max(
        0,
        min(rect["y"], height - 1)
    )
    w = max(
        1,
        min(rect["w"], width - x)
    )
    h = max(
        1,
        min(rect["h"], height - y)
    )

    return x, y, w, h


def _field_type(text):

    normalized = str(text or "").lower()

    for field, pattern in KEY_FIELD_PATTERNS:
        if re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE
        ):
            return field

    return "unknown"


def _is_instruction_text(text):

    normalized = str(text or "").strip().lower()

    return bool(
        re.search(
            r"\b(government|aadhaar|identity|proof|instruction|valid|issued|authority|unique identification|help|www|uidai)\b",
            normalized
        )
    )


def _is_id_like(text):

    normalized = str(text or "").strip()
    digits = len(re.findall(r"\d", normalized))

    return digits >= 8 or bool(
        re.fullmatch(
            r"[\d\s/-]{6,}",
            normalized
        )
    )


def _metrics(image, gray, rect):

    height, width = gray.shape[:2]
    x, y, w, h = _bounded(
        rect,
        width,
        height
    )
    pad = max(
        2,
        int(h * 0.15)
    )
    y1 = max(0, y - pad)
    y2 = min(height, y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(width, x + w + pad)

    roi_gray = gray[
        y1:y2,
        x1:x2
    ]
    roi_color = image[
        y1:y2,
        x1:x2
    ]

    if roi_gray.size == 0:
        return None

    edges = cv2.Canny(
        roi_gray,
        60,
        160
    )

    dark_pixels = roi_gray[
        roi_gray < np.percentile(roi_gray, 45)
    ]
    bright_pixels = roi_gray[
        roi_gray > np.percentile(roi_gray, 65)
    ]
    hsv = cv2.cvtColor(
        roi_color,
        cv2.COLOR_BGR2HSV
    )

    _, binary = cv2.threshold(
        roi_gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    distance = cv2.distanceTransform(
        binary,
        cv2.DIST_L2,
        3
    )
    stroke = float(
        np.mean(distance[binary > 0])
        if np.any(binary > 0)
        else 0
    )

    return {
        "height": float(h),
        "edge_density": float(np.mean(edges > 0)),
        "sharpness": float(cv2.Laplacian(roi_gray, cv2.CV_64F).var()),
        "background_brightness": float(np.mean(bright_pixels) if bright_pixels.size else np.mean(roi_gray)),
        "text_darkness": float(np.mean(dark_pixels) if dark_pixels.size else np.mean(roi_gray)),
        "contrast": float(np.std(roi_gray)),
        "stroke": stroke,
        "saturation": float(np.mean(hsv[:, :, 1]))
    }


def _robust_z(value, values):

    if not values:
        return 0

    median = float(np.median(values))
    mad = float(
        np.median(
            np.abs(
                np.array(values, dtype=float)
                - median
            )
        )
    )

    if mad < 0.001:
        return 0

    return abs(value - median) / (1.4826 * mad)


def analyze_text_consistency(
    image_path,
    ocr_lines,
    extracted_text=None,
    visual_regions=None
) -> dict:

    image = cv2.imread(image_path)

    if image is None:
        return {
            "font_mismatch_detected": False,
            "field_mismatch_score": 0,
            "suspicious_fields": [],
            "suspicious_regions": [],
            "comparisons_used": 0,
            "comparisons_skipped": 0,
            "reasons": [],
            "error": f"Cannot read image: {image_path}"
        }

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )
    height, width = gray.shape[:2]
    image_area = float(width * height) if width and height else 1.0
    visual_regions = visual_regions or []

    analyzed = []
    comparisons_skipped = 0

    for line in ocr_lines or []:
        text = str(
            line.get("text", "")
        ).strip()
        rect = _rect_from_bbox(
            line.get("bbox")
        )

        if not text or not rect:
            comparisons_skipped += 1
            continue

        confidence = float(line.get("confidence", 0))

        if confidence < 0.62:
            comparisons_skipped += 1
            continue

        area_ratio = (rect["w"] * rect["h"]) / image_area

        if area_ratio < 0.0001 or area_ratio > 0.08:
            comparisons_skipped += 1
            continue

        if rect["y"] < height * 0.08 or rect["y"] + rect["h"] > height * 0.94:
            comparisons_skipped += 1
            continue

        if _is_instruction_text(text):
            comparisons_skipped += 1
            continue

        metrics = _metrics(
            image,
            gray,
            rect
        )

        if not metrics:
            comparisons_skipped += 1
            continue

        if metrics["saturation"] > 75:
            comparisons_skipped += 1
            continue

        analyzed.append({
            "text": text,
            "field": _field_type(text),
            "rect": rect,
            "confidence": confidence,
            "metrics": metrics
        })

    if len(analyzed) < 3:
        return {
            "font_mismatch_detected": False,
            "field_mismatch_score": 0,
            "suspicious_fields": [],
            "suspicious_regions": [],
            "comparisons_used": 0,
            "comparisons_skipped": comparisons_skipped,
            "reasons": []
        }

    heights = [
        item["rect"]["h"]
        for item in analyzed
        if item["rect"]["h"] > 0
    ]
    median_height = float(np.median(heights)) if heights else 20.0

    def center(rect):
        return (
            rect["x"] + rect["w"] / 2,
            rect["y"] + rect["h"] / 2
        )

    def same_script(a, b):
        a_alpha = bool(re.search(r"[A-Za-z]", a))
        b_alpha = bool(re.search(r"[A-Za-z]", b))
        a_digit = bool(re.search(r"\d", a))
        b_digit = bool(re.search(r"\d", b))

        if a_alpha != b_alpha and not (a_digit and b_digit):
            return False

        return True

    def local_references(item):
        refs = []
        rect = item["rect"]
        cx, cy = center(rect)

        for candidate in analyzed:
            if candidate is item:
                continue

            if candidate["confidence"] < 0.72:
                comparisons_skipped_local[0] += 1
                continue

            if not same_script(item["text"], candidate["text"]):
                comparisons_skipped_local[0] += 1
                continue

            if _is_id_like(item["text"]) != _is_id_like(candidate["text"]):
                comparisons_skipped_local[0] += 1
                continue

            other = candidate["rect"]
            ox, oy = center(other)
            vertical_distance = abs(cy - oy)
            horizontal_distance = abs(cx - ox)
            row_close = vertical_distance <= median_height * 1.8
            adjacent_row = vertical_distance <= median_height * 2.4 and horizontal_distance <= max(width * 0.32, 220)
            same_block = (
                row_close
                or adjacent_row
            )

            if not same_block:
                comparisons_skipped_local[0] += 1
                continue

            size_ratio = max(rect["h"], other["h"]) / float(max(min(rect["h"], other["h"]), 1))

            if size_ratio > 1.35:
                comparisons_skipped_local[0] += 1
                continue

            distance = float(
                np.hypot(
                    cx - ox,
                    cy - oy
                )
            )
            refs.append(
                (
                    distance,
                    candidate
                )
            )

        return [
            candidate
            for _, candidate in sorted(refs)[:5]
        ]

    suspicious_fields = []
    suspicious_regions = []
    comparisons_used = 0
    comparisons_skipped_local = [comparisons_skipped]

    for item in analyzed:
        if item["field"] == "unknown":
            continue

        references = local_references(item)

        if len(references) < 2:
            comparisons_skipped_local[0] += 1
            continue

        metrics = item["metrics"]
        evidence = []
        score = 0
        reference_metrics = {
            key: [
                reference["metrics"][key]
                for reference in references
            ]
            for key in [
                "height",
                "edge_density",
                "sharpness",
                "background_brightness",
                "text_darkness",
                "contrast",
                "stroke"
            ]
        }

        checks = [
            ("height", 2.6, 8, "local text height differs"),
            ("edge_density", 2.8, 8, "local edge density differs"),
            ("sharpness", 3.0, 9, "local sharpness differs"),
            ("background_brightness", 3.0, 7, "local background differs"),
            ("text_darkness", 3.0, 7, "local text darkness differs"),
            ("contrast", 3.0, 6, "local contrast differs"),
            ("stroke", 3.0, 8, "local stroke thickness differs")
        ]

        for key, threshold, weight, label in checks:
            z_score = _robust_z(
                metrics[key],
                reference_metrics[key]
            )

            if z_score >= threshold:
                score += weight
                evidence.append(label)

        comparisons_used += len(references)

        if score < 22 or len(evidence) < 2:
            continue

        rect = item["rect"]
        region = {
            "x": rect["x"],
            "y": rect["y"],
            "w": rect["w"],
            "h": rect["h"]
        }
        visual_support = any_region_near(
            region,
            visual_regions,
            image_shape=image.shape
        )

        if visual_support:
            score += 6

        if score < 32:
            continue

        reason = "Local field text style differs from nearby reference text"
        region = {
            "x": rect["x"],
            "y": rect["y"],
            "w": rect["w"],
            "h": rect["h"],
            "type": "text_consistency",
            "source": "Text Consistency",
            "reason": reason,
            "scoring_eligible": True,
            "annotation_eligible": True,
            "near_visual_evidence": visual_support,
            "source": "TextMismatch",
            "area_ratio": round(
                (rect["w"] * rect["h"]) / image_area,
                5
            ),
            "overlaps_qr": False,
            "overlaps_photo": False,
            "overlaps_logo": False,
            "overlaps_dense_text": False,
            "overlaps_header_footer": False,
            "overlaps_damage_or_fold": False,
            "overlaps_editable_field": True,
            "editable_field_name": item["field"],
            "suppression_reason": None
        }
        nearest_distance = min(
            float(
                np.hypot(
                    center(item["rect"])[0] - center(reference["rect"])[0],
                    center(item["rect"])[1] - center(reference["rect"])[1]
                )
            )
            for reference in references
        )

        suspicious_fields.append({
            "field": item["field"],
            "value": item["text"],
            "reason": reason,
            "score": int(min(score, 50)),
            "region": region,
            "nearby_reference_count": len(references),
            "comparison_distance": round(nearest_distance, 2),
            "evidence": evidence[:4]
        })
        suspicious_regions.append(region)

    total_score = int(
        min(
            sum(field["score"] for field in suspicious_fields),
            50
        )
    )

    reasons = [
        "Local field text style differs from nearby reference text"
    ] if suspicious_fields else []

    return {
        "font_mismatch_detected": bool(suspicious_fields),
        "field_mismatch_score": total_score,
        "suspicious_fields": suspicious_fields[:5],
        "suspicious_regions": suspicious_regions[:5],
        "comparisons_used": comparisons_used,
        "comparisons_skipped": comparisons_skipped_local[0],
        "reasons": reasons
    }
