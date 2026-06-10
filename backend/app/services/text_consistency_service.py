import re

import cv2
import numpy as np


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

    if re.fullmatch(
        r"[A-Za-z .'-]{3,}",
        str(text or "").strip()
    ):
        return "name"

    return "unknown"


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
    extracted_text=None
) -> dict:

    image = cv2.imread(image_path)

    if image is None:
        return {
            "font_mismatch_detected": False,
            "field_mismatch_score": 0,
            "suspicious_fields": [],
            "suspicious_regions": [],
            "reasons": [],
            "error": f"Cannot read image: {image_path}"
        }

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )
    height, width = gray.shape[:2]
    image_area = float(width * height) if width and height else 1.0

    analyzed = []

    for line in ocr_lines or []:
        text = str(
            line.get("text", "")
        ).strip()
        rect = _rect_from_bbox(
            line.get("bbox")
        )

        if not text or not rect:
            continue

        area_ratio = (rect["w"] * rect["h"]) / image_area

        if area_ratio < 0.0001 or area_ratio > 0.08:
            continue

        metrics = _metrics(
            image,
            gray,
            rect
        )

        if not metrics:
            continue

        if metrics["saturation"] > 75:
            continue

        analyzed.append({
            "text": text,
            "field": _field_type(text),
            "rect": rect,
            "confidence": float(line.get("confidence", 0)),
            "metrics": metrics
        })

    if len(analyzed) < 5:
        return {
            "font_mismatch_detected": False,
            "field_mismatch_score": 0,
            "suspicious_fields": [],
            "suspicious_regions": [],
            "reasons": []
        }

    baseline = {
        key: [
            item["metrics"][key]
            for item in analyzed
            if item["field"] == "unknown"
            or item["confidence"] >= 0.75
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

    suspicious_fields = []
    suspicious_regions = []

    for item in analyzed:
        if item["field"] == "unknown":
            continue

        metrics = item["metrics"]
        evidence = []
        score = 0

        checks = [
            ("height", 2.8, 12, "text height differs"),
            ("edge_density", 3.0, 12, "edge density differs"),
            ("sharpness", 3.0, 12, "sharpness differs"),
            ("background_brightness", 3.0, 10, "local background differs"),
            ("text_darkness", 3.0, 10, "text darkness differs"),
            ("contrast", 3.0, 8, "contrast differs"),
            ("stroke", 3.0, 10, "stroke thickness differs")
        ]

        for key, threshold, weight, label in checks:
            z_score = _robust_z(
                metrics[key],
                baseline[key]
            )

            if z_score >= threshold:
                score += weight
                evidence.append(label)

        if score < 26 or len(evidence) < 2:
            continue

        rect = item["rect"]
        reason = "Text style differs from surrounding document text"
        region = {
            "x": rect["x"],
            "y": rect["y"],
            "w": rect["w"],
            "h": rect["h"],
            "type": "text_consistency",
            "reason": reason
        }

        suspicious_fields.append({
            "field": item["field"],
            "value": item["text"],
            "reason": reason,
            "score": int(min(score, 100)),
            "region": region,
            "evidence": evidence[:4]
        })
        suspicious_regions.append(region)

    total_score = int(
        min(
            sum(field["score"] for field in suspicious_fields),
            100
        )
    )

    reasons = [
        "Text field visual style differs from surrounding document text"
    ] if suspicious_fields else []

    return {
        "font_mismatch_detected": bool(suspicious_fields),
        "field_mismatch_score": total_score,
        "suspicious_fields": suspicious_fields[:5],
        "suspicious_regions": suspicious_regions[:5],
        "reasons": reasons
    }
