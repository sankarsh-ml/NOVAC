import cv2
import numpy as np

from app.services.visual_consistency_service import _image_metrics


def _entropy(values):

    histogram, _ = np.histogram(
        values,
        bins=32,
        range=(0, 255),
        density=True
    )

    histogram = histogram[
        histogram > 0
    ]

    if histogram.size == 0:
        return 0.0

    return float(
        -np.sum(
            histogram
            * np.log2(histogram)
        )
    )


def _empty_result(error=None):

    result = {
        "photo_region_detected": False,
        "photo_replacement_detected": False,
        "ai_photo_suspected": False,
        "critical_photo_issue": False,
        "printed_photo_likely": False,
        "photo_quality_issue": False,
        "positive_photo_evidence_count": 0,
        "replacement_score": 0,
        "photo_regions": [],
        "reasons": [],
        "supporting_reasons": [],
        "suppressed_reasons": []
    }

    if error:
        result["error"] = error

    return result


def _expand_box(x, y, w, h, width, height, scale=1.7):

    cx = x + w / 2.0
    cy = y + h / 2.0
    nw = w * scale
    nh = h * scale

    nx = int(
        max(0, cx - nw / 2.0)
    )

    ny = int(
        max(0, cy - nh / 2.0)
    )

    nw = int(
        min(width - nx, nw)
    )

    nh = int(
        min(height - ny, nh)
    )

    return nx, ny, nw, nh


def _detect_face_regions(gray, width, height):

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        return []

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(
            max(24, width // 18),
            max(24, height // 18)
        )
    )

    regions = []

    for x, y, w, h in faces:
        regions.append(
            _expand_box(
                x,
                y,
                w,
                h,
                width,
                height,
                scale=2.1
            )
        )

    return regions


def _detect_photo_like_regions(gray, width, height):

    edges = cv2.Canny(
        gray,
        60,
        160
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    image_area = float(width * height) if width and height else 1.0

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / image_area

        if area_ratio < 0.015 or area_ratio > 0.28:
            continue

        aspect = w / float(h) if h else 0

        if aspect < 0.55 or aspect > 1.35:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.03 * perimeter,
            True
        )

        if len(approx) < 4 or len(approx) > 8:
            continue

        candidates.append((x, y, w, h))

    candidates = sorted(
        candidates,
        key=lambda box: box[2] * box[3],
        reverse=True
    )

    return candidates[:3]


def analyze_photo_replacement(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return _empty_result(
            f"Cannot read image: {image_path}"
        )

    height, width = image.shape[:2]
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    global_metrics = _image_metrics(
        gray,
        image
    )

    regions = _detect_face_regions(
        gray,
        width,
        height
    )

    if not regions:
        regions = _detect_photo_like_regions(
            gray,
            width,
            height
        )

    photo_regions = []
    reasons = []
    supporting_reasons = []
    suppressed_reasons = []
    total_score = 0
    total_positive_evidence = 0

    for x, y, w, h in regions:

        roi_gray = gray[
            y:y + h,
            x:x + w
        ]

        roi_color = image[
            y:y + h,
            x:x + w
        ]

        if roi_gray.size == 0:
            continue

        metrics = _image_metrics(
            roi_gray,
            roi_color
        )

        noise_delta = abs(
            metrics["noise"]
            - global_metrics["noise"]
        )

        sharpness_ratio = (
            metrics["sharpness"]
            / max(global_metrics["sharpness"], 1.0)
        )

        brightness_delta = abs(
            metrics["brightness"]
            - global_metrics["brightness"]
        )

        border = gray[
            max(y - 3, 0):min(y + h + 3, height),
            max(x - 3, 0):min(x + w + 3, width)
        ]

        border_edges = cv2.Canny(
            border,
            80,
            180
        )

        border_density = float(
            np.mean(border_edges > 0)
        )

        roi_blur = cv2.GaussianBlur(
            roi_gray,
            (5, 5),
            0
        )

        roi_residual = cv2.absdiff(
            roi_gray,
            roi_blur
        )

        photo_noise = float(
            np.std(roi_residual)
        )

        photo_entropy = _entropy(
            roi_residual
        )

        positive_score = 0
        support_score = 0
        positive_reasons = []
        supporting_region_reasons = []

        printed_photo_likely = (
            photo_noise < 4.5
            and photo_entropy < 1.8
            and sharpness_ratio < 1.05
            and border_density < 0.24
        )

        photo_quality_issue = (
            sharpness_ratio < 0.55
            or photo_noise < 3.2
            or photo_entropy < 1.5
        )

        if noise_delta > 14:
            positive_score += 12
            positive_reasons.append(
                "major noise mismatch with surrounding document"
            )

        elif noise_delta > 8:
            support_score += 4
            supporting_region_reasons.append(
                "minor noise mismatch"
            )

        if sharpness_ratio > 3.0:
            positive_score += 12
            positive_reasons.append(
                "photo is much sharper than the document"
            )

        elif sharpness_ratio < 0.35 and noise_delta > 10:
            positive_score += 10
            positive_reasons.append(
                "photo is much blurrier with a different noise pattern"
            )

        elif sharpness_ratio < 0.55:
            support_score += 3
            supporting_region_reasons.append(
                "soft or low-detail printed photo"
            )

        if brightness_delta > 48:
            positive_score += 10
            positive_reasons.append(
                "photo lighting differs strongly from document"
            )

        elif brightness_delta > 34:
            support_score += 3
            supporting_region_reasons.append(
                "minor photo lighting mismatch"
            )

        if border_density > 0.30:
            positive_score += 14
            positive_reasons.append(
                "hard pasted-photo boundary"
            )

        elif border_density > 0.20:
            support_score += 4
            supporting_region_reasons.append(
                "visible photo boundary"
            )

        if photo_noise < 2.6:
            support_score += 3
            supporting_region_reasons.append(
                "low natural photo noise"
            )

        if photo_entropy < 1.35:
            support_score += 3
            supporting_region_reasons.append(
                "low photo texture entropy"
            )

        if (
            not printed_photo_likely
            and photo_noise < 2.8
            and photo_entropy < 1.35
            and sharpness_ratio > 1.25
        ):
            positive_score += 10
            positive_reasons.append(
                "synthetic-looking smooth photo texture with clean edges"
            )

        positive_evidence_count = len(positive_reasons)

        region_score = positive_score

        if positive_evidence_count:
            region_score += min(
                support_score,
                8
            )

        if positive_reasons:
            reason = (
                "Possible replaced photo region: "
                + ", ".join(positive_reasons)
            )
            reasons.append(reason)
        elif printed_photo_likely:
            reason = "Printed or low-detail physical photo traits; AI-photo signal suppressed"
            suppressed_reasons.append(reason)
        elif supporting_region_reasons:
            reason = (
                "Photo quality/supporting traits: "
                + ", ".join(supporting_region_reasons)
            )
            supporting_reasons.append(reason)
        else:
            reason = "Photo region detected without major replacement indicators"

        total_score += region_score
        total_positive_evidence += positive_evidence_count

        ai_photo_suspected = (
            positive_evidence_count >= 2
            and not printed_photo_likely
        )

        photo_regions.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(w * h),
            "type": "photo",
            "score": int(min(region_score, 40)),
            "ai_photo_suspected": ai_photo_suspected,
            "printed_photo_likely": printed_photo_likely,
            "photo_quality_issue": photo_quality_issue,
            "positive_photo_evidence_count": positive_evidence_count,
            "reason": reason,
            "metrics": {
                "noise_delta": round(noise_delta, 2),
                "sharpness_ratio": round(sharpness_ratio, 2),
                "brightness_delta": round(brightness_delta, 2),
                "border_density": round(border_density, 3),
                "photo_noise": round(photo_noise, 2),
                "photo_entropy": round(photo_entropy, 2)
            }
        })

    replacement_score = int(
        min(total_score, 100)
    )

    ai_photo_detected = any(
        region.get("ai_photo_suspected")
        for region in photo_regions
    )

    printed_photo_detected = any(
        region.get("printed_photo_likely")
        for region in photo_regions
    )

    quality_issue_detected = any(
        region.get("photo_quality_issue")
        for region in photo_regions
    )

    return {
        "photo_region_detected": len(photo_regions) > 0,
        "photo_replacement_detected": (
            replacement_score >= 22
            and total_positive_evidence >= 1
        ),
        "ai_photo_suspected": ai_photo_detected,
        "critical_photo_issue": (
            ai_photo_detected
            or (
                replacement_score >= 45
                and total_positive_evidence >= 2
            )
        ),
        "printed_photo_likely": printed_photo_detected,
        "photo_quality_issue": quality_issue_detected,
        "positive_photo_evidence_count": total_positive_evidence,
        "replacement_score": replacement_score,
        "photo_regions": photo_regions[:5],
        "reasons": reasons[:6],
        "supporting_reasons": supporting_reasons[:6],
        "suppressed_reasons": suppressed_reasons[:6]
    }
