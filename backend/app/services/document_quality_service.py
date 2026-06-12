import cv2
import numpy as np


# Document quality service for NOVAC
# High quality_score = better input quality
# High damage_score / fold_tear_score = worse physical document condition


def _default_result(error=None):
    result = {
        "analysis_reliable": True,
        "rejection_recommended": False,
        "quality_score": 100,
        "damage_score": 0,
        "blur_score": 0,
        "glare_score": 0,
        "fold_tear_score": 0,
        "crease_score": 0,
        "wrinkle_score": 0,
        "physical_damage_score": 0,
        "low_resolution": False,
        "poor_lighting": False,
        "excessive_noise": False,
        "severe_issue_count": 0,
        "weak_issue_count": 0,
        "readable_text": False,
        "quality_status": "good",
        "analysis_confidence": 100,
        "quality_reliable": True,
        "quality_warning": False,
        "reasons": [],
        "metrics": {},
    }

    if error:
        result["analysis_reliable"] = False
        result["rejection_recommended"] = True
        result["quality_score"] = 0
        result["damage_score"] = 100
        result["fold_tear_score"] = 100
        result["physical_damage_score"] = 100
        result["severe_issue_count"] = 1
        result["quality_status"] = "unprocessable"
        result["analysis_confidence"] = 0
        result["quality_reliable"] = False
        result["quality_warning"] = True
        result["reasons"] = [error]
        result["error"] = error

    return result


def _clamp(value, low=0, high=100):
    return int(min(max(round(float(value)), low), high))


def _score_from_laplacian(variance):
    """
    Higher blur_score = worse blur.
    Mild camera blur is intentionally lenient.
    Severe blur matters only when OCR is also weak.
    """

    if variance >= 180:
        return 0
    if variance >= 90:
        return 18
    if variance >= 45:
        return 48
    return 85


def _detector_scatter(detector_results, width, height, image_area):
    """
    Debug-only detector scatter.
    Do not directly reject document quality from detector regions.
    """

    if not detector_results or image_area <= 0:
        return 0, 0

    regions = []

    for key in ["forgery", "mvss", "ela", "text_consistency"]:
        result = detector_results.get(key, {}) or {}
        regions.extend(
            result.get("suspicious_regions", [])
            or result.get("annotation_regions", [])
            or []
        )

    if not regions:
        return 0, 0

    total_area = 0.0
    quadrants = set()

    for region in regions:
        w = float(region.get("w", 0) or 0)
        h = float(region.get("h", 0) or 0)
        x = float(region.get("x", 0) or 0)
        y = float(region.get("y", 0) or 0)
        area = float(region.get("area") or (w * h))

        total_area += max(area, 0)

        cx = x + (w / 2)
        cy = y + (h / 2)

        quadrants.add(
            (
                0 if cx < width * 0.5 else 1,
                0 if cy < height * 0.5 else 1,
            )
        )

    spread_penalty = 0
    area_ratio = total_area / image_area

    if len(regions) >= 8:
        spread_penalty += 22
    elif len(regions) >= 5:
        spread_penalty += 12

    if area_ratio > 0.35:
        spread_penalty += 25
    elif area_ratio > 0.18:
        spread_penalty += 12

    if len(quadrants) >= 3 and len(regions) >= 5:
        spread_penalty += 12

    return min(spread_penalty, 45), len(regions)


def _normalize_box(box, width, height):
    """
    Accept common document box formats and return x, y, w, h.
    """

    if not box:
        return None

    try:
        if isinstance(box, dict):
            x = box.get("x", box.get("left", 0))
            y = box.get("y", box.get("top", 0))
            w = box.get("w", box.get("width", 0))
            h = box.get("h", box.get("height", 0))
        elif isinstance(box, (list, tuple)) and len(box) >= 4:
            x, y, w, h = box[:4]
        else:
            return None

        x = int(float(x))
        y = int(float(y))
        w = int(float(w))
        h = int(float(h))

        if w <= 0 or h <= 0:
            return None

        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))

        if w * h < width * height * 0.08:
            return None

        return x, y, w, h

    except Exception:
        return None


def _estimate_document_box(image, gray):
    """
    Best-effort document region.
    If uncertain, fall back to almost the whole image instead of failing.
    """

    height, width = gray.shape[:2]
    image_area = float(width * height)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Broad paper/document mask. Keeps camera backgrounds mostly out.
    mask = ((gray > 70) & (hsv[:, :, 1] < 210)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)

        if area < image_area * 0.12:
            continue
        if area > image_area * 0.985:
            continue

        aspect = w / float(h) if h else 0
        if aspect < 0.45 or aspect > 4.8:
            continue

        candidates.append((area, x, y, w, h))

    if candidates:
        _, x, y, w, h = max(candidates, key=lambda item: item[0])
        pad_x = int(w * 0.025)
        pad_y = int(h * 0.025)
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(width - x, w + 2 * pad_x)
        h = min(height - y, h + 2 * pad_y)
        return x, y, w, h

    margin_x = int(width * 0.03)
    margin_y = int(height * 0.03)
    return margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y


def _get_document_roi(image, gray, document_condition_result):
    height, width = gray.shape[:2]
    document_condition_result = document_condition_result or {}

    box = _normalize_box(document_condition_result.get("document_box"), width, height)

    if not box:
        box = _estimate_document_box(image, gray)

    x, y, w, h = box
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))

    return image[y:y + h, x:x + w], gray[y:y + h, x:x + w], {
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
    }


def _resize_for_damage_analysis(gray):
    height, width = gray.shape[:2]
    max_side = max(width, height)

    if max_side <= 1100:
        return gray, 1.0

    scale = 1100.0 / float(max_side)
    resized = cv2.resize(
        gray,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


def _line_length(x1, y1, x2, y2):
    return float(np.hypot(x2 - x1, y2 - y1))


def _line_angle(x1, y1, x2, y2):
    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    if angle > 90:
        angle = 180 - angle
    return float(angle)


def _is_border_like_line(x1, y1, x2, y2, width, height):
    margin_x = width * 0.055
    margin_y = height * 0.055

    near_top = y1 < margin_y and y2 < margin_y
    near_bottom = y1 > height - margin_y and y2 > height - margin_y
    near_left = x1 < margin_x and x2 < margin_x
    near_right = x1 > width - margin_x and x2 > width - margin_x

    return near_top or near_bottom or near_left or near_right


def _is_printed_layout_line(x1, y1, x2, y2, angle, width, height):
    horizontal = angle <= 4
    vertical = angle >= 86

    if not horizontal and not vertical:
        return False

    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0
    length = _line_length(x1, y1, x2, y2)

    near_common_horizontal_rule = horizontal and (
        my < height * 0.16
        or my > height * 0.82
        or height * 0.48 <= my <= height * 0.58
        or height * 0.64 <= my <= height * 0.76
    )
    near_common_box_edge = vertical and (
        mx < width * 0.20
        or width * 0.70 <= mx <= width * 0.92
    )
    very_straight_long_rule = (
        (horizontal and length > width * 0.32)
        or (vertical and length > height * 0.32)
    )

    return near_common_horizontal_rule or near_common_box_edge or very_straight_long_rule


def _crosses_central_area(x1, y1, x2, y2, width, height):
    central_x1 = width * 0.12
    central_x2 = width * 0.88
    central_y1 = height * 0.12
    central_y2 = height * 0.88

    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0

    if central_x1 <= mx <= central_x2 and central_y1 <= my <= central_y2:
        return True

    if central_x1 <= x1 <= central_x2 and central_y1 <= y1 <= central_y2:
        return True

    if central_x1 <= x2 <= central_x2 and central_y1 <= y2 <= central_y2:
        return True

    return False


def _line_local_edge_density(edges, x1, y1, x2, y2, pad=12):
    height, width = edges.shape[:2]

    x_min = max(0, min(x1, x2) - pad)
    x_max = min(width, max(x1, x2) + pad)
    y_min = max(0, min(y1, y2) - pad)
    y_max = min(height, max(y1, y2) + pad)

    patch = edges[y_min:y_max, x_min:x_max]

    if patch.size <= 0:
        return 0.0

    return float(np.mean(patch > 0))


def _line_brightness_features(gray, x1, y1, x2, y2):
    length = _line_length(x1, y1, x2, y2)

    if length <= 0:
        return {
            "brightness_discontinuity": 0.0,
            "ridge_contrast": 0.0,
            "paired_highlight_shadow": False,
        }

    sample_count = int(min(max(length / 8, 16), 90))
    xs = np.linspace(x1, x2, sample_count)
    ys = np.linspace(y1, y2, sample_count)
    nx = -(y2 - y1) / length
    ny = (x2 - x1) / length

    side_distance = 5
    height, width = gray.shape[:2]

    side_a_x = np.clip(np.round(xs + nx * side_distance).astype(int), 0, width - 1)
    side_a_y = np.clip(np.round(ys + ny * side_distance).astype(int), 0, height - 1)
    side_b_x = np.clip(np.round(xs - nx * side_distance).astype(int), 0, width - 1)
    side_b_y = np.clip(np.round(ys - ny * side_distance).astype(int), 0, height - 1)
    center_x = np.clip(np.round(xs).astype(int), 0, width - 1)
    center_y = np.clip(np.round(ys).astype(int), 0, height - 1)

    side_a = gray[side_a_y, side_a_x].astype(np.float32)
    side_b = gray[side_b_y, side_b_x].astype(np.float32)
    center = gray[center_y, center_x].astype(np.float32)

    side_a_mean = float(np.mean(side_a))
    side_b_mean = float(np.mean(side_b))
    center_mean = float(np.mean(center))
    side_mean = (side_a_mean + side_b_mean) / 2.0

    brightness_discontinuity = abs(side_a_mean - side_b_mean)
    ridge_contrast = abs(center_mean - side_mean)
    paired_highlight_shadow = (
        brightness_discontinuity >= 10
        and (max(side_a_mean, side_b_mean) - center_mean >= 8 or ridge_contrast >= 12)
    )

    return {
        "brightness_discontinuity": float(brightness_discontinuity),
        "ridge_contrast": float(ridge_contrast),
        "paired_highlight_shadow": bool(paired_highlight_shadow),
    }


def _detect_moire_texture(gray):
    resized, _ = _resize_for_damage_analysis(gray)
    height, width = resized.shape[:2]

    if width < 160 or height < 160:
        return {
            "detected": False,
            "parallel_line_count": 0,
            "periodic_profile_strength": 0.0,
            "uniform_texture_ratio": 0.0,
        }

    blurred = cv2.GaussianBlur(resized, (0, 0), 3)
    high_freq = cv2.absdiff(resized, blurred)
    tile_strengths = []

    for row in np.array_split(high_freq, 4, axis=0):
        for tile in np.array_split(row, 4, axis=1):
            if tile.size:
                tile_strengths.append(float(np.mean(tile)))

    uniform_texture_ratio = 0.0
    if tile_strengths:
        mean_strength = float(np.mean(tile_strengths))
        strength_std = float(np.std(tile_strengths))
        uniform_texture_ratio = mean_strength / max(strength_std + 1.0, 1.0)

    row_profile = np.mean(high_freq, axis=1)
    col_profile = np.mean(high_freq, axis=0)

    def periodic_strength(profile):
        profile = profile.astype(np.float32) - float(np.mean(profile))
        if profile.size < 32 or float(np.std(profile)) < 0.1:
            return 0.0

        spectrum = np.abs(np.fft.rfft(profile))
        if spectrum.size <= 4:
            return 0.0

        body = spectrum[3:]
        return float(np.max(body) / max(np.mean(body) + 1e-6, 1e-6))

    periodic_profile_strength = max(
        periodic_strength(row_profile),
        periodic_strength(col_profile),
    )

    edges = cv2.Canny(high_freq, 30, 90)
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(24, int(min(width, height) * 0.035)),
        minLineLength=max(28, int(min(width, height) * 0.08)),
        maxLineGap=max(6, int(min(width, height) * 0.02)),
    )

    parallel_line_count = 0
    if raw_lines is not None:
        for raw_line in raw_lines:
            x1, y1, x2, y2 = map(int, raw_line[0])
            angle = _line_angle(x1, y1, x2, y2)
            if angle <= 7 or angle >= 83:
                parallel_line_count += 1

    detected = (
        periodic_profile_strength >= 8.0
        and uniform_texture_ratio >= 1.6
    ) or (
        parallel_line_count >= 18
        and uniform_texture_ratio >= 1.25
    )

    return {
        "detected": bool(detected),
        "parallel_line_count": int(parallel_line_count),
        "periodic_profile_strength": round(periodic_profile_strength, 3),
        "uniform_texture_ratio": round(uniform_texture_ratio, 3),
    }


def _dedupe_lines(lines):
    buckets = {}

    for item in lines:
        x1, y1, x2, y2 = item["line"]
        angle = item["angle"]
        theta = np.radians(angle)
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)

        key = (int(round(angle / 8.0)), int(round(rho / 24.0)))

        if key not in buckets or item["length"] > buckets[key]["length"]:
            buckets[key] = item

    return list(buckets.values())


def _detect_crease_and_wrinkle_scores(image, gray, document_condition_result=None):
    """
    Detect physical document damage.

    This targets real artifacts:
    - sharp fold/crease lines
    - wrinkles/bends on the document surface

    It tries to avoid normal camera artifacts:
    - background
    - mild blur
    - perspective
    - QR code/text density
    """

    _, roi_gray, roi_box = _get_document_roi(
        image,
        gray,
        document_condition_result,
    )

    if roi_gray is None or roi_gray.size == 0:
        return {
            "crease_score": 0,
            "wrinkle_score": 0,
            "physical_damage_score": 0,
            "crease_confidence": "none",
            "metrics": {
                "document_roi_box": roi_box,
                "crease_confidence": "none",
                "moire_texture_detected": False,
                "printed_line_guard_applied": False,
            },
        }

    resized_gray, scale = _resize_for_damage_analysis(roi_gray)
    height, width = resized_gray.shape[:2]
    moire_texture = _detect_moire_texture(roi_gray)

    if width < 120 or height < 120:
        return {
            "crease_score": 0,
            "wrinkle_score": 0,
            "physical_damage_score": 0,
            "crease_confidence": "none",
            "metrics": {
                "document_roi_box": roi_box,
                "damage_analysis_width": int(width),
                "damage_analysis_height": int(height),
                "damage_analysis_scale": round(scale, 4),
                "crease_confidence": "none",
                "moire_texture_detected": bool(moire_texture["detected"]),
                "printed_line_guard_applied": False,
                **moire_texture,
            },
        }

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized_gray)
    enhanced = cv2.bilateralFilter(enhanced, 7, 40, 40)

    edges = cv2.Canny(enhanced, 55, 145)

    # Connect broken crease lines but do not join whole text blocks too much.
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    edges_for_lines = cv2.bitwise_or(
        cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel),
        cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel),
    )

    min_side = min(width, height)
    diagonal = float(np.hypot(width, height))

    long_min_length = max(70, int(min_side * 0.23))
    medium_min_length = max(32, int(min_side * 0.085))

    raw_lines = cv2.HoughLinesP(
        edges_for_lines,
        rho=1,
        theta=np.pi / 180,
        threshold=max(30, int(min_side * 0.045)),
        minLineLength=medium_min_length,
        maxLineGap=max(10, int(min_side * 0.035)),
    )

    if raw_lines is None:
        return {
            "crease_score": 0,
            "wrinkle_score": 0,
            "physical_damage_score": 0,
            "crease_confidence": "none",
            "metrics": {
                "document_roi_box": roi_box,
                "damage_analysis_width": int(width),
                "damage_analysis_height": int(height),
                "damage_analysis_scale": round(scale, 4),
                "long_internal_line_count": 0,
                "central_long_line_count": 0,
                "medium_internal_line_count": 0,
                "crease_confidence": "none",
                "moire_texture_detected": bool(moire_texture["detected"]),
                "printed_line_guard_applied": False,
                **moire_texture,
            },
        }

    long_lines = []
    medium_lines = []
    printed_line_guard_applied = False
    rejected_printed_line_count = 0
    rejected_dense_line_count = 0
    max_crease_feature_count = 0

    for raw_line in raw_lines:
        x1, y1, x2, y2 = map(int, raw_line[0])
        length = _line_length(x1, y1, x2, y2)

        if length < medium_min_length:
            continue

        angle = _line_angle(x1, y1, x2, y2)

        if _is_border_like_line(x1, y1, x2, y2, width, height):
            printed_line_guard_applied = True
            rejected_printed_line_count += 1
            continue

        if _is_printed_layout_line(x1, y1, x2, y2, angle, width, height):
            printed_line_guard_applied = True
            rejected_printed_line_count += 1
            continue

        central = _crosses_central_area(x1, y1, x2, y2, width, height)
        local_density = _line_local_edge_density(edges, x1, y1, x2, y2)

        # QR codes, logos, and text usually have dense local edges.
        # Creases are more isolated line structures.
        if local_density > 0.34:
            rejected_dense_line_count += 1
            continue

        brightness_features = _line_brightness_features(
            enhanced,
            x1,
            y1,
            x2,
            y2,
        )
        brightness_discontinuity = brightness_features["brightness_discontinuity"]
        ridge_contrast = brightness_features["ridge_contrast"]
        paired_highlight_shadow = brightness_features["paired_highlight_shadow"]
        near_perfect_layout_angle = angle <= 4 or angle >= 86
        spans_multiple_zones = central and (
            abs(x2 - x1) > width * 0.38
            or abs(y2 - y1) > height * 0.38
        )

        feature_count = 0
        feature_count += 1 if length >= long_min_length and central else 0
        feature_count += 1 if paired_highlight_shadow else 0
        feature_count += 1 if brightness_discontinuity >= 12 else 0
        feature_count += 1 if not near_perfect_layout_angle else 0
        feature_count += 1 if spans_multiple_zones else 0
        feature_count += 1 if ridge_contrast >= 16 else 0
        max_crease_feature_count = max(max_crease_feature_count, feature_count)

        if length >= long_min_length and feature_count < 2:
            rejected_printed_line_count += 1
            continue

        item = {
            "line": (x1, y1, x2, y2),
            "length": length,
            "angle": angle,
            "central": central,
            "local_edge_density": local_density,
            "brightness_discontinuity": brightness_discontinuity,
            "ridge_contrast": ridge_contrast,
            "crease_feature_count": feature_count,
        }

        if length >= long_min_length and feature_count >= 2:
            long_lines.append(item)
        elif central and local_density < 0.28 and feature_count >= 1:
            medium_lines.append(item)

    long_lines = _dedupe_lines(long_lines)
    medium_lines = _dedupe_lines(medium_lines)

    central_long_lines = [item for item in long_lines if item["central"]]

    long_count = len(long_lines)
    central_long_count = len(central_long_lines)
    medium_count = len(medium_lines)

    total_long_length = sum(item["length"] for item in long_lines)
    central_long_length = sum(item["length"] for item in central_long_lines)
    total_medium_length = sum(item["length"] for item in medium_lines)

    long_length_ratio = total_long_length / max(diagonal, 1.0)
    central_long_length_ratio = central_long_length / max(diagonal, 1.0)
    medium_length_ratio = total_medium_length / max(diagonal, 1.0)
    screen_texture_guard_applied = (
        moire_texture["detected"]
        and (
            long_count >= 90
            or long_length_ratio >= 18
            or central_long_length_ratio >= 18
        )
    )

    crease_score = 0.0
    crease_score += central_long_count * 24
    crease_score += max(0, long_count - central_long_count) * 10
    crease_score += central_long_length_ratio * 42
    crease_score += long_length_ratio * 18

    wrinkle_score = 0.0
    wrinkle_score += medium_count * 4.5
    wrinkle_score += medium_length_ratio * 28

    if screen_texture_guard_applied:
        crease_score = min(crease_score, 45)
        wrinkle_score = min(wrinkle_score, 40)
    elif moire_texture["detected"]:
        wrinkle_score *= 0.35
        crease_score *= 0.65

    # Avoid accidental tiny detections becoming document damage.
    if central_long_count == 0 and medium_count < 5:
        wrinkle_score *= 0.35

    if long_count == 0:
        crease_score *= 0.45

    crease_score = _clamp(crease_score)
    wrinkle_score = _clamp(wrinkle_score)

    if max_crease_feature_count >= 3 and central_long_count >= 1:
        crease_confidence = "high"
    elif max_crease_feature_count >= 2 and long_count >= 1:
        crease_confidence = "medium"
    elif crease_score > 0 or wrinkle_score > 0:
        crease_confidence = "low"
    else:
        crease_confidence = "none"

    if screen_texture_guard_applied:
        crease_confidence = "low" if crease_score > 0 or wrinkle_score > 0 else "none"
        crease_score = min(crease_score, 45)
        wrinkle_score = min(wrinkle_score, 40)
    elif crease_confidence == "high":
        crease_score = max(
            crease_score,
            min(
                95,
                central_long_count * 30
                + central_long_length_ratio * 55
                + max_crease_feature_count * 6,
            ),
        )
    elif crease_confidence == "medium":
        crease_score = _clamp(crease_score * 0.82)
    elif crease_confidence in {"low", "none"}:
        crease_score = _clamp(crease_score * 0.55)

    physical_damage_score = max(
        crease_score,
        wrinkle_score,
        min(100, crease_score * 0.72 + wrinkle_score * 0.42),
    )

    if wrinkle_score >= 70 and crease_score < 55:
        physical_damage_score = min(physical_damage_score, 55)

    physical_damage_score = _clamp(physical_damage_score)

    return {
        "crease_score": crease_score,
        "wrinkle_score": wrinkle_score,
        "physical_damage_score": physical_damage_score,
        "crease_confidence": crease_confidence,
        "metrics": {
            "document_roi_box": roi_box,
            "damage_analysis_width": int(width),
            "damage_analysis_height": int(height),
            "damage_analysis_scale": round(scale, 4),
            "long_internal_line_count": int(long_count),
            "central_long_line_count": int(central_long_count),
            "medium_internal_line_count": int(medium_count),
            "total_internal_line_length_ratio": round(long_length_ratio, 4),
            "central_internal_line_length_ratio": round(central_long_length_ratio, 4),
            "medium_internal_line_length_ratio": round(medium_length_ratio, 4),
            "long_min_length": int(long_min_length),
            "medium_min_length": int(medium_min_length),
            "crease_confidence": crease_confidence,
            "max_crease_feature_count": int(max_crease_feature_count),
            "rejected_printed_line_count": int(rejected_printed_line_count),
            "rejected_dense_line_count": int(rejected_dense_line_count),
            "moire_texture_detected": bool(moire_texture["detected"]),
            "printed_line_guard_applied": bool(printed_line_guard_applied),
            "screen_texture_guard_applied": bool(screen_texture_guard_applied),
            **moire_texture,
        },
    }


def analyze_document_quality(
    image_path,
    ocr_result=None,
    document_condition_result=None,
    detector_results=None,
):
    """
    Determines whether the uploaded document image is reliable enough.

    Important behavior:
    - Mild camera artifacts are allowed.
    - Heavy camera blur/glare matters only when readability is affected.
    - Physical damage such as creases, folds, bends, wrinkles, tears, and
      cracked lamination raises damage_score even if OCR is readable.
    """

    image = cv2.imread(image_path)

    if image is None:
        return _default_result(
            f"Cannot read image for document quality analysis: {image_path}"
        )

    height, width = image.shape[:2]
    image_area = float(width * height) if width and height else 1.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ocr_result = ocr_result or {}
    ocr_lines = ocr_result.get("lines", []) or []
    ocr_text = str(ocr_result.get("text", "") or "")

    avg_confidence = float(
        ocr_result.get("avg_confidence", 1.0)
        if ocr_result.get("avg_confidence", None) is not None
        else 1.0
    )

    useful_line_count = len(ocr_lines)
    text_length = len(ocr_text.strip())

    readable_text = (
        avg_confidence >= 0.65
        and useful_line_count >= 4
    ) or text_length >= 40
    very_readable_text = avg_confidence >= 0.80 and useful_line_count >= 8

    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = _score_from_laplacian(laplacian_variance)

    brightness = float(np.mean(gray))
    brightness_std = float(np.std(gray))

    tile_means = []
    for row in np.array_split(gray, 4, axis=0):
        for tile in np.array_split(row, 4, axis=1):
            if tile.size:
                tile_means.append(float(np.mean(tile)))

    lighting_range = max(tile_means) - min(tile_means) if tile_means else 0

    severe_lighting = brightness < 38 or brightness > 247 or lighting_range > 170
    weak_lighting = brightness < 58 or brightness > 232 or lighting_range > 130
    poor_lighting = bool(severe_lighting)

    glare_ratio = float(np.mean((gray > 245) & (hsv[:, :, 1] < 45)))
    glare_score = int(min(glare_ratio * 1200, 100))

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_residual = cv2.absdiff(gray, blur)
    noise_level = float(np.mean(noise_residual))
    excessive_noise = noise_level > 13.5

    low_resolution = min(width, height) < 620 or image_area < 620000

    document_condition_result = document_condition_result or {}

    raw_fold_tear_score = float(
        document_condition_result.get("condition_score", 0) or 0
    )

    condition_confidence = document_condition_result.get(
        "condition_confidence",
        "low",
    )

    external_fold_tear_score = raw_fold_tear_score

    if condition_confidence == "low":
        external_fold_tear_score *= 0.40
    elif condition_confidence == "medium":
        external_fold_tear_score *= 0.75

    visual_damage = _detect_crease_and_wrinkle_scores(
        image,
        gray,
        document_condition_result,
    )

    crease_score = int(visual_damage["crease_score"])
    wrinkle_score = int(visual_damage["wrinkle_score"])
    visual_physical_damage_score = int(visual_damage["physical_damage_score"])
    crease_confidence = visual_damage.get("crease_confidence", "none")
    visual_metrics = visual_damage.get("metrics", {})
    central_line_ratio = float(
        visual_metrics.get("central_internal_line_length_ratio", 0)
        or 0
    )

    if (
        crease_confidence == "high"
        and central_line_ratio < 3.0
        and raw_fold_tear_score < 35
    ):
        crease_score = min(crease_score, 35)
        wrinkle_score = min(wrinkle_score, 25)
        visual_physical_damage_score = min(visual_physical_damage_score, 35)
        crease_confidence = "low"
        visual_damage["crease_confidence"] = crease_confidence
        visual_metrics["crease_confidence"] = crease_confidence
        visual_metrics["layout_edge_guard_applied"] = True

    if (
        blur_score >= 85
        and readable_text
        and raw_fold_tear_score < 35
    ):
        crease_score = min(crease_score, 35)
        wrinkle_score = min(wrinkle_score, 30)
        visual_physical_damage_score = min(visual_physical_damage_score, 40)
        crease_confidence = "low"
        visual_damage["crease_confidence"] = crease_confidence
        visual_metrics["crease_confidence"] = crease_confidence
        visual_metrics["readable_blur_guard_applied"] = True

    physical_damage_score = _clamp(
        max(external_fold_tear_score, visual_physical_damage_score)
    )

    fold_tear_score = _clamp(max(external_fold_tear_score, physical_damage_score))

    condition_error = str(document_condition_result.get("error", "") or "").lower()
    boundary_unknown = not document_condition_result.get("document_box")

    document_boundary_unreliable = (
        (
            boundary_unknown
            or "too small" in condition_error
            or "no document boundary" in condition_error
        )
        and avg_confidence < 0.50
        and low_resolution
    )

    detector_scatter_score, scattered_region_count = _detector_scatter(
        detector_results,
        width,
        height,
        image_area,
    )

    detector_scatter_quality_penalty = 0

    severe_blur_low_ocr = blur_score >= 85 and avg_confidence < 0.55
    severe_glare_low_ocr = glare_score >= 75 and avg_confidence < 0.60
    severe_lighting_low_ocr = severe_lighting and avg_confidence < 0.60
    severe_noise_low_ocr = excessive_noise and avg_confidence < 0.60
    very_low_resolution_low_ocr = low_resolution and avg_confidence < 0.55

    camera_capture_penalty = (
        (32 if severe_blur_low_ocr else 0)
        + (28 if severe_glare_low_ocr else 0)
        + (24 if severe_lighting_low_ocr else 0)
        + (18 if severe_noise_low_ocr else 0)
        + (14 if very_low_resolution_low_ocr else 0)
        + (18 if document_boundary_unreliable else 0)
        + detector_scatter_quality_penalty
    )

    if not readable_text:
        if blur_score >= 55 and not severe_blur_low_ocr:
            camera_capture_penalty += 8
        if glare_score >= 35 and not severe_glare_low_ocr:
            camera_capture_penalty += 7
        if weak_lighting and not severe_lighting_low_ocr:
            camera_capture_penalty += 6
        if excessive_noise and not severe_noise_low_ocr:
            camera_capture_penalty += 6
        if low_resolution and not very_low_resolution_low_ocr:
            camera_capture_penalty += 5

    camera_capture_penalty = _clamp(camera_capture_penalty)
    camera_damage_score = camera_capture_penalty

    external_high_confidence_damage = (
        raw_fold_tear_score >= 85
        and condition_confidence in {"medium", "high"}
    )
    high_confidence_physical_damage = (
        physical_damage_score >= 80
        or (crease_score >= 75 and crease_confidence == "high")
        or external_high_confidence_damage
    )

    physical_condition_penalty = physical_damage_score * 0.6
    damage_score = _clamp(max(physical_damage_score, min(100, camera_damage_score)))

    quality_score = 100.0 - camera_capture_penalty - physical_condition_penalty

    if avg_confidence < 0.50:
        quality_score -= (0.50 - avg_confidence) * 60
    elif avg_confidence < 0.70 and useful_line_count < 5:
        quality_score -= (0.70 - avg_confidence) * 30

    if readable_text and physical_damage_score < 70:
        quality_score = max(quality_score, 65)
    elif very_readable_text and physical_damage_score < 35:
        quality_score += 4

    quality_score = _clamp(quality_score)

    severe_issue_count = 0
    weak_issue_count = 0

    if high_confidence_physical_damage:
        severe_issue_count += 1
    elif physical_damage_score >= 35:
        weak_issue_count += 1

    if crease_score >= 75 and crease_confidence == "high":
        severe_issue_count += 1
    elif crease_score >= 45 and crease_confidence in {"medium", "high"}:
        weak_issue_count += 1

    if wrinkle_score >= 35:
        weak_issue_count += 1

    if severe_blur_low_ocr:
        severe_issue_count += 1
    elif blur_score >= 55 and not readable_text:
        weak_issue_count += 1

    if severe_glare_low_ocr:
        severe_issue_count += 1
    elif glare_score >= 35 and not readable_text:
        weak_issue_count += 1

    if severe_lighting_low_ocr:
        severe_issue_count += 1
    elif weak_lighting and not readable_text:
        weak_issue_count += 1

    if severe_noise_low_ocr:
        severe_issue_count += 1
    elif excessive_noise and not readable_text:
        weak_issue_count += 1

    if very_low_resolution_low_ocr:
        severe_issue_count += 1
    elif low_resolution and not readable_text:
        weak_issue_count += 1

    if document_boundary_unreliable:
        severe_issue_count += 1
    elif boundary_unknown and avg_confidence < 0.65:
        weak_issue_count += 1

    reasons = []

    if crease_score >= 75 and crease_confidence == "high":
        reasons.append("Sharp crease or fold lines detected across the document")
    elif crease_score >= 45 and crease_confidence in {"medium", "high"}:
        reasons.append("Possible crease or fold lines detected on the document")

    if wrinkle_score >= 70 and crease_score >= 55 and crease_confidence in {"medium", "high"}:
        reasons.append("Wrinkled or bent document surface detected")
    elif wrinkle_score >= 35:
        reasons.append("Possible surface texture or wrinkle artifacts detected")

    if external_fold_tear_score >= 75:
        reasons.append("Document appears heavily folded, torn, or physically damaged")
    elif external_fold_tear_score >= 35:
        reasons.append("Possible physical fold or tear indicators detected")

    if severe_blur_low_ocr:
        reasons.append("Document image is severely blurred and text readability is low")
    elif blur_score >= 55 and not readable_text:
        reasons.append("Document image has blur that may affect readability")

    if severe_lighting_low_ocr:
        reasons.append("Document lighting is severely uneven or outside reliable range")
    elif weak_lighting and not readable_text:
        reasons.append("Document lighting has mild unevenness")

    if severe_glare_low_ocr:
        reasons.append("Strong glare or overexposed patches reduce readability")
    elif glare_score >= 35 and not readable_text:
        reasons.append("Mild glare or bright patches detected")

    if severe_noise_low_ocr:
        reasons.append("Excessive image noise reduces document readability")
    elif excessive_noise and not readable_text:
        reasons.append("Mild image noise detected")

    if very_low_resolution_low_ocr:
        reasons.append("Document image resolution is too low for reliable verification")
    elif low_resolution and not readable_text:
        reasons.append("Document image resolution is somewhat low")

    if document_boundary_unreliable:
        reasons.append("Document boundary is unclear and OCR readability is weak")

    if detector_scatter_score >= 20:
        reasons.append(
            "Visual detectors fired broadly; treated as debug signal, not direct quality rejection"
        )

    if avg_confidence < 0.35 and useful_line_count < 4:
        reasons.append("Text readability is too low for reliable verification")

    severe_camera_failure = (
        not readable_text
        and (
            severe_blur_low_ocr
            or severe_glare_low_ocr
            or severe_lighting_low_ocr
            or very_low_resolution_low_ocr
        )
        and avg_confidence < 0.45
    )

    truly_unprocessable = (
        (
            avg_confidence < 0.35
            and useful_line_count < 4
            and (
                blur_score >= 55
                or excessive_noise
                or severe_lighting
            )
        )
        or (
            avg_confidence < 0.30
            and useful_line_count < 3
            and (
                blur_score >= 85
                or glare_score >= 85
                or severe_lighting
                or document_boundary_unreliable
            )
        )
        or severe_camera_failure
    )

    # Readable OCR can rescue camera artifacts, but not severe physical damage.
    if (
        readable_text
        and physical_damage_score < 70
        and not high_confidence_physical_damage
    ):
        quality_score = max(quality_score, 65)

    if truly_unprocessable:
        quality_status = "unprocessable"
    elif quality_score < 45 or physical_damage_score >= 75:
        quality_status = "bad"
    elif quality_score < 65 or physical_damage_score >= 45:
        quality_status = "warning"
    else:
        quality_status = "good"

    rejection_recommended = quality_status == "unprocessable"
    quality_reliable = quality_status != "unprocessable"
    quality_warning = quality_status in {"warning", "bad"}
    analysis_confidence = _clamp(
        min(
            quality_score,
            100 - (20 if quality_status == "bad" else 0)
        )
    )

    if quality_status == "unprocessable":
        reasons.append("Document could not be analyzed reliably due to readability or visibility issues")

    reasons = list(dict.fromkeys(reasons))

    merged_metrics = {
        "width": int(width),
        "height": int(height),
        "laplacian_variance": round(laplacian_variance, 2),
        "brightness": round(brightness, 2),
        "brightness_std": round(brightness_std, 2),
        "lighting_range": round(lighting_range, 2),
        "noise_level": round(noise_level, 2),
        "avg_ocr_confidence": round(avg_confidence, 3),
        "useful_line_count": int(useful_line_count),
        "ocr_text_length": int(text_length),
        "scattered_region_count": int(scattered_region_count),
        "detector_scatter_score_debug_only": int(detector_scatter_score),
        "boundary_unknown": bool(boundary_unknown),
        "document_boundary_unreliable": bool(document_boundary_unreliable),
        "condition_confidence": condition_confidence,
        "raw_fold_tear_score": round(raw_fold_tear_score, 2),
        "external_fold_tear_score": round(float(external_fold_tear_score), 2),
        "external_high_confidence_damage": bool(external_high_confidence_damage),
        "high_confidence_physical_damage": bool(high_confidence_physical_damage),
        "truly_unprocessable": bool(truly_unprocessable),
        "severe_lighting": bool(severe_lighting),
        "weak_lighting": bool(weak_lighting),
        "severe_blur_low_ocr": bool(severe_blur_low_ocr),
        "severe_glare_low_ocr": bool(severe_glare_low_ocr),
        "severe_lighting_low_ocr": bool(severe_lighting_low_ocr),
        "severe_noise_low_ocr": bool(severe_noise_low_ocr),
        "very_low_resolution_low_ocr": bool(very_low_resolution_low_ocr),
        "camera_damage_score": round(float(camera_damage_score), 2),
        "camera_capture_penalty": round(float(camera_capture_penalty), 2),
        "physical_condition_penalty": round(float(physical_condition_penalty), 2),
        "camera_capture_quality": "weak" if camera_capture_penalty >= 35 else "usable",
        "physical_document_condition": "damaged" if high_confidence_physical_damage else "usable",
        "visual_physical_damage_score": int(visual_physical_damage_score),
        "crease_confidence": crease_confidence,
    }

    merged_metrics.update(visual_damage.get("metrics", {}))

    return {
        "analysis_reliable": not bool(rejection_recommended),
        "rejection_recommended": bool(rejection_recommended),
        "quality_score": int(quality_score),
        "damage_score": int(damage_score),
        "blur_score": int(min(max(round(blur_score), 0), 100)),
        "glare_score": int(min(max(round(glare_score), 0), 100)),
        "fold_tear_score": int(fold_tear_score),
        "crease_score": int(crease_score),
        "wrinkle_score": int(wrinkle_score),
        "physical_damage_score": int(physical_damage_score),
        "low_resolution": bool(low_resolution),
        "poor_lighting": bool(poor_lighting),
        "excessive_noise": bool(excessive_noise),
        "severe_issue_count": int(severe_issue_count),
        "weak_issue_count": int(weak_issue_count),
        "readable_text": bool(readable_text),
        "quality_status": quality_status,
        "analysis_confidence": int(analysis_confidence),
        "quality_reliable": bool(quality_reliable),
        "quality_warning": bool(quality_warning),
        "reasons": reasons,
        "metrics": merged_metrics,
    }
