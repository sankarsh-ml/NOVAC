import cv2
import numpy as np


def _empty_result(error=None):

    result = {
        "fold_detected": False,
        "tear_detected": False,
        "condition_score": 0,
        "condition_confidence": "low",
        "document_box": None,
        "damaged_regions": [],
        "debug_candidates": [],
        "reasons": []
    }

    if error:
        result["error"] = error

    return result


def _rect(x, y, w, h, reason, region_type):

    return {
        "x": int(max(x, 0)),
        "y": int(max(y, 0)),
        "w": int(max(w, 0)),
        "h": int(max(h, 0)),
        "area": int(max(w, 0) * max(h, 0)),
        "type": region_type,
        "reason": reason
    }


def _corner_signals(patch, edge_patch):

    if patch.size == 0:
        return []

    signals = []
    dark_ratio = float(
        np.mean(patch < 45)
    )
    bright_ratio = float(
        np.mean(patch > 245)
    )
    edge_density = float(
        np.mean(edge_patch > 0)
    )
    contrast = float(
        np.std(patch)
    )

    if dark_ratio > 0.26:
        signals.append("dark corner shadow")

    if bright_ratio > 0.82 and contrast > 20:
        signals.append("missing or washed-out corner")

    if edge_density > 0.24 and contrast > 24:
        signals.append("irregular corner edge texture")

    return signals


def _diagonal_lines(edges, width, height, document_box):

    x, y, w, h = document_box

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=110,
        minLineLength=int(min(width, height) * 0.32),
        maxLineGap=10
    )

    candidates = []

    if lines is None:
        return candidates

    for line in lines[:60]:

        x1, y1, x2, y2 = line[0]
        line_length = float(
            np.hypot(x2 - x1, y2 - y1)
        )

        if line_length < min(width, height) * 0.32:
            continue

        angle = abs(
            np.degrees(
                np.arctan2(
                    y2 - y1,
                    x2 - x1
                )
            )
        )

        near_axis = (
            angle < 12
            or abs(angle - 90) < 12
            or abs(angle - 180) < 12
        )

        if near_axis:
            continue

        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2

        # Ignore exterior scan noise and border decorations.
        if not (
            x + w * 0.12 < midpoint_x < x + w * 0.88
            and y + h * 0.12 < midpoint_y < y + h * 0.88
        ):
            continue

        candidates.append(
            _rect(
                min(x1, x2),
                min(y1, y2),
                max(abs(x2 - x1), 12),
                max(abs(y2 - y1), 12),
                "Diagonal crease candidate",
                "condition"
            )
        )

        if len(candidates) >= 3:
            break

    return candidates


def _line_shadow_score(gray, x1, y1, x2, y2):

    line_length = float(
        np.hypot(x2 - x1, y2 - y1)
    )

    if line_length < 1:
        return 0.0

    x_min = max(
        min(x1, x2) - 8,
        0
    )
    y_min = max(
        min(y1, y2) - 8,
        0
    )
    x_max = min(
        max(x1, x2) + 8,
        gray.shape[1]
    )
    y_max = min(
        max(y1, y2) + 8,
        gray.shape[0]
    )

    patch = gray[
        y_min:y_max,
        x_min:x_max
    ]

    if patch.size == 0:
        return 0.0

    if patch.shape[0] >= patch.shape[1]:
        profile = np.mean(
            patch,
            axis=1
        )
    else:
        profile = np.mean(
            patch,
            axis=0
        )

    if profile.size < 6:
        return 0.0

    center = np.mean(
        profile[
            profile.size // 3:profile.size * 2 // 3
        ]
    )
    sides = np.mean(
        np.concatenate([
            profile[:profile.size // 3],
            profile[profile.size * 2 // 3:]
        ])
    )

    return float(
        abs(center - sides)
    )


def _internal_crease_lines(image, gray, edges, width, height, document_box):

    x, y, w, h = document_box

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=85,
        minLineLength=int(min(w, h) * 0.26),
        maxLineGap=18
    )

    candidates = []

    if lines is None:
        return candidates

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    for line in lines[:100]:

        x1, y1, x2, y2 = line[0]
        length = float(
            np.hypot(x2 - x1, y2 - y1)
        )

        if length < min(w, h) * 0.26:
            continue

        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2

        if not (
            x + w * 0.08 < midpoint_x < x + w * 0.92
            and y + h * 0.10 < midpoint_y < y + h * 0.90
        ):
            continue

        angle = abs(
            np.degrees(
                np.arctan2(
                    y2 - y1,
                    x2 - x1
                )
            )
        )
        angle = min(
            angle,
            abs(180 - angle)
        )

        if angle < 5 or abs(angle - 90) < 5:
            continue

        rx = max(
            min(x1, x2) - 6,
            0
        )
        ry = max(
            min(y1, y2) - 6,
            0
        )
        rw = min(
            max(abs(x2 - x1), 12) + 12,
            width - rx
        )
        rh = min(
            max(abs(y2 - y1), 12) + 12,
            height - ry
        )

        sat_patch = hsv[
            ry:ry + rh,
            rx:rx + rw,
            1
        ]

        if sat_patch.size and float(np.mean(sat_patch)) > 92:
            continue

        shadow_score = _line_shadow_score(
            gray,
            x1,
            y1,
            x2,
            y2
        )

        if shadow_score < 8:
            continue

        candidates.append(
            _rect(
                rx,
                ry,
                rw,
                rh,
                "Internal fold or crease line candidate",
                "condition"
            )
        )

        if len(candidates) >= 4:
            break

    return candidates


def analyze_document_condition(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return _empty_result(
            f"Cannot read image: {image_path}"
        )

    height, width = image.shape[:2]
    image_area = float(width * height) if width and height else 1.0

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        50,
        150
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return _empty_result(
            "No document boundary found"
        )

    large_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= image_area * 0.08
    ]

    if not large_contours:
        return _empty_result(
            "Document boundary is too small"
        )

    largest = max(
        large_contours,
        key=cv2.contourArea
    )

    contour_area = cv2.contourArea(largest)
    x, y, w, h = cv2.boundingRect(largest)
    box_area = float(w * h) if w and h else 1.0
    extent = contour_area / box_area

    perimeter = cv2.arcLength(
        largest,
        True
    )

    approx = cv2.approxPolyDP(
        largest,
        0.02 * perimeter,
        True
    )

    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull) or 1.0
    solidity = contour_area / hull_area

    document_box = {
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
        "area": int(contour_area),
        "solidity": round(solidity, 3),
        "extent": round(extent, 3),
        "corner_count": int(len(approx))
    }

    clean_boundary = (
        solidity >= 0.91
        and extent >= 0.74
        and len(approx) <= 8
    )

    damaged_regions = []
    debug_candidates = []
    reasons = []
    score = 0

    severe_boundary_damage = (
        solidity < 0.84
        or extent < 0.62
        or len(approx) > 12
    )

    moderate_boundary_damage = (
        solidity < 0.88
        or extent < 0.68
        or len(approx) > 10
    )

    if severe_boundary_damage:
        score += 24
        reason = "Irregular outer boundary suggests torn or damaged document edge"
        reasons.append(reason)
        damaged_regions.append(
            _rect(
                x,
                y,
                w,
                h,
                reason,
                "condition"
            )
        )

    elif moderate_boundary_damage:
        debug_candidates.append({
            "reason": "Mild outer-boundary irregularity",
            "solidity": round(solidity, 3),
            "extent": round(extent, 3),
            "corner_count": int(len(approx))
        })

    corner_size = int(
        max(
            min(w, h) * 0.11,
            28
        )
    )

    corners = [
        ("top-left", x, y),
        ("top-right", x + w - corner_size, y),
        ("bottom-left", x, y + h - corner_size),
        ("bottom-right", x + w - corner_size, y + h - corner_size),
    ]

    for name, cx, cy in corners:

        cx = max(
            0,
            min(cx, width - corner_size)
        )

        cy = max(
            0,
            min(cy, height - corner_size)
        )

        patch = gray[
            cy:cy + corner_size,
            cx:cx + corner_size
        ]

        edge_patch = edges[
            cy:cy + corner_size,
            cx:cx + corner_size
        ]

        signals = _corner_signals(
            patch,
            edge_patch
        )

        if not signals:
            continue

        candidate = {
            "corner": name,
            "signals": signals,
            "x": int(cx),
            "y": int(cy),
            "w": int(corner_size),
            "h": int(corner_size)
        }

        debug_candidates.append(candidate)

        # A clean rectangle with one noisy corner should not become a fold/tear.
        if clean_boundary and len(signals) < 3:
            continue

        if len(signals) >= 2 and moderate_boundary_damage:
            score += 10
            reason = f"Possible physical damage near {name} corner"
            reasons.append(reason)
            damaged_regions.append(
                _rect(
                    cx,
                    cy,
                    corner_size,
                    corner_size,
                    reason,
                    "condition"
                )
            )

    crease_regions = _diagonal_lines(
        edges,
        width,
        height,
        (x, y, w, h)
    )

    internal_crease_regions = _internal_crease_lines(
        image,
        gray,
        edges,
        width,
        height,
        (x, y, w, h)
    )

    if crease_regions and not clean_boundary:
        score += min(
            len(crease_regions) * 12,
            24
        )
        reasons.append(
            f"{len(crease_regions)} possible crease/fold line(s) detected"
        )
        damaged_regions.extend(
            crease_regions
        )
    else:
        debug_candidates.extend(
            {
                "reason": region["reason"],
                "x": region["x"],
                "y": region["y"],
                "w": region["w"],
                "h": region["h"]
            }
            for region in crease_regions
        )

    if internal_crease_regions:
        score += min(
            len(internal_crease_regions) * 12,
            26
        )
        reasons.append(
            f"{len(internal_crease_regions)} internal fold/crease line(s) detected"
        )
        damaged_regions.extend(
            internal_crease_regions
        )

    if score >= 34:
        confidence = "high"

    elif score >= 20:
        confidence = "medium"

    else:
        confidence = "low"
        score = 0
        reasons = []
        damaged_regions = []

    return {
        "fold_detected": (
            confidence in {"medium", "high"}
            and any("crease" in reason.lower() or "fold" in reason.lower() for reason in reasons)
        ),
        "tear_detected": (
            confidence in {"medium", "high"}
            and any("tear" in reason.lower() or "damage" in reason.lower() or "irregular" in reason.lower() for reason in reasons)
        ),
        "condition_score": int(min(score, 100)),
        "condition_confidence": confidence,
        "document_box": document_box,
        "damaged_regions": damaged_regions[:8],
        "debug_candidates": debug_candidates[:12],
        "reasons": reasons[:8]
    }
