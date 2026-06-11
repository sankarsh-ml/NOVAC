import cv2
import numpy as np


def _default_result(error=None):

    result = {
        "analysis_reliable": True,
        "rejection_recommended": False,
        "quality_score": 100,
        "damage_score": 0,
        "blur_score": 0,
        "glare_score": 0,
        "fold_tear_score": 0,
        "low_resolution": False,
        "poor_lighting": False,
        "excessive_noise": False,
        "reasons": []
    }

    if error:
        result["analysis_reliable"] = False
        result["rejection_recommended"] = True
        result["quality_score"] = 0
        result["damage_score"] = 100
        result["reasons"] = [error]
        result["error"] = error

    return result


def _score_from_laplacian(variance):

    if variance >= 180:
        return 0

    if variance >= 90:
        return 25

    if variance >= 45:
        return 55

    return 85


def _detector_scatter(detector_results, width, height, image_area):

    if not detector_results or image_area <= 0:
        return 0, 0

    regions = []

    for key in [
        "forgery",
        "mvss",
        "ela",
        "text_consistency"
    ]:
        result = detector_results.get(key, {}) or {}
        regions.extend(
            result.get("suspicious_regions", [])
            or result.get("annotation_regions", [])
            or []
        )

    if not regions:
        return 0, 0

    total_area = 0
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
                0 if cy < height * 0.5 else 1
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


def analyze_document_quality(
    image_path,
    ocr_result=None,
    document_condition_result=None,
    detector_results=None
):

    image = cv2.imread(image_path)

    if image is None:
        return _default_result(
            f"Cannot read image for document quality analysis: {image_path}"
        )

    height, width = image.shape[:2]
    image_area = float(width * height) if width and height else 1.0
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    laplacian_variance = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )
    blur_score = _score_from_laplacian(
        laplacian_variance
    )

    brightness = float(np.mean(gray))
    brightness_std = float(np.std(gray))
    grid_rows = np.array_split(gray, 4, axis=0)
    tile_means = []

    for row in grid_rows:
        for tile in np.array_split(row, 4, axis=1):
            if tile.size:
                tile_means.append(float(np.mean(tile)))

    lighting_range = max(tile_means) - min(tile_means) if tile_means else 0
    poor_lighting = (
        brightness < 65
        or brightness > 225
        or lighting_range > 105
        or brightness_std < 22
    )

    glare_ratio = float(
        np.mean(
            (gray > 245)
            & (hsv[:, :, 1] < 45)
        )
    )
    glare_score = int(
        min(glare_ratio * 1200, 100)
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )
    noise_residual = cv2.absdiff(
        gray,
        blur
    )
    noise_level = float(np.mean(noise_residual))
    excessive_noise = noise_level > 11.5

    low_resolution = (
        min(width, height) < 650
        or image_area < 700000
    )

    document_condition_result = document_condition_result or {}
    fold_tear_score = float(
        document_condition_result.get("condition_score", 0)
        or 0
    )
    condition_confidence = document_condition_result.get(
        "condition_confidence",
        "low"
    )
    condition_error = str(
        document_condition_result.get("error", "")
        or ""
    ).lower()
    document_boundary_unreliable = (
        not document_condition_result.get("document_box")
        or "too small" in condition_error
        or "no document boundary" in condition_error
    )

    if condition_confidence == "low":
        fold_tear_score *= 0.35

    avg_confidence = float(
        (ocr_result or {}).get("avg_confidence", 1.0)
        or 0
    )
    detector_scatter_score, scattered_region_count = _detector_scatter(
        detector_results,
        width,
        height,
        image_area
    )

    damage_score = min(
        100,
        max(
            fold_tear_score,
            blur_score * 0.45
            + glare_score * 0.20
            + (24 if poor_lighting else 0)
            + (22 if excessive_noise else 0)
            + (18 if low_resolution else 0)
            + (30 if document_boundary_unreliable else 0)
            + detector_scatter_score
        )
    )

    quality_score = 100
    quality_score -= blur_score * 0.38
    quality_score -= glare_score * 0.22
    quality_score -= 18 if poor_lighting else 0
    quality_score -= 16 if excessive_noise else 0
    quality_score -= 12 if low_resolution else 0
    quality_score -= 28 if document_boundary_unreliable else 0
    quality_score -= fold_tear_score * 0.32
    quality_score -= detector_scatter_score * 0.55

    if avg_confidence < 0.72:
        quality_score -= (0.72 - avg_confidence) * 55

    quality_score = int(
        min(max(round(quality_score), 0), 100)
    )
    damage_score = int(
        min(max(round(damage_score), 0), 100)
    )

    reasons = []

    if quality_score < 45:
        reasons.append(
            "Document quality is too poor for reliable fraud analysis"
        )

    if blur_score >= 55:
        reasons.append(
            "Document image appears blurred"
        )

    if poor_lighting:
        reasons.append(
            "Document lighting is uneven or outside reliable range"
        )

    if glare_score >= 25:
        reasons.append(
            "Glare or overexposed patches detected"
        )

    if excessive_noise:
        reasons.append(
            "Excessive image noise detected"
        )

    if low_resolution:
        reasons.append(
            "Document image resolution is low"
        )

    if document_boundary_unreliable:
        reasons.append(
            "Document boundary is unclear or too small in the image"
        )

    if fold_tear_score >= 35:
        reasons.append(
            "Document appears folded, torn, or physically damaged"
        )

    if detector_scatter_score >= 20:
        reasons.append(
            "Visual detectors fired broadly, which may indicate physical damage or scan noise"
        )

    if avg_confidence < 0.45 and (
        blur_score >= 55
        or excessive_noise
        or fold_tear_score >= 35
    ):
        reasons.append(
            "Text readability is too low for reliable verification"
        )

    rejection_recommended = (
        quality_score < 45
        or damage_score > 65
        or (
            document_boundary_unreliable
            and quality_score < 60
        )
        or (
            avg_confidence < 0.45
            and (
                blur_score >= 55
                or excessive_noise
                or fold_tear_score >= 35
            )
        )
    )

    if rejection_recommended:
        reasons.append(
            "Please upload a clearer, flatter, well-lit document scan"
        )

    return {
        "analysis_reliable": not rejection_recommended,
        "rejection_recommended": bool(rejection_recommended),
        "quality_score": quality_score,
        "damage_score": damage_score,
        "blur_score": int(min(max(round(blur_score), 0), 100)),
        "glare_score": int(min(max(round(glare_score), 0), 100)),
        "fold_tear_score": int(min(max(round(fold_tear_score), 0), 100)),
        "low_resolution": bool(low_resolution),
        "poor_lighting": bool(poor_lighting),
        "excessive_noise": bool(excessive_noise),
        "reasons": list(dict.fromkeys(reasons)),
        "metrics": {
            "width": int(width),
            "height": int(height),
            "laplacian_variance": round(laplacian_variance, 2),
            "brightness": round(brightness, 2),
            "lighting_range": round(lighting_range, 2),
            "noise_level": round(noise_level, 2),
            "scattered_region_count": scattered_region_count
        }
    }
