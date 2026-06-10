import cv2
import numpy as np


def _entropy(values):

    values = values.astype("uint8").flatten()

    histogram, _ = np.histogram(
        values,
        bins=32,
        range=(0, 256)
    )

    total = np.sum(histogram)

    if total <= 0:
        return 0.0

    probability = histogram / total
    probability = probability[
        probability > 0
    ]

    return float(
        -np.sum(
            probability
            * np.log2(probability)
        )
    )


def _tile_uniformity(gray):

    height, width = gray.shape[:2]

    tile_size = max(
        min(height, width) // 6,
        24
    )

    std_values = []

    for y in range(0, height - tile_size + 1, tile_size):

        for x in range(0, width - tile_size + 1, tile_size):

            tile = gray[
                y:y + tile_size,
                x:x + tile_size
            ]

            if tile.size:
                std_values.append(
                    float(np.std(tile))
                )

    if not std_values:
        return 0.0

    return float(
        np.std(std_values)
    )


def _text_edge_stats(gray, ocr_lines):

    if not ocr_lines:
        return 0.0, 0.0

    height, width = gray.shape[:2]
    values = []

    for line in ocr_lines[:30]:

        bbox = line.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        xs = [
            point[0]
            for point in bbox
        ]

        ys = [
            point[1]
            for point in bbox
        ]

        x = int(
            max(
                0,
                min(xs)
            )
        )

        y = int(
            max(
                0,
                min(ys)
            )
        )

        w = int(
            min(
                width - x,
                max(xs) - min(xs)
            )
        )

        h = int(
            min(
                height - y,
                max(ys) - min(ys)
            )
        )

        if w <= 0 or h <= 0:
            continue

        roi = gray[
            y:y + h,
            x:x + w
        ]

        if roi.size == 0:
            continue

        edges = cv2.Canny(
            roi,
            70,
            180
        )

        values.append(
            float(np.mean(edges > 0))
        )

    if not values:
        return 0.0, 0.0

    return float(np.mean(values)), float(np.std(values))


def _patch_repetition_score(gray):

    height, width = gray.shape[:2]

    patch_size = max(
        min(height, width) // 8,
        24
    )

    descriptors = []

    for y in range(0, height - patch_size + 1, patch_size):

        for x in range(0, width - patch_size + 1, patch_size):

            patch = gray[
                y:y + patch_size,
                x:x + patch_size
            ]

            if patch.size == 0:
                continue

            hist = cv2.calcHist(
                [patch],
                [0],
                None,
                [16],
                [0, 256]
            )

            hist = cv2.normalize(
                hist,
                hist
            ).flatten()

            descriptors.append(hist)

    if len(descriptors) < 6:
        return 0.0

    repeated = 0
    comparisons = 0

    for i in range(len(descriptors)):

        for j in range(i + 1, len(descriptors)):

            similarity = cv2.compareHist(
                descriptors[i].astype("float32"),
                descriptors[j].astype("float32"),
                cv2.HISTCMP_CORREL
            )

            comparisons += 1

            if similarity > 0.96:
                repeated += 1

    if comparisons == 0:
        return 0.0

    return repeated / comparisons


def _real_capture_evidence(image, gray):

    height, width = gray.shape[:2]

    image_area = float(width * height) if width and height else 1.0

    reasons = []

    tile_size = max(
        min(height, width) // 5,
        32
    )

    tile_means = []

    for y in range(0, height - tile_size + 1, tile_size):

        for x in range(0, width - tile_size + 1, tile_size):

            tile = gray[
                y:y + tile_size,
                x:x + tile_size
            ]

            if tile.size:
                tile_means.append(
                    float(np.mean(tile))
                )

    if tile_means and np.std(tile_means) > 34:
        reasons.append(
            "Uneven lighting or capture shadows"
        )

    blur = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    residual = cv2.absdiff(
        gray,
        blur
    )

    if float(np.std(residual)) >= 4.2:
        reasons.append(
            "Natural camera or scan noise present"
        )

    edges = cv2.Canny(
        gray,
        55,
        150
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    large_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= image_area * 0.08
    ]

    if large_contours:

        largest = max(
            large_contours,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(
            largest
        )

        contour_area = cv2.contourArea(
            largest
        )

        frame_ratio = (w * h) / image_area

        extent = contour_area / float(w * h or 1)

        if frame_ratio < 0.88:
            reasons.append(
                "Document photographed with visible background"
            )

        if extent < 0.72:
            reasons.append(
                "Perspective or imperfect physical document boundary"
            )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    value = hsv[:, :, 2]

    bright_ratio = float(
        np.mean(value > 242)
    )

    if 0.015 <= bright_ratio <= 0.20 and np.std(value) > 42:
        reasons.append(
            "Localized glare or laminated reflection"
        )

    return reasons[:5]


def _hard_physical_capture_reasons(real_capture_reasons):

    hard_reasons = []

    markers = [
        "visible background",
        "perspective",
        "physical document boundary",
        "localized glare",
        "laminated reflection",
        "capture shadows",
        "camera",
        "scan noise"
    ]

    for reason in real_capture_reasons:

        reason_lower = reason.lower()

        if any(marker in reason_lower for marker in markers):
            hard_reasons.append(reason)

    return hard_reasons


def _risk_label(score):

    if score >= 65:
        return "high"

    if score >= 40:
        return "medium"

    return "low"


def analyze_ai_generated_image(
    image_path,
    metadata_result=None,
    ocr_result=None
):

    image = cv2.imread(image_path)

    if image is None:
        return {
            "ai_generated_suspected": False,
            "strong_ai_generated_signal": False,
            "printed_document_likely": False,
            "synthetic_clean_document_signal": False,
            "synthetic_risk": "low",
            "positive_synthetic_evidence_count": 0,
            "real_capture_evidence_count": 0,
            "hard_physical_capture_count": 0,
            "ai_generation_score": 0,
            "confidence": 0,
            "reasons": [],
            "supporting_reasons": [],
            "real_capture_reasons": [],
            "suppressed_reasons": [],
            "error": f"Cannot read image: {image_path}"
        }

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    residual = cv2.absdiff(
        gray,
        blur
    )

    noise_std = float(
        np.std(residual)
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    texture_entropy = _entropy(
        residual
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    saturation_std = float(
        np.std(hsv[:, :, 1])
    )

    edges = cv2.Canny(
        gray,
        60,
        160
    )

    edge_density = float(
        np.mean(edges > 0)
    )

    tile_uniformity = _tile_uniformity(
        gray
    )

    patch_repetition = _patch_repetition_score(
        gray
    )

    metadata_result = metadata_result or {}

    metadata = metadata_result.get(
        "metadata",
        {}
    )

    ocr_result = ocr_result or {}

    ocr_lines = ocr_result.get(
        "lines",
        []
    )

    avg_confidence = ocr_result.get(
        "avg_confidence",
        0
    )

    text_edge_density, text_edge_variance = _text_edge_stats(
        gray,
        ocr_lines
    )

    real_capture_reasons = _real_capture_evidence(
        image,
        gray
    )

    real_capture_evidence_count = len(
        real_capture_reasons
    )

    hard_capture_reasons = _hard_physical_capture_reasons(
        real_capture_reasons
    )

    hard_physical_capture_count = len(
        hard_capture_reasons
    )

    supporting_score = 0
    positive_score = 0

    supporting_reasons = []
    positive_reasons = []
    suppressed_reasons = []

    # -----------------------------
    # Supporting weak signals
    # These are not enough alone.
    # -----------------------------

    if noise_std < 2.5:
        supporting_score += 4
        supporting_reasons.append(
            "Low natural camera noise"
        )

    if texture_entropy < 1.35:
        supporting_score += 4
        supporting_reasons.append(
            "Low residual texture entropy"
        )

    if edge_density < 0.028 and len(ocr_lines) > 4:
        supporting_score += 4
        supporting_reasons.append(
            "Readable OCR with weak visual edges"
        )

    # Missing metadata is common after uploads/compression.
    # Keep as context only. Do not increase score.
    if not metadata:
        supporting_reasons.append(
            "No image metadata available"
        )

    software = " ".join(
        str(value).lower()
        for value in metadata.values()
    )

    # -----------------------------
    # Strong positive synthetic clues
    # -----------------------------

    if any(
        marker in software
        for marker in [
            "openai",
            "midjourney",
            "stable diffusion",
            "dall",
            "canva",
            "firefly",
            "flux"
        ]
    ):
        positive_score += 35
        positive_reasons.append(
            "Metadata references generative or design software"
        )

    if patch_repetition > 0.42 and tile_uniformity < 6.0:
        positive_score += 16
        positive_reasons.append(
            "Repeated local texture patterns detected"
        )

    if (
        sharpness > 320
        and noise_std < 2.2
        and edge_density > 0.045
    ):
        positive_score += 14
        positive_reasons.append(
            "Very sharp edges with unusually low camera noise"
        )

    if (
        len(ocr_lines) >= 5
        and text_edge_variance > 0.12
        and avg_confidence < 0.92
    ):
        positive_score += 12
        positive_reasons.append(
            "Inconsistent text-edge rendering across OCR lines"
        )

    if (
        saturation_std < 12
        and tile_uniformity < 3.2
        and sharpness > 160
        and edge_density > 0.04
    ):
        positive_score += 12
        positive_reasons.append(
            "Overly uniform synthetic-looking surface with clean edges"
        )

    # -----------------------------
    # Clean digital/generated document signal
    # This does NOT depend heavily on OCR lines,
    # because OCR may fail or may not pass lines correctly.
    # -----------------------------

    clean_generated_surface_signal = (
        not metadata
        and texture_entropy < 1.75
        and noise_std < 4.5
        and tile_uniformity < 10.0
        and hard_physical_capture_count == 0
    )

    clean_digital_document_signal = (
        sharpness > 120
        and edge_density > 0.025
        and noise_std < 5.0
        and texture_entropy < 2.0
        and hard_physical_capture_count == 0
    )

    synthetic_clean_document_signal = (
        clean_generated_surface_signal
        or clean_digital_document_signal
    )

    if synthetic_clean_document_signal:
        positive_score += 45
        positive_reasons.append(
            "Clean digital document surface without real camera or scan evidence"
        )

    if (
        real_capture_evidence_count == 0
        and not metadata
        and texture_entropy < 1.75
        and tile_uniformity < 8.0
        and noise_std < 5.0
    ):
        positive_score += 18
        positive_reasons.append(
            "Clean document surface lacks real camera or scan artifacts"
        )

    # -----------------------------
    # Suppression logic
    # Important:
    # readable OCR alone must NOT suppress AI detection.
    # Only hard physical capture evidence should suppress.
    # -----------------------------

    printed_document_likely = (
        len(ocr_lines) >= 3
        and avg_confidence >= 0.80
        and len(positive_reasons) == 0
        and hard_physical_capture_count >= 2
        and not synthetic_clean_document_signal
    )

    if printed_document_likely:

        suppressed_reasons.extend([
            "Stable OCR text structure",
            "No positive synthetic artifacts detected"
        ])

        suppressed_reasons.extend(
            hard_capture_reasons[:3]
        )

        if noise_std < 4 or texture_entropy < 1.6:
            suppressed_reasons.append(
                "Flat printed or laminated document surface"
            )

    # -----------------------------
    # Final score
    # -----------------------------

    score = positive_score

    if not printed_document_likely:
        score += min(
            supporting_score,
            12
        )

    elif positive_score:
        score += min(
            supporting_score,
            6
        )

    score = int(
        min(score, 100)
    )

    positive_count = len(
        positive_reasons
    )

    synthetic_risk = _risk_label(
        score
    )

    ai_generated_suspected = (
        score >= 40
        and positive_count >= 1
        and not printed_document_likely
    )

    strong_ai_generated_signal = (
        score >= 65
        and positive_count >= 2
        and hard_physical_capture_count == 0
    ) or any(
        "generative" in reason.lower()
        for reason in positive_reasons
    )

    return {
        "ai_generated_suspected": ai_generated_suspected,
        "strong_ai_generated_signal": strong_ai_generated_signal,
        "printed_document_likely": printed_document_likely,
        "synthetic_clean_document_signal": synthetic_clean_document_signal,
        "synthetic_risk": synthetic_risk,
        "positive_synthetic_evidence_count": positive_count,
        "real_capture_evidence_count": real_capture_evidence_count,
        "hard_physical_capture_count": hard_physical_capture_count,
        "ai_generation_score": score,
        "confidence": round(score / 100, 2),
        "statistics": {
            "noise_std": round(noise_std, 2),
            "sharpness": round(sharpness, 2),
            "texture_entropy": round(texture_entropy, 2),
            "saturation_std": round(saturation_std, 2),
            "edge_density": round(edge_density, 4),
            "tile_uniformity": round(tile_uniformity, 2),
            "patch_repetition": round(patch_repetition, 3),
            "text_edge_density": round(text_edge_density, 4),
            "text_edge_variance": round(text_edge_variance, 4),
            "clean_generated_surface_signal": clean_generated_surface_signal,
            "clean_digital_document_signal": clean_digital_document_signal
        },
        "reasons": positive_reasons[:8],
        "supporting_reasons": supporting_reasons[:8],
        "real_capture_reasons": real_capture_reasons[:8],
        "suppressed_reasons": suppressed_reasons[:8]
    }