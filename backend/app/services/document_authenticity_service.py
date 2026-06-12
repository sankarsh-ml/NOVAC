import os
import re
from pathlib import Path

import cv2
import numpy as np

try:
    import fitz
except Exception:
    fitz = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


AADHAAR_INDICATORS = [
    "enrolment no",
    "unique identification authority of india",
    "government of india",
    "aadhaar",
    "vid",
    "details as on",
    "aadhaar no. issued",
    "address",
]

PLACEHOLDER_PATTERNS = [
    r"\b1234\s*5678\s*9012\b",
    r"\b9876\s*5432\s*1098\s*7654\b",
    r"\bxxxx\s*xxxx\s*xxxx\b",
    r"\brohan\s+kumar\b",
    r"\bname\s*/\s*naam\b",
]


def _clamp(value, low=0, high=100):
    return int(min(max(round(float(value)), low), high))


def _add_reason(reasons, reason):
    if reason and reason not in reasons:
        reasons.append(reason)


def _normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _inspect_pdf_with_fitz(file_path):
    if fitz is None:
        return None

    doc = fitz.open(file_path)
    text_parts = []
    image_count = 0
    drawing_count = 0

    for page in doc:
        text_parts.append(page.get_text() or "")
        image_count += len(page.get_images(full=True))
        drawing_count += len(page.get_drawings())

    metadata = doc.metadata or {}
    page_count = doc.page_count
    doc.close()

    return {
        "text": "\n".join(text_parts),
        "page_count": page_count,
        "image_count": image_count,
        "drawing_count": drawing_count,
        "metadata": metadata,
        "metadata_available": bool(metadata),
    }


def _inspect_pdf_with_pypdf(file_path):
    if PdfReader is None:
        return None

    reader = PdfReader(file_path)
    text_parts = []
    image_count = 0

    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
        resources = page.get("/Resources") or {}
        xobjects = resources.get("/XObject") if hasattr(resources, "get") else None

        if not xobjects:
            continue

        for item in xobjects.values():
            try:
                obj = item.get_object()
                if obj.get("/Subtype") == "/Image":
                    image_count += 1
            except Exception:
                continue

    metadata = {
        str(key): str(value)
        for key, value in (reader.metadata or {}).items()
    }

    return {
        "text": "\n".join(text_parts),
        "page_count": len(reader.pages),
        "image_count": image_count,
        "drawing_count": 0,
        "metadata": metadata,
        "metadata_available": bool(metadata),
    }


def _inspect_pdf(file_path):
    info = _inspect_pdf_with_fitz(file_path)

    if info is not None:
        return info

    info = _inspect_pdf_with_pypdf(file_path)

    if info is not None:
        return info

    return {
        "text": "",
        "page_count": 0,
        "image_count": 0,
        "drawing_count": 0,
        "metadata": {},
        "metadata_available": False,
        "error": "PDF inspection unavailable: install PyMuPDF or pypdf",
    }


def _pdf_authenticity(file_path):
    pdf_info = _inspect_pdf(file_path)
    text = pdf_info.get("text", "")
    normalized = _normalize_text(text)
    indicator_hits = [
        indicator
        for indicator in AADHAAR_INDICATORS
        if indicator in normalized
    ]
    has_embedded_text = len(normalized) >= 120
    has_address_block = "address" in normalized or "pin code" in normalized
    has_id_patterns = bool(
        re.search(r"\b\d{4}\s+\d{4}\s+\d{4}\b", normalized)
        or re.search(r"\bvid\s*[:\-]?\s*\d{4}\s+\d{4}\s+\d{4}\s+\d{4}\b", normalized)
    )

    embedded_text_score = _clamp(
        min(len(normalized), 900) / 900 * 100
    )
    structured_score = _clamp(
        len(indicator_hits) * 13
        + (15 if has_address_block else 0)
        + (15 if has_id_patterns else 0)
    )
    official_layout_score = _clamp(
        (18 if pdf_info.get("page_count", 0) >= 1 else 0)
        + (22 if pdf_info.get("image_count", 0) >= 1 else 0)
        + (20 if "aadhaar" in normalized else 0)
        + (20 if "government of india" in normalized else 0)
        + (20 if "details as on" in normalized or "aadhaar no. issued" in normalized else 0)
    )
    official_digital_pdf_detected = (
        has_embedded_text
        and structured_score >= 55
        and official_layout_score >= 55
    )

    reasons = []
    synthetic_score = 18
    authenticity_score = 82
    acquisition_type = "scanned_document"

    if official_digital_pdf_detected:
        acquisition_type = "official_digital_pdf"
        synthetic_score = min(synthetic_score, 35)
        authenticity_score = max(authenticity_score, 70)
        _add_reason(reasons, "Official digital PDF structure detected")
        _add_reason(reasons, "Embedded document text present")
    elif has_embedded_text:
        acquisition_type = "generated_image_or_template"
        synthetic_score = 48
        authenticity_score = 52
        _add_reason(reasons, "Embedded PDF text present but official document structure is weak")
    else:
        synthetic_score = 38
        authenticity_score = 62
        _add_reason(reasons, "PDF has limited embedded document text")

    return {
        "synthetic_detected": False if official_digital_pdf_detected else synthetic_score >= 70,
        "synthetic_score": int(synthetic_score),
        "authenticity_score": int(authenticity_score),
        "ai_generated_score": int(synthetic_score),
        "analysis_reliable": True,
        "acquisition_type": acquisition_type,
        "official_digital_pdf_detected": bool(official_digital_pdf_detected),
        "reasons": reasons,
        "metrics": {
            "pdf_embedded_text_score": int(embedded_text_score),
            "pdf_structured_document_score": int(structured_score),
            "pdf_official_layout_score": int(official_layout_score),
            "pdf_has_images": bool(pdf_info.get("image_count", 0) > 0),
            "pdf_has_vector_text": bool(has_embedded_text),
            "pdf_metadata_available": bool(pdf_info.get("metadata_available")),
            "pdf_page_count": int(pdf_info.get("page_count", 0) or 0),
            "pdf_image_count": int(pdf_info.get("image_count", 0) or 0),
            "pdf_drawing_count": int(pdf_info.get("drawing_count", 0) or 0),
            "pdf_text_length": int(len(text)),
            "pdf_aadhaar_indicator_hits": indicator_hits,
            "official_digital_pdf_detected": bool(official_digital_pdf_detected),
            "pdf_metadata": pdf_info.get("metadata", {}),
            "pdf_inspection_error": pdf_info.get("error"),
        },
    }


def _image_metrics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = gray.shape[:2]

    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = cv2.absdiff(gray, blur)
    noise_level = float(np.mean(residual))
    noise_std = float(np.std(residual))
    brightness = float(np.mean(gray))
    brightness_std = float(np.std(gray))
    saturation = float(np.mean(hsv[:, :, 1]))
    white_mask = (gray > 235) & (hsv[:, :, 1] < 45)
    white_area_ratio = float(np.mean(white_mask))
    soft_white_mask = (gray > 220) & (hsv[:, :, 1] < 70)
    white_area_uniformity = (
        100 - min(float(np.std(gray[soft_white_mask])) * 10, 100)
        if np.any(soft_white_mask)
        else 0
    )
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))

    tile_means = []
    for row in np.array_split(gray, 4, axis=0):
        for tile in np.array_split(row, 4, axis=1):
            if tile.size:
                tile_means.append(float(np.mean(tile)))

    lighting_range = max(tile_means) - min(tile_means) if tile_means else 0
    low_noise = noise_level < 2.2 and laplacian_variance > 80
    perfect_edges = laplacian_variance > 250 and 0.018 <= edge_density <= 0.085
    flat_region_ratio = white_area_ratio

    return {
        "width": int(width),
        "height": int(height),
        "laplacian_variance": round(laplacian_variance, 3),
        "noise_level": round(noise_level, 3),
        "noise_std": round(noise_std, 3),
        "brightness": round(brightness, 3),
        "brightness_std": round(brightness_std, 3),
        "saturation": round(saturation, 3),
        "lighting_range": round(lighting_range, 3),
        "flat_region_ratio": round(flat_region_ratio, 4),
        "white_area_uniformity": round(white_area_uniformity, 3),
        "edge_density": round(edge_density, 4),
        "low_noise": bool(low_noise),
        "perfect_edges": bool(perfect_edges),
    }


def _placeholder_score(ocr_result, file_path):
    text = _normalize_text((ocr_result or {}).get("text", ""))
    filename = _normalize_text(Path(file_path).name)
    haystack = f"{text} {filename}"
    hits = [
        pattern
        for pattern in PLACEHOLDER_PATTERNS
        if re.search(pattern, haystack, flags=re.IGNORECASE)
    ]

    score = min(len(hits) * 18, 45)

    if "chatgpt" in filename or "dall" in filename or "generated" in filename:
        score += 25
        hits.append("generated-image filename marker")

    return _clamp(score), hits


def _camera_capture_score(metrics, extension):
    score = 0

    if extension in {".jpg", ".jpeg"}:
        score += 12
    if metrics["laplacian_variance"] < 80:
        score += 25
    elif metrics["laplacian_variance"] < 180:
        score += 12
    if metrics["flat_region_ratio"] < 0.12:
        score += 14
    if metrics["brightness_std"] >= 25:
        score += 10
    if metrics["lighting_range"] >= 65:
        score += 14
    if metrics["saturation"] >= 45:
        score += 8
    if metrics["edge_density"] < 0.025:
        score += 8

    return _clamp(score)


def _raster_authenticity(file_path, ocr_result=None):
    image = cv2.imread(file_path)

    if image is None:
        return {
            "synthetic_detected": False,
            "synthetic_score": 0,
            "authenticity_score": 0,
            "ai_generated_score": 0,
            "analysis_reliable": False,
            "acquisition_type": "unknown",
            "official_digital_pdf_detected": False,
            "reasons": [f"Cannot read image for authenticity analysis: {file_path}"],
            "metrics": {},
        }

    extension = Path(file_path).suffix.lower()
    metrics = _image_metrics(image)
    camera_capture_score = _camera_capture_score(metrics, extension)
    placeholder_score, placeholder_hits = _placeholder_score(ocr_result, file_path)

    clean_render_score = 0
    clean_render_score += 22 if metrics["flat_region_ratio"] >= 0.28 else 0
    clean_render_score += 16 if metrics["white_area_uniformity"] >= 42 else 0
    clean_render_score += 14 if metrics["perfect_edges"] else 0
    clean_render_score += 10 if metrics["laplacian_variance"] >= 450 else 0
    clean_render_score += 10 if extension == ".png" else 0
    clean_render_score = _clamp(clean_render_score)

    weak_camera_trace_score = max(0, 55 - camera_capture_score)
    synthetic_score = (
        clean_render_score * 0.62
        + placeholder_score
        + weak_camera_trace_score * 0.35
    )

    strong_ai_evidence = placeholder_score >= 30 or clean_render_score >= 55

    if camera_capture_score >= 55 and not strong_ai_evidence:
        synthetic_score = min(synthetic_score, 44)

    if placeholder_score >= 45 and camera_capture_score < 45:
        synthetic_score = max(synthetic_score, 76)
    elif placeholder_score >= 30 and clean_render_score >= 38:
        synthetic_score = max(synthetic_score, 72)

    synthetic_score = _clamp(synthetic_score)
    synthetic_detected = synthetic_score >= 70
    authenticity_score = _clamp(100 - synthetic_score)

    if camera_capture_score >= 55 and not synthetic_detected:
        acquisition_type = "camera_capture"
    elif synthetic_detected or placeholder_score >= 30:
        acquisition_type = "generated_image_or_template"
    elif metrics["flat_region_ratio"] >= 0.20:
        acquisition_type = "scanned_document"
    else:
        acquisition_type = "camera_capture"

    reasons = []

    if synthetic_detected:
        _add_reason(reasons, "Document appears digitally generated or synthetic")

    if camera_capture_score >= 55 and not synthetic_detected:
        _add_reason(reasons, "Natural camera/print acquisition traces detected")
    elif camera_capture_score < 45:
        _add_reason(reasons, "Weak natural camera/print acquisition traces")

    if placeholder_hits:
        _add_reason(reasons, "Placeholder-like ID number pattern detected")

    if not reasons:
        _add_reason(reasons, "No strong synthetic document indicators detected")

    return {
        "synthetic_detected": bool(synthetic_detected),
        "synthetic_score": int(synthetic_score),
        "authenticity_score": int(authenticity_score),
        "ai_generated_score": int(synthetic_score),
        "analysis_reliable": True,
        "acquisition_type": acquisition_type,
        "official_digital_pdf_detected": False,
        "reasons": reasons,
        "metrics": {
            **metrics,
            "camera_capture_score": int(camera_capture_score),
            "clean_render_score": int(clean_render_score),
            "weak_camera_trace_score": int(weak_camera_trace_score),
            "placeholder_score": int(placeholder_score),
            "placeholder_hits": placeholder_hits,
            "flat_region_suspicion_applied": True,
            "raster_cleanliness_signals_applied": True,
        },
    }


def analyze_document_authenticity(
    file_path,
    analysis_image_path=None,
    ocr_result=None,
    embedded_text=None,
):
    """
    Detects likely synthetic/generated document images separately from quality.

    Official digital PDFs are inspected as PDFs first so clean typography,
    vector text, and low camera noise are treated as normal issuance signals.
    """

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return _pdf_authenticity(file_path)

    enriched_ocr_result = {
        **(ocr_result or {})
    }

    if embedded_text and not enriched_ocr_result.get("text"):
        enriched_ocr_result["text"] = embedded_text

    return _raster_authenticity(
        analysis_image_path or file_path,
        ocr_result=enriched_ocr_result,
    )
