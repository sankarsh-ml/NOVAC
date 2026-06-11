from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import os
import re
import shutil
import uuid

from app.services.ocr_service import extract_text

from app.services.pdf_service import (
    extract_pdf_text,
    pdf_to_image
)

from app.services.metadata_service import (
    analyze_metadata
)

from app.services.ela_services import (
    analyze_ela
)

from app.services.correlation_service import (
    correlate
)

from app.services.scoring_service import (
    calculate_fraud_score
)

from app.services.tampering_runner import (
    analyze_tampering
)

from app.services.storage_service import (
    save_analysis
)

from app.services.annotation_service import (
    create_annotated_image
)

from app.services.preprocessing_service import (
    remove_qr_code_with_metadata
)

from app.services.masking_detection_service import detect_masking

from app.services.document_condition_service import analyze_document_condition
from app.services.document_quality_service import analyze_document_quality

from app.services.photo_replacement_service import analyze_photo_replacement

from app.services.visual_consistency_service import analyze_visual_consistency

from app.services.field_extraction_service import extract_fields

from app.services.forgery_localization_service import analyze_forgery_localization

from app.services.text_consistency_service import analyze_text_consistency

from app.services.visual_region_utils import (
    any_region_near,
    box_iou,
    classify_region_context,
    classify_regions,
    normalize_score
)


router = APIRouter()

UPLOAD_DIR = "uploads"
CONFIDENCE_THRESHOLD = 0.80
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png"
}

os.makedirs(UPLOAD_DIR, exist_ok=True)


def convert_keys_to_strings(obj):

    if isinstance(obj, dict):

        return {
            str(k): convert_keys_to_strings(v)
            for k, v in obj.items()
        }

    elif isinstance(obj, list):

        return [
            convert_keys_to_strings(item)
            for item in obj
        ]

    return obj


def response_path(path):

    if not path:
        return None

    return path.replace("\\", "/")


def sanitize_upload_filename(filename):

    original_name = os.path.basename(filename or "").strip()

    if not original_name:
        raise HTTPException(
            status_code=400,
            detail="Missing upload filename"
        )

    stem, extension = os.path.splitext(original_name)
    extension = extension.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a PDF, JPG, JPEG, or PNG file."
        )

    safe_stem = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        stem
    ).strip("._-")

    if not safe_stem:
        safe_stem = "document"

    unique_prefix = uuid.uuid4().hex[:12]

    return (
        original_name,
        f"{unique_prefix}_{safe_stem}{extension}",
        extension
    )


def run_required_step(step_name, func, *args):

    try:
        return func(*args)

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{step_name} failed: {exc}"
        )


def safe_metadata_analysis(file_path):

    try:
        return analyze_metadata(file_path)

    except Exception as exc:
        return {
            "file_type": None,
            "metadata": {},
            "flags": [
                f"Metadata analysis failed: {exc}"
            ],
            "risk_score": 0
        }


def safe_ela_analysis(image_path):

    try:
        return analyze_ela(image_path)

    except Exception as exc:
        return {
            "ela_score": 0,
            "statistics": {
                "average_brightness": 0,
                "max_brightness": 0,
                "suspicious_region_count": 0,
                "total_suspicious_area": 0
            },
            "ela_image": None,
            "marked_image": None,
            "suspicious_regions": [],
            "error": f"ELA analysis failed: {exc}"
        }


def safe_tampering_analysis(image_path):

    try:
        return analyze_tampering(image_path)

    except Exception as exc:
        return {
            "tampering_detected": False,
            "tampering_score": 0,
            "tampered_area_percent": 0,
            "mask_path": None,
            "mvss_confidence": 0,
            "raw_region_count": 0,
            "scoring_region_count": 0,
            "annotation_region_count": 0,
            "suspicious_region_count": 0,
            "suspicious_regions": [],
            "suppressed_regions": [],
            "suppressed_region_count": 0,
            "error": f"MVSS analysis failed: {exc}"
        }


def _box_iou(region, other):
    return box_iou(region, other)


def _mvss_score_from_area(tampered_percent, confidence):

    if confidence < 0.35 or tampered_percent <= 0:
        return 0

    if confidence >= 0.75 and tampered_percent >= 2:
        return 30

    if confidence >= 0.65 and tampered_percent >= 5:
        return 35

    if confidence >= 0.55 and tampered_percent >= 0.8:
        return 20

    if tampered_percent >= 1.5:
        return 15

    return 10


def filter_mvss_regions(
    tampering_result,
    preprocessing_result,
    image_path,
    supporting_regions=None,
    ocr_lines=None,
    photo_regions=None,
    damage_regions=None
):

    result = {
        **(tampering_result or {})
    }

    image = cv2.imread(image_path)

    if image is None:
        return result

    height, width = image.shape[:2]
    image_area = float(width * height) if width and height else 1.0
    confidence = float(
        result.get("mvss_confidence", 0)
        or 0
    )
    min_area = max(1200, image_area * 0.002)
    scoring_area_ratio = 0.003
    removed_regions = preprocessing_result.get(
        "removed_regions",
        preprocessing_result.get("qr_regions", [])
    ) or []
    supporting_regions = supporting_regions or []

    scoring_regions = []
    annotation_regions = []
    suppressed_regions = list(
        result.get("suppressed_regions", [])
        or []
    )
    raw_regions = list(result.get("suspicious_regions", []) or [])

    for region in raw_regions:

        w = int(region.get("w", 0))
        h = int(region.get("h", 0))
        area = float(
            region.get("area")
            or (w * h)
        )
        area_ratio = area / image_area
        base_region = {
            **region,
            "w": w,
            "h": h,
            "area": int(area),
            "area_ratio": round(area_ratio, 5),
            "confidence": confidence,
            "source": "MVSS",
            "type": "mvss",
            "scoring_eligible": False,
            "annotation_eligible": False,
            "suppression_reason": None
        }
        base_region = classify_region_context(
            base_region,
            "MVSS",
            image.shape,
            ocr_lines=ocr_lines,
            qr_regions=removed_regions,
            photo_regions=photo_regions,
            damage_regions=damage_regions,
            default_type="mvss"
        )
        has_support = any_region_near(
            base_region,
            supporting_regions,
            image_shape=image.shape
        )
        suppression_reason = base_region.get("suppression_reason")

        if w < 18 or h < 18:
            suppression_reason = "Region dimensions below MVSS scoring threshold"

        elif area_ratio > 0.35 and confidence < 0.85:
            suppression_reason = "Region covers too much of document for reliable MVSS evidence"

        elif area < min_area and not (confidence >= 0.85 and has_support):
            suppression_reason = "Region too small for reliable MVSS evidence"

        elif area_ratio < scoring_area_ratio and not (confidence >= 0.85 and has_support):
            suppression_reason = "Region area ratio below MVSS scoring threshold"

        else:
            aspect = max(w, h) / float(max(min(w, h), 1))

            if aspect > 8 and area_ratio < 0.03 and not has_support:
                suppression_reason = "Region is a long thin noise strip"

        overlapping_qr = None
        overlap_score = 0

        if not suppression_reason:
            for removed_region in removed_regions:
                iou = _box_iou(
                    base_region,
                    removed_region
                )

                if iou > overlap_score:
                    overlap_score = iou
                    overlapping_qr = removed_region

            if overlapping_qr and overlap_score > 0.30:
                suppression_reason = "Region overlaps removed QR-like area"

        if suppression_reason:
            suppressed = {
                **base_region,
                "suppression_reason": suppression_reason,
                "reason": suppression_reason
            }

            if overlapping_qr:
                suppressed["overlapping_region"] = overlapping_qr
                suppressed["iou"] = round(overlap_score, 3)

            suppressed_regions.append(suppressed)
            continue

        valid_region = {
            **base_region,
            "scoring_eligible": True,
            "annotation_eligible": True,
            "reason": "MVSS detected meaningful suspicious visual manipulation region"
        }
        scoring_regions.append(valid_region)
        annotation_regions.append(valid_region)

    scoring_regions = sorted(
        scoring_regions,
        key=lambda item: item.get("area", 0),
        reverse=True
    )[:3]
    annotation_regions = sorted(
        annotation_regions,
        key=lambda item: item.get("area", 0),
        reverse=True
    )[:3]

    total_area = sum(
        region.get("area", 0)
        for region in scoring_regions
    )
    tampered_percent = (
        total_area / image_area
    ) * 100
    reasons = []

    if scoring_regions:
        reasons.extend([
            "MVSS detected scoring-eligible suspicious visual manipulation region",
            "Suspicious MVSS region passed area and confidence filters"
        ])

    if any(
        item.get("reason") == "Region overlaps removed QR-like area"
        for item in suppressed_regions
    ):
        reasons.append(
            "QR-overlapping regions suppressed before scoring"
        )

    result["raw_region_count"] = len(raw_regions) + int(
        result.get("raw_region_count", 0)
        if not raw_regions
        else 0
    )
    result["scoring_region_count"] = len(scoring_regions)
    result["annotation_region_count"] = len(annotation_regions)
    result["suspicious_regions"] = scoring_regions
    result["annotation_regions"] = annotation_regions
    result["suspicious_region_count"] = len(scoring_regions)
    result["valid_suspicious_region_count"] = len(scoring_regions)
    result["suppressed_regions"] = suppressed_regions
    result["suppressed_region_count"] = len(suppressed_regions)
    result["tampered_area_percent"] = round(tampered_percent, 2)
    result["tampering_score"] = float(
        min(
            _mvss_score_from_area(
            tampered_percent,
            confidence
            ),
            40
        )
    )
    result["tampering_detected"] = len(scoring_regions) > 0
    result["reasons"] = reasons

    return result


def safe_document_condition_analysis(image_path):

    try:
        return analyze_document_condition(image_path)

    except Exception as exc:
        return {
            "fold_detected": False,
            "tear_detected": False,
            "condition_score": 0,
            "condition_confidence": "low",
            "document_box": None,
            "damaged_regions": [],
            "debug_candidates": [],
            "reasons": [],
            "error": f"Document condition analysis failed: {exc}"
        }


def safe_photo_replacement_analysis(image_path):

    try:
        return analyze_photo_replacement(image_path)

    except Exception as exc:
        return {
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
            "suppressed_reasons": [],
            "error": f"Photo replacement analysis failed: {exc}"
        }


def safe_visual_consistency_analysis(
    image_path,
    region_groups
):

    try:
        return analyze_visual_consistency(
            image_path,
            region_groups
        )

    except Exception as exc:
        return {
            "consistency_score": 0,
            "inconsistent_regions": [],
            "reasons": [],
            "error": f"Visual consistency analysis failed: {exc}"
        }


def safe_forgery_localization_analysis(image_path):

    try:
        return analyze_forgery_localization(
            image_path
        )

    except Exception as exc:
        return {
            "model_available": False,
            "model": "TruFor",
            "manipulation_detected": False,
            "forgery_score": 0,
            "confidence": 0,
            "suspicious_regions": [],
            "localization_map_path": None,
            "reasons": [f"Forgery localization failed: {exc}"],
            "model_error": f"Forgery localization failed: {exc}",
            "elapsed_time_seconds": 0
        }


def safe_document_quality_analysis(
    image_path,
    ocr_result=None,
    document_condition_result=None,
    detector_results=None
):

    try:
        return analyze_document_quality(
            image_path,
            ocr_result=ocr_result,
            document_condition_result=document_condition_result,
            detector_results=detector_results
        )

    except Exception as exc:
        return {
            "analysis_reliable": False,
            "rejection_recommended": True,
            "quality_score": 0,
            "damage_score": 100,
            "blur_score": 0,
            "glare_score": 0,
            "fold_tear_score": 0,
            "low_resolution": False,
            "poor_lighting": False,
            "excessive_noise": False,
            "reasons": [
                f"Document quality analysis failed: {exc}"
            ],
            "error": f"Document quality analysis failed: {exc}"
        }


def safe_text_consistency_analysis(
    image_path,
    ocr_lines,
    extracted_text,
    visual_regions=None
):

    try:
        return analyze_text_consistency(
            image_path,
            ocr_lines,
            extracted_text,
            visual_regions=visual_regions
        )

    except Exception as exc:
        return {
            "font_mismatch_detected": False,
            "field_mismatch_score": 0,
            "suspicious_fields": [],
            "suspicious_regions": [],
            "comparisons_used": 0,
            "comparisons_skipped": 0,
            "reasons": [],
            "error": f"Text consistency analysis failed: {exc}"
        }


def safe_field_extraction(ocr_result):

    try:
        return extract_fields(
            ocr_result
        )

    except Exception as exc:
        return {
            "fields": {},
            "field_confidences": {},
            "field_sources": {},
            "field_details": {},
            "possible_values": [],
            "unmapped_lines": [],
            "field_count": 0,
            "possible_value_count": 0,
            "extraction_mode": "strict_label_anchor",
            "error": f"Field extraction failed: {exc}"
        }


def dedupe_regions(regions, iou_threshold=0.50):

    selected = []

    for region in regions or []:
        duplicate = False

        for existing in selected:
            if _box_iou(region, existing) >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            selected.append(region)

    return selected


def build_visual_manipulation_analysis(
    tampering_result,
    forgery_result,
    text_consistency_result,
    ela_result
):

    regions = []
    reasons = []

    for region in tampering_result.get("suspicious_regions", []) or []:
        if not region.get("scoring_eligible", True):
            continue

        regions.append({
            **region,
            "type": "mvss"
        })

    for region in forgery_result.get("suspicious_regions", []) or []:
        regions.append({
            **region,
            "type": "forgery_model"
        })

    for region in text_consistency_result.get("suspicious_regions", []) or []:
        regions.append({
            **region,
            "type": "text_consistency"
        })

    if tampering_result.get("tampering_detected") and tampering_result.get("scoring_region_count", tampering_result.get("suspicious_region_count", 0)) > 0:
        reasons.append(
            "MVSS detected scoring-eligible suspicious visual manipulation region"
        )

    if forgery_result.get("manipulation_detected"):
        reasons.append(
            "TruFor detected possible manipulated region"
        )

    if text_consistency_result.get("font_mismatch_detected"):
        reasons.append(
            "Text field style differs from surrounding document text"
        )

    score = min(
        100,
        int(
            tampering_result.get("tampering_score", 0) * 4
            + normalize_score(forgery_result.get("forgery_score", 0)) * 0.35
            + text_consistency_result.get("field_mismatch_score", 0) * 0.5
            + min(len(ela_result.get("suspicious_regions", []) or []) * 3, 10)
        )
    )

    return {
        "visual_manipulation_detected": bool(regions),
        "visual_manipulation_score": score,
        "regions": dedupe_regions(regions),
        "reasons": reasons,
        "signals": {
            "mvss": {
                "tampering_detected": tampering_result.get("tampering_detected", False),
                "tampering_score": tampering_result.get("tampering_score", 0),
                "suspicious_region_count": tampering_result.get("suspicious_region_count", 0),
                "raw_region_count": tampering_result.get("raw_region_count", 0),
                "scoring_region_count": tampering_result.get("scoring_region_count", 0)
            },
            "trufor": {
                "model_available": forgery_result.get("model_available", False),
                "manipulation_detected": forgery_result.get("manipulation_detected", False),
                "forgery_score": forgery_result.get("forgery_score", 0)
            },
            "text_consistency": {
                "font_mismatch_detected": text_consistency_result.get("font_mismatch_detected", False),
                "field_mismatch_score": text_consistency_result.get("field_mismatch_score", 0)
            }
        }
    }


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    original_filename, stored_filename, extension = sanitize_upload_filename(
        file.filename
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        stored_filename
    )

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save uploaded file: {exc}"
        )

    if os.path.getsize(file_path) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    metadata_result = convert_keys_to_strings(
        safe_metadata_analysis(file_path)
    )

    embedded_text = ""

    if extension == ".pdf":

        embedded_text = run_required_step(
            "PDF text extraction",
            extract_pdf_text,
            file_path
        )

        image_path = run_required_step(
            "PDF image conversion",
            pdf_to_image,
            file_path
        )

        ocr_result = run_required_step(
            "OCR analysis",
            extract_text,
            image_path
        )

        analysis_image_path = image_path

    else:

        ocr_result = run_required_step(
            "OCR analysis",
            extract_text,
            file_path
        )

        analysis_image_path = file_path

    if (
        ocr_result["avg_confidence"]
        < CONFIDENCE_THRESHOLD
    ):

        field_extraction_result = safe_field_extraction(
            ocr_result
        )

        response = {

            "filename": original_filename,

            "stored_filename": stored_filename,

            "status": "unclear_image",

            "avg_confidence":
                ocr_result["avg_confidence"],

            "metadata_analysis":
                metadata_result,

            "message":
                "Image quality too low for reliable fraud analysis.",

            "lines":
                ocr_result["lines"],

            "field_extraction_analysis":
                field_extraction_result,

            "document_quality_analysis":
                safe_document_quality_analysis(
                    analysis_image_path,
                    ocr_result=ocr_result
                ),

            "file_path":
                response_path(file_path),

            "analysis_image_path":
                response_path(analysis_image_path),

            "annotated_image_path": None

        }

        case_id = save_analysis(
            response
        )

        response["case_id"] = case_id

        return response

    masking_result = detect_masking(
        ocr_result
    )

    field_extraction_result = safe_field_extraction(
        ocr_result
    )

    document_condition_result = safe_document_condition_analysis(
        analysis_image_path
    )

    photo_replacement_result = safe_photo_replacement_analysis(
        analysis_image_path
    )

    qr_preprocessing_result = remove_qr_code_with_metadata(
        analysis_image_path
    )

    mvss_image_path = qr_preprocessing_result.get(
        "preprocessed_image_path",
        qr_preprocessing_result.get(
            "output_path",
            analysis_image_path
        )
    )

    forgery_localization_result = safe_forgery_localization_analysis(
        analysis_image_path
    )

    analysis_image = cv2.imread(
        analysis_image_path
    )

    if analysis_image is not None:
        forgery_localization_result["forgery_score"] = normalize_score(
            forgery_localization_result.get("forgery_score", 0)
        )
        forgery_localization_result["confidence"] = normalize_score(
            forgery_localization_result.get("confidence", 0)
        )
        forgery_localization_result["suspicious_regions"] = classify_regions(
            forgery_localization_result.get("suspicious_regions", []),
            "TruFor",
            analysis_image.shape,
            ocr_lines=ocr_result.get("lines", []),
            qr_regions=qr_preprocessing_result.get(
                "removed_regions",
                qr_preprocessing_result.get("qr_regions", [])
            ),
            photo_regions=photo_replacement_result.get("photo_regions", []),
            damage_regions=document_condition_result.get("damaged_regions", []),
            default_type="forgery_model"
        )

    tampering_result = safe_tampering_analysis(
        mvss_image_path
    )

    tampering_result = filter_mvss_regions(
        tampering_result,
        qr_preprocessing_result,
        analysis_image_path,
        supporting_regions=forgery_localization_result.get(
            "suspicious_regions",
            []
        ),
        ocr_lines=ocr_result.get("lines", []),
        photo_regions=photo_replacement_result.get("photo_regions", []),
        damage_regions=document_condition_result.get("damaged_regions", [])
    )

    mvss_preprocess_analysis = {
        "qr_removed": bool(
            qr_preprocessing_result.get("qr_removed")
        ),
        "removed_region_count": int(
            qr_preprocessing_result.get("removed_region_count", 0)
        ),
        "removed_regions": qr_preprocessing_result.get(
            "removed_regions",
            qr_preprocessing_result.get("qr_regions", [])
        ),
        "preprocessed_image_path": response_path(
            qr_preprocessing_result.get("preprocessed_image_path")
        ),
        "method": qr_preprocessing_result.get(
            "method",
            "none"
        ),
        "reasons": qr_preprocessing_result.get(
            "reasons",
            []
        )
    }

    if qr_preprocessing_result.get("error"):
        mvss_preprocess_analysis["error"] = qr_preprocessing_result["error"]

    preprocessing_analysis = {
        **mvss_preprocess_analysis,
        "input_path": response_path(
            qr_preprocessing_result.get("input_path")
        ),
        "output_path": response_path(
            qr_preprocessing_result.get("output_path")
        ),
        "qr_regions": mvss_preprocess_analysis["removed_regions"]
    }

    tampering_result["analysis_image_path"] = response_path(
        mvss_image_path
    )

    text_visual_regions = (
        forgery_localization_result.get("suspicious_regions", [])
        + tampering_result.get("suspicious_regions", [])
    )

    text_consistency_result = safe_text_consistency_analysis(
        analysis_image_path,
        ocr_result.get("lines", []),
        ocr_result.get("text", ""),
        visual_regions=text_visual_regions
    )

    ela_result = safe_ela_analysis(
        analysis_image_path
    )

    if analysis_image is not None:
        ela_result["suspicious_regions"] = classify_regions(
            ela_result.get("suspicious_regions", []),
            "ELA",
            analysis_image.shape,
            ocr_lines=ocr_result.get("lines", []),
            qr_regions=qr_preprocessing_result.get(
                "removed_regions",
                qr_preprocessing_result.get("qr_regions", [])
            ),
            photo_regions=photo_replacement_result.get("photo_regions", []),
            damage_regions=document_condition_result.get("damaged_regions", []),
            default_type="ela"
        )

        text_consistency_result["suspicious_regions"] = classify_regions(
            text_consistency_result.get("suspicious_regions", []),
            "TextMismatch",
            analysis_image.shape,
            ocr_lines=ocr_result.get("lines", []),
            qr_regions=qr_preprocessing_result.get(
                "removed_regions",
                qr_preprocessing_result.get("qr_regions", [])
            ),
            photo_regions=photo_replacement_result.get("photo_regions", []),
            damage_regions=document_condition_result.get("damaged_regions", []),
            default_type="text_consistency"
        )

    visual_consistency_result = safe_visual_consistency_analysis(
        analysis_image_path,
        {
            "ela": ela_result.get(
                "suspicious_regions",
                []
            ),
            "mvss": tampering_result.get(
                "suspicious_regions",
                []
            ),
            "condition": document_condition_result.get(
                "damaged_regions",
                []
            ),
            "photo": photo_replacement_result.get(
                "photo_regions",
                []
            ),
            "forgery_model": forgery_localization_result.get(
                "suspicious_regions",
                []
            ),
            "text_consistency": text_consistency_result.get(
                "suspicious_regions",
                []
            )
        }
    )

    visual_manipulation_result = build_visual_manipulation_analysis(
        tampering_result,
        forgery_localization_result,
        text_consistency_result,
        ela_result
    )

    document_quality_result = safe_document_quality_analysis(
        analysis_image_path,
        ocr_result=ocr_result,
        document_condition_result=document_condition_result,
        detector_results={
            "forgery": forgery_localization_result,
            "mvss": tampering_result,
            "ela": ela_result,
            "text_consistency": text_consistency_result
        }
    )

    correlation_result = correlate(

        ocr_result,

        ela_result,

        tampering_result,

        photo_replacement_result,

        visual_consistency_result

    )

    fraud_result = calculate_fraud_score(

        metadata_result,

        ocr_result,

        ela_result,

        tampering_result,

        correlation_result,

        type=extension.lstrip("."),

        masking_detected=masking_result["masking_detected"],

        document_condition_result=document_condition_result,

        photo_replacement_result=photo_replacement_result,

        forgery_localization_result=forgery_localization_result,

        text_consistency_result=text_consistency_result,

        visual_consistency_result=visual_consistency_result,

        document_quality_result=document_quality_result

    )

    annotated_image_path = None

    if fraud_result["risk_level"].lower() != "low":

        mvss_regions = tampering_result.get(
            "annotation_regions",
            tampering_result.get(
                "suspicious_regions",
                []
            )
        )
        mvss_regions = [
            region
            for region in mvss_regions or []
            if region.get("annotation_eligible", True)
        ]

        ela_regions = ela_result.get(
            "suspicious_regions",
            []
        )

        condition_regions = document_condition_result.get(
            "damaged_regions",
            []
        )

        photo_regions = photo_replacement_result.get(
            "photo_regions",
            []
        )

        visual_regions = visual_consistency_result.get(
            "inconsistent_regions",
            []
        )

        forgery_regions = forgery_localization_result.get(
            "suspicious_regions",
            []
        )

        text_regions = text_consistency_result.get(
            "suspicious_regions",
            []
        )

        for region in mvss_regions:
            region["type"] = "mvss"

        for region in ela_regions:
            region["type"] = "ela"

        for region in condition_regions:
            region["type"] = "condition"

        for region in photo_regions:
            region["type"] = "photo"

        for region in visual_regions:
            region["type"] = "visual"

        for region in forgery_regions:
            region["type"] = "forgery_model"

        for region in text_regions:
            region["type"] = "text_consistency"

        combined_regions = (
            (forgery_regions or [])
            + (mvss_regions or [])
            + (text_regions or [])
            + (ela_regions or [])
            + (condition_regions or [])
            + (photo_regions or [])
            + (visual_regions or [])
        )

        if document_quality_result.get("rejection_recommended"):
            combined_regions.append({
                "x": 12,
                "y": 12,
                "w": 520,
                "h": 52,
                "type": "quality",
                "source": "Document Quality",
                "label": "Low reliability / Rescan recommended",
                "reason": "Document condition prevents reliable automated verification",
                "annotation_eligible": True,
                "scoring_eligible": False
            })

        annotated_image_path = create_annotated_image(
            analysis_image_path,
            combined_regions,
            masking_result.get("masked_regions", []),
            os.path.splitext(stored_filename)[0]
        )

    response = {

        "filename": original_filename,

        "stored_filename": stored_filename,

        "status": fraud_result.get(
            "status",
            "success"
        ),

        "source":
            "embedded_pdf_text"
            if embedded_text
            else "ocr",

        "avg_confidence":
            ocr_result["avg_confidence"],

        "metadata_analysis":
            metadata_result,

        "ela_analysis":
            ela_result,

        "correlation_analysis":
            correlation_result,

        "masking_analysis": masking_result,

        "field_extraction_analysis":
            field_extraction_result,

        "document_condition_analysis":
            document_condition_result,

        "document_quality_analysis":
            document_quality_result,

        "photo_replacement_analysis":
            photo_replacement_result,

        "forgery_localization_analysis":
            forgery_localization_result,

        "text_consistency_analysis":
            text_consistency_result,

        "visual_consistency_analysis":
            visual_consistency_result,

        "visual_manipulation_analysis":
            visual_manipulation_result,

        "preprocessing_analysis":
            preprocessing_analysis,

        "mvss_preprocess_analysis":
            mvss_preprocess_analysis,

        "suspicious_fields":
            correlation_result.get(
                "suspicious_fields",
                []
            ),

        "fraud_analysis":
            fraud_result,

        "lines":
            ocr_result["lines"],

        "tampering_analysis":
            tampering_result,

        "text":
            embedded_text
            if embedded_text
            else ocr_result["text"],

        "file_path":
            response_path(file_path),

        "analysis_image_path":
            response_path(analysis_image_path),

        "annotated_image_path":
            response_path(annotated_image_path)

    }

    case_id = save_analysis(
        response
    )

    response["case_id"] = case_id

    return response
