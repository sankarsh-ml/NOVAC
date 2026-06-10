from fastapi import APIRouter, UploadFile, File, HTTPException
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

from app.services.photo_replacement_service import analyze_photo_replacement

from app.services.ai_generated_service import analyze_ai_generated_image

from app.services.visual_consistency_service import analyze_visual_consistency

from app.services.field_extraction_service import extract_fields


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
            "suspicious_region_count": 0,
            "suspicious_regions": [],
            "error": f"MVSS analysis failed: {exc}"
        }


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


def safe_ai_generated_analysis(
    image_path,
    metadata_result,
    ocr_result
):

    try:
        return analyze_ai_generated_image(
            image_path,
            metadata_result,
            ocr_result
        )

    except Exception as exc:
        return {
            "ai_generated_suspected": False,
            "strong_ai_generated_signal": False,
            "printed_document_likely": False,
            "positive_synthetic_evidence_count": 0,
            "real_capture_evidence_count": 0,
            "ai_generation_score": 0,
            "confidence": 0,
            "reasons": [],
            "supporting_reasons": [],
            "real_capture_reasons": [],
            "suppressed_reasons": [],
            "error": f"AI generated image analysis failed: {exc}"
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

    ai_generated_result = safe_ai_generated_analysis(
        analysis_image_path,
        metadata_result,
        ocr_result
    )

    ela_result = safe_ela_analysis(
        analysis_image_path
    )

    qr_preprocessing_result = remove_qr_code_with_metadata(
        analysis_image_path
    )

    qr_preprocessing_result["input_path"] = response_path(
        qr_preprocessing_result.get("input_path")
    )
    qr_preprocessing_result["output_path"] = response_path(
        qr_preprocessing_result.get("output_path")
    )

    mvss_image_path = qr_preprocessing_result.get(
        "output_path",
        analysis_image_path
    )

    tampering_result = safe_tampering_analysis(
        mvss_image_path
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
            )
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

        ai_generated_result=ai_generated_result,

        visual_consistency_result=visual_consistency_result

    )

    annotated_image_path = None

    if fraud_result["risk_level"].lower() != "low":

        mvss_regions = tampering_result.get(
            "suspicious_regions",
            []
        )

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

        combined_regions = (
            (mvss_regions or [])
            + (ela_regions or [])
            + (condition_regions or [])
            + (photo_regions or [])
            + (visual_regions or [])
        )

        annotated_image_path = create_annotated_image(
            analysis_image_path,
            combined_regions,
            masking_result.get("masked_regions", []),
            os.path.splitext(stored_filename)[0]
        )

    response = {

        "filename": original_filename,

        "stored_filename": stored_filename,

        "status": "success",

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

        "photo_replacement_analysis":
            photo_replacement_result,

        "ai_generated_analysis":
            ai_generated_result,

        "visual_consistency_analysis":
            visual_consistency_result,

        "preprocessing_analysis":
            qr_preprocessing_result,

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
