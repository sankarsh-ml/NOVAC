from fastapi import (
    APIRouter,
    HTTPException,
    Query
)
from fastapi.responses import StreamingResponse
from datetime import datetime
from io import BytesIO
import logging
from pathlib import Path
import time

from PIL import Image

from app.services.result_service import (

    get_all_results,

    get_result_by_case_id,

    delete_result,

    delete_all_results

)
from app.services.field_extraction_bridge import (
    prepare_field_extraction_input,
    run_field_extraction,
)
from app.services.storage_service import update_analysis_fields

router = APIRouter()
logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent


def _now():
    return datetime.utcnow().isoformat() + "Z"


def _resolve_uploaded_file_path(result):
    candidates = []

    file_path = result.get("file_path")
    if file_path:
        raw_path = Path(str(file_path))
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.extend([
                BACKEND_DIR / raw_path,
                PROJECT_DIR / raw_path,
            ])

    stored_filename = result.get("stored_filename")
    if stored_filename:
        candidates.extend([
            BACKEND_DIR / "uploads" / stored_filename,
            PROJECT_DIR / "uploads" / stored_filename,
        ])

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved.is_file():
            return resolved

    return None


def _resolve_existing_path(path_value):
    if not isinstance(path_value, str):
        return None

    raw_value = path_value.strip()
    if not raw_value:
        return None

    lowered = raw_value.lower()
    if (
        lowered.startswith("http://")
        or lowered.startswith("https://")
        or lowered.startswith("data:")
    ):
        return None

    normalized = raw_value.replace("\\", "/")
    relative_value = normalized.lstrip("/")
    raw_path = Path(normalized)
    relative_path = Path(relative_value)

    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([
            raw_path,
            Path.cwd() / relative_path,
            BACKEND_DIR / relative_path,
            PROJECT_DIR / relative_path,
        ])

    if relative_path.parts:
        candidates.extend([
            BACKEND_DIR / relative_path,
            PROJECT_DIR / relative_path,
        ])

    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue

        if resolved in seen:
            continue

        seen.add(resolved)
        if resolved.exists() and resolved.is_file():
            return resolved

    return None


def _has_pdf_extension(value):
    if not isinstance(value, str):
        return False
    return Path(value.split("?", 1)[0]).suffix.lower() == ".pdf"


def is_pdf_record(record):
    for key in (
        "file_path",
        "uploaded_file_path",
        "original_filename",
        "filename",
        "stored_filename",
    ):
        if _has_pdf_extension(record.get(key)):
            return True
    return False


def get_crop_source_image(record):
    extraction = record.get("field_extraction") or {}

    if is_pdf_record(record):
        return (
            extraction.get("source_image_path")
            or extraction.get("processed_image_path")
            or extraction.get("analysis_image_path")
            or extraction.get("input_path")
            or record.get("converted_image_path")
            or record.get("analysis_image_path")
            or record.get("processed_image_path")
            or record.get("preview_image_path")
            or record.get("file_path")
            or record.get("uploaded_file_path")
        )

    return (
        record.get("analysis_image_path")
        or record.get("processed_image_path")
        or record.get("preview_image_path")
        or record.get("image_path")
        or record.get("file_path")
        or record.get("uploaded_file_path")
    )


def _resolve_crop_image_path(result):
    source_value = get_crop_source_image(result)
    source_path = _resolve_existing_path(source_value)

    if source_path is not None:
        if source_path.suffix.lower() != ".pdf":
            return source_path

        if is_pdf_record(result):
            try:
                prepared_path = prepare_field_extraction_input(source_path)
            except Exception:
                logger.exception(
                    "Unable to prepare PDF field extraction crop source path=%s",
                    source_path
                )
                return None

            if prepared_path.exists() and prepared_path.suffix.lower() != ".pdf":
                return prepared_path.resolve()

    if is_pdf_record(result):
        for value in (
            result.get("file_path"),
            result.get("uploaded_file_path"),
            result.get("stored_filename") and f"uploads/{result.get('stored_filename')}",
        ):
            pdf_path = _resolve_existing_path(value)
            if pdf_path is None or pdf_path.suffix.lower() != ".pdf":
                continue

            try:
                prepared_path = prepare_field_extraction_input(pdf_path)
            except Exception:
                logger.exception(
                    "Unable to prepare fallback PDF field extraction crop source path=%s",
                    pdf_path
                )
                continue

            if prepared_path.exists() and prepared_path.suffix.lower() != ".pdf":
                return prepared_path.resolve()

        return None

    stored_filename = result.get("stored_filename")
    if stored_filename:
        path = _resolve_existing_path(f"uploads/{stored_filename}")
        if path is not None and path.suffix.lower() != ".pdf":
            return path

    return None


def _crop_from_bbox(image_path, bbox, padding=10):
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
    ):
        raise ValueError("missing_bbox")

    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError) as exc:
        raise ValueError("missing_bbox") from exc

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
        width, height = image.size

        left = max(0, int(round(min(x1, x2))) - padding)
        top = max(0, int(round(min(y1, y2))) - padding)
        right = min(width, int(round(max(x1, x2))) + padding)
        bottom = min(height, int(round(max(y1, y2))) + padding)

        if right <= left or bottom <= top:
            raise ValueError("missing_bbox")

        image_size = (width, height)
        buffer = BytesIO()
        image.crop((left, top, right, bottom)).save(buffer, format="PNG")
        buffer.seek(0)
        return buffer, image_size


def _fraud_analysis_complete(result):
    return bool(result.get("fraud_analysis"))


def _field_extraction_allowed(result):
    fraud = result.get("fraud_analysis", {}) or {}
    status_values = {
        str(fraud.get("result_status") or "").strip().lower(),
        str(fraud.get("status") or "").strip().lower(),
        str(result.get("result_status") or "").strip().lower(),
        str(result.get("status") or "").strip().lower(),
    }
    risk_level = str(fraud.get("risk_level") or result.get("risk_level") or "").strip().lower()
    fraud_score = fraud.get("fraud_score")

    blocked_statuses = {
        "fraud_suspected",
        "synthetic_suspected",
        "unprocessable",
        "failed",
        "error",
    }
    if status_values.intersection(blocked_statuses):
        return False

    blocked_risks = {
        "high risk",
        "high",
        "critical",
        "synthetic document suspected",
        "analysis inconclusive",
        "analysis limited",
    }
    if risk_level in blocked_risks:
        return False

    if "passed" in status_values or "success" in status_values:
        return True

    if risk_level in {"low risk", "low", "safe", "real"}:
        return True

    try:
        return float(fraud_score) < 25
    except Exception:
        return False


def _save_field_extraction(case_id, extraction_result):
    payload = {
        **(extraction_result or {}),
        "created_at": _now(),
    }
    update_analysis_fields(
        case_id,
        {
            "field_extraction": payload
        }
    )
    return payload


@router.get("/results")
def get_results():

    return get_all_results()


@router.get("/results/case/{analysis_id}")
def get_result(
    analysis_id: str
):

    result = get_result_by_case_id(
        analysis_id
    )

    if result is None:

        raise HTTPException(

            status_code=404,

            detail="Analysis not found"

        )

    return result


@router.post("/api/extract-fields/{analysis_id}")
def extract_fields_for_analysis(
    analysis_id: str,
    force: bool = Query(False)
):
    started_at = time.perf_counter()
    logger.info(
        "Field extraction request started analysis_id=%s force=%s",
        analysis_id,
        force
    )

    result = get_result_by_case_id(
        analysis_id
    )

    if result is None:
        logger.warning("Field extraction request analysis_id=%s not found", analysis_id)
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )

    if not _fraud_analysis_complete(result):
        logger.info("Field extraction request blocked analysis_id=%s reason=fraud_incomplete", analysis_id)
        raise HTTPException(
            status_code=409,
            detail="Fraud analysis is not complete yet."
        )

    saved_extraction = result.get("field_extraction")
    if (
        isinstance(saved_extraction, dict)
        and saved_extraction.get("status") == "completed"
        and not force
    ):
        logger.info(
            "Field extraction request returned cached result analysis_id=%s elapsed=%.3fs",
            analysis_id,
            time.perf_counter() - started_at
        )
        return saved_extraction

    if not _field_extraction_allowed(result):
        skipped = _save_field_extraction(
            analysis_id,
            {
                "status": "skipped",
                "reason": "Field extraction skipped because the document did not pass fraud verification.",
                "fields": {},
                "missing_fields": [],
                "warnings": []
            }
        )
        logger.info(
            "Field extraction skipped analysis_id=%s elapsed=%.3fs",
            analysis_id,
            time.perf_counter() - started_at
        )
        return skipped

    file_path = _resolve_uploaded_file_path(result)
    if file_path is None:
        failed = _save_field_extraction(
            analysis_id,
            {
                "status": "failed",
                "error": "Uploaded file is missing.",
                "fields": {},
                "missing_fields": [],
                "warnings": []
            }
        )
        logger.error(
            "Field extraction failed analysis_id=%s error=uploaded_file_missing elapsed=%.3fs",
            analysis_id,
            time.perf_counter() - started_at
        )
        return failed

    logger.info(
        "Field extraction invoking bridge analysis_id=%s input_path=%s",
        analysis_id,
        file_path
    )
    extraction_result = run_field_extraction(
        str(file_path)
    )
    saved = _save_field_extraction(
        analysis_id,
        extraction_result
    )
    if saved.get("status") == "failed":
        logger.error(
            "Field extraction finished analysis_id=%s status=failed error=%s elapsed=%.3fs",
            analysis_id,
            saved.get("error"),
            time.perf_counter() - started_at
        )
    else:
        logger.info(
            "Field extraction finished analysis_id=%s status=%s elapsed=%.3fs",
            analysis_id,
            saved.get("status"),
            time.perf_counter() - started_at
        )
    return saved


@router.get("/api/extraction-result/{analysis_id}")
def get_extraction_result(
    analysis_id: str
):
    result = get_result_by_case_id(
        analysis_id
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )

    field_extraction = result.get("field_extraction")
    if not field_extraction:
        return {
            "status": "not_run",
            "analysis_id": analysis_id,
            "message": "Field extraction has not been run for this analysis yet."
        }

    return field_extraction


@router.get("/api/extraction-field-crop/{analysis_id}/{field_name}")
def get_extraction_field_crop(
    analysis_id: str,
    field_name: str
):
    result = get_result_by_case_id(
        analysis_id
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )

    fields = (
        result.get("field_extraction", {})
        or {}
    ).get("fields", {}) or {}

    if field_name not in fields:
        raise HTTPException(
            status_code=404,
            detail="Field not found."
        )

    field_data = fields.get(field_name) or {}
    bbox = field_data.get("bbox")
    if not bbox:
        raise HTTPException(
            status_code=404,
            detail="Crop unavailable for this field."
        )

    pdf_record = is_pdf_record(result)
    image_path = _resolve_crop_image_path(result)
    if image_path is None:
        logger.warning(
            "Field crop source unavailable analysis_id=%s field_name=%s is_pdf=%s bbox=%s",
            analysis_id,
            field_name,
            pdf_record,
            bbox
        )
        raise HTTPException(
            status_code=404,
            detail="Original image not found for crop."
        )

    if (
        pdf_record
        and not (
            result.get("field_extraction", {})
            or {}
        ).get("source_image_path")
        and image_path.name.endswith("_field_extraction_source.png")
    ):
        update_analysis_fields(
            analysis_id,
            {
                "field_extraction.source_image_path": str(image_path)
            }
        )

    try:
        crop_buffer, image_size = _crop_from_bbox(image_path, bbox)
    except ValueError as exc:
        logger.warning(
            "Field crop bbox unavailable analysis_id=%s field_name=%s is_pdf=%s source_image_path=%s bbox=%s",
            analysis_id,
            field_name,
            pdf_record,
            image_path,
            bbox
        )
        raise HTTPException(
            status_code=404,
            detail="Crop unavailable for this field."
        ) from exc
    except OSError as exc:
        logger.exception(
            "Unable to generate field crop analysis_id=%s field_name=%s image_path=%s",
            analysis_id,
            field_name,
            image_path
        )
        raise HTTPException(
            status_code=404,
            detail="Original image not found for crop."
        ) from exc

    logger.info(
        "Field crop generated analysis_id=%s field_name=%s is_pdf=%s source_image_path=%s image_size=%s bbox=%s",
        analysis_id,
        field_name,
        pdf_record,
        image_path,
        image_size,
        bbox
    )

    return StreamingResponse(
        crop_buffer,
        media_type="image/png"
    )

@router.delete("/results/case/{case_id}")
def delete_case(case_id: str):

    success = delete_result(case_id)

    if not success:

        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    return {
        "message": f"{case_id} deleted successfully"
    }

@router.delete("/results")
def delete_all():

    deleted_count = delete_all_results()

    return {
        "message": "All records deleted",
        "deleted_count": deleted_count
    }
