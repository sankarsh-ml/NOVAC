import argparse
import os
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.chdir(BACKEND_DIR)

from app.services.ela_services import analyze_ela
from app.services.document_condition_service import analyze_document_condition
from app.services.document_quality_service import analyze_document_quality
from app.services.field_extraction_service import extract_fields
from app.services.forgery_localization_service import analyze_forgery_localization
from app.services.masking_detection_service import detect_masking
from app.services.preprocessing_service import remove_qr_code_with_metadata
from app.services.scoring_service import calculate_fraud_score
from app.services.tampering_runner import analyze_tampering
from app.services.text_consistency_service import analyze_text_consistency
from app.services.visual_region_utils import (
    any_region_near,
    box_iou,
    classify_region_context,
    classify_regions,
    normalize_score
)

try:
    from app.services.pdf_service import pdf_to_image

except Exception as exc:
    PDF_IMPORT_ERROR = str(exc)

    def pdf_to_image(pdf_path):
        raise RuntimeError(
            f"PDF conversion unavailable in this shell: {PDF_IMPORT_ERROR}"
        )

try:
    from app.services.ocr_service import extract_text

except Exception as exc:
    OCR_IMPORT_ERROR = str(exc)

    def extract_text(image_path):
        return {
            "text": "",
            "lines": [],
            "avg_confidence": 0,
            "ocr_engine": "unavailable",
            "ocr_warning": f"OCR unavailable in this shell: {OCR_IMPORT_ERROR}"
        }


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PDF_SUFFIXES = {".pdf"}


def _safe_call(default, func, *args, **kwargs):

    try:
        return func(*args, **kwargs)

    except Exception as exc:
        result = {
            **default
        }
        result["error"] = str(exc)
        return result


def _filter_mvss_regions(
    tampering_result,
    preprocessing_result,
    image_path,
    supporting_regions=None,
    ocr_lines=None,
    damage_regions=None
):

    import cv2

    result = {
        **(tampering_result or {})
    }
    image = cv2.imread(str(image_path))

    if image is None:
        return result

    height, width = image.shape[:2]
    image_area = float(width * height) if width and height else 1.0
    confidence = float(result.get("mvss_confidence", 0) or 0)
    min_area = max(1200, image_area * 0.002)
    removed_regions = preprocessing_result.get(
        "removed_regions",
        preprocessing_result.get("qr_regions", [])
    ) or []
    supporting_regions = supporting_regions or []
    raw_regions = list(result.get("suspicious_regions", []) or [])
    scoring_regions = []
    suppressed_regions = list(result.get("suppressed_regions", []) or [])

    for region in raw_regions:
        w = int(region.get("w", 0))
        h = int(region.get("h", 0))
        area = float(region.get("area") or (w * h))
        area_ratio = area / image_area
        base = {
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
        base = classify_region_context(
            base,
            "MVSS",
            image.shape,
            ocr_lines=ocr_lines,
            qr_regions=removed_regions,
            damage_regions=damage_regions,
            default_type="mvss"
        )
        has_support = any_region_near(base, supporting_regions, image_shape=image.shape)
        reason = base.get("suppression_reason")

        if w < 18 or h < 18:
            reason = "Region dimensions below MVSS scoring threshold"
        elif area_ratio > 0.35 and confidence < 0.85:
            reason = "Region covers too much of document for reliable MVSS evidence"
        elif area < min_area and not (confidence >= 0.85 and has_support):
            reason = "Region too small for reliable MVSS evidence"
        elif area_ratio < 0.003 and not (confidence >= 0.85 and has_support):
            reason = "Region area ratio below MVSS scoring threshold"
        elif max(w, h) / float(max(min(w, h), 1)) > 8 and area_ratio < 0.03 and not has_support:
            reason = "Region is a long thin noise strip"

        for removed_region in removed_regions:
            if box_iou(base, removed_region) > 0.30:
                reason = "Region overlaps removed QR-like area"
                break

        if reason:
            suppressed_regions.append({
                **base,
                "suppression_reason": reason,
                "reason": reason
            })
        else:
            scoring_regions.append({
                **base,
                "scoring_eligible": True,
                "annotation_eligible": True,
                "reason": "MVSS detected meaningful suspicious visual manipulation region"
            })

    total_area = sum(region.get("area", 0) for region in scoring_regions[:3])
    tampered_percent = (total_area / image_area) * 100

    if confidence < 0.35 or not scoring_regions:
        tampering_score = 0
    elif confidence >= 0.75 and tampered_percent >= 2:
        tampering_score = 30
    elif confidence >= 0.55 and tampered_percent >= 0.8:
        tampering_score = 20
    else:
        tampering_score = 10

    result.update({
        "raw_region_count": len(raw_regions),
        "scoring_region_count": len(scoring_regions[:3]),
        "annotation_region_count": len(scoring_regions[:3]),
        "suspicious_region_count": len(scoring_regions[:3]),
        "valid_suspicious_region_count": len(scoring_regions[:3]),
        "suspicious_regions": scoring_regions[:3],
        "annotation_regions": scoring_regions[:3],
        "suppressed_regions": suppressed_regions,
        "suppressed_region_count": len(suppressed_regions),
        "tampered_area_percent": round(tampered_percent, 2),
        "tampering_score": float(min(tampering_score, 40)),
        "tampering_detected": bool(scoring_regions)
    })

    return result


def _candidate_files(upload_dir):

    skip_parts = {
        "forgery_maps",
        "tampering",
        "ela",
        "ocr_variants"
    }

    for path in sorted(upload_dir.rglob("*")):
        if not path.is_file():
            continue

        if any(part in skip_parts for part in path.parts):
            continue

        if path.name.endswith("_annotated.png"):
            continue

        if path.suffix.lower() in IMAGE_SUFFIXES | PDF_SUFFIXES:
            yield path


def _expected_label(filename):

    name = filename.lower()

    for label, tokens in {
        "damaged": ["damaged", "worn", "torn", "blur", "fold", "crumpled"],
        "fake": ["fake", "tampered", "forged", "edited"],
        "real": ["real", "original", "clean"]
    }.items():
        if any(token in name for token in tokens):
            return label

    return "unknown"


def _analyze_file(path):

    analysis_path = path

    if path.suffix.lower() in PDF_SUFFIXES:
        try:
            analysis_path = Path(pdf_to_image(str(path)))
        except Exception as exc:
            return {
                "filename": path.name,
                "expected": _expected_label(path.name),
                "ocr_confidence": 0,
                "field_count": 0,
                "trufor_available": False,
                "trufor_score": 0,
                "trufor_regions": 0,
                "mvss_raw": 0,
                "mvss_scoring": 0,
                "mvss_score": 0,
                "ela_score": 0,
                "text_score": 0,
                "risk_score": 0,
                "risk_level": "Skipped",
                "status": "skipped",
                "quality_score": 0,
                "rejection_recommended": False,
                "trufor_contribution": 0,
                "mvss_contribution": 0,
                "text_contribution": 0,
                "ela_contribution": 0,
                "agreement_contribution": 0,
                "reasons": str(exc)
            }

    ocr = _safe_call(
        {
            "text": "",
            "lines": [],
            "avg_confidence": 0
        },
        extract_text,
        str(analysis_path)
    )
    fields = _safe_call(
        {
            "fields": {},
            "field_count": 0
        },
        extract_fields,
        ocr
    )
    masking = _safe_call(
        {
            "masking_detected": False
        },
        detect_masking,
        ocr
    )
    forgery = _safe_call(
        {
            "model_available": False,
            "forgery_score": 0,
            "suspicious_regions": [],
            "model_error": "TruFor failed"
        },
        analyze_forgery_localization,
        str(analysis_path)
    )
    condition = _safe_call(
        {
            "condition_score": 0,
            "condition_confidence": "low",
            "damaged_regions": [],
            "reasons": []
        },
        analyze_document_condition,
        str(analysis_path)
    )
    import cv2

    image = cv2.imread(str(analysis_path))
    image_shape = image.shape if image is not None else None
    preprocessing = _safe_call(
        {
            "removed_regions": [],
            "qr_removed": False
        },
        remove_qr_code_with_metadata,
        str(analysis_path)
    )
    if image_shape is not None:
        forgery["forgery_score"] = normalize_score(
            forgery.get("forgery_score", 0)
        )
        forgery["confidence"] = normalize_score(
            forgery.get("confidence", 0)
        )
        forgery["suspicious_regions"] = classify_regions(
            forgery.get("suspicious_regions", []),
            "TruFor",
            image_shape,
            ocr_lines=ocr.get("lines", []),
            qr_regions=preprocessing.get(
                "removed_regions",
                preprocessing.get("qr_regions", [])
            ),
            damage_regions=condition.get("damaged_regions", []),
            default_type="forgery_model"
        )
    mvss_path = preprocessing.get(
        "preprocessed_image_path",
        preprocessing.get(
            "output_path",
            str(analysis_path)
        )
    )
    mvss = _safe_call(
        {
            "tampering_score": 0,
            "suspicious_regions": [],
            "raw_region_count": 0,
            "scoring_region_count": 0,
            "suppressed_regions": []
        },
        analyze_tampering,
        mvss_path
    )
    mvss = _filter_mvss_regions(
        mvss,
        preprocessing,
        str(analysis_path),
        supporting_regions=forgery.get(
            "suspicious_regions",
            []
        ),
        ocr_lines=ocr.get("lines", []),
        damage_regions=condition.get("damaged_regions", [])
    )
    text = _safe_call(
        {
            "font_mismatch_detected": False,
            "field_mismatch_score": 0,
            "suspicious_regions": [],
            "comparisons_used": 0,
            "comparisons_skipped": 0
        },
        analyze_text_consistency,
        str(analysis_path),
        ocr.get("lines", []),
        ocr.get("text", ""),
        visual_regions=(
            forgery.get("suspicious_regions", [])
            + mvss.get("suspicious_regions", [])
        )
    )
    ela = _safe_call(
        {
            "ela_score": 0,
            "suspicious_regions": []
        },
        analyze_ela,
        str(analysis_path)
    )
    if image_shape is not None:
        text["suspicious_regions"] = classify_regions(
            text.get("suspicious_regions", []),
            "TextMismatch",
            image_shape,
            ocr_lines=ocr.get("lines", []),
            qr_regions=preprocessing.get(
                "removed_regions",
                preprocessing.get("qr_regions", [])
            ),
            damage_regions=condition.get("damaged_regions", []),
            default_type="text_consistency"
        )
        ela["suspicious_regions"] = classify_regions(
            ela.get("suspicious_regions", []),
            "ELA",
            image_shape,
            ocr_lines=ocr.get("lines", []),
            qr_regions=preprocessing.get(
                "removed_regions",
                preprocessing.get("qr_regions", [])
            ),
            damage_regions=condition.get("damaged_regions", []),
            default_type="ela"
        )
    quality = _safe_call(
        {
            "quality_score": 100,
            "damage_score": 0,
            "rejection_recommended": False,
            "analysis_reliable": True,
            "reasons": []
        },
        analyze_document_quality,
        str(analysis_path),
        ocr_result=ocr,
        document_condition_result=condition,
        detector_results={
            "forgery": forgery,
            "mvss": mvss,
            "ela": ela,
            "text_consistency": text
        }
    )
    fraud = calculate_fraud_score(
        {},
        ocr,
        ela,
        mvss,
        {},
        type=path.suffix.lower().lstrip("."),
        masking_detected=masking.get("masking_detected", False),
        document_condition_result=condition,
        forgery_localization_result=forgery,
        text_consistency_result=text,
        document_quality_result=quality
    )
    contributions = fraud.get("detector_contributions", {})

    return {
        "filename": path.name,
        "expected": _expected_label(path.name),
        "ocr_confidence": ocr.get("avg_confidence", 0),
        "field_count": fields.get("field_count", len(fields.get("fields", {}))),
        "trufor_available": forgery.get("model_available", False),
        "trufor_score": normalize_score(forgery.get("forgery_score", 0)),
        "trufor_contribution": contributions.get("trufor", {}).get("contribution", 0),
        "trufor_regions": len(forgery.get("suspicious_regions", [])),
        "mvss_raw": mvss.get("raw_region_count", 0),
        "mvss_scoring": mvss.get("scoring_region_count", 0),
        "mvss_contribution": contributions.get("mvss", {}).get("contribution", 0),
        "mvss_score": mvss.get("tampering_score", 0),
        "ela_score": ela.get("ela_score", 0),
        "ela_contribution": contributions.get("ela", {}).get("contribution", 0),
        "text_score": text.get("field_mismatch_score", 0),
        "text_contribution": contributions.get("text_consistency", {}).get("contribution", 0),
        "agreement_contribution": contributions.get("detector_agreement", {}).get("contribution", 0),
        "quality_score": quality.get("quality_score", 100),
        "rejection_recommended": quality.get("rejection_recommended", False),
        "risk_score": fraud.get("fraud_score", 0),
        "risk_level": fraud.get("risk_level", "Unknown"),
        "status": fraud.get("status", "success"),
        "reasons": "; ".join(fraud.get("reasons", [])[:3])
    }


def _print_table(rows):

    headers = [
        "filename",
        "expected",
        "ocr",
        "quality",
        "reject",
        "trufor_norm",
        "trufor_contrib",
        "mvss_raw",
        "mvss_ok",
        "mvss_contrib",
        "text_score",
        "text_contrib",
        "ela_score",
        "ela_contrib",
        "agreement",
        "risk",
        "level",
        "status",
        "reasons"
    ]
    print("\t".join(headers))

    for row in rows:
        print("\t".join([
            str(row["filename"]),
            str(row["expected"]),
            str(row["ocr_confidence"]),
            str(row["quality_score"]),
            str(row["rejection_recommended"]),
            str(row["trufor_score"]),
            str(row["trufor_contribution"]),
            str(row["mvss_raw"]),
            str(row["mvss_scoring"]),
            str(row["mvss_contribution"]),
            str(row["text_score"]),
            str(row["text_contribution"]),
            str(row["ela_score"]),
            str(row["ela_contribution"]),
            str(row["agreement_contribution"]),
            str(row["risk_score"]),
            str(row["risk_level"]),
            str(row["status"]),
            row["reasons"]
        ]))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uploads",
        default=str(BACKEND_DIR / "uploads")
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5
    )
    args = parser.parse_args()

    upload_dir = Path(args.uploads)
    rows = []

    for path in _candidate_files(upload_dir):
        rows.append(
            _analyze_file(path)
        )

        if len(rows) >= args.limit:
            break

    _print_table(rows)


if __name__ == "__main__":
    main()
