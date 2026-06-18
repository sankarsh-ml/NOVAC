from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


os.environ.setdefault("FLAGS_use_onednn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")

MODULE_DIR = Path(__file__).resolve().parent
MODELS_DIR = MODULE_DIR / "models"

MODEL_FILES = {
    "Id_Classifier": "Id_Classifier.pt",
    "Aadhaar": "Aadhaar_Card.pt",
    "Pan_Card": "Pan_Card.pt",
    "Passport": "Passport.pt",
    "Voter_Id": "Voter_Id.pt",
    "Driving_License": "Driving_License.pt",
}

FALLBACK_CLASSES = {
    "Aadhaar": ["Aadhaar", "DOB", "Gender", "Name", "Address"],
    "Pan_Card": ["PAN", "Name", "Father's Name", "DOB", "Pan Card"],
    "Passport": [
        "Address",
        "Code",
        "DOB",
        "DOI",
        "EXP",
        "Gender",
        "MRZ2",
        "MRZ1",
        "MRZ2",
        "Name",
        "Nationality",
        "Nation",
        "POI",
    ],
    "Voter_Id": [
        "Address",
        "Age",
        "DOB",
        "Card Voter ID 1 Back",
        "Card Voter ID 2 Front",
        "Card Voter ID 2 Back",
        "Card Voter ID 1 Front",
        "DOB",
        "Date of Issue",
        "Election",
        "Father",
        "Gender",
        "Name",
        "Point",
        "Portrait",
        "Symbol",
        "Voter ID",
        "Portrait",
        "Card Voter ID 1 Back",
    ],
    "Driving_License": [
        "Address",
        "Blood Group",
        "DL No",
        "DOB",
        "Name",
        "Relation With",
        "RTO",
        "State",
        "Vehicle Type",
    ],
}

DOC_TYPE_TO_MODEL = {
    "aadhar_front": "Aadhaar",
    "aadhar_back": "Aadhaar",
    "aadhaar": "Aadhaar",
    "pan_card_front": "Pan_Card",
    "pan": "Pan_Card",
    "pan_card": "Pan_Card",
    "passport": "Passport",
    "voter_id": "Voter_Id",
    "voter": "Voter_Id",
    "driving_license_front": "Driving_License",
    "driving_license_back": "Driving_License",
    "driving_license": "Driving_License",
}

PUBLIC_DOC_TYPES = {
    "aadhar_front": "aadhaar",
    "aadhar_back": "aadhaar",
    "aadhaar": "aadhaar",
    "pan_card_front": "pan",
    "pan_card": "pan",
    "pan": "pan",
    "passport": "passport",
    "voter_id": "voter_id",
    "voter": "voter_id",
    "driving_license_front": "driving_license",
    "driving_license_back": "driving_license",
    "driving_license": "driving_license",
}

FIELD_LABELS = {
    "aadhaar": {
        "aadhaar": "aadhaar_number",
        "aadhaar_number": "aadhaar_number",
        "number": "aadhaar_number",
        "aadhaar_card": None,
        "dob": "dob",
        "date_of_birth": "dob",
        "gender": "gender",
        "name": "name",
        "address": "address",
    },
    "pan": {
        "pan": "pan_number",
        "pan_number": "pan_number",
        "card": None,
        "pan_card": None,
        "name": "name",
        "father_name": "father_name",
        "fathers_name": "father_name",
        "dob": "dob",
        "date_of_birth": "dob",
    },
    "passport": {
        "code": "passport_number",
        "passport_number": "passport_number",
        "name": "name",
        "dob": "dob",
        "doi": "date_of_issue",
        "exp": "expiry_date",
        "gender": "gender",
        "mrz1": "mrz1",
        "mrz2": "mrz2",
        "nationality": "nationality",
        "nation": "country",
        "poi": "place_of_issue",
        "address": "address",
    },
    "voter_id": {
        "voter_id": "voter_id_number",
        "name": "name",
        "father": "father_name",
        "gender": "gender",
        "dob": "dob",
        "age": "age",
        "address": "address",
        "date_of_issue": "date_of_issue",
        "election": "election",
        "portrait": None,
        "symbol": None,
        "point": None,
    },
    "driving_license": {
        "dl_no": "license_number",
        "name": "name",
        "dob": "dob",
        "address": "address",
        "blood_group": "blood_group",
        "relation_with": "relation_name",
        "rto": "rto",
        "state": "state",
        "vehicle_type": "vehicle_type",
    },
}

EXPECTED_FIELDS = {
    "aadhaar": ["name", "dob", "gender", "aadhaar_number", "address"],
    "pan": ["name", "father_name", "dob", "pan_number"],
    "passport": ["passport_number", "name", "dob", "gender", "nationality"],
    "voter_id": ["voter_id_number", "name"],
    "driving_license": ["license_number", "name", "dob"],
}

JUNK_NAME_LINES = {
    "government of india",
    "govt of india",
    "unique identification authority of india",
    "income tax department",
    "election commission of india",
    "republic of india",
}


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("ultralytics").setLevel(logging.WARNING)
LOGGER = logging.getLogger("field_extraction")


def _json_failure(input_path: str, error: str, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": "failed",
        "input_path": input_path,
        "error": error,
        "fields": {},
        "missing_fields": [],
        "warnings": warnings or [],
    }


def _write_json(output_path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _model_path(model_key: str) -> Path:
    return MODELS_DIR / MODEL_FILES[model_key]


def _require_model(model_key: str) -> Path:
    path = _model_path(model_key)
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    return path


def _load_yolo(model_key: str):
    from ultralytics import YOLO

    return YOLO(str(_require_model(model_key)))


def _load_ocr():
    from paddleocr import PaddleOCR

    try:
        return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    except ValueError:
        return PaddleOCR(use_angle_cls=True, lang="en")
    except TypeError:
        return PaddleOCR(use_textline_orientation=True, lang="en")


def _load_image_bgr(input_path: str | Path) -> np.ndarray:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".pdf":
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PDF input requires PyMuPDF. Install requirements.txt in field_extraction_venv.") from exc

        doc = fitz.open(str(path))
        if doc.page_count == 0:
            raise ValueError("PDF has no pages")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def _canonical_label(label: str) -> str:
    label = label.strip().replace("'", "")
    label = re.sub(r"(?i)^aadhaar[_\s-]+", "", label)
    label = re.sub(r"(?i)^aadhar[_\s-]+", "", label)
    label = re.sub(r"(?i)^pan[_\s-]+", "", label)
    label = re.sub(r"(?i)^passport[_\s-]+", "", label)
    label = re.sub(r"(?i)^voter[_\s-]+", "voter_", label)
    label = re.sub(r"(?i)^driving[_\s-]+license[_\s-]+", "", label)
    label = re.sub(r"[^A-Za-z0-9]+", "_", label)
    return label.strip("_").lower()


def _field_key(document_type: str, raw_label: str) -> str | None:
    canonical = _canonical_label(raw_label)
    return FIELD_LABELS.get(document_type, {}).get(canonical, canonical)


def _public_doc_type(raw_doc_type: str) -> str:
    return PUBLIC_DOC_TYPES.get(raw_doc_type, raw_doc_type)


def _class_name(result: Any, model_key: str, cls_index: int) -> str:
    names = getattr(result, "names", None)
    if isinstance(names, dict) and cls_index in names:
        candidate = str(names[cls_index])
        if not candidate.isdigit():
            return candidate
    if isinstance(names, list) and cls_index < len(names):
        candidate = str(names[cls_index])
        if not candidate.isdigit():
            return candidate
    fallback = FALLBACK_CLASSES.get(model_key, [])
    if cls_index < len(fallback):
        return fallback[cls_index]
    return str(cls_index)


def _detect_document_type(image: np.ndarray, requested_document_type: str | None) -> tuple[str, float, list[str]]:
    warnings: list[str] = []
    if requested_document_type:
        raw_type = requested_document_type.strip().lower().replace("-", "_")
        if raw_type not in DOC_TYPE_TO_MODEL:
            raise ValueError(f"Unsupported document type override: {requested_document_type}")
        return raw_type, 1.0, warnings

    classifier_path = _model_path("Id_Classifier")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Missing classifier model file: {classifier_path}")

    classifier = _load_yolo("Id_Classifier")
    results = classifier(image, verbose=False)
    if not results or getattr(results[0], "probs", None) is None:
        raise RuntimeError("Classifier did not return probabilities")

    result = results[0]
    doc_type = result.names[result.probs.top1]
    confidence = float(result.probs.top1conf.item())
    return str(doc_type), confidence, warnings


def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    if max(h, w) < 900:
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    crop = cv2.fastNlMeansDenoisingColored(crop, None, 8, 8, 7, 21)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def _iter_ocr_items(ocr_result: Any):
    if not ocr_result:
        return
    for line in ocr_result:
        if line is None:
            continue
        if isinstance(line, dict):
            texts = line.get("rec_texts") or []
            scores = line.get("rec_scores") or []
            for index, text in enumerate(texts):
                yield str(text), float(scores[index]) if index < len(scores) else None
            continue
        for item in line:
            if not item or len(item) < 2:
                continue
            text_score = item[1]
            if isinstance(text_score, (list, tuple)) and text_score:
                text = str(text_score[0])
                score = float(text_score[1]) if len(text_score) > 1 and text_score[1] is not None else None
                yield text, score


def _run_ocr(ocr: Any, crop: np.ndarray) -> tuple[str, float | None]:
    crop = _preprocess_crop(crop)
    try:
        result = ocr.ocr(crop, cls=True)
    except TypeError:
        result = ocr.ocr(crop)

    texts: list[str] = []
    scores: list[float] = []
    for text, score in _iter_ocr_items(result):
        if text.strip():
            texts.append(text.strip())
        if score is not None:
            scores.append(score)
    confidence = round(sum(scores) / len(scores), 4) if scores else None
    return " ".join(texts), confidence


def _strip_labels(text: str, labels: list[str]) -> str:
    for label in labels:
        text = re.sub(rf"(?i)\b{re.escape(label)}\b\s*[:\-]?", " ", text)
    return text


def _normalize_date(text: str) -> str:
    cleaned = _strip_labels(text, ["DOB", "D.O.B", "Date of Birth", "Birth", "DOI", "EXP"])
    cleaned = re.sub(r"[^0-9A-Za-z/.\- ]", " ", cleaned)
    patterns = [
        r"(\d{1,2})[\/.\- ](\d{1,2})[\/.\- ](\d{2,4})",
        r"(\d{4})[\/.\- ](\d{1,2})[\/.\- ](\d{1,2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if not match:
            continue
        parts = match.groups()
        if len(parts[0]) == 4:
            year, month, day = parts
        else:
            day, month, year = parts
        if len(year) == 2:
            year = f"19{year}" if int(year) > 30 else f"20{year}"
        return f"{int(day):02d}/{int(month):02d}/{int(year):04d}"
    return _collapse_spaces(cleaned)


def _normalize_gender(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\b(f|female)\b", lowered):
        return "Female"
    if re.search(r"\b(m|male)\b", lowered):
        return "Male"
    if "trans" in lowered or "other" in lowered:
        return "Other"
    return _collapse_spaces(_strip_labels(text, ["Gender", "Sex"]))


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def _clean_text(text: str, field_key: str) -> str:
    text = _collapse_spaces(text)
    if not text:
        return ""

    label_words = [
        "Name",
        "DOB",
        "Date of Birth",
        "Gender",
        "Address",
        "Father's Name",
        "Father Name",
        "PAN",
        "Permanent Account Number",
        "Aadhaar",
        "Voter ID",
        "DL No",
        "Passport No",
    ]
    text = _collapse_spaces(_strip_labels(text, label_words))

    if field_key in {"dob", "date_of_issue", "expiry_date"}:
        return _normalize_date(text)
    if field_key == "gender":
        return _normalize_gender(text)
    if field_key in {"name", "father_name", "relation_name"}:
        pieces = [piece.strip(" :-") for piece in re.split(r"[|;]", text) if piece.strip()]
        pieces = [piece for piece in pieces if piece.lower() not in JUNK_NAME_LINES]
        text = " ".join(pieces) if pieces else text
        return _collapse_spaces(re.sub(r"[^A-Za-z .'-]", " ", text)).upper()
    if field_key == "pan_number":
        candidate = re.search(r"[A-Z]{5}\d{4}[A-Z]", text.upper().replace(" ", ""))
        return candidate.group(0) if candidate else text.upper().replace(" ", "")
    if field_key in {"passport_number", "license_number", "voter_id_number"}:
        return re.sub(r"[^A-Za-z0-9/ -]", "", text).strip().upper()
    return text


def _mask_aadhaar(raw_text: str) -> tuple[str, str]:
    digits = re.sub(r"\D", "", raw_text)
    if len(digits) >= 12:
        aadhaar = digits[-12:]
        return f"xxxx-xxxx-{aadhaar[-4:]}", aadhaar
    return _collapse_spaces(raw_text), digits


def _clamp_bbox(xyxy: list[float], width: int, height: int, padding: int = 4) -> list[int]:
    x1, y1, x2, y2 = xyxy
    return [
        max(0, int(round(x1)) - padding),
        max(0, int(round(y1)) - padding),
        min(width, int(round(x2)) + padding),
        min(height, int(round(y2)) + padding),
    ]


def extract_document(input_path: str, document_type: str | None = None) -> dict[str, Any]:
    image = _load_image_bgr(input_path)
    height, width = image.shape[:2]

    raw_doc_type, doc_confidence, warnings = _detect_document_type(image, document_type)
    public_doc_type = _public_doc_type(raw_doc_type)
    model_key = DOC_TYPE_TO_MODEL.get(raw_doc_type)
    if model_key is None:
        raise ValueError(f"No field detection model mapped for document type: {raw_doc_type}")

    detector = _load_yolo(model_key)
    ocr = _load_ocr()
    results = detector(image, verbose=False)

    fields: dict[str, dict[str, Any]] = {}
    if not results:
        warnings.append("Field detector returned no results.")

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            cls_index = int(box.cls[0].item())
            raw_label = _class_name(result, model_key, cls_index)
            field_key = _field_key(public_doc_type, raw_label)
            if field_key is None:
                continue

            box_confidence = round(float(box.conf[0].item()), 4)
            bbox = _clamp_bbox(box.xyxy[0].tolist(), width, height)
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                warnings.append(f"Skipping empty crop for {raw_label}.")
                continue

            if field_key in fields and fields[field_key]["box_confidence"] >= box_confidence:
                continue

            crop = image[y1:y2, x1:x2]
            raw_text, ocr_confidence = _run_ocr(ocr, crop)
            cleaned = _clean_text(raw_text, field_key)
            field_payload: dict[str, Any] = {
                "value": cleaned,
                "box_confidence": box_confidence,
                "ocr_confidence": ocr_confidence,
                "bbox": bbox,
            }

            if field_key == "aadhaar_number":
                masked, raw_digits = _mask_aadhaar(raw_text)
                field_payload["value"] = masked
                field_payload["raw_value"] = raw_digits

            fields[field_key] = field_payload

    expected = EXPECTED_FIELDS.get(public_doc_type, [])
    missing_fields = [field for field in expected if field not in fields or not fields[field].get("value")]

    return {
        "status": "completed",
        "input_path": input_path,
        "document_type": public_doc_type,
        "document_type_confidence": round(doc_confidence, 4),
        "fields": fields,
        "missing_fields": missing_fields,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standalone Indian ID field extraction.")
    parser.add_argument("--input", required=True, help="Path to image or PDF input.")
    parser.add_argument("--output", required=True, help="Path to output JSON.")
    parser.add_argument(
        "--document-type",
        choices=sorted(DOC_TYPE_TO_MODEL.keys()),
        help="Optional document type override. Defaults to Id_Classifier.",
    )
    args = parser.parse_args(argv)

    try:
        payload = extract_document(args.input, args.document_type)
    except Exception as exc:
        LOGGER.exception("Extraction failed")
        payload = _json_failure(args.input, str(exc))

    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
