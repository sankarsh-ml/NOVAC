import os
import re
from datetime import datetime
from xml.sax.saxutils import escape

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle
)

from app.services.result_service import get_result_by_case_id


router = APIRouter()

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

PAGE_WIDTH, PAGE_HEIGHT = A4


def _register_unicode_font():
    candidates = [
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]

    for font_path in candidates:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont("NOVACUnicode", font_path))
                return "NOVACUnicode"
            except Exception:
                continue

    return "Helvetica"


def _safe(value):
    if value is None:
        return ""

    return escape(str(value))


def _plain(value):
    if value is None:
        return ""

    return str(value)


def _num(value, default=0):
    try:
        return float(value)
    except Exception:
        return default


def _score(value, suffix=""):
    try:
        number = round(float(value), 2)
        if number == int(number):
            number = int(number)
        return f"{number}{suffix}"
    except Exception:
        return "N/A"


def _format_datetime(value):
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M UTC")

    text = _plain(value).strip()

    if not text:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    return text


def _pct(value):
    try:
        return f"{round(float(value) * 100, 1)}%"
    except Exception:
        return "N/A"


def _dedupe(items, limit=8):
    seen = set()
    output = []

    for item in items or []:
        text = _plain(item).strip()
        key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

        if not key or key in seen:
            continue

        seen.add(key)
        output.append(text)

        if len(output) >= limit:
            break

    return output


def _risk_color(risk_level):
    text = _plain(risk_level).lower()

    if "critical" in text or "synthetic" in text:
        return colors.HexColor("#b91c1c")
    if "high" in text:
        return colors.HexColor("#dc2626")
    if "medium" in text:
        return colors.HexColor("#d97706")
    if "low" in text:
        return colors.HexColor("#16a34a")
    if "quality" in text or "warning" in text or "unclear" in text:
        return colors.HexColor("#d97706")

    return colors.HexColor("#64748b")


def _badge_color(label):
    text = _plain(label).lower()

    if "synthetic" in text or "high" in text or "critical" in text:
        return colors.HexColor("#b91c1c")
    if "medium" in text or "warning" in text or "unclear" in text or "rescan" in text:
        return colors.HexColor("#d97706")
    if "good" in text or "low" in text or "pass" in text:
        return colors.HexColor("#16a34a")

    return colors.HexColor("#475569")


def group_aadhaar_number(value):
    digits = re.sub(r"\D", "", _plain(value))

    if len(digits) != 12:
        return _plain(value).strip()

    return f"{digits[0:4]} {digits[4:8]} {digits[8:12]}"


def group_vid_number(value):
    digits = re.sub(r"\D", "", _plain(value))

    if len(digits) != 16:
        return _plain(value).strip()

    return f"{digits[0:4]} {digits[4:8]} {digits[8:12]} {digits[12:16]}"


def format_pan_number(value):
    text = re.sub(r"[^A-Za-z0-9]", "", _plain(value)).upper()
    match = re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", text)

    return match.group(0) if match else _plain(value).strip().upper()


def _canonical_name_text(text):
    value = _plain(text).lower()
    value = re.sub(r"(?<=[a-z])(?=ofindia\b)", " ", value)
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = value.replace("govt of india", "government of india")
    value = value.replace("govt india", "government of india")
    value = value.replace("government ofindia", "government of india")

    return value


NAME_HEADER_BLACKLIST = [
    "government of india",
    "govt of india",
    "unique identification authority of india",
    "unique identification",
    "income tax department",
    "aadhaar",
    "uidai",
    "permanent account number",
    "signature",
    "enrolment no",
    "enrolment number",
    "vid",
    "address",
    "details as on",
    "aadhaar no issued",
    "document keywords",
    "mobile",
    "pin code",
    "date of birth",
    "dob",
    "gender",
    "male",
    "female",
    "transgender",
    "father",
    "name"
]


NAME_HEADER_KEYWORDS = [
    "government",
    "authority",
    "department",
    "aadhaar",
    "uidai",
    "income tax",
    "permanent account",
    "signature",
    "enrolment",
    "details as on",
    "mobile",
    "pin code"
]


def normalize_ocr_text(text):
    normalized = _plain(text)
    normalized = normalized.replace("|", "I")
    normalized = re.sub(r"\bD0B\b", "DOB", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bDOB\s*/?\s*[A-Za-z]*\b", "DOB", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bDate\s+of\s+Birth\b", "DOB", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bGender\s*/?\s*[A-Za-z]*\b", "Gender", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bMale\s*/?\s*[A-Za-z]*\b", "Male", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bFemale\s*/?\s*[A-Za-z]*\b", "Female", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bTransgender\s*/?\s*[A-Za-z]*\b", "Transgender", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bAadhaar\b|\bAadhar\b|\bAdhaar\b", "Aadhaar", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bGovernment\s+of\s+India\b", "Government of India", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s*:\s*", ": ", normalized)
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\bVID:?\s*(\d[\d\s]{12,20}\d)", lambda m: f"VID: {group_vid_number(m.group(1))}", normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"(?<!\d)(\d{12})(?!\d)",
        lambda m: group_aadhaar_number(m.group(1)),
        normalized
    )
    normalized = re.sub(
        r"(?<!\d)(\d{16})(?!\d)",
        lambda m: group_vid_number(m.group(1)),
        normalized
    )

    return normalized.strip(" -:/\n\t")


def _line_text(line):
    if isinstance(line, dict):
        return _plain(line.get("text", ""))

    return _plain(line)


def _line_confidence(line):
    if isinstance(line, dict):
        return _num(line.get("confidence", 0), 0)

    return 0


def _line_entries(ocr_result, combined_text):
    lines = []

    if isinstance(ocr_result, dict):
        for line in ocr_result.get("lines", []) or []:
            text = normalize_ocr_text(_line_text(line))
            if text:
                lines.append({
                    "text": text,
                    "raw_text": _line_text(line),
                    "confidence": _line_confidence(line)
                })

    if not lines and combined_text:
        for item in _plain(combined_text).splitlines():
            text = normalize_ocr_text(item)
            if text:
                lines.append({
                    "text": text,
                    "raw_text": item,
                    "confidence": 0
                })

    return lines


def _is_header_or_label(text):
    value = _canonical_name_text(text)

    if not value:
        return True

    if any(item in value for item in NAME_HEADER_BLACKLIST):
        return True

    if any(item in value for item in NAME_HEADER_KEYWORDS):
        return True

    return False


def _format_person_name(value):
    text = normalize_ocr_text(value)

    return text


def is_person_name_candidate(line, confidence=0, relaxed=False):
    text = normalize_ocr_text(line)
    canonical = _canonical_name_text(text)
    compact = re.sub(r"[^A-Za-z0-9]", "", text)
    words = re.findall(r"[A-Za-z]+", text)
    confidence = _num(confidence, 0)

    if confidence and confidence < 0.65:
        return False

    if confidence and confidence < 0.75 and not relaxed:
        return False

    if not text or not re.search(r"[A-Za-z]", text):
        return False

    if _is_header_or_label(text):
        return False

    if re.search(r"\d", text):
        return False

    if len(words) < 1 or len(words) > 5:
        return False

    if len(compact) < 3:
        return False

    if len(compact) < 4 and not relaxed:
        return False

    if re.fullmatch(r"[A-Za-z]{1,2}", text):
        return False

    if any(keyword in canonical for keyword in NAME_HEADER_KEYWORDS):
        return False

    if re.fullmatch(r"[A-Z]{4,10}", compact):
        vowels = len(re.findall(r"[AEIOU]", compact))

        if vowels == 0:
            return False

        if vowels <= 1 and len(compact) >= 6:
            common_initial_name = re.fullmatch(r"[A-Z][AEIOU]?[A-Z]{4,}", compact)

            if not common_initial_name:
                return False

    return True


def is_garbage_ocr_line(line, confidence=0):
    text = normalize_ocr_text(line)
    compact = re.sub(r"[^A-Za-z0-9]", "", text)

    if not text:
        return True

    useful_patterns = [
        r"\d{2}[/-]\d{2}[/-]\d{4}",
        r"\d{4}\s?\d{4}\s?\d{4}",
        r"\d{4}\s?\d{4}\s?\d{4}\s?\d{4}",
        r"[A-Z]{5}\d{4}[A-Z]",
        r"\b(name|dob|gender|aadhaar|vid|address|father|income|permanent|government)\b"
    ]

    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in useful_patterns):
        return False

    if confidence and confidence >= 0.78 and len(compact) >= 4 and re.search(r"[A-Za-z]", compact):
        return False

    if len(compact) <= 2:
        return True

    if confidence and confidence < 0.65:
        return True

    if re.fullmatch(r"[A-Za-z]{3,6}", compact) and len(set(compact.lower())) <= 4:
        return True

    if not re.search(r"[aeiouAEIOU]", compact) and re.fullmatch(r"[A-Za-z]{4,8}", compact):
        return True

    return False


def score_field_candidate(field_type, candidate, context=None):
    context = context or {}
    text = normalize_ocr_text(candidate)
    confidence = _num(context.get("confidence", 0), 0)
    near_label = bool(context.get("near_label"))
    score = 0

    if confidence >= 0.85:
        score += 25
    elif confidence >= 0.70:
        score += 15

    if near_label:
        score += 30

    if field_type == "name":
        relaxed_name = bool(
            near_label
            or context.get("pan_name_block")
            or context.get("after_to_label")
            or context.get("before_dob_or_gender")
        )

        if not is_person_name_candidate(
            text,
            confidence=confidence,
            relaxed=relaxed_name
        ):
            return -100
        if re.search(r"\b[A-Za-z]\s+[A-Za-z]{2,}\b", text):
            score += 25
        elif re.search(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}", text):
            score += 25
        elif re.fullmatch(r"[A-Z]{4,14}", re.sub(r"[^A-Za-z]", "", text)):
            score += 18
        if context.get("before_dob_or_gender"):
            score += 22
        if context.get("pan_name_block"):
            score += 28
        if context.get("after_to_label"):
            score += 25
        if re.search(r"\d", text):
            score -= 30
        if _is_header_or_label(text):
            score -= 100
    elif field_type == "dob":
        if re.search(r"\d{2}[/-]\d{2}[/-]\d{4}|\d{4}[/-]\d{2}[/-]\d{2}", text):
            score += 40
    elif field_type == "aadhaar_number":
        if re.fullmatch(r"\d{4}\s?\d{4}\s?\d{4}", text):
            score += 45
    elif field_type == "vid":
        if re.fullmatch(r"\d{4}\s?\d{4}\s?\d{4}\s?\d{4}", text):
            score += 45
    elif field_type == "pan_number":
        if re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", text):
            score += 45

    return score


def _field(label, value, confidence=0, source="OCR pattern match"):
    return {
        "field": label,
        "value": value,
        "confidence": round(float(confidence or 0), 2) if confidence else None,
        "source": source
    }


def _candidate(kind, value, reason, confidence=0):
    return {
        "type": kind,
        "value": value,
        "reason": reason,
        "confidence": round(float(confidence or 0), 2) if confidence else None
    }


def _best_candidate(field_type, candidates):
    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda item: item.get("score", 0),
        reverse=True
    )[0]


def extract_structured_fields(ocr_result, combined_text):
    entries = _line_entries(ocr_result, combined_text)
    text = normalize_ocr_text("\n".join(item["text"] for item in entries) or combined_text)
    text_lower = text.lower()
    raw_candidates = []
    fields = {}
    date_pattern = r"\d{2}[/-]\d{2}[/-]\d{4}|\d{4}[/-]\d{2}[/-]\d{2}"
    pan_pattern = r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"

    aadhaar_hits = [
        "aadhaar",
        "government of india",
        "uidai",
        "vid",
        "enrolment",
        "unique identification authority"
    ]
    pan_hits = [
        "income tax department",
        "permanent account number",
        " pan "
    ]

    if any(hit in text_lower for hit in aadhaar_hits):
        document_type = "Aadhaar"
    elif any(hit in f" {text_lower} " for hit in pan_hits):
        document_type = "PAN"
    else:
        document_type = "Unknown"

    fields["document_type"] = _field(
        "Document Type",
        document_type,
        0.90 if document_type != "Unknown" else 0,
        "Document keywords"
    )

    for match in re.finditer(r"\bVID:?\s*(\d(?:[\s-]?\d){15})\b", text, flags=re.IGNORECASE):
        value = group_vid_number(match.group(1))
        fields["vid"] = _field("VID", value, 0.90, "VID label")
        raw_candidates.append(_candidate("VID", value, "Detected after VID label", 0.90))
        break

    masked_text = re.sub(r"\bVID:?\s*\d(?:[\s-]?\d){15}\b", " ", text, flags=re.IGNORECASE)

    for match in re.finditer(r"(?<!\d)(\d{4}\s?\d{4}\s?\d{4})(?!\d)", masked_text):
        value = group_aadhaar_number(match.group(1))
        fields["aadhaar_number"] = _field("Aadhaar Number", value, 0.88, "12-digit Aadhaar pattern")
        raw_candidates.append(_candidate("Aadhaar Number", value, "12-digit Aadhaar-like pattern", 0.88))
        break

    for match in re.finditer(r"(?<!\d)(\d{4}\s?\d{4}\s?\d{4}\s?\d{4})(?!\d)", text):
        value = group_vid_number(match.group(1))
        if value != fields.get("vid", {}).get("value"):
            raw_candidates.append(_candidate("VID", value, "16-digit VID-like pattern", 0.78))
        if "vid" not in fields:
            fields["vid"] = _field("VID", value, 0.78, "16-digit VID-like pattern")
            break

    pan_match = re.search(pan_pattern, text.upper())
    if pan_match:
        pan = format_pan_number(pan_match.group(0))
        fields["pan_number"] = _field("PAN Number", pan, 0.90, "PAN regex pattern")
        raw_candidates.append(_candidate("PAN Number", pan, "PAN regex pattern", 0.90))
        if document_type == "Unknown":
            fields["document_type"] = _field("Document Type", "PAN", 0.85, "PAN number pattern")

    dob_candidates = []
    for index, entry in enumerate(entries):
        line = entry["text"]
        match = re.search(rf"\b({date_pattern})\b", line)
        if match:
            near_label = "dob" in line.lower() or any("dob" in entries[j]["text"].lower() for j in range(max(0, index - 1), index + 1))
            value = match.group(1).replace("-", "/")
            dob_candidates.append({
                "value": value,
                "confidence": entry["confidence"],
                "source": "DOB label" if near_label else "Date-like OCR line",
                "score": score_field_candidate("dob", value, {"confidence": entry["confidence"], "near_label": near_label})
            })

        yob = re.search(r"\b(?:year of birth|yob)[:\s]*(\d{4})\b", line, flags=re.IGNORECASE)
        if yob:
            fields["year_of_birth"] = _field("Year of Birth", yob.group(1), entry["confidence"], "Year of Birth label")

    best_dob = _best_candidate("dob", dob_candidates)
    if best_dob:
        fields["dob"] = _field("DOB", best_dob["value"], best_dob["confidence"], best_dob["source"])
        raw_candidates.append(_candidate("DOB", best_dob["value"], best_dob["source"], best_dob["confidence"]))

    for entry in entries:
        gender_match = re.search(r"\b(Transgender|Female|Male)\b", entry["text"], flags=re.IGNORECASE)

        if gender_match:
            value = gender_match.group(1).title()
            fields["gender"] = _field("Gender", value, entry["confidence"] or 0.86, "Gender keyword")
            break

    enrolment_match = re.search(r"\b\d{4}/\d{5}/\d{5}\b", text)
    if enrolment_match:
        fields["enrolment_number"] = _field("Enrolment Number", enrolment_match.group(0), 0.82, "Enrolment number pattern")

    issue_match = re.search(r"\b(?:details as on|issued on|issue date)[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})", text, flags=re.IGNORECASE)
    if issue_match:
        fields["issue_date"] = _field("Issue Date", issue_match.group(1), 0.78, "Issue/details date label")

    mobile_match = re.search(r"\b(?:mobile|phone)[:\s]*(\+?91[-\s]?)?([6-9]\d{9})\b", text, flags=re.IGNORECASE)
    if mobile_match:
        fields["mobile"] = _field("Mobile", mobile_match.group(2), 0.75, "Mobile number label")

    pan_header_index = None
    dob_index = None
    pan_number_index = None

    for index, entry in enumerate(entries):
        canonical = _canonical_name_text(entry["text"])
        upper_line = entry["text"].upper()

        if pan_header_index is None and (
            "income tax" in canonical
            or "government" in canonical
            or "govt" in canonical
        ):
            pan_header_index = index

        if dob_index is None and re.search(date_pattern, entry["text"]):
            dob_index = index

        if pan_number_index is None and re.search(pan_pattern, upper_line):
            pan_number_index = index

    pan_block_start = (pan_header_index + 1) if pan_header_index is not None else None
    pan_block_end_candidates = [
        index
        for index in [dob_index, pan_number_index]
        if index is not None
    ]
    pan_block_end = min(pan_block_end_candidates) if pan_block_end_candidates else None

    name_candidates = []
    for index, entry in enumerate(entries):
        line = entry["text"].strip()
        raw_line = entry["raw_text"].strip()

        label_stripped_line = line
        inline_name_label = False

        if re.search(r"\bname\b[:/]", raw_line, flags=re.IGNORECASE):
            inline_name_label = True
            label_stripped_line = re.sub(r"(?i)\bname\b\s*[:/]*", "", line).strip()

        near_label = False
        if index > 0 and re.search(r"\bname\b", entries[index - 1]["text"], flags=re.IGNORECASE):
            near_label = True
        if inline_name_label:
            near_label = True

        before_dob_or_gender = any(
            re.search(r"\b(dob|gender|male|female|transgender)\b", entries[j]["text"], flags=re.IGNORECASE)
            for j in range(index + 1, min(len(entries), index + 4))
        )
        pan_name_block = (
            fields.get("document_type", {}).get("value") == "PAN"
            and pan_block_start is not None
            and index >= pan_block_start
            and (pan_block_end is None or index < pan_block_end)
        )
        after_to_label = (
            index > 0
            and re.fullmatch(r"to", _canonical_name_text(entries[index - 1]["text"]))
        )

        if not is_person_name_candidate(
            label_stripped_line,
            confidence=entry["confidence"],
            relaxed=near_label or pan_name_block or before_dob_or_gender or after_to_label
        ):
            continue

        score = score_field_candidate(
            "name",
            label_stripped_line,
            {
                "confidence": entry["confidence"],
                "near_label": near_label,
                "before_dob_or_gender": before_dob_or_gender,
                "pan_name_block": pan_name_block,
                "after_to_label": after_to_label
            }
        )

        if score > 0:
            name_candidates.append({
                "value": _format_person_name(label_stripped_line),
                "confidence": entry["confidence"],
                "source": "OCR line after Name label" if near_label else "High-confidence person-name-like line",
                "score": score,
                "index": index,
                "near_label": near_label,
                "before_dob_or_gender": before_dob_or_gender,
                "pan_name_block": pan_name_block,
                "after_to_label": after_to_label
            })

    if fields.get("document_type", {}).get("value") == "PAN":
        pan_people = [
            item
            for item in sorted(name_candidates, key=lambda candidate: candidate["index"])
            if (
                pan_block_start is not None
                and item["index"] >= pan_block_start
                and (pan_block_end is None or item["index"] < pan_block_end)
                and item.get("pan_name_block")
            )
        ]

        if pan_people:
            fields["name"] = _field("Name", pan_people[0]["value"], pan_people[0]["confidence"], "First PAN name-block line")
            raw_candidates.append(_candidate("Name", pan_people[0]["value"], "First PAN name-block line", pan_people[0]["confidence"]))

        if len(pan_people) > 1:
            fields["father_name"] = _field("Father's Name", pan_people[1]["value"], pan_people[1]["confidence"], "Second PAN name-block line")

    else:
        selected_name = None

        for index, entry in enumerate(entries):
            if re.fullmatch(r"name", _canonical_name_text(entry["text"])):
                for lookahead in range(index + 1, min(len(entries), index + 4)):
                    candidate = next(
                        (
                            item
                            for item in name_candidates
                            if item["index"] == lookahead
                        ),
                        None
                    )

                    if candidate:
                        selected_name = {
                            **candidate,
                            "source": "OCR line after Name label"
                        }
                        break

            if selected_name:
                break

        if selected_name is None:
            for index, entry in enumerate(entries):
                if re.fullmatch(r"to", _canonical_name_text(entry["text"])):
                    for lookahead in range(index + 1, min(len(entries), index + 5)):
                        canonical = _canonical_name_text(entries[lookahead]["text"])

                        if canonical.startswith("c o") or canonical.startswith("co") or canonical.startswith("address"):
                            break

                        candidate = next(
                            (
                                item
                                for item in name_candidates
                                if item["index"] == lookahead
                            ),
                            None
                        )

                        if candidate:
                            selected_name = {
                                **candidate,
                                "source": "First valid person line after To"
                            }
                            break

                if selected_name:
                    break

        if selected_name is None:
            signal_indices = [
                index
                for index, entry in enumerate(entries)
                if re.search(r"\b(dob|gender|male|female|transgender)\b", entry["text"], flags=re.IGNORECASE)
            ]

            if signal_indices:
                first_signal = min(signal_indices)
                before_signal = [
                    item
                    for item in name_candidates
                    if item["index"] < first_signal
                ]

                if before_signal:
                    selected_name = sorted(
                        before_signal,
                        key=lambda item: (
                            abs(first_signal - item["index"]),
                            -item["score"]
                        )
                    )[0]
                    selected_name = {
                        **selected_name,
                        "source": "Nearest valid person line before DOB/Gender"
                    }

        if selected_name is None:
            selected_name = _best_candidate("name", name_candidates)

        if selected_name:
            fields["name"] = _field("Name", selected_name["value"], selected_name["confidence"], selected_name["source"])
            raw_candidates.append(_candidate("Name", selected_name["value"], selected_name["source"], selected_name["confidence"]))

    address_lines = []
    capture = False
    for entry in entries:
        line = entry["text"]

        if re.search(r"\b(address|to)\b[:\s]*", line, flags=re.IGNORECASE):
            capture = True
            cleaned = re.sub(r"(?i)\b(address|to)\b[:\s]*", "", line).strip()
            if cleaned:
                address_lines.append(cleaned)
            continue

        if capture:
            if re.search(r"\b(vid|aadhaar|uidai|government|dob|gender)\b", line, flags=re.IGNORECASE):
                break
            if re.search(r"\d{4}\s?\d{4}\s?\d{4}", line):
                break
            if not is_garbage_ocr_line(line, entry["confidence"]):
                address_lines.append(line)

        if len(address_lines) >= 4:
            break

    if address_lines:
        fields["address"] = _field("Address", ", ".join(address_lines), 0.70, "Address block")

    field_order = [
        "document_type",
        "name",
        "dob",
        "year_of_birth",
        "gender",
        "aadhaar_number",
        "vid",
        "pan_number",
        "father_name",
        "address",
        "enrolment_number",
        "issue_date",
        "mobile"
    ]

    ordered_fields = {
        key: fields[key]
        for key in field_order
        if key in fields and fields[key].get("value")
    }

    return {
        "document_type": ordered_fields.get("document_type", {}).get("value", "Unknown"),
        "fields": ordered_fields,
        "raw_candidates": raw_candidates[:20]
    }


def clean_ocr_lines(ocr_result):
    entries = _line_entries(ocr_result, "")
    cleaned = []

    for entry in entries:
        text = normalize_ocr_text(entry["text"])

        if is_garbage_ocr_line(text, entry["confidence"]):
            continue

        cleaned.append({
            **entry,
            "text": text
        })

    return cleaned


def _structured_field_rows(structured_fields, field_analysis, styles):
    fields = structured_fields.get("fields", {}) or {}
    legacy_fields = (field_analysis or {}).get("fields", {}) or {}
    legacy_confidences = (field_analysis or {}).get("field_confidences", {}) or {}

    for key, value in legacy_fields.items():
        if key not in fields and value:
            fields[key] = _field(
                key.replace("_", " ").title(),
                value,
                legacy_confidences.get(key),
                "Existing field extractor"
            )

    rows = [[
        Paragraph("<b>Field</b>", styles["TableHeader"]),
        Paragraph("<b>Value</b>", styles["TableHeader"]),
        Paragraph("<b>Confidence</b>", styles["TableHeader"]),
        Paragraph("<b>Source / Reason</b>", styles["TableHeader"])
    ]]

    for item in fields.values():
        rows.append([
            Paragraph(_safe(item.get("field", "")), styles["SmallBody"]),
            Paragraph(_safe(item.get("value", "")), styles["SmallBody"]),
            Paragraph(_score(item.get("confidence")) if item.get("confidence") is not None else "N/A", styles["SmallBody"]),
            Paragraph(_safe(item.get("source", "")), styles["SmallBody"])
        ])

    if len(rows) == 1:
        rows.append([
            Paragraph("-", styles["SmallBody"]),
            Paragraph("No structured fields were confidently extracted.", styles["SmallBody"]),
            Paragraph("-", styles["SmallBody"]),
            Paragraph("OCR did not contain enough anchored field evidence.", styles["SmallBody"])
        ])

    return rows


def _possible_value_rows(structured_fields, field_analysis, styles):
    possible = []
    possible.extend(structured_fields.get("raw_candidates", []) or [])
    possible.extend((field_analysis or {}).get("possible_values", []) or [])

    rows = [[
        Paragraph("<b>Type</b>", styles["TableHeader"]),
        Paragraph("<b>Value</b>", styles["TableHeader"]),
        Paragraph("<b>Reason</b>", styles["TableHeader"])
    ]]

    seen = set()
    for item in possible[:30]:
        kind = _plain(item.get("type", "")).replace("_", " ").title()
        value = _plain(item.get("value", ""))
        reason = _plain(item.get("reason", ""))
        key = (kind.lower(), value.lower())

        if not value or key in seen:
            continue

        seen.add(key)
        rows.append([
            Paragraph(_safe(kind), styles["SmallBody"]),
            Paragraph(_safe(value), styles["SmallBody"]),
            Paragraph(_safe(reason), styles["SmallBody"])
        ])

    if len(rows) == 1:
        rows.append([
            Paragraph("-", styles["SmallBody"]),
            Paragraph("No uncertain candidates detected.", styles["SmallBody"]),
            Paragraph("Only confident fields are shown above.", styles["SmallBody"])
        ])

    return rows


def _cleaned_text(structured_fields, cleaned_lines):
    fields = structured_fields.get("fields", {}) or {}
    canonical = []

    label_map = {
        "document_type": "Document Type",
        "name": "Name",
        "dob": "DOB",
        "year_of_birth": "Year of Birth",
        "gender": "Gender",
        "aadhaar_number": "Aadhaar Number",
        "vid": "VID",
        "pan_number": "PAN Number",
        "father_name": "Father's Name",
        "address": "Address",
        "enrolment_number": "Enrolment Number",
        "issue_date": "Issue Date",
        "mobile": "Mobile"
    }

    for key, label in label_map.items():
        if key in fields:
            canonical.append(f"{label}: {fields[key].get('value', '')}")

    if canonical:
        return "\n".join(canonical)

    return "\n".join(item["text"] for item in cleaned_lines[:80]) or "No cleaned OCR text available."


def _raw_ocr_rows(ocr_lines, styles):
    rows = [[
        Paragraph("<b>#</b>", styles["TableHeader"]),
        Paragraph("<b>Raw OCR Line</b>", styles["TableHeader"]),
        Paragraph("<b>Confidence</b>", styles["TableHeader"])
    ]]

    for index, line in enumerate(ocr_lines[:80], start=1):
        rows.append([
            Paragraph(str(index), styles["SmallBody"]),
            Paragraph(_safe(_line_text(line)), styles["SmallBody"]),
            Paragraph(_score(_line_confidence(line)), styles["SmallBody"])
        ])

    if len(rows) == 1:
        rows.append([
            Paragraph("-", styles["SmallBody"]),
            Paragraph("No raw OCR lines available.", styles["SmallBody"]),
            Paragraph("-", styles["SmallBody"])
        ])

    return rows


def _make_styles():
    base = getSampleStyleSheet()
    body_font = _register_unicode_font()

    return {
        **base.byName,
        "Title": ParagraphStyle(
            "NOVACTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0f172a"),
            alignment=TA_CENTER,
            spaceAfter=5
        ),
        "Subtitle": ParagraphStyle(
            "NOVACSubtitle",
            parent=base["BodyText"],
            fontName=body_font,
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#475569"),
            alignment=TA_CENTER,
            spaceAfter=10
        ),
        "Section": ParagraphStyle(
            "NOVACSection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=8,
            spaceAfter=7
        ),
        "Body": ParagraphStyle(
            "NOVACBody",
            parent=base["BodyText"],
            fontName=body_font,
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#1f2937"),
            alignment=TA_LEFT
        ),
        "SmallBody": ParagraphStyle(
            "NOVACSmallBody",
            parent=base["BodyText"],
            fontName=body_font,
            fontSize=8,
            leading=10.5,
            textColor=colors.HexColor("#334155")
        ),
        "Muted": ParagraphStyle(
            "NOVACMuted",
            parent=base["BodyText"],
            fontName=body_font,
            fontSize=8,
            leading=10.5,
            textColor=colors.HexColor("#64748b")
        ),
        "CardLabel": ParagraphStyle(
            "NOVACCardLabel",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9,
            textColor=colors.HexColor("#64748b")
        ),
        "CardValue": ParagraphStyle(
            "NOVACCardValue",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#0f172a")
        ),
        "TableHeader": ParagraphStyle(
            "NOVACTableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white
        )
    }


def _page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(doc.leftMargin, 12 * mm, "NOVAC automated document analysis")
    canvas.drawRightString(PAGE_WIDTH - doc.rightMargin, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _section(content, title, styles):
    content.append(Spacer(1, 9))
    content.append(Paragraph(title, styles["Section"]))


def _wrapped_table(rows, col_widths, repeat_rows=1, row_background=True):
    table = Table(
        rows,
        colWidths=col_widths,
        hAlign="LEFT",
        repeatRows=repeat_rows
    )
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
    ]

    if row_background:
        for row_index in range(1, len(rows)):
            if row_index % 2 == 0:
                style.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#f8fafc")))

    table.setStyle(TableStyle(style))
    return table


def build_badge(label, styles, color=None):
    color = color or _badge_color(label)
    table = Table(
        [[Paragraph(f"<b>{_safe(label)}</b>", styles["SmallBody"])]],
        colWidths=[120],
        hAlign="LEFT"
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0, color),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def build_score_card(label, value, styles, accent=None):
    accent = accent or colors.HexColor("#2563eb")
    card = Table(
        [[
            Paragraph(_safe(label), styles["CardLabel"]),
            Paragraph(_safe(value), styles["CardValue"])
        ]],
        colWidths=[92, 78],
        hAlign="LEFT"
    )
    card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#cbd5e1")),
        ("LINEBEFORE", (0, 0), (0, -1), 4, accent),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return card


def _score_card_grid(cards):
    rows = []
    for index in range(0, len(cards), 3):
        row = cards[index:index + 3]
        while len(row) < 3:
            row.append("")
        rows.append(row)

    table = Table(rows, colWidths=[170, 170, 170], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _resolve_path(path):
    if not path:
        return None

    if os.path.exists(path):
        return path

    candidate = os.path.join(os.getcwd(), path)
    if os.path.exists(candidate):
        return candidate

    backend_candidate = os.path.join(os.getcwd(), "backend", path)
    if os.path.exists(backend_candidate):
        return backend_candidate

    return None


def _scaled_image(path, max_width, max_height):
    actual_path = _resolve_path(path)

    if not actual_path:
        return None

    try:
        image = Image(actual_path)
        width, height = image.imageWidth, image.imageHeight
        scale = min(max_width / float(width or 1), max_height / float(height or 1), 1)
        image.drawWidth = width * scale
        image.drawHeight = height * scale
        return image
    except Exception:
        return None


def _summary_text(result, fraud_analysis, quality_analysis, authenticity_analysis):
    fraud_score = _num(fraud_analysis.get("fraud_score", 0), 0)
    risk_level = _plain(fraud_analysis.get("risk_level", "Unknown"))
    reasons = _dedupe(fraud_analysis.get("reasons", []), 3)
    quality_status = _plain(quality_analysis.get("quality_status", "unknown"))

    if authenticity_analysis.get("synthetic_detected") or fraud_analysis.get("result_status") == "synthetic_suspected":
        conclusion = "Authenticity concern detected."
    elif fraud_analysis.get("result_status") == "unprocessable":
        conclusion = "Document quality prevents reliable automated verification."
    elif fraud_score >= 50:
        conclusion = "High-risk document indicators detected."
    elif fraud_score >= 25:
        conclusion = "Moderate risk indicators detected."
    else:
        conclusion = "No major fraud indicators detected."

    reason = reasons[0] if reasons else "No single dominant risk reason was identified."
    quality_note = result.get("quality_notice") or fraud_analysis.get("quality_notice")

    if not quality_note:
        quality_note = (
            "Quality acceptable for automated analysis."
            if quality_status in ("good", "acceptable", "")
            else f"Document quality status: {quality_status}."
        )

    return f"{conclusion} Risk level is {risk_level}. Key reason: {reason} {quality_note}"


def _key_findings(result, fraud_analysis, quality_analysis, authenticity_analysis):
    findings = []
    findings.extend(fraud_analysis.get("reasons", []) or [])

    if authenticity_analysis.get("synthetic_detected"):
        findings.append("Synthetic document authenticity signal detected")
    if authenticity_analysis.get("acquisition_type") in ("digital_or_synthetic", "unknown"):
        findings.append("Weak natural camera or print acquisition traces")
    if quality_analysis.get("quality_status") in ("good", "acceptable"):
        findings.append("Quality acceptable for automated analysis")
    elif quality_analysis.get("quality_status"):
        findings.append(f"Document quality status: {quality_analysis.get('quality_status')}")
    if result.get("quality_notice"):
        findings.append(result.get("quality_notice"))

    return _dedupe(findings, 8)


def _detector_status(ok_text, warning_text, condition):
    return warning_text if condition else ok_text


def _detector_rows(result, styles):
    fraud_analysis = result.get("fraud_analysis", {}) or {}
    metadata = result.get("metadata_analysis", {}) or {}
    tampering = result.get("tampering_analysis", {}) or {}
    ela = result.get("ela_analysis", {}) or {}
    masking = result.get("masking_analysis", {}) or {}
    quality = result.get("document_quality_analysis", {}) or {}
    authenticity = result.get("document_authenticity_analysis", {}) or {}
    photo = result.get("photo_replacement_analysis", {}) or {}
    trufor = result.get("forgery_localization_analysis", {}) or {}
    text_consistency = result.get("text_consistency_analysis", {}) or {}
    visual = result.get("visual_consistency_analysis", {}) or {}
    correlation = result.get("correlation_analysis", {}) or {}

    rows = [[
        Paragraph("<b>Detector</b>", styles["TableHeader"]),
        Paragraph("<b>Score / Metric</b>", styles["TableHeader"]),
        Paragraph("<b>Status</b>", styles["TableHeader"]),
        Paragraph("<b>Explanation</b>", styles["TableHeader"])
    ]]

    detector_data = [
        (
            "Metadata",
            f"Risk {_score(metadata.get('risk_score', 0))}",
            "Clear" if not metadata.get("flags") else "Flagged",
            "; ".join(metadata.get("flags", [])[:3]) or "No metadata-specific flags were found."
        ),
        (
            "OCR / Masking",
            f"OCR {_safe(result.get('avg_confidence', 'N/A'))}",
            "Masked fields detected" if masking.get("masking_detected") else "No masking signal",
            "; ".join(masking.get("reasons", [])[:3]) or "OCR completed and no strong masking signal was reported."
        ),
        (
            "TruFor",
            f"Score {_score(trufor.get('forgery_score', 0))}",
            "Unavailable" if trufor.get("model_available") is False else ("Signal detected" if trufor.get("manipulation_detected") else "No strong signal"),
            "; ".join(trufor.get("reasons", [])[:3]) or trufor.get("model_error") or "No strong TruFor forgery localization signal."
        ),
        (
            "MVSS",
            f"Score {_score(tampering.get('tampering_score', 0))}",
            "Inconclusive" if tampering.get("completed") is False else ("Signal detected" if tampering.get("tampering_detected") else "No strong signal"),
            "; ".join(tampering.get("reasons", [])[:3]) or tampering.get("error") or f"{tampering.get('scoring_region_count', tampering.get('suspicious_region_count', 0))} scoring-eligible region(s)."
        ),
        (
            "Text Consistency",
            f"Score {_score(text_consistency.get('field_mismatch_score', 0))}",
            "Mismatch detected" if text_consistency.get("font_mismatch_detected") else "No strong signal",
            "; ".join(text_consistency.get("reasons", [])[:3]) or "No strong local field-level text mismatch detected."
        ),
        (
            "ELA",
            f"Score {_score(ela.get('ela_score', 0))}",
            "Supporting signal" if ela.get("suspicious_regions") else "No strong signal",
            f"{len(ela.get('suspicious_regions', []) or [])} suspicious compression region(s) reported."
        ),
        (
            "Visual Consistency",
            f"Score {_score(visual.get('consistency_score', 0))}",
            "Inconsistency detected" if visual.get("inconsistent_regions") else "No major inconsistency",
            "; ".join(visual.get("reasons", [])[:3]) or "No major region-level visual inconsistency was reported."
        ),
        (
            "Document Quality",
            f"Quality {_score(quality.get('quality_score', 'N/A'))}",
            quality.get("quality_status", "Unknown"),
            "; ".join(quality.get("reasons", [])[:3]) or "Quality context was incorporated into the decision."
        ),
        (
            "Document Authenticity",
            f"Synthetic {_score(authenticity.get('synthetic_score', 0))}",
            "Synthetic suspected" if authenticity.get("synthetic_detected") else "No strong synthetic signal",
            "; ".join(authenticity.get("reasons", [])[:3]) or "No strong synthetic-document indicators were reported."
        ),
        (
            "Photo Integrity",
            f"Score {_score(photo.get('replacement_score', 0))}",
            "Photo concern" if photo.get("critical_photo_issue") else "No strong signal",
            "; ".join(photo.get("reasons", [])[:3]) or "No strong photo replacement or print integrity signal."
        ),
        (
            "Detector Agreement",
            f"{correlation.get('suspicious_field_count', 0)} overlap(s)",
            "Agreement found" if correlation.get("suspicious_field_count", 0) else "No strong agreement",
            "OCR fields overlap visual evidence." if correlation.get("suspicious_field_count", 0) else "No suspicious OCR-field overlap was reported."
        )
    ]

    for detector, metric, status, explanation in detector_data:
        rows.append([
            Paragraph(_safe(detector), styles["SmallBody"]),
            Paragraph(_safe(metric), styles["SmallBody"]),
            Paragraph(_safe(status), styles["SmallBody"]),
            Paragraph(_safe(explanation), styles["SmallBody"])
        ])

    return rows


def _contribution_rows(result, styles):
    contributions = ((result.get("fraud_analysis") or {}).get("detector_contributions", {}) or {})
    rows = [[
        Paragraph("<b>Detector</b>", styles["TableHeader"]),
        Paragraph("<b>Contribution</b>", styles["TableHeader"]),
        Paragraph("<b>Reason</b>", styles["TableHeader"])
    ]]

    for key, item in contributions.items():
        contribution = _num(item.get("contribution", 0), 0)
        reason = item.get("reason") or ("No strong signal detected" if contribution == 0 else "Contributed to risk")
        rows.append([
            Paragraph(_safe(key.replace("_", " ").title()), styles["SmallBody"]),
            Paragraph(_safe(_score(contribution)), styles["SmallBody"]),
            Paragraph(_safe(reason), styles["SmallBody"])
        ])

    if len(rows) == 1:
        rows.append([
            Paragraph("-", styles["SmallBody"]),
            Paragraph("0", styles["SmallBody"]),
            Paragraph("No detector contribution details were available.", styles["SmallBody"])
        ])

    return rows


def _verdict(result, fraud_analysis, quality_analysis, authenticity_analysis):
    fraud_score = _num(fraud_analysis.get("fraud_score", 0), 0)
    result_status = fraud_analysis.get("result_status") or result.get("result_status")

    if result_status == "unprocessable":
        return (
            "Document quality notice",
            "The upload could not be analyzed reliably because quality issues may reduce confidence.",
            "Request a clearer rescan before accepting this document."
        )

    if result_status == "synthetic_suspected" or authenticity_analysis.get("synthetic_detected"):
        return (
            "Document authenticity concern detected",
            "Authenticity checks found signals consistent with a synthetic, AI-generated, or digitally fabricated document.",
            "Manual verification is strongly recommended before accepting this document."
        )

    if fraud_score >= 75:
        return (
            "High-risk document indicators detected",
            "Multiple detector signals indicate a high probability of manipulation or integrity concerns.",
            "Review suspicious regions and detector findings manually before accepting this document."
        )

    if fraud_score >= 50:
        return (
            "High-risk document indicators detected",
            "Several suspicious indicators were detected across the analysis pipeline.",
            "Manual review is recommended before acceptance."
        )

    if fraud_score >= 25:
        return (
            "Moderate document risk detected",
            "Limited or moderate indicators were detected.",
            "Proceed with additional standard verification for high-value or sensitive workflows."
        )

    return (
        "No major fraud indicators detected",
        "The available detectors did not report major fraud or integrity signals.",
        "Document may proceed to standard verification."
    )


@router.get("/report/{case_id}")
def generate_report(case_id: str):
    result = get_result_by_case_id(case_id)

    if not result:
        raise HTTPException(status_code=404, detail="Case not found")

    pdf_path = os.path.join(REPORT_DIR, f"report_{case_id}.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=32,
        leftMargin=32,
        topMargin=32,
        bottomMargin=34
    )
    styles = _make_styles()
    content = []

    fraud_analysis = result.get("fraud_analysis", {}) or {}
    quality_analysis = result.get("document_quality_analysis", {}) or {}
    authenticity_analysis = result.get("document_authenticity_analysis", {}) or {}
    field_analysis = result.get("field_extraction_analysis", {}) or {}
    ocr_result = {
        "lines": result.get("lines", []) or [],
        "avg_confidence": result.get("avg_confidence", 0)
    }
    combined_text = result.get("text", "") or ""
    structured_fields = extract_structured_fields(ocr_result, combined_text)
    cleaned_lines = clean_ocr_lines(ocr_result)
    cleaned_text = _cleaned_text(structured_fields, cleaned_lines)

    fraud_score = fraud_analysis.get("fraud_score", 0)
    risk_level = fraud_analysis.get("risk_level", "Unknown")
    quality_badge = fraud_analysis.get("quality_badge") or result.get("quality_badge")
    authenticity_score = (
        authenticity_analysis.get("authenticity_score")
        if authenticity_analysis.get("authenticity_score") is not None
        else 100 - _num(authenticity_analysis.get("synthetic_score", 0), 0)
    )
    quality_score = quality_analysis.get("quality_score", "N/A")
    evidence_count = len(_dedupe(fraud_analysis.get("reasons", []), 50))

    content.append(Paragraph("NOVAC Document Analysis Report", styles["Title"]))
    content.append(Paragraph("Document integrity, authenticity, and forensic review", styles["Subtitle"]))

    header_rows = [
        [
            Paragraph("<b>Case ID</b>", styles["SmallBody"]),
            Paragraph(_safe(result.get("case_id", case_id)), styles["SmallBody"]),
            Paragraph("<b>File name</b>", styles["SmallBody"]),
            Paragraph(_safe(result.get("filename", "N/A")), styles["SmallBody"])
        ],
        [
            Paragraph("<b>Analysis date/time</b>", styles["SmallBody"]),
            Paragraph(_safe(_format_datetime(result.get("timestamp"))), styles["SmallBody"]),
            Paragraph("<b>Risk Level</b>", styles["SmallBody"]),
            build_badge(_plain(risk_level), styles, _risk_color(risk_level))
        ]
    ]

    if quality_badge:
        header_rows.append([
            Paragraph("<b>Quality Badge</b>", styles["SmallBody"]),
            build_badge(_plain(quality_badge), styles),
            Paragraph("<b>Status</b>", styles["SmallBody"]),
            Paragraph(_safe(result.get("status", "N/A")), styles["SmallBody"])
        ])

    header = Table(header_rows, colWidths=[105, 165, 95, 145], hAlign="LEFT")
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
    ]))
    content.append(header)
    content.append(Spacer(1, 10))

    cards = [
        build_score_card("Fraud Score", f"{_score(fraud_score)}/100", styles, _risk_color(risk_level)),
        build_score_card("Risk Level", _plain(risk_level), styles, _risk_color(risk_level)),
        build_score_card("Authenticity", _score(authenticity_score), styles, _badge_color("good" if _num(authenticity_score, 0) >= 70 else "warning")),
        build_score_card("Document Quality", _score(quality_score), styles, _badge_color(quality_badge or quality_analysis.get("quality_status", ""))),
        build_score_card("OCR Confidence", _safe(result.get("avg_confidence", "N/A")), styles, colors.HexColor("#2563eb")),
        build_score_card("Evidence Count", str(evidence_count), styles, colors.HexColor("#7c3aed")),
    ]
    content.append(_score_card_grid(cards))

    _section(content, "Executive Summary", styles)
    content.append(Paragraph(_safe(_summary_text(result, fraud_analysis, quality_analysis, authenticity_analysis)), styles["Body"]))

    _section(content, "Key Findings", styles)
    findings = _key_findings(result, fraud_analysis, quality_analysis, authenticity_analysis)
    if findings:
        content.append(Paragraph("<br/>".join(f"- {_safe(item)}" for item in findings), styles["Body"]))
    else:
        content.append(Paragraph("No significant findings were reported by the available detectors.", styles["Body"]))

    content.append(PageBreak())
    _section(content, "Document Evidence", styles)

    image_items = []
    original_image = _scaled_image(result.get("analysis_image_path") or result.get("file_path"), 245, 315)
    annotated_image = _scaled_image(result.get("annotated_image_path"), 245, 315)

    if original_image:
        image_items.append((original_image, "Original / analyzed document"))
    if annotated_image:
        image_items.append((annotated_image, "Annotated evidence image"))

    if len(image_items) == 2:
        image_table = Table(
            [
                [image_items[0][0], image_items[1][0]],
                [Paragraph(image_items[0][1], styles["Muted"]), Paragraph(image_items[1][1], styles["Muted"])]
            ],
            colWidths=[255, 255],
            hAlign="CENTER"
        )
    elif len(image_items) == 1:
        image_table = Table(
            [[image_items[0][0]], [Paragraph(image_items[0][1], styles["Muted"])]],
            colWidths=[510],
            hAlign="CENTER"
        )
    else:
        image_table = Table(
            [[Paragraph("No document preview image was available for this report.", styles["Body"])]],
            colWidths=[510]
        )

    image_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5)
    ]))
    content.append(image_table)

    content.append(PageBreak())
    _section(content, "Detector Summary", styles)
    content.append(_wrapped_table(_detector_rows(result, styles), [95, 85, 95, 235]))

    _section(content, "Detector Contributions", styles)
    content.append(_wrapped_table(_contribution_rows(result, styles), [120, 85, 305]))

    content.append(PageBreak())
    _section(content, "Extracted Text", styles)
    content.append(Paragraph("Structured Fields", styles["Body"]))
    content.append(Spacer(1, 5))
    content.append(_wrapped_table(_structured_field_rows(structured_fields, field_analysis, styles), [105, 185, 75, 145]))

    content.append(Spacer(1, 10))
    content.append(Paragraph("Possible Detected Values", styles["Body"]))
    content.append(Spacer(1, 5))
    content.append(_wrapped_table(_possible_value_rows(structured_fields, field_analysis, styles), [110, 165, 235]))

    content.append(Spacer(1, 10))
    content.append(Paragraph("Cleaned Extracted Text", styles["Body"]))
    text_panel = Table(
        [[Paragraph(_safe(cleaned_text[:3500]).replace("\n", "<br/>"), styles["Body"])]],
        colWidths=[510],
        hAlign="LEFT"
    )
    text_panel.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9)
    ]))
    content.append(text_panel)

    content.append(PageBreak())
    _section(content, "OCR Details", styles)
    content.append(Paragraph("Raw OCR lines are shown here for auditability. Low-confidence fragments are intentionally kept out of the main cleaned text.", styles["Muted"]))
    content.append(Spacer(1, 6))
    content.append(_wrapped_table(_raw_ocr_rows(result.get("lines", []) or [], styles), [32, 405, 73]))

    _section(content, "Final Verdict", styles)
    verdict_title, verdict_body, recommendation = _verdict(result, fraud_analysis, quality_analysis, authenticity_analysis)
    verdict_box = Table(
        [[
            Paragraph(f"<b>{_safe(verdict_title)}</b><br/>{_safe(verdict_body)}<br/><br/><b>Recommended action:</b> {_safe(recommendation)}", styles["Body"])
        ]],
        colWidths=[510],
        hAlign="LEFT"
    )
    verdict_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.8, _risk_color(risk_level)),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10)
    ]))
    content.append(KeepTogether(verdict_box))
    content.append(Spacer(1, 8))
    content.append(Paragraph(
        "Disclaimer: This report is automated document risk analysis and is not proof of fraud. Manual verification is recommended before making acceptance or rejection decisions.",
        styles["Muted"]
    ))

    doc.build(content, onFirstPage=_page_footer, onLaterPages=_page_footer)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{case_id}.pdf"
    )
