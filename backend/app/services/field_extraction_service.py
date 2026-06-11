import re


FIELD_LABELS = {
    "name": [
        "name",
        "full name",
        "applicant name",
        "candidate name",
        "student name",
        "holder name",
        "customer name"
    ],
    "dob": [
        "dob",
        "d.o.b",
        "date of birth",
        "birth date",
        "born"
    ],
    "gender": [
        "gender",
        "sex"
    ],
    "address": [
        "address",
        "addr",
        "residence",
        "permanent address",
        "current address"
    ],
    "mobile": [
        "mobile",
        "phone",
        "contact",
        "contact number"
    ],
    "father_name": [
        "father name",
        "father",
        "father's name",
        "s/o",
        "son of"
    ],
    "mother_name": [
        "mother name",
        "mother",
        "mother's name",
        "d/o",
        "daughter of"
    ],
    "document_number": [
        "document number",
        "id number",
        "id no",
        "id",
        "card number",
        "account number",
        "account no",
        "permanent account number",
        "aadhaar no",
        "aadhar no",
        "aadhaar number",
        "aadhar number",
        "pan no",
        "pan number",
        "passport no",
        "passport number",
        "license no",
        "licence no",
        "enrolment no",
        "enrollment no",
        "registration number",
        "reg no",
        "roll no"
    ],
    "vid": [
        "vid",
        "virtual id"
    ],
    "pin_code": [
        "pin code",
        "pincode",
        "postal code",
        "zip code"
    ],
    "issue_date": [
        "issue date",
        "issued on",
        "date of issue"
    ],
    "expiry_date": [
        "expiry",
        "expires",
        "expiry date",
        "valid upto",
        "valid until",
        "valid till",
        "date of expiry"
    ]
}


FIELD_ORDER = [
    "name",
    "father_name",
    "mother_name",
    "dob",
    "gender",
    "document_number",
    "vid",
    "mobile",
    "address",
    "pin_code",
    "issue_date",
    "expiry_date"
]


LABEL_BLOCKLIST_VALUES = {
    "aadhaar",
    "government of india",
    "govt of india",
    "income tax department",
    "income tax",
    "department",
    "govt",
    "government",
    "india",
    "unique identification authority of india",
    "permanent account number",
    "your aadhaar no",
    "aadhaar no",
    "aadhar no",
    "signature",
    "information",
    "identity card",
    "certificate",
    "document",
    "male female"
}


VALUE_LABEL_WORDS = {
    "name",
    "dob",
    "date",
    "birth",
    "gender",
    "sex",
    "address",
    "mobile",
    "phone",
    "father",
    "mother",
    "document",
    "number",
    "account",
    "enrolment",
    "enrollment",
    "vid",
    "pin",
    "code",
    "signature",
    "holder",
    "issued",
    "expiry",
    "valid"
}


def _clean(text):
    return re.sub(
        r"\s+",
        " ",
        str(text or "")
    ).strip()


def _normalize(text):
    normalized = re.sub(
        r"[^a-z0-9/ ]+",
        " ",
        _clean(text).lower()
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _bbox_rect(bbox):
    if not bbox or len(bbox) != 4:
        return {
            "x": 0,
            "y": 0,
            "w": 0,
            "h": 0
        }

    try:
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]

        return {
            "x": int(min(xs)),
            "y": int(min(ys)),
            "w": int(max(xs) - min(xs)),
            "h": int(max(ys) - min(ys))
        }

    except Exception:
        return {
            "x": 0,
            "y": 0,
            "w": 0,
            "h": 0
        }


def _prepare_lines(ocr_result):
    lines = []

    for index, line in enumerate(ocr_result.get("lines", [])):
        text = _clean(
            line.get("text", "")
        )

        if not text:
            continue

        rect = line.get("region") or _bbox_rect(
            line.get("bbox")
        )

        lines.append({
            "index": index,
            "text": text,
            "norm": _normalize(text),
            "confidence": float(
                line.get(
                    "confidence",
                    0
                )
            ),
            "bbox": line.get("bbox"),
            "rect": rect
        })

    return sorted(
        lines,
        key=lambda item: (
            item["rect"]["y"],
            item["rect"]["x"]
        )
    )


def _label_match(line):
    """
    Stricter label detection.

    Avoids matching weak labels like "id" or "name" when they appear
    somewhere in the middle of normal document text.
    """

    norm = line["norm"]
    raw_text = line["text"]

    if not norm:
        return None, None

    for field, labels in FIELD_LABELS.items():
        for label in labels:
            label_norm = _normalize(label)

            if not label_norm:
                continue

            # Exact label line.
            if norm == label_norm:
                return field, label_norm

            # Label at the start, e.g. "Name Rahul Sharma".
            if norm.startswith(label_norm + " "):
                return field, label_norm

            # Label with punctuation, e.g. "Name:", "DOB -".
            if re.search(
                rf"^\s*{re.escape(label)}\s*[:=\-]",
                raw_text,
                flags=re.IGNORECASE
            ):
                return field, label_norm

    return None, None


def _strip_label_from_text(text, label_norm):
    """
    Extracts same-line values:
    Name: Rahul Sharma
    DOB - 12/05/2001
    Sex F
    """

    raw = _clean(text)
    norm = _normalize(raw)

    if not label_norm or not norm.startswith(label_norm):
        return ""

    # Strong case: "Name: Rahul", "DOB - 12/05/2001".
    match = re.match(
        r"^\s*(.+?)\s*[:=\-]\s*(.+)$",
        raw
    )

    if match:
        return _clean(match.group(2))

    # Soft case: "Name Rahul Sharma", "DOB 12/05/2001".
    raw_words = raw.split()
    label_words = label_norm.split()

    if len(raw_words) > len(label_words):
        return _clean(
            " ".join(raw_words[len(label_words):])
        )

    return ""


def _is_bad_value(value):
    norm = _normalize(value)

    if not norm:
        return True

    if norm in LABEL_BLOCKLIST_VALUES:
        return True

    if len(norm) <= 1:
        return True

    return False


def _letters_only(value):
    return re.sub(
        r"[^A-Za-z ]+",
        " ",
        value
    )


def _looks_like_date(value):
    value = _clean(value)

    return bool(
        re.search(
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            value
        )
        or re.search(
            r"\b\d{1,2}[.]\d{1,2}[.]\d{2,4}\b",
            value
        )
        or re.search(
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
            value
        )
        or re.search(
            r"\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{2,4}\b",
            value,
            flags=re.IGNORECASE
        )
    )


def _looks_like_gender(value):
    norm = _normalize(value).strip()

    return norm in {
        "male",
        "female",
        "other",
        "m",
        "f",
        "transgender"
    }


def _normalize_gender(value):
    norm = _normalize(value).strip()

    if norm == "m":
        return "Male"

    if norm == "f":
        return "Female"

    if norm in {"male", "female", "other", "transgender"}:
        return norm.title()

    return _clean(value)


def _looks_like_document_number(value):
    compact = re.sub(
        r"[^A-Za-z0-9]+",
        "",
        _clean(value)
    ).upper()

    if not compact:
        return False

    if not re.search(r"\d", compact):
        return False

    return bool(
        re.fullmatch(
            r"[A-Z]{5}\d{4}[A-Z]",
            compact
        )
        or re.fullmatch(
            r"\d{12}",
            compact
        )
        or re.fullmatch(
            r"\d{16}",
            compact
        )
        or re.fullmatch(
            r"[A-Z0-9]{8,20}",
            compact
        )
    )


def _valid_value_for_field(field, value):
    value = _clean(value).strip(" :;,-")
    norm = _normalize(value)

    if _is_bad_value(value):
        return False

    if field in {"dob", "issue_date", "expiry_date"}:
        return _looks_like_date(value)

    if field == "gender":
        return _looks_like_gender(value)

    if field == "vid":
        compact = re.sub(
            r"\D+",
            "",
            value
        )
        return len(compact) == 16

    if field == "mobile":
        compact = re.sub(
            r"\D+",
            "",
            value
        )
        return bool(
            re.fullmatch(
                r"[6-9]\d{9}",
                compact
            )
        )

    if field == "pin_code":
        compact = re.sub(
            r"\D+",
            "",
            value
        )
        return bool(
            re.fullmatch(
                r"\d{6}",
                compact
            )
        )

    if field == "document_number":
        return _looks_like_document_number(value)

    if field in {"name", "father_name", "mother_name"}:
        if not re.fullmatch(
            r"[A-Za-z .'-]+",
            value
        ):
            return False

        if any(
            word in norm.split()
            for word in VALUE_LABEL_WORDS
        ):
            return False

        if re.search(r"\d", value):
            return False

        compact = re.sub(
            r"[^A-Za-z0-9]+",
            "",
            value
        )

        if re.fullmatch(
            r"[A-Z0-9]{6,}",
            compact,
            flags=re.IGNORECASE
        ) and re.search(r"\d", compact):
            return False

        letters = _letters_only(
            value
        )
        words = [
            word
            for word in letters.split()
            if len(word) > 1
        ]

        if not words or len(" ".join(words)) < 3:
            return False

        if len(words) > 6:
            return False

        return True

    if field == "address":
        if len(value) < 8:
            return False

        if re.fullmatch(
            r"[\d\s/:-]+",
            value
        ):
            return False

        return True

    return True


def _vertical_overlap(a, b):
    top = max(a["y"], b["y"])
    bottom = min(a["y"] + a["h"], b["y"] + b["h"])

    if bottom <= top:
        return 0

    return (bottom - top) / max(
        min(a["h"], b["h"]),
        1
    )


def _candidate_score(label, candidate, field):
    """
    Scores candidate values near a label.

    This reduces wrong matches like:
    name -> document number
    dob -> F
    """

    value = candidate["text"]

    if not _valid_value_for_field(field, value):
        return -9999

    label_rect = label["rect"]
    rect = candidate["rect"]

    score = 0.0

    score += candidate["confidence"] * 20

    overlap = _vertical_overlap(
        label_rect,
        rect
    )

    x_gap = rect["x"] - (
        label_rect["x"]
        + label_rect["w"]
    )

    # Same-row value on the right.
    if overlap >= 0.45 and -12 <= x_gap <= 380:
        score += 35
        score -= min(abs(x_gap) / 35, 12)

    # Value below label.
    y_gap = rect["y"] - (
        label_rect["y"]
        + label_rect["h"]
    )

    x_delta = abs(
        rect["x"]
        - label_rect["x"]
    )

    if 0 <= y_gap <= max(label_rect["h"] * 2.2, 55):
        score += 20
        score -= min(y_gap / 10, 10)

        if x_delta <= max(label_rect["w"] * 1.4, 160):
            score += 10
            score -= min(x_delta / 45, 8)

    # Penalize values that are far above or far below.
    if y_gap < -20:
        score -= 30

    if y_gap > max(label_rect["h"] * 5, 140):
        score -= 25

    # Field-specific confidence boost.
    if field in {"dob", "issue_date", "expiry_date"} and _looks_like_date(value):
        score += 25

    if field == "gender" and _looks_like_gender(value):
        score += 25

    if field == "document_number" and _looks_like_document_number(value):
        score += 22

    if field in {"name", "father_name", "mother_name"}:
        score += 20

    return score


def _find_adjacent_value(lines, label_position, field):
    label = lines[label_position]
    candidates = []

    for candidate in lines:
        if candidate["index"] == label["index"]:
            continue

        if _label_match(candidate)[0]:
            continue

        score = _candidate_score(
            label,
            candidate,
            field
        )

        if score <= -999:
            continue

        candidates.append(
            (
                -score,
                abs(candidate["rect"]["y"] - label["rect"]["y"]),
                abs(candidate["rect"]["x"] - label["rect"]["x"]),
                candidate["index"],
                candidate
            )
        )

    if candidates:
        return sorted(candidates)[0][4]

    return None


def _average_confidence(lines):
    if not lines:
        return 0

    return round(
        sum(line["confidence"] for line in lines)
        / len(lines),
        3
    )


def _add_confirmed_field(
    fields,
    confidences,
    sources,
    details,
    key,
    value,
    source_lines
):
    value = _clean(value)

    if key == "gender":
        value = _normalize_gender(value)

    if not _valid_value_for_field(key, value):
        return False

    if key in fields:
        return False

    fields[key] = value

    confidences[key] = _average_confidence(
        source_lines
    )

    sources[key] = [
        {
            "text": line["text"],
            "confidence": line["confidence"],
            "bbox": line["bbox"]
        }
        for line in source_lines
    ]

    details[key] = {
        "value": value,
        "confidence": confidences[key],
        "source": "label_anchor",
        "source_lines": sources[key]
    }

    return True


def _collect_address(lines, label_position):
    label = lines[label_position]
    label_rect = label["rect"]
    collected = []

    for candidate in lines:
        if candidate["index"] == label["index"]:
            continue

        if _label_match(candidate)[0]:
            continue

        rect = candidate["rect"]

        y_gap = rect["y"] - (
            label_rect["y"]
            + label_rect["h"]
        )

        if y_gap < 0 or y_gap > max(label_rect["h"] * 8, 180):
            continue

        if abs(rect["x"] - label_rect["x"]) > max(label_rect["w"] * 2, 240):
            continue

        # Stop collecting if a strong ID-like number appears.
        if re.search(
            r"\b\d{4}\s+\d{4}\b",
            candidate["text"]
        ):
            break

        if _label_match(candidate)[0]:
            break

        if not _valid_value_for_field(
            "address",
            candidate["text"]
        ):
            continue

        collected.append(candidate)

        if len(collected) >= 5:
            break

    return collected


def _possible_value(
    possible,
    value_type,
    value,
    line,
    reason
):
    value = _clean(value)

    if not value:
        return

    key = (
        value_type,
        value
    )

    if any(
        item["type"] == key[0]
        and item["value"] == key[1]
        for item in possible
    ):
        return

    possible.append({
        "type": value_type,
        "value": value,
        "confidence": line["confidence"],
        "source": "pattern",
        "reason": reason,
        "source_line": {
            "text": line["text"],
            "confidence": line["confidence"],
            "bbox": line["bbox"]
        }
    })


def _extract_possible_values(lines):
    possible = []

    for line in lines:
        text = line["text"]

        for match in re.finditer(
            r"\b[A-Z]{5}\d{4}[A-Z]\b",
            text
        ):
            _possible_value(
                possible,
                "pan_like_number",
                match.group(0),
                line,
                "PAN-like alphanumeric pattern"
            )

        for match in re.finditer(
            r"\b\d{4}\s+\d{4}\s+\d{4}\s+\d{4}\b",
            text
        ):
            _possible_value(
                possible,
                "vid_like_number",
                match.group(0),
                line,
                "16-digit grouped number pattern"
            )

        for match in re.finditer(
            r"\b\d{4}\s+\d{4}\s+\d{4}\b",
            text
        ):
            _possible_value(
                possible,
                "id_like_number",
                match.group(0),
                line,
                "12-digit grouped number pattern"
            )

        for match in re.finditer(
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}[.]\d{1,2}[.]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
            text
        ):
            _possible_value(
                possible,
                "date_like_value",
                match.group(0),
                line,
                "date-like pattern"
            )

        for match in re.finditer(
            r"\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{2,4}\b",
            text,
            flags=re.IGNORECASE
        ):
            _possible_value(
                possible,
                "date_like_value",
                match.group(0),
                line,
                "month-name date pattern"
            )

        compact = re.sub(
            r"\s+",
            "",
            text
        )

        for match in re.finditer(
            r"\b[6-9]\d{9}\b",
            compact
        ):
            _possible_value(
                possible,
                "mobile_like_number",
                match.group(0),
                line,
                "10-digit mobile-like pattern"
            )

        gender_match = re.search(
            r"\b(male|female|other|transgender|m|f)\b",
            text,
            flags=re.IGNORECASE
        )

        if gender_match:
            _possible_value(
                possible,
                "gender_like_value",
                _normalize_gender(gender_match.group(1)),
                line,
                "gender-like value"
            )

    return possible


def _build_field_warnings(ordered_fields):
    warnings = []

    if "name" in ordered_fields and _looks_like_document_number(
        ordered_fields["name"]
    ):
        warnings.append(
            "Name field looks like a document number; review extraction."
        )

    if "dob" in ordered_fields and not _valid_value_for_field(
        "dob",
        ordered_fields["dob"]
    ):
        warnings.append(
            "DOB field does not look like a valid date."
        )

    if "gender" in ordered_fields and not _valid_value_for_field(
        "gender",
        ordered_fields["gender"]
    ):
        warnings.append(
            "Gender field does not look like a valid gender value."
        )

    return warnings


def extract_fields(ocr_result):
    lines = _prepare_lines(
        ocr_result
    )

    fields = {}
    confidences = {}
    sources = {}
    details = {}
    consumed_indexes = set()

    for position, line in enumerate(lines):
        field, label_norm = _label_match(
            line
        )

        if not field:
            continue

        value = _strip_label_from_text(
            line["text"],
            label_norm
        )

        source_lines = [line]

        if value and not _valid_value_for_field(field, value):
            value = ""

        if not value and field == "address":
            address_lines = _collect_address(
                lines,
                position
            )

            value = ", ".join(
                item["text"]
                for item in address_lines
            )

            source_lines.extend(address_lines)

        elif not value:
            adjacent = _find_adjacent_value(
                lines,
                position,
                field
            )

            if adjacent:
                value = adjacent["text"]
                source_lines.append(adjacent)

        if _add_confirmed_field(
            fields,
            confidences,
            sources,
            details,
            field,
            value,
            source_lines
        ):
            consumed_indexes.update(
                item["index"]
                for item in source_lines
            )

    possible_values = [
        item
        for item in _extract_possible_values(lines)
        if not any(
            item["value"] == value
            for value in fields.values()
        )
    ]

    unmapped_lines = [
        {
            "text": line["text"],
            "confidence": line["confidence"],
            "bbox": line["bbox"]
        }
        for line in lines
        if line["index"] not in consumed_indexes
    ]

    ordered_fields = {
        field: fields[field]
        for field in FIELD_ORDER
        if field in fields
    }

    ordered_confidences = {
        field: confidences[field]
        for field in ordered_fields
    }

    ordered_sources = {
        field: sources[field]
        for field in ordered_fields
    }

    ordered_details = {
        field: details[field]
        for field in ordered_fields
    }

    field_warnings = _build_field_warnings(
        ordered_fields
    )

    return {
        "fields": ordered_fields,
        "field_confidences": ordered_confidences,
        "field_sources": ordered_sources,
        "field_details": ordered_details,
        "possible_values": possible_values[:30],
        "unmapped_lines": unmapped_lines[:60],
        "field_count": len(ordered_fields),
        "possible_value_count": len(possible_values),
        "field_warnings": field_warnings,
        "extraction_mode": "strict_label_anchor"
    }