import re


FIELD_LABELS = {
    "name": [
        "name",
        "full name",
        "applicant name"
    ],
    "dob": [
        "dob",
        "date of birth",
        "birth date"
    ],
    "gender": [
        "gender",
        "sex"
    ],
    "address": [
        "address",
        "addr"
    ],
    "mobile": [
        "mobile",
        "phone",
        "contact"
    ],
    "father_name": [
        "father name",
        "father",
        "s/o",
        "son of"
    ],
    "mother_name": [
        "mother name",
        "mother",
        "d/o",
        "daughter of"
    ],
    "document_number": [
        "document number",
        "id number",
        "id no",
        "card number",
        "account number",
        "account no",
        "permanent account number",
        "aadhaar no",
        "aadhar no",
        "enrolment no",
        "enrollment no"
    ],
    "vid": [
        "vid",
        "virtual id"
    ],
    "pin_code": [
        "pin code",
        "pincode",
        "postal code"
    ],
    "issue_date": [
        "issue date",
        "issued on",
        "date of issue"
    ],
    "expiry_date": [
        "expiry",
        "expires",
        "valid upto",
        "valid until"
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
    "signature",
    "information"
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
    "holder"
}


def _clean(text):

    return re.sub(
        r"\s+",
        " ",
        str(text or "")
    ).strip()


def _normalize(text):

    return re.sub(
        r"[^a-z0-9/ ]+",
        " ",
        _clean(text).lower()
    )


def _bbox_rect(bbox):

    if not bbox or len(bbox) != 4:
        return {
            "x": 0,
            "y": 0,
            "w": 0,
            "h": 0
        }

    xs = [point[0] for point in bbox]
    ys = [point[1] for point in bbox]

    return {
        "x": int(min(xs)),
        "y": int(min(ys)),
        "w": int(max(xs) - min(xs)),
        "h": int(max(ys) - min(ys))
    }


def _prepare_lines(ocr_result):

    lines = []

    for index, line in enumerate(ocr_result.get("lines", [])):

        text = _clean(
            line.get("text", "")
        )

        if not text:
            continue

        rect = _bbox_rect(
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

    norm = line["norm"]

    for field, labels in FIELD_LABELS.items():
        for label in labels:
            label_norm = _normalize(label)

            if re.search(
                rf"\b{re.escape(label_norm)}\b",
                norm
            ):
                return field, label_norm

    return None, None


def _strip_label_from_text(text, label_norm):

    norm = _normalize(text)

    if label_norm not in norm:
        return ""

    parts = re.split(
        r"\s*[:=\-]\s*",
        text,
        maxsplit=1
    )

    if len(parts) == 2:
        return _clean(parts[1])

    # Only accept same-line values when the readable text starts with the label.
    if norm.startswith(label_norm):
        pattern = re.compile(
            re.escape(label_norm),
            flags=re.IGNORECASE
        )
        cleaned_norm = pattern.sub(
            "",
            norm,
            count=1
        ).strip(" /:-")

        if cleaned_norm and cleaned_norm != norm:
            return _clean(cleaned_norm)

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


def _valid_value_for_field(field, value):

    value = _clean(value).strip(" :;,-")
    norm = _normalize(value)

    if _is_bad_value(value):
        return False

    if field in {"dob", "issue_date", "expiry_date"}:
        return bool(
            re.fullmatch(
                r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
                value
            )
        )

    if field == "gender":
        return bool(
            re.fullmatch(
                r"(male|female|other)",
                norm
            )
        )

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
        compact = re.sub(
            r"[^A-Za-z0-9]+",
            "",
            value
        ).upper()

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
                r"[A-Z0-9]{8,16}",
                compact
            )
        )

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

        if re.search(r"\d", value):
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


def _find_adjacent_value(lines, label_position, field):

    label = lines[label_position]
    label_rect = label["rect"]
    candidates = []

    for candidate in lines:

        if candidate["index"] == label["index"]:
            continue

        if _label_match(candidate)[0]:
            continue

        value = candidate["text"]

        if not _valid_value_for_field(field, value):
            continue

        rect = candidate["rect"]
        overlap = _vertical_overlap(
            label_rect,
            rect
        )
        x_gap = rect["x"] - (
            label_rect["x"]
            + label_rect["w"]
        )

        if overlap >= 0.45 and -12 <= x_gap <= 300:
            candidates.append(
                (
                    abs(x_gap),
                    abs(rect["y"] - label_rect["y"]),
                    candidate["index"],
                    candidate
                )
            )

    if candidates:
        return sorted(candidates)[0][3]

    # Fallback: immediate next readable line below the label.
    below = []

    for candidate in lines:

        if candidate["index"] == label["index"]:
            continue

        if _label_match(candidate)[0]:
            continue

        value = candidate["text"]

        if not _valid_value_for_field(field, value):
            continue

        rect = candidate["rect"]
        y_gap = rect["y"] - (
            label_rect["y"]
            + label_rect["h"]
        )
        x_delta = abs(
            rect["x"]
            - label_rect["x"]
        )

        if (
            0 <= y_gap <= max(label_rect["h"] * 1.6, 38)
            and x_delta <= max(label_rect["w"] * 1.4, 150)
        ):
            below.append(
                (
                    y_gap,
                    x_delta,
                    candidate["index"],
                    candidate
                )
            )

    if below:
        return sorted(below)[0][3]

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

        if abs(rect["x"] - label_rect["x"]) > max(label_rect["w"] * 2, 220):
            continue

        if re.search(
            r"\b\d{4}\s+\d{4}\b",
            candidate["text"]
        ):
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
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            text
        ):
            _possible_value(
                possible,
                "date_like_value",
                match.group(0),
                line,
                "date-like pattern"
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

        if re.search(
            r"\b(male|female|other)\b",
            text,
            flags=re.IGNORECASE
        ):
            _possible_value(
                possible,
                "gender_like_value",
                re.search(
                    r"\b(male|female|other)\b",
                    text,
                    flags=re.IGNORECASE
                ).group(1).title(),
                line,
                "gender-like word"
            )

    return possible


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

    return {
        "fields": ordered_fields,
        "field_confidences": {
            field: confidences[field]
            for field in ordered_fields
        },
        "field_sources": {
            field: sources[field]
            for field in ordered_fields
        },
        "field_details": {
            field: details[field]
            for field in ordered_fields
        },
        "possible_values": possible_values[:30],
        "unmapped_lines": unmapped_lines[:60],
        "field_count": len(ordered_fields),
        "possible_value_count": len(possible_values),
        "extraction_mode": "strict_label_anchor"
    }
