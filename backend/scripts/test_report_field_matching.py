import sys
from pathlib import Path


def _backend_dir():
    return Path(__file__).resolve().parents[1]


def _fields_for(lines):
    backend_dir = _backend_dir()

    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    from app.api.report import extract_structured_fields

    normalized_lines = []

    for item in lines:
        if isinstance(item, tuple):
            text, confidence = item
        else:
            text, confidence = item, 0.95

        normalized_lines.append({
            "text": text,
            "confidence": confidence
        })

    ocr_result = {
        "lines": normalized_lines
    }
    result = extract_structured_fields(
        ocr_result,
        "\n".join(item["text"] for item in normalized_lines)
    )

    return {
        key: value["value"]
        for key, value in result.get("fields", {}).items()
    }


def _assert_equal(actual, expected, label):
    if actual != expected:
        raise AssertionError(
            f"{label}: expected {expected!r}, got {actual!r}"
        )


def test_aadhaar_name_matching():
    fields = _fields_for([
        "Government of India",
        "Unique Identification Authority of India.",
        "Sankarsh Ponnath",
        "p 51T6ir/DOB: 05/08/2006",
        "600r/MALE"
    ])

    _assert_equal(fields.get("document_type"), "Aadhaar", "Aadhaar document type")
    _assert_equal(fields.get("name"), "Sankarsh Ponnath", "Aadhaar name")
    _assert_equal(fields.get("dob"), "05/08/2006", "Aadhaar DOB")
    _assert_equal(fields.get("gender"), "Male", "Aadhaar gender")


def test_pan_name_matching():
    fields = _fields_for([
        ("replnla.", 0.58),
        ("TCRHTST", 0.57),
        ("PAH", 0.51),
        ("INCOME TAX DEPARTMENT", 0.95),
        ("Government.OFINDIA", 0.99),
        ("DDINESH", 0.99),
        ("PADMANABHAN DIVAKARAN", 0.98),
        ("20/04/1978", 1.0),
        ("Permanent Account Number", 0.96),
        ("AHRPD5455C", 1.0),
        ("Signature", 1.0)
    ])

    _assert_equal(fields.get("document_type"), "PAN", "PAN document type")

    if fields.get("name") not in {"DDINESH", "D DINESH"}:
        raise AssertionError(
            f"PAN name: expected 'DDINESH' or 'D DINESH', got {fields.get('name')!r}"
        )

    _assert_equal(
        fields.get("father_name"),
        "PADMANABHAN DIVAKARAN",
        "PAN father's name"
    )
    _assert_equal(fields.get("dob"), "20/04/1978", "PAN DOB")
    _assert_equal(fields.get("pan_number"), "AHRPD5455C", "PAN number")


def main():
    test_aadhaar_name_matching()
    test_pan_name_matching()
    print("Report field matching tests passed")


if __name__ == "__main__":
    main()
