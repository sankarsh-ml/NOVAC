from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)
PAGE_WIDTH, PAGE_HEIGHT = A4


def _register_font():
    candidates = [
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_path in candidates:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont("NOVACExtraction", font_path))
                return "NOVACExtraction"
            except Exception:
                continue
    return "Helvetica"


BASE_FONT = _register_font()


def _safe(value):
    if value is None:
        return ""
    return escape(str(value))


def _format_datetime(value=None):
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S UTC")
    if isinstance(value, str) and value:
        return value
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _styles():
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ExtractionTitle",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0f172a"),
            alignment=TA_CENTER,
            spaceAfter=5,
        ),
        "subtitle": ParagraphStyle(
            "ExtractionSubtitle",
            parent=sample["BodyText"],
            fontName=BASE_FONT,
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#475569"),
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "section": ParagraphStyle(
            "ExtractionSection",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=8,
            spaceAfter=7,
        ),
        "body": ParagraphStyle(
            "ExtractionBody",
            parent=sample["BodyText"],
            fontName=BASE_FONT,
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#1f2937"),
            alignment=TA_LEFT,
        ),
        "small": ParagraphStyle(
            "ExtractionSmall",
            parent=sample["BodyText"],
            fontName=BASE_FONT,
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#475569"),
        ),
        "muted": ParagraphStyle(
            "ExtractionMuted",
            parent=sample["BodyText"],
            fontName=BASE_FONT,
            fontSize=8,
            leading=10.5,
            textColor=colors.HexColor("#64748b"),
        ),
        "card_label": ParagraphStyle(
            "ExtractionCardLabel",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9,
            textColor=colors.HexColor("#64748b"),
        ),
        "card_value": ParagraphStyle(
            "ExtractionCardValue",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=colors.HexColor("#0f172a"),
        ),
        "table_header": ParagraphStyle(
            "ExtractionTableHeader",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
        ),
        "warning": ParagraphStyle(
            "ExtractionWarning",
            parent=sample["BodyText"],
            fontName=BASE_FONT,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#92400e"),
        ),
    }


def _resolve_path(path):
    if not path:
        return None

    normalized = str(path).replace("\\", "/")
    candidates = []
    raw_path = Path(normalized)
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        backend_dir = Path(__file__).resolve().parents[2]
        project_dir = backend_dir.parent
        candidates.extend([
            Path.cwd() / raw_path,
            backend_dir / raw_path,
            project_dir / raw_path,
        ])

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists():
            return str(resolved)
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


def _format_confidence(value):
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if 0 <= number <= 1:
        return f"{number * 100:.1f}%"
    return f"{number:.1f}%"


def _label(key):
    return str(key).replace("_", " ").strip().title()


def _plain_label(value):
    return _label(value or "Unknown")


def _format_aadhaar(raw_value):
    digits = re.sub(r"\D", "", str(raw_value or ""))
    if len(digits) == 12:
        return f"{digits[:4]}-{digits[4:8]}-{digits[8:]}"
    return str(raw_value or "")


def _field_value(field_key, field_data, document_type, mask_aadhaar):
    if (
        str(document_type).lower() in {"aadhaar", "aadhaar card", "aadhar"}
        and field_key == "aadhaar_number"
        and not mask_aadhaar
        and field_data.get("raw_value")
    ):
        return _format_aadhaar(field_data.get("raw_value"))
    return field_data.get("value", "")


def _table(rows, widths, header=True):
    table = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    style = [
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        style.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ])
        for row_index in range(1, len(rows)):
            if row_index % 2 == 0:
                style.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#f8fafc")))
    table.setStyle(TableStyle(style))
    return table


def _metadata_table(rows):
    table = Table(rows, colWidths=[105, 165, 95, 145], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _status_color(status):
    normalized = str(status or "").strip().lower()
    if normalized == "completed":
        return colors.HexColor("#16a34a")
    if normalized in {"failed", "error"}:
        return colors.HexColor("#dc2626")
    if normalized == "skipped":
        return colors.HexColor("#d97706")
    return colors.HexColor("#2563eb")


def _score_card(label, value, styles, accent=None):
    accent = accent or colors.HexColor("#2563eb")
    card = Table(
        [[
            Paragraph(_safe(label), styles["card_label"]),
            Paragraph(_safe(value), styles["card_value"]),
        ]],
        colWidths=[78, 92],
        hAlign="LEFT",
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


def _section(content, title, styles):
    content.append(Spacer(1, 9))
    content.append(Paragraph(title, styles["section"]))


def _page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(doc.leftMargin, 12 * mm, "NOVAC field extraction report")
    canvas.drawRightString(PAGE_WIDTH - doc.rightMargin, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_extraction_report(result, mask_aadhaar=True):
    case_id = result.get("case_id", "unknown")
    extraction = result.get("field_extraction")
    if not extraction:
        raise ValueError("Field extraction has not been run for this analysis.")

    document_type = extraction.get("document_type", "Unknown")
    report_suffix = "masked" if mask_aadhaar else "full"
    pdf_path = os.path.join(REPORT_DIR, f"extraction_report_{case_id}_{report_suffix}.pdf")
    styles = _styles()
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=32,
        leftMargin=32,
        topMargin=32,
        bottomMargin=34,
    )

    content = [
        Paragraph("Document Field Extraction Report", styles["title"]),
        Paragraph("Extracted document information only. Fraud and tamper analysis are excluded.", styles["subtitle"]),
    ]

    header_rows = [
        [
            Paragraph("<b>Case ID</b>", styles["small"]),
            Paragraph(_safe(case_id), styles["small"]),
            Paragraph("<b>File name</b>", styles["small"]),
            Paragraph(_safe(result.get("filename", "N/A")), styles["small"]),
        ],
        [
            Paragraph("<b>Generated</b>", styles["small"]),
            Paragraph(_safe(_format_datetime()), styles["small"]),
            Paragraph("<b>Extraction status</b>", styles["small"]),
            Paragraph(_safe(extraction.get("status", "unknown")), styles["small"]),
        ],
        [
            Paragraph("<b>Document type</b>", styles["small"]),
            Paragraph(_safe(document_type), styles["small"]),
            Paragraph("<b>Type confidence</b>", styles["small"]),
            Paragraph(_safe(_format_confidence(extraction.get("document_type_confidence"))), styles["small"]),
        ],
    ]
    content.append(_metadata_table(header_rows))
    content.append(Spacer(1, 10))

    fields = extraction.get("fields") or {}
    missing_fields = extraction.get("missing_fields") or []
    warnings = extraction.get("warnings") or []
    error = extraction.get("error")

    cards = [
        _score_card(
            "Status",
            _plain_label(extraction.get("status", "unknown")),
            styles,
            _status_color(extraction.get("status")),
        ),
        _score_card(
            "Document Type",
            _plain_label(document_type),
            styles,
            colors.HexColor("#2563eb"),
        ),
        _score_card(
            "Type Confidence",
            _format_confidence(extraction.get("document_type_confidence")),
            styles,
            colors.HexColor("#7c3aed"),
        ),
        _score_card(
            "Extracted Fields",
            str(len(fields)),
            styles,
            colors.HexColor("#0f766e"),
        ),
        _score_card(
            "Missing Fields",
            str(len(missing_fields)),
            styles,
            colors.HexColor("#d97706" if missing_fields else "#16a34a"),
        ),
        _score_card(
            "Report Mode",
            "Masked" if mask_aadhaar else "Full",
            styles,
            colors.HexColor("#64748b" if mask_aadhaar else "#dc2626"),
        ),
    ]
    content.append(_score_card_grid(cards))

    if not mask_aadhaar and str(document_type).lower() in {"aadhaar", "aadhaar card", "aadhar"}:
        content.append(Spacer(1, 8))
        warning_box = Table(
            [[Paragraph(
                "<b>Sensitive report:</b> This report contains an unmasked Aadhaar number and should be handled securely.",
                styles["warning"],
            )]],
            colWidths=[510],
            hAlign="LEFT",
        )
        warning_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff7ed")),
            ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#fed7aa")),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        content.append(warning_box)

    _section(content, "Document Evidence", styles)
    preview = _scaled_image(result.get("analysis_image_path") or result.get("file_path"), 260, 330)
    if preview:
        image_table = Table(
            [[preview], [Paragraph("Original / analyzed document", styles["muted"])]],
            colWidths=[510],
            hAlign="CENTER",
        )
        image_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        content.append(image_table)
    else:
        content.append(Paragraph("No image preview was available for this extraction report.", styles["body"]))

    content.append(PageBreak())

    field_rows = [[
        Paragraph("Field", styles["table_header"]),
        Paragraph("Value", styles["table_header"]),
        Paragraph("Box Confidence", styles["table_header"]),
        Paragraph("OCR Confidence", styles["table_header"]),
    ]]

    for field_key, field_data in fields.items():
        if not isinstance(field_data, dict):
            field_data = {"value": field_data}
        field_rows.append([
            Paragraph(_safe(_label(field_key)), styles["small"]),
            Paragraph(_safe(_field_value(field_key, field_data, document_type, mask_aadhaar)), styles["small"]),
            Paragraph(_safe(_format_confidence(field_data.get("box_confidence"))), styles["small"]),
            Paragraph(_safe(_format_confidence(field_data.get("ocr_confidence"))), styles["small"]),
        ])

    if len(field_rows) == 1:
        field_rows.append([
            Paragraph("No fields extracted.", styles["small"]),
            Paragraph("", styles["small"]),
            Paragraph("", styles["small"]),
            Paragraph("", styles["small"]),
        ])
    content.append(KeepTogether([
        Spacer(1, 9),
        Paragraph("Extracted Fields", styles["section"]),
        _table(field_rows, [105, 255, 75, 75]),
    ]))

    _section(content, "Missing Fields", styles)
    missing_text = ", ".join(_label(field) for field in missing_fields) if missing_fields else "None reported."
    content.append(Paragraph(_safe(missing_text), styles["body"]))

    _section(content, "Warnings / Errors", styles)
    messages = []
    if warnings:
        messages.extend(str(item) for item in warnings)
    if error:
        messages.append(str(error))
    content.append(Paragraph(_safe("; ".join(messages) if messages else "None reported."), styles["body"]))

    content.append(Spacer(1, 8))
    disclaimer = Table(
        [[Paragraph(
            "This report summarizes structured field extraction only. It excludes fraud scoring, tamper localization, and final document acceptance decisions.",
            styles["muted"],
        )]],
        colWidths=[510],
        hAlign="LEFT",
    )
    disclaimer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#cbd5e1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    content.append(KeepTogether(disclaimer))

    doc.build(content, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return pdf_path
