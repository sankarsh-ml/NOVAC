import os
from xml.sax.saxutils import escape

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle
)

from app.services.result_service import (
    get_result_by_case_id
)

router = APIRouter()


REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


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
                pdfmetrics.registerFont(
                    TTFont(
                        "NOVACUnicode",
                        font_path
                    )
                )
                return "NOVACUnicode"
            except Exception:
                continue

    return "Helvetica"


def _risk_color(risk_level):

    return {
        "Low": colors.HexColor("#16a34a"),
        "Medium": colors.HexColor("#ca8a04"),
        "High": colors.HexColor("#ea580c"),
        "Critical": colors.HexColor("#dc2626")
    }.get(
        risk_level,
        colors.HexColor("#64748b")
    )


def _safe(value):

    return escape(
        str(value)
    )


def _score(value):

    try:
        return str(
            round(
                float(value),
                2
            )
        )
    except Exception:
        return "N/A"


def _table(data, col_widths=None, header=True):

    table = Table(
        data,
        colWidths=col_widths,
        hAlign="LEFT"
    )

    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]

    if header:
        style.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])

    table.setStyle(
        TableStyle(style)
    )

    return table


def _section(content, title, styles):

    content.append(
        Spacer(1, 14)
    )
    content.append(
        Paragraph(
            title,
            styles["NovacSection"]
        )
    )
    content.append(
        Spacer(1, 7)
    )


def _reasons_block(reasons, styles):

    if not reasons:
        return Paragraph(
            "No significant fraud indicators detected.",
            styles["NovacBody"]
        )

    return Paragraph(
        "<br/>".join(
            f"- {_safe(reason)}"
            for reason in reasons[:14]
        ),
        styles["NovacBody"]
    )


def _ocr_lines_table(lines, styles):

    rows = [
        [
            Paragraph("<b>#</b>", styles["NovacBody"]),
            Paragraph("<b>Extracted Text</b>", styles["NovacBody"]),
            Paragraph("<b>Confidence</b>", styles["NovacBody"])
        ]
    ]

    for index, line in enumerate((lines or [])[:60], start=1):
        rows.append([
            str(index),
            Paragraph(
                _safe(line.get("text", "")),
                styles["NovacBody"]
            ),
            _score(
                line.get(
                    "confidence",
                    "N/A"
                )
            )
        ])

    if len(rows) == 1:
        rows.append([
            "-",
            Paragraph(
                "No OCR lines available.",
                styles["NovacBody"]
            ),
            "-"
        ])

    return _table(
        rows,
        col_widths=[32, 380, 80]
    )


def _field_table(field_analysis, styles):

    fields = field_analysis.get(
        "fields",
        {}
    )
    confidences = field_analysis.get(
        "field_confidences",
        {}
    )

    rows = [
        [
            Paragraph("<b>Field</b>", styles["NovacBody"]),
            Paragraph("<b>Value</b>", styles["NovacBody"]),
            Paragraph("<b>Confidence</b>", styles["NovacBody"])
        ]
    ]

    for key, value in fields.items():
        rows.append([
            Paragraph(
                _safe(key.replace("_", " ").title()),
                styles["NovacBody"]
            ),
            Paragraph(
                _safe(value),
                styles["NovacBody"]
            ),
            _score(
                confidences.get(
                    key,
                    "N/A"
                )
            )
        ])

    if len(rows) == 1:
        rows.append([
            "-",
            Paragraph(
                "No structured fields were confidently extracted.",
                styles["NovacBody"]
            ),
            "-"
        ])

    return _table(
        rows,
        col_widths=[130, 300, 70]
    )


def _possible_values_table(field_analysis, styles):

    possible_values = field_analysis.get(
        "possible_values",
        []
    )

    rows = [
        [
            Paragraph("<b>Type</b>", styles["NovacBody"]),
            Paragraph("<b>Value</b>", styles["NovacBody"]),
            Paragraph("<b>Reason</b>", styles["NovacBody"])
        ]
    ]

    for item in possible_values[:20]:
        rows.append([
            Paragraph(
                _safe(item.get("type", "").replace("_", " ").title()),
                styles["NovacBody"]
            ),
            Paragraph(
                _safe(item.get("value", "")),
                styles["NovacBody"]
            ),
            Paragraph(
                _safe(item.get("reason", "")),
                styles["NovacBody"]
            )
        ])

    if len(rows) == 1:
        rows.append([
            "-",
            Paragraph(
                "No unanchored pattern values detected.",
                styles["NovacBody"]
            ),
            "-"
        ])

    return _table(
        rows,
        col_widths=[130, 170, 200]
    )


@router.get("/report/{case_id}")
def generate_report(case_id: str):

    result = get_result_by_case_id(case_id)

    if not result:
        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    pdf_path = os.path.join(
        REPORT_DIR,
        f"report_{case_id}.pdf"
    )

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=34,
        bottomMargin=34
    )

    base_styles = getSampleStyleSheet()
    body_font = _register_unicode_font()

    styles = {
        **base_styles.byName,
        "NovacTitle": ParagraphStyle(
            "NovacTitle",
            parent=base_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=colors.HexColor("#0f172a"),
            alignment=TA_CENTER,
            spaceAfter=6
        ),
        "NovacSubtitle": ParagraphStyle(
            "NovacSubtitle",
            parent=base_styles["BodyText"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#475569"),
            alignment=TA_CENTER,
            spaceAfter=12
        ),
        "NovacSection": ParagraphStyle(
            "NovacSection",
            parent=base_styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=4,
            spaceAfter=4
        ),
        "NovacBody": ParagraphStyle(
            "NovacBody",
            parent=base_styles["BodyText"],
            fontName=body_font,
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#1f2937")
        ),
        "NovacSmall": ParagraphStyle(
            "NovacSmall",
            parent=base_styles["BodyText"],
            fontName=body_font,
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#64748b")
        )
    }

    content = []

    fraud_analysis = result.get("fraud_analysis", {})
    metadata_analysis = result.get("metadata_analysis", {})
    tampering_analysis = result.get("tampering_analysis", {})
    ela_analysis = result.get("ela_analysis", {})
    correlation_analysis = result.get("correlation_analysis", {})
    masking_analysis = result.get("masking_analysis", {})
    condition_analysis = result.get("document_condition_analysis", {})
    photo_analysis = result.get("photo_replacement_analysis", {})
    forgery_analysis = result.get("forgery_localization_analysis", {})
    text_consistency_analysis = result.get("text_consistency_analysis", {})
    consistency_analysis = result.get("visual_consistency_analysis", {})
    field_analysis = result.get("field_extraction_analysis", {})
    preprocessing_analysis = (
        result.get("mvss_preprocess_analysis", {})
        or result.get("preprocessing_analysis", {})
    )

    fraud_score = fraud_analysis.get("fraud_score", 0)
    risk_level = fraud_analysis.get("risk_level", "Unknown")
    reasons = fraud_analysis.get("reasons", [])
    components = fraud_analysis.get("components", {})
    escalations = fraud_analysis.get("escalations", [])

    content.append(
        Paragraph(
            "NOVAC Fraud Analysis Report",
            styles["NovacTitle"]
        )
    )

    content.append(
        Paragraph(
            "Document integrity, tampering, and visual consistency review",
            styles["NovacSubtitle"]
        )
    )

    risk = _risk_color(
        risk_level
    )

    summary = Table(
        [
            [
                Paragraph("<b>Case ID</b>", styles["NovacBody"]),
                Paragraph(_safe(result.get("case_id", "N/A")), styles["NovacBody"]),
                Paragraph("<b>Risk Level</b>", styles["NovacBody"]),
                Paragraph(f"<b>{_safe(risk_level)}</b>", styles["NovacBody"])
            ],
            [
                Paragraph("<b>Filename</b>", styles["NovacBody"]),
                Paragraph(_safe(result.get("filename", "N/A")), styles["NovacBody"]),
                Paragraph("<b>Fraud Score</b>", styles["NovacBody"]),
                Paragraph(f"<b>{_safe(fraud_score)}/100</b>", styles["NovacBody"])
            ],
            [
                Paragraph("<b>Status</b>", styles["NovacBody"]),
                Paragraph(_safe(result.get("status", "N/A")), styles["NovacBody"]),
                Paragraph("<b>OCR Confidence</b>", styles["NovacBody"]),
                Paragraph(_safe(result.get("avg_confidence", "N/A")), styles["NovacBody"])
            ],
        ],
        colWidths=[90, 190, 100, 130],
        hAlign="LEFT"
    )

    summary.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
            ("BACKGROUND", (3, 0), (3, 0), risk),
            ("TEXTCOLOR", (3, 0), (3, 0), colors.white),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])
    )

    content.append(summary)

    _section(
        content,
        "Key Findings",
        styles
    )

    content.append(
        _reasons_block(
            reasons,
            styles
        )
    )

    if escalations:
        _section(
            content,
            "Critical Escalations",
            styles
        )

        content.append(
            _reasons_block(
                escalations,
                styles
            )
        )

    _section(
        content,
        "Score Breakdown",
        styles
    )

    component_rows = [
        ["Signal", "Contribution"]
    ]

    for key, value in components.items():
        component_rows.append([
            key.replace("_", " ").title(),
            _score(value)
        ])

    content.append(
        _table(
            component_rows,
            col_widths=[300, 120]
        )
    )

    content.append(PageBreak())

    _section(
        content,
        "Document Evidence",
        styles
    )

    image_cells = []
    caption_cells = []

    try:
        if result.get("analysis_image_path"):
            image_cells.append(
                Image(
                    result["analysis_image_path"],
                    width=235,
                    height=300
                )
            )
            caption_cells.append(
                Paragraph("Original / analyzed image", styles["NovacSmall"])
            )

        if result.get("annotated_image_path"):
            image_cells.append(
                Image(
                    result["annotated_image_path"],
                    width=235,
                    height=300
                )
            )
            caption_cells.append(
                Paragraph("Annotated evidence regions", styles["NovacSmall"])
            )

        if image_cells:
            image_table = Table(
                [image_cells, caption_cells],
                hAlign="CENTER"
            )
            image_table.setStyle(
                TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ])
            )
            content.append(image_table)
        else:
            content.append(
                Paragraph(
                    "No document images available.",
                    styles["NovacBody"]
                )
            )

    except Exception:
        content.append(
            Paragraph(
                "Unable to load document images.",
                styles["NovacBody"]
            )
        )

    _section(
        content,
        "Detector Summary",
        styles
    )

    detector_rows = [
        ["Detector", "Primary Metric", "Result"],
        ["Metadata", f"Risk {_score(metadata_analysis.get('risk_score', 0))}", ", ".join(metadata_analysis.get("flags", [])) or "No metadata flags"],
        ["OCR / Masking", f"Confidence {_safe(result.get('avg_confidence', 'N/A'))}", "Masked fields detected" if masking_analysis.get("masking_detected") else "No masked fields detected"],
        ["ELA", f"Score {_score(ela_analysis.get('ela_score', 0))}", f"{len(ela_analysis.get('suspicious_regions', []))} suspicious region(s)"],
        ["Visual Tampering / MVSS", f"Score {_score(tampering_analysis.get('tampering_score', 0))}", f"{tampering_analysis.get('valid_suspicious_region_count', tampering_analysis.get('suspicious_region_count', 0))} valid region(s), {tampering_analysis.get('suppressed_region_count', 0)} suppressed"],
        ["MVSS Preprocess", preprocessing_analysis.get("method", "none"), f"{preprocessing_analysis.get('removed_region_count', len(preprocessing_analysis.get('removed_regions', preprocessing_analysis.get('qr_regions', []))))} QR-like region(s) removed" if preprocessing_analysis.get("qr_removed") else "No QR region removed before MVSS"],
        ["Physical Condition", f"Score {_score(condition_analysis.get('condition_score', 0))}", f"Confidence: {condition_analysis.get('condition_confidence', 'low')}. " + ("; ".join(condition_analysis.get("reasons", [])[:2]) or "No major fold/tear indicators")],
        ["Photo Replacement", f"Score {_score(photo_analysis.get('replacement_score', 0))}", ("Synthetic photo suspected; " if photo_analysis.get("ai_photo_suspected") else "") + ("Printed photo likely; " if photo_analysis.get("printed_photo_likely") else "") + ("; ".join(photo_analysis.get("reasons", [])[:2]) or "No photo replacement indicators")],
        ["Forgery Localization", f"Score {_score(forgery_analysis.get('forgery_score', 0))}", "Forgery localization model unavailable" if forgery_analysis.get("model_available") is False else ("Possible manipulated region detected" if forgery_analysis.get("manipulation_detected") else "No strong forgery localization signal")],
        ["Field Text Consistency", f"Score {_score(text_consistency_analysis.get('field_mismatch_score', 0))}", ("; ".join(text_consistency_analysis.get("reasons", [])[:2]) if text_consistency_analysis.get("font_mismatch_detected") else "No strong field-level text mismatch detected")],
        ["Visual Consistency", f"Score {_score(consistency_analysis.get('consistency_score', 0))}", "; ".join(consistency_analysis.get("reasons", [])[:2]) or "No major region inconsistency"],
        ["Correlation", f"{correlation_analysis.get('suspicious_field_count', 0)} field(s)", "OCR fields overlapping visual evidence" if correlation_analysis.get("suspicious_field_count", 0) else "No suspicious OCR-field overlap"],
    ]

    content.append(
        _table(
            detector_rows,
            col_widths=[120, 110, 280]
        )
    )

    content.append(PageBreak())

    _section(
        content,
        "Extracted Text",
        styles
    )

    extracted_text = result.get(
        "text",
        "No text extracted."
    )

    ocr_lines = result.get(
        "lines",
        []
    )

    content.append(
        _table(
            [
                ["Metric", "Value"],
                ["OCR Confidence", _safe(result.get("avg_confidence", "N/A"))],
                ["Confirmed Fields", str(field_analysis.get("field_count", 0))],
                ["Possible Values", str(field_analysis.get("possible_value_count", 0))],
                ["Lines Extracted", str(len(ocr_lines))],
                ["Character Count", str(len(extracted_text))]
            ],
            col_widths=[160, 260]
        )
    )

    content.append(
        Spacer(1, 10)
    )

    content.append(
        Paragraph(
            "<b>Confirmed Structured Fields</b>",
            styles["NovacBody"]
        )
    )

    content.append(
        Spacer(1, 6)
    )

    content.append(
        _field_table(
            field_analysis,
            styles
        )
    )

    content.append(
        Spacer(1, 12)
    )

    content.append(
        Paragraph(
            "<b>Possible Detected Values</b>",
            styles["NovacBody"]
        )
    )

    content.append(
        Spacer(1, 6)
    )

    content.append(
        _possible_values_table(
            field_analysis,
            styles
        )
    )

    content.append(
        Spacer(1, 12)
    )

    content.append(
        Paragraph(
            "<b>Raw OCR Lines</b>",
            styles["NovacBody"]
        )
    )

    content.append(
        Spacer(1, 6)
    )

    content.append(
        _ocr_lines_table(
            ocr_lines,
            styles
        )
    )

    content.append(
        Spacer(1, 10)
    )

    content.append(
        Paragraph(
            "<b>Combined Extracted Text</b>",
            styles["NovacBody"]
        )
    )

    text_panel = Table(
        [[
            Paragraph(
                _safe(extracted_text[:3000]).replace(
                    "\n",
                    "<br/>"
                ),
                styles["NovacBody"]
            )
        ]],
        colWidths=[500],
        hAlign="LEFT"
    )

    text_panel.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ])
    )

    content.append(
        text_panel
    )

    _section(
        content,
        "Final Verdict",
        styles
    )

    if fraud_score >= 75:

        verdict = (
            "Critical risk: multiple document integrity signals indicate a "
            "high probability of tampering or digital manipulation. Manual "
            "verification is strongly recommended."
        )

    elif fraud_score >= 50:

        verdict = (
            "High risk: several suspicious indicators were detected. The "
            "document should be reviewed before acceptance."
        )

    elif fraud_score >= 25:

        verdict = (
            "Medium risk: limited or moderate indicators were detected. "
            "Additional review is recommended if the document is high value."
        )

    else:

        verdict = (
            "Low risk: no significant fraud indicators were detected by the "
            "available analysis modules."
        )

    content.append(
        Paragraph(
            verdict,
            styles["NovacBody"]
        )
    )

    doc.build(content)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{case_id}.pdf"
    )
