def _cap(value, maximum):

    return min(
        max(float(value), 0),
        maximum
    )


def _extend_reasons(reasons, incoming, limit=3):

    for reason in incoming or []:
        if reason and reason not in reasons:
            reasons.append(reason)
        if len(reasons) >= limit:
            break


def _risk_level(score):

    if score < 25:
        return "Low"

    if score < 50:
        return "Medium"

    if score < 75:
        return "High"

    return "Critical"


def calculate_fraud_score(
    metadata_result: dict,
    ocr_result: dict,
    ela_result: dict,
    tampering_result: dict,
    correlation_result: dict = None,
    type: str = "unknown",
    masking_detected: bool = False,
    document_condition_result: dict = None,
    photo_replacement_result: dict = None,
    ai_generated_result: dict = None,
    visual_consistency_result: dict = None
) -> dict:

    reasons = []
    components = {}
    active_groups = []

    metadata_result = metadata_result or {}
    ocr_result = ocr_result or {}
    ela_result = ela_result or {}
    tampering_result = tampering_result or {}
    correlation_result = correlation_result or {}
    document_condition_result = document_condition_result or {}
    photo_replacement_result = photo_replacement_result or {}
    ai_generated_result = ai_generated_result or {}
    visual_consistency_result = visual_consistency_result or {}

    metadata_score = _cap(
        metadata_result.get("risk_score", 0),
        15
    )

    components["metadata"] = round(metadata_score, 2)

    if metadata_score >= 8:
        active_groups.append("metadata")
        _extend_reasons(
            reasons,
            metadata_result.get("flags", []),
            limit=4
        )

    avg_confidence = ocr_result.get(
        "avg_confidence",
        1.0
    )

    ocr_score = 0

    if avg_confidence < 0.90:
        ocr_score += _cap(
            (0.90 - avg_confidence) * 90,
            16
        )
        reasons.append(
            f"Low OCR confidence ({avg_confidence:.3f})"
        )

    if masking_detected:
        ocr_score += 12
        reasons.append(
            "Masked fields detected in OCR"
        )

    ocr_score = _cap(
        ocr_score,
        25
    )

    components["ocr_and_masking"] = round(ocr_score, 2)

    if ocr_score >= 8:
        active_groups.append("ocr")

    ela_score_raw = ela_result.get(
        "ela_score",
        0
    )

    if type == "pdf":
        ela_score = _cap(
            ela_score_raw * 0.08,
            12
        )
    else:
        ela_score = _cap(
            ela_score_raw * 0.20,
            20
        )

    components["ela"] = round(ela_score, 2)

    if ela_score >= 8:
        active_groups.append("ela")
        reasons.append(
            f"Suspicious ELA signal ({ela_score_raw})"
        )

    ela_regions = ela_result.get(
        "suspicious_regions",
        []
    )

    if ela_regions and ela_score >= 5:
        reasons.append(
            f"{len(ela_regions)} ELA suspicious region(s) detected"
        )

    tampered_area = tampering_result.get(
        "tampered_area_percent",
        0
    )

    mvss_confidence = tampering_result.get(
        "mvss_confidence",
        0
    )

    mvss_score = 0

    if tampering_result.get("tampering_detected"):
        mvss_score += _cap(
            tampered_area * 1.5,
            12
        )
        mvss_score += _cap(
            mvss_confidence * 8,
            8
        )

    mvss_score = _cap(
        mvss_score,
        20
    )

    components["mvss"] = round(mvss_score, 2)

    if mvss_score >= 8:
        active_groups.append("mvss")
        reasons.append(
            f"MVSS suspicious area: {tampered_area:.2f}%"
        )

    condition_raw = document_condition_result.get(
        "condition_score",
        0
    )

    condition_confidence = document_condition_result.get(
        "condition_confidence",
        "low"
    )

    if condition_confidence in {"medium", "high"}:
        condition_score = _cap(
            condition_raw * 0.20,
            20
        )
    else:
        condition_score = 0

    components["document_condition"] = round(condition_score, 2)

    if condition_score >= 8 and condition_confidence in {"medium", "high"}:
        active_groups.append("condition")
        _extend_reasons(
            reasons,
            document_condition_result.get("reasons", []),
            limit=len(reasons) + 3
        )

    photo_raw = photo_replacement_result.get(
        "replacement_score",
        0
    )

    photo_score = _cap(
        photo_raw * 0.30,
        30
    )

    components["photo_replacement"] = round(photo_score, 2)

    if photo_score >= 8:
        active_groups.append("photo")
        _extend_reasons(
            reasons,
            photo_replacement_result.get("reasons", []),
            limit=len(reasons) + 3
        )

    ai_raw = ai_generated_result.get(
        "ai_generation_score",
        0
    )

    ai_score = _cap(
        ai_raw * 0.25,
        25
    )

    components["ai_generated"] = round(ai_score, 2)

    if (
        ai_score >= 8
        and ai_generated_result.get("ai_generated_suspected")
    ):
        active_groups.append("ai_generated")
        _extend_reasons(
            reasons,
            ai_generated_result.get("reasons", []),
            limit=len(reasons) + 3
        )

    consistency_raw = visual_consistency_result.get(
        "consistency_score",
        0
    )

    consistency_score = _cap(
        consistency_raw * 0.25,
        25
    )

    components["visual_consistency"] = round(consistency_score, 2)

    if consistency_score >= 8:
        active_groups.append("visual_consistency")
        _extend_reasons(
            reasons,
            visual_consistency_result.get("reasons", []),
            limit=len(reasons) + 3
        )

    suspicious_fields = correlation_result.get(
        "suspicious_fields",
        []
    )

    correlation_score = _cap(
        len(suspicious_fields) * 8,
        25
    )

    components["correlation"] = round(correlation_score, 2)

    if correlation_score >= 8:
        active_groups.append("correlation")

        for field in suspicious_fields[:4]:
            field_text = field.get(
                "text",
                "Unknown Field"
            )
            reasons.append(
                f"Suspicious field overlap: {field_text}"
            )

    unique_groups = []

    for group in active_groups:
        if group not in unique_groups:
            unique_groups.append(group)

    subtotal = sum(
        components.values()
    )

    agreement_bonus = 0

    if len(unique_groups) >= 3:
        agreement_bonus = 10

    elif len(unique_groups) == 2:
        agreement_bonus = 5

    components["agreement_bonus"] = agreement_bonus

    score = subtotal + agreement_bonus

    if len(unique_groups) <= 1:
        score = min(
            score,
            45
        )

    elif len(unique_groups) == 2:
        score = min(
            score,
            70
        )

    score = int(
        min(
            max(score, 0),
            100
        )
    )

    escalations = []

    if masking_detected:
        escalations.append(
            "Masked fields detected; masking is treated as a critical document integrity issue"
        )

    if ai_generated_result.get("strong_ai_generated_signal"):
        escalations.append(
            "Strong full-document AI generation signal detected"
        )

    elif (
        ai_generated_result.get("ai_generated_suspected")
        and ai_generated_result.get("positive_synthetic_evidence_count", 0) >= 1
        and not ai_generated_result.get("printed_document_likely")
        and tampering_result.get("tampering_detected")
    ):
        escalations.append(
            "Positive AI-generation evidence combined with MVSS tampering evidence"
        )

    if photo_replacement_result.get("ai_photo_suspected"):
        escalations.append(
            "AI-generated photo or portrait region suspected"
        )

    if photo_replacement_result.get("critical_photo_issue"):
        escalations.append(
            "Critical photo integrity issue detected"
        )

    photo_overlap_count = correlation_result.get(
        "photo_region_overlap_count",
        0
    )

    if (
        photo_overlap_count > 0
        and photo_replacement_result.get("photo_replacement_detected")
    ):
        escalations.append(
            "Photo region overlaps tampering, ELA, or visual inconsistency evidence"
        )

    if (
        ai_generated_result.get("positive_synthetic_evidence_count", 0) >= 1
        and photo_overlap_count > 0
        and tampering_result.get("tampering_detected")
    ):
        escalations.append(
            "Synthetic-image indicators plus MVSS evidence on a photo region"
        )

    if escalations:
        score = max(
            score,
            85
        )

    if not reasons and not escalations:
        reasons.append(
            "No significant fraud indicators detected"
        )

    for escalation in escalations:
        if escalation not in reasons:
            reasons.insert(
                0,
                escalation
            )

    risk_level = _risk_level(score)

    print("\n===== FRAUD SCORE =====")
    print("Score:", score)
    print("Risk Level:", risk_level)
    print("Components:", components)
    print("Evidence Groups:", unique_groups)
    print("Escalations:", escalations)
    print("Reasons:", reasons)

    return {

        "fraud_score": score,

        "risk_level": risk_level,

        "reasons": reasons,

        "components": components,

        "evidence_groups": unique_groups,

        "escalations": escalations

    }
