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


def _box_iou(region, other):

    x1 = max(region.get("x", 0), other.get("x", 0))
    y1 = max(region.get("y", 0), other.get("y", 0))
    x2 = min(
        region.get("x", 0) + region.get("w", 0),
        other.get("x", 0) + other.get("w", 0)
    )
    y2 = min(
        region.get("y", 0) + region.get("h", 0),
        other.get("y", 0) + other.get("h", 0)
    )

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, region.get("w", 0)) * max(0, region.get("h", 0))
    area_b = max(0, other.get("w", 0)) * max(0, other.get("h", 0))
    union = area_a + area_b - intersection

    if union <= 0:
        return 0

    return intersection / float(union)


def _regions_overlap(regions_a, regions_b, threshold=0.20):

    for region_a in regions_a or []:
        for region_b in regions_b or []:
            if _box_iou(region_a, region_b) >= threshold:
                return True

    return False


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
    forgery_localization_result: dict = None,
    text_consistency_result: dict = None,
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
    forgery_localization_result = forgery_localization_result or {}
    text_consistency_result = text_consistency_result or {}
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
    mvss_regions = tampering_result.get(
        "suspicious_regions",
        []
    )
    valid_mvss_count = tampering_result.get(
        "valid_suspicious_region_count",
        tampering_result.get(
            "suspicious_region_count",
            len(mvss_regions)
        )
    )

    if tampering_result.get("tampering_detected") and valid_mvss_count > 0:

        if tampered_area < 0.2 or mvss_confidence < 0.35:
            mvss_score = 0

        elif tampered_area < 1:
            mvss_score = 3

        elif tampered_area < 3:
            mvss_score = 6

        elif tampered_area < 8:
            mvss_score = 9

        else:
            mvss_score = 11

        if mvss_confidence >= 0.70 and tampered_area >= 1:
            mvss_score += 1

    mvss_score = _cap(
        mvss_score,
        12
    )

    components["mvss"] = round(mvss_score, 2)

    if mvss_score >= 5:
        active_groups.append("mvss")
        reasons.append(
            "MVSS detected valid suspicious visual region"
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
            condition_raw * 0.05,
            3
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

    mvss_ela_synergy = 0

    if (
        mvss_score >= 5
        and ela_regions
        and ela_score >= 5
    ):
        mvss_ela_synergy = 4
        reasons.append(
            "MVSS signal supported by ELA visual inconsistency"
        )

    components["mvss_ela_support"] = mvss_ela_synergy

    forgery_score_raw = forgery_localization_result.get(
        "forgery_score",
        0
    )
    forgery_confidence = forgery_localization_result.get(
        "confidence",
        0
    )
    forgery_regions = forgery_localization_result.get(
        "suspicious_regions",
        []
    )

    forgery_score = 0

    if (
        forgery_localization_result.get("model_available")
        and forgery_localization_result.get("manipulation_detected")
    ):
        if forgery_confidence >= 0.75 or forgery_score_raw >= 70:
            forgery_score = 12
        elif forgery_confidence >= 0.45 or forgery_score_raw >= 40:
            forgery_score = 7
        else:
            forgery_score = 3

    components["forgery_localization"] = round(
        _cap(
            forgery_score,
            12
        ),
        2
    )

    if forgery_score >= 4:
        active_groups.append("forgery")
        reasons.append(
            "Forgery localization model detected possible manipulated region"
        )

    text_score_raw = text_consistency_result.get(
        "field_mismatch_score",
        0
    )
    text_fields = text_consistency_result.get(
        "suspicious_fields",
        []
    )
    text_regions = text_consistency_result.get(
        "suspicious_regions",
        []
    )

    text_score = 0

    if text_consistency_result.get("font_mismatch_detected"):
        critical_count = len([
            field
            for field in text_fields
            if field.get("field") in {
                "name",
                "dob",
                "aadhaar_number",
                "vid",
                "gender"
            }
        ])

        if critical_count:
            text_score = _cap(
                max(10, text_score_raw * 0.15),
                15
            )
        else:
            text_score = _cap(
                text_score_raw * 0.08,
                6
            )

    components["text_consistency"] = round(text_score, 2)

    if text_score >= 6:
        active_groups.append("text_consistency")
        reasons.append(
            "Text field visual style differs from surrounding document text"
        )

    mvss_forgery_synergy = 0

    if (
        mvss_score >= 5
        and forgery_score >= 4
        and _regions_overlap(mvss_regions, forgery_regions)
    ):
        mvss_forgery_synergy = 5
        reasons.append(
            "MVSS signal overlaps forgery localization model region"
        )

    components["mvss_forgery_support"] = mvss_forgery_synergy

    mvss_text_synergy = 0

    if (
        mvss_score >= 5
        and text_score >= 6
        and _regions_overlap(mvss_regions, text_regions, threshold=0.12)
    ):
        mvss_text_synergy = 5
        reasons.append(
            "MVSS signal supported by nearby text field mismatch"
        )

    components["mvss_text_support"] = mvss_text_synergy

    ocr_text_synergy = 0

    if text_score >= 6 and (
        correlation_result.get("suspicious_field_count", 0) > 0
        or ocr_score >= 8
    ):
        ocr_text_synergy = 5
        reasons.append(
            "Visual tampering signal supported by OCR/correlation anomaly"
        )

    components["ocr_text_support"] = ocr_text_synergy

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

    if photo_replacement_result.get("ai_photo_suspected"):
        escalations.append(
            "Synthetic photo or portrait region suspected"
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
