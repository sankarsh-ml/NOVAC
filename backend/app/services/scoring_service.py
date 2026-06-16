from app.services.visual_region_utils import (
    any_region_near,
    editable_regions,
    eligible_regions,
    meaningful_regions,
    normalize_score,
    regions_near
)


def _cap(value, maximum):

    return min(
        max(float(value or 0), 0),
        maximum
    )


def _risk_level(score):

    if score < 25:
        return "Low"

    if score < 50:
        return "Medium"

    if score < 75:
        return "High"

    return "Critical"


def _fraud_risk_level(score):

    if score >= 75:
        return "Critical"

    if score >= 50:
        return "High Risk"

    if score >= 25:
        return "Medium Risk"

    return "Low Risk"


def _add_reason(reasons, reason):

    if reason and reason not in reasons:
        reasons.append(reason)


def _regions_near_any(regions_a, regions_b, threshold=0.12):

    for region_a in regions_a or []:
        for region_b in regions_b or []:
            if regions_near(
                region_a,
                region_b,
                iou_threshold=threshold
            ):
                return True

    return False


def _editable_field_name(regions):

    for region in regions or []:
        if region.get("overlaps_editable_field"):
            return region.get("editable_field_name") or "field"

    return None


def _normal_region_count(regions):

    count = 0

    for region in regions or []:
        if region.get("suppression_reason"):
            count += 1

    return count


def _contribution(raw_score=0, normalized_score=None, contribution=0, reason=""):

    result = {
        "raw_score": raw_score,
        "contribution": round(float(contribution or 0), 2),
        "reason": reason
    }

    if normalized_score is not None:
        result["normalized_score"] = round(
            float(normalized_score or 0),
            2
        )

    return result


def _trufor_component(result, mvss_regions, reasons, active_groups):

    result = result or {}
    raw_score = result.get("forgery_score", 0)
    normalized = normalize_score(raw_score)
    confidence = normalize_score(result.get("confidence", 0))
    regions = result.get("suspicious_regions", []) or []
    meaningful = meaningful_regions(regions)
    editable = editable_regions(regions)
    near_mvss = _regions_near_any(
        meaningful,
        mvss_regions
    )

    if result.get("skipped"):
        reason = result.get("skip_reason") or "Skipped due to decisive early signal"
        return 0, {
            **_contribution(
                raw_score,
                normalized,
                0,
                f"TruFor skipped: {reason}"
            ),
            "confidence": confidence,
            "meaningful_region_count": 0,
            "downweighted_region_count": 0,
            "skipped": True,
            "cancelled": bool(result.get("cancelled")),
            "status": result.get("status")
        }

    if not result.get("model_available"):
        return 0, {
            **_contribution(
                raw_score,
                normalized,
                0,
                result.get("model_error") or "TruFor unavailable"
            ),
            "confidence": confidence,
            "meaningful_region_count": 0,
            "downweighted_region_count": _normal_region_count(regions)
        }

    if not result.get("manipulation_detected") or normalized < 20:
        return 0, {
            **_contribution(
                raw_score,
                normalized,
                0,
                "No strong TruFor forgery localization signal"
            ),
            "confidence": confidence,
            "meaningful_region_count": len(meaningful),
            "downweighted_region_count": _normal_region_count(regions)
        }

    if not meaningful:
        contribution = 5 if normalized >= 75 else 2 if normalized >= 45 else 0
        reason = "TruFor detected a region, but it overlapped normal/damaged document structure and was downweighted"

        if contribution:
            _add_reason(reasons, reason)

        return contribution, {
            **_contribution(
                raw_score,
                normalized,
                contribution,
                reason
            ),
            "confidence": confidence,
            "meaningful_region_count": 0,
            "downweighted_region_count": _normal_region_count(regions)
        }

    field_name = _editable_field_name(editable)

    if field_name and near_mvss:
        contribution = 30
        reason = f"Suspicious region overlaps editable field: {field_name}"
    elif field_name:
        contribution = 23
        reason = f"TruFor detected possible manipulation on editable field: {field_name}"
    elif near_mvss:
        contribution = 20 if normalized >= 55 else 14
        reason = "TruFor and MVSS independently detected suspicious visual manipulation"
    elif normalized >= 80:
        contribution = 18
        reason = "TruFor detected a meaningful suspicious manipulation region"
    elif normalized >= 45:
        contribution = 12
        reason = "TruFor detected a localized suspicious manipulation signal"
    else:
        contribution = 4
        reason = "Weak isolated TruFor localization signal"

    contribution = min(contribution, 32)

    if contribution:
        active_groups.append("forgery")
        _add_reason(reasons, reason)

    return contribution, {
        **_contribution(
            raw_score,
            normalized,
            contribution,
            reason
        ),
        "confidence": confidence,
        "meaningful_region_count": len(meaningful),
        "editable_region_count": len(editable),
        "downweighted_region_count": _normal_region_count(regions)
    }


def _mvss_component(result, trufor_regions, trufor_active, reasons, active_groups):

    result = result or {}
    regions = eligible_regions(
        result.get("suspicious_regions", []),
        "scoring_eligible"
    )
    meaningful = meaningful_regions(regions)
    editable = editable_regions(meaningful)
    raw_score = float(result.get("tampering_score", 0) or 0)
    normalized = normalize_score(raw_score)
    near_trufor = (
        trufor_active
        and _regions_near_any(
            meaningful,
            meaningful_regions(trufor_regions)
        )
    )

    if result.get("skipped"):
        reason = result.get("skip_reason") or "Skipped due to decisive early signal"
        return 0, {
            **_contribution(
                raw_score,
                normalized,
                0,
                f"MVSS skipped: {reason}"
            ),
            "raw_region_count": 0,
            "eligible_region_count": 0,
            "downweighted_region_count": 0,
            "skipped": True,
            "cancelled": bool(result.get("cancelled")),
            "status": result.get("status")
        }

    if not meaningful:
        reason = (
            "Raw MVSS regions were suppressed as small/noisy/normal/damage regions"
            if result.get("raw_region_count", 0)
            else "No scoring-eligible MVSS tampering region"
        )
        return 0, {
            **_contribution(
                raw_score,
                normalized,
                0,
                reason
            ),
            "raw_region_count": result.get("raw_region_count", 0),
            "eligible_region_count": 0,
            "downweighted_region_count": _normal_region_count(result.get("suppressed_regions", []))
        }

    field_name = _editable_field_name(editable)

    if field_name and near_trufor:
        contribution = 22
        reason = f"MVSS agrees near editable field: {field_name}"
    elif field_name:
        contribution = 15
        reason = f"MVSS detected suspicious visual manipulation on editable field: {field_name}"
    elif near_trufor:
        contribution = 18
        reason = "MVSS and TruFor overlap or are nearby"
    elif raw_score >= 20:
        contribution = 10
        reason = "MVSS detected scoring-eligible suspicious visual manipulation regions"
    else:
        contribution = 5
        reason = "Weak isolated MVSS visual manipulation signal"

    contribution = min(contribution, 25)

    if contribution:
        active_groups.append("mvss")
        _add_reason(reasons, reason)

    return contribution, {
        **_contribution(
            raw_score,
            normalized,
            contribution,
            reason
        ),
        "raw_region_count": result.get("raw_region_count", 0),
        "eligible_region_count": len(meaningful),
        "editable_region_count": len(editable),
        "downweighted_region_count": _normal_region_count(result.get("suppressed_regions", []))
    }


def _text_component(result, visual_regions, reasons, active_groups):

    result = result or {}

    if not result.get("font_mismatch_detected"):
        return 0, _contribution(
            result.get("field_mismatch_score", 0),
            normalize_score(result.get("field_mismatch_score", 0)),
            0,
            "No strong local field-level text mismatch detected"
        )

    raw = result.get("field_mismatch_score", 0)
    normalized = normalize_score(raw)
    comparisons = int(result.get("comparisons_used", 0) or 0)
    regions = meaningful_regions(result.get("suspicious_regions", []))

    if comparisons < 2 or not regions:
        return 0, _contribution(
            raw,
            normalized,
            0,
            "Text consistency had no reliable local reference"
        )

    near_visual = any(
        any_region_near(
            region,
            visual_regions
        )
        for region in regions
    )
    editable = editable_regions(regions)

    if editable and near_visual and normalized >= 45:
        contribution = 10
        reason = "Possible local field text style mismatch is supported by visual evidence"
    elif editable and normalized >= 35:
        contribution = 6
        reason = "Possible local field text style mismatch detected"
    elif near_visual and normalized >= 40:
        contribution = 5
        reason = "Text mismatch is near a visual detector signal"
    elif normalized >= 45:
        contribution = 3
        reason = "Weak isolated field text consistency signal"
    else:
        contribution = 0
        reason = "No strong local field-level text mismatch detected"

    if contribution:
        active_groups.append("text_consistency")
        _add_reason(reasons, reason)

    return contribution, _contribution(
        raw,
        normalized,
        contribution,
        reason
    )


def _ela_component(result, visual_regions, reasons, active_groups, document_type):

    result = result or {}
    raw = result.get("ela_score", 0)
    normalized = normalize_score(raw)
    regions = meaningful_regions(result.get("suspicious_regions", []))
    supported = _regions_near_any(
        regions,
        visual_regions
    )

    if document_type == "pdf":
        normalized *= 0.65

    if not regions:
        contribution = 0
        reason = "No strong compression consistency signal"
    elif not supported:
        contribution = 2 if normalized >= 70 else 0
        reason = "ELA signal is isolated and used only as weak supporting evidence"
    elif normalized >= 70:
        contribution = 8
        reason = "Supporting compression consistency signal"
    elif normalized >= 45:
        contribution = 5
        reason = "Localized ELA signal supports another detector"
    else:
        contribution = 2
        reason = "Weak supporting ELA signal"

    contribution = min(contribution, 10)

    if contribution >= 4:
        active_groups.append("ela")
        _add_reason(reasons, reason)

    return contribution, _contribution(
        raw,
        normalized,
        contribution,
        reason
    )


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
    visual_consistency_result: dict = None,
    document_quality_result: dict = None,
    document_authenticity_result: dict = None
) -> dict:

    reasons = []
    components = {}
    detector_contributions = {}
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
    document_quality_result = document_quality_result or {}
    document_authenticity_result = document_authenticity_result or {}

    metadata_score = _cap(
        metadata_result.get("risk_score", 0) * 0.55,
        8
    )
    components["metadata"] = round(metadata_score, 2)

    if metadata_score >= 6:
        active_groups.append("metadata")
        for flag in metadata_result.get("flags", [])[:3]:
            _add_reason(reasons, flag)

    avg_confidence = float(
        ocr_result.get("avg_confidence", 1.0)
        or 0
    )
    ocr_score = 0

    if avg_confidence < 0.86 and not document_quality_result.get("rejection_recommended"):
        ocr_score += _cap((0.86 - avg_confidence) * 35, 8)
        _add_reason(
            reasons,
            f"Low OCR confidence ({avg_confidence:.3f})"
        )

    if masking_detected:
        ocr_score += 18
        _add_reason(
            reasons,
            "Masked ID or document fields detected"
        )

    field_warnings = (
        correlation_result.get("field_warnings", [])
        or []
    )

    if field_warnings:
        ocr_score += min(len(field_warnings) * 4, 8)

    ocr_score = _cap(ocr_score, 24)
    components["ocr_and_masking"] = round(ocr_score, 2)

    if ocr_score >= 8:
        active_groups.append("ocr")

    forgery_regions = forgery_localization_result.get("suspicious_regions", [])
    mvss_regions = eligible_regions(
        tampering_result.get("suspicious_regions", []),
        "scoring_eligible"
    )
    text_regions = text_consistency_result.get("suspicious_regions", [])
    ela_regions = ela_result.get("suspicious_regions", [])
    visual_fraud_regions = (
        meaningful_regions(forgery_regions)
        + meaningful_regions(mvss_regions)
    )

    forgery_score, detector_contributions["trufor"] = _trufor_component(
        forgery_localization_result,
        meaningful_regions(mvss_regions),
        reasons,
        active_groups
    )
    components["forgery_localization"] = round(forgery_score, 2)

    mvss_score, detector_contributions["mvss"] = _mvss_component(
        tampering_result,
        forgery_regions,
        bool(forgery_score),
        reasons,
        active_groups
    )
    components["mvss"] = round(mvss_score, 2)

    text_score, detector_contributions["text_consistency"] = _text_component(
        text_consistency_result,
        visual_fraud_regions,
        reasons,
        active_groups
    )
    components["text_consistency"] = round(text_score, 2)

    ela_score, detector_contributions["ela"] = _ela_component(
        ela_result,
        visual_fraud_regions + meaningful_regions(text_regions),
        reasons,
        active_groups,
        type
    )
    components["ela"] = round(ela_score, 2)

    consistency_raw = float(
        visual_consistency_result.get("consistency_score", 0)
        or 0
    )
    consistency_score = 0

    if visual_fraud_regions:
        consistency_score = _cap(consistency_raw * 0.04, 4)

    components["visual_consistency"] = round(consistency_score, 2)

    if consistency_score >= 3:
        active_groups.append("visual_consistency")
        for reason in visual_consistency_result.get("reasons", [])[:1]:
            _add_reason(reasons, reason)

    condition_score = 0
    components["document_condition"] = round(condition_score, 2)

    photo_score = 0
    if photo_replacement_result.get("critical_photo_issue"):
        photo_score = _cap(
            float(photo_replacement_result.get("replacement_score", 0) or 0) * 0.16,
            10
        )
        active_groups.append("photo")
        for reason in photo_replacement_result.get("reasons", [])[:2]:
            _add_reason(reasons, reason)

    components["photo_integrity"] = round(photo_score, 2)

    suspicious_fields = correlation_result.get(
        "suspicious_fields",
        []
    )
    correlation_score = _cap(len(suspicious_fields) * 4, 8)
    components["correlation"] = round(correlation_score, 2)

    if correlation_score >= 5:
        active_groups.append("correlation")
        _add_reason(
            reasons,
            "OCR fields overlap visual evidence"
        )

    synergy = 0
    synergy_reasons = []
    meaningful_forgery = meaningful_regions(forgery_regions)
    meaningful_mvss = meaningful_regions(mvss_regions)
    meaningful_text = meaningful_regions(text_regions)
    meaningful_ela = meaningful_regions(ela_regions)
    editable_visual = editable_regions(
        meaningful_forgery + meaningful_mvss + meaningful_text
    )

    if forgery_score and mvss_score and _regions_near_any(meaningful_forgery, meaningful_mvss):
        if editable_visual:
            synergy += 15
            field_name = _editable_field_name(editable_visual)
            synergy_reasons.append(
                f"Visual detector agreement increased risk score near editable field: {field_name}"
            )
        else:
            synergy += 8
            synergy_reasons.append(
                "Visual detector agreement increased risk score"
            )

    if forgery_score and text_score and _regions_near_any(meaningful_forgery, meaningful_text):
        synergy += 5
        synergy_reasons.append(
            "TruFor signal is near a local field text mismatch"
        )

    if mvss_score and text_score and _regions_near_any(meaningful_mvss, meaningful_text):
        synergy += 4
        synergy_reasons.append(
            "MVSS signal is near a local field text mismatch"
        )

    if (
        (forgery_score or mvss_score)
        and ela_score
        and _regions_near_any(meaningful_forgery + meaningful_mvss, meaningful_ela)
    ):
        synergy += 3
        synergy_reasons.append(
            "ELA / compression consistency supports a nearby visual detector"
        )

    if ocr_score >= 10 and (forgery_score or mvss_score or text_score):
        synergy += 4
        synergy_reasons.append(
            "OCR or masking anomaly is supported by visual detector evidence"
        )

    detector_agreement = min(synergy, 25)
    components["detector_agreement"] = detector_agreement
    detector_contributions["detector_agreement"] = {
        "contribution": round(detector_agreement, 2),
        "reason": "; ".join(synergy_reasons) if synergy_reasons else "No meaningful detector agreement"
    }

    for reason in synergy_reasons:
        _add_reason(reasons, reason)

    quality_score = float(
        document_quality_result.get("quality_score", 100)
        if document_quality_result
        else 100
    )
    quality_rejection = bool(
        document_quality_result.get("rejection_recommended", False)
    )
    detector_contributions["document_quality"] = {
        "quality_score": quality_score,
        "damage_score": document_quality_result.get("damage_score", 0),
        "rejection_recommended": quality_rejection,
        "contribution": 0,
        "reason": (
            "Document condition prevents reliable automated verification"
            if quality_rejection
            else "Document quality did not require rejection"
        )
    }
    components["document_quality"] = 0

    synthetic_score_raw = float(
        document_authenticity_result.get("synthetic_score", 0)
        or document_authenticity_result.get("ai_generated_score", 0)
        or 0
    )
    synthetic_detected = bool(
        document_authenticity_result.get("synthetic_detected", False)
    )
    official_digital_pdf = bool(
        document_authenticity_result.get("official_digital_pdf_detected", False)
    )
    authenticity_contribution = 0
    authenticity_reason = "No strong synthetic document signal"

    if official_digital_pdf:
        authenticity_reason = "Official digital PDF structure detected"
    elif synthetic_detected or synthetic_score_raw >= 70:
        authenticity_contribution = _cap(synthetic_score_raw * 0.32, 32)
        authenticity_reason = "Document appears digitally generated or synthetic"
        active_groups.append("document_authenticity")
        _add_reason(reasons, authenticity_reason)
    elif synthetic_score_raw >= 45:
        authenticity_contribution = _cap(synthetic_score_raw * 0.10, 8)
        authenticity_reason = "Weak synthetic document indicators detected"

        if authenticity_contribution >= 5:
            active_groups.append("document_authenticity")
            _add_reason(reasons, authenticity_reason)

    components["document_authenticity"] = round(authenticity_contribution, 2)
    detector_contributions["document_authenticity"] = {
        "raw_score": round(synthetic_score_raw, 2),
        "authenticity_score": document_authenticity_result.get("authenticity_score", 0),
        "synthetic_detected": synthetic_detected,
        "acquisition_type": document_authenticity_result.get("acquisition_type"),
        "official_digital_pdf_detected": official_digital_pdf,
        "contribution": round(authenticity_contribution, 2),
        "reason": authenticity_reason
    }

    if quality_rejection:
        components["detector_agreement"] = 0
        detector_contributions["detector_agreement"] = {
            "contribution": 0,
            "reason": "Detector agreement suppressed because document quality is too poor for reliable analysis"
        }
        reasons = [
            reason
            for reason in reasons
            if reason not in synergy_reasons
        ]

    unique_groups = []

    for group in active_groups:
        if group not in unique_groups:
            unique_groups.append(group)

    score = sum(components.values())

    if len(unique_groups) <= 1:
        score = min(score, 42)
    elif len(unique_groups) == 2:
        score = min(score, 74)

    score = int(min(max(score, 0), 100))
    escalations = []

    if masking_detected and (forgery_score or mvss_score or text_score):
        escalations.append(
            "Masked fields are supported by additional document integrity signals"
        )
        score = max(score, 80)

    if photo_replacement_result.get("critical_photo_issue") and len(unique_groups) >= 2:
        escalations.append(
            "Critical photo integrity issue is supported by other detector evidence"
        )
        score = max(score, 75)

    if synthetic_detected and not official_digital_pdf:
        escalations.append(
            "Synthetic document authenticity signal detected"
        )
        score = max(score, 80)

    if not reasons and not escalations:
        reasons.append(
            "No significant fraud indicators detected"
        )

    for escalation in escalations:
        _add_reason(reasons, escalation)

    quality_status = document_quality_result.get("quality_status")

    if not quality_status:
        if quality_rejection:
            quality_status = "unprocessable"
        elif quality_score < 45:
            quality_status = "bad"
        elif quality_score < 65:
            quality_status = "warning"
        else:
            quality_status = "good"

    physical_damage_score = float(
        document_quality_result.get("physical_damage_score", 0)
        or document_quality_result.get("damage_score", 0)
        or 0
    )
    fold_tear_score = float(
        document_quality_result.get("fold_tear_score", 0)
        or 0
    )
    quality_badge = None
    quality_notice = None

    if quality_status == "unprocessable":
        quality_badge = "Unprocessable Document"
        quality_notice = (
            "The upload is too blurred, cropped, damaged, or unreadable for "
            "reliable automated verification."
        )
    elif max(physical_damage_score, fold_tear_score) >= 65:
        quality_badge = "Unclear Document"
        quality_notice = (
            "The document was analyzed, but visible physical damage, folds, "
            "tears, or wrinkles may reduce confidence in some detector signals."
        )
    elif quality_status in {"warning", "bad"}:
        quality_badge = "Quality Warning"
        quality_notice = (
            "The document was analyzed, but image quality may reduce confidence "
            "in some detector signals."
        )

    authenticity_score = float(
        document_authenticity_result.get("authenticity_score", 100)
        or 0
    )
    synthetic_document = (
        synthetic_detected
        or synthetic_score_raw >= 65
        or authenticity_score <= 40
    )
    poor_quality_document = (
        quality_status in {"bad", "unprocessable"}
        or quality_badge in {"Unclear Document", "Unprocessable Document"}
        or quality_rejection
        or physical_damage_score >= 70
        or float(document_quality_result.get("damage_score", 0) or 0) >= 70
        or float(document_quality_result.get("crease_score", 0) or 0) >= 70
    )

    if synthetic_document:
        score = 100
    elif poor_quality_document:
        score = 50

    high_threshold = 50

    if synthetic_document:
        result_status = "synthetic_suspected"
        rejection_reason_type = "authenticity"
        risk_level = "Synthetic Document Suspected"
        status = "synthetic_suspected"
        banner_title = "Synthetic document detected."
        banner_body = "Entire document flagged as synthetic/AI-generated. Region-level annotation is not required."

    elif quality_status == "unprocessable":
        result_status = "unprocessable"
        rejection_reason_type = "quality"
        risk_level = "Analysis Inconclusive"
        status = "unprocessable"
        banner_title = "Document could not be analyzed reliably."
        banner_body = "The upload is too blurred, cropped, damaged, or unreadable for reliable automated verification."

    elif poor_quality_document:
        result_status = "quality_warning"
        rejection_reason_type = None
        risk_level = "Analysis Limited"
        status = "quality_warning"
        banner_title = "Document quality limits reliable analysis."
        banner_body = "Document quality is poor. Region-level forensic annotation is not required."

    elif score >= high_threshold:
        result_status = "fraud_suspected"
        rejection_reason_type = "fraud"
        risk_level = "High Risk"
        status = "fraud_suspected"
        banner_title = "Fraud indicators detected."
        banner_body = "The document contains suspicious evidence that may indicate tampering or fraud."

    elif quality_status in {"warning", "bad"}:
        result_status = "quality_warning"
        rejection_reason_type = None
        risk_level = _fraud_risk_level(score)
        status = "quality_warning"
        banner_title = "No major fraud indicators detected."
        banner_body = "The document was analyzed successfully. Any quality concerns are shown separately and did not override the fraud or authenticity result."

    else:
        result_status = "passed"
        rejection_reason_type = None
        risk_level = _fraud_risk_level(score)
        status = "success"
        banner_title = "No major fraud indicators detected."
        banner_body = "The document passed the available automated checks."

    analysis_confidence = int(
        document_quality_result.get(
            "analysis_confidence",
            0 if quality_status == "unprocessable" else 100
        )
    )
    quality_warning = bool(
        document_quality_result.get(
            "quality_warning",
            quality_status in {"warning", "bad"}
        )
    )
    quality_reliable = bool(
        document_quality_result.get(
            "quality_reliable",
            quality_status != "unprocessable"
        )
    )

    print("\n===== FRAUD SCORE =====")
    print("Score:", score)
    print("Risk Level:", risk_level)
    print("Status:", status)
    print("Components:", components)
    print("Detector Contributions:", detector_contributions)
    print("Evidence Groups:", unique_groups)
    print("Escalations:", escalations)
    print("Reasons:", reasons)

    return {
        "fraud_score": score,
        "risk_level": risk_level,
        "status": status,
        "result_status": result_status,
        "rejection_reason_type": rejection_reason_type,
        "banner_title": banner_title,
        "banner_body": banner_body,
        "quality_status": quality_status,
        "analysis_confidence": analysis_confidence,
        "quality_reliable": quality_reliable,
        "quality_warning": quality_warning,
        "quality_badge": quality_badge,
        "quality_notice": quality_notice,
        "reasons": reasons,
        "components": components,
        "detector_contributions": detector_contributions,
        "evidence_groups": unique_groups,
        "escalations": escalations
    }
