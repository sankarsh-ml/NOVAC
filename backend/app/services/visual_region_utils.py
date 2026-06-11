import math
import re


EDITABLE_FIELD_PATTERNS = [
    ("name", r"\b(name|s/o|d/o|father|husband)\b|^[A-Za-z .'-]{3,}$"),
    ("dob", r"\b(dob|date of birth|birth|yob|year of birth|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"),
    ("gender", r"\b(gender|sex|male|female|other)\b"),
    ("document_number", r"\b(\d{4}\s+\d{4}\s+\d{4}|aadhaar|uid|id no|number)\b"),
    ("vid", r"\b(vid|\d{4}\s+\d{4}\s+\d{4}\s+\d{4})\b"),
    ("address", r"\b(address|care of|c/o|house|road|street|district|state|pin)\b"),
    ("issue_or_expiry_date", r"\b(issue|expiry|valid|validity)\b"),
    ("signature", r"\b(signature|signed)\b")
]

NORMAL_STRUCTURE_SOURCES = {
    "QR",
    "Logo",
    "HeaderFooter",
    "DenseText",
    "Damage",
    "Photo"
}


def normalize_score(value):

    try:
        if value is None:
            return 0.0

        score = float(value)

    except Exception:
        return 0.0

    if 0 <= score <= 1:
        score *= 100

    return float(min(max(score, 0), 100))


def box_iou(region, other):

    if not region or not other:
        return 0

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


def center_distance(region, other):

    if not region or not other:
        return float("inf")

    ax = region.get("x", 0) + region.get("w", 0) / 2
    ay = region.get("y", 0) + region.get("h", 0) / 2
    bx = other.get("x", 0) + other.get("w", 0) / 2
    by = other.get("y", 0) + other.get("h", 0) / 2

    return math.hypot(ax - bx, ay - by)


def regions_near(region, other, image_shape=None, iou_threshold=0.12):

    if box_iou(region, other) >= iou_threshold:
        return True

    distance = center_distance(region, other)
    scale = max(
        region.get("w", 0),
        region.get("h", 0),
        other.get("w", 0),
        other.get("h", 0),
        1
    )

    if distance <= scale * 1.25:
        return True

    if image_shape:
        height, width = image_shape[:2]
        diagonal = math.hypot(width, height)
        return distance <= diagonal * 0.045

    return False


def any_region_near(region, others, image_shape=None, iou_threshold=0.12):

    return any(
        regions_near(
            region,
            other,
            image_shape=image_shape,
            iou_threshold=iou_threshold
        )
        for other in others or []
    )


def eligible_regions(regions, flag="scoring_eligible"):

    selected = []

    for region in regions or []:
        if flag not in region or region.get(flag):
            selected.append(region)

    return selected


def rect_from_ocr_line(line):

    bbox = (line or {}).get("bbox")

    if not bbox or len(bbox) != 4:
        return None

    try:
        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]
    except Exception:
        return None

    x = int(min(xs))
    y = int(min(ys))
    w = int(max(xs) - min(xs))
    h = int(max(ys) - min(ys))

    if w <= 0 or h <= 0:
        return None

    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h
    }


def _field_name(text):

    normalized = str(text or "").strip().lower()

    if not normalized:
        return None

    for field_name, pattern in EDITABLE_FIELD_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return field_name

    return None


def _overlap_ratio(region, other):

    if not region or not other:
        return 0.0

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
    region_area = max(1, region.get("w", 0) * region.get("h", 0))

    return intersection / float(region_area)


def classify_region_context(
    region,
    source,
    image_shape,
    ocr_lines=None,
    qr_regions=None,
    photo_regions=None,
    damage_regions=None,
    default_type=None
):

    height, width = image_shape[:2]
    image_area = float(width * height) if width and height else 1.0
    w = int(region.get("w", 0) or 0)
    h = int(region.get("h", 0) or 0)
    area = float(region.get("area") or (w * h))
    area_ratio = area / image_area
    classified = {
        **region,
        "source": source,
        "type": default_type or region.get("type"),
        "w": w,
        "h": h,
        "area": int(area),
        "area_ratio": round(area_ratio, 5),
        "overlaps_qr": False,
        "overlaps_photo": False,
        "overlaps_logo": False,
        "overlaps_dense_text": False,
        "overlaps_header_footer": False,
        "overlaps_damage_or_fold": False,
        "overlaps_editable_field": False,
        "editable_field_name": None,
        "scoring_eligible": region.get("scoring_eligible", True),
        "annotation_eligible": region.get("annotation_eligible", True),
        "suppression_reason": region.get("suppression_reason")
    }

    if not width or not height or w <= 0 or h <= 0:
        classified["scoring_eligible"] = False
        classified["annotation_eligible"] = False
        classified["suppression_reason"] = "Invalid region geometry"
        return classified

    if region.get("y", 0) < height * 0.08 or region.get("y", 0) + h > height * 0.94:
        classified["overlaps_header_footer"] = True

    for qr_region in qr_regions or []:
        if box_iou(classified, qr_region) >= 0.10 or _overlap_ratio(classified, qr_region) >= 0.25:
            classified["overlaps_qr"] = True
            break

    for photo_region in photo_regions or []:
        if box_iou(classified, photo_region) >= 0.08 or _overlap_ratio(classified, photo_region) >= 0.25:
            classified["overlaps_photo"] = True
            break

    for damage_region in damage_regions or []:
        if box_iou(classified, damage_region) >= 0.08 or _overlap_ratio(classified, damage_region) >= 0.18:
            classified["overlaps_damage_or_fold"] = True
            break

    overlapping_text = 0
    editable_fields = []

    for line in ocr_lines or []:
        rect = rect_from_ocr_line(line)

        if not rect:
            continue

        if _overlap_ratio(classified, rect) < 0.12 and box_iou(classified, rect) < 0.02:
            continue

        overlapping_text += 1
        field_name = _field_name(line.get("text", ""))

        if field_name:
            editable_fields.append(field_name)

    if overlapping_text >= 4 and not editable_fields:
        classified["overlaps_dense_text"] = True

    if editable_fields:
        classified["overlaps_editable_field"] = True
        classified["editable_field_name"] = editable_fields[0]

    if (
        area_ratio > 0.28
        and not classified["overlaps_editable_field"]
    ):
        classified["overlaps_dense_text"] = True

    if (
        region.get("x", 0) < width * 0.18
        and region.get("y", 0) < height * 0.18
        and area_ratio < 0.025
    ):
        classified["overlaps_logo"] = True

    normal_flags = [
        classified["overlaps_qr"],
        classified["overlaps_photo"],
        classified["overlaps_logo"],
        classified["overlaps_dense_text"],
        classified["overlaps_header_footer"],
        classified["overlaps_damage_or_fold"]
    ]

    if any(normal_flags) and not classified["overlaps_editable_field"]:
        reason = "Visual region overlapped normal document structure and was downweighted"

        if classified["overlaps_damage_or_fold"]:
            reason = "Visual region overlaps document damage or fold"
        elif classified["overlaps_qr"]:
            reason = "Visual region overlaps QR-like document structure"
        elif classified["overlaps_photo"]:
            reason = "Visual region overlaps photo area"
        elif classified["overlaps_header_footer"]:
            reason = "Visual region overlaps header or footer band"
        elif classified["overlaps_dense_text"]:
            reason = "Visual region overlaps dense text or instruction block"

        classified["scoring_eligible"] = False
        classified["suppression_reason"] = classified.get("suppression_reason") or reason

    if classified["suppression_reason"]:
        classified["annotation_eligible"] = False

    return classified


def classify_regions(
    regions,
    source,
    image_shape,
    ocr_lines=None,
    qr_regions=None,
    photo_regions=None,
    damage_regions=None,
    default_type=None
):

    return [
        classify_region_context(
            region,
            source,
            image_shape,
            ocr_lines=ocr_lines,
            qr_regions=qr_regions,
            photo_regions=photo_regions,
            damage_regions=damage_regions,
            default_type=default_type
        )
        for region in regions or []
    ]


def meaningful_regions(regions):

    return [
        region
        for region in regions or []
        if region.get("scoring_eligible", True)
        and not region.get("suppression_reason")
    ]


def editable_regions(regions):

    return [
        region
        for region in meaningful_regions(regions)
        if region.get("overlaps_editable_field")
    ]
