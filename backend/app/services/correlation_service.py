# correlation_service.py

def has_overlap(box1, box2):
    """
    Check if two bounding boxes overlap.

    box format:
    {
        "x": ...,
        "y": ...,
        "w": ...,
        "h": ...
    }
    """

    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])

    x2 = min(
        box1["x"] + box1["w"],
        box2["x"] + box2["w"]
    )

    y2 = min(
        box1["y"] + box1["h"],
        box2["y"] + box2["h"]
    )

    return x2 > x1 and y2 > y1


def bbox_to_rect(bbox):
    """
    Convert OCR bbox to rectangle.

    OCR bbox format:
    [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """

    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]

    return {
        "x": int(min(xs)),
        "y": int(min(ys)),
        "w": int(max(xs) - min(xs)),
        "h": int(max(ys) - min(ys))
    }


def correlate(
    ocr_result,
    ela_result,
    tampering_result,
    photo_replacement_result=None,
    visual_consistency_result=None
):
    """
    Correlate OCR fields with:
    1. ELA suspicious regions
    2. MVSS suspicious regions
    """

    suspicious_fields = []

    ela_regions = ela_result.get(
        "suspicious_regions",
        []
    )

    mvss_regions = tampering_result.get(
        "suspicious_regions",
        []
    )

    photo_regions = (
        photo_replacement_result
        or {}
    ).get(
        "photo_regions",
        []
    )

    visual_regions = (
        visual_consistency_result
        or {}
    ).get(
        "inconsistent_regions",
        []
    )

    ocr_lines = ocr_result.get(
        "lines",
        []
    )

    for line in ocr_lines:

        text = line.get("text", "")

        bbox = line.get("bbox")

        # Defensive: skip malformed bboxes
        if not bbox:
            continue

        if len(bbox) != 4:
            continue


        field_box = bbox_to_rect(bbox)

        ela_overlap = False
        mvss_overlap = False

        # Check ELA overlap
        for region in ela_regions:

            if has_overlap(
                field_box,
                region
            ):
                ela_overlap = True
                break

        # Check MVSS overlap
        for region in mvss_regions:

            if has_overlap(
                field_box,
                region
            ):
                mvss_overlap = True
                break

        if ela_overlap or mvss_overlap:

            confidence = 0.5
            reason = []

            if ela_overlap:
                confidence += 0.25
                reason.append("ELA overlap")

            if mvss_overlap:
                confidence += 0.25
                reason.append("MVSS overlap")

            suspicious_fields.append(
                {
                    "text": text,
                    "confidence": round(
                        confidence,
                        2
                    ),
                    "reason": ", ".join(reason),
                    "ela_overlap": ela_overlap,
                    "mvss_overlap": mvss_overlap,
                    "ocr_bbox": field_box
                }
            )

    return {
        "suspicious_field_count":
            len(suspicious_fields),

        "suspicious_fields":
            suspicious_fields,

        "photo_region_overlap_count":
            len(
                correlate_photo_regions(
                    photo_regions,
                    ela_regions,
                    mvss_regions,
                    visual_regions
                )
            ),

        "photo_region_overlaps":
            correlate_photo_regions(
                photo_regions,
                ela_regions,
                mvss_regions,
                visual_regions
            )
    }


def correlate_photo_regions(
    photo_regions,
    ela_regions,
    mvss_regions,
    visual_regions
):

    overlaps = []

    for photo in photo_regions or []:

        evidence = []

        for region in ela_regions or []:
            if has_overlap(photo, region):
                evidence.append("ELA")
                break

        for region in mvss_regions or []:
            if has_overlap(photo, region):
                evidence.append("MVSS")
                break

        for region in visual_regions or []:
            if has_overlap(photo, region):
                evidence.append("Visual consistency")
                break

        if evidence:
            overlaps.append({
                "photo_region": photo,
                "overlap_sources": evidence,
                "reason": (
                    "Photo region overlaps "
                    + ", ".join(evidence)
                    + " evidence"
                )
            })

    return overlaps
