import cv2
import logging
import os


logger = logging.getLogger(__name__)


REGION_STYLES = {
    "mvss": {
        "color": (0, 0, 255),
        "label": "MVSS"
    },
    "ela": {
        "color": (0, 165, 255),
        "label": "ELA"
    },
    "condition": {
        "color": (0, 255, 255),
        "label": "Damage"
    },
    "photo": {
        "color": (255, 0, 255),
        "label": "Photo"
    },
    "visual": {
        "color": (255, 128, 0),
        "label": "Visual"
    },
    "forgery_model": {
        "color": (0, 80, 255),
        "label": "TruFor"
    },
    "text_consistency": {
        "color": (80, 255, 80),
        "label": "Text Mismatch"
    },
    "quality": {
        "color": (0, 255, 255),
        "label": "Quality Issue"
    },
    "masking": {
        "color": (0, 215, 255),
        "label": "Masked Field"
    },
    "unknown": {
        "color": (0, 0, 255),
        "label": "Suspicious"
    }
}


def _draw_region(image, region, img_area):

    if region.get("suppression_reason"):
        return

    if region.get("type") == "mvss" and not region.get("annotation_eligible", False):
        return

    if region.get("type") == "text_consistency" and not region.get("annotation_eligible", True):
        return

    x = int(region.get("x", 0))
    y = int(region.get("y", 0))
    w = int(region.get("w", 0))
    h = int(region.get("h", 0))

    region_area = float(w * h)

    if region_area <= 0:
        return

    region_type = region.get(
        "type",
        "unknown"
    )

    tiny_threshold = {
        "ela": 0.0005,
        "condition": 0.0005,
        "photo": 0.001,
        "visual": 0.001,
        "forgery_model": 0.001,
        "text_consistency": 0.0002,
        "mvss": 0.005
    }.get(
        region_type,
        0.003
    )

    if region_area / img_area < tiny_threshold:
        return

    style = REGION_STYLES.get(
        region_type,
        REGION_STYLES["unknown"]
    )

    color = style["color"]
    label = region.get("label") or style["label"]

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        color,
        3
    )

    label_y = max(y - 10, 20)

    if hasattr(_draw_region, "_label_positions"):
        while any(
            abs(label_y - existing_y) < 18
            and abs(x - existing_x) < 120
            for existing_x, existing_y in _draw_region._label_positions
        ):
            label_y = min(
                label_y + 18,
                image.shape[0] - 8
            )
        _draw_region._label_positions.append((x, label_y))

    cv2.putText(
        image,
        label,
        (x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )


def _draw_masked_region(image, region):

    geometry = _masked_display_geometry(region, image.shape)

    if not geometry:
        return

    original_region = geometry["original_region"]
    display_region = geometry["display_region"]
    x = display_region["x"]
    y = display_region["y"]
    w = display_region["w"]
    h = display_region["h"]

    if w <= 0 or h <= 0:
        return

    style = REGION_STYLES["masking"]
    label = _masked_label(region.get("label") or style["label"])
    logger.info(
        "Masked field annotation region: %s",
        {
            "label": label,
            "masked_text": region.get("masked_text") or region.get("text"),
            "ocr_region": region.get("region") or region.get("original_region"),
            "original_region": original_region,
            "display_region": display_region,
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "scale_x": geometry["scale_applied"].get("x"),
            "scale_y": geometry["scale_applied"].get("y"),
            "coordinate_source": geometry["coordinate_source"],
            "scale_applied": geometry["scale_applied"],
            "source": "ocr_mask_pattern",
            "source_detector": "masking",
        }
    )
    region["original_region"] = original_region
    region["display_region"] = display_region

    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (x, y),
        (x + w, y + h),
        style["color"],
        -1
    )
    cv2.addWeighted(
        overlay,
        0.18,
        image,
        0.82,
        0,
        image
    )

    thickness = max(4, int(round(max(image.shape[0], image.shape[1]) / 420)))
    cv2.rectangle(
        image,
        (max(0, x - thickness), max(0, y - thickness)),
        (
            min(image.shape[1] - 1, x + w + thickness),
            min(image.shape[0] - 1, y + h + thickness)
        ),
        (255, 255, 255),
        max(1, thickness // 2)
    )
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        style["color"],
        thickness
    )

    font_scale = max(0.95, min(1.55, max(image.shape[0], image.shape[1]) / 1600))
    text_thickness = max(2, int(round(font_scale * 2)))
    (label_w, label_h), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_thickness
    )
    padding_x = max(10, int(label_h * 0.45))
    padding_y = max(7, int(label_h * 0.35))
    label_x = max(0, min(x, image.shape[1] - label_w - (padding_x * 2)))
    label_top = y - label_h - baseline - (padding_y * 2) - 4

    if label_top < 0:
        label_top = y + max(0, min(8, h - label_h - baseline - (padding_y * 2)))

    label_top = max(0, min(label_top, image.shape[0] - label_h - baseline - (padding_y * 2)))
    label_bottom = min(image.shape[0] - 1, label_top + label_h + baseline + (padding_y * 2))
    label_right = min(image.shape[1] - 1, label_x + label_w + (padding_x * 2))
    text_x = label_x + padding_x
    text_y = min(image.shape[0] - baseline - padding_y, label_top + padding_y + label_h)

    cv2.rectangle(
        image,
        (label_x, label_top),
        (label_right, label_bottom),
        style["color"],
        -1
    )
    cv2.rectangle(
        image,
        (label_x, label_top),
        (label_right, label_bottom),
        (0, 0, 0),
        max(1, thickness // 3)
    )
    cv2.putText(
        image,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        text_thickness + 2
    )
    cv2.putText(
        image,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness
    )


def _masked_label(label):
    normalized = str(label or "").strip().lower()

    if "hidden" in normalized and "critical" in normalized:
        return "Hidden Critical Field"

    if "hidden" in normalized:
        return "Hidden Field"

    if "critical" in normalized:
        return "Masked Critical Field"

    return "Masked Field"


def _masked_display_geometry(region, image_shape):
    img_h, img_w = image_shape[:2]
    rect, source, scale = _masked_region_rect(region, img_w, img_h)

    if not rect:
        return None

    original = _clamp_rect(rect, img_w, img_h)
    display = _expanded_display_rect(original, img_w, img_h)

    return {
        "original_region": _rect_to_dict(original),
        "display_region": _rect_to_dict(display),
        "coordinate_source": source,
        "scale_applied": scale
    }


def _rect_to_dict(rect):
    x, y, w, h = rect
    return {
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h)
    }


def _clamp_rect(rect, image_width, image_height):
    x, y, w, h = rect
    x = max(0, min(int(round(x)), image_width - 1))
    y = max(0, min(int(round(y)), image_height - 1))
    w = max(1, min(int(round(w)), image_width - x))
    h = max(1, min(int(round(h)), image_height - y))
    return x, y, w, h


def _expanded_display_rect(rect, image_width, image_height):
    x, y, w, h = rect
    padding_x = int(round(min(16, max(8, image_width * 0.006))))
    padding_y = int(round(min(12, max(6, image_height * 0.006))))

    return _clamp_rect(
        (
            x - padding_x,
            y - padding_y,
            w + (padding_x * 2),
            h + (padding_y * 2)
        ),
        image_width,
        image_height
    )


def _scale_rect_if_needed(rect, region, image_width, image_height):
    x, y, w, h = rect

    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
        return (
            x * image_width,
            y * image_height,
            w * image_width,
            h * image_height
        ), {
            "x": image_width,
            "y": image_height
        }, "normalized"

    source_w = region.get("image_width") or region.get("source_image_width")
    source_h = region.get("image_height") or region.get("source_image_height")

    try:
        source_w = float(source_w)
        source_h = float(source_h)
    except Exception:
        source_w = None
        source_h = None

    if source_w and source_h and (abs(source_w - image_width) > 1 or abs(source_h - image_height) > 1):
        scale_x = image_width / source_w
        scale_y = image_height / source_h
        return (
            x * scale_x,
            y * scale_y,
            w * scale_x,
            h * scale_y
        ), {
            "x": round(scale_x, 6),
            "y": round(scale_y, 6)
        }, "source_image_scaled"

    return rect, {
        "x": 1,
        "y": 1
    }, "image_pixels"


def _masked_region_rect(region, image_width, image_height):
    source = region.get("original_region") or region.get("region") or region.get("display_region") or region

    try:
        if all(source.get(key) is not None for key in ("x", "y", "w", "h")):
            rect = (
                float(source.get("x", 0)),
                float(source.get("y", 0)),
                float(source.get("w", 0)),
                float(source.get("h", 0))
            )
            scaled_rect, scale, scale_source = _scale_rect_if_needed(
                rect,
                region,
                image_width,
                image_height
            )
            return scaled_rect, f"{scale_source}:xywh", scale
    except Exception:
        pass

    bbox = region.get("bbox")

    try:
        if isinstance(bbox, dict):
            rect = (
                float(bbox.get("x", bbox.get("left", 0))),
                float(bbox.get("y", bbox.get("top", 0))),
                float(bbox.get("w", bbox.get("width", 0))),
                float(bbox.get("h", bbox.get("height", 0)))
            )
            scaled_rect, scale, scale_source = _scale_rect_if_needed(
                rect,
                region,
                image_width,
                image_height
            )
            return scaled_rect, f"{scale_source}:bbox_dict", scale

        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            if all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in bbox):
                xs = [float(point[0]) for point in bbox]
                ys = [float(point[1]) for point in bbox]
                rect = (
                    min(xs),
                    min(ys),
                    max(xs) - min(xs),
                    max(ys) - min(ys)
                )
                scaled_rect, scale, scale_source = _scale_rect_if_needed(
                    rect,
                    region,
                    image_width,
                    image_height
                )
                return scaled_rect, f"{scale_source}:bbox_polygon", scale

            x, y, w, h = bbox
            rect = (float(x), float(y), float(w), float(h))
            scaled_rect, scale, scale_source = _scale_rect_if_needed(
                rect,
                region,
                image_width,
                image_height
            )
            return scaled_rect, f"{scale_source}:bbox_xywh", scale

    except Exception:
        logger.debug("Unable to normalize masked annotation bbox", exc_info=True)

    return None, "unknown", {"x": 1, "y": 1}


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


def _dedupe_regions(regions):

    selected = []

    for region in regions or []:
        if any(
            _box_iou(region, existing) >= 0.55
            for existing in selected
        ):
            continue

        selected.append(region)

    return selected


def create_annotated_image(
    image_path,
    suspicious_regions,
    masking_regions,
    filename
):

    image = cv2.imread(image_path)

    if image is None:
        return None

    img_h, img_w = image.shape[:2]
    img_area = float(img_w * img_h) if img_w and img_h else 1.0
    _draw_region._label_positions = []
    has_masked_regions = bool(masking_regions)

    priority = {
        "text_consistency": 0,
        "forgery_model": 1,
        "mvss": 2,
        "ela": 3,
        "condition": 4,
        "photo": 5,
        "visual": 6,
        "quality": 7
    }
    annotation_regions = suspicious_regions or []

    if has_masked_regions:
        low_priority = {"photo", "visual", "condition", "quality"}
        annotation_regions = [
            region
            for region in annotation_regions
            if region.get("type") not in low_priority
        ]

    ordered_regions = sorted(
        annotation_regions,
        key=lambda item: priority.get(item.get("type"), 99)
    )

    for region in _dedupe_regions(ordered_regions):
        _draw_region(
            image,
            region,
            img_area
        )

    for region in masking_regions or []:
        _draw_masked_region(
            image,
            region
        )

    output_path = os.path.join(
        "uploads",
        f"{filename}_annotated.png"
    )

    cv2.imwrite(
        output_path,
        image
    )

    return output_path
