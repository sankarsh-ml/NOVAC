import cv2
import os


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
        "label": "Forgery"
    },
    "text_consistency": {
        "color": (80, 255, 80),
        "label": "Text Mismatch"
    },
    "masking": {
        "color": (255, 0, 0),
        "label": "Masked"
    },
    "unknown": {
        "color": (0, 0, 255),
        "label": "Suspicious"
    }
}


def _draw_region(image, region, img_area):

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
    label = style["label"]

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        color,
        3
    )

    cv2.putText(
        image,
        label,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )


def _draw_masked_region(image, region):

    bbox = region.get("bbox")

    if not bbox or len(bbox) < 3:
        return

    x = int(bbox[0][0])
    y = int(bbox[0][1])
    w = int(bbox[2][0] - bbox[0][0])
    h = int(bbox[2][1] - bbox[0][1])

    if w <= 0 or h <= 0:
        return

    style = REGION_STYLES["masking"]

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        style["color"],
        2
    )

    cv2.putText(
        image,
        style["label"],
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        style["color"],
        2
    )


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

    for region in suspicious_regions or []:
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
