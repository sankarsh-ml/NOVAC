import cv2
import os
import numpy as np

def crop_document(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        50,
        150
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image_path

    largest = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(
        largest
    )

    cropped = image[
        y:y+h,
        x:x+w
    ]

    base, ext = os.path.splitext(
        image_path
    )

    output_path = (
        f"{base}_cropped{ext}"
    )

    cv2.imwrite(
        output_path,
        cropped
    )

    return output_path


def _fill_region(image, points):

    pts = np.array(
        points,
        dtype=np.int32
    ).reshape(-1, 2)

    x, y, w, h = cv2.boundingRect(pts)
    pad = max(
        int(min(w, h) * 0.08),
        6
    )

    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, image.shape[1])
    y2 = min(y + h + pad, image.shape[0])

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        thickness=-1
    )

    return {
        "x": int(x1),
        "y": int(y1),
        "w": int(max(x2 - x1, 0)),
        "h": int(max(y2 - y1, 0))
    }


def _overlap_ratio(region, other):

    x1 = max(region["x"], other["x"])
    y1 = max(region["y"], other["y"])
    x2 = min(region["x"] + region["w"], other["x"] + other["w"])
    y2 = min(region["y"] + region["h"], other["y"] + other["h"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    smaller = min(
        region["w"] * region["h"],
        other["w"] * other["h"]
    )

    if smaller <= 0:
        return 0

    return intersection / float(smaller)


def _is_probable_photo_region(image, region):

    x = region["x"]
    y = region["y"]
    w = region["w"]
    h = region["h"]

    roi = image[
        y:y + h,
        x:x + w
    ]

    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    saturation = hsv[:, :, 1]

    return float(np.mean(saturation > 45)) > 0.35


def _qr_like_regions(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    height, width = gray.shape[:2]
    image_area = float(width * height) if width and height else 1.0

    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    closed = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        area_ratio = area / image_area

        if area_ratio < 0.005 or area_ratio > 0.15:
            continue

        aspect = w / float(h or 1)

        if aspect < 0.65 or aspect > 1.45:
            continue

        if min(w, h) < max(28, int(min(width, height) * 0.045)):
            continue

        candidate = {
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h)
        }

        if _is_probable_photo_region(image, candidate):
            continue

        roi = gray[
            y:y + h,
            x:x + w
        ]

        if roi.size == 0:
            continue

        edges = cv2.Canny(
            roi,
            60,
            160
        )
        edge_density = float(
            np.mean(edges > 0)
        )

        small = cv2.resize(
            roi,
            (48, 48),
            interpolation=cv2.INTER_AREA
        )

        _, small_binary = cv2.threshold(
            small,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        horizontal_transitions = np.mean(
            small_binary[:, 1:] != small_binary[:, :-1]
        )
        vertical_transitions = np.mean(
            small_binary[1:, :] != small_binary[:-1, :]
        )
        transition_density = float(
            (horizontal_transitions + vertical_transitions) / 2
        )

        contrast = float(
            np.std(roi)
        )
        dark_ratio = float(
            np.mean(roi < 95)
        )
        bright_ratio = float(
            np.mean(roi > 185)
        )

        if (
            edge_density < 0.08
            or transition_density < 0.12
            or contrast < 38
            or dark_ratio < 0.12
            or bright_ratio < 0.18
        ):
            continue

        regions.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "edge_density": round(edge_density, 3),
            "transition_density": round(transition_density, 3),
            "contrast": round(contrast, 2)
        })

    regions = sorted(
        regions,
        key=lambda item: item["w"] * item["h"],
        reverse=True
    )

    selected = []

    for region in regions:
        overlaps = False

        for existing in selected:
            if _overlap_ratio(region, existing) > 0.45:
                overlaps = True
                break

        if not overlaps:
            selected.append(region)

        if len(selected) >= 3:
            break

    return selected


def remove_qr_code_with_metadata(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return {
            "input_path": image_path,
            "output_path": image_path,
            "preprocessed_image_path": image_path,
            "qr_removed": False,
            "qr_regions": [],
            "removed_regions": [],
            "removed_region_count": 0,
            "method": "error",
            "reasons": [],
            "error": f"Cannot read image: {image_path}"
        }

    detector = cv2.QRCodeDetector()
    removed_regions = []
    method = "none"

    try:
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(image)

        if retval and points is not None:
            for point_set in points:
                removed_regions.append(
                    _fill_region(
                        image,
                        point_set
                    )
                )
            method = "opencv_multi"

    except Exception:
        retval = False

    if not removed_regions:
        retval, points = detector.detect(image)

        if retval and points is not None:
            removed_regions.append(
                _fill_region(
                    image,
                    points
                )
            )
            method = "opencv"

    if not removed_regions:
        fallback_regions = _qr_like_regions(
            image
        )

        for region in fallback_regions:
            x = region["x"]
            y = region["y"]
            w = region["w"]
            h = region["h"]
            removed_regions.append(
                _fill_region(
                    image,
                    [
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]
                    ]
                )
            )

        if removed_regions:
            method = "qr_like_fallback"

    if not removed_regions:
        return {
            "input_path": image_path,
            "output_path": image_path,
            "preprocessed_image_path": image_path,
            "qr_removed": False,
            "qr_regions": [],
            "removed_regions": [],
            "removed_region_count": 0,
            "method": "none",
            "reasons": [
                "No QR or QR-like dense square region detected"
            ]
        }

    base, ext = os.path.splitext(
        image_path
    )

    output_path = f"{base}_noqr{ext}"

    cv2.imwrite(
        output_path,
        image
    )

    return {
        "input_path": image_path,
        "output_path": output_path,
        "preprocessed_image_path": output_path,
        "qr_removed": True,
        "qr_regions": removed_regions,
        "removed_regions": removed_regions,
        "removed_region_count": len(removed_regions),
        "method": method,
        "reasons": [
            "QR-like region removed before MVSS"
        ]
    }


def remove_qr_code(image_path):

    return remove_qr_code_with_metadata(
        image_path
    )["output_path"]
