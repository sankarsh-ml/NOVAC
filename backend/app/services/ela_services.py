from PIL import Image, ImageChops, ImageEnhance
import cv2
import numpy as np
import os


BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
ELA_DIR = os.path.join(UPLOADS_DIR, "ela")

os.makedirs(ELA_DIR, exist_ok=True)


def analyze_ela(image_path):

    temp_path = os.path.join(
        ELA_DIR,
        "temp_ela.jpg"
    )

    ela_output = os.path.join(
        ELA_DIR,
        "ela_output.png"
    )

    marked_output = os.path.join(
        ELA_DIR,
        "ela_marked.png"
    )

    # =========================
    # Create ELA Image
    # =========================

    original = Image.open(image_path).convert("RGB")

    original.save(
        temp_path,
        "JPEG",
        quality=90
    )

    compressed = Image.open(temp_path)

    ela_image = ImageChops.difference(
        original,
        compressed
    )

    extrema = ela_image.getextrema()

    max_diff = max(
        channel[1]
        for channel in extrema
    )

    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(
        ela_image
    ).enhance(scale)

    ela_image.save(ela_output)

    # =========================
    # OpenCV Analysis
    # =========================

    ela_cv = cv2.imread(ela_output)

    gray = cv2.cvtColor(
        ela_cv,
        cv2.COLOR_BGR2GRAY
    )

    avg_brightness = float(
        np.mean(gray)
    )

    max_brightness = int(
        np.max(gray)
    )

    _, thresh = cv2.threshold(
        gray,
        50,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    suspicious_regions = []
    total_area = 0

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        area = w * h

        # Ignore tiny noise
        if area < 5000:
            continue

        total_area += area

        suspicious_regions.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(area)
        })

        cv2.rectangle(
            ela_cv,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

    cv2.imwrite(
        marked_output,
        ela_cv
    )

    # =========================
    # Better Scoring
    # =========================

    region_score = min(
        len(suspicious_regions) * 10,
        30
    )

    area_score = min(
        total_area / 5000,
        40
    )

    brightness_score = min(
        avg_brightness / 3,
        30
    )

    ela_score = int(
        region_score +
        area_score +
        brightness_score
    )

    ela_score = min(
        ela_score,
        100
    )

    # =========================
    # Return
    # =========================

    return {

        "ela_score": ela_score,

        "statistics": {
            "average_brightness": round(
                avg_brightness,
                2
            ),
            "max_brightness": max_brightness,
            "suspicious_region_count": len(
                suspicious_regions
            ),
            "total_suspicious_area": int(
                total_area
            )
        },

        "ela_image": ela_output,

        "marked_image": marked_output,

        "suspicious_regions": suspicious_regions
    }