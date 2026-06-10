import cv2
import numpy as np


def _image_metrics(gray, color):

    if gray.size == 0 or color.size == 0:
        return {
            "sharpness": 0.0,
            "noise": 0.0,
            "brightness": 0.0,
            "color_std": 0.0
        }

    blur = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    residual = cv2.absdiff(
        gray,
        blur
    )

    return {
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "noise": float(np.std(residual)),
        "brightness": float(np.mean(gray)),
        "color_std": float(np.mean(np.std(color, axis=(0, 1))))
    }


def _bounded_region(region, width, height):

    x = int(region.get("x", 0))
    y = int(region.get("y", 0))
    w = int(region.get("w", 0))
    h = int(region.get("h", 0))

    x = max(
        0,
        min(x, width - 1)
    )

    y = max(
        0,
        min(y, height - 1)
    )

    w = max(
        1,
        min(w, width - x)
    )

    h = max(
        1,
        min(h, height - y)
    )

    return x, y, w, h


def analyze_visual_consistency(
    image_path,
    region_groups=None
):

    image = cv2.imread(image_path)

    if image is None:
        return {
            "consistency_score": 0,
            "inconsistent_regions": [],
            "reasons": [],
            "error": f"Cannot read image: {image_path}"
        }

    height, width = image.shape[:2]
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    global_metrics = _image_metrics(
        gray,
        image
    )

    region_groups = region_groups or {}
    inconsistent_regions = []
    reasons = []
    score = 0

    for source, regions in region_groups.items():

        for region in regions or []:

            x, y, w, h = _bounded_region(
                region,
                width,
                height
            )

            area_ratio = (w * h) / float(width * height)

            if area_ratio < 0.0004:
                continue

            roi_gray = gray[
                y:y + h,
                x:x + w
            ]

            roi_color = image[
                y:y + h,
                x:x + w
            ]

            metrics = _image_metrics(
                roi_gray,
                roi_color
            )

            noise_delta = abs(
                metrics["noise"]
                - global_metrics["noise"]
            )

            sharpness_ratio = (
                metrics["sharpness"]
                / max(global_metrics["sharpness"], 1.0)
            )

            brightness_delta = abs(
                metrics["brightness"]
                - global_metrics["brightness"]
            )

            region_score = 0
            region_reasons = []

            if noise_delta > 8:
                region_score += 8
                region_reasons.append(
                    "noise mismatch"
                )

            if sharpness_ratio > 2.4 or sharpness_ratio < 0.35:
                region_score += 8
                region_reasons.append(
                    "sharpness mismatch"
                )

            if brightness_delta > 38:
                region_score += 6
                region_reasons.append(
                    "lighting mismatch"
                )

            if region_score:
                score += region_score
                reason = (
                    f"{source} region has "
                    + ", ".join(region_reasons)
                )
                reasons.append(reason)
                inconsistent_regions.append({
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "area": int(w * h),
                    "type": "visual",
                    "source": source,
                    "reason": reason,
                    "metrics": {
                        "noise_delta": round(noise_delta, 2),
                        "sharpness_ratio": round(sharpness_ratio, 2),
                        "brightness_delta": round(brightness_delta, 2)
                    }
                })

    return {
        "consistency_score": int(min(score, 100)),
        "global_metrics": {
            "sharpness": round(global_metrics["sharpness"], 2),
            "noise": round(global_metrics["noise"], 2),
            "brightness": round(global_metrics["brightness"], 2),
            "color_std": round(global_metrics["color_std"], 2)
        },
        "inconsistent_regions": inconsistent_regions[:8],
        "reasons": reasons[:8]
    }
