import logging
import time

import cv2


logger = logging.getLogger(__name__)


def build_shared_preprocessing(image_path):
    started_at = time.perf_counter()
    image_bgr = cv2.imread(image_path) if image_path else None

    if image_bgr is None:
        return {
            "image_path": image_path,
            "original_image_bgr": None,
            "original_image_rgb": None,
            "grayscale": None,
            "image_width": 0,
            "image_height": 0,
            "timing_seconds": round(time.perf_counter() - started_at, 3),
            "error": f"Cannot read image: {image_path}"
        }

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = image_bgr.shape[:2]

    return {
        "image_path": image_path,
        "original_image_bgr": image_bgr,
        "original_image_rgb": image_rgb,
        "grayscale": grayscale,
        "resized_for_models": {},
        "normalized_tensor": None,
        "image_width": int(width),
        "image_height": int(height),
        "timing_seconds": round(time.perf_counter() - started_at, 3)
    }
