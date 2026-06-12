import os
import cv2
import logging
import threading
import time
import torch
import numpy as np

from pathlib import Path

# MVSS imports
import sys

MVSS_PATH = r"D:\novac\backend\MVSS-Net"

if MVSS_PATH not in sys.path:
    sys.path.append(MVSS_PATH)

from models.mvssnet import get_mvss
from common.tools import inference_single

logger = logging.getLogger(__name__)

MVSS_MODEL_VERSION = "mvssnet_casia"
MVSS_CHECKPOINT_PATH = r"D:\novac\backend\MVSS-Net\ckpt\mvssnet_casia.pt"
USE_FP16_INFERENCE = os.getenv("USE_FP16_INFERENCE", "false").lower() == "true"
MVSS_DEVICE = os.getenv("MVSS_DEVICE", "cpu").lower()


class TamperingService:

    def __init__(self):

        self.device = None
        self.model = None
        self._model_lock = threading.Lock()

    def _load_model(self):

        logger.info("Loading MVSS model...")
        started_at = time.perf_counter()
        logger.info("MVSS configured for CPU mode")
        logger.info("MVSS torch version: %s", torch.__version__)
        selected_device = "cpu"

        if MVSS_DEVICE == "cuda":
            logger.warning(
                "MVSS_DEVICE=cuda was requested, but MVSS defaults to CPU for compatibility. "
                "Set MVSS_ALLOW_CUDA=true only after validating output parity."
            )

            if os.getenv("MVSS_ALLOW_CUDA", "false").lower() == "true" and torch.cuda.is_available():
                selected_device = "cuda"

        device = torch.device(selected_device)
        logger.info("MVSS selected device: %s", device.type)
        logger.info("MVSS running on %s", device.type)

        if device.type == "cpu":
            thread_count = os.getenv("MVSS_CPU_THREADS")

            if thread_count:
                try:
                    torch.set_num_threads(max(1, int(thread_count)))
                except Exception:
                    logger.exception("Unable to set MVSS_CPU_THREADS=%s", thread_count)

        model = get_mvss(
            backbone="resnet50",
            pretrained_base=True,
            nclass=1,
            sobel=True,
            constrain=True,
            n_input=3
        )

        checkpoint_path = (
            MVSS_CHECKPOINT_PATH
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint,
            strict=True
        )

        model.to(device)

        model.eval()

        if USE_FP16_INFERENCE and device.type == "cuda":
            model.half()

        self.device = device
        logger.info(
            "MVSS model loaded in %.3f seconds",
            time.perf_counter() - started_at
        )

        return model

    def _get_model(self):

        if self.model is not None:
            logger.info("Using cached MVSS model")
            return self.model

        with self._model_lock:
            if self.model is None:
                self.model = self._load_model()
            else:
                logger.info("Using cached MVSS model")

        return self.model

    def analyze(self, image_path, shared_preprocessing=None):

        total_started_at = time.perf_counter()
        timings = {}
        preprocess_started_at = time.perf_counter()
        model = self._get_model()

        image = None

        if shared_preprocessing:
            image = shared_preprocessing.get("original_image_bgr")

        if image is None:
            image = cv2.imread(image_path)

        if image is None:
            raise Exception(
                f"Cannot read image: {image_path}"
            )

        original_h, original_w = image.shape[:2]

        resized = cv2.resize(
            image,
            (512, 512)
        )

        timings["mvss_preprocess_seconds"] = round(
            time.perf_counter() - preprocess_started_at,
            3
        )
        inference_started_at = time.perf_counter()

        with torch.inference_mode():

            mask, confidence = inference_single(
                img=resized,
                model=model,
                th=0.5
            )

        timings["mvss_inference_seconds"] = round(
            time.perf_counter() - inference_started_at,
            3
        )
        postprocess_started_at = time.perf_counter()

        result = self._process_mask(
            mask,
            image_path,
            original_w,
            original_h,
            confidence
        )
        timings["mvss_postprocess_seconds"] = round(
            time.perf_counter() - postprocess_started_at,
            3
        )
        timings["mvss_total_seconds"] = round(
            time.perf_counter() - total_started_at,
            3
        )
        result["timings"] = timings
        result["model_device"] = self.device.type if self.device else "unknown"
        result["model_version"] = MVSS_MODEL_VERSION
        result["cache_hit"] = False
        result["completed"] = True
        result["enabled"] = True
        result["timed_out"] = False
        result["score"] = result.get("tampering_score", 0)
        timings["mvss_cache_hit"] = False
        timings["mvss_timed_out"] = False

        logger.info("MVSS preprocessing took %.3f seconds", timings["mvss_preprocess_seconds"])
        logger.info("MVSS model inference took %.3f seconds", timings["mvss_inference_seconds"])
        logger.info("MVSS postprocessing took %.3f seconds", timings["mvss_postprocess_seconds"])

        return result

    def _process_mask(
        self,
        mask,
        image_path,
        original_w,
        original_h,
        confidence
    ):

        mask = mask.astype(np.uint8)

        mask = cv2.resize(
            mask,
            (original_w, original_h)
        )

        _, thresh = cv2.threshold(
            mask,
            127,
            255,
            cv2.THRESH_BINARY
        )

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)

        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            kernel
        )

        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        suspicious_regions = []
        suppressed_regions = []

        total_area = 0

        image_area = original_w * original_h

        MIN_AREA = max(
            1200,
            image_area * 0.002
        )
        MAX_AREA = image_area * 0.35
        raw_region_count = len(contours)

        for cnt in contours:

            area = cv2.contourArea(cnt)
            area_ratio = area / float(image_area or 1)
            x, y, w, h = cv2.boundingRect(cnt)

            if area < MIN_AREA:
                suppressed_regions.append({
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area": int(area),
                    "area_ratio": round(area_ratio, 5),
                    "source": "MVSS",
                    "scoring_eligible": False,
                    "annotation_eligible": False,
                    "suppression_reason": "Region too small for reliable MVSS evidence",
                    "reason": "Region too small for reliable MVSS evidence"
                })
                continue

            if area > MAX_AREA and confidence < 0.85:
                suppressed_regions.append({
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area": int(area),
                    "area_ratio": round(area_ratio, 5),
                    "source": "MVSS",
                    "scoring_eligible": False,
                    "annotation_eligible": False,
                    "suppression_reason": "Region covers too much of document for reliable MVSS evidence",
                    "reason": "Region covers too much of document for reliable MVSS evidence"
                })
                continue

            if w < 18 or h < 18:
                suppressed_regions.append({
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area": int(area),
                    "area_ratio": round(area_ratio, 5),
                    "source": "MVSS",
                    "scoring_eligible": False,
                    "annotation_eligible": False,
                    "suppression_reason": "Region dimensions too small for reliable MVSS evidence",
                    "reason": "Region dimensions too small for reliable MVSS evidence"
                })
                continue

            aspect = max(w, h) / float(max(min(w, h), 1))

            if aspect > 8 and area_ratio < 0.03:
                suppressed_regions.append({
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area": int(area),
                    "area_ratio": round(area_ratio, 5),
                    "source": "MVSS",
                    "scoring_eligible": False,
                    "annotation_eligible": False,
                    "suppression_reason": "Region is a long thin noise strip",
                    "reason": "Region is a long thin noise strip"
                })
                continue

            suspicious_regions.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": int(area),
                "area_ratio": round(area_ratio, 5),
                "confidence": float(confidence),
                "source": "MVSS",
                "type": "mvss",
                "scoring_eligible": True,
                "annotation_eligible": True,
                "suppression_reason": None,
                "reason": "MVSS detected meaningful suspicious visual manipulation region"
            })

            total_area += area

        # Keep only top 3 largest regions
        suspicious_regions = sorted(
            suspicious_regions,
            key=lambda r: r["area"],
            reverse=True
        )[:3]

        total_area = sum(
            region["area"]
            for region in suspicious_regions
        )

        tampered_percent = (
            total_area / image_area
        ) * 100

        reasons = []

        # Suppress very weak detections
        if confidence < 0.35:

            suspicious_regions = []

            total_area = 0

            tampered_percent = 0

            tampering_score = 0
            reasons.append(
                "MVSS confidence below meaningful visual tampering threshold"
            )

        else:

            if not suspicious_regions:
                tampering_score = 0

            elif confidence >= 0.75 and tampered_percent >= 2:
                tampering_score = 30

            elif confidence >= 0.65 and tampered_percent >= 5:
                tampering_score = 35

            elif confidence >= 0.55 and tampered_percent >= 0.8:
                tampering_score = 20

            elif tampered_percent >= 1.5 or len(suspicious_regions) >= 2:
                tampering_score = 15

            else:
                tampering_score = 10

            if suspicious_regions:
                reasons.extend([
                    "MVSS detected meaningful suspicious visual manipulation region",
                    "Suspicious region passed area and confidence filters"
                ])

        tampering_score = min(
            tampering_score,
            40
        )

        output_dir = (
            r"D:\novac\backend\uploads\tampering"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        mask_path = os.path.join(
            output_dir,
            Path(image_path).stem + "_mask.png"
        )

        cv2.imwrite(
            mask_path,
            thresh
        )

        return {

            "tampering_detected":
                len(suspicious_regions) > 0,

            "tampering_score":
                float(tampering_score),

            "tampered_area_percent":
                round(
                    tampered_percent,
                    2
                ),

            "mask_path":
                mask_path,

            "mvss_confidence":
                float(confidence),

            "raw_region_count":
                raw_region_count,

            "scoring_region_count":
                len(suspicious_regions),

            "annotation_region_count":
                len(suspicious_regions),

            "suspicious_region_count":
                len(suspicious_regions),

            "suspicious_regions":
                suspicious_regions,

            "annotation_regions":
                suspicious_regions,

            "suppressed_regions":
                suppressed_regions,

            "suppressed_region_count":
                len(suppressed_regions),

            "reasons":
                reasons
        }


tampering_service = TamperingService()
