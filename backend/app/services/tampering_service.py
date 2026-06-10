import os
import cv2
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


class TamperingService:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = self._load_model()

    def _load_model(self):

        model = get_mvss(
            backbone="resnet50",
            pretrained_base=True,
            nclass=1,
            sobel=True,
            constrain=True,
            n_input=3
        )

        checkpoint_path = (
            r"D:\novac\backend\MVSS-Net\ckpt\mvssnet_casia.pt"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        model.load_state_dict(
            checkpoint,
            strict=True
        )

        model.to(self.device)

        model.eval()

        return model

    def analyze(self, image_path):

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

        with torch.no_grad():

            mask, confidence = inference_single(
                img=resized,
                model=self.model,
                th=0.5
            )

        return self._process_mask(
            mask,
            image_path,
            original_w,
            original_h,
            confidence
        )

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

        total_area = 0

        image_area = original_w * original_h

        MIN_AREA = max(
            1000,
            image_area * 0.001
        )

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            suspicious_regions.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": int(area)
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

        # Suppress very weak detections
        if confidence < 0.35:

            suspicious_regions = []

            total_area = 0

            tampered_percent = 0

            tampering_score = 0

        else:

            if tampered_percent < 0.2:
                tampering_score = 0

            elif tampered_percent < 1:
                tampering_score = 2

            elif tampered_percent < 3:
                tampering_score = 4

            elif tampered_percent < 8:
                tampering_score = 6

            elif tampered_percent < 15:
                tampering_score = 8

            else:
                tampering_score = 10

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

            "suspicious_region_count":
                len(suspicious_regions),

            "suspicious_regions":
                suspicious_regions
        }


tampering_service = TamperingService()