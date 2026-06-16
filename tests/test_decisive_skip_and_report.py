import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.api.report import _cleaned_text, _detector_rows, _make_styles
from app.api.upload import (
    apply_document_level_overrides,
    get_decisive_skip_reason,
    should_generate_annotation,
    skipped_mvss_result,
    skipped_trufor_result,
)
from app.services.annotation_service import _masked_display_geometry
from app.services.masking_detection_service import detect_masking


class DecisiveSkipReasonTests(unittest.TestCase):
    def test_synthetic_has_priority_over_masking_and_quality(self):
        reason = get_decisive_skip_reason(
            {"masking_detected": True},
            {"quality_status": "bad", "physical_damage_score": 90},
            {"synthetic_detected": True}
        )

        self.assertEqual(reason, "synthetic_detected")

    def test_masking_has_priority_over_quality(self):
        reason = get_decisive_skip_reason(
            {"masked_field_count": 1},
            {"quality_status": "bad"},
            {"synthetic_detected": False, "synthetic_score": 0, "authenticity_score": 90}
        )

        self.assertEqual(reason, "masked_fields_detected")

    def test_bad_quality_triggers_poor_quality(self):
        cases = [
            {"quality_status": "bad"},
            {"quality_status": "unprocessable"},
            {"rejection_recommended": True},
            {"quality_badge": "Unclear Document"},
            {"physical_damage_score": 70},
            {"crease_score": 70},
            {"damage_score": 70},
        ]

        for quality in cases:
            with self.subTest(quality=quality):
                reason = get_decisive_skip_reason(
                    {},
                    quality,
                    {"synthetic_detected": False, "synthetic_score": 0, "authenticity_score": 90}
                )
                self.assertEqual(reason, "poor_quality")

    def test_warning_quality_without_severe_damage_does_not_skip(self):
        reason = get_decisive_skip_reason(
            {},
            {"quality_status": "warning", "physical_damage_score": 69},
            {"synthetic_detected": False, "synthetic_score": 0, "authenticity_score": 90}
        )

        self.assertIsNone(reason)

    def test_skipped_detector_schemas_use_null_scores(self):
        mvss = skipped_mvss_result("synthetic_detected", cancelled=True, cancellation_requested=True)
        trufor = skipped_trufor_result("synthetic_detected")

        self.assertIsNone(mvss["score"])
        self.assertIsNone(mvss["tampering_score"])
        self.assertTrue(mvss["skipped"])
        self.assertTrue(mvss["cancelled"])
        self.assertEqual(mvss["status"], "cancelled_due_to_decisive_signal")
        self.assertIsNone(trufor["score"])
        self.assertIsNone(trufor["forgery_score"])
        self.assertTrue(trufor["skipped"])
        self.assertEqual(trufor["status"], "skipped_due_to_decisive_signal")

    def test_document_level_score_overrides(self):
        synthetic = apply_document_level_overrides(
            {"fraud_score": 80, "risk_level": "High Risk", "reasons": []},
            {"quality_status": "good"},
            {"synthetic_detected": True, "synthetic_score": 80, "authenticity_score": 20}
        )
        poor = apply_document_level_overrides(
            {"fraud_score": 5, "risk_level": "Low Risk", "reasons": []},
            {"quality_status": "bad", "physical_damage_score": 75},
            {"synthetic_detected": False, "synthetic_score": 0, "authenticity_score": 90}
        )

        self.assertEqual(synthetic["fraud_score"], 100)
        self.assertEqual(synthetic["score_override_reason"], "synthetic_detected")
        self.assertEqual(poor["fraud_score"], 50)
        self.assertEqual(poor["score_override_reason"], "poor_quality")

    def test_annotation_generation_decision(self):
        self.assertFalse(should_generate_annotation({
            "deep_skip_reason": "synthetic_detected",
            "document_authenticity_analysis": {"synthetic_detected": True}
        }))
        self.assertFalse(should_generate_annotation({
            "deep_skip_reason": "poor_quality",
            "document_quality_analysis": {"quality_status": "bad"}
        }))
        self.assertTrue(should_generate_annotation({
            "masking_analysis": {"masking_detected": True}
        }))

    def test_masking_regions_are_annotation_ready(self):
        result = detect_masking({
            "lines": [
                {
                    "text": "XXXX XXXX XXXX",
                    "confidence": 0.91,
                    "region": {"x": 710, "y": 1040, "w": 420, "h": 55},
                    "bbox": [[710, 1040], [1130, 1040], [1130, 1095], [710, 1095]]
                }
            ]
        })

        region = result["masked_regions"][0]
        self.assertTrue(result["masking_detected"])
        self.assertEqual(region["label"], "Masked field")
        self.assertEqual(region["source_detector"], "masking")
        self.assertEqual(region["original_region"], {"x": 710, "y": 1040, "w": 420, "h": 55})
        self.assertEqual(region["x"], 710)
        self.assertEqual(region["w"], 420)
        self.assertEqual(region["reason"], "Masked identifier pattern detected in OCR text")
        self.assertEqual(region["source"], "ocr_mask_pattern")

    def test_masked_ocr_line_has_priority_over_nearby_text(self):
        result = detect_masking({
            "lines": [
                {
                    "text": "Aadhaar S number has been hidden",
                    "confidence": 0.42,
                    "region": {"x": 520, "y": 835, "w": 520, "h": 58},
                },
                {
                    "text": "XXXX XXXX XXXX",
                    "confidence": 0.88,
                    "region": {"x": 680, "y": 1008, "w": 460, "h": 62},
                },
            ]
        })

        self.assertTrue(result["masking_detected"])
        self.assertEqual(len(result["masked_regions"]), 1)
        region = result["masked_regions"][0]
        self.assertEqual(region["text"], "XXXX XXXX XXXX")
        self.assertEqual(region["original_region"], {"x": 680, "y": 1008, "w": 460, "h": 62})
        self.assertEqual(region["reason"], "Masked identifier pattern detected in OCR text")

    def test_masked_display_region_expands_small_boxes(self):
        geometry = _masked_display_geometry(
            {
                "label": "Masked field",
                "original_region": {"x": 750, "y": 1050, "w": 40, "h": 12},
            },
            (1200, 1600, 3)
        )

        self.assertEqual(geometry["original_region"], {"x": 750, "y": 1050, "w": 40, "h": 12})
        self.assertGreater(geometry["display_region"]["w"], 40)
        self.assertGreater(geometry["display_region"]["h"], 12)
        self.assertLessEqual(geometry["display_region"]["w"], 72)
        self.assertLessEqual(geometry["display_region"]["h"], 36)
        self.assertGreaterEqual(geometry["display_region"]["x"], 0)
        self.assertGreaterEqual(geometry["display_region"]["y"], 0)

    def test_masked_display_region_scales_normalized_coordinates(self):
        geometry = _masked_display_geometry(
            {
                "label": "Masked field",
                "region": {"x": 0.5, "y": 0.8, "w": 0.1, "h": 0.02},
            },
            (1000, 2000, 3)
        )

        self.assertEqual(geometry["original_region"], {"x": 1000, "y": 800, "w": 200, "h": 20})
        self.assertIn("normalized", geometry["coordinate_source"])


class ReportDisplayTests(unittest.TestCase):
    def test_cleaned_text_omits_readable_ocr_by_default(self):
        structured = {
            "fields": {
                "name": {"value": "Manisha Dhakad"},
                "gender": {"value": "Female"},
            }
        }
        lines = [
            {"text": "GOVERNMENT OF INDIA"},
            {"text": "Husband: SANJAY SINGH"},
        ]

        default_text = _cleaned_text(structured, lines)
        debug_text = _cleaned_text(structured, lines, include_readable=True)

        self.assertNotIn("Readable OCR Text", default_text)
        self.assertNotIn("GOVERNMENT OF INDIA", default_text)
        self.assertIn("Readable OCR Text", debug_text)

    def test_detector_rows_show_skipped_as_not_run(self):
        rows = _detector_rows(
            {
                "forgery_localization_analysis": skipped_trufor_result("synthetic_detected"),
                "tampering_analysis": skipped_mvss_result("synthetic_detected", cancelled=True),
            },
            _make_styles()
        )
        row_text = "\n".join(
            cell.getPlainText()
            for row in rows
            for cell in row
            if hasattr(cell, "getPlainText")
        )

        self.assertIn("Not run", row_text)
        self.assertIn("Skipped because synthetic document was already detected.", row_text)
        self.assertNotIn("No scoring-eligible MVSS tampering region", row_text)


if __name__ == "__main__":
    unittest.main()
