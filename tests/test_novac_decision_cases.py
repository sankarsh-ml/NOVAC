import json
import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.document_authenticity_service import analyze_document_authenticity
from app.services.document_quality_service import analyze_document_quality
from app.services.scoring_service import calculate_fraud_score


FIXTURES = {
    "chatgpt_aadhaar": Path(r"c:\Users\hp\Downloads\ChatGPT Image Jun 8, 2026, 10_24_16 AM.png"),
    "govtid": Path(r"d:\LaptopBills\govtid.jpg"),
    "torn": Path(r"c:\Users\hp\Downloads\torn.jpg"),
    "child_aadhaar": Path(r"c:\Users\hp\Downloads\IMG_6635.jpg"),
    "eaadhaar_pdf": Path(r"c:\Users\hp\Downloads\EAadhaar_0013070160836520241213145234_2609202517433_unlocked.pdf"),
}


OCR_FIXTURES = {
    "chatgpt_aadhaar": {
        "avg_confidence": 0.90,
        "text": "Government of India Rohan Kumar DOB 15/08/1995 Male 1234 5678 9012 VID 9876 5432 1098 7654",
        "lines": [{} for _ in range(8)],
    },
    "govtid": {
        "avg_confidence": 0.75,
        "text": "INCOME TAX DEPARTMENT GOVT OF INDIA D DINESH PADMANABHAN DIVAKARAN 20/04/1978 AHRPD5455C",
        "lines": [{} for _ in range(6)],
    },
    "torn": {
        "avg_confidence": 0.82,
        "text": "GOVERNMENT OF INDIA Manisha Dhakad Year of Birth 1983 Female 6433 4737 4657",
        "lines": [{} for _ in range(8)],
    },
    "child_aadhaar": {
        "avg_confidence": 0.70,
        "text": "Government of India Annya Singh DOB 26/07/2019 Female XXXX XXXX XXXX",
        "lines": [{} for _ in range(6)],
    },
    "eaadhaar_pdf": {
        "avg_confidence": 0.95,
        "text": "Enrolment No Government of India Aadhaar VID Details as on Aadhaar no issued Address",
        "lines": [{} for _ in range(8)],
    },
}


def analyze_case(name):
    path = FIXTURES[name]
    ocr = OCR_FIXTURES[name]
    authenticity = analyze_document_authenticity(str(path), ocr_result=ocr)

    if authenticity.get("official_digital_pdf_detected"):
        quality = {
            "quality_status": "good",
            "quality_score": 100,
            "damage_score": 0,
            "physical_damage_score": 0,
            "rejection_recommended": False,
            "analysis_confidence": 100,
            "quality_reliable": True,
            "quality_warning": False,
            "reasons": [],
        }
    else:
        quality = analyze_document_quality(str(path), ocr_result=ocr)

    fraud = calculate_fraud_score(
        {},
        ocr,
        {},
        {},
        {},
        document_quality_result=quality,
        document_authenticity_result=authenticity,
    )

    summary = {
        "document_quality": quality,
        "authenticity": authenticity,
        "fraud_score": fraud.get("fraud_score"),
        "risk_level": fraud.get("risk_level"),
        "result_status": fraud.get("result_status"),
        "rejection_reason_type": fraud.get("rejection_reason_type"),
        "quality_badge": fraud.get("quality_badge"),
        "quality_notice": fraud.get("quality_notice"),
        "banner_title": fraud.get("banner_title"),
        "banner_body": fraud.get("banner_body"),
    }
    print(f"\n{name}\n{json.dumps(summary, indent=2, ensure_ascii=False)}")

    return summary


class NovacDecisionCaseTests(unittest.TestCase):
    def require_fixture(self, name):
        if not FIXTURES[name].exists():
            self.skipTest(f"Fixture not available: {FIXTURES[name]}")

    def test_chatgpt_aadhaar_synthetic_not_quality(self):
        self.require_fixture("chatgpt_aadhaar")
        result = analyze_case("chatgpt_aadhaar")
        self.assertEqual(result["document_quality"]["quality_status"], "good")
        self.assertGreaterEqual(result["document_quality"]["quality_score"], 75)
        self.assertTrue(result["authenticity"]["synthetic_detected"])
        self.assertGreaterEqual(result["authenticity"]["synthetic_score"], 65)
        self.assertEqual(result["result_status"], "synthetic_suspected")
        self.assertEqual(result["risk_level"], "Synthetic Document Suspected")
        self.assertIsNone(result["quality_badge"])

    def test_govtid_camera_capture_low_risk(self):
        self.require_fixture("govtid")
        result = analyze_case("govtid")
        self.assertIn(result["document_quality"]["quality_status"], {"good", "warning"})
        self.assertFalse(result["authenticity"]["synthetic_detected"])
        self.assertLess(result["authenticity"]["synthetic_score"], 45)
        self.assertNotEqual(result["risk_level"], "Unreliable Scan")
        self.assertNotEqual(result["quality_badge"], "Unprocessable Document")

    def test_official_eaadhaar_pdf_passes(self):
        self.require_fixture("eaadhaar_pdf")
        result = analyze_case("eaadhaar_pdf")
        self.assertEqual(result["authenticity"]["acquisition_type"], "official_digital_pdf")
        self.assertFalse(result["authenticity"]["synthetic_detected"])
        self.assertLess(result["authenticity"]["synthetic_score"], 35)
        self.assertGreaterEqual(result["authenticity"]["authenticity_score"], 70)
        self.assertEqual(result["document_quality"]["quality_status"], "good")
        self.assertEqual(result["result_status"], "passed")
        self.assertIsNone(result["quality_badge"])

    def test_torn_aadhaar_quality_bad_but_not_override(self):
        self.require_fixture("torn")
        result = analyze_case("torn")
        self.assertIn(result["document_quality"]["quality_status"], {"bad", "warning"})
        self.assertGreaterEqual(result["document_quality"]["physical_damage_score"], 65)
        self.assertNotEqual(result["risk_level"], "Unreliable Scan")
        self.assertEqual(result["quality_badge"], "Unclear Document")
        self.assertIsNotNone(result["quality_notice"])

    def test_child_aadhaar_screen_photo_not_unprocessable(self):
        self.require_fixture("child_aadhaar")
        result = analyze_case("child_aadhaar")
        self.assertIn(result["document_quality"]["quality_status"], {"good", "warning"})
        self.assertLess(result["document_quality"]["physical_damage_score"], 70)
        self.assertFalse(result["authenticity"]["synthetic_detected"])
        self.assertNotEqual(result["risk_level"], "Unreliable Scan")
        self.assertNotEqual(result["quality_badge"], "Unprocessable Document")


if __name__ == "__main__":
    unittest.main()
