import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services import result_service


class DeleteResult:
    def __init__(self, deleted_count):
        self.deleted_count = deleted_count


class FakeCollection:
    def __init__(self, documents):
        self.documents = list(documents)

    def find_one(self, query):
        case_id = query.get("case_id")
        return next(
            (
                document
                for document in self.documents
                if document.get("case_id") == case_id
            ),
            None
        )

    def delete_one(self, query):
        case_id = query.get("case_id")
        before_count = len(self.documents)
        self.documents = [
            document
            for document in self.documents
            if document.get("case_id") != case_id
        ]

        return DeleteResult(
            before_count - len(self.documents)
        )

    def find(self):
        return list(self.documents)

    def delete_many(self, _query):
        deleted_count = len(self.documents)
        self.documents = []

        return DeleteResult(deleted_count)


class ResultServiceCleanupTests(unittest.TestCase):
    def test_delete_result_removes_case_artifacts_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            backend_uploads = temp_path / "backend" / "uploads"
            backend_reports = temp_path / "backend" / "reports"
            project_uploads = temp_path / "project" / "uploads"
            project_reports = temp_path / "project" / "reports"

            for directory in (
                backend_uploads,
                backend_reports,
                project_uploads,
                project_reports
            ):
                directory.mkdir(parents=True)

            expected_deleted = [
                backend_uploads / "original.png",
                backend_uploads / "original_noqr.png",
                backend_uploads / "original_annotated.png",
                backend_uploads / "forgery_maps" / "map.png",
                backend_reports / "report_NOVAC-TEST.pdf"
            ]

            for path in expected_deleted:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("artifact", encoding="utf-8")

            outside_file = temp_path / "outside.txt"
            outside_file.write_text("keep", encoding="utf-8")

            document = {
                "case_id": "NOVAC-TEST",
                "stored_filename": "original.png",
                "file_path": "uploads/original.png",
                "analysis_image_path": "uploads/original.png",
                "annotated_image_path": "/uploads/original_annotated.png",
                "preprocessing_analysis": {
                    "output_path": "uploads/original_noqr.png"
                },
                "forgery_localization_analysis": {
                    "localization_map_path": "uploads/forgery_maps/map.png"
                },
                "metadata_analysis": {
                    "source_path": str(outside_file)
                }
            }

            roots = {
                "uploads": (
                    backend_uploads,
                    project_uploads
                ),
                "reports": (
                    backend_reports,
                    project_reports
                )
            }

            collection = FakeCollection([document])

            with (
                patch.object(result_service, "LOCAL_ARTIFACT_ROOTS", roots),
                patch.object(result_service, "analysis_collection", collection)
            ):
                self.assertTrue(
                    result_service.delete_result("NOVAC-TEST")
                )

            for path in expected_deleted:
                self.assertFalse(
                    path.exists(),
                    f"{path} should have been deleted"
                )

            self.assertTrue(outside_file.exists())
            self.assertEqual(collection.documents, [])

    def test_delete_all_results_cleans_artifacts_for_each_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            backend_uploads = temp_path / "backend" / "uploads"
            backend_reports = temp_path / "backend" / "reports"
            backend_uploads.mkdir(parents=True)
            backend_reports.mkdir(parents=True)

            first_file = backend_uploads / "first.png"
            second_file = backend_uploads / "second.png"
            first_report = backend_reports / "report_NOVAC-FIRST.pdf"

            for path in (
                first_file,
                second_file,
                first_report
            ):
                path.write_text("artifact", encoding="utf-8")

            roots = {
                "uploads": (backend_uploads,),
                "reports": (backend_reports,)
            }

            collection = FakeCollection([
                {
                    "case_id": "NOVAC-FIRST",
                    "file_path": "uploads/first.png"
                },
                {
                    "case_id": "NOVAC-SECOND",
                    "file_path": "uploads/second.png"
                }
            ])

            with (
                patch.object(result_service, "LOCAL_ARTIFACT_ROOTS", roots),
                patch.object(result_service, "analysis_collection", collection)
            ):
                self.assertEqual(
                    result_service.delete_all_results(),
                    2
                )

            self.assertFalse(first_file.exists())
            self.assertFalse(second_file.exists())
            self.assertFalse(first_report.exists())
            self.assertEqual(collection.documents, [])


if __name__ == "__main__":
    unittest.main()
