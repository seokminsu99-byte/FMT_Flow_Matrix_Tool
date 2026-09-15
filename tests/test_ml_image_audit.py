import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import gold_labels
import ml_image_audit
import solution


class MLImageAuditTests(unittest.TestCase):
    def test_directory_audit_returns_rows_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = np.full((160, 240, 3), 255, dtype=np.uint8)
            cv2.arrowedLine(image, (30, 80), (210, 80), (0, 0, 0), 6, tipLength=0.16)
            image_path = root / "sample.png"
            ok, data = cv2.imencode(".png", image)
            self.assertTrue(ok)
            data.tofile(str(image_path))

            records = ml_image_audit.evaluate_image_directory(root, rows=(12, 16))
            summary = ml_image_audit.summarize_records(records)

        self.assertEqual(len(records), 2)
        self.assertEqual(summary["image_count"], 1)
        self.assertEqual(summary["row_runs"], 2)
        self.assertIn("object_count", records[0])
        self.assertIn("candidate_count_by_row", records[0])
        self.assertIn("centerline_coverage", records[0])
        self.assertIn("centerline_unassigned", records[0])
        self.assertGreaterEqual(float(records[0]["centerline_coverage"]), 0.0)
        self.assertLessEqual(float(records[0]["centerline_coverage"]), 1.0)

    def test_directory_audit_reports_gold_metrics_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gold_root = root / "gold"
            image = np.full((160, 240, 3), 255, dtype=np.uint8)
            cv2.arrowedLine(image, (30, 80), (210, 80), (0, 0, 0), 6, tipLength=0.16)
            image_path = root / "sample.png"
            ok, data = cv2.imencode(".png", image)
            self.assertTrue(ok)
            data.tofile(str(image_path))

            matrix, _occ, _boxes, _meta = solution.compute_direction_matrix_with_meta(image, 12)
            gold_labels.save_gold_label(gold_root, image, str(image_path), matrix, prediction_matrix=matrix)

            records = ml_image_audit.evaluate_image_directory(root, rows=(12,), gold_root=gold_root)
            summary = ml_image_audit.summarize_records(records)

        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["has_gold"])
        self.assertAlmostEqual(float(records[0]["final_pipe_f1"]), 1.0)
        self.assertAlmostEqual(float(records[0]["final_direction_strict_recall"]), 1.0)
        self.assertEqual(summary["gold_row_runs"], 1)


if __name__ == "__main__":
    unittest.main()
