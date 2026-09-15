import tempfile
import unittest
from pathlib import Path

import numpy as np

import gold_labels


class GoldLabelTests(unittest.TestCase):
    def test_save_find_and_load_gold_label_by_image_hash_and_shape(self) -> None:
        image = np.zeros((8, 9, 3), dtype=np.uint8)
        image[1, 2] = (10, 20, 30)
        matrix = np.array([[1, 0, 2], [0, 3, 4]], dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = gold_labels.save_gold_label(tmpdir, image, r"C:\data\sample.png", matrix)
            matches = gold_labels.find_matching_gold_labels(tmpdir, image, 2, 3)
            loaded = gold_labels.load_gold_label(path)

        self.assertEqual(len(matches), 1)
        self.assertTrue(np.array_equal(matches[0].matrix, matrix))
        self.assertTrue(np.array_equal(loaded.matrix, matrix))
        self.assertTrue(np.array_equal(loaded.review_mask, np.ones_like(matrix)))
        self.assertEqual(loaded.review_scope, "full")
        self.assertEqual(loaded.rows, 2)
        self.assertEqual(loaded.cols, 3)

    def test_evaluate_prediction_separates_pipe_presence_and_direction(self) -> None:
        gold = np.array(
            [
                [1, 2, 0],
                [0, 3, 4],
            ],
            dtype=np.uint8,
        )
        pred = np.array(
            [
                [1, 3, 1],
                [0, 0, 4],
            ],
            dtype=np.uint8,
        )

        metrics = gold_labels.evaluate_prediction(pred, gold)

        self.assertEqual(metrics["pipe"]["tp"], 3)
        self.assertEqual(metrics["pipe"]["fp"], 1)
        self.assertEqual(metrics["pipe"]["fn"], 1)
        self.assertAlmostEqual(metrics["pipe"]["precision"], 0.75)
        self.assertAlmostEqual(metrics["pipe"]["recall"], 0.75)
        self.assertEqual(metrics["direction"]["strict_exact"], 2)
        self.assertEqual(metrics["direction"]["gold_total"], 4)
        self.assertAlmostEqual(metrics["direction"]["strict_recall"], 0.5)

    def test_sparse_review_mask_limits_evaluation(self) -> None:
        gold = np.array([[1, 2, 0], [0, 3, 4]], dtype=np.uint8)
        pred = np.array([[1, 3, 1], [4, 0, 4]], dtype=np.uint8)
        reviewed = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)

        metrics = gold_labels.evaluate_prediction(pred, gold, review_mask=reviewed)

        self.assertEqual(metrics["review"]["reviewed_cells"], 2)
        self.assertEqual(metrics["pipe"]["tp"], 2)
        self.assertEqual(metrics["pipe"]["fp"], 0)
        self.assertEqual(metrics["direction"]["strict_exact"], 1)
        self.assertEqual(metrics["direction"]["gold_total"], 2)

    def test_sparse_gold_saves_merge_reviewed_cells(self) -> None:
        image = np.zeros((8, 9, 3), dtype=np.uint8)
        first = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        second = np.array([[0, 0], [0, 4]], dtype=np.uint8)
        first_mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        second_mask = np.array([[0, 0], [0, 1]], dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = gold_labels.save_gold_label(
                tmpdir,
                image,
                r"C:\data\sample.png",
                first,
                review_mask=first_mask,
                metadata={"review_scope": "corrected_cells_only"},
            )
            gold_labels.save_gold_label(
                tmpdir,
                image,
                r"C:\data\sample.png",
                second,
                review_mask=second_mask,
                metadata={"review_scope": "corrected_cells_only"},
            )
            loaded = gold_labels.load_gold_label(path)

        self.assertEqual(loaded.matrix.tolist(), [[1, 0], [0, 4]])
        self.assertEqual(loaded.review_mask.tolist(), [[1, 0], [0, 1]])


if __name__ == "__main__":
    unittest.main()
