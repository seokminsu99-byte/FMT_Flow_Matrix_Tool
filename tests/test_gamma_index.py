import tempfile
import unittest
from pathlib import Path

import numpy as np

from gamma_index import compute_gamma_index, load_plena_input_matrix, save_gamma_outputs
from plena_plots import render_gamma_plot


_DELTAS = {1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}


def _reference_trace(matrix: np.ndarray, active: np.ndarray, start: int):
    rows, cols = matrix.shape
    current = int(start)
    path = []
    seen = set()
    while True:
        if current in seen:
            return None
        seen.add(current)
        row = current % rows
        col = current // rows
        code = int(matrix[row, col])
        if not active[row, col] or code not in _DELTAS:
            return None
        path.append(current)
        dr, dc = _DELTAS[code]
        row2, col2 = row + dr, col + dc
        if row2 < 0 or row2 >= rows or col2 < 0 or col2 >= cols or not active[row2, col2]:
            return len(path) - 1, current, path
        current = row2 + col2 * rows


def _reference_gamma(matrix: np.ndarray):
    rows, _cols = matrix.shape
    active0 = matrix > 0
    traced = {}
    for raw_index in np.flatnonzero(active0.ravel(order="F")):
        index = int(raw_index)
        result = _reference_trace(matrix, active0, index)
        if result is not None:
            traced[index] = result
    if not traced:
        return None
    sink_counts = {}
    for index in sorted(traced):
        sink = traced[index][1]
        sink_counts[sink] = sink_counts.get(sink, 0) + 1
    max_count = max(sink_counts.values())
    main_sink = min(sink for sink, count in sink_counts.items() if count == max_count)
    basin = np.zeros_like(active0)
    for index, (_distance, sink, _path) in traced.items():
        if sink == main_sink:
            basin[index % rows, index // rows] = True

    active = basin.copy()
    lengths = []
    while np.any(active):
        selected = None
        selected_distance = -1
        selected_path = None
        for raw_index in np.flatnonzero(active.ravel(order="F")):
            index = int(raw_index)
            result = _reference_trace(matrix, active, index)
            if result is None:
                continue
            distance, _sink, path = result
            if distance > selected_distance:
                selected = index
                selected_distance = distance
                selected_path = path
        if selected is None or selected_path is None:
            break
        lengths.append(len(selected_path))
        for index in selected_path:
            active[index % rows, index // rows] = False
    lengths_array = np.asarray(lengths, dtype=int)
    unique, counts = np.unique(lengths_array, return_counts=True)
    weighted = unique * counts
    center = float(np.sum(unique * weighted) / np.sum(weighted))
    return main_sink, int(np.count_nonzero(basin)), lengths_array, center / float(np.max(unique))


class GammaIndexTests(unittest.TestCase):
    def test_branching_network_matches_hand_calculated_gamma(self) -> None:
        matrix = np.asarray(
            [
                [2, 0, 2],
                [1, 1, 2],
                [0, 0, 2],
            ],
            dtype=np.uint8,
        )

        result = compute_gamma_index(matrix, outlet=(2, 2))

        np.testing.assert_array_equal(result.removed_lengths_cell, np.asarray([5, 1]))
        np.testing.assert_array_equal(result.removed_lengths_edge, np.asarray([4, 0]))
        np.testing.assert_array_equal(result.unique_lengths, np.asarray([1, 5]))
        np.testing.assert_array_equal(result.occurrence_counts, np.asarray([1, 1]))
        self.assertEqual(result.contributing_cells, 6)
        self.assertAlmostEqual(result.gravity_center_x, 26.0 / 6.0)
        self.assertAlmostEqual(result.gamma, 13.0 / 15.0)

    def test_random_matrices_match_independent_matlab_style_reference(self) -> None:
        rng = np.random.default_rng(20260730)
        compared = 0
        for _ in range(80):
            matrix = rng.integers(0, 5, size=(5, 6), dtype=np.uint8)
            expected = _reference_gamma(matrix)
            if expected is None:
                continue
            result = compute_gamma_index(matrix)
            main_sink, contributing, lengths, gamma = expected
            expected_outlet = (main_sink % matrix.shape[0], main_sink // matrix.shape[0])
            self.assertEqual(result.outlet, expected_outlet)
            self.assertEqual(result.contributing_cells, contributing)
            np.testing.assert_array_equal(result.removed_lengths_cell, lengths)
            self.assertAlmostEqual(result.gamma, gamma)
            compared += 1
        self.assertGreaterEqual(compared, 60)

    def test_cycle_only_matrix_is_rejected(self) -> None:
        matrix = np.asarray([[1, 3]], dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "no valid downstream path"):
            compute_gamma_index(matrix)

    def test_plena_loader_and_saved_outputs_are_strict_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "sample.txt"
            input_path.write_text("3 3\n3 3\n2 0 2\n1 1 2\n0 0 2\n", encoding="utf-8")

            matrix, outlet = load_plena_input_matrix(input_path)
            result = compute_gamma_index(matrix, outlet=outlet)
            summary_path, csv_path = save_gamma_outputs(input_path, result)

            self.assertEqual(outlet, (2, 2))
            self.assertIn("Gamma: 0.8666666667", summary_path.read_text(encoding="utf-8"))
            self.assertIn("removed_flowline_length", csv_path.read_text(encoding="utf-8-sig"))

    def test_gamma_figure_contains_distribution_and_reference_lines(self) -> None:
        matrix = np.asarray([[2, 0, 2], [1, 1, 2], [0, 0, 2]], dtype=np.uint8)
        result = compute_gamma_index(matrix, outlet=(2, 2))

        image = render_gamma_plot(result, size=(900, 650))
        pixels = np.asarray(image)

        self.assertEqual(image.size, (900, 650))
        self.assertGreater(int(np.count_nonzero(np.any(pixels < 245, axis=2))), 10_000)
        red = (pixels[:, :, 0] > 180) & (pixels[:, :, 1] < 80) & (pixels[:, :, 2] < 80)
        self.assertGreater(int(np.count_nonzero(red)), 500)


if __name__ == "__main__":
    unittest.main()
