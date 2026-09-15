"""Focused regression tests for explicit basin-area routing, not pipe inference."""
from collections import deque
import unittest

import numpy as np

from solution import fill_basin_to_network, direction_cells_reaching_outlets, analyze_direction_network


class FillBasinTests(unittest.TestCase):
    @staticmethod
    def competing_pipes():
        matrix = np.zeros((5, 7), dtype=np.uint8)
        matrix[:, (1, 5)] = 2
        matrix[-1, 1:5] = 1
        basin = np.zeros_like(matrix)
        basin[:4, 2:6] = 1
        return matrix, basin

    def test_nearer_pipe_outside_fill_basin_wins(self):
        matrix, basin = self.competing_pipes()
        original = matrix.copy()
        result, added, stats = fill_basin_to_network(
            matrix, basin, [(4, 5)], network_mask=np.ones_like(matrix),
        )
        # Left pipe is one step away; the in-basin right pipe is three away.
        self.assertEqual(result[1, 2], 3)
        self.assertEqual(stats['seed_count'], int(np.count_nonzero(matrix)))
        np.testing.assert_array_equal(matrix, original)
        np.testing.assert_array_equal(result[matrix > 0], matrix[matrix > 0])
        self.assertFalse(added[basin == 0].any())
        self.assertFalse(result[(basin == 0) & (matrix == 0)].any())
        reached = direction_cells_reaching_outlets(result, [(4, 5)])
        self.assertTrue(np.all(reached[added > 0]))

    def test_only_external_pipe_can_drain_basin(self):
        matrix = np.zeros((5, 5), dtype=np.uint8)
        matrix[:, 1] = 2
        basin = np.zeros_like(matrix)
        basin[:, 2:] = 1
        result, added, stats = fill_basin_to_network(
            matrix, basin, [(4, 1)], network_mask=np.ones_like(matrix),
        )
        np.testing.assert_array_equal(result[:, 2:], np.full((5, 3), 3))
        np.testing.assert_array_equal(added, basin)
        self.assertEqual(stats['filled_count'], 15)
        self.assertEqual(stats['unfilled_count'], 0)
        again, extra, _ = fill_basin_to_network(
            result, basin, [(4, 1)], network_mask=np.ones_like(matrix),
        )
        np.testing.assert_array_equal(again, result)
        self.assertFalse(extra.any())

    def test_external_pipe_cannot_fill_across_non_basin_zero_gap(self):
        matrix = np.zeros((5, 5), dtype=np.uint8)
        matrix[:, 0] = 2
        basin = np.zeros_like(matrix)
        basin[:, 2:] = 1
        result, added, stats = fill_basin_to_network(
            matrix, basin, [(4, 0)], network_mask=np.ones_like(matrix),
        )
        np.testing.assert_array_equal(result, matrix)
        self.assertFalse(added.any())
        self.assertEqual(stats['unfilled_count'], 15)

    def test_lasso_excludes_pipe_but_downstream_outlet_status_does_not(self):
        matrix, basin = self.competing_pipes()
        for exclude_path_only in (False, True):
            with self.subTest(exclude_path_only=exclude_path_only):
                allowed = np.ones_like(matrix)
                if exclude_path_only:
                    allowed[-1, 1:3] = 0  # nearer pipe cannot reach Outlet inside LASSO
                else:
                    allowed[:, :2] = 0  # nearer pipe itself is outside LASSO
                result, added, _ = fill_basin_to_network(
                    matrix, basin, [(4, 5)], network_mask=allowed,
                )
                self.assertEqual(result[1, 2], 3 if exclude_path_only else 1)
                self.assertFalse(added[allowed == 0].any())
                pipe_cells = list(map(tuple, np.argwhere((matrix > 0) & (allowed > 0))))
                reached = direction_cells_reaching_outlets(np.where(allowed, result, 0), pipe_cells)
                self.assertTrue(np.all(reached[added > 0]))

    def test_nearer_disconnected_pipe_wins_regardless_of_outlet_choice(self):
        matrix, basin = self.competing_pipes()
        matrix[:4, 1] = 4  # nearby branch does not reach the right outlet
        reference = None
        for outlets in ([(4, 5)], [(0, 1)], []):
            with self.subTest(outlets=outlets):
                result, added, stats = fill_basin_to_network(
                    matrix, basin, outlets, network_mask=np.ones_like(matrix),
                )
                self.assertEqual(result[1, 2], 3)  # one step left, never three right
                if reference is None:
                    reference = result
                np.testing.assert_array_equal(result, reference)
                np.testing.assert_array_equal(result[matrix > 0], matrix[matrix > 0])
                self.assertEqual(stats['outlet_checked'], bool(outlets))
                if not outlets:
                    self.assertIsNone(stats['existing_unreachable_count'])

    def test_fill_does_not_require_an_outlet(self):
        result, added, _ = fill_basin_to_network([[0, 4, 0]], [[1, 1, 1]])
        np.testing.assert_array_equal(result, [[1, 4, 3]])
        np.testing.assert_array_equal(added, [[1, 0, 1]])
        with self.assertRaisesRegex(ValueError, '기존 관로'):
            fill_basin_to_network([[0, 0]], [[1, 1]])

    def test_separate_domains_match_independent_per_cell_shortest_paths(self):
        rng = np.random.default_rng(20260908)
        steps = {1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
        for case in range(20):
            matrix = np.zeros((8, 11), dtype=np.uint8)
            matrix[:, (1, 9)] = 2
            matrix[-1, 1:9] = 1
            matrix[2, 4:6] = [1, 3]  # disconnected pipes are still nearest-pipe candidates
            basin = (rng.random(matrix.shape) > .3).astype(np.uint8)
            basin[:, 1] = 0  # do not force existing pipes into the fill domain
            allowed = np.ones_like(matrix)
            if case % 2:
                allowed[-1, 3] = 0  # breaks left pipe's downstream path
            basin &= allowed
            result, added, _ = fill_basin_to_network(matrix, basin, [(7, 9)], network_mask=allowed)
            for start in map(tuple, np.argwhere((basin > 0) & (matrix == 0))):
                # Search from each blank, stopping at the first original
                # pipe. Unlike production this is not a multi-source flood fill.
                queue = deque([(start, 0)])
                seen = {start}
                expected = None
                while queue and expected is None:
                    (row, col), distance = queue.popleft()
                    for dr, dc in steps.values():
                        nr, nc = row + dr, col + dc
                        if not (0 <= nr < 8 and 0 <= nc < 11 and allowed[nr, nc]):
                            continue
                        if matrix[nr, nc]:
                            expected = distance + 1
                            break
                        elif basin[nr, nc] and (nr, nc) not in seen:
                            seen.add((nr, nc))
                            queue.append(((nr, nc), distance + 1))
                if expected is None:
                    self.assertEqual(result[start], 0, (case, start))
                    continue
                row, col = start
                for _ in range(expected):
                    self.assertEqual(matrix[row, col], 0, (case, start))
                    self.assertTrue(basin[row, col])
                    self.assertIn(int(result[row, col]), steps, (case, start))
                    dr, dc = steps[int(result[row, col])]
                    row, col = row + dr, col + dc
                self.assertNotEqual(matrix[row, col], 0, (case, start))
                self.assertTrue(allowed[row, col])
            np.testing.assert_array_equal(result[matrix > 0], matrix[matrix > 0])
            self.assertFalse(added[basin == 0].any())
            pipe_cells = list(map(tuple, np.argwhere((matrix > 0) & (allowed > 0))))
            reached = direction_cells_reaching_outlets(np.where(allowed, result, 0), pipe_cells)
            self.assertTrue(np.all(reached[added > 0]))

    def test_shortest_paths_and_outlet_reachability_random_holes(self):
        rng = np.random.default_rng(42)
        for _ in range(35):
            matrix = np.zeros((13, 17), dtype=np.uint8)
            matrix[:, 8] = 2
            matrix[-1, 8] = 1
            matrix[4, 2:8] = 1  # tributary joining the trunk
            basin = (rng.random(matrix.shape) > .25).astype(np.uint8)
            basin[matrix > 0] = 1
            original = matrix.copy()
            result, added, stats = fill_basin_to_network(matrix, basin, [(12, 8)])
            # Independent shortest-distance oracle over basin cells.
            distance = np.full(matrix.shape, -1, dtype=int)
            queue = deque(map(tuple, np.argwhere(matrix > 0)))
            distance[matrix > 0] = 0
            while queue:
                row, col = queue.popleft()
                for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < 13 and 0 <= nc < 17 and basin[nr, nc] and distance[nr, nc] < 0:
                        distance[nr, nc] = distance[row, col] + 1
                        queue.append((nr, nc))
            np.testing.assert_array_equal(added > 0, (distance > 0))
            steps = {1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
            for row, col in np.argwhere(added):
                dr, dc = steps[int(result[row, col])]
                self.assertEqual(distance[row + dr, col + dc], distance[row, col] - 1)
            np.testing.assert_array_equal(matrix, original)
            np.testing.assert_array_equal(result[matrix > 0], matrix[matrix > 0])
            self.assertFalse(result[basin == 0].any())
            reached = direction_cells_reaching_outlets(result, [(12, 8)])
            self.assertTrue(np.all(reached[result > 0]))
            self.assertEqual(stats['filled_count'], int(added.sum()))
            again, extra, _ = fill_basin_to_network(result, basin, [(12, 8)])
            np.testing.assert_array_equal(again, result)
            self.assertFalse(extra.any())

    def test_unconnected_pipe_cycle_is_a_seed_without_changing_its_directions(self):
        matrix = np.zeros((5, 7), dtype=np.uint8)
        matrix[:, 1] = 2
        matrix[-1, 1] = 1
        matrix[2, 4:6] = [1, 3]
        basin = np.ones_like(matrix)
        basin[:, 3] = 0
        result, added, stats = fill_basin_to_network(matrix, basin, [(4, 1)])
        np.testing.assert_array_equal(result[matrix > 0], matrix[matrix > 0])
        self.assertEqual(result[1, 4], 2)
        self.assertEqual(result[2, 6], 3)
        self.assertEqual(stats['existing_unreachable_count'], 2)
        self.assertEqual(stats['unfilled_count'], 0)
        self.assertFalse(added[:, 3].any())
        report = analyze_direction_network(result, [(4, 1)])
        self.assertEqual(report['cycle_cell_count'], 2)

    def test_fill_reaches_local_pipe_without_following_its_downstream_path(self):
        matrix = np.array([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.uint8)
        basin = np.array([[1, 0, 1, 1], [1, 0, 1, 1]], dtype=np.uint8)
        result, added, _ = fill_basin_to_network(matrix, basin, [(0, 3)])
        self.assertEqual(result[1, 0], 4)
        self.assertEqual(added[0, 1], 0)
        self.assertEqual(result[0, 1], 1)  # pure API never edits original nonzeros

    def test_multiple_outlets_have_deterministic_tie(self):
        matrix = np.array([[4, 0, 0, 0, 4]], dtype=np.uint8)
        a, _, _ = fill_basin_to_network(matrix, np.ones_like(matrix), [(0, 4), (0, 0)])
        b, _, _ = fill_basin_to_network(matrix, np.ones_like(matrix), [(0, 0), (0, 4)])
        np.testing.assert_array_equal(a, [[4, 3, 3, 1, 4]])
        np.testing.assert_array_equal(a, b)

    def test_rejects_invalid_values_masks_and_outlets(self):
        for matrix in (np.array([[257]]), np.array([[1.5]]), np.array([[float('nan')]])):
            with self.assertRaises(ValueError):
                fill_basin_to_network(matrix, [[1]], [(0, 0)])
        for mask, outlets in (([[2]], [(0, 0)]), ([[1, 1]], [(0, 0)]), ([[0]], [(0, 0)]), ([[1]], [(1, 0)]), ([[1]], [(0.5, 0)])):
            with self.assertRaises(ValueError):
                fill_basin_to_network([[1]], mask, outlets)
        with self.assertRaises(ValueError):
            fill_basin_to_network([[0]], [[1]], [(0, 0)])


if __name__ == '__main__':
    unittest.main()
