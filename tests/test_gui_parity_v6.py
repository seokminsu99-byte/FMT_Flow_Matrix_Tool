"""Compatibility checks for rectangular editing and explicit LASSO clipping."""

import unittest
from unittest import mock

import numpy as np

from solution import direction_cells_reaching_outlets, fill_basin_to_network
from test_workspace_v6 import make_app


class GuiParityV6Tests(unittest.TestCase):
    def test_pending_gis_row_change_never_erases_existing_network(self):
        app = self._network_crossing_basin_edge()
        app.source_kind = "gis"
        app.rows_var = mock.Mock()
        app.rows_var.get.return_value = 20
        original = app.current_matrix.copy()
        self.assertFalse(app._ensure_grid())
        np.testing.assert_array_equal(app.current_matrix, original)
        self.assertIn("GIS 방향 보정", app.status_var.set.call_args.args[0])

    def test_gis_shape_is_not_inferred_again_from_preview_aspect_ratio(self):
        app = self._network_crossing_basin_edge()
        app.source_kind = "gis"
        app.rows_var = mock.Mock()
        app.rows_var.get.return_value = 2
        app.img_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        app.compressed_grid_shape = None
        app.trimmed_grid_shape = None
        app.roi_polygon_points = []
        app._sync_grid_shape()
        self.assertEqual((app.A, app.B), (2, 4))

    @staticmethod
    def _network_crossing_basin_edge():
        # The surveyed pipe leaves the fill basin before reaching its outlet.
        # A fill-only boundary must not delete this existing downstream path.
        app = make_app([[1, 1, 2, 0], [0, 0, 1, 1]])
        app.outlet_cell = (1, 3)
        app.outlet_cells = {(1, 3)}
        app._basin_cell_mask = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.uint8)
        return app

    def test_gis_fill_boundary_does_not_block_rectangular_cell_editing(self):
        app = self._network_crossing_basin_edge()
        self.assertTrue(app._cell_is_inside_active_roi((0, 3)))
        self.assertTrue(app._cell_is_inside_active_roi(app.outlet_cell))
        self.assertFalse(app._cell_is_inside_active_roi((-1, 0)))
        self.assertFalse(app._cell_is_inside_active_roi((2, 0)))

    def test_explicit_lasso_blocks_edits_but_does_not_inherit_gis_fill_boundary(self):
        app = self._network_crossing_basin_edge()
        app.roi_cell_mask = np.ones_like(app.current_matrix)
        app.roi_cell_mask[:, 0] = 0
        self.assertFalse(app._cell_is_inside_active_roi((0, 0)))
        self.assertTrue(app._cell_is_inside_active_roi((0, 3)))
        self.assertTrue(app._cell_is_inside_active_roi(app.outlet_cell))

    def test_plena_preserves_existing_outlet_path_outside_gis_fill_boundary(self):
        app = self._network_crossing_basin_edge()
        original = app.current_matrix.copy()
        exported, removed_roi, removed_unreachable = app._matrix_for_plena()
        np.testing.assert_array_equal(exported, original)
        np.testing.assert_array_equal(app.current_matrix, original)
        self.assertEqual((removed_roi, removed_unreachable), (0, 0))

    def test_plena_still_zeros_explicit_lasso_exterior(self):
        app = self._network_crossing_basin_edge()
        app.roi_cell_mask = np.ones_like(app.current_matrix)
        app.roi_cell_mask[:, 0] = 0
        original = app.current_matrix.copy()
        exported, removed_roi, removed_unreachable = app._matrix_for_plena()
        expected = original.copy()
        expected[:, 0] = 0
        np.testing.assert_array_equal(exported, expected)
        np.testing.assert_array_equal(app.current_matrix, original)
        self.assertEqual((removed_roi, removed_unreachable), (1, 0))

    def test_empty_grid_restores_edit_mode_and_keeps_explicit_lasso(self):
        app = self._network_crossing_basin_edge()
        app._ensure_grid = mock.Mock(return_value=True)
        app.matrix_arrow_view = True
        app.matrix_arrow_button = mock.Mock()
        app.outlet_pick_mode = True
        app.lasso_mode = True
        app.roi_cell_mask = np.ones_like(app.current_matrix)
        app.roi_cell_mask[:, 0] = 0
        explicit_roi = app.roi_cell_mask.copy()
        app.start_empty_grid()
        self.assertFalse(app.matrix_arrow_view)
        self.assertFalse(app.outlet_pick_mode)
        self.assertFalse(app.lasso_mode)
        self.assertFalse(np.any(app.current_matrix))
        np.testing.assert_array_equal(app.roi_cell_mask, explicit_roi)
        app.canvas.focus_set.assert_called()

    def test_fill_uses_existing_network_outside_basin_without_filling_exterior_zeros(self):
        app = self._network_crossing_basin_edge()
        original = app.current_matrix.copy()
        basin = app._basin_cell_mask.copy()
        result, added, stats = fill_basin_to_network(
            original, basin, [app.outlet_cell], network_mask=np.ones_like(original),
        )
        expected_added = np.array([[0, 0, 0, 0], [1, 1, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(added, expected_added)
        np.testing.assert_array_equal(result[basin == 0], original[basin == 0])
        np.testing.assert_array_equal(result[original > 0], original[original > 0])
        np.testing.assert_array_equal(app.current_matrix, original)
        self.assertEqual(stats["filled_count"], 2)
        reached = direction_cells_reaching_outlets(result, [app.outlet_cell])
        self.assertTrue(np.all(reached[added > 0]))

    def test_region_application_and_bfs_postprocessing_preserve_network_outside_basin(self):
        app = self._network_crossing_basin_edge()
        original = app.current_matrix.copy()
        original_support = app.support_mask.copy()
        app._apply_workspace_region()
        np.testing.assert_array_equal(app.current_matrix, original)
        np.testing.assert_array_equal(app.support_mask, original_support)
        app._ensure_grid = mock.Mock(return_value=True)
        app._processing_image_and_roi_mask = mock.Mock(
            return_value=(np.full((20, 40, 3), 255, dtype=np.uint8), None),
        )
        app._assist_direction_matrix_for_active_outlets = mock.Mock(return_value=(
            original.copy(), original_support.copy(),
            {"component_fill": 0, "outlet_fill": 0}, np.zeros_like(original),
        ))
        app.apply_bfs_assist()
        app._assist_direction_matrix_for_active_outlets.assert_called_once()
        np.testing.assert_array_equal(app.current_matrix, original)
        np.testing.assert_array_equal(app.support_mask, original_support)
        self.assertFalse(np.any(app.unresolved_mask))

    def test_gis_lasso_remains_an_explicit_edit_and_export_boundary(self):
        app = self._network_crossing_basin_edge()
        app._basin_is_lasso = True
        app._basin_cell_mask = np.ones_like(app.current_matrix)
        app._basin_cell_mask[:, 0] = 0
        self.assertFalse(app._cell_is_inside_active_roi((0, 0)))
        self.assertTrue(app._cell_is_inside_active_roi((0, 1)))
        self.assertTrue(app._cell_is_inside_active_roi(app.outlet_cell))
        np.testing.assert_array_equal(app._active_edit_mask(), app._basin_cell_mask)
        exported, removed_roi, removed_unreachable = app._matrix_for_plena()
        expected = app.current_matrix.copy()
        expected[:, 0] = 0
        np.testing.assert_array_equal(exported, expected)
        self.assertEqual((removed_roi, removed_unreachable), (1, 0))
        app._apply_workspace_region()
        np.testing.assert_array_equal(app.current_matrix, expected)


if __name__ == "__main__":
    unittest.main()
