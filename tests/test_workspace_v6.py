from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import gui
from gamma_index import load_plena_input_matrix
from workspace_state import atomic_write_text, basin_mask_from_world_rings, load_settings_object


def make_app(matrix):
    app = object.__new__(gui.NFMATApp)
    app._initialize_workspace_actions()
    app._plena_process = None
    app.current_matrix = np.array(matrix, dtype=np.uint8)
    app.base_matrix = app.current_matrix.copy()
    app.A, app.B = app.current_matrix.shape
    for name in ("occ_matrix", "path_occ_matrix", "support_mask"):
        setattr(app, name, (app.current_matrix > 0).astype(np.uint8))
    app.confidence_matrix = (app.current_matrix > 0).astype(np.float32)
    app.base_confidence_matrix = app.confidence_matrix.copy()
    app.model_applied_mask = np.zeros_like(app.current_matrix)
    app.base_model_applied_mask = app.model_applied_mask.copy()
    app.unresolved_mask = np.zeros_like(app.current_matrix)
    app.user_edit_mask = np.zeros_like(app.current_matrix)
    app.roi_cell_mask = None
    app.outlet_cell = (app.A - 1, app.B // 2)
    app.outlet_cells = {app.outlet_cell}
    app.selected_cell = app.outlet_cell
    app.status_var = mock.Mock()
    app.canvas = mock.Mock()
    app._set_selected_cell = mock.Mock()
    app._queue_render = mock.Mock()
    app._update_matrix_preview = mock.Mock()
    app.display_image = mock.Mock()
    app._start_compute_task = lambda label, compute, finish, **kwargs: finish(compute(None))
    app._record_support_baseline()
    return app


class WorkspaceV6Tests(unittest.TestCase):
    def test_fill_attaches_to_nearest_external_pipe_and_exports_then_undoes(self):
        matrix = np.zeros((5, 7), dtype=np.uint8)
        matrix[:, (1, 5)] = 2
        matrix[-1, 1:5] = 1
        for explicit_lasso in (False, True):
            with self.subTest(explicit_lasso=explicit_lasso):
                app = make_app(matrix)
                app.outlet_cell = (4, 5)
                app.outlet_cells = {app.outlet_cell}
                app._basin_cell_mask = np.zeros_like(matrix)
                app._basin_cell_mask[:4, 2:6] = 1
                expected_original = matrix.copy()
                if explicit_lasso:
                    app.roi_cell_mask = np.ones_like(matrix)
                    app.roi_cell_mask[:, :2] = 0
                    app._apply_workspace_region()
                    expected_original[:, :2] = 0
                app.fill_empty_cells()
                self.assertEqual(app.current_matrix[1, 2], 1 if explicit_lasso else 3)
                self.assertFalse(app._area_fill_mask[app._basin_cell_mask == 0].any())
                filled = app.current_matrix.copy()
                app.fill_empty_cells()  # never reclassify an existing fill as pipes
                np.testing.assert_array_equal(app.current_matrix, filled)
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / 'nearest_pipe.txt'
                    app._write_plena_input_file(str(path))
                    exported, outlet = load_plena_input_matrix(path)
                np.testing.assert_array_equal(exported, filled)
                self.assertEqual(outlet, app.outlet_cell)
                app.undo_fill_empty_cells()
                np.testing.assert_array_equal(app.current_matrix, expected_original)

    def test_fill_and_undo_preserve_existing_network_and_support(self):
        matrix = np.zeros((5, 5), dtype=np.uint8)
        matrix[:, 2] = 2
        matrix[-1, 2] = 1
        matrix[1, :2] = 1
        app = make_app(matrix)
        basin = np.ones_like(matrix)
        basin[3:, 0] = 0
        app._basin_cell_mask = basin
        occ = app.occ_matrix.copy()
        support = app.support_mask.copy()
        app.fill_empty_cells()
        self.assertGreater(np.count_nonzero(app._area_fill_mask), 0)
        np.testing.assert_array_equal(app.current_matrix[matrix > 0], matrix[matrix > 0])
        self.assertFalse(np.any(app.current_matrix[basin == 0]))
        np.testing.assert_array_equal(app.occ_matrix, occ)
        np.testing.assert_array_equal(app.support_mask, support)
        app.undo_fill_empty_cells()
        np.testing.assert_array_equal(app.current_matrix, matrix)
        np.testing.assert_array_equal(app.support_mask, support)
        self.assertFalse(app._has_area_fill())

    def test_fill_obeys_intersection_of_basin_and_lasso_and_exports_added_flow(self):
        matrix = np.zeros((5, 5), dtype=np.uint8)
        matrix[:, 2] = 2
        matrix[-1, 2] = 1
        app = make_app(matrix)
        app._basin_cell_mask = np.ones_like(matrix)
        app.roi_cell_mask = np.ones_like(matrix)
        app.roi_cell_mask[:, 0] = 0
        app.fill_empty_cells()
        self.assertEqual(np.count_nonzero(app.current_matrix), 20)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "filled.txt"
            app._write_plena_input_file(str(path))
            exported, outlet = load_plena_input_matrix(path)
        np.testing.assert_array_equal(exported, app.current_matrix)
        self.assertEqual(outlet, app.outlet_cell)
        self.assertEqual(app.last_plena_sanitized_count, 0)

    def test_fill_requires_explicit_basin_and_does_not_guess_rectangle(self):
        app = make_app([[1, 1, 1], [0, 0, 0]])
        with mock.patch.object(gui.messagebox, "showwarning") as warning:
            app.fill_empty_cells()
        self.assertIn("유역 경계", warning.call_args.args[1])
        np.testing.assert_array_equal(app.current_matrix, app.base_matrix)

    def test_clear_edits_restores_original_occupancy_and_support(self):
        app = make_app(np.zeros((3, 3), dtype=np.uint8))
        app.current_matrix[1, 1] = 1
        app.occ_matrix[1, 1] = app.path_occ_matrix[1, 1] = app.support_mask[1, 1] = 1
        app.clear_edits()
        for name in ("current_matrix", "occ_matrix", "path_occ_matrix", "support_mask"):
            self.assertFalse(np.any(getattr(app, name)), name)

    def test_roi_merge_zeros_old_outside_values(self):
        app = make_app(np.ones((3, 3), dtype=np.uint8))
        roi = np.eye(3, dtype=np.uint8)
        merged = app._merge_roi_matrix(app.current_matrix, np.full((3, 3), 2, dtype=np.uint8), roi)
        np.testing.assert_array_equal(merged, roi * 2)

    def test_trim_does_not_discard_empty_basin_area_waiting_to_be_filled(self):
        app = make_app(np.zeros((9, 9), dtype=np.uint8))
        app.current_matrix[4, 4] = 1
        app.outlet_cell = (4, 4)
        app.outlet_cells = {(4, 4)}
        app._basin_cell_mask = np.ones((9, 9), dtype=np.uint8)
        self.assertEqual(app._current_trim_bounds(padding=1), (0, 9, 0, 9))

    def test_invalid_direction_cannot_wrap_to_valid_export_code(self):
        app = make_app([[1, 1, 1]])
        app.current_matrix = np.array([[257, 1, 1]], dtype=np.int64)
        app.outlet_cell = (0, 2)
        app.outlet_cells = {(0, 2), (0, 1)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.txt"
            with self.assertRaises(ValueError):
                app._write_plena_input_file(str(path))
            self.assertFalse(path.exists())
        self.assertEqual(app.outlet_cells, {(0, 2), (0, 1)})

    def test_export_header_uses_actual_matrix_shape(self):
        app = make_app([[1, 1, 1]])
        app.A = 99
        app.B = 99
        app.outlet_cell = (0, 2)
        app.outlet_cells = {(0, 2)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.txt"
            app._write_plena_input_file(str(path))
            self.assertEqual(path.read_text(encoding="utf-8").splitlines()[0], "1 3")

    def test_cancel_plena_options_does_not_save_or_change_outlet(self):
        app = make_app([[1, 1, 1]])
        app._write_plena_input_file = mock.Mock()
        original_outlets = app.outlet_cells.copy()
        # Cancellation must not depend on a separately installed PLENA binary.
        with tempfile.TemporaryDirectory() as folder:
            executable = Path(folder) / "PLENA.exe"
            executable.touch()  # Never executed: the options dialog is cancelled.
            with mock.patch.object(gui, "_resource_path", return_value=executable), \
                 mock.patch.object(gui.simpledialog, "askfloat", return_value=None):
                app.run_plena()
        app._write_plena_input_file.assert_not_called()
        self.assertEqual(app.outlet_cells, original_outlets)

    def test_atomic_save_failure_keeps_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            atomic_write_text(path, '{"old": true}')
            with mock.patch("workspace_state.os.replace", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    atomic_write_text(path, '{"new": true}')
            self.assertEqual(path.read_text(encoding="utf-8"), '{"old": true}')
            self.assertEqual(len(list(Path(folder).iterdir())), 1)

    def test_corrupt_settings_root_does_not_break_startup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            for value in ("[]", "null", "7", "broken"):
                atomic_write_text(path, value)
                self.assertEqual(load_settings_object(path), {})
                self.assertEqual(path.read_text(encoding="utf-8"), value)

    def test_basin_preserves_holes_and_disconnected_features(self):
        outer = [(0, 0), (5, 0), (5, 5), (0, 5)]
        hole = [(2, 2), (3, 2), (3, 3), (2, 3)]
        mask = basin_mask_from_world_rings([[outer, hole]], (5, 5), (0, 0, 5, 5))
        self.assertEqual(mask.sum(), 24)
        self.assertEqual(mask[2, 2], 0)
        outside = [(10, 10), (11, 10), (11, 11), (10, 11)]
        self.assertFalse(basin_mask_from_world_rings([[outside]], (5, 5), (0, 0, 5, 5)).any())

    def test_rotated_basin_mask_matches_quarter_turn(self):
        ring = [(0, 0), (2, 0), (2, 5), (0, 5)]
        normal = basin_mask_from_world_rings([[ring]], (5, 5), (0, 0, 5, 5))
        rotated = basin_mask_from_world_rings([[ring]], (5, 5), (0, 0, 5, 5), 90)
        np.testing.assert_array_equal(rotated, np.rot90(normal))


if __name__ == "__main__":
    unittest.main()
