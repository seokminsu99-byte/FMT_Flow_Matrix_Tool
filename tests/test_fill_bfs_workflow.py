"""Area drainage stays separate throughout fill -> pipe BFS -> PLENA."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import gui
from gamma_index import load_plena_input_matrix


def make_app(matrix):
    app = object.__new__(gui.NFMATApp)
    app._initialize_workspace_actions()
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
    app.outlet_cell = None
    app.outlet_cells = set()
    app.selected_cell = None
    app.source_kind = "gis"
    app.matrix_arrow_view = False
    app.lasso_mode = False
    app.outlet_pick_mode = False
    app.status_var = mock.Mock()
    app.canvas = mock.Mock()
    for name in ("_set_selected_cell", "_queue_render", "_update_matrix_preview", "display_image",
                 "_sync_workspace_task_controls", "_update_action_states"):
        setattr(app, name, mock.Mock())
    app._ensure_grid = mock.Mock(return_value=True)
    app._processing_image_and_roi_mask = mock.Mock(
        return_value=(np.full((app.A * 20, app.B * 20, 3), 255, dtype=np.uint8), None))
    app._start_compute_task = lambda label, compute, finish, **kwargs: finish(compute(None))
    app._record_support_baseline()
    app._basin_cell_mask = np.ones_like(app.current_matrix)
    return app


def bfs_result(matrix):
    return (matrix.copy(), (matrix > 0).astype(np.uint8), {
        "head_fill": 0, "head_neighbor_fill": 0, "component_fill": 0,
        "outlet_fill": 0, "cycle_fix": 0, "unresolved_count": 0,
    }, np.zeros_like(matrix))


STATE_FIELDS = (
    "current_matrix", "base_matrix", "occ_matrix", "path_occ_matrix", "support_mask",
    "confidence_matrix", "base_confidence_matrix", "model_applied_mask",
    "base_model_applied_mask", "unresolved_mask", "user_edit_mask", "_area_fill_mask",
    "_area_fill_values",
)


def state_snapshot(app):
    return {name: value.copy() if isinstance((value := vars(app).get(name)), np.ndarray) else value
            for name in STATE_FIELDS}


class FillBFSWorkflowTests(unittest.TestCase):
    def assert_state_equal(self, app, snapshot):
        for name, value in snapshot.items():
            with self.subTest(field=name):
                np.testing.assert_array_equal(vars(app).get(name), value)

    @staticmethod
    def gap_app():
        matrix = np.zeros((3, 5), dtype=np.uint8)
        matrix[1, 1] = matrix[1, 3] = 1
        app = make_app(matrix)
        app.fill_empty_cells()
        app.outlet_cell = (1, 3)
        app.outlet_cells = {app.outlet_cell}
        return app, matrix

    def test_fill_without_outlet_then_select_only_existing_pipe(self):
        matrix = np.zeros((3, 5), dtype=np.uint8)
        matrix[1, 1:4] = 1
        app = make_app(matrix)
        app._basin_cell_mask[:, 0] = 0
        app.fill_empty_cells()
        self.assertTrue(app._has_area_fill())
        self.assertIsNone(app.outlet_cell)
        app.enable_outlet_pick_mode()
        self.assertTrue(app.outlet_pick_mode)
        for cell in ((0, 2), (0, 0)):
            app._event_to_cell = mock.Mock(return_value=cell)
            app.on_canvas_click(mock.Mock())
            self.assertIsNone(app.outlet_cell)
            self.assertTrue(app.outlet_pick_mode)
        app._event_to_cell = mock.Mock(return_value=(1, 3))
        app.on_canvas_click(mock.Mock())
        self.assertEqual(app.outlet_cell, (1, 3))
        self.assertFalse(app.outlet_pick_mode)

    def test_bfs_uses_pipe_snapshot_and_undo_keeps_bfs_bridge(self):
        app, original = self.gap_app()
        displayed = app.current_matrix.copy()
        baseline = {key: value.copy() for key, value in app._base_support_state.items()}
        corrected = original.copy()
        corrected[1, 2] = 1  # This used to be area drainage; BFS makes it a pipe.

        def assist(_img, matrix, _rows, **kwargs):
            np.testing.assert_array_equal(matrix, original)
            np.testing.assert_array_equal(app.current_matrix, displayed)
            np.testing.assert_array_equal(kwargs["occ"], original > 0)
            return bfs_result(corrected)

        with mock.patch.object(gui, "assist_direction_matrix", side_effect=assist):
            app.apply_bfs_assist()
        self.assertEqual(app._area_fill_mask[1, 2], 0)
        self.assertEqual(app.occ_matrix[1, 2], 1)
        self.assertAlmostEqual(float(app.confidence_matrix[1, 2]), 0.55)
        area = app._area_fill_mask > 0
        self.assertTrue(area.any())
        for name in ("occ_matrix", "path_occ_matrix", "support_mask", "confidence_matrix"):
            self.assertFalse(np.any(getattr(app, name)[area]), name)
        np.testing.assert_array_equal(app._area_fill_values, np.where(area, app.current_matrix, 0))
        np.testing.assert_array_equal(app.base_matrix, original)
        for key, value in baseline.items():
            np.testing.assert_array_equal(app._base_support_state[key], value)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "after_bfs.txt"
            app._write_plena_input_file(str(path))
            exported, outlet = load_plena_input_matrix(path)
        np.testing.assert_array_equal(exported, app.current_matrix)
        self.assertEqual(outlet, (1, 3))
        self.assertEqual(app.last_plena_sanitized_count, 0)
        app.undo_fill_empty_cells()
        np.testing.assert_array_equal(app.current_matrix, corrected)
        self.assertFalse(app._has_area_fill())

    def test_clear_after_bfs_and_refill_restores_original_baseline(self):
        app, original = self.gap_app()
        corrected = original.copy()
        corrected[1, 2] = 1
        with mock.patch.object(gui, "assist_direction_matrix", return_value=bfs_result(corrected)):
            app.apply_bfs_assist()
        app.clear_edits()
        np.testing.assert_array_equal(app.current_matrix, original)
        np.testing.assert_array_equal(app.confidence_matrix, app.base_confidence_matrix)
        for name in ("occ_matrix", "path_occ_matrix", "support_mask"):
            np.testing.assert_array_equal(getattr(app, name), original > 0)
        self.assertFalse(app._has_area_fill())

    def test_refill_keeps_nearest_disconnected_pipe_before_zeroing(self):
        matrix = np.zeros((5, 7), dtype=np.uint8)
        matrix[:, 1] = 4  # Disconnected left pipe exits north.
        matrix[:, 5] = 2
        app = make_app(matrix)
        app.fill_empty_cells()
        app.outlet_cell = (4, 5)
        app.outlet_cells = {app.outlet_cell}
        self.assertEqual(app.current_matrix[1, 2], 3)
        with mock.patch.object(gui, "assist_direction_matrix", return_value=bfs_result(matrix)):
            app.apply_bfs_assist()
        self.assertEqual(app.current_matrix[1, 2], 0)  # Never reroute to distant right pipe.
        self.assertEqual(app._area_fill_mask[1, 2], 0)
        self.assertEqual(app._area_fill_values[1, 2], 0)
        self.assertEqual(app.current_matrix[1, 4], 1)
        self.assertEqual(app._area_fill_mask[1, 4], 1)
        app.undo_fill_empty_cells()
        expected = matrix.copy()
        expected[:, 1] = 0
        np.testing.assert_array_equal(app.current_matrix, expected)

    def test_multi_outlet_helper_never_uses_live_fill_for_partition(self):
        matrix = np.zeros((3, 5), dtype=np.uint8)
        matrix[1, :2] = 3
        matrix[1, 3:] = 1
        app = make_app(matrix)
        app.fill_empty_cells()
        displayed = app.current_matrix.copy()
        captured = []

        def assist(_img, part_matrix, _rows, **kwargs):
            captured.append(part_matrix.copy())
            self.assertFalse(part_matrix[matrix == 0].any())
            np.testing.assert_array_equal(app.current_matrix, displayed)
            return bfs_result(part_matrix)

        with mock.patch.object(gui, "assist_direction_matrix", side_effect=assist):
            assisted, *_ = app._assist_direction_matrix_for_active_outlets(
                np.full((60, 100, 3), 255, dtype=np.uint8), matrix=matrix.copy(),
                occ=app.occ_matrix.copy(), path_occ=app.path_occ_matrix.copy(),
                confidence=app.confidence_matrix.copy(), support_mask=app.support_mask.copy(),
                outlets=[(1, 0), (1, 4)],
            )
        self.assertEqual(len(captured), 2)
        np.testing.assert_array_equal(assisted, matrix)
        np.testing.assert_array_equal(app.current_matrix, displayed)

    def test_compute_error_and_cancel_keep_filled_workspace(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                app, original = self.gap_app()
                before = state_snapshot(app)
                pending = {}

                def start(_label, compute, finish, **_kwargs):
                    pending.update(compute=compute, finish=finish)
                    return True

                app._start_compute_task = start
                app.apply_bfs_assist()
                self.assert_state_equal(app, before)
                app._task_on_done = pending["finish"]
                app._task_busy = True
                if cancelled:
                    corrected = original.copy()
                    corrected[1, 2] = 1
                    with mock.patch.object(gui, "assist_direction_matrix", return_value=bfs_result(corrected)):
                        value = pending["compute"](None)
                    app._task_cancel_event.set()
                    error = None
                else:
                    with mock.patch.object(gui, "assist_direction_matrix", side_effect=RuntimeError("BFS failed")):
                        with self.assertRaisesRegex(RuntimeError, "BFS failed"):
                            pending["compute"](None)
                    value, error = None, RuntimeError("BFS failed")
                app._task_queue.put((app._task_generation, value, error))
                with mock.patch.object(gui.messagebox, "showerror"):
                    app._poll_compute_task()
                self.assert_state_equal(app, before)

    def test_refill_failure_does_not_commit_partial_bfs_state(self):
        app, original = self.gap_app()
        before = state_snapshot(app)
        corrected = original.copy()
        corrected[1, 2] = 1
        with mock.patch.object(gui, "assist_direction_matrix", return_value=bfs_result(corrected)), \
                mock.patch.object(gui, "fill_basin_to_network", side_effect=RuntimeError("refill failed")):
            with self.assertRaisesRegex(RuntimeError, "refill failed"):
                app.apply_bfs_assist()
        self.assert_state_equal(app, before)

    def test_empty_bfs_result_clears_old_area_without_refill_error(self):
        app, original = self.gap_app()
        empty = np.zeros_like(original)
        with mock.patch.object(gui, "assist_direction_matrix", return_value=bfs_result(empty)), \
                mock.patch.object(gui, "fill_basin_to_network") as refill:
            app.apply_bfs_assist()
        refill.assert_not_called()
        np.testing.assert_array_equal(app.current_matrix, empty)
        np.testing.assert_array_equal(app._area_fill_mask, empty)
        np.testing.assert_array_equal(app._area_fill_values, empty)
        np.testing.assert_array_equal(app.base_matrix, original)
        self.assertFalse(app._has_area_fill())

    def test_real_bfs_corrects_pipe_direction_after_fill(self):
        matrix = np.zeros((5, 5), dtype=np.uint8)
        matrix[:, 2] = 2
        matrix[1, 1] = 3
        app = make_app(matrix)
        app.fill_empty_cells()
        app.outlet_cell = (4, 2)
        app.outlet_cells = {app.outlet_cell}
        app.apply_bfs_assist()
        self.assertEqual(app.current_matrix[1, 1], 1)
        exported, _, removed = app._matrix_for_plena()
        np.testing.assert_array_equal(exported, app.current_matrix)
        self.assertEqual(removed, 0)
        app.undo_fill_empty_cells()
        corrected = matrix.copy()
        corrected[1, 1] = 1
        np.testing.assert_array_equal(app.current_matrix, corrected)


if __name__ == "__main__":
    unittest.main()
