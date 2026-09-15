import tempfile
import struct
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch

import cell_training as ct
import crs_support
import gis_matrix
import gui
import matrix_arrows
import solution
import synthetic_arrows


class FrozenResourcePathTests(unittest.TestCase):
    def test_resource_path_falls_back_to_exe_directory_when_meipass_lacks_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app_dir = root / "NFMAT"
            internal_dir = app_dir / "_internal"
            app_dir.mkdir()
            internal_dir.mkdir()
            exe_path = app_dir / "NFMAT.exe"
            exe_path.write_bytes(b"")
            plena_path = app_dir / "PLENA.exe"
            plena_path.write_bytes(b"PLENA")

            with mock.patch.object(gui.sys, "frozen", True, create=True), mock.patch.object(
                gui.sys,
                "_MEIPASS",
                str(internal_dir),
                create=True,
            ), mock.patch.object(gui.sys, "executable", str(exe_path)):
                self.assertEqual(gui._resource_path("PLENA.exe").resolve(), plena_path.resolve())

    def test_resource_path_prefers_meipass_when_bundled_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app_dir = root / "NFMAT"
            internal_dir = app_dir / "_internal"
            app_dir.mkdir()
            internal_dir.mkdir()
            exe_path = app_dir / "NFMAT.exe"
            exe_path.write_bytes(b"")
            (app_dir / "icon.png").write_bytes(b"outer")
            bundled_icon = internal_dir / "icon.png"
            bundled_icon.write_bytes(b"inner")

            with mock.patch.object(gui.sys, "frozen", True, create=True), mock.patch.object(
                gui.sys,
                "_MEIPASS",
                str(internal_dir),
                create=True,
            ), mock.patch.object(gui.sys, "executable", str(exe_path)):
                self.assertEqual(gui._resource_path("icon.png").resolve(), bundled_icon.resolve())


class OutputNamingTests(unittest.TestCase):
    @staticmethod
    def _make_app(rotation_degrees: float = -20.0) -> gui.NFMATApp:
        app = object.__new__(gui.NFMATApp)
        app.img_path = r"C:\data\Secho4.shp"
        app.A = 30
        app.B = 30
        app.rotation_var = mock.Mock()
        app.rotation_var.get.return_value = rotation_degrees
        return app

    def test_output_dataset_name_normalizes_negative_angle(self) -> None:
        app = self._make_app(-20.0)

        self.assertEqual(app._output_dataset_name(), "Secho4_row30_Angle340")

    def test_output_dataset_name_preserves_fine_angle(self) -> None:
        app = self._make_app(20.125)

        self.assertEqual(app._output_dataset_name(), "Secho4_row30_Angle20.125")

    def test_output_dataset_name_includes_selected_gis_region(self) -> None:
        app = self._make_app(0.0)
        app._gis_boundary_value = "Seocho 4"

        self.assertEqual(app._output_dataset_name(), "Secho4_Seocho_4_row30_Angle0")

    def test_default_plena_path_uses_shared_folder_and_file_name(self) -> None:
        app = self._make_app(-20.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            app.output_root = Path(tmpdir)

            path = Path(app._default_plena_input_path())

            self.assertEqual(path.name, "Secho4_row30_Angle340.txt")
            self.assertEqual(path.parent.name, "Secho4_row30_Angle340")
            self.assertTrue(path.parent.is_dir())

    def test_write_plena_input_file_uses_plena_text_format(self) -> None:
        app = self._make_app(0.0)
        app.A = 2
        app.B = 2
        app.current_matrix = np.array([[1, 2], [4, 3]], dtype=np.uint8)
        app.outlet_cell = (1, 1)
        app.unresolved_mask = np.zeros((2, 2), dtype=np.uint8)
        app.manual_edits_since_autogen = False
        app.last_plena_sanitized_count = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "Secho4_row2_Angle0.txt"
            app._write_plena_input_file(str(path))

            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                ["2 2", "2 2", "1 2", "4 3"],
            )

    def test_write_plena_input_file_preserves_selected_outlets_and_removes_noncontributing_flow(self) -> None:
        app = self._make_app(0.0)
        app.A = 2
        app.B = 3
        app.current_matrix = np.array([[1, 1, 1], [3, 3, 3]], dtype=np.uint8)
        app.outlet_cell = (0, 2)
        app.outlet_cells = {(0, 2), (1, 0)}
        app.unresolved_mask = np.ones((2, 3), dtype=np.uint8)
        app.unresolved_mask[0, 2] = 0
        app.manual_edits_since_autogen = False
        app.last_plena_sanitized_count = 0
        app.last_plena_outlet_reduced_count = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "Secho4_row2_Angle0.txt"
            app._write_plena_input_file(str(path))

            self.assertEqual(path.read_text(encoding="utf-8").splitlines()[1], "1 3")

        self.assertEqual(app.outlet_cell, (0, 2))
        self.assertEqual(app.outlet_cells, {(0, 2), (1, 0)})
        self.assertEqual(app.last_plena_outlet_reduced_count, 1)
        self.assertEqual(app.last_plena_lasso_removed_count, 0)
        self.assertEqual(app.last_plena_unreachable_count, 3)
        self.assertEqual(app.last_plena_sanitized_count, 3)

    def test_write_plena_input_file_hard_clips_lasso_and_unreachable_manual_cells(self) -> None:
        app = self._make_app(0.0)
        app.A = 3
        app.B = 5
        app.current_matrix = np.array(
            [
                [1, 1, 1, 1, 3],
                [1, 3, 0, 0, 0],
                [1, 1, 1, 1, 4],
            ],
            dtype=np.uint8,
        )
        app.outlet_cell = (0, 4)
        app.outlet_cells = {(0, 4)}
        app.roi_cell_mask = np.ones((3, 5), dtype=np.uint8)
        app.roi_cell_mask[2, :] = 0
        app.unresolved_mask = np.zeros((3, 5), dtype=np.uint8)
        app.manual_edits_since_autogen = True

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lasso.txt"
            app._write_plena_input_file(str(path))
            written = np.asarray(
                [[int(value) for value in line.split()] for line in path.read_text(encoding="utf-8").splitlines()[2:]],
                dtype=np.uint8,
            )

        self.assertTrue(np.array_equal(written[0], np.array([1, 1, 1, 1, 3], dtype=np.uint8)))
        self.assertEqual(int(np.count_nonzero(written[1:])), 0)
        self.assertEqual(app.last_plena_lasso_removed_count, 5)
        self.assertEqual(app.last_plena_unreachable_count, 2)
        self.assertEqual(app.last_plena_sanitized_count, 7)

    def test_export_matrix_uses_shared_name_and_forces_txt(self) -> None:
        app = self._make_app(-20.0)
        app.current_matrix = np.ones((30, 30), dtype=np.uint8)
        app.outlet_cell = (0, 0)
        app.status_var = mock.Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            app.output_root = Path(tmpdir)
            selected_path = Path(tmpdir) / "custom.csv"
            with mock.patch.object(
                gui.filedialog,
                "asksaveasfilename",
                return_value=str(selected_path),
            ) as save_dialog, mock.patch.object(
                gui.NFMATApp,
                "_write_plena_input_file",
                autospec=True,
                return_value=str(selected_path.with_suffix(".txt")),
            ) as write_plena:
                app.export_matrix()

            self.assertEqual(save_dialog.call_args.kwargs["initialfile"], "Secho4_row30_Angle340.txt")
            self.assertEqual(save_dialog.call_args.kwargs["filetypes"], [("PLENA 입력 TXT", "*.txt")])
            write_plena.assert_called_once_with(app, str(selected_path.with_suffix(".txt")))

    def test_plena_match_summary_uses_actual_sanitized_input_matrix(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app.current_matrix = np.ones((2, 2), dtype=np.uint8)
        app._last_plena_input_matrix = np.array([[1, 0], [0, 0]], dtype=np.uint8)

        summary = app._plena_match_summary(app._last_plena_input_matrix.copy())

        self.assertIn("PLENA 입력 행렬", summary)
        self.assertIn("4/4", summary)

    def test_apply_lasso_to_matrix_removes_only_outside_cells(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app.current_matrix = np.ones((3, 3), dtype=np.uint8)
        app.base_matrix = app.current_matrix.copy()
        app.base_confidence_matrix = np.ones((3, 3), dtype=np.float32)
        app.confidence_matrix = app.base_confidence_matrix.copy()
        app.base_model_applied_mask = np.ones((3, 3), dtype=np.uint8)
        app.model_applied_mask = app.base_model_applied_mask.copy()
        app.occ_matrix = np.ones((3, 3), dtype=np.uint8)
        app.path_occ_matrix = np.ones((3, 3), dtype=np.uint8)
        app.support_mask = np.ones((3, 3), dtype=np.uint8)
        app.unresolved_mask = np.ones((3, 3), dtype=np.uint8)
        app.user_edit_mask = np.ones((3, 3), dtype=np.uint8)
        app._flood_cell_mask = np.ones((3, 3), dtype=np.uint8)
        app.roi_cell_mask = np.zeros((3, 3), dtype=np.uint8)
        app.roi_cell_mask[1, 1] = 1
        app.selected_cell = (1, 1)
        app.outlet_cell = (0, 0)
        app.manual_edits_since_autogen = True
        app.status_var = mock.Mock()
        app.canvas = mock.Mock()
        app._invalidate_detection_cache = mock.Mock()
        app._invalidate_matrix_arrow_cache = mock.Mock()
        app._queue_render = mock.Mock()
        app._update_matrix_preview = mock.Mock()

        app.apply_lasso_to_matrix()

        self.assertEqual(int(np.count_nonzero(app.current_matrix)), 1)
        self.assertEqual(int(app.current_matrix[1, 1]), 1)
        for name in (
            "base_matrix",
            "base_confidence_matrix",
            "confidence_matrix",
            "base_model_applied_mask",
            "model_applied_mask",
            "occ_matrix",
            "path_occ_matrix",
            "support_mask",
            "unresolved_mask",
            "user_edit_mask",
            "_flood_cell_mask",
        ):
            self.assertEqual(int(np.count_nonzero(getattr(app, name))), 1, name)
        self.assertIsNone(app.outlet_cell)
        self.assertFalse(app.manual_edits_since_autogen)

    def test_apply_bfs_assist_zeros_direction_cells_that_do_not_reach_outlet(self) -> None:
        app = object.__new__(gui.NFMATApp)
        assisted = np.array(
            [
                [1, 1, 1, 1, 3],
                [1, 3, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        support = (assisted > 0).astype(np.uint8)
        app.A, app.B = assisted.shape
        app.current_matrix = assisted.copy()
        app.outlet_cell = (0, 4)
        app.outlet_cells = {(0, 4)}
        app.roi_cell_mask = None
        app.confidence_matrix = np.ones_like(assisted, dtype=np.float32)
        app.model_applied_mask = np.ones_like(assisted, dtype=np.uint8)
        app.occ_matrix = support.copy()
        app.path_occ_matrix = support.copy()
        app.support_mask = support.copy()
        app.unresolved_mask = np.zeros_like(assisted, dtype=np.uint8)
        app.selected_cell = (0, 4)
        app.status_var = mock.Mock()
        app.canvas = mock.Mock()
        app._ensure_grid = mock.Mock(return_value=True)
        app._processing_image_and_roi_mask = mock.Mock(
            return_value=(np.full((20, 50, 3), 255, dtype=np.uint8), None)
        )
        app._assist_direction_matrix_for_active_outlets = mock.Mock(
            return_value=(
                assisted.copy(),
                support.copy(),
                {
                    "head_fill": 0,
                    "head_neighbor_fill": 0,
                    "component_fill": 0,
                    "outlet_fill": 0,
                    "cycle_fix": 0,
                    "unresolved_count": 0,
                },
                np.zeros_like(assisted, dtype=np.uint8),
            )
        )
        app._set_selected_cell = mock.Mock()
        app._queue_render = mock.Mock()
        app._update_matrix_preview = mock.Mock()

        app._start_compute_task = lambda label, compute, finish: finish(compute(None))
        app.apply_bfs_assist()

        self.assertTrue(np.array_equal(app.current_matrix[0], assisted[0]))
        self.assertEqual(int(np.count_nonzero(app.current_matrix[1])), 0)
        self.assertEqual(int(np.count_nonzero(app.unresolved_mask[1])), 2)
        self.assertEqual(int(np.count_nonzero(app.model_applied_mask[1, :2])), 0)
        self.assertIn("Outlet 미도달 0 처리 2칸", app.status_var.set.call_args.args[0])

    def test_manual_edit_cannot_reintroduce_value_outside_lasso(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app.current_matrix = np.zeros((3, 3), dtype=np.uint8)
        app.roi_cell_mask = np.zeros((3, 3), dtype=np.uint8)
        app.roi_cell_mask[1, 1] = 1
        app.selected_cell = (0, 0)
        app.manual_edits_since_autogen = False
        app.user_edit_mask = np.zeros((3, 3), dtype=np.uint8)
        app.model_applied_mask = np.zeros((3, 3), dtype=np.uint8)
        app.occ_matrix = np.zeros((3, 3), dtype=np.uint8)
        app.path_occ_matrix = np.zeros((3, 3), dtype=np.uint8)
        app.confidence_matrix = np.zeros((3, 3), dtype=np.float32)
        app.status_var = mock.Mock()
        app.canvas = mock.Mock()
        app._ensure_grid = mock.Mock(return_value=True)

        app._set_selected_value(1)

        self.assertEqual(int(np.count_nonzero(app.current_matrix)), 0)
        self.assertFalse(app.manual_edits_since_autogen)
        self.assertIn("항상 0", app.status_var.set.call_args.args[0])

    def test_lasso_cell_membership_uses_cell_center_not_small_edge_overlap(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app.img_bgr = np.full((100, 100, 3), 255, dtype=np.uint8)
        app.A = 2
        app.B = 2
        app.roi_polygon_points = [(0.0, 0.0), (49.0, 0.0), (49.0, 49.0), (0.0, 49.0)]
        app._build_roi_mask_pyramid = mock.Mock()

        cell_mask = app._roi_cell_mask_from_polygon()

        self.assertTrue(np.array_equal(cell_mask, np.array([[1, 0], [0, 0]], dtype=np.uint8)))

    def test_gis_project_lasso_pixels_convert_to_world_coordinates(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app._gis_map_bbox = (100.0, 200.0, 300.0, 400.0)
        app.img_pil = gui.Image.new("RGB", (201, 101))
        app.roi_polygon_points = [(0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)]

        rings = app._lasso_world_rings()

        self.assertEqual(rings[0][0], (100.0, 400.0))
        self.assertEqual(rings[0][2], (300.0, 200.0))


class ToolbarLayoutTests(unittest.TestCase):
    def test_toolbar_partition_uses_balanced_three_rows_at_minimum_width(self) -> None:
        widths = [431, 337, 428, 533, 323, 218, 323]

        layout = gui.NFMATApp._partition_toolbar_widths(widths, available_width=1036)

        self.assertEqual(layout, ((0, 1), (2, 3), (4, 5, 6)))

    def test_toolbar_partition_reduces_to_two_rows_when_space_allows(self) -> None:
        widths = [431, 337, 428, 533, 323, 218, 323]

        layout = gui.NFMATApp._partition_toolbar_widths(widths, available_width=1836)

        self.assertEqual(layout, ((0, 1, 2), (3, 4, 5, 6)))

    def test_toolbar_partition_never_exceeds_available_width(self) -> None:
        widths = [431, 337, 428, 533, 323, 218, 323]
        available = 1036

        layout = gui.NFMATApp._partition_toolbar_widths(widths, available_width=available)
        row_widths = [sum(widths[index] for index in row) + 6 * (len(row) - 1) for row in layout]

        self.assertTrue(all(width <= available for width in row_widths))


class GISProjectInteractionTests(unittest.TestCase):
    def test_world_bbox_for_grid_bounds_preserves_trimmed_gis_origin(self) -> None:
        bbox = gui.NFMATApp._world_bbox_for_grid_bounds(
            (0.0, 0.0, 80.0, 60.0),
            (6, 8),
            (1, 5, 2, 6),
        )

        self.assertEqual(bbox, (20.0, 10.0, 60.0, 50.0))

    def test_trim_matrix_margins_crops_all_aligned_state_and_preserves_coordinates(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app.A, app.B = 6, 8
        matrix = np.zeros((6, 8), dtype=np.uint8)
        matrix[2, 3] = 1
        matrix[3, 3] = 2
        matrix[3, 4] = 3
        original = matrix.copy()
        app.base_matrix = matrix.copy()
        app.current_matrix = matrix.copy()
        app.base_confidence_matrix = (matrix > 0).astype(np.float32)
        app.confidence_matrix = app.base_confidence_matrix.copy()
        app.base_model_applied_mask = np.zeros_like(matrix)
        app.model_applied_mask = np.zeros_like(matrix)
        app.occ_matrix = (matrix > 0).astype(np.uint8)
        app.path_occ_matrix = app.occ_matrix.copy()
        app.support_mask = app.occ_matrix.copy()
        app.unresolved_mask = np.zeros_like(matrix)
        app.user_edit_mask = np.zeros_like(matrix)
        app._flood_cell_mask = np.zeros_like(matrix)
        app._flood_cell_mask[3, 4] = 1
        app.roi_cell_mask = np.zeros_like(matrix)
        app.roi_cell_mask[2:4, 3:5] = 1
        app.roi_mask = np.full((60, 80), 255, dtype=np.uint8)
        app.roi_polygon_points = [(25.0, 15.0), (55.0, 15.0), (55.0, 45.0)]
        app._lasso_preview_point = (40.0, 30.0)
        app._roi_mask_pil = gui.Image.fromarray(app.roi_mask, mode="L")
        app._roi_mask_pyramid = []
        app._image_pyramid = []
        app.img_bgr = np.zeros((60, 80, 3), dtype=np.uint8)
        app.img_pil = gui.Image.fromarray(np.zeros((60, 80, 3), dtype=np.uint8))
        app.source_kind = "gis"
        app.arrow_boxes = [(28, 18, 52, 42)]
        app.selected_cell = (2, 3)
        app.outlet_cell = (3, 4)
        app.outlet_cells = {(3, 4)}
        app.outlet_pick_mode = False
        app.compressed_grid_shape = None
        app.compression_source_shape = None
        app.trimmed_grid_shape = None
        app.trim_source_shape = None
        app.trim_bounds = None
        app._gis_matrix_render_bbox = None
        app._gis_active_render_bbox = (0.0, 0.0, 80.0, 60.0)
        app._gis_matrix_meta = {
            "bbox": [0.0, 0.0, 80.0, 60.0],
            "rows_A": 6,
            "cols_B": 8,
            "cell_width": 10.0,
            "cell_height": 10.0,
        }
        app._gis_viewport_render_cache = {}
        app._gis_viewport_render_generation = 0
        app._gis_viewport_pending_keys = set()
        app._matrix_arrow_pil = None
        app._matrix_arrow_matrix_key = None
        app._matrix_arrow_pyramid = []
        app._detection_cache = None
        app._detection_cache_key = None
        app.rows_var = mock.Mock()
        app.selected_var = mock.Mock()
        app.status_var = mock.Mock()
        app.canvas = mock.Mock()
        app._ensure_grid = mock.Mock(return_value=True)
        app._update_matrix_preview = mock.Mock()
        app._update_flood_region_label = mock.Mock()
        app._update_gis_cell_size_label = mock.Mock()
        app.reset_view = mock.Mock()

        with mock.patch.object(gui.messagebox, "askyesno", return_value=True), mock.patch.object(
            gui.messagebox, "showerror"
        ) as showerror:
            app.trim_matrix_margins()

        showerror.assert_not_called()
        self.assertEqual((app.A, app.B), (4, 4))
        self.assertTrue(np.array_equal(app.current_matrix, original[1:5, 2:6]))
        self.assertTrue(np.array_equal(app.base_matrix, original[1:5, 2:6]))
        self.assertEqual(app.img_bgr.shape[:2], (40, 40))
        self.assertEqual(app.outlet_cell, (2, 2))
        self.assertEqual(app.outlet_cells, {(2, 2)})
        self.assertEqual(app.selected_cell, (1, 1))
        self.assertEqual(app.arrow_boxes, [(8, 8, 32, 32)])
        self.assertEqual(app._flood_cell_mask.shape, (4, 4))
        self.assertEqual(app.roi_cell_mask.shape, (4, 4))
        self.assertEqual(app.roi_mask.shape, (40, 40))
        self.assertEqual(app.roi_polygon_points[0], (5.0, 5.0))
        self.assertEqual(app.trim_source_shape, (6, 8))
        self.assertEqual(app.trim_bounds, (1, 5, 2, 6))
        self.assertEqual(app._gis_matrix_render_bbox, (20.0, 10.0, 60.0, 50.0))
        self.assertEqual(app._gis_matrix_meta["bbox"], [20.0, 10.0, 60.0, 50.0])
        self.assertEqual(app._current_trim_bounds(padding=1), (0, 4, 0, 4))

    def test_repeated_wheel_zoom_keeps_anchor_image_point_fixed(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app.matrix_arrow_view = False
        app.img_pil = gui.Image.new("RGB", (1000, 500))
        app._scale = 0.8
        app._fit_scale = 0.8
        app._zoom = 1.0
        app._display_origin = (0.0, 100.0)
        app._display_origin_initialized = True
        app._display_size = (800, 400)
        app.source_kind = "gis_project"
        app.canvas = mock.Mock()
        app.canvas.winfo_width.return_value = 800
        app.canvas.winfo_height.return_value = 600
        app._queue_render = mock.Mock()
        event = mock.Mock(x=625.0, y=275.0, delta=120)
        expected = (
            (event.x - app._display_origin[0]) / app._scale,
            (event.y - app._display_origin[1]) / app._scale,
        )

        app.on_mouse_wheel(event)
        app.on_mouse_wheel(event)

        actual = (
            (event.x - app._display_origin[0]) / app._scale,
            (event.y - app._display_origin[1]) / app._scale,
        )
        self.assertAlmostEqual(actual[0], expected[0])
        self.assertAlmostEqual(actual[1], expected[1])

    def test_gis_overview_resolution_and_zoom_range_are_extended(self) -> None:
        self.assertGreaterEqual(gui.GIS_OVERVIEW_RENDER_SIZE, 4096)
        self.assertGreaterEqual(gui.ZOOM_MAX, 64.0)

    def test_world_bbox_from_image_rect_maps_visible_pixels_to_gis_coordinates(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app._gis_map_bbox = (0.0, 0.0, 100.0, 100.0)
        app.img_pil = gui.Image.new("RGB", (100, 100))

        bbox = app._world_bbox_from_image_rect((25.0, 25.0, 50.0, 50.0))

        self.assertEqual(bbox, (25.0, 50.0, 50.0, 75.0))

    def test_full_image_edge_rect_maps_exactly_to_source_bbox(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app.img_pil = gui.Image.new("RGB", (320, 180))
        source_bbox = (100.0, 200.0, 740.0, 560.0)

        bbox = app._world_bbox_from_image_rect_for_bbox((0.0, 0.0, 320.0, 180.0), source_bbox)

        self.assertEqual(bbox, source_bbox)

    def test_rotated_active_render_extent_matches_matrix_extent(self) -> None:
        records = [gis_matrix.GISPolyline(parts=[[(8.0, 5.0), (12.0, 5.0)]])]
        source_bbox = (0.0, 0.0, 20.0, 10.0)
        result = gis_matrix.matrix_from_loaded_polylines(
            records,
            rows=10,
            bbox=source_bbox,
            rotation_degrees=90.0,
        )
        app = object.__new__(gui.NFMATApp)
        app._gis_active_records = records
        app._gis_active_bbox = source_bbox
        app._gis_active_render_records = None
        app._gis_active_render_bbox = None
        app._gis_active_render_spatial_index = None
        app._gis_active_render_rotation = None
        app._gis_viewport_render_cache = {}
        app._gis_viewport_render_generation = 0
        app._gis_viewport_pending_keys = set()
        app.rotation_var = mock.Mock()
        app.rotation_var.get.return_value = 90.0

        app._refresh_active_gis_render_geometry()

        matrix_bbox = tuple(float(value) for value in result.meta["bbox"])
        for actual, expected in zip(app._gis_active_render_bbox, matrix_bbox):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(matrix_bbox, (5.0, -5.0, 15.0, 15.0)):
            self.assertAlmostEqual(actual, expected)

    def test_gis_result_bbox_is_available_before_display_refresh(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app.trimmed_grid_shape = None
        app.trim_source_shape = None
        app.trim_bounds = None
        app._gis_matrix_render_bbox = None
        app._gis_matrix_meta = None
        app.source_kind = "gis"
        app._gis_active_records = []
        expected_bbox = (5.0, -5.0, 15.0, 15.0)
        app._set_display_image_from_bgr = mock.Mock(
            side_effect=lambda _image: self.assertEqual(app._gis_matrix_render_bbox, expected_bbox)
        )
        app._refresh_active_gis_render_geometry = mock.Mock()
        app._clear_compressed_grid_shape = mock.Mock()
        app.rows_var = mock.Mock()
        app.A, app.B = 2, 3
        app._sync_grid_shape = mock.Mock()
        app._clear_roi = mock.Mock()
        app._invalidate_detection_cache = mock.Mock()
        app._rebuild_flood_cell_mask = mock.Mock()
        app._invalidate_matrix_arrow_cache = mock.Mock()
        app._update_gis_cell_size_label = mock.Mock()
        app._default_selected_cell = mock.Mock(return_value=(0, 0))
        app._set_selected_cell = mock.Mock()
        app.reset_view = mock.Mock()
        app.matrix_arrow_button = mock.Mock()
        result = mock.Mock()
        result.image_bgr = np.zeros((20, 30, 3), dtype=np.uint8)
        result.matrix = np.zeros((2, 3), dtype=np.uint8)
        result.confidence = np.zeros((2, 3), dtype=np.float32)
        result.occ = np.zeros((2, 3), dtype=np.uint8)
        result.path_occ = np.zeros((2, 3), dtype=np.uint8)
        result.support_mask = np.zeros((2, 3), dtype=np.uint8)
        result.meta = {"bbox": list(expected_bbox)}

        app._apply_gis_result_to_state(result)

        self.assertEqual(app._gis_matrix_render_bbox, expected_bbox)
        app._refresh_active_gis_render_geometry.assert_called_once_with()

    def test_gis_viewport_render_uses_intersecting_vector_features(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app.matrix_arrow_view = False
        app.source_kind = "gis_project"
        app._zoom = gui.GIS_VIEWPORT_RENDER_ZOOM_THRESHOLD
        app._gis_map_bbox = (0.0, 0.0, 100.0, 100.0)
        app.img_pil = gui.Image.new("RGB", (100, 100))
        inside = gis_matrix.GISPolyline(parts=[[(30.0, 60.0), (40.0, 60.0)]])
        outside = gis_matrix.GISPolyline(parts=[[(90.0, 10.0), (95.0, 10.0)]])
        app._gis_layers = {
            "pipe": {
                "kind": "polyline",
                "visible": True,
                "geometry": [inside, outside],
                "style": {"color": (1, 2, 3), "opacity": 1.0, "line_width": 1.0},
                "spatial_index": gis_matrix.PolylineSpatialIndex.build([inside, outside]),
            }
        }
        app._gis_viewport_render_cache = {}

        with mock.patch.object(gui, "render_gis_layers", return_value=np.zeros((64, 64, 3), dtype=np.uint8)) as render:
            image = app._render_gis_viewport_image((25.0, 25.0, 50.0, 50.0), (80, 60))

        self.assertEqual(image.size, (80, 60))
        _polyline_layers, _boundary_layers, bbox = render.call_args.args[:3]
        self.assertEqual(bbox, (25.0, 50.0, 50.0, 75.0))
        layer_stack = render.call_args.kwargs["layer_stack"]
        self.assertEqual(len(layer_stack), 1)
        self.assertEqual(layer_stack[0][0], "polyline")
        self.assertEqual(layer_stack[0][1][0], [inside])
        self.assertEqual(render.call_args.kwargs["output_size"], (108, 81))

    def test_reconstructed_gis_viewport_render_uses_active_spatial_index(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app.matrix_arrow_view = False
        app.source_kind = "gis"
        app._zoom = gui.GIS_VIEWPORT_RENDER_ZOOM_THRESHOLD
        app._gis_active_bbox = (0.0, 0.0, 100.0, 100.0)
        app.img_pil = gui.Image.new("RGB", (100, 100))
        inside = gis_matrix.GISPolyline(parts=[[(30.0, 60.0), (40.0, 60.0)]])
        outside = gis_matrix.GISPolyline(parts=[[(90.0, 10.0), (95.0, 10.0)]])
        app._gis_active_records = [inside, outside]
        app._gis_active_spatial_index = gis_matrix.PolylineSpatialIndex.build(app._gis_active_records)
        app._gis_viewport_render_cache = {}

        with mock.patch.object(gui, "render_gis_layers", return_value=np.zeros((64, 64, 3), dtype=np.uint8)) as render:
            image = app._render_gis_viewport_image((25.0, 25.0, 50.0, 50.0), (80, 60))

        self.assertEqual(image.size, (80, 60))
        self.assertEqual(render.call_args.args[2], (25.0, 50.0, 50.0, 75.0))
        layer_stack = render.call_args.kwargs["layer_stack"]
        self.assertEqual(layer_stack[0][0], "polyline")
        self.assertEqual(layer_stack[0][1][0], [inside])
        self.assertEqual(render.call_args.kwargs["output_size"], (108, 81))

    def test_forced_gis_viewport_render_bypasses_zoom_threshold_after_layer_change(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app.matrix_arrow_view = False
        app.source_kind = "gis_project"
        app._zoom = 1.0
        app._gis_force_viewport_render = True
        app._gis_sync_viewport_once = True
        app._gis_map_bbox = (0.0, 0.0, 100.0, 100.0)
        app.img_pil = gui.Image.new("RGB", (100, 100))
        inside = gis_matrix.GISPolyline(parts=[[(30.0, 60.0), (40.0, 60.0)]])
        app._gis_layers = {
            "pipe": {
                "kind": "polyline",
                "visible": True,
                "geometry": [inside],
                "style": {"color": (1, 2, 3), "opacity": 1.0, "line_width": 1.0},
                "spatial_index": gis_matrix.PolylineSpatialIndex.build([inside]),
            }
        }
        app._gis_viewport_render_cache = {}

        with mock.patch.object(gui, "render_gis_layers", return_value=np.zeros((32, 32, 3), dtype=np.uint8)) as render:
            image = app._render_gis_viewport_image((25.0, 25.0, 50.0, 50.0), (80, 60))

        self.assertEqual(image.size, (80, 60))
        self.assertFalse(app._gis_sync_viewport_once)
        self.assertEqual(render.call_args.args[2], (25.0, 50.0, 50.0, 75.0))
        self.assertEqual(render.call_args.kwargs["output_size"], (108, 81))

    def test_forced_gis_viewport_render_shows_blank_when_all_layers_hidden(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app.matrix_arrow_view = False
        app.source_kind = "gis_project"
        app._zoom = 1.0
        app._gis_force_viewport_render = True
        app._gis_sync_viewport_once = True
        app._gis_map_bbox = (0.0, 0.0, 100.0, 100.0)
        app.img_pil = gui.Image.new("RGB", (100, 100))
        app._gis_layers = {
            "pipe": {
                "kind": "polyline",
                "visible": False,
                "geometry": [],
                "style": {"color": (1, 2, 3), "opacity": 1.0, "line_width": 1.0},
            }
        }
        app._gis_viewport_render_cache = {}

        image = app._render_gis_viewport_image((0.0, 0.0, 100.0, 100.0), (80, 60))

        self.assertEqual(image.size, (80, 60))
        self.assertFalse(app._gis_sync_viewport_once)
        self.assertTrue(np.all(np.asarray(image) == 250))

    def test_gis_render_line_width_is_capped_for_visibility(self) -> None:
        self.assertEqual(gui.NFMATApp._gis_render_line_width(12.0, 5.0), 2.0)
        self.assertEqual(gui.NFMATApp._gis_render_line_width(0.1, 0.1), 0.5)

    def test_boundary_hover_overlay_is_thicker_than_selected_overlay(self) -> None:
        selected_inner, selected_halo = gui.NFMATApp._gis_boundary_overlay_widths(1.0, 16.0, hover=False)
        hover_inner, hover_halo = gui.NFMATApp._gis_boundary_overlay_widths(1.0, 16.0, hover=True)

        self.assertGreater(hover_inner, selected_inner)
        self.assertGreater(hover_halo, selected_halo)
        self.assertEqual((hover_inner, hover_halo), (4.0, 7.0))

    def test_non_role_layers_default_to_half_opacity(self) -> None:
        pipe_style = gui.NFMATApp._default_gis_layer_style("polyline", 0)
        boundary_style = gui.NFMATApp._default_gis_layer_style("boundary", 0)

        pipe = gui.NFMATApp._apply_non_role_default_opacity(pipe_style, "polyline", False)
        boundary = gui.NFMATApp._apply_non_role_default_opacity(boundary_style, "boundary", False)

        self.assertEqual(pipe["opacity"], 0.5)
        self.assertEqual(boundary["fill_opacity"], 0.5)
        self.assertEqual(boundary["outline_opacity"], 0.5)

    def test_flood_layer_filename_detection_uses_separate_gis_names(self) -> None:
        self.assertTrue(gui.NFMATApp._looks_like_flood_layer_name("2010침수지역"))
        self.assertTrue(gui.NFMATApp._looks_like_flood_layer_name("Flood_Area"))
        self.assertFalse(gui.NFMATApp._looks_like_flood_layer_name("서울_행정경계"))

    def test_binary_flood_mask_resize_preserves_any_covered_source_cell(self) -> None:
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, 0] = 1
        mask[3, 3] = 1

        resized = gui.NFMATApp._resize_binary_mask_any(mask, 2, 2)

        self.assertEqual(resized.tolist(), [[1, 0], [0, 1]])

    def test_world_point_from_canvas_event_uses_current_gis_extent(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app.matrix_arrow_view = False
        app.img_pil = gui.Image.new("RGB", (201, 101))
        app._gis_map_bbox = (100.0, 200.0, 300.0, 400.0)
        app._display_origin = (10.0, 20.0)
        app._scale = 2.0
        event = mock.Mock(x=210.0, y=120.0)

        point = app._world_point_from_event(event)

        self.assertEqual(point, (200.0, 300.0))

    def test_gis_project_single_click_immediately_clips_clicked_boundary(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app.matrix_arrow_view = False
        app.source_kind = "gis_project"
        app.lasso_mode = False
        app.status_var = mock.Mock()
        app._world_point_from_event = mock.Mock(return_value=(5.0, 5.0))
        app._boundary_at_world_point = mock.Mock(return_value=("boundary", 3))
        app._clip_boundary_feature_to_matrix = mock.Mock()

        app.on_canvas_click(mock.Mock())

        app._clip_boundary_feature_to_matrix.assert_called_once_with("boundary", 3)

    def test_boundary_matrix_worker_prepares_result_off_main_thread(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app._gis_boundary_job_queue = gui.queue.Queue()
        records = [gis_matrix.GISPolyline(parts=[[(0.0, 5.0), (10.0, 5.0)]])]
        payload = {
            "pipe_records": records,
            "rings": [[(0.0, 0.0), (6.0, 0.0), (6.0, 10.0), (0.0, 10.0), (0.0, 0.0)]],
            "spatial_index": gis_matrix.PolylineSpatialIndex.build(records),
            "rows": 10,
            "source_meta": {},
            "rotation_degrees": 0.0,
        }

        app._boundary_matrix_worker(7, payload)
        generation, result = app._gis_boundary_job_queue.get_nowait()

        self.assertEqual(generation, 7)
        self.assertIsInstance(result, dict)
        self.assertGreater(int(np.count_nonzero(result["prepared_result"].matrix)), 0)

    def test_pipe_layer_matrix_worker_prepares_full_pipe_result(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app._gis_pipe_job_queue = gui.queue.Queue()
        records = [gis_matrix.GISPolyline(parts=[[(0.0, 5.0), (10.0, 5.0)]])]
        payload = {
            "records": records,
            "bbox": (0.0, 0.0, 10.0, 10.0),
            "rows": 10,
            "source_meta": {},
            "rotation_degrees": 0.0,
        }

        app._pipe_layer_matrix_worker(9, payload)
        generation, result = app._gis_pipe_job_queue.get_nowait()

        self.assertEqual(generation, 9)
        self.assertIsInstance(result, dict)
        self.assertGreater(int(np.count_nonzero(result["prepared_result"].matrix)), 0)

    def test_convert_selected_pipe_layer_to_matrix_uses_selected_polyline_without_boundary(self) -> None:
        app = object.__new__(gui.NFMATApp)
        records = [gis_matrix.GISPolyline(parts=[[(0.0, 5.0), (10.0, 5.0)]])]
        pipe_layer = {
            "kind": "polyline",
            "name": "pipe_only",
            "path": "pipe.shp",
            "bbox": (0.0, 0.0, 10.0, 10.0),
            "meta": {},
            "geometry": records,
        }
        result = gis_matrix.matrix_from_loaded_polylines(records, rows=10, bbox=pipe_layer["bbox"], source_meta={})
        app._gis_layers = {"pipe": pipe_layer}
        app._gis_pipe_layer_id = None
        app._gis_flood_layer_id = None
        app._gis_flood_layer_ids = set()
        app.rows_var = mock.Mock()
        app.rows_var.get.return_value = 10
        app.rotation_var = mock.Mock()
        app.rotation_var.get.return_value = 0.0
        app._selected_gis_layer_id = mock.Mock(return_value="pipe")
        app._gis_pipe_result_cache = {
            app._pipe_layer_matrix_cache_key("pipe", pipe_layer, 10, None, None): (records, pipe_layer["bbox"], result)
        }
        app._sync_default_role_opacities = mock.Mock()
        app._refresh_gis_layer_tree = mock.Mock()
        app._apply_clipped_records_to_matrix = mock.Mock()
        app.status_var = mock.Mock()

        app.convert_selected_pipe_layer_to_matrix()

        self.assertEqual(app._gis_pipe_layer_id, "pipe")
        app._apply_clipped_records_to_matrix.assert_called_once()
        self.assertIn("전체 관로", app._gis_boundary_value)

    def test_boundary_feature_label_uses_user_selected_field(self) -> None:
        boundary = gis_matrix.GISBoundary(
            rings=[[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]],
            attributes={"CODE": "11650", "KOR_NAME": "Seocho"},
        )

        field, value = gui.NFMATApp._boundary_feature_label(boundary, 0, "KOR_NAME")

        self.assertEqual((field, value), ("KOR_NAME", "Seocho"))

    def test_gis_hover_refreshes_only_boundary_overlay(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app.matrix_arrow_view = False
        app.source_kind = "gis_project"
        app.lasso_mode = False
        app._gis_hover_boundary = None
        app._gis_hover_after_id = None
        app._gis_hover_canvas_point = None
        app.status_var = mock.Mock()
        app.after = mock.Mock(side_effect=lambda _delay, callback: callback())
        app._world_point_from_canvas_xy = mock.Mock(return_value=(5.0, 5.0))
        app._boundary_at_world_point = mock.Mock(return_value=("boundary", 0))
        app._refresh_gis_boundary_overlay = mock.Mock()
        app._queue_render = mock.Mock()
        app._gis_layers = {
            "boundary": {
                "geometry": [
                    gis_matrix.GISBoundary(
                        rings=[[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 0.0)]],
                        attributes={"NAME": "Area"},
                    )
                ],
                "label_field": "NAME",
            }
        }

        app.on_canvas_motion(mock.Mock(x=10, y=20))

        app._refresh_gis_boundary_overlay.assert_called_once_with()
        app._queue_render.assert_not_called()

    def test_gis_pan_drag_moves_existing_canvas_without_rerender(self) -> None:
        for source_kind in ("gis_project", "gis"):
            with self.subTest(source_kind=source_kind):
                app = object.__new__(gui.NFMATApp)
                app.source_kind = source_kind
                app._pan_start = (10, 10)
                app._pan_last_canvas = (10, 10)
                app._pan_origin = (100.0, 200.0)
                app._display_origin = app._pan_origin
                app.canvas = mock.Mock()
                app._queue_render = mock.Mock()
                event = mock.Mock(x=25, y=35)

                app.on_pan_drag(event)

                self.assertEqual(app._display_origin, (115.0, 225.0))
                app.canvas.move.assert_called_once_with("all", 15, 25)
                app._queue_render.assert_not_called()

    def test_boundary_style_migrates_legacy_color_to_fill_and_outline(self) -> None:
        layer = {"kind": "boundary", "style": {"color": (10, 20, 30), "opacity": 0.4}}

        style = gui.NFMATApp._normalized_gis_layer_style(layer)

        self.assertEqual(style["fill_color"], (10, 20, 30))
        self.assertEqual(style["outline_color"], (10, 20, 30))
        self.assertEqual(style["fill_opacity"], 0.4)
        self.assertGreater(float(style["outline_opacity"]), 0.0)

    def test_return_to_gis_map_confirms_before_discarding_matrix_work(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app.source_kind = "gis"
        app._has_rotation_reset_risk = mock.Mock(return_value=True)

        with mock.patch.object(gui.messagebox, "askyesno", return_value=False) as ask:
            allowed = app._confirm_gis_map_reset()

        self.assertFalse(allowed)
        ask.assert_called_once()

    def test_return_to_full_gis_map_clears_transient_selection_state(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app._gis_layers = {
            "pipe": {"kind": "polyline"},
            "boundary": {"kind": "boundary"},
        }
        app._gis_pipe_layer_id = "pipe"
        app._gis_boundary_layer_id = "boundary"
        app._gis_flood_layer_id = None
        app._gis_render_generation = 4
        app._gis_render_polling = True
        app._gis_hover_boundary = ("boundary", 0)
        app._gis_last_clicked_boundary = ("boundary", 1)
        app._confirm_gis_map_reset = mock.Mock(return_value=True)
        app._clear_roi = mock.Mock()
        app._render_gis_project = mock.Mock()
        app.status_var = mock.Mock()
        app.canvas = mock.Mock()

        self.assertTrue(app.return_to_full_gis_map())

        self.assertEqual(app._gis_render_generation, 5)
        self.assertFalse(app._gis_render_polling)
        self.assertIsNone(app._gis_hover_boundary)
        self.assertIsNone(app._gis_last_clicked_boundary)
        app._clear_roi.assert_called_once_with()
        app._render_gis_project.assert_called_once_with()

    def test_return_to_full_gis_map_allows_pipe_only_project(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app._gis_layers = {"pipe": {"kind": "polyline"}}
        app._gis_pipe_layer_id = "pipe"
        app._gis_boundary_layer_id = None
        app._gis_render_generation = 1
        app._gis_render_polling = False
        app._gis_hover_boundary = None
        app._gis_last_clicked_boundary = None
        app._confirm_gis_map_reset = mock.Mock(return_value=True)
        app._clear_roi = mock.Mock()
        app._render_gis_project = mock.Mock()
        app.status_var = mock.Mock()
        app.canvas = mock.Mock()

        self.assertTrue(app.return_to_full_gis_map())

        app._render_gis_project.assert_called_once_with()
        app.status_var.set.assert_called_with("관로만 불러온 상태입니다. [선택 관로 → 행렬]로 전체 관로를 행렬로 변환할 수 있습니다.")

    def test_apply_gis_project_image_remembers_fast_overview_snapshot(self) -> None:
        app = object.__new__(gui.NFMATApp)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        bbox = (0.0, 1.0, 2.0, 3.0)
        app._set_display_image_from_bgr = mock.Mock()
        app._reset_image_matrix_state = mock.Mock()
        app.reset_view = mock.Mock()
        app._queue_render = mock.Mock()
        app._update_matrix_preview = mock.Mock()

        app._apply_gis_project_image(image, bbox, "map.shp", reset_matrix=False)

        self.assertIs(app._gis_last_overview_image, image)
        self.assertEqual(app._gis_last_overview_bbox, bbox)
        self.assertEqual(app._gis_last_overview_path, "map.shp")
        self.assertEqual(app.source_kind, "gis_project")
        app._reset_image_matrix_state.assert_not_called()

    def test_large_gis_render_shows_last_overview_before_background_refresh(self) -> None:
        app = object.__new__(gui.NFMATApp)
        previous_image = np.zeros((8, 8, 3), dtype=np.uint8)
        previous_bbox = (0.0, 0.0, 10.0, 10.0)
        app._gis_layers = {
            "pipe": {
                "kind": "polyline",
                "geometry": [],
                "path": "pipe.shp",
                "bbox": previous_bbox,
                "point_count": gui.GIS_BACKGROUND_RENDER_POINT_THRESHOLD,
            }
        }
        app._gis_render_cache = {}
        app._gis_render_generation = 0
        app._gis_render_polling = False
        app._gis_last_overview_image = previous_image
        app._gis_last_overview_bbox = previous_bbox
        app._gis_last_overview_path = "pipe.shp"
        app._gis_overview_line_scale = mock.Mock(return_value=1.0)
        app._apply_gis_project_image = mock.Mock()
        app.status_var = mock.Mock()
        app.after = mock.Mock()
        thread = mock.Mock()

        with mock.patch.object(gui.threading, "Thread", return_value=thread):
            app._render_gis_project(reset_matrix=True)

        app._apply_gis_project_image.assert_called_once_with(previous_image, previous_bbox, "pipe.shp", True)
        self.assertTrue(app._gis_render_polling)
        thread.start.assert_called_once_with()
        app.after.assert_called_once_with(30, app._poll_gis_render_result)

    def test_interactive_large_gis_render_debounces_overview_worker(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app._gis_layers = {
            "pipe": {
                "kind": "polyline",
                "geometry": [],
                "path": "pipe.shp",
                "bbox": (0.0, 0.0, 10.0, 10.0),
                "point_count": gui.GIS_BACKGROUND_RENDER_POINT_THRESHOLD,
                "visible": True,
            }
        }
        app._gis_render_cache = {}
        app._gis_render_generation = 0
        app._gis_render_polling = False
        app._gis_deferred_overview_after_id = None
        app._gis_last_overview_image = None
        app._gis_last_overview_bbox = None
        app._gis_last_overview_path = None
        app.source_kind = "gis_project"
        app.img_pil = gui.Image.new("RGB", (8, 8))
        app._gis_overview_line_scale = mock.Mock(return_value=1.0)
        app._queue_render = mock.Mock()
        app.status_var = mock.Mock()
        app.after = mock.Mock(return_value="deferred-job")
        thread = mock.Mock()

        with mock.patch.object(gui.threading, "Thread", return_value=thread):
            app._render_gis_project(reset_matrix=False)
            thread.start.assert_not_called()
            callback = app.after.call_args.args[1]
            self.assertEqual(app.after.call_args.args[0], gui.GIS_INTERACTIVE_OVERVIEW_DEBOUNCE_MS)

            callback()

        thread.start.assert_called_once_with()
        self.assertIsNone(app._gis_deferred_overview_after_id)

    def test_invalidate_gis_render_cache_cancels_deferred_overview_worker(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app._gis_deferred_overview_after_id = "deferred-job"
        app.after_cancel = mock.Mock()
        app._gis_render_generation = 0
        app._gis_render_polling = True
        app._gis_boundary_job_generation = 0
        app._gis_boundary_job_polling = True
        app._gis_render_cache = {}
        app._gis_last_overview_image = None
        app._gis_last_overview_bbox = None
        app._gis_last_overview_path = None
        app._gis_viewport_render_cache = {}
        app._gis_viewport_render_generation = 0
        app._gis_viewport_render_polling = True
        app._gis_viewport_pending_keys = {"pending"}
        app._gis_boundary_overlay_cache = {}
        app._gis_hover_boundary = ("boundary", 0)
        app.source_kind = None

        app._invalidate_gis_render_cache()

        app.after_cancel.assert_called_once_with("deferred-job")
        self.assertIsNone(app._gis_deferred_overview_after_id)
        self.assertFalse(app._gis_render_polling)
        self.assertFalse(app._gis_viewport_pending_keys)

    def test_async_gis_poll_uses_current_generation_even_if_stale_result_arrives_last(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app._gis_render_result_queue = gui.queue.Queue()
        app._gis_render_generation = 2
        app._gis_render_polling = True
        app._gis_render_cache = {}
        app._apply_gis_project_image = mock.Mock()
        app.status_var = mock.Mock()
        app.after = mock.Mock()
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        bbox = (0.0, 0.0, 1.0, 1.0)
        app._gis_render_result_queue.put((2, ("current",), image, bbox, "current.shp", True))
        app._gis_render_result_queue.put((1, ("stale",), image, bbox, "stale.shp", True))

        app._poll_gis_render_result()

        app._apply_gis_project_image.assert_called_once_with(image, bbox, "current.shp", True)
        self.assertFalse(app._gis_render_polling)
        app.after.assert_not_called()

    def test_layer_roles_distinguish_boundary_and_separate_flood_layer(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app._gis_pipe_layer_id = "pipe"
        app._gis_boundary_layer_id = "district"
        app._gis_flood_layer_id = "flood"

        self.assertEqual(app._gis_layer_role("district"), "경계")
        self.assertEqual(app._gis_layer_role("flood"), "침수")

    def test_set_selected_flood_layer_assigns_separate_polygon_role(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app._gis_layers = {
            "pipe": {"kind": "polyline", "path": "pipe.shp", "bbox": (0.0, 0.0, 10.0, 10.0)},
            "district": {"kind": "boundary", "path": "district.shp", "bbox": (0.0, 0.0, 10.0, 10.0)},
            "flood": {
                "kind": "boundary",
                "name": "flood_area",
                "path": "flood.shp",
                "bbox": (0.0, 0.0, 10.0, 10.0),
                "geometry": [],
            },
        }
        app._gis_pipe_layer_id = "pipe"
        app._gis_boundary_layer_id = "district"
        app._gis_flood_layer_id = None
        app._flood_cell_mask = np.ones((2, 2), dtype=np.uint8)
        app.source_kind = None
        app.current_matrix = None
        app._selected_gis_layer_id = mock.Mock(return_value="flood")
        app._confirm_pipe_polygon_compatibility = mock.Mock(return_value=True)
        app._sync_default_role_opacities = mock.Mock()
        app._invalidate_gis_render_cache = mock.Mock()
        app._refresh_gis_layer_tree = mock.Mock()
        app._update_flood_region_label = mock.Mock()
        app.status_var = mock.Mock()

        app.set_selected_flood_layer()

        self.assertEqual(app._gis_flood_layer_id, "flood")
        self.assertIsNone(app._flood_cell_mask)
        app._confirm_pipe_polygon_compatibility.assert_called_once()

    def test_set_selected_flood_layer_toggles_one_layer_without_clearing_others(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app._gis_layers = {
            "pipe": {"kind": "polyline", "path": "pipe.shp", "bbox": (0.0, 0.0, 10.0, 10.0)},
            "flood_a": {"kind": "boundary", "name": "A", "path": "a.shp", "bbox": (0.0, 0.0, 10.0, 10.0)},
            "flood_b": {"kind": "boundary", "name": "B", "path": "b.shp", "bbox": (0.0, 0.0, 10.0, 10.0)},
        }
        app._gis_pipe_layer_id = "pipe"
        app._gis_boundary_layer_id = None
        app._gis_flood_layer_id = "flood_a"
        app._gis_flood_layer_ids = {"flood_a", "flood_b"}
        app._flood_cell_mask = None
        app.source_kind = None
        app.current_matrix = None
        app._selected_gis_layer_id = mock.Mock(return_value="flood_a")
        app._confirm_pipe_polygon_compatibility = mock.Mock(return_value=True)
        app._sync_default_role_opacities = mock.Mock()
        app._invalidate_gis_render_cache = mock.Mock()
        app._refresh_gis_layer_tree = mock.Mock()
        app._update_flood_region_label = mock.Mock()
        app.status_var = mock.Mock()

        app.set_selected_flood_layer()

        self.assertEqual(app._gis_flood_layer_ids, {"flood_b"})
        self.assertEqual(app._gis_flood_layer_id, "flood_b")

    def test_flood_mask_uses_all_features_from_separate_flood_layer(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app.A = 10
        app.B = 10
        app.current_matrix = np.zeros((10, 10), dtype=np.uint8)
        app.current_matrix[5, :] = 1
        app.support_mask = (app.current_matrix > 0).astype(np.uint8)
        app._gis_flood_layer_id = "flood"
        app._gis_active_bbox = (0.0, 0.0, 10.0, 10.0)
        app._gis_layers = {
            "district": {
                "kind": "boundary",
                "geometry": [
                    gis_matrix.GISBoundary(
                        rings=[[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]],
                        attributes={},
                    )
                ],
            },
            "flood": {
                "kind": "boundary",
                "geometry": [
                    gis_matrix.GISBoundary(
                        rings=[[(0.0, 0.0), (2.9, 0.0), (2.9, 10.0), (0.0, 10.0), (0.0, 0.0)]],
                        attributes={},
                    ),
                    gis_matrix.GISBoundary(
                        rings=[[(7.1, 0.0), (10.0, 0.0), (10.0, 10.0), (7.1, 10.0), (7.1, 0.0)]],
                        attributes={},
                    )
                ],
            }
        }
        app.rotation_var = mock.Mock()
        app.rotation_var.get.return_value = 0.0
        app._update_flood_region_label = mock.Mock()

        app._rebuild_flood_cell_mask()

        self.assertGreater(int(np.count_nonzero(app._flood_cell_mask)), 0)
        self.assertEqual(int(np.count_nonzero(app._flood_cell_mask[:5, :])), 0)
        self.assertEqual(int(np.count_nonzero(app._flood_cell_mask[:, 4:6])), 0)
        self.assertGreater(int(np.count_nonzero(app._flood_cell_mask[:, 7:])), 0)

    def test_flood_mask_unions_multiple_selected_flood_layers(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app.A = 10
        app.B = 10
        app.current_matrix = np.ones((10, 10), dtype=np.uint8)
        app.support_mask = np.ones((10, 10), dtype=np.uint8)
        app._gis_flood_layer_id = "left"
        app._gis_flood_layer_ids = {"left", "right"}
        app._gis_active_bbox = (0.0, 0.0, 10.0, 10.0)
        app._gis_layers = {
            "left": {
                "kind": "boundary",
                "geometry": [
                    gis_matrix.GISBoundary(
                        rings=[[(0.0, 0.0), (2.0, 0.0), (2.0, 10.0), (0.0, 10.0), (0.0, 0.0)]],
                        attributes={},
                    )
                ],
            },
            "right": {
                "kind": "boundary",
                "geometry": [
                    gis_matrix.GISBoundary(
                        rings=[[(8.0, 0.0), (10.0, 0.0), (10.0, 10.0), (8.0, 10.0), (8.0, 0.0)]],
                        attributes={},
                    )
                ],
            },
        }
        app.rotation_var = mock.Mock()
        app.rotation_var.get.return_value = 0.0
        app._update_flood_region_label = mock.Mock()

        app._rebuild_flood_cell_mask()

        self.assertGreater(int(np.count_nonzero(app._flood_cell_mask[:, :3])), 0)
        self.assertEqual(int(np.count_nonzero(app._flood_cell_mask[:, 4:6])), 0)
        self.assertGreater(int(np.count_nonzero(app._flood_cell_mask[:, 7:])), 0)


class CellTrainingTests(unittest.TestCase):
    def test_extract_features_and_labels_uses_manual_labels_by_default(self) -> None:
        img = np.full((64, 64, 3), 255, dtype=np.uint8)
        A = 4
        B = ct.derive_grid_B(A, img.shape[1], img.shape[0])
        reference = np.zeros((A, B), dtype=np.uint8)
        reference[1, 1] = 1

        X_list, X_patch_list, y_list = ct.extract_features_and_labels(
            img,
            A,
            corrected_labels={(1, 1): 1},
            reference_matrix=reference,
            context_radius=ct.CONTEXT_RADIUS,
        )

        self.assertGreaterEqual(len(X_list), 1)
        self.assertEqual(len(X_list), len(X_patch_list))
        self.assertIn(1, y_list)
        self.assertNotIn(0, y_list)
        self.assertEqual(tuple(np.asarray(X_patch_list[0]).shape), ct.expected_patch_shape())

    def test_extract_features_and_labels_can_opt_into_reliable_zero_context(self) -> None:
        img = np.full((64, 64, 3), 255, dtype=np.uint8)
        A = 4
        B = ct.derive_grid_B(A, img.shape[1], img.shape[0])
        reference = np.zeros((A, B), dtype=np.uint8)
        reference[1, 1] = 1

        X_list, X_patch_list, y_list = ct.extract_features_and_labels(
            img,
            A,
            corrected_labels={(1, 1): 1},
            reference_matrix=reference,
            context_radius=ct.CONTEXT_RADIUS,
            include_reliable_context=True,
        )

        self.assertGreater(len(X_list), 1)
        self.assertEqual(len(X_list), len(X_patch_list))
        self.assertIn(1, y_list)
        self.assertIn(0, y_list)
        self.assertEqual(tuple(np.asarray(X_patch_list[0]).shape), ct.expected_patch_shape())

    def test_collect_corrections_keeps_explicit_zero_input(self) -> None:
        app = gui.NFMATApp.__new__(gui.NFMATApp)
        app.base_matrix = np.zeros((2, 2), dtype=np.uint8)
        app.current_matrix = np.zeros((2, 2), dtype=np.uint8)
        app.user_edit_mask = np.zeros((2, 2), dtype=np.uint8)
        app.user_edit_mask[0, 1] = 1

        corrections = gui.NFMATApp._collect_corrections(app)

        self.assertEqual(corrections, {(0, 1): 0})

    def test_coverage_mode_requires_all_five_classes(self) -> None:
        y = np.array([1, 2, 3, 4] * 25, dtype=np.int64)

        mode, reason = ct._coverage_mode(y)

        self.assertEqual(mode, "restricted")
        self.assertIn("클래스 종류가 4개", reason)

    def test_predict_with_model_with_meta_can_apply_zero_override(self) -> None:
        expected_dim = ct.expected_feature_dim(ct.CONTEXT_RADIUS)
        checkpoint = {
            "model_kind": "knn",
            "model_schema_version": int(ct.MODEL_SCHEMA_VERSION),
            "input_dim": int(expected_dim),
            "feature_mean": np.zeros((1, expected_dim), dtype=np.float32),
            "feature_std": np.ones((1, expected_dim), dtype=np.float32),
            "num_classes": int(ct.NUM_CLASSES),
            "schema_version": int(ct.DATASET_SCHEMA_VERSION),
            "device_label": "cpu",
            "coverage_mode": "restricted",
            "coverage_reason": "",
            "reference_features": np.zeros((1, expected_dim + int(np.prod(ct.expected_patch_shape()))), dtype=np.float32),
            "reference_labels": np.array([0], dtype=np.int64),
            "class_counts": np.array([5, 0, 0, 0, 0], dtype=np.int32),
            "distance_scale": 1.0,
            "distance_limit": 1.0,
            "knn_k": 1,
            "patch_channels": ct.PATCH_CHANNELS,
            "patch_size": ct.RAW_PATCH_SIZE,
        }
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        morph_tuple = (
            np.array([[1]], dtype=np.uint8),
            np.array([[1]], dtype=np.uint8),
            np.array([[0.01]], dtype=np.float32),
            np.array([[0.0]], dtype=np.float32),
            np.array([[0.0]], dtype=np.float32),
            np.array([[0.0]], dtype=np.float32),
            np.array([[0.10]], dtype=np.float32),
        )
        global_maps = {
            "row_occ_density": np.array([1.0], dtype=np.float32),
            "col_occ_density": np.array([1.0], dtype=np.float32),
            "row_head_density": np.array([0.0], dtype=np.float32),
            "col_head_density": np.array([0.0], dtype=np.float32),
            "component_ratio": np.array([[0.0]], dtype=np.float32),
            "direction_run": np.array([[0.0]], dtype=np.float32),
            "occ_degree": np.array([[0.0]], dtype=np.float32),
            "incoming_degree": np.array([[0.0]], dtype=np.float32),
        }

        with mock.patch("cell_training.os.path.exists", return_value=True), mock.patch(
            "cell_training.Path.read_bytes",
            return_value=b"MODEL\n",
        ), mock.patch(
            "cell_training.torch.load",
            return_value=checkpoint,
        ), mock.patch.object(
            ct,
            "_compute_morphological_features",
            return_value=morph_tuple,
        ), mock.patch.object(
            ct,
            "_compute_global_context_maps",
            return_value=global_maps,
        ), mock.patch.object(
            ct,
            "_cell_feature_vector",
            return_value=[0.0] * expected_dim,
        ), mock.patch.object(
            ct,
            "_compute_patch_source_maps",
            return_value=(
                np.zeros((10, 10), dtype=np.float32),
                np.zeros((10, 10), dtype=np.float32),
                np.zeros((10, 10), dtype=np.float32),
            ),
        ), mock.patch.object(
            ct,
            "_extract_cell_patch_tensor",
            return_value=np.zeros(ct.expected_patch_shape(), dtype=np.float32),
        ), mock.patch.object(
            ct,
            "_v2_predict_knn_probabilities",
            return_value=(np.array([0.95, 0.01, 0.01, 0.01, 0.02], dtype=np.float32), 0.05),
        ), mock.patch.object(
            ct,
            "_v2_query_closeness",
            return_value=(1.0, True),
        ), mock.patch.object(
            ct,
            "_v2_class_support_score",
            return_value=1.0,
        ):
            final_dir, _morph_dir, meta = ct.predict_with_model_with_meta(
                img,
                1,
                model_path="mock_model.pt",
            )

        self.assertEqual(int(final_dir[0, 0]), 0)
        self.assertEqual(int(meta["model_applied_mask"][0, 0]), 1)

    def test_predict_with_model_with_meta_does_not_fill_blank_without_patch_support(self) -> None:
        expected_dim = ct.expected_feature_dim(ct.CONTEXT_RADIUS)
        checkpoint = {
            "model_kind": "knn",
            "model_schema_version": int(ct.MODEL_SCHEMA_VERSION),
            "input_dim": int(expected_dim),
            "feature_mean": np.zeros((1, expected_dim), dtype=np.float32),
            "feature_std": np.ones((1, expected_dim), dtype=np.float32),
            "num_classes": int(ct.NUM_CLASSES),
            "schema_version": int(ct.DATASET_SCHEMA_VERSION),
            "device_label": "cpu",
            "coverage_mode": "normal",
            "coverage_reason": "",
            "reference_features": np.zeros((1, expected_dim + int(np.prod(ct.expected_patch_shape()))), dtype=np.float32),
            "reference_labels": np.array([1], dtype=np.int64),
            "class_counts": np.array([10, 10, 10, 10, 10], dtype=np.int32),
            "distance_scale": 1.0,
            "distance_limit": 1.0,
            "knn_k": 1,
            "patch_channels": ct.PATCH_CHANNELS,
            "patch_size": ct.RAW_PATCH_SIZE,
        }
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        morph_tuple = (
            np.array([[0]], dtype=np.uint8),
            np.array([[0]], dtype=np.uint8),
            np.array([[0.005]], dtype=np.float32),
            np.array([[0.0]], dtype=np.float32),
            np.array([[0.0]], dtype=np.float32),
            np.array([[0.0]], dtype=np.float32),
            np.array([[0.10]], dtype=np.float32),
        )
        global_maps = {
            "row_occ_density": np.array([0.0], dtype=np.float32),
            "col_occ_density": np.array([0.0], dtype=np.float32),
            "row_head_density": np.array([0.0], dtype=np.float32),
            "col_head_density": np.array([0.0], dtype=np.float32),
            "component_ratio": np.array([[0.0]], dtype=np.float32),
            "direction_run": np.array([[0.0]], dtype=np.float32),
            "occ_degree": np.array([[0.0]], dtype=np.float32),
            "incoming_degree": np.array([[0.0]], dtype=np.float32),
        }

        with mock.patch("cell_training.os.path.exists", return_value=True), mock.patch(
            "cell_training.Path.read_bytes",
            return_value=b"MODEL\n",
        ), mock.patch(
            "cell_training.torch.load",
            return_value=checkpoint,
        ), mock.patch.object(
            ct,
            "_compute_morphological_features",
            return_value=morph_tuple,
        ), mock.patch.object(
            ct,
            "_compute_global_context_maps",
            return_value=global_maps,
        ), mock.patch.object(
            ct,
            "_cell_feature_vector",
            return_value=[0.0] * expected_dim,
        ), mock.patch.object(
            ct,
            "_compute_patch_source_maps",
            return_value=(
                np.zeros((10, 10), dtype=np.float32),
                np.zeros((10, 10), dtype=np.float32),
                np.zeros((10, 10), dtype=np.float32),
            ),
        ), mock.patch.object(
            ct,
            "_extract_cell_patch_tensor",
            return_value=np.zeros(ct.expected_patch_shape(), dtype=np.float32),
        ), mock.patch.object(
            ct,
            "_v2_predict_knn_probabilities",
            return_value=(np.array([0.01, 0.95, 0.02, 0.01, 0.01], dtype=np.float32), 0.05),
        ), mock.patch.object(
            ct,
            "_v2_query_closeness",
            return_value=(1.0, True),
        ), mock.patch.object(
            ct,
            "_v2_class_support_score",
            return_value=1.0,
        ):
            final_dir, _morph_dir, meta = ct.predict_with_model_with_meta(
                img,
                1,
                model_path="mock_model.pt",
            )

        self.assertEqual(int(final_dir[0, 0]), 0)
        self.assertEqual(int(meta["model_applied_mask"][0, 0]), 0)

    def test_predict_with_model_with_meta_can_override_confident_wrong_direction(self) -> None:
        expected_dim = ct.expected_feature_dim(ct.CONTEXT_RADIUS)
        checkpoint = {
            "model_kind": "knn",
            "model_schema_version": int(ct.MODEL_SCHEMA_VERSION),
            "input_dim": int(expected_dim),
            "feature_mean": np.zeros((1, expected_dim), dtype=np.float32),
            "feature_std": np.ones((1, expected_dim), dtype=np.float32),
            "num_classes": int(ct.NUM_CLASSES),
            "schema_version": int(ct.DATASET_SCHEMA_VERSION),
            "device_label": "cpu",
            "coverage_mode": "normal",
            "coverage_reason": "",
            "reference_features": np.zeros((1, expected_dim + int(np.prod(ct.expected_patch_shape()))), dtype=np.float32),
            "reference_labels": np.array([2], dtype=np.int64),
            "class_counts": np.array([10, 18, 18, 18, 18], dtype=np.int32),
            "distance_scale": 1.0,
            "distance_limit": 1.0,
            "knn_k": 1,
            "patch_channels": ct.PATCH_CHANNELS,
            "patch_size": ct.RAW_PATCH_SIZE,
        }
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        morph_tuple = (
            np.array([[1]], dtype=np.uint8),
            np.array([[1]], dtype=np.uint8),
            np.array([[0.30]], dtype=np.float32),
            np.array([[0.15]], dtype=np.float32),
            np.array([[0.20]], dtype=np.float32),
            np.array([[0.0]], dtype=np.float32),
            np.array([[0.92]], dtype=np.float32),
        )
        global_maps = {
            "row_occ_density": np.array([1.0], dtype=np.float32),
            "col_occ_density": np.array([1.0], dtype=np.float32),
            "row_head_density": np.array([0.0], dtype=np.float32),
            "col_head_density": np.array([0.0], dtype=np.float32),
            "component_ratio": np.array([[0.0]], dtype=np.float32),
            "direction_run": np.array([[0.70]], dtype=np.float32),
            "occ_degree": np.array([[0.80]], dtype=np.float32),
            "incoming_degree": np.array([[0.50]], dtype=np.float32),
        }

        with mock.patch("cell_training.os.path.exists", return_value=True), mock.patch(
            "cell_training.Path.read_bytes",
            return_value=b"MODEL\n",
        ), mock.patch(
            "cell_training.torch.load",
            return_value=checkpoint,
        ), mock.patch.object(
            ct,
            "_compute_morphological_features",
            return_value=morph_tuple,
        ), mock.patch.object(
            ct,
            "_compute_global_context_maps",
            return_value=global_maps,
        ), mock.patch.object(
            ct,
            "_cell_feature_vector",
            return_value=[0.0] * expected_dim,
        ), mock.patch.object(
            ct,
            "_compute_patch_source_maps",
            return_value=(
                np.zeros((10, 10), dtype=np.float32),
                np.zeros((10, 10), dtype=np.float32),
                np.zeros((10, 10), dtype=np.float32),
            ),
        ), mock.patch.object(
            ct,
            "_extract_cell_patch_tensor",
            return_value=np.zeros(ct.expected_patch_shape(), dtype=np.float32),
        ), mock.patch.object(
            ct,
            "_v2_predict_knn_probabilities",
            return_value=(np.array([0.01, 0.10, 0.82, 0.04, 0.03], dtype=np.float32), 0.05),
        ), mock.patch.object(
            ct,
            "_v2_query_closeness",
            return_value=(0.95, True),
        ), mock.patch.object(
            ct,
            "_v2_class_support_score",
            return_value=0.85,
        ):
            final_dir, _morph_dir, meta = ct.predict_with_model_with_meta(
                img,
                1,
                model_path="mock_model.pt",
            )

        self.assertEqual(int(final_dir[0, 0]), 2)
        self.assertEqual(int(meta["model_applied_mask"][0, 0]), 1)

    def test_train_cell_model_preserves_zero_class_counts_and_metadata(self) -> None:
        expected_dim = ct.expected_feature_dim(ct.CONTEXT_RADIUS)
        rng = np.random.default_rng(7)
        X = rng.normal(size=(16, expected_dim)).astype(np.float32)
        X_patch = rng.random(size=(16, *ct.expected_patch_shape()), dtype=np.float32)
        y = np.array([0] * 8 + [1] * 8, dtype=np.int64)
        group_ids = np.arange(len(y), dtype=np.int64)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
            dataset_path = Path(tmpdir) / "cell_dataset.npz"
            model_path = Path(tmpdir) / "cell_model.pt"
            np.savez(
                dataset_path,
                X=X,
                X_patch=X_patch,
                y=y,
                group_ids=group_ids,
                schema_version=np.array([ct.DATASET_SCHEMA_VERSION], dtype=np.int32),
                num_classes=np.array([ct.NUM_CLASSES], dtype=np.int32),
                patch_channels=np.array([ct.PATCH_CHANNELS], dtype=np.int32),
                patch_size=np.array([ct.RAW_PATCH_SIZE], dtype=np.int32),
            )
            history = ct.train_cell_model(
                dataset_path=str(dataset_path),
                model_path=str(model_path),
                device_preference="cpu",
            )
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        self.assertEqual(history["model_kind"], "knn")
        self.assertEqual(int(np.asarray(checkpoint["class_counts"])[0]), 8)
        self.assertIn(0, set(int(v) for v in np.asarray(checkpoint["seen_labels"]).tolist()))
        self.assertIn("best_epoch", checkpoint)
        self.assertIn("split_strategy", checkpoint)
        self.assertIn("collapse_warning", checkpoint)
        self.assertEqual(int(checkpoint["patch_channels"]), ct.PATCH_CHANNELS)
        self.assertEqual(int(checkpoint["patch_size"]), ct.RAW_PATCH_SIZE)

    def test_train_cell_model_saves_hybrid_mlp_metadata(self) -> None:
        expected_dim = ct.expected_feature_dim(ct.CONTEXT_RADIUS)
        rng = np.random.default_rng(13)
        samples_per_class = 20
        X = rng.normal(size=(samples_per_class * ct.NUM_CLASSES, expected_dim)).astype(np.float32)
        X_patch = rng.random(size=(samples_per_class * ct.NUM_CLASSES, *ct.expected_patch_shape()), dtype=np.float32)
        y = np.repeat(np.arange(ct.NUM_CLASSES, dtype=np.int64), samples_per_class)
        group_ids = np.arange(len(y), dtype=np.int64)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
            dataset_path = Path(tmpdir) / "cell_dataset.npz"
            model_path = Path(tmpdir) / "cell_model.pt"
            np.savez(
                dataset_path,
                X=X,
                X_patch=X_patch,
                y=y,
                group_ids=group_ids,
                schema_version=np.array([ct.DATASET_SCHEMA_VERSION], dtype=np.int32),
                num_classes=np.array([ct.NUM_CLASSES], dtype=np.int32),
                patch_channels=np.array([ct.PATCH_CHANNELS], dtype=np.int32),
                patch_size=np.array([ct.RAW_PATCH_SIZE], dtype=np.int32),
            )
            history = ct.train_cell_model(
                dataset_path=str(dataset_path),
                model_path=str(model_path),
                device_preference="cpu",
                epochs=4,
                batch_size=16,
            )
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        self.assertEqual(history["model_kind"], "mlp")
        self.assertIn("model", checkpoint)
        self.assertEqual(int(checkpoint["patch_channels"]), ct.PATCH_CHANNELS)
        self.assertEqual(int(checkpoint["patch_size"]), ct.RAW_PATCH_SIZE)
        self.assertIn("best_epoch", checkpoint)
        self.assertIn("split_strategy", checkpoint)
        self.assertIn("collapse_warning", checkpoint)

    def test_strict_deployment_ready_rejects_limited_checkpoint(self) -> None:
        expected_dim = ct.expected_feature_dim(ct.CONTEXT_RADIUS)
        checkpoint = {
            "model_kind": "knn",
            "model_schema_version": int(ct.MODEL_SCHEMA_VERSION),
            "input_dim": int(expected_dim),
            "feature_mean": np.zeros((1, expected_dim), dtype=np.float32),
            "feature_std": np.ones((1, expected_dim), dtype=np.float32),
            "num_classes": int(ct.NUM_CLASSES),
            "schema_version": int(ct.DATASET_SCHEMA_VERSION),
            "device_label": "cpu",
            "coverage_mode": "restricted",
            "coverage_reason": "제한 모드",
            "reference_features": np.zeros((1, expected_dim + int(np.prod(ct.expected_patch_shape()))), dtype=np.float32),
            "reference_labels": np.array([1], dtype=np.int64),
            "class_counts": np.array([0, 10, 0, 0, 0], dtype=np.int32),
            "seen_labels": np.array([1], dtype=np.int64),
            "patch_channels": ct.PATCH_CHANNELS,
            "patch_size": ct.RAW_PATCH_SIZE,
        }
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        morph_tuple = (
            np.array([[1]], dtype=np.uint8),
            np.array([[1]], dtype=np.uint8),
            np.array([[0.01]], dtype=np.float32),
            np.array([[0.0]], dtype=np.float32),
            np.array([[0.0]], dtype=np.float32),
            np.array([[0.0]], dtype=np.float32),
            np.array([[0.10]], dtype=np.float32),
        )

        with mock.patch("cell_training.os.path.exists", return_value=True), mock.patch(
            "cell_training.Path.read_bytes",
            return_value=b"MODEL\n",
        ), mock.patch(
            "cell_training.torch.load",
            return_value=checkpoint,
        ), mock.patch.object(
            ct,
            "_compute_morphological_features",
            return_value=morph_tuple,
        ), mock.patch.object(
            ct,
            "_compute_patch_source_maps",
            return_value=(
                np.zeros((10, 10), dtype=np.float32),
                np.zeros((10, 10), dtype=np.float32),
                np.zeros((10, 10), dtype=np.float32),
            ),
        ):
            final_dir, morph_dir, meta = ct.predict_with_model_with_meta(
                img,
                1,
                model_path="mock_model.pt",
                strict_deployment_ready=True,
            )

        self.assertTrue(np.array_equal(final_dir, morph_dir))
        self.assertIn("배포형", str(meta["model_notice"].reshape(-1)[0]))

    def test_append_to_dataset_uses_stable_group_key(self) -> None:
        expected_dim = ct.expected_feature_dim(ct.CONTEXT_RADIUS)
        patch = np.zeros(ct.expected_patch_shape(), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = str(Path(tmpdir) / "dataset.npz")
            ct.append_to_dataset(
                [[0.0] * expected_dim],
                [patch],
                [1],
                dataset_path=dataset_path,
                group_key="same-image",
            )
            ct.append_to_dataset(
                [[0.0] * expected_dim],
                [patch],
                [2],
                dataset_path=dataset_path,
                group_key="same-image",
            )
            with np.load(dataset_path) as data:
                group_ids = np.asarray(data["group_ids"], dtype=np.int64)
                labels = np.asarray(data["y"], dtype=np.int64)
                self.assertEqual(group_ids.shape[0], 1)
                self.assertEqual(labels.tolist(), [2])

    def test_reference_distance_scale_ignores_exact_duplicate_rows(self) -> None:
        X = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [3.0, 4.0],
            ],
            dtype=np.float32,
        )

        nearest = ct._pairwise_nearest_distances(X)

        self.assertTrue(np.isfinite(nearest[0]))
        self.assertGreater(float(nearest[0]), 1.0)

    def test_predict_with_model_with_meta_reuses_detection_cache(self) -> None:
        expected_dim = ct.expected_feature_dim(ct.CONTEXT_RADIUS)
        checkpoint = {
            "model_kind": "knn",
            "model_schema_version": int(ct.MODEL_SCHEMA_VERSION),
            "input_dim": int(expected_dim),
            "feature_mean": np.zeros((1, expected_dim), dtype=np.float32),
            "feature_std": np.ones((1, expected_dim), dtype=np.float32),
            "num_classes": int(ct.NUM_CLASSES),
            "schema_version": int(ct.DATASET_SCHEMA_VERSION),
            "device_label": "cpu",
            "coverage_mode": "normal",
            "coverage_reason": "",
            "reference_features": np.zeros((1, expected_dim + int(np.prod(ct.expected_patch_shape()))), dtype=np.float32),
            "reference_labels": np.array([2], dtype=np.int64),
            "class_counts": np.array([10, 10, 10, 10, 10], dtype=np.int32),
            "distance_scale": 1.0,
            "distance_limit": 1.0,
            "knn_k": 1,
            "patch_channels": ct.PATCH_CHANNELS,
            "patch_size": ct.RAW_PATCH_SIZE,
        }
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        detection_cache = {
            "A": 1,
            "morph_dir": np.array([[1]], dtype=np.uint8),
            "occ": np.array([[1]], dtype=np.uint8),
            "bboxes": [],
            "meta": {
                "bw": np.zeros((10, 10), dtype=np.uint8),
                "edges": np.zeros((10, 10), dtype=np.uint8),
                "confidence": np.array([[0.2]], dtype=np.float32),
                "overlap_count": np.array([[2]], dtype=np.int32),
                "support_level": np.array([[2]], dtype=np.uint8),
                "path_occ": np.array([[1]], dtype=np.uint8),
                "support_mask": np.array([[1]], dtype=np.uint8),
            },
        }
        global_maps = {
            "row_occ_density": np.array([1.0], dtype=np.float32),
            "col_occ_density": np.array([1.0], dtype=np.float32),
            "row_head_density": np.array([0.0], dtype=np.float32),
            "col_head_density": np.array([0.0], dtype=np.float32),
            "component_ratio": np.array([[0.0]], dtype=np.float32),
            "direction_run": np.array([[0.0]], dtype=np.float32),
            "occ_degree": np.array([[0.0]], dtype=np.float32),
            "incoming_degree": np.array([[0.0]], dtype=np.float32),
        }

        with mock.patch("cell_training.os.path.exists", return_value=True), mock.patch(
            "cell_training.Path.read_bytes",
            return_value=b"MODEL\n",
        ), mock.patch(
            "cell_training.torch.load",
            return_value=checkpoint,
        ), mock.patch.object(
            ct,
            "compute_direction_matrix_with_meta",
            side_effect=AssertionError("detector should not be called when cache is provided"),
        ), mock.patch.object(
            ct,
            "_compute_global_context_maps",
            return_value=global_maps,
        ), mock.patch.object(
            ct,
            "_cell_feature_vector",
            return_value=[0.0] * expected_dim,
        ), mock.patch.object(
            ct,
            "_compute_patch_source_maps",
            return_value=(
                np.zeros((10, 10), dtype=np.float32),
                np.zeros((10, 10), dtype=np.float32),
                np.zeros((10, 10), dtype=np.float32),
            ),
        ), mock.patch.object(
            ct,
            "_extract_cell_patch_tensor",
            return_value=np.zeros(ct.expected_patch_shape(), dtype=np.float32),
        ), mock.patch.object(
            ct,
            "_v2_predict_knn_probabilities",
            return_value=(np.array([0.01, 0.02, 0.92, 0.03, 0.02], dtype=np.float32), 0.05),
        ), mock.patch.object(
            ct,
            "_v2_query_closeness",
            return_value=(1.0, True),
        ), mock.patch.object(
            ct,
            "_v2_class_support_score",
            return_value=1.0,
        ):
            final_dir, _morph_dir, meta = ct.predict_with_model_with_meta(
                img,
                1,
                model_path="mock_model.pt",
                detection_cache=detection_cache,
            )

        self.assertEqual(int(final_dir[0, 0]), 2)
        self.assertEqual(int(meta["model_candidate_mask"][0, 0]), 1)
        self.assertEqual(int(meta["overlap_count"][0, 0]), 2)

    def test_training_uses_exemplars_when_independent_groups_are_too_few(self) -> None:
        expected_dim = ct.expected_feature_dim(ct.CONTEXT_RADIUS)
        rng = np.random.default_rng(31)
        samples_per_class = 20
        sample_count = samples_per_class * ct.NUM_CLASSES
        X = rng.normal(size=(sample_count, expected_dim)).astype(np.float32)
        X_patch = rng.random(
            size=(sample_count, *ct.expected_patch_shape()),
            dtype=np.float32,
        )
        y = np.repeat(np.arange(ct.NUM_CLASSES, dtype=np.int64), samples_per_class)
        group_ids = np.arange(sample_count, dtype=np.int64) % (ct.MIN_NEURAL_TRAINING_GROUPS - 2)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
            dataset_path = Path(tmpdir) / "cell_dataset.npz"
            model_path = Path(tmpdir) / "cell_model.pt"
            np.savez(
                dataset_path,
                X=X,
                X_patch=X_patch,
                y=y,
                group_ids=group_ids,
                schema_version=np.array([ct.DATASET_SCHEMA_VERSION], dtype=np.int32),
                num_classes=np.array([ct.NUM_CLASSES], dtype=np.int32),
                patch_channels=np.array([ct.PATCH_CHANNELS], dtype=np.int32),
                patch_size=np.array([ct.RAW_PATCH_SIZE], dtype=np.int32),
            )
            history = ct.train_cell_model(
                dataset_path=str(dataset_path),
                model_path=str(model_path),
                device_preference="cpu",
            )

        self.assertEqual(history["model_kind"], "knn")
        self.assertEqual(history["coverage_mode"], "restricted")
        self.assertIn("independent correction groups", str(history["coverage_reason"]))

    def test_low_validation_mlp_uses_human_exemplar_to_fill_supported_cell(self) -> None:
        expected_dim = ct.expected_feature_dim(ct.CONTEXT_RADIUS)
        reference_dim = expected_dim + int(np.prod(ct.expected_patch_shape()))
        checkpoint = {
            "model_kind": "mlp",
            "model_schema_version": int(ct.MODEL_SCHEMA_VERSION),
            "input_dim": int(expected_dim),
            "feature_mean": np.zeros((1, expected_dim), dtype=np.float32),
            "feature_std": np.ones((1, expected_dim), dtype=np.float32),
            "num_classes": int(ct.NUM_CLASSES),
            "schema_version": int(ct.DATASET_SCHEMA_VERSION),
            "device_label": "cpu",
            "coverage_mode": "normal",
            "coverage_reason": "",
            "val_best_macro_acc": 0.50,
            "reference_features": np.zeros((1, reference_dim), dtype=np.float32),
            "reference_labels": np.array([2], dtype=np.int64),
            "class_counts": np.array([10, 10, 10, 10, 10], dtype=np.int32),
            "distance_scale": 1.0,
            "distance_limit": 1.0,
            "knn_k": 1,
            "patch_channels": ct.PATCH_CHANNELS,
            "patch_size": ct.RAW_PATCH_SIZE,
        }
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        morph_tuple = (
            np.array([[0]], dtype=np.uint8),
            np.array([[1]], dtype=np.uint8),
            np.array([[0.4]], dtype=np.float32),
            np.array([[0.2]], dtype=np.float32),
            np.array([[0.4]], dtype=np.float32),
            np.array([[0.6]], dtype=np.float32),
            np.array([[0.1]], dtype=np.float32),
        )

        with mock.patch("cell_training.os.path.exists", return_value=True), mock.patch(
            "cell_training.Path.read_bytes",
            return_value=b"MODEL\n",
        ), mock.patch(
            "cell_training.torch.load",
            return_value=checkpoint,
        ), mock.patch.object(
            ct,
            "_compute_morphological_features",
            return_value=morph_tuple,
        ), mock.patch.object(
            ct,
            "_cell_feature_vector",
            return_value=[0.0] * expected_dim,
        ), mock.patch.object(
            ct,
            "_extract_cell_patch_tensor",
            return_value=np.ones(ct.expected_patch_shape(), dtype=np.float32),
        ), mock.patch.object(
            ct,
            "_v2_predict_knn_probabilities",
            return_value=(np.array([0.01, 0.01, 0.96, 0.01, 0.01], dtype=np.float32), 0.05),
        ), mock.patch.object(
            ct,
            "_v2_query_closeness",
            return_value=(1.0, True),
        ), mock.patch.object(
            ct,
            "_v2_class_support_score",
            return_value=1.0,
        ), mock.patch.object(
            ct,
            "_local_pipe_support_score",
            return_value=1.0,
        ), mock.patch.object(
            ct,
            "_has_local_directional_signal",
            return_value=True,
        ):
            final_dir, _morph_dir, meta = ct.predict_with_model_with_meta(
                img,
                1,
                model_path="mock_model.pt",
            )

        self.assertEqual(int(final_dir[0, 0]), 2)
        self.assertEqual(int(meta["model_applied_mask"][0, 0]), 1)
        self.assertEqual(int(meta["model_direct_mask"][0, 0]), 1)
        self.assertIn("human-exemplar correction", str(meta["model_notice"].reshape(-1)[0]))


class SolutionTests(unittest.TestCase):
    def test_direction_cells_reaching_outlets_excludes_cycles_and_dead_ends(self) -> None:
        matrix = np.array(
            [
                [1, 1, 1, 3, 0],
                [4, 0, 0, 0, 0],
                [1, 3, 0, 0, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=np.uint8,
        )

        reached = solution.direction_cells_reaching_outlets(matrix, [(0, 3)])

        expected = np.zeros_like(matrix)
        expected[0, 0:4] = 1
        expected[1, 0] = 1
        self.assertTrue(np.array_equal(reached, expected))

    def test_direction_reachability_matches_independent_random_trace(self) -> None:
        rng = np.random.default_rng(20260815)
        steps = {1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
        for _case in range(200):
            rows = int(rng.integers(2, 9))
            cols = int(rng.integers(2, 9))
            matrix = rng.integers(0, 5, size=(rows, cols), dtype=np.uint8)
            outlet = (int(rng.integers(rows)), int(rng.integers(cols)))
            actual = solution.direction_cells_reaching_outlets(matrix, [outlet]).astype(bool)
            expected = np.zeros((rows, cols), dtype=bool)
            for start_row in range(rows):
                for start_col in range(cols):
                    if int(matrix[start_row, start_col]) == 0 and (start_row, start_col) != outlet:
                        continue
                    current = (start_row, start_col)
                    visited = set()
                    while current not in visited:
                        if current == outlet:
                            expected[start_row, start_col] = True
                            break
                        visited.add(current)
                        direction = int(matrix[current])
                        if direction not in steps:
                            break
                        drow, dcol = steps[direction]
                        nxt = (current[0] + drow, current[1] + dcol)
                        if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                            break
                        if nxt != outlet and int(matrix[nxt]) == 0:
                            break
                        current = nxt
            self.assertTrue(np.array_equal(actual, expected))

    def test_outer_zero_margin_bounds_preserves_internal_empty_rows_and_is_idempotent(self) -> None:
        matrix = np.zeros((8, 10), dtype=np.uint8)
        matrix[2, 3] = 1
        matrix[4, 6] = 3

        bounds = solution.outer_zero_margin_bounds(matrix, padding=1)

        self.assertEqual(bounds, (1, 6, 2, 8))
        cropped = matrix[bounds[0] : bounds[1], bounds[2] : bounds[3]]
        self.assertEqual(int(np.count_nonzero(cropped[2])), 0)
        self.assertEqual(
            solution.outer_zero_margin_bounds(cropped, padding=1),
            (0, cropped.shape[0], 0, cropped.shape[1]),
        )

    def test_outer_zero_margin_bounds_includes_support_and_protected_outlet(self) -> None:
        matrix = np.zeros((7, 8), dtype=np.uint8)
        support = np.zeros_like(matrix)
        support[4, 5] = 1

        bounds = solution.outer_zero_margin_bounds(
            matrix,
            padding=1,
            support_masks=[support],
            protected_cells=[(2, 2)],
        )

        self.assertEqual(bounds, (1, 6, 1, 7))

    def test_outer_zero_margin_bounds_rejects_misaligned_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match matrix shape"):
            solution.outer_zero_margin_bounds(
                np.zeros((5, 6), dtype=np.uint8),
                support_masks=[np.zeros((4, 6), dtype=np.uint8)],
            )

    def test_non_arrow_lines_do_not_create_direction_matrix(self) -> None:
        cases = []
        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        cv2.line(img, (60, 150), (240, 150), (0, 0, 0), 3)
        cases.append(img)

        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        cv2.line(img, (150, 60), (150, 240), (0, 0, 0), 3)
        cases.append(img)

        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        cv2.line(img, (60, 150), (240, 150), (0, 0, 0), 3)
        cv2.line(img, (150, 60), (150, 150), (0, 0, 0), 3)
        cases.append(img)

        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        for x in range(0, 300, 30):
            cv2.line(img, (x, 0), (x, 299), (0, 0, 0), 1)
        for y in range(0, 300, 30):
            cv2.line(img, (0, y), (299, y), (0, 0, 0), 1)
        cases.append(img)

        for img in cases:
            matrix, _occ, boxes, meta = solution.compute_direction_matrix_with_meta(img, 50)

            self.assertEqual(int(np.count_nonzero(matrix)), 0)
            self.assertEqual(len(boxes), 0)
            self.assertEqual(int(meta["object_count"][0]), 0)

    def test_arrow_detail_mask_excludes_colored_boundary(self) -> None:
        img = np.full((180, 260, 3), 255, dtype=np.uint8)
        cv2.line(img, (20, 50), (240, 50), (0, 0, 0), 7, cv2.LINE_AA)
        cv2.line(img, (20, 130), (240, 130), (70, 175, 70), 7, cv2.LINE_AA)

        detail = solution.build_arrow_detail_binary(img)

        self.assertGreater(int(np.count_nonzero(detail[44:57, 20:241])), 100)
        self.assertEqual(int(np.count_nonzero(detail[124:137, 20:241])), 0)

    def test_pipe_mask_preserves_thin_black_pipe_and_excludes_colored_boundary(self) -> None:
        img = np.full((180, 260, 3), 255, dtype=np.uint8)
        cv2.line(img, (15, 25), (245, 90), (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(img, (15, 145), (245, 145), (70, 175, 70), 7, cv2.LINE_AA)

        pipe = solution.build_pipe_binary(img, solution.FeatureConfig(A=30))

        self.assertGreater(int(np.count_nonzero(pipe[20:96, 10:251])), 120)
        self.assertEqual(int(np.count_nonzero(pipe[137:153, 10:251])), 0)

    def test_single_pixel_pipe_touch_is_kept_as_occupied(self) -> None:
        img = np.full((180, 260, 3), 255, dtype=np.uint8)
        cv2.line(img, (8, 171), (251, 8), (0, 0, 0), 1, cv2.LINE_AA)
        rows = 30
        matrix, occ, _boxes, _meta = solution.compute_direction_matrix_with_meta(img, rows)
        detail = solution.build_pipe_binary(img, solution.FeatureConfig(A=rows))
        touched = solution.build_support_mask_from_binary(detail, rows, matrix.shape[1], min_pixels=1)

        self.assertTrue(np.array_equal(occ, touched))
        self.assertEqual(int(np.count_nonzero(matrix)), 0)

    def test_inline_arrow_detection_is_independent_of_output_rows(self) -> None:
        img = np.full((320, 320, 3), 255, dtype=np.uint8)
        cv2.line(img, (30, 160), (290, 160), (0, 0, 0), 6, cv2.LINE_AA)
        cv2.fillConvexPoly(
            img,
            np.array([[220, 160], [184, 139], [184, 181]], dtype=np.int32),
            (0, 0, 0),
        )

        signatures = []
        for rows in (16, 30, 50):
            matrix, _occ, boxes, meta = solution.compute_direction_matrix_with_meta(img, rows)
            directions = tuple(int(v) for v in np.asarray(meta["object_directions"]).tolist())
            signatures.append((int(meta["object_count"][0]), directions, len(boxes)))
            nonzero_directions = set(int(v) for v in matrix[matrix > 0].tolist())
            self.assertEqual(nonzero_directions, {1})
            self.assertEqual(tuple(int(v) for v in meta["candidate_detection_rows"].tolist()), (20, 30, 50))

        self.assertEqual(signatures, [(1, (1,), 1), (1, (1,), 1), (1, (1,), 1)])

    def test_inline_arrow_propagates_along_body_but_not_t_branch(self) -> None:
        img = np.full((240, 320, 3), 255, dtype=np.uint8)
        cv2.line(img, (20, 120), (300, 120), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, (160, 35), (160, 120), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.fillConvexPoly(
            img,
            np.array([[225, 120], [190, 101], [190, 139]], dtype=np.int32),
            (0, 0, 0),
        )

        matrix, _occ, boxes, meta = solution.compute_direction_matrix_with_meta(img, 16)
        horizontal = matrix[7:9, 1:20]
        vertical_branch = matrix[2:7, 9:12]

        self.assertEqual(len(boxes), 1)
        self.assertGreaterEqual(int(np.count_nonzero(horizontal == 1)), 28)
        self.assertEqual(int(np.count_nonzero(vertical_branch)), 0)
        self.assertEqual(set(int(v) for v in matrix[matrix > 0].tolist()), {1})
        self.assertGreater(int(meta["propagated_path_nonzero"][0]), 10)

    def test_width_peaks_read_inline_filled_arrow_directions(self) -> None:
        bw = np.zeros((180, 720), dtype=np.uint8)
        cases = [
            (np.array([1.0, 0.0], dtype=np.float32), 1),
            (np.array([0.0, 1.0], dtype=np.float32), 2),
            (np.array([-1.0, 0.0], dtype=np.float32), 3),
            (np.array([0.0, -1.0], dtype=np.float32), 4),
        ]
        gold = []
        for index, (unit, code) in enumerate(cases):
            center = np.array([90.0 + index * 180.0, 90.0], dtype=np.float32)
            perpendicular = np.array([-unit[1], unit[0]], dtype=np.float32)
            tip = center + unit * 10.0
            base = tip - unit * 28.0
            cv2.line(
                bw,
                tuple(np.rint(center - unit * 62.0).astype(int)),
                tuple(np.rint(center + unit * 62.0).astype(int)),
                255,
                5,
                cv2.LINE_AA,
            )
            cv2.fillConvexPoly(
                bw,
                np.array([tip, base + perpendicular * 16.0, base - perpendicular * 16.0], dtype=np.int32),
                255,
                cv2.LINE_AA,
            )
            gold.append((float(tip[0]), float(tip[1]), int(code)))

        dist_map, skel, stroke_radius, peaks = solution._arrow_width_peak_evidence(bw)
        self.assertEqual(len(peaks), 4)
        unused = set(range(len(peaks)))
        for tip_x, tip_y, expected_code in gold:
            index = min(
                unused,
                key=lambda item: (peaks[item][1] - tip_x) ** 2 + (peaks[item][2] - tip_y) ** 2,
            )
            unused.remove(index)
            _radius, peak_x, peak_y = peaks[index]
            evidence = solution._width_profile_direction_at_peak(
                dist_map,
                skel,
                peak_x,
                peak_y,
                stroke_radius,
            )
            self.assertIsNotNone(evidence)
            self.assertEqual(solution._direction_from_vector(evidence[0]), expected_code)

    def test_repeated_detection_is_idempotent(self) -> None:
        img = np.full((220, 320, 3), 255, dtype=np.uint8)
        cv2.line(img, (20, 110), (300, 110), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.fillConvexPoly(
            img,
            np.array([[230, 110], [196, 92], [196, 128]], dtype=np.int32),
            (0, 0, 0),
        )

        first_matrix, first_occ, first_boxes, first_meta = solution.compute_direction_matrix_with_meta(img, 20)
        second_matrix, second_occ, second_boxes, second_meta = solution.compute_direction_matrix_with_meta(img, 20)

        self.assertTrue(np.array_equal(first_matrix, second_matrix))
        self.assertTrue(np.array_equal(first_occ, second_occ))
        self.assertEqual(first_boxes, second_boxes)
        self.assertTrue(
            np.array_equal(first_meta["centerline_support_mask"], second_meta["centerline_support_mask"])
        )
        self.assertTrue(np.array_equal(first_meta["object_directions"], second_meta["object_directions"]))

    def test_local_arrow_path_does_not_use_global_component_tail(self) -> None:
        nodes = [(10, x) for x in range(21)]
        adj = {}
        for idx, node in enumerate(nodes):
            neighbors = []
            if idx > 0:
                neighbors.append(nodes[idx - 1])
            if idx + 1 < len(nodes):
                neighbors.append(nodes[idx + 1])
            adj[node] = neighbors

        path = solution._trace_local_tail_to_head_path(
            head_node=nodes[-1],
            component_nodes=nodes,
            adj=adj,
            direction=(1.0, 0.0),
            max_pixels=5.0,
        )

        self.assertEqual(path[-1], nodes[-1])
        self.assertNotEqual(path[0], nodes[0])
        self.assertLessEqual(len(path), 7)

    def test_crossed_arrows_keep_separate_contour_heads(self) -> None:
        img = np.full((320, 320, 3), 255, dtype=np.uint8)
        cv2.arrowedLine(img, (40, 160), (280, 160), (0, 0, 0), 8, tipLength=0.12)
        cv2.arrowedLine(img, (160, 40), (160, 280), (0, 0, 0), 8, tipLength=0.12)
        A = 16
        B = solution.derive_grid_B(A, img.shape[1], img.shape[0])
        feat = solution.build_features(img, solution.FeatureConfig(A=A, use_edges=True, use_gray=False))
        labels = solution._label_binary_components(feat["bw"])

        candidates = solution.detect_arrow_candidates(feat["bw"], A, B, component_labels=labels)
        directions = {solution._direction_from_vector(candidate.direction) for candidate in candidates}

        self.assertIn(1, directions)
        self.assertIn(2, directions)
        self.assertNotIn(3, directions)
        self.assertNotIn(4, directions)

    def test_candidate_direction_refinement_compares_reverse_direction(self) -> None:
        img = np.full((240, 320, 3), 255, dtype=np.uint8)
        cv2.arrowedLine(img, (40, 120), (270, 120), (0, 0, 0), 8, tipLength=0.16)
        A = 16
        B = solution.derive_grid_B(A, img.shape[1], img.shape[0])
        feat = solution.build_features(img, solution.FeatureConfig(A=A, use_edges=True, use_gray=False))
        candidate = solution.ArrowCandidate(
            bbox=(240, 90, 300, 150),
            tip=(270.0, 120.0),
            direction=(-1.0, 0.0),
            score=0.95,
            area=100.0,
            component_id=1,
        )

        refined, geometry_score = solution._refine_candidate_direction_by_geometry(feat["bw"], candidate, A, B)

        self.assertGreater(geometry_score, 0.0)
        self.assertEqual(solution._direction_from_vector(refined.direction), 1)

    def test_cardinal_arrow_direction_refinement_uses_shaft_support(self) -> None:
        cases = [
            ((40, 120), (270, 120), (-1.0, 0.0), 1),
            ((270, 120), (40, 120), (1.0, 0.0), 3),
            ((160, 40), (160, 210), (0.0, -1.0), 2),
            ((160, 210), (160, 40), (0.0, 1.0), 4),
        ]
        A = 16
        for start, end, reversed_direction, expected in cases:
            img = np.full((240, 320, 3), 255, dtype=np.uint8)
            cv2.arrowedLine(img, start, end, (0, 0, 0), 8, tipLength=0.16)
            B = solution.derive_grid_B(A, img.shape[1], img.shape[0])
            feat = solution.build_features(img, solution.FeatureConfig(A=A, use_edges=True, use_gray=False))
            candidate = solution.ArrowCandidate(
                bbox=(max(0, end[0] - 35), max(0, end[1] - 35), min(img.shape[1], end[0] + 36), min(img.shape[0], end[1] + 36)),
                tip=(float(end[0]), float(end[1])),
                direction=reversed_direction,
                score=0.95,
                area=100.0,
                component_id=1,
            )

            refined, geometry_score = solution._refine_candidate_direction_by_geometry(feat["bw"], candidate, A, B)

            self.assertGreater(geometry_score, 0.0)
            self.assertEqual(solution._direction_from_vector(refined.direction), expected)

    def test_object_rasterization_suppresses_reverse_steps(self) -> None:
        object_path = solution.ArrowObjectPath(
            component_id=1,
            candidate_index=0,
            bbox=(0, 0, 4, 1),
            head_cell=(0, 3),
            tail_cell=(0, 0),
            raw_cell_path=((0, 0), (0, 1), (0, 0), (0, 1), (0, 2), (0, 3)),
            vote_cell_path=((0, 0), (0, 1), (0, 0), (0, 1), (0, 2), (0, 3)),
            score=1.0,
            direction_code=1,
        )

        matrix, support, overlap = solution.rasterize_arrow_object_paths([object_path], 1, 4)

        self.assertEqual(int(np.count_nonzero(matrix == 3)), 0)
        self.assertGreaterEqual(int(np.count_nonzero(matrix == 1)), 3)
        self.assertEqual(int(np.count_nonzero(support)), 4)
        self.assertGreaterEqual(int(np.max(overlap)), 1)

    def test_crossed_arrows_do_not_reverse_main_paths(self) -> None:
        img = np.full((320, 320, 3), 255, dtype=np.uint8)
        cv2.arrowedLine(img, (40, 160), (280, 160), (0, 0, 0), 8, tipLength=0.12)
        cv2.arrowedLine(img, (160, 40), (160, 280), (0, 0, 0), 8, tipLength=0.12)

        matrix, _occ, _boxes, meta = solution.compute_direction_matrix_with_meta(img, 16)

        left_horizontal = matrix[7:9, 1:7].reshape(-1)
        upper_vertical = matrix[1:7, 7:9].reshape(-1)
        self.assertGreaterEqual(int(np.count_nonzero(left_horizontal == 1)), 8)
        self.assertGreaterEqual(int(np.count_nonzero(upper_vertical == 2)), 8)
        self.assertEqual(int(meta["object_count"][0]), 2)
        self.assertEqual(np.asarray(meta["object_head_cells"]).shape, (2, 2))
        self.assertEqual(np.asarray(meta["object_tail_cells"]).shape, (2, 2))
        self.assertEqual(np.asarray(meta["object_directions"]).shape, (2,))
        object_matrix = np.asarray(meta["object_direction_matrix"])
        object_support = np.asarray(meta["object_support_mask"])
        object_overlap = np.asarray(meta["object_overlap_count"])
        self.assertEqual(object_matrix.shape, matrix.shape)
        self.assertEqual(object_support.shape, matrix.shape)
        self.assertEqual(object_overlap.shape, matrix.shape)
        self.assertEqual(set(int(v) for v in np.asarray(meta["object_directions"]).tolist()), {1, 2})
        self.assertGreaterEqual(int(np.count_nonzero(object_matrix[7:9, 1:7] == 1)), 5)
        self.assertGreaterEqual(int(np.count_nonzero(object_matrix[1:7, 7:9] == 2)), 5)

    def test_touching_opposite_heads_do_not_become_vertical(self) -> None:
        img = np.full((320, 320, 3), 255, dtype=np.uint8)
        cv2.arrowedLine(img, (40, 160), (170, 160), (0, 0, 0), 8, tipLength=0.18)
        cv2.arrowedLine(img, (280, 160), (150, 160), (0, 0, 0), 8, tipLength=0.18)
        A = 16
        B = solution.derive_grid_B(A, img.shape[1], img.shape[0])
        feat = solution.build_features(img, solution.FeatureConfig(A=A, use_edges=True, use_gray=False))
        labels = solution._label_binary_components(feat["bw"])

        candidates = solution.detect_arrow_candidates(feat["bw"], A, B, component_labels=labels)
        directions = {solution._direction_from_vector(candidate.direction) for candidate in candidates}
        matrix, _occ, _boxes, _meta = solution.compute_direction_matrix_with_meta(img, A)

        self.assertEqual(directions, {1, 3})
        self.assertGreaterEqual(int(np.count_nonzero(matrix[7:9, 1:7] == 1)), 8)
        self.assertGreaterEqual(int(np.count_nonzero(matrix[7:9, 9:15] == 3)), 8)

    def test_resolve_cell_direction_prefers_transition_on_overlap(self) -> None:
        overlap_count = np.array([[2, 0]], dtype=np.int32)
        axis_x = np.array([[1.0, 0.0]], dtype=np.float32)
        axis_y = np.array([[0.0, 0.0]], dtype=np.float32)
        axis_conf = np.array([[1.0, 0.0]], dtype=np.float32)
        cell_votes = {1: 2.0, 2: 2.4}
        transition_votes = {((0, 0), (0, 1)): 3.5}

        direction = solution._resolve_cell_direction(
            (0, 0),
            cell_votes,
            transition_votes,
            overlap_count,
            axis_x,
            axis_y,
            axis_conf,
            1,
            2,
        )

        self.assertEqual(direction, 1)

    def test_bfs_repair_breaks_loop_to_reach_outlet(self) -> None:
        matrix = np.zeros((3, 3), dtype=np.uint8)
        matrix[0, 0] = 1
        matrix[0, 1] = 2
        matrix[1, 1] = 3
        matrix[1, 0] = 4
        support = (matrix > 0).astype(np.uint8)
        outlet = (1, 2)

        fixed, changed = solution._repair_unreachable_cells_to_outlet(matrix, support, outlet, [(0, 0)])
        route = (fixed > 0).astype(np.uint8)

        self.assertGreater(changed, 0)
        self.assertLess(changed, int(np.count_nonzero(support)))
        for cell in map(tuple, np.argwhere(support > 0)):
            self.assertTrue(solution._path_reaches_outlet(fixed, route, cell, outlet), f"{cell} did not reach outlet")

    def test_recommend_outlet_candidates_ranks_natural_flow_terminals(self) -> None:
        matrix = np.zeros((4, 5), dtype=np.uint8)
        matrix[0, 0:4] = 1
        matrix[2, 0:2] = 1

        ranked = solution.recommend_outlet_candidates(matrix)

        self.assertEqual(ranked[0], ((0, 3), 4))
        self.assertEqual(ranked[1], ((2, 1), 2))

    def test_partition_support_by_multiple_outlets_assigns_each_route_cell(self) -> None:
        support = np.ones((1, 5), dtype=np.uint8)

        labels = gui.NFMATApp._partition_support_by_outlets(support, [(0, 0), (0, 4)])

        self.assertEqual(labels.tolist(), [[0, 0, 0, 1, 1]])

    def test_assist_direction_matrix_preserves_existing_detour_to_outlet(self) -> None:
        img = np.full((90, 90, 3), 255, dtype=np.uint8)
        matrix = np.zeros((3, 3), dtype=np.uint8)
        matrix[0, 0] = 2
        matrix[1, 0] = 1
        matrix[1, 1] = 4
        matrix[0, 1] = 1
        outlet = (0, 2)
        support = (matrix > 0).astype(np.uint8)

        assisted, _support, stats, unresolved = solution.assist_direction_matrix(
            img,
            matrix,
            3,
            outlet=outlet,
            occ=support,
            path_occ=support,
        )

        self.assertTrue(np.array_equal(assisted, matrix))
        self.assertEqual(int(stats["cycle_fix"]), 0)
        self.assertEqual(int(np.count_nonzero(unresolved)), 0)

    def test_assist_direction_matrix_clears_internal_loop(self) -> None:
        img = np.full((90, 90, 3), 255, dtype=np.uint8)
        matrix = np.zeros((3, 3), dtype=np.uint8)
        matrix[0, 0] = 1
        matrix[0, 1] = 2
        matrix[1, 1] = 3
        matrix[1, 0] = 4
        outlet = (1, 2)
        support = (matrix > 0).astype(np.uint8)

        assisted, _support, stats, unresolved = solution.assist_direction_matrix(
            img,
            matrix,
            3,
            outlet=outlet,
            occ=support,
            path_occ=support,
        )
        route = (assisted > 0).astype(np.uint8)

        self.assertEqual(int(stats["unresolved_count"]), 0)
        self.assertEqual(int(np.count_nonzero(unresolved)), 0)
        for cell in map(tuple, np.argwhere(support > 0)):
            self.assertTrue(solution._path_reaches_outlet(assisted, route, cell, outlet), f"{cell} did not reach outlet")


class SyntheticArrowTests(unittest.TestCase):
    def test_matrix_arrow_renderer_connects_vertical_run(self) -> None:
        matrix = np.array(
            [
                [0, 2, 0],
                [0, 2, 0],
                [0, 2, 0],
            ],
            dtype=np.uint8,
        )

        img = matrix_arrows.render_direction_matrix_arrows(matrix, cell_px=40, margin=20)
        center_x = 20 + 40 + 20
        vertical_dark = np.count_nonzero(np.any(img[35:140, center_x - 3:center_x + 4] < 210, axis=2))
        left_dark = np.count_nonzero(np.any(img[35:140, 20:35] < 210, axis=2))

        self.assertEqual(tuple(img.shape), (160, 160, 3))
        self.assertGreater(int(vertical_dark), 500)
        self.assertLess(int(left_dark), int(vertical_dark))

    def test_matrix_arrow_renderer_marks_flood_cells_with_visible_square(self) -> None:
        matrix = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        flood = np.array([[1, 0], [0, 0]], dtype=np.uint8)

        img = matrix_arrows.render_direction_matrix_arrows(matrix, cell_px=40, margin=20, highlight_mask=flood)

        red_pixels = (img[:, :, 2] > 245) & (img[:, :, 1] < 70) & (img[:, :, 0] < 110)
        self.assertGreater(int(np.count_nonzero(red_pixels)), 0)

    def test_render_arrow_matrix_image_varies_styles(self) -> None:
        matrix = np.zeros((4, 4), dtype=np.uint8)
        matrix[0, 0] = 1
        matrix[1, 1] = 2
        matrix[2, 2] = 3
        matrix[3, 3] = 4

        img_a, styles_a = synthetic_arrows.render_arrow_matrix_image(matrix, cell_px=24, variant=0, seed=7)
        img_b, styles_b = synthetic_arrows.render_arrow_matrix_image(matrix, cell_px=24, variant=1, seed=7)

        self.assertEqual(tuple(img_a.shape), (96, 96, 3))
        self.assertGreater(int(np.count_nonzero(img_a < 220)), 0)
        self.assertGreater(sum(styles_a.values()), 0)
        self.assertGreater(sum(styles_b.values()), 0)
        self.assertFalse(np.array_equal(img_a, img_b))

    def test_synthetic_arrow_training_batch_extracts_all_direction_labels(self) -> None:
        matrix = np.zeros((4, 4), dtype=np.uint8)
        matrix[0, 0] = 1
        matrix[0, 3] = 2
        matrix[3, 3] = 3
        matrix[3, 0] = 4

        batch = synthetic_arrows.build_synthetic_arrow_training_batch(
            matrix,
            variants=2,
            cell_px=24,
            max_directional_cells_per_variant=20,
            seed=11,
        )

        self.assertEqual(len(batch.X), len(batch.X_patch))
        self.assertEqual(len(batch.X), len(batch.y))
        self.assertGreater(len(batch.preview_images), 0)
        for code in range(5):
            self.assertIn(code, batch.label_counts)


class GISMatrixTests(unittest.TestCase):
    @staticmethod
    def _write_polyline_shp(path: Path, shape_type: int, points: list[tuple[float, float]]) -> None:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        content = bytearray()
        content += struct.pack("<i", shape_type)
        content += struct.pack("<4d", *bbox)
        content += struct.pack("<2i", 1, len(points))
        content += struct.pack("<i", 0)
        for x, y in points:
            content += struct.pack("<2d", x, y)
        if shape_type == 13:
            content += struct.pack("<2d", 0.0, 0.0)
            for _point in points:
                content += struct.pack("<d", 0.0)
            content += struct.pack("<2d", -1.0e38, -1.0e38)
            for _point in points:
                content += struct.pack("<d", -1.0e38)

        file_length_words = (100 + 8 + len(content)) // 2
        header = bytearray()
        header += struct.pack(">i", 9994)
        header += struct.pack(">5i", 0, 0, 0, 0, 0)
        header += struct.pack(">i", file_length_words)
        header += struct.pack("<2i", 1000, shape_type)
        header += struct.pack("<4d", *bbox)
        header += struct.pack("<4d", 0.0, 0.0, 0.0, 0.0)
        record_header = struct.pack(">2i", 1, len(content) // 2)
        path.write_bytes(bytes(header) + record_header + bytes(content))

    @staticmethod
    def _write_polygon_shp(path: Path, rings: list[list[tuple[float, float]]], shape_type: int = 5) -> None:
        all_points = [point for ring in rings for point in ring]
        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        contents: list[bytes] = []
        for ring in rings:
            closed = list(ring)
            if closed[0] != closed[-1]:
                closed.append(closed[0])
            ring_xs = [point[0] for point in closed]
            ring_ys = [point[1] for point in closed]
            content = bytearray()
            content += struct.pack("<i", shape_type)
            content += struct.pack("<4d", min(ring_xs), min(ring_ys), max(ring_xs), max(ring_ys))
            content += struct.pack("<2i", 1, len(closed))
            content += struct.pack("<i", 0)
            for x, y in closed:
                content += struct.pack("<2d", x, y)
            if shape_type == 15:
                content += struct.pack("<2d", 0.0, 0.0)
                for _point in closed:
                    content += struct.pack("<d", 0.0)
                content += struct.pack("<2d", -1.0e38, -1.0e38)
                for _point in closed:
                    content += struct.pack("<d", -1.0e38)
            elif shape_type == 25:
                content += struct.pack("<2d", -1.0e38, -1.0e38)
                for _point in closed:
                    content += struct.pack("<d", -1.0e38)
            contents.append(bytes(content))

        total_bytes = 100 + sum(8 + len(content) for content in contents)
        header = bytearray()
        header += struct.pack(">i", 9994)
        header += struct.pack(">5i", 0, 0, 0, 0, 0)
        header += struct.pack(">i", total_bytes // 2)
        header += struct.pack("<2i", 1000, shape_type)
        header += struct.pack("<4d", *bbox)
        header += struct.pack("<4d", 0.0, 0.0, 0.0, 0.0)
        data = bytearray(header)
        for index, content in enumerate(contents, start=1):
            data += struct.pack(">2i", index, len(content) // 2)
            data += content
        path.write_bytes(bytes(data))

    @staticmethod
    def _write_dbf(path: Path, fields: list[tuple[str, int]], rows: list[dict[str, float]]) -> None:
        header_len = 32 + 32 * len(fields) + 1
        record_len = 1 + sum(length for _name, length in fields)
        header = bytearray(32)
        header[0] = 0x03
        header[1:4] = bytes([126, 5, 6])
        header[4:8] = struct.pack("<I", len(rows))
        header[8:10] = struct.pack("<H", header_len)
        header[10:12] = struct.pack("<H", record_len)
        data = bytearray(header)
        for name, length in fields:
            desc = bytearray(32)
            desc[0: min(len(name), 11)] = name.encode("ascii")[:11]
            desc[11] = ord("N")
            desc[16] = length
            desc[17] = 3
            data += desc
        data.append(0x0D)
        for row in rows:
            record = bytearray(b" ")
            for name, length in fields:
                text = f"{float(row[name]):>{length}.3f}"[-length:]
                record += text.encode("ascii")
            data += record
        data.append(0x1A)
        path.write_bytes(bytes(data))

    @staticmethod
    def _write_text_dbf(path: Path, field_name: str, values: list[str], field_len: int = 20) -> None:
        header_len = 32 + 32 + 1
        record_len = 1 + field_len
        header = bytearray(32)
        header[0] = 0x03
        header[1:4] = bytes([126, 6, 9])
        header[4:8] = struct.pack("<I", len(values))
        header[8:10] = struct.pack("<H", header_len)
        header[10:12] = struct.pack("<H", record_len)
        desc = bytearray(32)
        desc[0: min(len(field_name), 11)] = field_name.encode("ascii")[:11]
        desc[11] = ord("C")
        desc[16] = field_len
        data = bytearray(header) + desc + bytes([0x0D])
        for value in values:
            encoded = value.encode("utf-8")[:field_len]
            data += b" " + encoded.ljust(field_len, b" ")
        data.append(0x1A)
        path.write_bytes(bytes(data))

    def test_diagonal_polyline_is_expanded_to_four_neighbor_path(self) -> None:
        result = gis_matrix.matrix_from_polylines(
            [
                gis_matrix.GISPolyline(
                    parts=[[(0.0, 0.0), (10.0, 10.0)]],
                    start_height=10.0,
                    end_height=9.0,
                )
            ],
            rows=10,
            cols=10,
            bbox=(0.0, 0.0, 10.0, 10.0),
        )
        cells = set(map(tuple, np.argwhere(result.matrix > 0)))
        self.assertGreater(len(cells), 10)
        start = next(iter(cells))
        seen = {start}
        stack = [start]
        while stack:
            row, col = stack.pop()
            for nb in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        self.assertEqual(seen, cells)

    def test_height_rising_geometry_is_reversed(self) -> None:
        result = gis_matrix.matrix_from_polylines(
            [
                gis_matrix.GISPolyline(
                    parts=[[(0.0, 5.0), (10.0, 5.0)]],
                    start_height=1.0,
                    end_height=2.0,
                )
            ],
            rows=5,
            cols=10,
            bbox=(0.0, 0.0, 10.0, 10.0),
        )
        self.assertGreater(int(np.count_nonzero(result.matrix == 3)), 0)
        self.assertEqual(int(np.count_nonzero(result.matrix == 1)), 0)

    def test_polylinez_shp_without_shx_or_dbf_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = Path(tmpdir) / "network.shp"
            self._write_polyline_shp(shp_path, 13, [(0.0, 5.0), (10.0, 5.0)])

            info = gis_matrix.inspect_gis_source(shp_path)
            result = gis_matrix.load_gis_matrix(shp_path, rows=5, start_height_field=None, end_height_field=None)

            self.assertEqual(info["shape_type_name"], "PolyLineZ")
            self.assertFalse(bool(result.meta["dbf_available"]))
            self.assertFalse(bool(result.meta["shx_available"]))
            self.assertGreater(int(np.count_nonzero(result.matrix == 1)), 0)

    def test_read_shp_shape_info_classifies_polyline_and_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipe_path = root / "network.shp"
            boundary_path = root / "district.shp"
            self._write_polyline_shp(pipe_path, 3, [(0.0, 5.0), (10.0, 5.0)])
            self._write_polygon_shp(
                boundary_path,
                [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]],
            )

            pipe_info = gis_matrix.read_shp_shape_info(pipe_path)
            boundary_info = gis_matrix.read_shp_shape_info(boundary_path)

            self.assertEqual(pipe_info["kind"], "polyline")
            self.assertEqual(pipe_info["shape_type_name"], "PolyLine")
            self.assertEqual(boundary_info["kind"], "boundary")
            self.assertEqual(boundary_info["shape_type_name"], "Polygon")

    def test_gis_load_worker_defers_spatial_index_building(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipe_path = root / "network.shp"
            boundary_path = root / "district.shp"
            self._write_polyline_shp(pipe_path, 3, [(0.0, 5.0), (10.0, 5.0)])
            self._write_polygon_shp(
                boundary_path,
                [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]],
            )
            app = object.__new__(gui.NFMATApp)
            app._gis_load_result_queue = gui.queue.Queue()

            app._load_gis_layers_worker((str(pipe_path), str(boundary_path)))

            loaded, errors = app._gis_load_result_queue.get_nowait()
            self.assertEqual(errors, [])
            self.assertEqual([layer["kind"] for layer in loaded], ["polyline", "boundary"])
            self.assertNotIn("spatial_index", loaded[0])
            self.assertNotIn("feature_index", loaded[1])

    def test_finish_add_gis_layers_preserves_existing_project_view(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app._gis_layers = {
            "layer_1": {
                "kind": "polyline",
                "path": "pipe.shp",
                "name": "pipe",
                "bbox": (0.0, 0.0, 10.0, 10.0),
                "geometry": [],
                "meta": {},
                "visible": True,
                "style": {},
            }
        }
        app._gis_layer_sequence = 1
        app._gis_pipe_layer_id = "layer_1"
        app._gis_boundary_layer_id = None
        app._gis_flood_layer_ids = set()
        app._gis_flood_layer_id = None
        app.source_kind = "gis_project"
        app.img_pil = gui.Image.new("RGB", (16, 16))
        app._gis_load_in_progress = True
        app._gis_render_polling = False
        app._invalidate_gis_analysis_cache = mock.Mock()
        app._looks_like_flood_layer_name = mock.Mock(return_value=False)
        app._default_gis_layer_style = mock.Mock(return_value={"fill_color": (0, 0, 0), "fill_opacity": 0.5, "outline_color": (0, 0, 0), "outline_opacity": 1.0, "line_width": 1.0})
        app._apply_non_role_default_opacity = mock.Mock(side_effect=lambda style, _kind, _is_role: style)
        app._active_flood_layer_ids = mock.Mock(return_value=set())
        app._sync_legacy_flood_layer_id = mock.Mock()
        app._sync_default_role_opacities = mock.Mock()
        app._confirm_pipe_polygon_compatibility = mock.Mock(return_value=True)
        app._invalidate_gis_render_cache = mock.Mock()
        app._refresh_gis_layer_tree = mock.Mock()
        app._update_flood_region_label = mock.Mock()
        app._render_gis_project = mock.Mock()
        app._schedule_gis_index_build = mock.Mock()
        app.status_var = mock.Mock()
        loaded = [
            {
                "kind": "boundary",
                "path": "flood.shp",
                "name": "2023 침수흔적도 최신수정",
                "bbox": (1.0, 1.0, 2.0, 2.0),
                "geometry": [],
                "meta": {},
                "visible": True,
                "label_field": "",
            }
        ]

        app._finish_add_gis_layers(loaded, [])

        app._render_gis_project.assert_called_once_with(reset_matrix=False)
        app._schedule_gis_index_build.assert_called_once_with(["layer_2"])

    def test_custom_dbf_height_fields_without_shx_reverse_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "network"
            shp_path = base.with_suffix(".shp")
            self._write_polyline_shp(shp_path, 3, [(0.0, 5.0), (10.0, 5.0)])
            self._write_dbf(
                base.with_suffix(".dbf"),
                [("UP_H", 10), ("DN_H", 10)],
                [{"UP_H": 1.0, "DN_H": 2.0}],
            )

            result = gis_matrix.load_gis_matrix(
                shp_path,
                rows=5,
                start_height_field="UP_H",
                end_height_field="DN_H",
            )

            self.assertEqual(result.meta["height_start_field"], "UP_H")
            self.assertEqual(result.meta["height_end_field"], "DN_H")
            self.assertFalse(bool(result.meta["shx_available"]))
            self.assertGreater(int(np.count_nonzero(result.matrix == 3)), 0)
            self.assertEqual(int(np.count_nonzero(result.matrix == 1)), 0)

    def test_load_gis_matrix_applies_rotation_degrees(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = Path(tmpdir) / "network.shp"
            self._write_polyline_shp(shp_path, 3, [(0.0, 5.0), (10.0, 5.0)])

            result = gis_matrix.load_gis_matrix(
                shp_path,
                rows=5,
                start_height_field=None,
                end_height_field=None,
                rotation_degrees=90.0,
            )

            self.assertEqual(float(result.meta["rotation_degrees"]), 90.0)
            self.assertGreater(int(np.count_nonzero(result.matrix == 4)), 0)
            self.assertEqual(int(np.count_nonzero(result.matrix == 1)), 0)

    def test_gis_cell_size_metadata_uses_projected_meter_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = Path(tmpdir) / "network.shp"
            self._write_polyline_shp(shp_path, 3, [(0.0, 0.0), (10.0, 10.0)])
            shp_path.with_suffix(".prj").write_text(
                'PROJCS["Test",GEOGCS["GCS",UNIT["Degree",0.0174532925199433]],UNIT["metre",1.0]]',
                encoding="utf-8",
            )

            result = gis_matrix.load_gis_matrix(shp_path, rows=5, start_height_field=None, end_height_field=None)

            self.assertEqual(result.meta["coordinate_unit_type"], "meter")
            self.assertEqual(result.meta["rows_A"], 5)
            self.assertEqual(result.meta["cols_B"], 5)
            self.assertAlmostEqual(float(result.meta["cell_width"]), 2.0)
            self.assertAlmostEqual(float(result.meta["cell_height"]), 2.0)
            self.assertAlmostEqual(float(result.meta["cell_width_m"]), 2.0)
            self.assertAlmostEqual(float(result.meta["cell_height_m"]), 2.0)

    def test_gis_occupied_cell_average_slope_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "network"
            shp_path = base.with_suffix(".shp")
            self._write_polyline_shp(shp_path, 3, [(0.0, 0.0), (10.0, 10.0)])
            shp_path.with_suffix(".prj").write_text(
                'PROJCS["Test",GEOGCS["GCS",UNIT["Degree",0.0174532925199433]],UNIT["metre",1.0]]',
                encoding="utf-8",
            )
            self._write_dbf(
                base.with_suffix(".dbf"),
                [("UP_H", 10), ("DN_H", 10)],
                [{"UP_H": 10.0, "DN_H": 9.0}],
            )

            result = gis_matrix.load_gis_matrix(
                shp_path,
                rows=5,
                start_height_field="UP_H",
                end_height_field="DN_H",
            )

            expected_slope = 1.0 / float(np.hypot(10.0, 10.0))
            self.assertTrue(bool(result.meta["occupied_slope_available"]))
            self.assertTrue(bool(result.meta["occupied_slope_metric"]))
            self.assertGreater(int(result.meta["occupied_slope_cell_count"]), 0)
            self.assertAlmostEqual(float(result.meta["occupied_slope_mean"]), expected_slope)
            self.assertAlmostEqual(float(result.meta["occupied_slope_percent"]), expected_slope * 100.0)

    def test_rotate_bgr_image_expands_canvas_for_fine_angle(self) -> None:
        image = np.full((20, 40, 3), 255, dtype=np.uint8)
        cv2.line(image, (5, 10), (35, 10), (0, 0, 0), 2)

        rotated = gui.NFMATApp._rotate_bgr_image(image, 12.5)

        self.assertGreaterEqual(rotated.shape[0], image.shape[0])
        self.assertGreater(rotated.shape[1], image.shape[1])
        self.assertLess(int(np.min(rotated)), 255)

    def test_boundary_values_and_selected_region_clip_pipe_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipe_path = root / "seoul_pipes.shp"
            boundary_path = root / "districts.shp"
            self._write_polyline_shp(pipe_path, 3, [(0.0, 5.0), (20.0, 5.0)])
            self._write_polygon_shp(
                boundary_path,
                [
                    [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
                    [(10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)],
                ],
            )
            self._write_text_dbf(boundary_path.with_suffix(".dbf"), "DISTRICT", ["West", "East"])

            self.assertEqual(gis_matrix.list_boundary_values(boundary_path, "DISTRICT"), ["East", "West"])
            result = gis_matrix.load_gis_matrix(
                pipe_path,
                rows=10,
                start_height_field=None,
                end_height_field=None,
                boundary_shp_path=boundary_path,
                boundary_field="DISTRICT",
                boundary_value="East",
            )

            self.assertEqual(result.meta["boundary_value"], "East")
            self.assertEqual(result.meta["boundary_selected_feature_count"], 1)
            self.assertEqual(result.meta["boundary_clipped_pipe_records"], 1)
            self.assertEqual(result.meta["bbox"], [10.0, 0.0, 20.0, 10.0])
            self.assertGreater(int(np.count_nonzero(result.matrix)), 0)

    def test_polygonz_boundary_source_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            boundary_path = Path(tmpdir) / "district_z.shp"
            self._write_polygon_shp(
                boundary_path,
                [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]],
                shape_type=15,
            )

            info = gis_matrix.inspect_boundary_source(boundary_path)
            boundaries, _bbox, _meta = gis_matrix.read_boundary_shp(boundary_path)

            self.assertEqual(info["shape_type_name"], "PolygonZ")
            self.assertEqual(len(boundaries), 1)

    def test_boundary_clipping_preserves_original_pipe_length_for_slope(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipe_path = root / "slope_pipe.shp"
            boundary_path = root / "district.shp"
            self._write_polyline_shp(pipe_path, 3, [(0.0, 5.0), (20.0, 5.0)])
            self._write_dbf(
                pipe_path.with_suffix(".dbf"),
                [("START_H", 10), ("END_H", 10)],
                [{"START_H": 10.0, "END_H": 8.0}],
            )
            self._write_polygon_shp(
                boundary_path,
                [[(10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)]],
            )
            self._write_text_dbf(boundary_path.with_suffix(".dbf"), "DISTRICT", ["East"])

            result = gis_matrix.load_gis_matrix(
                pipe_path,
                rows=10,
                start_height_field="START_H",
                end_height_field="END_H",
                boundary_shp_path=boundary_path,
                boundary_field="DISTRICT",
                boundary_value="East",
            )

            self.assertTrue(bool(result.meta["occupied_slope_available"]))
            self.assertAlmostEqual(float(result.meta["occupied_slope_mean"]), 0.1)

    def test_render_gis_layers_combines_pipe_and_boundary_layers(self) -> None:
        records = [gis_matrix.GISPolyline(parts=[[(0.0, 5.0), (10.0, 5.0)]])]
        boundaries = [
            gis_matrix.GISBoundary(
                rings=[[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]],
                attributes={"NAME": "Area"},
            )
        ]

        image = gis_matrix.render_gis_layers(
            [(records, (20, 80, 200))],
            [(boundaries, (80, 160, 100))],
            (0.0, 0.0, 10.0, 10.0),
            max_size=600,
        )

        self.assertEqual(image.shape[:2], (600, 600))
        self.assertLess(int(np.min(image)), 250)

    def test_render_gis_layers_honors_exact_output_size_for_viewport_tiles(self) -> None:
        records = [gis_matrix.GISPolyline(parts=[[(0.0, 2.0), (10.0, 2.0)]])]

        image = gis_matrix.render_gis_layers(
            [(records, (0, 0, 0))],
            [],
            (0.0, 0.0, 20.0, 10.0),
            max_size=600,
            output_size=(80, 60),
        )

        self.assertEqual(image.shape[:2], (60, 80))
        self.assertLess(int(np.min(image)), 250)

    def test_render_gis_layers_applies_layer_opacity(self) -> None:
        records = [gis_matrix.GISPolyline(parts=[[(0.0, 5.0), (10.0, 5.0)]])]

        hidden = gis_matrix.render_gis_layers(
            [(records, (0, 0, 0), 0.0)],
            [],
            (0.0, 0.0, 10.0, 10.0),
            max_size=500,
        )
        visible = gis_matrix.render_gis_layers(
            [(records, (0, 0, 0), 1.0)],
            [],
            (0.0, 0.0, 10.0, 10.0),
            max_size=500,
        )

        self.assertEqual(int(np.min(hidden)), 250)
        self.assertLess(int(np.min(visible)), 250)

    def test_boundary_outline_renders_when_fill_is_transparent(self) -> None:
        boundaries = [
            gis_matrix.GISBoundary(
                rings=[[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]],
                attributes={},
            )
        ]

        image = gis_matrix.render_gis_layers(
            [],
            [(boundaries, (0, 255, 0), 0.0, (0, 0, 255), 1.0, 7.0)],
            (0.0, 0.0, 10.0, 10.0),
            max_size=500,
        )

        self.assertTrue(np.array_equal(image[250, 250], np.array([250, 250, 250], dtype=np.uint8)))
        self.assertGreater(int(np.count_nonzero((image[:, :, 2] > 200) & (image[:, :, 1] < 80))), 0)

    def test_ordered_layer_stack_draws_later_layer_on_top(self) -> None:
        records = [gis_matrix.GISPolyline(parts=[[(0.0, 5.0), (10.0, 5.0)]])]
        blue = (255, 0, 0)
        red = (0, 0, 255)

        image = gis_matrix.render_gis_layers(
            [],
            [],
            (0.0, 0.0, 10.0, 10.0),
            max_size=500,
            layer_stack=[
                ("polyline", (records, blue, 1.0, 9.0)),
                ("polyline", (records, red, 1.0, 3.0)),
            ],
        )

        self.assertGreater(int(image[250, 250, 2]), int(image[250, 250, 0]))

    def test_gis_compatibility_blocks_different_prj_definitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipe = root / "pipe.shp"
            boundary = root / "boundary.shp"
            pipe.write_bytes(b"")
            boundary.write_bytes(b"")
            pipe.with_suffix(".prj").write_text(crs_support.parse_crs("EPSG:5186").wkt, encoding="utf-8")
            boundary.with_suffix(".prj").write_text(crs_support.parse_crs("EPSG:4326").wkt, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "coordinate systems differ"):
                gis_matrix.validate_gis_compatibility(
                    pipe,
                    (0.0, 0.0, 10.0, 10.0),
                    boundary,
                    (0.0, 0.0, 10.0, 10.0),
                )

    def test_clip_polylines_excludes_hole_in_boundary(self) -> None:
        outer = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
        hole = [(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0), (4.0, 4.0)]
        records = [gis_matrix.GISPolyline(parts=[[(-1.0, 5.0), (11.0, 5.0)]])]

        clipped = gis_matrix.clip_polylines_to_boundary(records, [outer, hole])

        self.assertEqual(len(clipped), 1)
        self.assertEqual(len(clipped[0].parts), 2)
        self.assertAlmostEqual(clipped[0].parts[0][0][0], 0.0)
        self.assertAlmostEqual(clipped[0].parts[0][-1][0], 4.0)
        self.assertAlmostEqual(clipped[0].parts[1][0][0], 6.0)
        self.assertAlmostEqual(clipped[0].parts[1][-1][0], 10.0)

    def test_boundary_grid_mask_preserves_holes_and_unions_overlapping_features(self) -> None:
        outer = [(0.0, 0.0), (7.0, 0.0), (7.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
        hole = [(2.0, 4.0), (3.0, 4.0), (3.0, 6.0), (2.0, 6.0), (2.0, 4.0)]
        overlapping = [(5.0, 0.0), (10.0, 0.0), (10.0, 10.0), (5.0, 10.0), (5.0, 0.0)]

        mask = gis_matrix.boundary_grid_mask(
            [[outer, hole], [overlapping]],
            rows=101,
            cols=101,
            bbox=(0.0, 0.0, 10.0, 10.0),
        )

        self.assertEqual(int(mask[50, 25]), 0)
        self.assertEqual(int(mask[50, 60]), 1)
        self.assertEqual(int(mask[50, 90]), 1)

    def test_boundary_grid_mask_ignores_feature_outside_active_bbox(self) -> None:
        outside = [(20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 30.0), (20.0, 20.0)]

        mask = gis_matrix.boundary_grid_mask(
            [[outside]],
            rows=20,
            cols=20,
            bbox=(0.0, 0.0, 10.0, 10.0),
        )

        self.assertEqual(int(np.count_nonzero(mask)), 0)

    def test_boundary_feature_index_prefers_smallest_overlapping_boundary(self) -> None:
        outer = gis_matrix.GISBoundary(
            rings=[[(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0), (0.0, 0.0)]],
            attributes={"NAME": "outer"},
        )
        inner = gis_matrix.GISBoundary(
            rings=[[(4.0, 4.0), (8.0, 4.0), (8.0, 8.0), (4.0, 8.0), (4.0, 4.0)]],
            attributes={"NAME": "inner"},
        )

        index = gis_matrix.BoundaryFeatureIndex.build([outer, inner])

        self.assertEqual(index.find_containing((6.0, 6.0)), 1)
        self.assertEqual(index.find_containing((2.0, 2.0)), 0)
        self.assertIsNone(index.find_containing((30.0, 30.0)))

    def test_polyline_spatial_index_reduces_boundary_clip_candidates(self) -> None:
        records = [
            gis_matrix.GISPolyline(parts=[[(float(index * 10), 0.0), (float(index * 10 + 5), 0.0)]])
            for index in range(100)
        ]
        spatial_index = gis_matrix.PolylineSpatialIndex.build(records)
        ring = [(495.0, -1.0), (510.0, -1.0), (510.0, 1.0), (495.0, 1.0), (495.0, -1.0)]

        candidates = spatial_index.query((495.0, -1.0, 510.0, 1.0))
        clipped = gis_matrix.clip_polylines_to_boundary(records, [ring], spatial_index=spatial_index)

        self.assertLess(len(candidates), len(records) // 10)
        self.assertGreater(len(clipped), 0)

    def test_active_gis_viewport_uses_rotated_render_geometry(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app._gis_active_records = [gis_matrix.GISPolyline(parts=[[(0.0, 5.0), (10.0, 5.0)]])]
        app._gis_active_bbox = (0.0, 0.0, 10.0, 10.0)
        app._gis_active_render_records = None
        app._gis_active_render_bbox = None
        app._gis_active_render_spatial_index = None
        app._gis_active_render_rotation = None
        app._gis_viewport_render_cache = {}
        app._gis_viewport_render_generation = 0
        app._gis_viewport_pending_keys = set()
        app._zoom = 4.0
        app.source_kind = "gis"
        app.rotation_var = mock.Mock()
        app.rotation_var.get.return_value = 90.0

        stack = gui.NFMATApp._active_gis_viewport_layer_stack(app, (4.0, -1.0, 6.0, 11.0))

        self.assertTrue(stack)
        geometry = stack[0][1][0]
        xs = [point[0] for record in geometry for part in record.parts for point in part]
        ys = [point[1] for record in geometry for part in record.parts for point in part]
        self.assertTrue(all(abs(float(x) - 5.0) < 1e-6 for x in xs))
        self.assertAlmostEqual(min(ys), 0.0)
        self.assertAlmostEqual(max(ys), 10.0)


if __name__ == "__main__":
    unittest.main()
