# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Uses NumPy: https://github.com/numpy/numpy (BSD-3-Clause plus bundled notices).
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
"""Real-Tk regression: GIS row change, rectangular editing and rotation."""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest import mock

import numpy as np

import gui
from gis_matrix import GISPolyline, matrix_from_loaded_polylines, clip_polylines_to_boundary


def wait_for_job(app):
    deadline = time.monotonic() + 30
    while app._task_busy and time.monotonic() < deadline:
        app.update()
        time.sleep(.01)
    assert not app._task_busy, "GIS worker timeout"


def main():
    app = gui.NFMATApp()
    app.withdraw()
    app._save_settings = mock.Mock()  # QA must not change saved user settings.
    errors = []
    app.report_callback_exception = lambda *error: errors.append(str(error))
    records = [GISPolyline(parts=[[(1., 8.), (6., 8.), (6., 1.)]], start_height=20., end_height=10.)]
    bbox = (0., 0., 10., 10.)
    app.source_kind = "gis"
    app.img_path = "synthetic_GIS.shp"
    app.rotation_var.set(0.)
    app._gis_source_path = app.img_path
    app._gis_rows = 10
    app._gis_active_records = records
    app._gis_active_bbox = bbox
    app._gis_active_source_meta = {}
    app._basin_ring_groups = [[[(0., 0.), (10., 0.), (0., 10.)]]]
    app._apply_gis_result_to_state(matrix_from_loaded_polylines(records, rows=10, bbox=bbox))
    original = app.current_matrix.copy()
    app.rows_var.set(20)
    assert not app._ensure_grid()
    np.testing.assert_array_equal(app.current_matrix, original)
    app.recalculate_gis_directions()
    wait_for_job(app)
    expected = matrix_from_loaded_polylines(records, rows=20, bbox=bbox).matrix
    np.testing.assert_array_equal(app.current_matrix, expected)
    assert app.A == 20 and app._gis_rows == 20

    # A GIS fill boundary must not remove the surrounding rectangular editor.
    outside = tuple(map(int, np.argwhere((app._basin_cell_mask == 0) & (app.current_matrix == 0))[0]))
    app._set_selected_cell(*outside)
    app._set_selected_value(3)
    assert app.current_matrix[outside] == 3
    app.matrix_arrow_view = True
    app.outlet_pick_mode = True
    app.start_empty_grid()
    assert not app.matrix_arrow_view and not app.outlet_pick_mode
    app._set_selected_cell(*outside)
    app.on_canvas_key(SimpleNamespace(keysym="1"))
    assert app.current_matrix[outside] == 1

    # Rotation uses original GIS data, not the manually edited/blank preview.
    app.rotation_var.set(90.)
    with mock.patch.object(gui.messagebox, "askyesno", return_value=True):
        app.apply_rotation()
    rotated = matrix_from_loaded_polylines(records, rows=20, bbox=bbox, rotation_degrees=90.).matrix
    np.testing.assert_array_equal(app.current_matrix, rotated)

    # Explicit LASSO is preserved when regridding; cancelled regrid is read-only.
    app.roi_cell_mask = np.ones_like(rotated)
    app.roi_cell_mask[:, :10] = 0
    app.rows_var.set(30)
    with mock.patch.object(gui.messagebox, "askyesno", return_value=False):
        app.recalculate_gis_directions()
    np.testing.assert_array_equal(app.current_matrix, rotated)
    with mock.patch.object(gui.messagebox, "askyesno", return_value=True):
        app.recalculate_gis_directions()
    wait_for_job(app)
    assert app.A == 30
    assert app.roi_cell_mask.shape == app.current_matrix.shape
    assert not app.current_matrix[:, :15].any()

    # Global GIS -> draft LASSO -> direction correction must analyze only
    # selected/clipped vectors, not the entire pipe layer or the old subset.
    project_records = [
        GISPolyline(parts=[[(5., 5.), (6., 6.)]], start_height=20., end_height=10.),
        GISPolyline(parts=[[(0., 8.), (15., 8.)]], start_height=30., end_height=10.),
        GISPolyline(parts=[[(16., 16.), (19., 19.)]], start_height=30., end_height=10.),
    ]
    app.source_kind = "gis_project"
    app.rotation_var.set(0.)
    app._gis_map_bbox = (0., 0., 20., 20.)
    app._set_display_image_from_bgr(np.full((401, 401, 3), 255, dtype=np.uint8))
    app.roi_polygon_points = [(80., 320.), (240., 320.), (80., 160.)]
    app.roi_cell_mask = None
    app.lasso_mode = True
    app.rows_var.set(18)
    app._gis_pipe_layer_id = "qa_pipe"
    app._gis_layers = {"qa_pipe": {"path": "synthetic_GIS.shp", "name": "QA pipes", "kind": "pipe",
                                    "geometry": project_records, "bbox": app._gis_map_bbox, "meta": {}}}
    rings = [[(4., 4.), (12., 4.), (4., 12.)]]
    expected_records = clip_polylines_to_boundary(project_records, rings)
    with mock.patch.object(app, "_load_selected_pipe_records", return_value=({}, project_records, None, None)), \
         mock.patch.object(app, "convert_selected_pipe_layer_to_matrix") as whole_layer:
        app.recalculate_gis_directions()
    whole_layer.assert_not_called()
    assert app.source_kind == "gis" and app._basin_is_lasso
    assert app._gis_active_bbox == (4., 4., 12., 12.)
    assert app.A == 18 and len(app._gis_active_records) == 2
    assert app._gis_active_records == expected_records
    expected = matrix_from_loaded_polylines(expected_records, rows=18, bbox=app._gis_active_bbox).matrix
    expected[app._active_edit_mask() == 0] = 0
    np.testing.assert_array_equal(app.current_matrix, expected)
    assert not app.current_matrix[app._active_edit_mask() == 0].any()
    assert not errors, errors
    app.destroy()
    print("GIS_ROWS_EDIT_ROTATE_LASSO_OK rows=10->20->30 callbacks=0")
    print("GIS_GLOBAL_LASSO_ONLY_OK input_records=3 clipped_records=2 rows=18 outside_lasso=0")


if __name__ == "__main__":
    main()
