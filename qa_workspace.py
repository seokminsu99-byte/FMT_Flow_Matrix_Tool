# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Uses OpenCV: https://github.com/opencv/opencv (Apache-2.0); opencv-python wheel notices apply.
# Uses NumPy: https://github.com/numpy/numpy (BSD-3-Clause plus bundled notices).
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
"""Reproducible synthetic trunk/branch workspace for UI and export verification."""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import cv2
import numpy as np

import gui
from gamma_index import load_plena_input_matrix
from solution import direction_cells_reaching_outlets


def populate_demo(app, *, external_pipe=False, reverse_branch=False):
    matrix = np.zeros((30, 38), dtype=np.uint8)
    matrix[3:27, 24] = 2
    matrix[26, 24] = 1
    matrix[8, 4:24] = 1
    matrix[14, 12:24] = 1
    matrix[20, 25:34] = 3
    matrix[4:8, 8] = 2
    if external_pipe:
        # Surveyed branch outside the fill basin, joining the same trunk/outlet.
        matrix[2:8, 1] = 2
        matrix[8, 1:4] = 1
        if reverse_branch:
            matrix[8, 1] = 4  # wrong downstream direction; later BFS must repair it
    basin = np.zeros_like(matrix)
    basin[2:28, 2:36] = 1
    basin[22:, :10] = 0
    image = np.full((660, 836, 3), 255, dtype=np.uint8)
    colors = {1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
    for row, col in np.argwhere(matrix > 0):
        dr, dc = colors[int(matrix[row, col])]
        a = (int((col + .5) * 22), int((row + .5) * 22))
        b = (int((col + .5 + dc) * 22), int((row + .5 + dr) * 22))
        cv2.arrowedLine(image, a, b, (85, 65, 40), 2, cv2.LINE_AA, tipLength=.28)
    app.source_kind = "image"
    app.img_path = "NFMAT6_검증용_간선지선.png"
    app._source_image_bgr = image.copy()
    app._set_display_image_from_bgr(image)
    app.A, app.B = matrix.shape
    app.rows_var.set(app.A)
    app.base_matrix = matrix.copy()
    app.current_matrix = matrix.copy()
    app.base_confidence_matrix = (matrix > 0).astype(np.float32)
    app.confidence_matrix = app.base_confidence_matrix.copy()
    app.base_model_applied_mask = np.zeros_like(matrix)
    app.model_applied_mask = np.zeros_like(matrix)
    for name in ("occ_matrix", "path_occ_matrix", "support_mask"):
        setattr(app, name, (matrix > 0).astype(np.uint8))
    app.unresolved_mask = np.zeros_like(matrix)
    app.user_edit_mask = np.zeros_like(matrix)
    app._basin_cell_mask = basin
    app.outlet_cell = (26, 24)
    app.outlet_cells = {(26, 24)}
    app.selected_cell = (20, 24)
    app._record_support_baseline()
    app.reset_view()
    app._update_matrix_preview()
    app.status_var.set("검증용 합성 관망 · 간선 1개와 지선 합류 · [빈 칸 채우기]로 면적 배수를 확인하세요.")
    return matrix, basin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--size", default="1450x900")
    parser.add_argument("--external-pipe", action="store_true")
    parser.add_argument("--reverse-branch", action="store_true")
    parser.add_argument("--bfs-after-fill", action="store_true")
    args = parser.parse_args()
    app = gui.NFMATApp()
    app.title("NFMAT6 · 합성 관망 검증")
    app.geometry(args.size)
    original, basin = populate_demo(
        app, external_pipe=args.external_pipe or args.reverse_branch,
        reverse_branch=args.reverse_branch,
    )
    if args.demo:
        app.mainloop()
        return
    app.withdraw()
    callbacks = []
    app.report_callback_exception = lambda *error: callbacks.append(str(error))
    selected_outlet = app.outlet_cell
    if args.bfs_after_fill:
        app._set_outlet_cells(set())  # fill must not require any Outlet
    app.fill_empty_cells()
    deadline = time.monotonic() + 30
    while app._task_busy and time.monotonic() < deadline:
        app.update()
        time.sleep(.01)
    if app._task_busy:
        app.destroy()
        raise RuntimeError("Fill UI job did not finish in 30 seconds")
    assert not callbacks, callbacks
    assert app._has_area_fill()
    assert "저신뢰 셀: 0" in app.summary_var.get()
    assert "수정된 셀: 0" in app.summary_var.get()
    np.testing.assert_array_equal(app.current_matrix[original > 0], original[original > 0])
    np.testing.assert_array_equal(app.current_matrix[basin == 0], original[basin == 0])
    if args.external_pipe or args.reverse_branch:
        # Attach one step left to the nearby branch, not the distant trunk or
        # the path with fewer downstream steps to the outlet.
        assert app.current_matrix[5, 2] == 3
        assert app._area_fill_mask[5, 2] == 1
    pipe_cells = list(map(tuple, np.argwhere(original > 0)))
    reached = direction_cells_reaching_outlets(app.current_matrix, pipe_cells)
    assert np.all(reached[app.current_matrix > 0])
    if args.reverse_branch:
        reaches_outlet = direction_cells_reaching_outlets(app.current_matrix, [selected_outlet])
        assert not reaches_outlet[5, 2]  # still attach to nearest pipe before BFS
    expected_after_undo = original
    if args.bfs_after_fill:
        app.enable_outlet_pick_mode()
        assert app.outlet_pick_mode
        origin_x, origin_y = app._display_origin
        disp_w, disp_h = app._display_size
        app.on_canvas_click(SimpleNamespace(
            x=origin_x + (selected_outlet[1] + .5) * disp_w / app.B,
            y=origin_y + (selected_outlet[0] + .5) * disp_h / app.A,
        ))
        assert app.outlet_cell == selected_outlet
        app.apply_bfs_assist()
        deadline = time.monotonic() + 30
        while app._task_busy and time.monotonic() < deadline:
            app.update()
            time.sleep(.01)
        assert not app._task_busy, 'BFS UI job did not finish in 30 seconds'
        assert not callbacks, callbacks
        assert 'BFS' in app.status_var.get(), app.status_var.get()
        assert app._has_area_fill()
        if args.reverse_branch:
            assert app.current_matrix[5, 2] == 3  # nearby branch repaired, still one step left
        expected_after_undo = app.current_matrix.copy()
        expected_after_undo[app._area_fill_mask > 0] = 0
        reached = direction_cells_reaching_outlets(app.current_matrix, [app.outlet_cell])
        assert np.all(reached[app.current_matrix > 0])
    filled_count = int(app._area_fill_mask.sum())
    with tempfile.TemporaryDirectory() as directory:
        exported_path = Path(directory) / "synthetic_basin.txt"
        app._write_plena_input_file(str(exported_path))
        exported, outlet = load_plena_input_matrix(exported_path)
        expected_export, _, _ = app._matrix_for_plena()
        np.testing.assert_array_equal(exported, expected_export)
        if not args.reverse_branch or args.bfs_after_fill:
            np.testing.assert_array_equal(exported, app.current_matrix)
        assert outlet == app.outlet_cell
    app.undo_fill_empty_cells()
    np.testing.assert_array_equal(app.current_matrix, expected_after_undo)
    app.destroy()
    print(f"GUI_FILL_EXPORT_UNDO_OK original={int(np.count_nonzero(original))} filled={filled_count} callbacks={len(callbacks)} external_pipe={args.external_pipe} reverse_branch={args.reverse_branch} bfs_after_fill={args.bfs_after_fill}")


if __name__ == "__main__":
    main()
