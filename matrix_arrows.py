# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Uses OpenCV: https://github.com/opencv/opencv (Apache-2.0); opencv-python wheel notices apply.
# Uses NumPy: https://github.com/numpy/numpy (BSD-3-Clause plus bundled notices).
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


Cell = Tuple[int, int]

_DIR_TO_STEP: Dict[int, Cell] = {
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
    4: (-1, 0),
}

_DIR_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    1: (64, 64, 230),
    2: (72, 176, 72),
    3: (230, 126, 64),
    4: (196, 80, 176),
}


def _cell_center(row: int, col: int, cell_px: int, margin: int) -> Tuple[int, int]:
    return (
        int(round(margin + (col + 0.5) * cell_px)),
        int(round(margin + (row + 0.5) * cell_px)),
    )


def _draw_arrow_head(
    img: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    head_len: int,
    head_width: int,
) -> None:
    vec = np.asarray([float(end[0] - start[0]), float(end[1] - start[1])], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return
    unit = vec / norm
    perp = np.asarray([-unit[1], unit[0]], dtype=np.float32)
    tip = np.asarray([float(end[0]), float(end[1])], dtype=np.float32)
    base = tip - unit * float(head_len)
    pts = np.asarray(
        [
            [tip[0], tip[1]],
            [base[0] + perp[0] * float(head_width), base[1] + perp[1] * float(head_width)],
            [base[0] - perp[0] * float(head_width), base[1] - perp[1] * float(head_width)],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(img, pts, color, lineType=cv2.LINE_AA)


def render_direction_matrix_arrows(
    matrix: np.ndarray,
    *,
    cell_px: int = 42,
    margin: int = 26,
    show_grid: bool = True,
    highlight_mask: np.ndarray | None = None,
    area_fill_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Render a direction matrix as connected arrow paths.

    Direction codes follow the program convention: 1=east, 2=south, 3=west, 4=north.
    Each non-zero cell emits one segment in its own flow direction. Connected
    neighbor cells are drawn center-to-center, and every segment receives a
    visible arrow head so long runs still show direction locally.
    """
    mat = np.asarray(matrix, dtype=np.uint8)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    rows, cols = mat.shape
    highlight = None
    if highlight_mask is not None:
        highlight = np.asarray(highlight_mask, dtype=np.uint8)
        if highlight.shape != mat.shape:
            raise ValueError("highlight_mask shape must match matrix")
    area = None
    if area_fill_mask is not None:
        area = np.asarray(area_fill_mask, dtype=np.uint8)
        if area.shape != mat.shape:
            raise ValueError("area_fill_mask shape must match matrix")
    cell_px = max(4, int(cell_px))
    margin = max(8, int(margin))
    width = cols * cell_px + margin * 2
    height = rows * cell_px + margin * 2
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    if area is not None:
        for row, col in np.argwhere((area > 0) & (mat > 0)):
            cv2.rectangle(img, (margin + int(col) * cell_px, margin + int(row) * cell_px),
                          (margin + (int(col) + 1) * cell_px, margin + (int(row) + 1) * cell_px),
                          (248, 238, 217), -1)

    if show_grid:
        grid_color = (232, 235, 239)
        for row in range(rows + 1):
            y = margin + row * cell_px
            cv2.line(img, (margin, y), (width - margin, y), grid_color, 1, lineType=cv2.LINE_AA)
        for col in range(cols + 1):
            x = margin + col * cell_px
            cv2.line(img, (x, margin), (x, height - margin), grid_color, 1, lineType=cv2.LINE_AA)

    segments = []
    for row in range(rows):
        for col in range(cols):
            direction = int(mat[row, col])
            step = _DIR_TO_STEP.get(direction)
            if step is None:
                continue
            start = _cell_center(row, col, cell_px, margin)
            next_cell = (row + step[0], col + step[1])
            if 0 <= next_cell[0] < rows and 0 <= next_cell[1] < cols and int(mat[next_cell]) != 0:
                end = _cell_center(next_cell[0], next_cell[1], cell_px, margin)
            else:
                end = (
                    int(round(start[0] + step[1] * cell_px * 0.48)),
                    int(round(start[1] + step[0] * cell_px * 0.48)),
                )
                end = (
                    int(np.clip(end[0], margin, width - margin)),
                    int(np.clip(end[1], margin, height - margin)),
                )
            segments.append((start, end, direction, bool(area[row, col]) if area is not None else False))

    if not segments:
        return img

    thickness = max(1, int(round(cell_px * 0.10)))
    halo_thickness = thickness + max(3, int(round(cell_px * 0.08)))
    head_len = max(2, int(round(cell_px * 0.34)))
    head_width = max(1, int(round(cell_px * 0.21)))

    for start, end, _direction, _is_area in segments:
        cv2.line(img, start, end, (255, 255, 255), halo_thickness, lineType=cv2.LINE_AA)
        _draw_arrow_head(img, start, end, (255, 255, 255), head_len + 3, head_width + 3)

    for start, end, direction, is_area in segments:
        color = (173, 138, 22) if is_area else _DIR_COLORS_BGR.get(direction, (64, 72, 88))
        cv2.line(img, start, end, color, thickness, lineType=cv2.LINE_AA)
        _draw_arrow_head(img, start, end, color, head_len, head_width)

    if highlight is not None:
        flood_cells = np.argwhere((highlight > 0) & (mat > 0))
        outer_width = max(4, int(round(cell_px * 0.13)))
        inner_width = max(2, int(round(cell_px * 0.07)))
        inset = max(2, int(round(cell_px * 0.08)))
        for row, col in flood_cells:
            x1 = margin + int(col) * cell_px + inset
            y1 = margin + int(row) * cell_px + inset
            x2 = margin + (int(col) + 1) * cell_px - inset
            y2 = margin + (int(row) + 1) * cell_px - inset
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), outer_width, lineType=cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), (75, 45, 255), inner_width, lineType=cv2.LINE_AA)

    return img
