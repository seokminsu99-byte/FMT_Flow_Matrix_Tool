# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Uses NumPy: https://github.com/numpy/numpy (BSD-3-Clause plus bundled notices).
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
"""Small state/IO helpers shared by the desktop workspace and its tests."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_settings_object(path: str | Path) -> dict:
    """Invalid settings never prevent startup and are never silently overwritten."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def validated_direction_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix)
    if values.ndim != 2 or not values.size:
        raise ValueError("행렬은 비어 있지 않은 2차원 배열이어야 합니다.")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError("행렬에 숫자가 아닌 값 또는 유효하지 않은 값이 있습니다.")
    if not np.all(np.isin(values, [0, 1, 2, 3, 4])):
        raise ValueError("방향값은 정수 0, 1, 2, 3, 4만 사용할 수 있습니다.")
    return values.astype(np.uint8, copy=True)


def intersect_region_masks(shape: tuple[int, int], *masks) -> np.ndarray | None:
    combined = None
    for mask in masks:
        if mask is None:
            continue
        value = np.asarray(mask)
        if value.shape != tuple(shape) or not np.all(np.isin(value, [0, 1])):
            raise ValueError(f"유역/LASSO 마스크가 현재 격자 {shape}와 일치하지 않습니다.")
        combined = (value > 0).copy() if combined is None else combined & (value > 0)
    return None if combined is None else combined.astype(np.uint8)


def resample_mask_centers(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Keep a boundary by sampling new cell centers, never by expanding any overlap."""
    source = np.asarray(mask)
    rows, cols = map(int, shape)
    if source.ndim != 2 or not source.size or rows < 1 or cols < 1:
        raise ValueError("잘못된 마스크 크기입니다.")
    ri = np.minimum(((np.arange(rows) + .5) * source.shape[0] / rows).astype(int), source.shape[0] - 1)
    ci = np.minimum(((np.arange(cols) + .5) * source.shape[1] / cols).astype(int), source.shape[1] - 1)
    return (source[np.ix_(ri, ci)] > 0).astype(np.uint8)


def basin_mask_from_world_rings(ring_groups, shape, bbox, rotation_degrees=0.0) -> np.ndarray:
    """Cell-center point-in-polygon, with even/odd holes and union of features.

    Inverse-rotate grid centers about the original GIS extent, matching the
    matrix renderer. No clamping of outside vertices onto the grid boundary.
    """
    rows, cols = map(int, shape)
    xmin, ymin, xmax, ymax = map(float, bbox)
    if rows <= 0 or cols <= 0 or not np.all(np.isfinite([xmin, ymin, xmax, ymax, rotation_degrees])):
        raise ValueError("유역 격자/좌표 범위가 유효하지 않습니다.")
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("유역 좌표 범위의 폭과 높이는 양수여야 합니다.")
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    angle = math.radians(float(rotation_degrees))
    c, s = math.cos(angle), math.sin(angle)
    corners = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]], dtype=float)
    dx, dy = corners[:, 0] - cx, corners[:, 1] - cy
    rx, ry = cx + c * dx - s * dy, cy + s * dx + c * dy
    gx = rx.min() + (np.arange(cols) + .5) / cols * (rx.max() - rx.min())
    gy = ry.max() - (np.arange(rows) + .5) / rows * (ry.max() - ry.min())
    xx, yy = np.meshgrid(gx - cx, gy - cy)
    x, y = cx + c * xx + s * yy, cy - s * xx + c * yy
    result = np.zeros((rows, cols), dtype=bool)
    tolerance = max(xmax - xmin, ymax - ymin) * 1e-10
    for rings in ring_groups:
        inside_feature = np.zeros_like(result)
        on_boundary = np.zeros_like(result)
        for ring in rings:
            points = np.asarray(ring, dtype=float)
            if len(points) < 3:
                continue
            if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
                raise ValueError("유역 경계에 잘못된 좌표가 있습니다.")
            in_ring = np.zeros_like(result)
            for (x0, y0), (x1, y1) in zip(points, np.roll(points, -1, axis=0)):
                if x0 == x1 and y0 == y1:
                    continue
                if y0 != y1:
                    crossing = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0)
                    in_ring ^= crossing
                cross = (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)
                on_boundary |= (np.abs(cross) <= tolerance * math.hypot(x1 - x0, y1 - y0)) & (x >= min(x0, x1) - tolerance) & (x <= max(x0, x1) + tolerance) & (y >= min(y0, y1) - tolerance) & (y <= max(y0, y1) + tolerance)
            inside_feature ^= in_ring
        result |= inside_feature | on_boundary
    return result.astype(np.uint8)
