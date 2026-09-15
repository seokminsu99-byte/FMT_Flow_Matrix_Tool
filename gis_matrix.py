# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Uses OpenCV: https://github.com/opencv/opencv (Apache-2.0); opencv-python wheel notices apply.
# Uses NumPy: https://github.com/numpy/numpy (BSD-3-Clause plus bundled notices).
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import re
import struct
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from crs_support import (
    CRSDefinition,
    crs_equivalent,
    crs_from_meta,
    read_shapefile_crs_meta,
    transform_nested_points,
)


Cell = Tuple[int, int]
Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
DEFAULT_START_HEIGHT_FIELD = "ST_PIP_HIT"
DEFAULT_END_HEIGHT_FIELD = "ET_PIP_HIT"
SUPPORTED_POLYLINE_TYPES = {3: "PolyLine", 13: "PolyLineZ"}
SUPPORTED_POLYGON_TYPES = {5: "Polygon", 15: "PolygonZ", 25: "PolygonM"}


@dataclass
class GISPolyline:
    parts: List[List[Point]]
    start_height: float | None = None
    end_height: float | None = None
    slope_reference_parts: List[List[Point]] | None = None


@dataclass
class GISBoundary:
    rings: List[List[Point]]
    attributes: Dict[str, str]


def bbox_from_polylines(records: Sequence[GISPolyline]) -> BBox:
    return _bbox_from_parts([part for record in records for part in record.parts])


def bbox_from_boundaries(boundaries: Sequence[GISBoundary]) -> BBox:
    return _bbox_from_parts([ring for boundary in boundaries for ring in boundary.rings])


def reproject_polylines(
    records: Sequence[GISPolyline],
    source_crs: CRSDefinition | str,
    target_crs: CRSDefinition | str,
) -> Tuple[List[GISPolyline], BBox]:
    normalized = list(records)
    primary_groups = [record.parts for record in normalized]
    slope_groups = [record.slope_reference_parts or [] for record in normalized]
    transformed = transform_nested_points(primary_groups + slope_groups, source_crs, target_crs)
    split = len(normalized)
    output = [
        GISPolyline(
            parts=transformed[index],
            start_height=record.start_height,
            end_height=record.end_height,
            slope_reference_parts=(
                transformed[split + index]
                if record.slope_reference_parts is not None
                else None
            ),
        )
        for index, record in enumerate(normalized)
    ]
    return output, bbox_from_polylines(output)


def reproject_boundaries(
    boundaries: Sequence[GISBoundary],
    source_crs: CRSDefinition | str,
    target_crs: CRSDefinition | str,
) -> Tuple[List[GISBoundary], BBox]:
    normalized = list(boundaries)
    transformed = transform_nested_points([boundary.rings for boundary in normalized], source_crs, target_crs)
    output = [
        GISBoundary(rings=transformed[index], attributes=dict(boundary.attributes))
        for index, boundary in enumerate(normalized)
    ]
    return output, bbox_from_boundaries(output)


def _grid_cell_for_point(point: Point, bbox: BBox, grid_size: int) -> Tuple[int, int]:
    xmin, ymin, xmax, ymax = bbox
    x_span = max(float(xmax) - float(xmin), 1e-12)
    y_span = max(float(ymax) - float(ymin), 1e-12)
    col = int(np.clip(math.floor((float(point[0]) - float(xmin)) / x_span * grid_size), 0, grid_size - 1))
    row = int(np.clip(math.floor((float(point[1]) - float(ymin)) / y_span * grid_size), 0, grid_size - 1))
    return row, col


def _grid_cells_for_bbox(query_bbox: BBox, bbox: BBox, grid_size: int) -> List[Tuple[int, int]]:
    first_row, first_col = _grid_cell_for_point((query_bbox[0], query_bbox[1]), bbox, grid_size)
    last_row, last_col = _grid_cell_for_point((query_bbox[2], query_bbox[3]), bbox, grid_size)
    return [
        (row, col)
        for row in range(first_row, last_row + 1)
        for col in range(first_col, last_col + 1)
    ]


def _add_bbox_to_bins(
    bins: Dict[Tuple[int, int], List[int]],
    feature_index: int,
    feature_bbox: BBox,
    bbox: BBox,
    grid_size: int,
) -> None:
    first_row, first_col = _grid_cell_for_point((feature_bbox[0], feature_bbox[1]), bbox, grid_size)
    last_row, last_col = _grid_cell_for_point((feature_bbox[2], feature_bbox[3]), bbox, grid_size)
    for row in range(first_row, last_row + 1):
        for col in range(first_col, last_col + 1):
            bins.setdefault((row, col), []).append(feature_index)


def _bbox_from_parts(parts: Sequence[Sequence[Point]]) -> BBox:
    xmin = ymin = math.inf
    xmax = ymax = -math.inf
    for part in parts:
        for point in part:
            x, y = float(point[0]), float(point[1])
            xmin = min(xmin, x)
            ymin = min(ymin, y)
            xmax = max(xmax, x)
            ymax = max(ymax, y)
    if not math.isfinite(xmin):
        raise ValueError("Geometry has no points.")
    return xmin, ymin, xmax, ymax


@dataclass
class BoundaryFeatureIndex:
    boundaries: List[GISBoundary]
    bboxes: List[BBox]
    bbox: BBox
    grid_size: int
    bins: Dict[Tuple[int, int], List[int]]
    clip_indexes: Dict[int, "BoundaryClipIndex"]

    @classmethod
    def build(cls, boundaries: Sequence[GISBoundary]) -> "BoundaryFeatureIndex":
        normalized = list(boundaries)
        bboxes = [_bbox_from_parts(boundary.rings) for boundary in normalized]
        bbox = combine_bboxes(bboxes) if bboxes else (0.0, 0.0, 0.0, 0.0)
        grid_size = int(np.clip(round(math.sqrt(max(len(normalized), 1)) * 1.5), 4, 128))
        bins: Dict[Tuple[int, int], List[int]] = {}
        for feature_index, feature_bbox in enumerate(bboxes):
            _add_bbox_to_bins(bins, feature_index, feature_bbox, bbox, grid_size)
        return cls(
            boundaries=normalized,
            bboxes=bboxes,
            bbox=bbox,
            grid_size=grid_size,
            bins=bins,
            clip_indexes={},
        )

    def candidates_for_point(self, point: Point) -> List[int]:
        if not self.boundaries or not _bbox_contains_point(self.bbox, point):
            return []
        cell = _grid_cell_for_point(point, self.bbox, self.grid_size)
        return [
            index
            for index in self.bins.get(cell, [])
            if _bbox_contains_point(self.bboxes[index], point)
        ]

    def find_containing(self, point: Point) -> int | None:
        matches: List[int] = []
        for index in self.candidates_for_point(point):
            clip_index = self.clip_indexes.get(index)
            if clip_index is None:
                clip_index = BoundaryClipIndex.build(self.boundaries[index].rings)
                self.clip_indexes[index] = clip_index
            if clip_index.contains(point):
                matches.append(index)
        if not matches:
            return None
        return min(
            matches,
            key=lambda index: max(self.bboxes[index][2] - self.bboxes[index][0], 0.0)
            * max(self.bboxes[index][3] - self.bboxes[index][1], 0.0),
        )


@dataclass
class PolylineSpatialIndex:
    records: List[GISPolyline]
    bboxes: List[BBox]
    bbox: BBox
    grid_size: int
    bins: Dict[Tuple[int, int], List[int]]

    @classmethod
    def build(cls, records: Sequence[GISPolyline]) -> "PolylineSpatialIndex":
        normalized = list(records)
        bboxes = [_bbox_from_parts(record.parts) for record in normalized]
        bbox = combine_bboxes(bboxes) if bboxes else (0.0, 0.0, 0.0, 0.0)
        grid_size = int(np.clip(round(math.sqrt(max(len(normalized), 1)) * 0.75), 8, 256))
        bins: Dict[Tuple[int, int], List[int]] = {}
        for record_index, record_bbox in enumerate(bboxes):
            _add_bbox_to_bins(bins, record_index, record_bbox, bbox, grid_size)
        return cls(records=normalized, bboxes=bboxes, bbox=bbox, grid_size=grid_size, bins=bins)

    def query(self, query_bbox: BBox) -> List[GISPolyline]:
        if not self.records or not _bbox_intersects(self.bbox, query_bbox):
            return []
        candidates: set[int] = set()
        for cell in _grid_cells_for_bbox(query_bbox, self.bbox, self.grid_size):
            candidates.update(self.bins.get(cell, []))
        return [
            self.records[index]
            for index in sorted(candidates)
            if _bbox_intersects(self.bboxes[index], query_bbox)
        ]


@dataclass
class BoundaryClipIndex:
    rings: List[List[Point]]
    bbox: BBox
    edges: List[Tuple[Point, Point, BBox]]
    y_bins: List[List[int]]

    @classmethod
    def build(cls, rings: Sequence[Sequence[Point]]) -> "BoundaryClipIndex":
        normalized = [list(ring) for ring in rings if len(ring) >= 3]
        points = [point for ring in normalized for point in ring]
        bbox = _bbox_from_points(points)
        edges: List[Tuple[Point, Point, BBox]] = []
        for ring in normalized:
            closed = list(ring)
            if closed[0] != closed[-1]:
                closed.append(closed[0])
            for start, end in zip(closed[:-1], closed[1:]):
                edge_bbox = (
                    min(float(start[0]), float(end[0])),
                    min(float(start[1]), float(end[1])),
                    max(float(start[0]), float(end[0])),
                    max(float(start[1]), float(end[1])),
                )
                edges.append((start, end, edge_bbox))
        bin_count = int(np.clip(round(math.sqrt(max(len(edges), 1)) * 2.0), 16, 256))
        y_bins: List[List[int]] = [[] for _ in range(bin_count)]
        ymin, ymax = float(bbox[1]), float(bbox[3])
        span = max(ymax - ymin, 1e-12)
        for edge_index, (_start, _end, edge_bbox) in enumerate(edges):
            first = int(np.clip(math.floor((edge_bbox[1] - ymin) / span * bin_count), 0, bin_count - 1))
            last = int(np.clip(math.floor((edge_bbox[3] - ymin) / span * bin_count), 0, bin_count - 1))
            for bin_index in range(first, last + 1):
                y_bins[bin_index].append(edge_index)
        return cls(rings=normalized, bbox=bbox, edges=edges, y_bins=y_bins)

    def _edge_indices_for_y_range(self, ymin: float, ymax: float) -> List[int]:
        bin_count = len(self.y_bins)
        span = max(float(self.bbox[3]) - float(self.bbox[1]), 1e-12)
        first = int(np.clip(math.floor((float(ymin) - float(self.bbox[1])) / span * bin_count), 0, bin_count - 1))
        last = int(np.clip(math.floor((float(ymax) - float(self.bbox[1])) / span * bin_count), 0, bin_count - 1))
        candidates: set[int] = set()
        for bin_index in range(first, last + 1):
            candidates.update(self.y_bins[bin_index])
        return list(candidates)

    def candidate_edges_for_segment(self, start: Point, end: Point) -> List[Tuple[Point, Point]]:
        segment_bbox = (
            min(float(start[0]), float(end[0])),
            min(float(start[1]), float(end[1])),
            max(float(start[0]), float(end[0])),
            max(float(start[1]), float(end[1])),
        )
        if not _bbox_intersects(segment_bbox, self.bbox):
            return []
        return [
            (self.edges[index][0], self.edges[index][1])
            for index in self._edge_indices_for_y_range(segment_bbox[1], segment_bbox[3])
            if _bbox_intersects(segment_bbox, self.edges[index][2])
        ]

    def contains(self, point: Point) -> bool:
        px, py = float(point[0]), float(point[1])
        if not (self.bbox[0] - 1e-9 <= px <= self.bbox[2] + 1e-9 and self.bbox[1] - 1e-9 <= py <= self.bbox[3] + 1e-9):
            return False
        inside = False
        for index in self._edge_indices_for_y_range(py, py):
            start, end, _edge_bbox = self.edges[index]
            if _point_on_segment(point, start, end):
                return True
            x0, y0 = float(start[0]), float(start[1])
            x1, y1 = float(end[0]), float(end[1])
            if (y0 > py) == (y1 > py):
                continue
            crossing_x = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
            if crossing_x > px:
                inside = not inside
        return inside


@dataclass
class GISMatrixResult:
    matrix: np.ndarray
    occ: np.ndarray
    path_occ: np.ndarray
    support_mask: np.ndarray
    confidence: np.ndarray
    overlap_mask: np.ndarray
    image_bgr: np.ndarray
    meta: Dict[str, object]


def combine_bboxes(bboxes: Sequence[BBox]) -> BBox:
    valid = [tuple(float(value) for value in bbox) for bbox in bboxes if len(bbox) == 4]
    if not valid:
        raise ValueError("No GIS extents are available.")
    return (
        min(bbox[0] for bbox in valid),
        min(bbox[1] for bbox in valid),
        max(bbox[2] for bbox in valid),
        max(bbox[3] for bbox in valid),
    )


def derive_grid_cols(rows: int, bbox: BBox) -> int:
    xmin, ymin, xmax, ymax = bbox
    width = float(xmax - xmin)
    height = float(ymax - ymin)
    rows = max(2, int(rows))
    if width <= 1e-9 and height <= 1e-9:
        return rows
    if height <= 1e-9:
        return rows
    if width <= 1e-9:
        return 2
    aspect = float(np.clip(width / height, 0.05, 20.0))
    return max(2, int(round(rows * aspect)))


def _shape_type_name(shape_type: int) -> str:
    return (SUPPORTED_POLYLINE_TYPES | SUPPORTED_POLYGON_TYPES).get(int(shape_type), f"shape_type_{int(shape_type)}")


def _read_prj_unit_meta(shp_path: Path) -> Dict[str, object]:
    meta = read_shapefile_crs_meta(shp_path)
    if bool(meta.get("crs_valid")) or not bool(meta.get("prj_available")):
        return meta

    # Preserve unit reporting for legacy, incomplete WKT while marking the CRS invalid.
    prj_path = shp_path.with_suffix(".prj")
    text = prj_path.read_text(encoding="utf-8-sig", errors="ignore")
    matches = re.findall(r'UNIT\s*\[\s*"([^"]+)"\s*,\s*([0-9.+\-Ee]+)', text)
    unit_name = ""
    unit_factor: float | None = None
    if matches:
        unit_name, factor_text = matches[-1]
        try:
            unit_factor = float(factor_text)
        except ValueError:
            unit_factor = None
    lower_name = unit_name.lower()
    lower_text = text.lower()
    unit_type = "unknown"
    to_meter: float | None = None
    if "degree" in lower_name or (unit_factor is not None and abs(unit_factor - 0.0174532925199433) < 1e-12):
        unit_type = "degree"
    elif any(token in lower_name for token in ("metre", "meter", "mètre")) or (
        "projcs" in lower_text and unit_factor is not None and abs(unit_factor - 1.0) < 1e-9
    ):
        unit_type = "meter"
        to_meter = 1.0
    elif "foot" in lower_name or "feet" in lower_name:
        unit_type = "linear"
        to_meter = float(unit_factor) if unit_factor is not None else 0.3048
    elif unit_factor is not None and "projcs" in lower_text:
        unit_type = "linear"
        to_meter = float(unit_factor)
    meta.update(
        {
            "coordinate_unit_name": unit_name,
            "coordinate_unit_type": unit_type,
            "coordinate_unit_to_meter": to_meter,
        }
    )
    return meta


def _direction_from_step(src: Cell, dst: Cell) -> int:
    dr = int(dst[0] - src[0])
    dc = int(dst[1] - src[1])
    if dc > 0 and dr == 0:
        return 1
    if dr > 0 and dc == 0:
        return 2
    if dc < 0 and dr == 0:
        return 3
    if dr < 0 and dc == 0:
        return 4
    return 0


def _cell_for_point(x: float, y: float, rows: int, cols: int, bbox: BBox) -> Cell:
    xmin, ymin, xmax, ymax = bbox
    cell_w = (xmax - xmin) / max(float(cols), 1.0)
    cell_h = (ymax - ymin) / max(float(rows), 1.0)
    col = int(math.floor((float(x) - xmin) / max(cell_w, 1e-12)))
    row = int(math.floor((ymax - float(y)) / max(cell_h, 1e-12)))
    return int(np.clip(row, 0, rows - 1)), int(np.clip(col, 0, cols - 1))


def _bresenham_cells(start: Cell, end: Cell) -> List[Cell]:
    """Incremental line rasterization following Bresenham (1965).

    Method reference: Algorithm for Computer Control of a Digital Plotter,
    IBM Systems Journal 4(1), 25-30; https://doi.org/10.1147/sj.41.0025.
    This function expresses the method locally; it is not a vendored library.
    The separate four-neighbor bridge policy below is an FMT integration rule.
    """
    r0, c0 = start
    r1, c1 = end
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    r, c = r0, c0
    cells: List[Cell] = [(r, c)]
    if dc >= dr:
        err = dc / 2.0
        while c != c1:
            c += sc
            err -= dr
            if err < 0:
                r += sr
                err += dc
            cells.append((r, c))
    else:
        err = dr / 2.0
        while r != r1:
            r += sr
            err -= dc
            if err < 0:
                c += sc
                err += dr
            cells.append((r, c))
    return cells


def _dedupe_consecutive(cells: Sequence[Cell]) -> List[Cell]:
    deduped: List[Cell] = []
    for cell in cells:
        if not deduped or deduped[-1] != cell:
            deduped.append(cell)
    return deduped


def _expand_diagonal_steps(cells: Sequence[Cell], dx: float, dy: float) -> List[Cell]:
    """FMT postprocessing: add one orthogonal cell per diagonal step.

    Prioritize x when abs(dx) >= abs(dy), otherwise y, using source-coordinate
    displacement. This preserves each path's four-neighbor continuity, not
    global direction consistency, physical conduit length or Outlet reachability.
    """
    if not cells:
        return []
    expanded: List[Cell] = [cells[0]]
    horizontal_first = abs(float(dx)) >= abs(float(dy))
    for dst in cells[1:]:
        src = expanded[-1]
        dr = int(dst[0] - src[0])
        dc = int(dst[1] - src[1])
        if abs(dr) == 1 and abs(dc) == 1:
            bridge = (src[0], dst[1]) if horizontal_first else (dst[0], src[1])
            if bridge != src:
                expanded.append(bridge)
        elif abs(dr) + abs(dc) > 1:
            bridge_path = _bresenham_cells(src, dst)
            for bridge_cell in _expand_diagonal_steps(bridge_path, dx, dy)[1:-1]:
                if bridge_cell != expanded[-1]:
                    expanded.append(bridge_cell)
        if dst != expanded[-1]:
            expanded.append(dst)
    return _dedupe_consecutive(expanded)


def _segment_cell_path(p0: Point, p1: Point, rows: int, cols: int, bbox: BBox) -> List[Cell]:
    start = _cell_for_point(p0[0], p0[1], rows, cols, bbox)
    end = _cell_for_point(p1[0], p1[1], rows, cols, bbox)
    base = _bresenham_cells(start, end)
    return _expand_diagonal_steps(base, p1[0] - p0[0], p1[1] - p0[1])


def _should_reverse_by_height(start_height: float | None, end_height: float | None) -> bool:
    if start_height is None or end_height is None:
        return False
    if abs(float(start_height)) <= 1e-9 and abs(float(end_height)) <= 1e-9:
        return False
    return float(start_height) < float(end_height)


def _rotate_point(point: Point, center: Point, cos_a: float, sin_a: float) -> Point:
    x, y = float(point[0]), float(point[1])
    cx, cy = float(center[0]), float(center[1])
    dx = x - cx
    dy = y - cy
    return cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a


def rotate_polylines(
    records: Sequence[GISPolyline],
    bbox: BBox,
    rotation_degrees: float,
) -> Tuple[List[GISPolyline], BBox]:
    angle = float(rotation_degrees)
    if abs(angle) <= 1e-9:
        return list(records), bbox
    xmin, ymin, xmax, ymax = bbox
    center = ((float(xmin) + float(xmax)) * 0.5, (float(ymin) + float(ymax)) * 0.5)
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    rotated_records: List[GISPolyline] = []
    all_points: List[Point] = []
    for record in records:
        rotated_parts: List[List[Point]] = []
        for part in record.parts:
            rotated_part = [_rotate_point(point, center, cos_a, sin_a) for point in part]
            rotated_parts.append(rotated_part)
            all_points.extend(rotated_part)
        rotated_records.append(
            GISPolyline(
                parts=rotated_parts,
                start_height=record.start_height,
                end_height=record.end_height,
                slope_reference_parts=(
                    [
                        [_rotate_point(point, center, cos_a, sin_a) for point in part]
                        for part in record.slope_reference_parts
                    ]
                    if record.slope_reference_parts is not None
                    else None
                ),
            )
        )
    if not all_points:
        corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
        all_points = [_rotate_point(point, center, cos_a, sin_a) for point in corners]
    xs = [float(point[0]) for point in all_points]
    ys = [float(point[1]) for point in all_points]
    return rotated_records, (min(xs), min(ys), max(xs), max(ys))


def _rotate_bbox(bbox: BBox, rotation_degrees: float) -> BBox:
    angle = float(rotation_degrees)
    if abs(angle) <= 1e-9:
        return bbox
    xmin, ymin, xmax, ymax = bbox
    center = ((float(xmin) + float(xmax)) * 0.5, (float(ymin) + float(ymax)) * 0.5)
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    rotated = [_rotate_point(point, center, cos_a, sin_a) for point in corners]
    xs = [point[0] for point in rotated]
    ys = [point[1] for point in rotated]
    return min(xs), min(ys), max(xs), max(ys)


def prepare_rotated_polylines(
    records: Sequence[GISPolyline],
    bbox: BBox,
    rotation_degrees: float,
) -> Tuple[List[GISPolyline], BBox]:
    """Rotate geometry and return the exact world extent used by the grid."""
    rotated_records, _record_bbox = rotate_polylines(records, bbox, rotation_degrees)
    return rotated_records, _rotate_bbox(bbox, rotation_degrees)


def boundary_grid_mask(
    ring_groups: Sequence[Sequence[Sequence[Point]]],
    rows: int,
    cols: int,
    bbox: BBox,
    *,
    rotation_degrees: float = 0.0,
) -> np.ndarray:
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    angle = float(rotation_degrees)
    xmin, ymin, xmax, ymax = bbox
    center = ((float(xmin) + float(xmax)) * 0.5, (float(ymin) + float(ymax)) * 0.5)
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    used_bbox = _rotate_bbox(bbox, angle)
    used_xmin, used_ymin, used_xmax, used_ymax = used_bbox
    x_span = max(float(used_xmax) - float(used_xmin), 1e-12)
    y_span = max(float(used_ymax) - float(used_ymin), 1e-12)
    union = np.zeros((rows, cols), dtype=np.uint8)

    for rings in ring_groups:
        feature_points = [
            _rotate_point(point, center, cos_a, sin_a) if abs(angle) > 1e-9 else point
            for ring in rings
            for point in ring
        ]
        if not feature_points:
            continue
        feature_x = [float(point[0]) for point in feature_points]
        feature_y = [float(point[1]) for point in feature_points]
        if (
            max(feature_x) < used_xmin
            or min(feature_x) > used_xmax
            or max(feature_y) < used_ymin
            or min(feature_y) > used_ymax
        ):
            continue
        contours: List[np.ndarray] = []
        for ring in rings:
            if len(ring) < 3:
                continue
            points = [
                _rotate_point(point, center, cos_a, sin_a) if abs(angle) > 1e-9 else point
                for point in ring
            ]
            coords = np.asarray(points, dtype=np.float64)
            px = np.rint((coords[:, 0] - used_xmin) / x_span * max(cols - 1, 1))
            py = np.rint((used_ymax - coords[:, 1]) / y_span * max(rows - 1, 1))
            contour = np.column_stack(
                (
                    np.clip(px, 0, cols - 1),
                    np.clip(py, 0, rows - 1),
                )
            ).astype(np.int32)
            if len(contour) >= 3:
                contours.append(contour.reshape((-1, 1, 2)))
        if contours:
            feature_mask = np.zeros((rows, cols), dtype=np.uint8)
            cv2.fillPoly(feature_mask, contours, color=1, lineType=cv2.LINE_8)
            union |= feature_mask
    return union


def _bbox_from_points(points: Sequence[Point]) -> BBox:
    if not points:
        raise ValueError("Boundary has no points.")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_intersects(first: BBox, second: BBox) -> bool:
    return not (
        float(first[2]) < float(second[0])
        or float(second[2]) < float(first[0])
        or float(first[3]) < float(second[1])
        or float(second[3]) < float(first[1])
    )


def _bbox_contains_point(bbox: BBox, point: Point) -> bool:
    return (
        float(bbox[0]) - 1e-9 <= float(point[0]) <= float(bbox[2]) + 1e-9
        and float(bbox[1]) - 1e-9 <= float(point[1]) <= float(bbox[3]) + 1e-9
    )


def _point_on_segment(point: Point, start: Point, end: Point, tolerance: float = 1e-9) -> bool:
    px, py = float(point[0]), float(point[1])
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    cross = (px - x0) * (y1 - y0) - (py - y0) * (x1 - x0)
    scale = max(abs(x1 - x0), abs(y1 - y0), 1.0)
    if abs(cross) > tolerance * scale:
        return False
    return (
        min(x0, x1) - tolerance <= px <= max(x0, x1) + tolerance
        and min(y0, y1) - tolerance <= py <= max(y0, y1) + tolerance
    )


def _point_in_rings(point: Point, rings: Sequence[Sequence[Point]]) -> bool:
    px, py = float(point[0]), float(point[1])
    inside = False
    for ring in rings:
        if len(ring) < 3:
            continue
        closed = list(ring)
        if closed[0] != closed[-1]:
            closed.append(closed[0])
        for start, end in zip(closed[:-1], closed[1:]):
            if _point_on_segment(point, start, end):
                return True
            x0, y0 = float(start[0]), float(start[1])
            x1, y1 = float(end[0]), float(end[1])
            if (y0 > py) == (y1 > py):
                continue
            crossing_x = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
            if crossing_x > px:
                inside = not inside
    return inside


def _segment_intersection_parameter(start: Point, end: Point, edge_start: Point, edge_end: Point) -> float | None:
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    x2, y2 = float(edge_start[0]), float(edge_start[1])
    x3, y3 = float(edge_end[0]), float(edge_end[1])
    rx, ry = x1 - x0, y1 - y0
    sx, sy = x3 - x2, y3 - y2
    denominator = rx * sy - ry * sx
    if abs(denominator) <= 1e-12:
        return None
    qpx, qpy = x2 - x0, y2 - y0
    t = (qpx * sy - qpy * sx) / denominator
    u = (qpx * ry - qpy * rx) / denominator
    if -1e-10 <= t <= 1.0 + 1e-10 and -1e-10 <= u <= 1.0 + 1e-10:
        return float(np.clip(t, 0.0, 1.0))
    return None


def _point_at_parameter(start: Point, end: Point, t: float) -> Point:
    return (
        float(start[0]) + (float(end[0]) - float(start[0])) * float(t),
        float(start[1]) + (float(end[1]) - float(start[1])) * float(t),
    )


def _clip_segment_to_rings(start: Point, end: Point, boundary_index: BoundaryClipIndex) -> List[Tuple[Point, Point]]:
    parameters = [0.0, 1.0]
    for edge_start, edge_end in boundary_index.candidate_edges_for_segment(start, end):
        t = _segment_intersection_parameter(start, end, edge_start, edge_end)
        if t is not None:
            parameters.append(t)
    ordered: List[float] = []
    for value in sorted(parameters):
        if not ordered or abs(value - ordered[-1]) > 1e-9:
            ordered.append(value)
    clipped: List[Tuple[Point, Point]] = []
    for left, right in zip(ordered[:-1], ordered[1:]):
        if right - left <= 1e-10:
            continue
        midpoint = _point_at_parameter(start, end, (left + right) * 0.5)
        if boundary_index.contains(midpoint):
            clipped.append((_point_at_parameter(start, end, left), _point_at_parameter(start, end, right)))
    return clipped


def _points_close(first: Point, second: Point, tolerance: float = 1e-8) -> bool:
    return abs(float(first[0]) - float(second[0])) <= tolerance and abs(float(first[1]) - float(second[1])) <= tolerance


def clip_polylines_to_boundary(
    records: Sequence[GISPolyline],
    rings: Sequence[Sequence[Point]],
    *,
    spatial_index: PolylineSpatialIndex | None = None,
) -> List[GISPolyline]:
    boundary_index = BoundaryClipIndex.build(rings)
    clipped_records: List[GISPolyline] = []
    candidate_records = spatial_index.query(boundary_index.bbox) if spatial_index is not None else records
    for record in candidate_records:
        clipped_parts: List[List[Point]] = []
        for part in record.parts:
            current: List[Point] = []
            for start, end in zip(part[:-1], part[1:]):
                for clipped_start, clipped_end in _clip_segment_to_rings(start, end, boundary_index):
                    if current and _points_close(current[-1], clipped_start):
                        if not _points_close(current[-1], clipped_end):
                            current.append(clipped_end)
                    else:
                        if len(current) >= 2:
                            clipped_parts.append(current)
                        current = [clipped_start, clipped_end]
            if len(current) >= 2:
                clipped_parts.append(current)
        if clipped_parts:
            clipped_records.append(
                GISPolyline(
                    parts=clipped_parts,
                    start_height=record.start_height,
                    end_height=record.end_height,
                    slope_reference_parts=record.slope_reference_parts or record.parts,
                )
            )
    return clipped_records


def _confidence_from_votes(votes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    directional = votes[:, :, 1:5]
    total = np.sum(directional, axis=2)
    best = np.max(directional, axis=2)
    ordered = np.sort(directional, axis=2)
    second = ordered[:, :, -2]
    confidence = np.zeros(total.shape, dtype=np.float32)
    mask = total > 1e-9
    dominance = np.zeros_like(total, dtype=np.float64)
    margin = np.zeros_like(total, dtype=np.float64)
    dominance[mask] = best[mask] / total[mask]
    margin[mask] = (best[mask] - second[mask]) / np.maximum(best[mask], 1e-9)
    confidence[mask] = np.clip(0.45 + 0.35 * dominance[mask] + 0.20 * margin[mask], 0.0, 1.0)
    active = directional > np.maximum(best[:, :, None] * 0.25, 1e-9)
    overlap = ((np.sum(active, axis=2) >= 2) & mask).astype(np.uint8)
    confidence[overlap == 1] *= np.float32(0.82)
    return confidence.astype(np.float32), overlap


def _degree_cell_size_to_meters(bbox: BBox, cell_width: float, cell_height: float) -> Tuple[float, float]:
    _xmin, ymin, _xmax, ymax = bbox
    center_lat = math.radians((float(ymin) + float(ymax)) * 0.5)
    meters_per_degree_lat = 111_132.92 - 559.82 * math.cos(2.0 * center_lat) + 1.175 * math.cos(4.0 * center_lat)
    meters_per_degree_lon = 111_412.84 * math.cos(center_lat) - 93.5 * math.cos(3.0 * center_lat)
    return abs(float(cell_width)) * abs(meters_per_degree_lon), abs(float(cell_height)) * abs(meters_per_degree_lat)


def _degree_segment_length_meters(p0: Point, p1: Point) -> float:
    lon0, lat0 = float(p0[0]), float(p0[1])
    lon1, lat1 = float(p1[0]), float(p1[1])
    center_lat = math.radians((lat0 + lat1) * 0.5)
    meters_per_degree_lat = 111_132.92 - 559.82 * math.cos(2.0 * center_lat) + 1.175 * math.cos(4.0 * center_lat)
    meters_per_degree_lon = 111_412.84 * math.cos(center_lat) - 93.5 * math.cos(3.0 * center_lat)
    dx = (lon1 - lon0) * meters_per_degree_lon
    dy = (lat1 - lat0) * meters_per_degree_lat
    return float(math.hypot(dx, dy))


def _segment_length_for_slope(p0: Point, p1: Point, unit_type: str, to_meter: object) -> Tuple[float, bool]:
    raw_len = float(math.hypot(float(p1[0] - p0[0]), float(p1[1] - p0[1])))
    if unit_type == "meter":
        return raw_len, True
    if unit_type == "linear" and to_meter is not None:
        return raw_len * float(to_meter), True
    if unit_type == "degree":
        return _degree_segment_length_meters(p0, p1), True
    return raw_len, False


def _apply_cell_size_unit_meta(meta: Dict[str, object], bbox: BBox) -> None:
    cell_width = float(meta.get("cell_width", 0.0) or 0.0)
    cell_height = float(meta.get("cell_height", 0.0) or 0.0)
    unit_type = str(meta.get("coordinate_unit_type", "unknown") or "unknown").lower()
    to_meter = meta.get("coordinate_unit_to_meter")
    meta["cell_width_m"] = None
    meta["cell_height_m"] = None
    meta["cell_size_note"] = ""
    if unit_type == "meter":
        meta["cell_width_m"] = abs(cell_width)
        meta["cell_height_m"] = abs(cell_height)
    elif unit_type == "linear" and to_meter is not None:
        factor = float(to_meter)
        meta["cell_width_m"] = abs(cell_width) * factor
        meta["cell_height_m"] = abs(cell_height) * factor
    elif unit_type == "degree":
        width_m, height_m = _degree_cell_size_to_meters(bbox, cell_width, cell_height)
        meta["cell_width_m"] = float(width_m)
        meta["cell_height_m"] = float(height_m)
        meta["cell_size_note"] = "geographic degree unit converted approximately by bbox latitude"


def _apply_occupied_slope_meta(
    meta: Dict[str, object],
    records: Sequence[GISPolyline],
    rows: int,
    cols: int,
    bbox: BBox,
) -> None:
    unit_type = str(meta.get("coordinate_unit_type", "unknown") or "unknown").lower()
    to_meter = meta.get("coordinate_unit_to_meter")
    slope_sum = np.zeros((rows, cols), dtype=np.float64)
    slope_weight = np.zeros((rows, cols), dtype=np.float64)
    feature_count = 0
    metric_length_known = unit_type in {"meter", "degree"} or (unit_type == "linear" and to_meter is not None)

    for record in records:
        if record.start_height is None or record.end_height is None:
            continue
        if abs(float(record.start_height)) <= 1e-9 and abs(float(record.end_height)) <= 1e-9:
            continue
        total_length = 0.0
        reference_parts = record.slope_reference_parts or record.parts
        for part in reference_parts:
            for p0, p1 in zip(part[:-1], part[1:]):
                length, is_metric = _segment_length_for_slope(p0, p1, unit_type, to_meter)
                metric_length_known = metric_length_known and bool(is_metric)
                total_length += length
        if total_length <= 1e-12:
            continue

        part_lengths: List[List[float]] = []
        for part in record.parts:
            lengths: List[float] = []
            for p0, p1 in zip(part[:-1], part[1:]):
                length, is_metric = _segment_length_for_slope(p0, p1, unit_type, to_meter)
                metric_length_known = metric_length_known and bool(is_metric)
                lengths.append(length)
            part_lengths.append(lengths)
        feature_count += 1
        slope = abs(float(record.start_height) - float(record.end_height)) / total_length
        for part, lengths in zip(record.parts, part_lengths):
            if len(part) < 2:
                continue
            for (p0, p1), seg_len in zip(zip(part[:-1], part[1:]), lengths):
                path = _segment_cell_path(p0, p1, rows, cols, bbox)
                if not path:
                    continue
                cell_weight = max(float(seg_len) / max(len(path), 1), 1e-12)
                for row, col in path:
                    slope_sum[row, col] += slope * cell_weight
                    slope_weight[row, col] += cell_weight

    mask = slope_weight > 0
    meta["occupied_slope_available"] = bool(np.any(mask))
    meta["occupied_slope_metric"] = bool(metric_length_known)
    meta["occupied_slope_cell_count"] = int(np.count_nonzero(mask))
    meta["occupied_slope_feature_count"] = int(feature_count)
    meta["occupied_slope_mean"] = None
    meta["occupied_slope_percent"] = None
    meta["occupied_slope_per_mille"] = None
    meta["occupied_slope_note"] = ""
    if not np.any(mask):
        if not records:
            meta["occupied_slope_note"] = "no GIS records"
        else:
            meta["occupied_slope_note"] = "height fields are missing or unusable"
        return
    per_cell_slope = slope_sum[mask] / np.maximum(slope_weight[mask], 1e-12)
    mean_slope = float(np.mean(per_cell_slope))
    meta["occupied_slope_mean"] = mean_slope
    if metric_length_known:
        meta["occupied_slope_percent"] = mean_slope * 100.0
        meta["occupied_slope_per_mille"] = mean_slope * 1000.0
        if unit_type == "degree":
            meta["occupied_slope_note"] = "geographic degree length converted approximately by segment latitude"
    else:
        meta["occupied_slope_note"] = "horizontal coordinate unit is not confirmed as meters"


def matrix_from_polylines(
    records: Sequence[GISPolyline],
    rows: int,
    bbox: BBox,
    cols: int | None = None,
) -> GISMatrixResult:
    rows = max(2, int(rows))
    cols = int(cols) if cols is not None else derive_grid_cols(rows, bbox)
    cols = max(2, cols)
    votes = np.zeros((rows, cols, 5), dtype=np.float64)
    height_counts: Counter[str] = Counter()
    segment_counts: Counter[int] = Counter()
    feature_count = 0
    segment_count = 0
    used_segment_count = 0

    for record in records:
        feature_count += 1
        reverse = _should_reverse_by_height(record.start_height, record.end_height)
        if record.start_height is None or record.end_height is None:
            height_counts["missing_geometry_order"] += 1
        elif abs(float(record.start_height)) <= 1e-9 and abs(float(record.end_height)) <= 1e-9:
            height_counts["zero_geometry_order"] += 1
        elif reverse:
            height_counts["reversed_by_height"] += 1
        elif float(record.start_height) > float(record.end_height):
            height_counts["start_to_end_by_height"] += 1
        else:
            height_counts["equal_height_geometry_order"] += 1

        for part in record.parts:
            if len(part) < 2:
                continue
            oriented = list(reversed(part)) if reverse else list(part)
            for p0, p1 in zip(oriented[:-1], oriented[1:]):
                segment_count += 1
                path = _segment_cell_path(p0, p1, rows, cols, bbox)
                if len(path) < 2:
                    continue
                seg_len = float(math.hypot(float(p1[0] - p0[0]), float(p1[1] - p0[1])))
                step_weight = max(seg_len / max(len(path) - 1, 1), 1.0)
                last_code = 0
                for src, dst in zip(path[:-1], path[1:]):
                    code = _direction_from_step(src, dst)
                    if code == 0:
                        continue
                    votes[src[0], src[1], code] += step_weight
                    segment_counts[code] += 1
                    last_code = code
                if last_code:
                    votes[path[-1][0], path[-1][1], last_code] += step_weight * 0.35
                    used_segment_count += 1

    directional = votes[:, :, 1:5]
    best_dirs = np.argmax(directional, axis=2) + 1
    best_vals = np.max(directional, axis=2)
    matrix = np.zeros((rows, cols), dtype=np.uint8)
    matrix[best_vals > 0] = best_dirs[best_vals > 0].astype(np.uint8)
    support = (best_vals > 0).astype(np.uint8)
    confidence, overlap = _confidence_from_votes(votes)
    image_bgr = render_polylines_image(records, bbox, rows, cols)
    xmin, ymin, xmax, ymax = bbox
    cell_width = float(xmax - xmin) / max(float(cols), 1.0)
    cell_height = float(ymax - ymin) / max(float(rows), 1.0)
    meta: Dict[str, object] = {
        "rows_A": int(rows),
        "cols_B": int(cols),
        "bbox": [float(v) for v in bbox],
        "cell_width": float(cell_width),
        "cell_height": float(cell_height),
        "feature_count": int(feature_count),
        "segment_count": int(segment_count),
        "used_segment_count": int(used_segment_count),
        "direction_step_counts": {str(k): int(v) for k, v in sorted(segment_counts.items())},
        "height_direction_counts": {str(k): int(v) for k, v in height_counts.items()},
        "nonzero_cells": int(np.count_nonzero(matrix)),
        "overlap_cells": int(np.count_nonzero(overlap)),
        "direction_cell_counts": {str(k): int(np.count_nonzero(matrix == k)) for k in range(5)},
        "diagonal_policy": "8-neighbor GIS cell paths are expanded into 4-neighbor stair steps",
    }
    return GISMatrixResult(
        matrix=matrix,
        occ=support.copy(),
        path_occ=support.copy(),
        support_mask=support.copy(),
        confidence=confidence,
        overlap_mask=overlap,
        image_bgr=image_bgr,
        meta=meta,
    )


def render_polylines_image(
    records: Sequence[GISPolyline],
    bbox: BBox,
    rows: int,
    cols: int,
    cell_px: int = 16,
) -> np.ndarray:
    height = max(1, int(rows) * int(cell_px))
    width = max(1, int(cols) * int(cell_px))
    xmin, ymin, xmax, ymax = bbox
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    def to_pixel_array(points: Sequence[Point]) -> np.ndarray:
        coords = np.asarray(points, dtype=np.float64)
        px = np.rint((coords[:, 0] - xmin) / max(xmax - xmin, 1e-9) * (width - 1))
        py = np.rint((ymax - coords[:, 1]) / max(ymax - ymin, 1e-9) * (height - 1))
        pixels = np.column_stack((np.clip(px, 0, width - 1), np.clip(py, 0, height - 1)))
        return pixels.astype(np.int32).reshape((-1, 1, 2))

    for record in records:
        reverse = _should_reverse_by_height(record.start_height, record.end_height)
        for part in record.parts:
            pts = list(reversed(part)) if reverse else list(part)
            if len(pts) < 2:
                continue
            arr = to_pixel_array(pts)
            cv2.polylines(img, [arr], isClosed=False, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    return img


def render_gis_layers(
    polyline_layers: Sequence[tuple],
    boundary_layers: Sequence[tuple],
    bbox: BBox,
    *,
    max_size: int = 1800,
    layer_stack: Sequence[Tuple[str, tuple]] | None = None,
    output_size: Tuple[int, int] | None = None,
) -> np.ndarray:
    xmin, ymin, xmax, ymax = bbox
    width_span = max(float(xmax - xmin), 1e-9)
    height_span = max(float(ymax - ymin), 1e-9)
    if output_size is not None:
        width = max(1, int(output_size[0]))
        height = max(1, int(output_size[1]))
    else:
        aspect = float(np.clip(width_span / height_span, 0.1, 10.0))
        if aspect >= 1.0:
            width = max(500, int(max_size))
            height = max(300, int(round(width / aspect)))
        else:
            height = max(500, int(max_size))
            width = max(300, int(round(height * aspect)))
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    stack_polyline_layers = [item for kind, item in layer_stack or [] if kind == "polyline"]
    stack_boundary_layers = [item for kind, item in layer_stack or [] if kind == "boundary"]

    def polyline_item_point_count(item: tuple) -> int:
        if len(item) >= 5 and isinstance(item[4], (int, np.integer)):
            return int(item[4])
        return sum(len(part) for record in item[0] for part in record.parts)

    def boundary_item_point_count(item: tuple) -> int:
        if len(item) >= 7 and isinstance(item[6], (int, np.integer)):
            return int(item[6])
        return sum(len(ring) for boundary in item[0] for ring in boundary.rings)

    point_count = sum(polyline_item_point_count(item) for item in list(polyline_layers) + stack_polyline_layers) + sum(
        boundary_item_point_count(item) for item in list(boundary_layers) + stack_boundary_layers
    )
    line_type = cv2.LINE_8 if point_count >= 100_000 else cv2.LINE_AA

    def to_pixel_array(points: Sequence[Point], *, closed: bool = False) -> np.ndarray:
        coords = np.asarray(points, dtype=np.float64)
        px = np.rint((coords[:, 0] - xmin) / width_span * (width - 1))
        py = np.rint((ymax - coords[:, 1]) / height_span * (height - 1))
        pixels = np.column_stack((np.clip(px, 0, width - 1), np.clip(py, 0, height - 1))).astype(np.int32)
        if len(pixels) >= 2:
            keep = np.ones(len(pixels), dtype=bool)
            keep[1:] = np.any(pixels[1:] != pixels[:-1], axis=1)
            pixels = pixels[keep]
        if point_count >= 100_000 and len(pixels) >= 8:
            pixels = cv2.approxPolyDP(
                pixels.reshape((-1, 1, 2)),
                epsilon=0.55,
                closed=closed,
            ).reshape((-1, 2))
        return pixels.reshape((-1, 1, 2))

    def draw_boundary(item: tuple) -> None:
        nonlocal image
        boundaries, fill_color = item[0], item[1]
        fill_opacity = float(np.clip(item[2] if len(item) >= 3 else 0.12, 0.0, 1.0))
        outline_color = tuple(int(value) for value in (item[3] if len(item) >= 4 else fill_color))
        outline_opacity = float(np.clip(item[4] if len(item) >= 5 else 1.0, 0.0, 1.0))
        line_width = max(1, int(round(float(item[5] if len(item) >= 6 else 2.0))))
        ring_arrays: List[np.ndarray] = []
        for boundary in boundaries:
            for ring in boundary.rings:
                if len(ring) < 3:
                    continue
                pixel_ring = to_pixel_array(ring, closed=True)
                if len(pixel_ring) >= 3:
                    ring_arrays.append(pixel_ring)
        if ring_arrays:
            if fill_opacity > 0.0:
                fill_overlay = image.copy()
                cv2.fillPoly(fill_overlay, ring_arrays, color=fill_color, lineType=line_type)
                image = cv2.addWeighted(fill_overlay, fill_opacity, image, 1.0 - fill_opacity, 0.0)
            if outline_opacity > 0.0:
                outline_overlay = image.copy()
                cv2.polylines(
                    outline_overlay,
                    ring_arrays,
                    isClosed=True,
                    color=outline_color,
                    thickness=line_width,
                    lineType=line_type,
                )
                image = cv2.addWeighted(outline_overlay, outline_opacity, image, 1.0 - outline_opacity, 0.0)

    def draw_polyline(item: tuple) -> None:
        nonlocal image
        records, color = item[0], item[1]
        opacity = float(np.clip(item[2] if len(item) >= 3 else 1.0, 0.0, 1.0))
        line_width = max(1, int(round(float(item[3] if len(item) >= 4 else 2.0))))
        target = image if opacity >= 0.999 else image.copy()
        for record in records:
            for part in record.parts:
                if len(part) < 2:
                    continue
                points = to_pixel_array(part)
                if len(points) < 2:
                    continue
                cv2.polylines(
                    target,
                    [points],
                    isClosed=False,
                    color=color,
                    thickness=line_width,
                    lineType=line_type,
                )
        if target is not image:
            image = cv2.addWeighted(target, opacity, image, 1.0 - opacity, 0.0)

    if layer_stack is not None:
        for kind, item in layer_stack:
            if kind == "boundary":
                draw_boundary(item)
            else:
                draw_polyline(item)
    else:
        for item in boundary_layers:
            draw_boundary(item)
        for item in polyline_layers:
            draw_polyline(item)
    return image


def _read_dbf_fields(dbf_path: Path) -> Tuple[int, int, int, Dict[str, Tuple[int, int, str]]]:
    with dbf_path.open("rb") as fh:
        header = fh.read(32)
        if len(header) < 32:
            raise ValueError(f"Invalid DBF header: {dbf_path}")
        record_count = struct.unpack("<I", header[4:8])[0]
        header_len = struct.unpack("<H", header[8:10])[0]
        record_len = struct.unpack("<H", header[10:12])[0]
        fields: Dict[str, Tuple[int, int, str]] = {}
        offset = 1
        while True:
            desc = fh.read(32)
            if not desc or desc[0] == 0x0D:
                break
            raw_name = desc[0:11].split(b"\x00", 1)[0]
            name = raw_name.decode("latin1", errors="replace").strip()
            field_type = chr(desc[11])
            field_len = int(desc[16])
            fields[name] = (offset, field_len, field_type)
            offset += field_len
    return int(record_count), int(header_len), int(record_len), fields


def _read_dbf_text(record: bytes, fields: Dict[str, Tuple[int, int, str]], name: str) -> str:
    if name not in fields:
        return ""
    offset, field_len, _field_type = fields[name]
    raw = record[offset: offset + field_len]
    for encoding in ("utf-8", "cp949", "euc-kr", "latin1"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("latin1", errors="replace").strip()


def _read_dbf_float(record: bytes, fields: Dict[str, Tuple[int, int, str]], name: str) -> float | None:
    text = _read_dbf_text(record, fields, name)
    if not text or "*" in text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_dbf_records(dbf_path: Path) -> Tuple[int, Dict[str, Tuple[int, int, str]], List[bytes | None]]:
    record_count, header_len, record_len, fields = _read_dbf_fields(dbf_path)
    records: List[bytes | None] = []
    with dbf_path.open("rb") as fh:
        fh.seek(header_len)
        for _idx in range(record_count):
            record = fh.read(record_len)
            if len(record) < record_len:
                break
            records.append(None if record[0:1] == b"*" else record)
    return int(record_count), fields, records


def _read_dbf_record_markers(dbf_path: Path) -> Tuple[int, Dict[str, Tuple[int, int, str]], List[bytes | None]]:
    record_count, header_len, record_len, fields = _read_dbf_fields(dbf_path)
    records: List[bytes | None] = []
    with dbf_path.open("rb") as fh:
        fh.seek(header_len)
        for _idx in range(record_count):
            marker = fh.read(1)
            if not marker:
                break
            records.append(None if marker == b"*" else b" ")
            fh.seek(max(record_len - 1, 0), 1)
    return int(record_count), fields, records


def _read_optional_dbf(
    dbf_path: Path,
    required_fields: Sequence[str | None] | None = None,
) -> Tuple[int, Dict[str, Tuple[int, int, str]], List[bytes | None]]:
    if not dbf_path.exists():
        return 0, {}, []
    if required_fields is not None:
        record_count, _header_len, _record_len, fields = _read_dbf_fields(dbf_path)
        requested = {str(field) for field in required_fields if field}
        if not requested.intersection(fields):
            return _read_dbf_record_markers(dbf_path)
    return _read_dbf_records(dbf_path)


# File-layout authority: ESRI Shapefile Technical Description (July 1998).
# https://www.esri.com/library/whitepapers/pdfs/shapefile.pdf
def _read_shp_header_for_types(
    shp_path: Path,
    supported_types: Dict[int, str],
    source_label: str,
) -> Tuple[int, BBox]:
    with shp_path.open("rb") as fh:
        header = fh.read(100)
    if len(header) < 100:
        raise ValueError(f"Invalid SHP header: {shp_path}")
    shape_type = struct.unpack("<i", header[32:36])[0]
    if int(shape_type) not in supported_types:
        supported = ", ".join(f"{code}={name}" for code, name in sorted(supported_types.items()))
        raise ValueError(f"Only {source_label} shapefiles are supported. shape_type={shape_type}, supported={supported}")
    bbox = tuple(float(v) for v in struct.unpack("<4d", header[36:68]))
    return int(shape_type), bbox


def _read_shp_header(shp_path: Path) -> Tuple[int, BBox]:
    return _read_shp_header_for_types(shp_path, SUPPORTED_POLYLINE_TYPES, "PolyLine/PolyLineZ")


def read_shp_shape_info(shp_path: str | Path) -> Dict[str, object]:
    shp_path = Path(shp_path)
    with shp_path.open("rb") as fh:
        header = fh.read(100)
    if len(header) < 100:
        raise ValueError(f"Invalid SHP header: {shp_path}")
    shape_type = int(struct.unpack("<i", header[32:36])[0])
    bbox = tuple(float(v) for v in struct.unpack("<4d", header[36:68]))
    if shape_type in SUPPORTED_POLYLINE_TYPES:
        kind = "polyline"
    elif shape_type in SUPPORTED_POLYGON_TYPES:
        kind = "boundary"
    else:
        kind = "unsupported"
    return {
        "source_shp": str(shp_path),
        "shape_type": shape_type,
        "shape_type_name": _shape_type_name(shape_type),
        "kind": kind,
        "bbox": [float(v) for v in bbox],
    }


def inspect_gis_source(shp_path: str | Path) -> Dict[str, object]:
    shp_path = Path(shp_path)
    if not shp_path.exists():
        raise FileNotFoundError(str(shp_path))
    shape_type, bbox = _read_shp_header(shp_path)
    dbf_path = shp_path.with_suffix(".dbf")
    shx_path = shp_path.with_suffix(".shx")
    record_count = 0
    fields: Dict[str, Tuple[int, int, str]] = {}
    if dbf_path.exists():
        record_count, _header_len, _record_len, fields = _read_dbf_fields(dbf_path)
    info = {
        "source_shp": str(shp_path),
        "shape_type": int(shape_type),
        "shape_type_name": _shape_type_name(shape_type),
        "bbox": [float(v) for v in bbox],
        "dbf_available": bool(dbf_path.exists()),
        "shx_available": bool(shx_path.exists()),
        "dbf_records": int(record_count),
        "fields": sorted(fields.keys()),
    }
    info.update(_read_prj_unit_meta(shp_path))
    return info


def inspect_boundary_source(shp_path: str | Path) -> Dict[str, object]:
    shp_path = Path(shp_path)
    if not shp_path.exists():
        raise FileNotFoundError(str(shp_path))
    shape_type, bbox = _read_shp_header_for_types(shp_path, SUPPORTED_POLYGON_TYPES, "Polygon/PolygonZ/PolygonM")
    dbf_path = shp_path.with_suffix(".dbf")
    shx_path = shp_path.with_suffix(".shx")
    record_count = 0
    fields: Dict[str, Tuple[int, int, str]] = {}
    if dbf_path.exists():
        record_count, _header_len, _record_len, fields = _read_dbf_fields(dbf_path)
    info = {
        "source_shp": str(shp_path),
        "shape_type": int(shape_type),
        "shape_type_name": _shape_type_name(shape_type),
        "bbox": [float(v) for v in bbox],
        "dbf_available": bool(dbf_path.exists()),
        "shx_available": bool(shx_path.exists()),
        "dbf_records": int(record_count),
        "fields": sorted(fields.keys()),
    }
    info.update(_read_prj_unit_meta(shp_path))
    return info


def _iter_shp_record_contents(shp_path: Path):
    with shp_path.open("rb") as fh:
        header = fh.read(100)
        if len(header) < 100:
            raise ValueError(f"Invalid SHP header: {shp_path}")
        while True:
            record_header = fh.read(8)
            if not record_header:
                break
            if len(record_header) < 8:
                raise ValueError(f"Truncated SHP record header: {shp_path}")
            record_number, content_words = struct.unpack(">2i", record_header)
            content_bytes = int(content_words) * 2
            content = fh.read(content_bytes)
            if len(content) < content_bytes:
                raise ValueError(f"Truncated SHP record content: record={record_number}")
            yield int(record_number), content


def _read_polyline_parts_from_content(content: bytes) -> List[List[Point]]:
    if len(content) < 4:
        return []
    shape_type = struct.unpack("<i", content[:4])[0]
    if shape_type == 0:
        return []
    if int(shape_type) not in SUPPORTED_POLYLINE_TYPES:
        supported = ", ".join(f"{code}={name}" for code, name in sorted(SUPPORTED_POLYLINE_TYPES.items()))
        raise ValueError(f"Only PolyLine/PolyLineZ records are supported. shape_type={shape_type}, supported={supported}")
    offset = 4 + 32
    if len(content) < offset + 8:
        raise ValueError("Truncated PolyLine record")
    num_parts, num_points = struct.unpack("<2i", content[offset: offset + 8])
    offset += 8
    if num_parts <= 0 or num_points <= 0:
        return []
    parts_bytes = 4 * num_parts
    points_bytes = 16 * num_points
    if len(content) < offset + parts_bytes + points_bytes:
        raise ValueError("Truncated PolyLine point array")
    parts = list(struct.unpack("<" + "i" * num_parts, content[offset: offset + parts_bytes]))
    offset += parts_bytes
    points = np.frombuffer(content, dtype="<f8", count=num_points * 2, offset=offset).reshape((-1, 2)).tolist()
    ranges = parts + [num_points]
    return [points[ranges[idx]: ranges[idx + 1]] for idx in range(num_parts)]


def _read_polygon_rings_from_content(content: bytes) -> List[List[Point]]:
    if len(content) < 4:
        return []
    shape_type = struct.unpack("<i", content[:4])[0]
    if shape_type == 0:
        return []
    if int(shape_type) not in SUPPORTED_POLYGON_TYPES:
        supported = ", ".join(f"{code}={name}" for code, name in sorted(SUPPORTED_POLYGON_TYPES.items()))
        raise ValueError(f"Only Polygon/PolygonZ/PolygonM records are supported. shape_type={shape_type}, supported={supported}")
    offset = 4 + 32
    if len(content) < offset + 8:
        raise ValueError("Truncated Polygon record")
    num_parts, num_points = struct.unpack("<2i", content[offset: offset + 8])
    offset += 8
    if num_parts <= 0 or num_points <= 0:
        return []
    parts_bytes = 4 * num_parts
    points_bytes = 16 * num_points
    if len(content) < offset + parts_bytes + points_bytes:
        raise ValueError("Truncated Polygon point array")
    parts = list(struct.unpack("<" + "i" * num_parts, content[offset: offset + parts_bytes]))
    offset += parts_bytes
    points = np.frombuffer(content, dtype="<f8", count=num_points * 2, offset=offset).reshape((-1, 2)).tolist()
    ranges = parts + [num_points]
    return [points[ranges[idx]: ranges[idx + 1]] for idx in range(num_parts)]


def _read_dbf_height(
    dbf_record: bytes | None,
    fields: Dict[str, Tuple[int, int, str]],
    field_name: str | None,
) -> float | None:
    if dbf_record is None or not field_name:
        return None
    return _read_dbf_float(dbf_record, fields, field_name)


def read_boundary_shp(shp_path: str | Path) -> Tuple[List[GISBoundary], BBox, Dict[str, object]]:
    shp_path = Path(shp_path)
    dbf_path = shp_path.with_suffix(".dbf")
    shx_path = shp_path.with_suffix(".shx")
    if not shp_path.exists():
        raise FileNotFoundError(str(shp_path))
    shape_type, bbox = _read_shp_header_for_types(shp_path, SUPPORTED_POLYGON_TYPES, "Polygon/PolygonZ/PolygonM")
    record_count, fields, dbf_records = _read_optional_dbf(dbf_path)
    boundaries: List[GISBoundary] = []
    skipped_deleted_dbf = 0
    shp_record_count = 0
    for idx, (_record_number, content) in enumerate(_iter_shp_record_contents(shp_path)):
        shp_record_count += 1
        dbf_record = dbf_records[idx] if idx < len(dbf_records) else None
        if dbf_records and dbf_record is None:
            skipped_deleted_dbf += 1
            continue
        rings = _read_polygon_rings_from_content(content)
        if not rings:
            continue
        attributes = {
            field_name: _read_dbf_text(dbf_record, fields, field_name)
            for field_name in fields
            if dbf_record is not None
        }
        boundaries.append(GISBoundary(rings=rings, attributes=attributes))
    meta = {
        "source_shp": str(shp_path),
        "shape_type": int(shape_type),
        "shape_type_name": _shape_type_name(shape_type),
        "bbox": [float(value) for value in bbox],
        "dbf_available": bool(dbf_path.exists()),
        "shx_available": bool(shx_path.exists()),
        "shx_used": False,
        "dbf_records": int(record_count),
        "shp_records": int(shp_record_count),
        "loaded_records": int(len(boundaries)),
        "skipped_deleted_dbf_records": int(skipped_deleted_dbf),
        "fields": sorted(fields.keys()),
    }
    meta.update(_read_prj_unit_meta(shp_path))
    return boundaries, bbox, meta


def list_boundary_values(shp_path: str | Path, field_name: str) -> List[str]:
    boundaries, _bbox, meta = read_boundary_shp(shp_path)
    fields = list(meta.get("fields", []))
    if field_name not in fields:
        raise ValueError(f"Boundary field not found: {field_name}")
    return sorted({boundary.attributes.get(field_name, "") for boundary in boundaries if boundary.attributes.get(field_name, "")})


def _selected_boundary_rings(
    boundary_shp_path: str | Path,
    boundary_field: str,
    boundary_value: str,
) -> Tuple[List[List[Point]], BBox, Dict[str, object]]:
    boundaries, _source_bbox, boundary_meta = read_boundary_shp(boundary_shp_path)
    selected = [
        boundary
        for boundary in boundaries
        if boundary.attributes.get(boundary_field, "") == str(boundary_value)
    ]
    if not selected:
        raise ValueError(f"No boundary feature matched {boundary_field}={boundary_value}")
    rings = [ring for boundary in selected for ring in boundary.rings if len(ring) >= 3]
    points = [point for ring in rings for point in ring]
    bbox = _bbox_from_points(points)
    boundary_meta = dict(boundary_meta)
    boundary_meta["selected_field"] = str(boundary_field)
    boundary_meta["selected_value"] = str(boundary_value)
    boundary_meta["selected_feature_count"] = int(len(selected))
    boundary_meta["selected_ring_count"] = int(len(rings))
    boundary_meta["selected_bbox"] = [float(value) for value in bbox]
    return rings, bbox, boundary_meta


def _normalized_prj_text(shp_path: str | Path) -> str:
    prj_path = Path(shp_path).with_suffix(".prj")
    if not prj_path.exists():
        return ""
    text = prj_path.read_text(encoding="utf-8", errors="ignore")
    return re.sub(r"\s+", "", text).lower()


def validate_gis_compatibility(
    pipe_shp_path: str | Path,
    pipe_bbox: BBox,
    boundary_shp_path: str | Path,
    boundary_bbox: BBox,
) -> str:
    pipe_meta = read_shapefile_crs_meta(pipe_shp_path)
    boundary_meta = read_shapefile_crs_meta(boundary_shp_path)
    for role, meta in (("Pipe", pipe_meta), ("Boundary", boundary_meta)):
        if bool(meta.get("prj_available")) and not bool(meta.get("crs_valid")):
            detail = str(meta.get("crs_error", "") or "unparseable CRS definition")
            raise ValueError(f"{role} GIS has an invalid PRJ definition: {detail}")
    pipe_wkt = str(pipe_meta.get("crs_wkt", "") or "")
    boundary_wkt = str(boundary_meta.get("crs_wkt", "") or "")
    if pipe_wkt and boundary_wkt and not crs_equivalent(pipe_wkt, boundary_wkt):
        pipe_label = str(pipe_meta.get("crs_label", "") or "unknown")
        boundary_label = str(boundary_meta.get("crs_label", "") or "unknown")
        raise ValueError(
            f"Pipe and boundary coordinate systems differ ({pipe_label} vs {boundary_label}). "
            "Reproject them to one project CRS before clipping."
        )
    if not _bbox_intersects(pipe_bbox, boundary_bbox):
        raise ValueError("Pipe GIS and boundary GIS coordinate ranges do not overlap.")
    if not pipe_wkt or not boundary_wkt:
        return "PRJ is missing on one or both sources; assign each layer CRS before multi-layer analysis"
    return ""


def _validate_boundary_compatibility(
    pipe_shp_path: str | Path,
    pipe_bbox: BBox,
    boundary_shp_path: str | Path,
    boundary_bbox: BBox,
) -> str:
    return validate_gis_compatibility(pipe_shp_path, pipe_bbox, boundary_shp_path, boundary_bbox)


def matrix_from_loaded_polylines(
    records: Sequence[GISPolyline],
    rows: int,
    bbox: BBox,
    *,
    source_meta: Dict[str, object] | None = None,
    rotation_degrees: float = 0.0,
) -> GISMatrixResult:
    rotated_records, used_bbox = prepare_rotated_polylines(records, bbox, rotation_degrees)
    result = matrix_from_polylines(rotated_records, rows=rows, bbox=used_bbox)
    if source_meta:
        result.meta.update(source_meta)
    result.meta["bbox"] = [float(value) for value in used_bbox]
    result.meta["rotation_degrees"] = float(rotation_degrees)
    _apply_cell_size_unit_meta(result.meta, used_bbox)
    _apply_occupied_slope_meta(
        result.meta,
        rotated_records,
        int(result.matrix.shape[0]),
        int(result.matrix.shape[1]),
        used_bbox,
    )
    return result


def read_polyline_shp(
    shp_path: str | Path,
    *,
    start_height_field: str | None = DEFAULT_START_HEIGHT_FIELD,
    end_height_field: str | None = DEFAULT_END_HEIGHT_FIELD,
) -> Tuple[List[GISPolyline], BBox, Dict[str, object]]:
    shp_path = Path(shp_path)
    dbf_path = shp_path.with_suffix(".dbf")
    shx_path = shp_path.with_suffix(".shx")
    if not shp_path.exists():
        raise FileNotFoundError(str(shp_path))
    shape_type, bbox = _read_shp_header(shp_path)
    record_count, fields, dbf_records = _read_optional_dbf(
        dbf_path,
        required_fields=(start_height_field, end_height_field),
    )
    records: List[GISPolyline] = []
    skipped_deleted_dbf = 0
    shp_record_count = 0
    for idx, (_record_number, content) in enumerate(_iter_shp_record_contents(shp_path)):
        shp_record_count += 1
        dbf_record = dbf_records[idx] if idx < len(dbf_records) else None
        if dbf_records and dbf_record is None:
            skipped_deleted_dbf += 1
            continue
        parts = _read_polyline_parts_from_content(content)
        if not parts:
            continue
        records.append(
            GISPolyline(
                parts=parts,
                start_height=_read_dbf_height(dbf_record, fields, start_height_field),
                end_height=_read_dbf_height(dbf_record, fields, end_height_field),
            )
        )

    meta = {
        "source_shp": str(shp_path),
        "shape_type": int(shape_type),
        "shape_type_name": _shape_type_name(shape_type),
        "bbox": [float(value) for value in bbox],
        "dbf_available": bool(dbf_path.exists()),
        "shx_available": bool(shx_path.exists()),
        "shx_used": False,
        "dbf_records": int(record_count),
        "shp_records": int(shp_record_count),
        "loaded_records": int(len(records)),
        "skipped_deleted_dbf_records": int(skipped_deleted_dbf),
        "height_start_field": start_height_field or "",
        "height_end_field": end_height_field or "",
        "fields": sorted(fields.keys()),
    }
    meta.update(_read_prj_unit_meta(shp_path))
    return records, bbox, meta


def load_gis_matrix(
    shp_path: str | Path,
    rows: int,
    *,
    start_height_field: str | None = DEFAULT_START_HEIGHT_FIELD,
    end_height_field: str | None = DEFAULT_END_HEIGHT_FIELD,
    rotation_degrees: float = 0.0,
    boundary_shp_path: str | Path | None = None,
    boundary_field: str | None = None,
    boundary_value: str | None = None,
) -> GISMatrixResult:
    records, bbox, source_meta = read_polyline_shp(
        shp_path,
        start_height_field=start_height_field,
        end_height_field=end_height_field,
    )
    original_source_bbox = bbox
    region_meta: Dict[str, object] = {}
    if boundary_shp_path is not None:
        if not boundary_field or boundary_value is None:
            raise ValueError("Boundary field and value are required when boundary GIS is provided.")
        rings, boundary_bbox, boundary_meta = _selected_boundary_rings(
            boundary_shp_path,
            str(boundary_field),
            str(boundary_value),
        )
        pipe_crs = crs_from_meta(source_meta)
        boundary_crs = crs_from_meta(boundary_meta)
        if pipe_crs is not None and boundary_crs is not None and not crs_equivalent(pipe_crs, boundary_crs):
            rings = transform_nested_points([rings], boundary_crs, pipe_crs)[0]
            boundary_bbox = _bbox_from_parts(rings)
            compatibility_note = f"Boundary reprojected from {boundary_crs.label} to {pipe_crs.label}"
        else:
            compatibility_note = _validate_boundary_compatibility(shp_path, bbox, boundary_shp_path, boundary_bbox)
        original_record_count = len(records)
        records = clip_polylines_to_boundary(records, rings)
        bbox = boundary_bbox
        region_meta = {
            "boundary_source_shp": str(Path(boundary_shp_path)),
            "boundary_field": str(boundary_field),
            "boundary_value": str(boundary_value),
            "boundary_shape_type": int(boundary_meta.get("shape_type", 0)),
            "boundary_shape_type_name": str(boundary_meta.get("shape_type_name", "")),
            "boundary_selected_feature_count": int(boundary_meta.get("selected_feature_count", 0)),
            "boundary_selected_ring_count": int(boundary_meta.get("selected_ring_count", 0)),
            "boundary_bbox": [float(value) for value in boundary_bbox],
            "boundary_original_pipe_records": int(original_record_count),
            "boundary_clipped_pipe_records": int(len(records)),
            "boundary_compatibility_note": compatibility_note,
        }
    result = matrix_from_loaded_polylines(
        records,
        rows,
        bbox,
        source_meta=source_meta,
        rotation_degrees=rotation_degrees,
    )
    used_bbox = tuple(float(value) for value in result.meta["bbox"])
    source_bbox_meta = source_meta.get("bbox")
    result.meta.update(region_meta)
    result.meta["source_shp"] = str(Path(shp_path))
    result.meta["source_bbox"] = (
        source_bbox_meta
        if source_bbox_meta is not None
        else [float(value) for value in original_source_bbox]
    )
    result.meta["bbox"] = [float(v) for v in used_bbox]
    result.meta["rotation_degrees"] = float(rotation_degrees)
    return result
