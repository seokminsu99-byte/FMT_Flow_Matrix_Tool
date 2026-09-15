# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Uses OpenCV: https://github.com/opencv/opencv (Apache-2.0); opencv-python wheel notices apply.
# Uses NumPy: https://github.com/numpy/numpy (BSD-3-Clause plus bundled notices).
# Uses scikit-image: https://github.com/scikit-image/scikit-image (BSD-3-Clause plus per-file notices).
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import heapq
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
# Library implementation of 2-D Zhang-Suen thinning; see docs/PROVENANCE.md.
from skimage.morphology import skeletonize


DirectionCode = int
Pixel = Tuple[int, int]
Cell = Tuple[int, int]
DEFAULT_INK_THRESHOLD = 0.003
_DIRECTION_CODE_TO_CELL_STEP: Dict[int, Tuple[int, int]] = {
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
    4: (-1, 0),
}


@dataclass
class FeatureConfig:
    """Configuration for image preprocessing."""

    A: int = 50
    patch_size: int = 96
    pad_ratio: float = 1.75
    strength: int = 3
    use_edges: bool = True
    use_gray: bool = False
    q_min: float = 0.08
    q_step: float = 0.03


@dataclass
class ArrowCandidate:
    """Detected arrow-head candidate."""

    bbox: Tuple[int, int, int, int]
    tip: Tuple[float, float]
    direction: Tuple[float, float]
    score: float
    area: float
    component_id: int


@dataclass
class ArrowObjectPath:
    """Ordered cell path for a single arrow object."""

    component_id: int
    candidate_index: int
    bbox: Tuple[int, int, int, int]
    head_cell: Cell
    tail_cell: Cell
    raw_cell_path: Tuple[Cell, ...]
    vote_cell_path: Tuple[Cell, ...]
    score: float
    direction_code: DirectionCode = 0


@dataclass(frozen=True)
class ArrowCoreEvidence:
    """Direction evidence from a shaft-suppressing erosion of one arrowhead core."""

    center: Tuple[float, float]
    tip: Tuple[float, float]
    direction: Tuple[float, float]
    asymmetry: float
    axis_ratio: float
    area: int
    bbox: Tuple[int, int, int, int]


def derive_grid_B(A: int, W: int, H: int) -> int:
    """Derive column count from image aspect ratio."""
    if A <= 0:
        raise ValueError("A must be positive")
    if H <= 0 or W <= 0:
        raise ValueError("image width and height must be positive")
    return max(1, int(round(A * (W / float(H)))))


def outer_zero_margin_bounds(
    matrix: np.ndarray,
    *,
    padding: int = 1,
    support_masks: Sequence[np.ndarray] = (),
    protected_cells: Sequence[Cell] = (),
) -> Tuple[int, int, int, int] | None:
    """Return half-open bounds that remove only empty outer rows and columns.

    Internal empty rows and columns are intentionally retained.  Support masks
    and protected cells let callers preserve occupied-but-unresolved cells and
    selected outlets even when their direction value is zero.
    """
    values = np.asarray(matrix)
    if values.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("matrix must not be empty")
    padding = int(padding)
    if padding < 0:
        raise ValueError("padding must be non-negative")

    occupied = values != 0
    for index, mask in enumerate(support_masks):
        support = np.asarray(mask)
        if support.shape != values.shape:
            raise ValueError(
                f"support mask {index} shape {support.shape} does not match matrix shape {values.shape}"
            )
        occupied |= support != 0

    rows, cols = values.shape
    for cell in protected_cells:
        row, col = int(cell[0]), int(cell[1])
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError(f"protected cell {(row, col)} is outside matrix shape {values.shape}")
        occupied[row, col] = True

    occupied_rows, occupied_cols = np.nonzero(occupied)
    if occupied_rows.size == 0:
        return None
    row_start = max(0, int(occupied_rows.min()) - padding)
    row_stop = min(rows, int(occupied_rows.max()) + padding + 1)
    col_start = max(0, int(occupied_cols.min()) - padding)
    col_stop = min(cols, int(occupied_cols.max()) + padding + 1)
    return row_start, row_stop, col_start, col_stop


def direction_cells_reaching_outlets(
    matrix: np.ndarray,
    outlets: Sequence[Cell],
) -> np.ndarray:
    """Return a mask of direction cells whose flow ends at a selected outlet.

    Each non-zero cell has exactly one outgoing edge according to the NFMAT
    direction codes (1=east, 2=south, 3=west, 4=north).  Starting at the
    outlets and walking the reverse edges marks every cell that genuinely
    drains to one of them.  Dead ends, out-of-bounds paths, and cycles that do
    not contain an outlet remain unmarked.
    """
    values = np.asarray(matrix)
    if values.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("matrix must not be empty")
    if np.any((values < 0) | (values > 4)):
        raise ValueError("matrix contains a direction code outside 0..4")

    rows, cols = values.shape
    normalized: set[Cell] = set()
    for outlet in outlets:
        row, col = int(outlet[0]), int(outlet[1])
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError(f"outlet {(row, col)} is outside matrix shape {values.shape}")
        normalized.add((row, col))

    reached = np.zeros(values.shape, dtype=np.uint8)
    queue: deque[Cell] = deque()
    for outlet in sorted(normalized):
        reached[outlet] = 1
        queue.append(outlet)

    # For a reached cell, these are the neighboring predecessor positions and
    # the direction code each predecessor must contain to flow into it.
    predecessor_steps = (
        (-1, 0, 2),
        (1, 0, 4),
        (0, -1, 1),
        (0, 1, 3),
    )
    while queue:
        row, col = queue.popleft()
        for drow, dcol, required_direction in predecessor_steps:
            pred_row, pred_col = row + drow, col + dcol
            if not (0 <= pred_row < rows and 0 <= pred_col < cols):
                continue
            if int(reached[pred_row, pred_col]) == 1:
                continue
            if int(values[pred_row, pred_col]) != required_direction:
                continue
            reached[pred_row, pred_col] = 1
            queue.append((pred_row, pred_col))
    return reached


def fill_basin_to_network(
    matrix: np.ndarray, basin_mask: np.ndarray, outlets: Sequence[Cell] = (),
    *, network_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Route only basin zeros to the nearest original pipe, not to an outlet.

    Distance is to the first original pipe, not to its downstream outlet: a
    four-neighbor path through basin zeros, ending at any allowed original
    pipe even outside the fill basin. Original directions are immutable;
    outlet reachability is diagnostic only and NEVER filters seed nodes.
    Row-major seeds and north/south/west/east expansion make ties reproducible.
    Each added direction points one BFS level closer to the original network.
    """
    from workspace_state import validated_direction_matrix, intersect_region_masks

    original = validated_direction_matrix(matrix)
    basin = intersect_region_masks(original.shape, basin_mask)
    if basin is None:
        raise ValueError("빈 칸 채우기에는 명시적인 유역 경계가 필요합니다.")
    # Optional workspace separates existing-pipe routing from area-fill domain.
    # By default both domains coincide, preserving the strict standalone API.
    allowed = basin if network_mask is None else intersect_region_masks(original.shape, network_mask)
    basin = (basin & allowed).astype(np.uint8)
    normalized = []
    rows, cols = original.shape
    for cell in outlets:
        if len(cell) != 2 or any(int(v) != v for v in cell):
            raise ValueError("Outlet 좌표는 정수 행과 열이어야 합니다.")
        row, col = int(cell[0]), int(cell[1])
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError("Outlet이 행렬 범위 밖에 있습니다.")
        if not allowed[row, col] or not original[row, col]:
            raise ValueError("작업 범위 내부의 0이 아닌 관망 셀을 Outlet으로 선택해 주세요.")
        normalized.append((row, col))
    inside_network = np.where(allowed, original, 0).astype(np.uint8)
    connected = direction_cells_reaching_outlets(inside_network, normalized) if normalized else None
    # Existing pipes throughout the allowed workspace are equally near at
    # distance zero. Only expansion into NEW cells is restricted to the basin.
    seeds = inside_network > 0
    if not np.any(seeds):
        raise ValueError("허용 영역에 연결할 기존 관로가 없습니다. 관로 행렬을 먼저 확인해 주세요.")
    result = original.copy()
    added = np.zeros(original.shape, dtype=np.uint8)
    queue = deque(zip(*np.nonzero(seeds)))
    # Neighbor -> current direction (the reverse of expansion).
    steps = ((-1, 0, 2), (1, 0, 4), (0, -1, 1), (0, 1, 3))
    while queue:
        row, col = queue.popleft()
        for dr, dc, direction in steps:
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols and basin[nr, nc] and result[nr, nc] == 0:
                result[nr, nc] = direction
                added[nr, nc] = 1
                queue.append((nr, nc))
    return result, added, {
        "filled_count": int(np.count_nonzero(added)),
        "unfilled_count": int(np.count_nonzero((basin > 0) & (result == 0))),
        "seed_count": int(np.count_nonzero(seeds)),
        "outlet_checked": bool(normalized),
        "existing_unreachable_count": (
            int(np.count_nonzero(seeds & (connected == 0))) if connected is not None else None
        ),
    }


def analyze_direction_network(matrix: np.ndarray, outlets: Sequence[Cell] = ()) -> dict:
    """Linear-time graph counts, treating selected outlets as terminal nodes."""
    from workspace_state import validated_direction_matrix

    values = validated_direction_matrix(matrix)
    rows, cols = values.shape
    active = values.ravel() > 0
    reached = direction_cells_reaching_outlets(values, outlets).ravel() > 0
    terminal = {int(r) * cols + int(c) for r, c in outlets}
    downstream = np.full(values.size, -1, dtype=np.int64)
    indegree = np.zeros(values.size, dtype=np.int32)
    dead_ends = exits = 0
    for index in np.flatnonzero(active):
        if index in terminal:
            continue
        row, col = divmod(int(index), cols)
        dr, dc = _DIRECTION_CODE_TO_CELL_STEP[int(values[row, col])]
        nr, nc = row + dr, col + dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            exits += 1
        elif not values[nr, nc]:
            dead_ends += 1
        else:
            target = nr * cols + nc
            downstream[index] = target
            indegree[target] += 1
    sources = int(np.count_nonzero(active & (indegree == 0)))
    junctions = int(np.count_nonzero(indegree >= 2))
    queue = deque(np.flatnonzero(active & (indegree == 0)))
    remaining = active.copy()
    while queue:
        node = queue.popleft()
        remaining[node] = False
        target = downstream[node]
        if target >= 0:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return {
        "active_count": int(np.count_nonzero(active)),
        "connected_count": int(np.count_nonzero(active & reached)),
        "unreachable_count": int(np.count_nonzero(active & ~reached)),
        "source_count": sources, "junction_count": junctions,
        "outlet_count": len(terminal), "cycle_cell_count": int(np.count_nonzero(remaining)),
        "dead_end_count": dead_ends, "boundary_exit_count": exits,
    }


def _odd_kernel(size: int) -> int:
    size = max(1, int(size))
    return size if size % 2 == 1 else size + 1


def _relative_kernel(shape: Sequence[int], scale: float, minimum: int = 1) -> int:
    return _odd_kernel(max(minimum, int(round(min(shape[:2]) * scale))))


def build_binary(gray: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Build a robust binary image using relative thresholds."""
    if gray.ndim != 2:
        raise ValueError("build_binary expects a single-channel image")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    q = float(np.clip(cfg.q_min + cfg.q_step * cfg.strength, 0.02, 0.35))
    dark = blur[blur < 248]
    if dark.size:
        q_thr = float(np.quantile(dark, min(0.95, q + 0.12)))
        _, quant = cv2.threshold(blur, q_thr, 255, cv2.THRESH_BINARY_INV)
    else:
        quant = np.zeros_like(gray)

    adaptive_block = _relative_kernel(gray.shape, 0.03, minimum=15)
    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block,
        2,
    )

    strong = cv2.bitwise_or(cv2.bitwise_and(otsu, adaptive), cv2.bitwise_and(quant, adaptive))
    strong = cv2.bitwise_or(strong, cv2.bitwise_and(otsu, quant))
    weak = cv2.bitwise_or(otsu, cv2.bitwise_or(quant, adaptive))
    support_k = _relative_kernel(gray.shape, 0.0040, minimum=3)
    support_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (support_k, support_k))
    support_zone = cv2.dilate(strong, support_kernel, iterations=1)
    bw = cv2.bitwise_or(strong, cv2.bitwise_and(weak, support_zone))

    open_k = _relative_kernel(gray.shape, 0.0025, minimum=1)
    close_k = _relative_kernel(gray.shape, 0.0035, minimum=1)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, open_kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # Only keep bridged pixels when they stay inside a small neighborhood of real ink.
    line_k = _relative_kernel(gray.shape, 0.0048, minimum=3)
    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (line_k, 1))
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_k))
    close_h = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, horizontal, iterations=1)
    close_v = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, vertical, iterations=1)
    bridge_zone = cv2.dilate(bw, support_kernel, iterations=1)
    bridged = cv2.bitwise_or(close_h, close_v)
    bw = cv2.bitwise_or(bw, cv2.bitwise_and(bridged, bridge_zone))
    return bw


def build_arrow_detail_binary(image: np.ndarray) -> np.ndarray:
    """Build a shape-preserving mask for black arrow glyphs.

    Pipe occupancy needs small morphological bridges, but those same operations
    widen shafts and destroy the arrowhead taper.  Arrow detection therefore
    uses this strict, relative-threshold mask while pipe tracing keeps using
    :func:`build_binary`.
    """
    if image.ndim == 2:
        gray = image
        chroma_ratio = None
    elif image.ndim == 3 and image.shape[2] >= 3:
        bgr = image[:, :, :3].astype(np.float32)
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        high = np.max(bgr, axis=2)
        low = np.min(bgr, axis=2)
        chroma_ratio = (high - low) / np.maximum(high, 1.0)
    else:
        raise ValueError("build_arrow_detail_binary expects a grayscale or BGR image")

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    _threshold, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if chroma_ratio is None:
        return dark.astype(np.uint8)

    dark_values = chroma_ratio[dark > 0]
    if dark_values.size == 0:
        return dark.astype(np.uint8)
    neutral_limit = float(np.clip(np.quantile(dark_values, 0.40) + 0.04, 0.08, 0.24))
    detail = np.where((dark > 0) & (chroma_ratio <= neutral_limit), 255, 0).astype(np.uint8)

    # Fall back only when neutral ink is effectively absent.  A ratio-based
    # fallback lets a large coloured GIS boundary overwhelm a valid thin black
    # pipe and incorrectly turns the boundary into pipe occupancy.
    if int(np.count_nonzero(detail)) < 8:
        return dark.astype(np.uint8)
    return detail


def build_pipe_binary(
    image: np.ndarray,
    cfg: FeatureConfig,
    robust_bw: np.ndarray | None = None,
) -> np.ndarray:
    """Build a shape-preserving mask for neutral pipe and arrow ink.

    The general morphology mask is useful for closing small scan gaps, but a
    square opening can erase a one- or two-pixel diagonal pipe.  The strict
    Otsu mask preserves that source ink.  For colour drawings, the neutral-ink
    mask also prevents administrative boundaries from becoming pipes.
    """
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] >= 3:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("build_pipe_binary expects a grayscale or BGR image")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    robust = build_binary(gray, cfg) if robust_bw is None else np.asarray(robust_bw, dtype=np.uint8)
    detail = build_arrow_detail_binary(image)
    if robust.shape != detail.shape:
        raise ValueError("robust and detail pipe masks must have the same shape")

    if image.ndim == 2:
        combined = cv2.bitwise_or(robust, detail)
    else:
        support_k = _relative_kernel(detail.shape, 0.0018, minimum=1)
        support_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (support_k, support_k))
        neutral_zone = cv2.dilate(detail, support_kernel, iterations=1)
        combined = cv2.bitwise_or(detail, cv2.bitwise_and(robust, neutral_zone))

    # Remove isolated single-pixel scan noise without changing thin connected lines.
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (combined > 0).astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        return combined.astype(np.uint8)
    keep = np.zeros((count,), dtype=np.uint8)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= 2).astype(np.uint8)
    return np.where(keep[labels] > 0, 255, 0).astype(np.uint8)


def build_features(bgr: np.ndarray, cfg: FeatureConfig) -> Dict[str, np.ndarray]:
    """Create grayscale, binary and edge images."""
    if bgr.ndim == 2:
        gray = bgr.copy()
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    robust_bw = build_binary(gray, cfg)
    bw = build_pipe_binary(bgr, cfg, robust_bw=robust_bw)
    features = {"gray": gray, "bw": bw, "robust_bw": robust_bw}
    if cfg.use_edges:
        lo = 40 + 6 * max(0, int(cfg.strength))
        hi = 110 + 10 * max(0, int(cfg.strength))
        edges = cv2.Canny(gray, lo, hi)
        edge_k = _relative_kernel(gray.shape, 0.0018, minimum=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_k, edge_k))
        features["edges"] = cv2.dilate(edges, kernel, iterations=1)
    else:
        features["edges"] = np.zeros_like(gray)
    return features


def occupancy_from_binary(bw: np.ndarray, A: int, B: int, ink_thr: float) -> np.ndarray:
    """Estimate which cells contain arrow ink."""
    H, W = bw.shape[:2]
    cell_h = H / float(A)
    cell_w = W / float(B)
    occ = np.zeros((A, B), dtype=np.uint8)

    for i in range(A):
        y1 = int(round(i * cell_h))
        y2 = int(round((i + 1) * cell_h))
        y1 = max(0, min(H - 1, y1))
        y2 = max(y1 + 1, min(H, y2))
        for j in range(B):
            x1 = int(round(j * cell_w))
            x2 = int(round((j + 1) * cell_w))
            x1 = max(0, min(W - 1, x1))
            x2 = max(x1 + 1, min(W, x2))

            cell = bw[y1:y2, x1:x2]
            if cell.size == 0:
                continue
            ratio = float(np.mean(cell > 0))
            count = int(np.count_nonzero(cell > 0))
            area = int(cell.size)
            min_count = max(5, int(round(area * max(ink_thr * 0.65, 0.0030))))
            occupied = ratio >= ink_thr or (
                count >= min_count and ratio >= max(ink_thr * 0.45, 0.0015)
            )
            occ[i, j] = 1 if occupied else 0
    return occ


def _cell_area(H: int, W: int, A: int, B: int) -> float:
    return max(1.0, (H / float(A)) * (W / float(B)))


def _label_binary_components(bw: np.ndarray) -> np.ndarray:
    """Label connected foreground objects in the binary image."""
    _count, labels = cv2.connectedComponents((bw > 0).astype(np.uint8), connectivity=8)
    return labels


def score_arrowhead_like(contour: np.ndarray, cell_area: float) -> float:
    """Score how likely a contour is to be an arrow head."""
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return -1.0

    peri = float(cv2.arcLength(contour, True))
    if peri <= 0:
        return -1.0

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull)) if hull is not None else area
    x, y, w, h = cv2.boundingRect(contour)
    aspect = max(w, h) / max(1.0, min(w, h))
    extent = area / max(1.0, w * h)
    solidity = area / max(hull_area, 1e-6)
    approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
    n_vertices = len(approx)
    area_ratio = area / max(cell_area, 1.0)

    vertex_score = 1.0 if 3 <= n_vertices <= 6 else 0.35
    solidity_score = 1.0 - min(abs(solidity - 0.82), 0.82) / 0.82
    extent_score = 1.0 - min(abs(extent - 0.42), 0.42) / 0.42
    aspect_score = 1.0 - min(abs(min(aspect, 5.0) - 1.7), 3.3) / 3.3
    size_score = 1.0 if 0.08 <= area_ratio <= 6.5 else 0.25

    return float(
        0.28 * vertex_score
        + 0.22 * solidity_score
        + 0.22 * extent_score
        + 0.18 * aspect_score
        + 0.10 * size_score
    )


def _tip_from_contour(contour: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-6:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        pts = contour.reshape(-1, 2)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))

    center = np.array([cx, cy], dtype=np.float32)
    pts = contour.reshape(-1, 2).astype(np.float32)

    hull = cv2.convexHull(contour, returnPoints=True)
    hull_pts = hull.reshape(-1, 2).astype(np.float32) if hull is not None else np.empty((0, 2), dtype=np.float32)
    if len(hull_pts) >= 3:
        best_tip: np.ndarray | None = None
        best_angle = float("inf")
        best_dist = -1.0
        for idx in range(len(hull_pts)):
            cur = hull_pts[idx]
            prev_vec = hull_pts[idx - 1] - cur
            next_vec = hull_pts[(idx + 1) % len(hull_pts)] - cur
            norm_prev = float(np.linalg.norm(prev_vec))
            norm_next = float(np.linalg.norm(next_vec))
            if norm_prev <= 1e-6 or norm_next <= 1e-6:
                continue
            cos_angle = float(np.clip(np.dot(prev_vec, next_vec) / (norm_prev * norm_next), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cos_angle)))
            dist = float(np.linalg.norm(cur - center))
            if angle < best_angle - 1e-3 or (abs(angle - best_angle) <= 2.0 and dist > best_dist):
                best_tip = cur
                best_angle = angle
                best_dist = dist
        if best_tip is not None and best_angle <= 120.0:
            direction = best_tip - center
            norm = float(np.linalg.norm(direction))
            if norm > 1e-6:
                direction = direction / norm
                return (float(best_tip[0]), float(best_tip[1])), (float(direction[0]), float(direction[1]))

    if len(pts) >= 2:
        shifted = pts - center
        cov = np.cov(shifted.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))].astype(np.float32)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm > 1e-6:
            axis = axis / axis_norm
            perp = np.array([-axis[1], axis[0]], dtype=np.float32)
            proj = shifted @ axis
            perp_proj = shifted @ perp
            tmin = float(np.min(proj))
            tmax = float(np.max(proj))
            band = max(4.0, 0.18 * (tmax - tmin))
            min_mask = proj <= (tmin + band)
            max_mask = proj >= (tmax - band)
            min_width = float(np.ptp(perp_proj[min_mask])) if np.any(min_mask) else 0.0
            max_width = float(np.ptp(perp_proj[max_mask])) if np.any(max_mask) else 0.0
            use_max = max_width >= min_width
            extreme_idx = int(np.argmax(proj) if use_max else np.argmin(proj))
            tip = pts[extreme_idx]
            direction = tip - center
            norm = float(np.linalg.norm(direction))
            if norm > 1e-6:
                direction = direction / norm
                return (float(tip[0]), float(tip[1])), (float(direction[0]), float(direction[1]))

    distances = np.sum((pts - center) ** 2, axis=1)
    tip = pts[int(np.argmax(distances))]
    direction = tip - center
    norm = float(np.linalg.norm(direction))
    if norm > 1e-6:
        direction = direction / norm
    else:
        direction = np.array([1.0, 0.0], dtype=np.float32)
    return (float(tip[0]), float(tip[1])), (float(direction[0]), float(direction[1]))


def _contour_tip_candidates(
    contour: np.ndarray,
    cell_area: float,
    margin: int,
    component_id: int,
    branch_nodes: Sequence[Pixel],
    H: int,
    W: int,
    max_tips: int = 4,
) -> List[ArrowCandidate]:
    """Return local contour-tip candidates for merged/overlapping arrow components."""
    base_score = score_arrowhead_like(contour, cell_area)
    if base_score < 0.05:
        return []
    if not branch_nodes:
        return []

    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-6:
        center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float32)
    else:
        pts = contour.reshape(-1, 2).astype(np.float32)
        center = np.mean(pts, axis=0, dtype=np.float32)

    hull = cv2.convexHull(contour, returnPoints=True)
    hull_pts = hull.reshape(-1, 2).astype(np.float32) if hull is not None else np.empty((0, 2), dtype=np.float32)
    if len(hull_pts) < 3:
        return []

    branch_xy = np.asarray([[float(x), float(y)] for y, x in branch_nodes], dtype=np.float32)
    probe_len = float(max(8, int(round(margin * 2.2))))
    lateral_limit = float(max(3.0, margin * 0.95))
    candidates: List[Tuple[float, ArrowCandidate]] = []

    for idx, tip in enumerate(hull_pts):
        vec = tip - center
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-6:
            continue
        unit = vec / norm

        prev_vec = hull_pts[idx - 1] - tip
        next_vec = hull_pts[(idx + 1) % len(hull_pts)] - tip
        prev_norm = float(np.linalg.norm(prev_vec))
        next_norm = float(np.linalg.norm(next_vec))
        if prev_norm <= 1e-6 or next_norm <= 1e-6:
            continue
        cos_angle = float(np.clip(np.dot(prev_vec, next_vec) / (prev_norm * next_norm), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos_angle)))
        if angle > 165.0:
            continue
        angle_score = float(np.clip((178.0 - angle) / 70.0, 0.0, 1.0))

        branch_score = 0.0
        branch_anchor = tip.copy()
        if branch_xy.size:
            rel = branch_xy - tip.reshape(1, 2)
            behind = -(rel @ unit)
            lateral = np.abs(rel[:, 0] * unit[1] - rel[:, 1] * unit[0])
            valid = (behind >= 2.0) & (behind <= probe_len) & (lateral <= lateral_limit)
            if np.any(valid):
                proximity = np.exp(-((behind[valid] / max(probe_len, 1e-6)) ** 2))
                lateral_score = np.exp(-((lateral[valid] / max(lateral_limit, 1e-6)) ** 2))
                scores = proximity * lateral_score
                best_idx = int(np.argmax(scores))
                valid_indices = np.flatnonzero(valid)
                anchor_idx = int(valid_indices[best_idx])
                branch_score = float(scores[best_idx])
                branch_anchor = branch_xy[anchor_idx]

        distance_score = float(np.clip(norm / max(np.sqrt(cell_area) * 1.4, 1.0), 0.0, 1.0))
        score = (
            0.34 * float(base_score)
            + 0.40 * branch_score
            + 0.16 * angle_score
            + 0.10 * distance_score
        )

        # Flat tail ends may be sharp and distant, but they lack a nearby branch behind
        # the putative tip. Keep only strong branch-supported tips in complex components.
        if branch_xy.size and branch_score < 0.22:
            continue
        if score < 0.18:
            continue

        head_margin = int(max(2, round(float(margin) * 1.65)))
        x1 = int(max(0, min(float(tip[0]), float(branch_anchor[0])) - head_margin))
        y1 = int(max(0, min(float(tip[1]), float(branch_anchor[1])) - head_margin))
        x2 = int(min(W, max(float(tip[0]), float(branch_anchor[0])) + head_margin + 1))
        y2 = int(min(H, max(float(tip[1]), float(branch_anchor[1])) + head_margin + 1))
        candidates.append(
            (
                float(score),
                ArrowCandidate(
                    bbox=(x1, y1, x2, y2),
                    tip=(float(tip[0]), float(tip[1])),
                    direction=(float(unit[0]), float(unit[1])),
                    score=float(score),
                    area=float(cv2.contourArea(contour)),
                    component_id=int(component_id),
                ),
            )
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    filtered: List[ArrowCandidate] = []
    min_tip_gap_sq = float(max(3, margin) ** 2)
    for _score, candidate in candidates:
        if any((candidate.tip[0] - existing.tip[0]) ** 2 + (candidate.tip[1] - existing.tip[1]) ** 2 <= min_tip_gap_sq for existing in filtered):
            continue
        filtered.append(candidate)
        if len(filtered) >= max(1, int(max_tips)):
            break
    return filtered


def _contour_bisector_candidates(
    contour: np.ndarray,
    bw: np.ndarray,
    A: int,
    B: int,
    margin: int,
    component_id: int,
    max_tips: int = 12,
) -> List[ArrowCandidate]:
    """Detect arrow tips from acute contour corners.

    For a filled/open arrow head, the flow direction is the opposite of the
    contour corner's interior angle bisector. This avoids using the whole
    component centroid, which is unstable when many arrows are connected.
    """
    H, W = bw.shape[:2]
    cell_span = float(max(H / float(A), W / float(B), 1.0))
    peri = float(cv2.arcLength(contour, True))
    if peri <= 0:
        return []
    approx = cv2.approxPolyDP(contour, max(1.5, cell_span * 0.055), True)
    pts = approx.reshape(-1, 2).astype(np.float32)
    if len(pts) < 3:
        return []

    candidates: List[ArrowCandidate] = []
    for idx, tip in enumerate(pts):
        prev_vec = pts[idx - 1] - tip
        next_vec = pts[(idx + 1) % len(pts)] - tip
        prev_norm = float(np.linalg.norm(prev_vec))
        next_norm = float(np.linalg.norm(next_vec))
        if prev_norm < cell_span * 0.22 or next_norm < cell_span * 0.22:
            continue
        prev_unit = prev_vec / prev_norm
        next_unit = next_vec / next_norm
        cos_angle = float(np.clip(np.dot(prev_unit, next_unit), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos_angle)))
        if angle < 22.0 or angle > 112.0:
            continue

        interior = prev_unit + next_unit
        interior_norm = float(np.linalg.norm(interior))
        if interior_norm <= 1e-6:
            continue
        direction = -(interior / interior_norm)
        radius = int(max(3, round(max(float(margin) * 1.55, cell_span * 1.35))))
        x1 = int(max(0, round(float(tip[0]) - radius)))
        y1 = int(max(0, round(float(tip[1]) - radius)))
        x2 = int(min(W, round(float(tip[0]) + radius + 1)))
        y2 = int(min(H, round(float(tip[1]) + radius + 1)))
        candidate = ArrowCandidate(
            bbox=(x1, y1, x2, y2),
            tip=(float(tip[0]), float(tip[1])),
            direction=(float(direction[0]), float(direction[1])),
            score=0.0,
            area=1.0,
            component_id=int(component_id),
        )
        front_ink_ratio = _candidate_front_ink_ratio(bw, candidate, A, B)
        if front_ink_ratio > 0.42:
            continue
        geometry_score = _candidate_local_geometry_score(bw, candidate, A, B)
        angle_score = float(np.clip((112.0 - angle) / 72.0, 0.0, 1.0))
        edge_score = float(np.clip(min(prev_norm, next_norm) / max(cell_span * 0.70, 1.0), 0.0, 1.0))
        front_score = 1.0 - min(1.0, front_ink_ratio / 0.42)
        score = float(np.clip(0.58 * geometry_score + 0.20 * angle_score + 0.14 * edge_score + 0.08 * front_score, 0.0, 1.0))
        if score < 0.48:
            continue
        candidates.append(
            ArrowCandidate(
                bbox=(x1, y1, x2, y2),
                tip=(float(tip[0]), float(tip[1])),
                direction=(float(direction[0]), float(direction[1])),
                score=score,
                area=float(max(1.0, cv2.contourArea(contour))),
                component_id=int(component_id),
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    filtered: List[ArrowCandidate] = []
    for candidate in candidates:
        if any(_bbox_iou(candidate.bbox, existing.bbox) > 0.34 for existing in filtered):
            continue
        if any(
            (candidate.tip[0] - existing.tip[0]) ** 2 + (candidate.tip[1] - existing.tip[1]) ** 2
            <= float((cell_span * 0.82) ** 2)
            for existing in filtered
        ):
            continue
        filtered.append(candidate)
        if len(filtered) >= max(1, int(max_tips)):
            break
    return filtered


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
    area_b = float(max(1, bx2 - bx1) * max(1, by2 - by1))
    return inter / max(area_a + area_b - inter, 1e-6)


def _candidate_direction_dot(a: ArrowCandidate, b: ArrowCandidate) -> float:
    va = np.asarray(a.direction, dtype=np.float32)
    vb = np.asarray(b.direction, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 1e-6 or nb <= 1e-6:
        return 1.0
    return float(np.dot(va / na, vb / nb))


def _foreground_at(bw: np.ndarray, x: float, y: float) -> bool:
    H, W = bw.shape[:2]
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    if xi < 0 or yi < 0 or xi >= W or yi >= H:
        return False
    return bool(bw[yi, xi] > 0)


def _candidate_local_geometry_score(
    bw: np.ndarray,
    candidate: ArrowCandidate,
    A: int,
    B: int,
) -> float:
    """Score a local arrowhead taper without assuming that the shaft ends at the tip."""
    H, W = bw.shape[:2]
    unit = np.asarray(candidate.direction, dtype=np.float32)
    norm = float(np.linalg.norm(unit))
    if norm <= 1e-6:
        return 0.0
    unit /= norm
    perp = np.asarray([-unit[1], unit[0]], dtype=np.float32)
    tip = np.asarray([float(candidate.tip[0]), float(candidate.tip[1])], dtype=np.float32)
    cell_span = float(max(H / float(A), W / float(B), 1.0))
    step = max(1.0, cell_span / 10.0)
    half = max(3.0, cell_span * 1.40)

    def cross_stats(distance_cells: float) -> Tuple[float, float]:
        center = tip - unit * float(distance_cells * cell_span)
        offsets = np.arange(-half, half + 0.5 * step, step, dtype=np.float32)
        hit_mask: List[bool] = []
        for offset in offsets:
            point = center + perp * float(offset)
            hit_mask.append(_foreground_at(bw, float(point[0]), float(point[1])))
        hit_indices = np.flatnonzero(np.asarray(hit_mask, dtype=bool))
        if hit_indices.size == 0:
            return 0.0, 0.0
        center_index = int(np.argmin(np.abs(offsets)))
        anchor_index = int(hit_indices[int(np.argmin(np.abs(hit_indices - center_index)))])
        if abs(float(offsets[anchor_index])) > max(step * 2.5, cell_span * 0.18):
            return 0.0, 0.0
        left_index = anchor_index
        right_index = anchor_index
        while left_index > 0 and bool(hit_mask[left_index - 1]):
            left_index -= 1
        while right_index + 1 < len(hit_mask) and bool(hit_mask[right_index + 1]):
            right_index += 1
        minimum = float(offsets[left_index])
        maximum = float(offsets[right_index])
        width = maximum - minimum
        neg = abs(min(0.0, minimum))
        pos = max(0.0, maximum)
        balance = min(neg, pos) / max(max(neg, pos), 1e-6)
        return float(width), float(balance)

    front_lateral_hits = 0
    front_lateral_total = 0
    for d in (0.35, 0.70, 1.05):
        center = tip + unit * float(d * cell_span)
        # Flow arrows in the source drawings are commonly embedded in a longer
        # pipe.  The centreline ahead is therefore neutral evidence; only ink
        # away from that centreline contradicts a sharp tip.
        for offset in (-0.42, -0.24, 0.24, 0.42):
            point = center + perp * float(offset * cell_span)
            front_lateral_hits += int(_foreground_at(bw, float(point[0]), float(point[1])))
            front_lateral_total += 1
    front_lateral_clear = 1.0 - float(front_lateral_hits) / max(1.0, float(front_lateral_total))

    behind_hits = 0
    behind_total = 0
    for d in (0.25, 0.55, 0.85, 1.20):
        center = tip - unit * float(d * cell_span)
        for offset in (-0.14, 0.0, 0.14):
            point = center + perp * float(offset * cell_span)
            behind_hits += int(_foreground_at(bw, float(point[0]), float(point[1])))
            behind_total += 1
    behind_support = float(behind_hits) / max(1.0, float(behind_total))

    shaft_hits = 0
    shaft_total = 0
    shaft_distance_hits = 0
    for d in (0.85, 1.30, 1.85, 2.45, 3.10):
        center = tip - unit * float(d * cell_span)
        local_hits = 0
        for offset in (-0.18, -0.08, 0.0, 0.08, 0.18):
            point = center + perp * float(offset * cell_span)
            hit = int(_foreground_at(bw, float(point[0]), float(point[1])))
            shaft_hits += hit
            local_hits += hit
            shaft_total += 1
        shaft_distance_hits += int(local_hits > 0)
    shaft_support = float(shaft_hits) / max(1.0, float(shaft_total))
    shaft_continuity = float(shaft_distance_hits) / 5.0

    tip_width, _tip_balance = cross_stats(0.18)
    mid_width, mid_balance = cross_stats(0.62)
    base_width, base_balance = cross_stats(0.95)
    ahead_width, _ahead_balance = cross_stats(-0.48)
    head_widths = [mid_width, base_width, cross_stats(1.35)[0], cross_stats(1.70)[0]]
    far_widths = [cross_stats(distance)[0] for distance in (2.40, 3.20, 4.00)]
    positive_far_widths = [width for width in far_widths if width > 0]
    far_shaft_width = min(positive_far_widths) if positive_far_widths else 0.0
    head_width = max(head_widths)
    head_bulge = float(
        np.clip((head_width - far_shaft_width) / max(cell_span * 0.42, 1.0), 0.0, 1.0)
    )
    width_gain = float(np.clip((head_width - tip_width) / max(cell_span * 0.34, 1.0), 0.0, 1.0))
    directional_taper = float(
        np.clip((head_width - ahead_width) / max(cell_span * 0.34, 1.0), 0.0, 1.0)
    )
    balance = max(mid_balance, base_balance)
    base_score = float(
        np.clip(
            0.16 * behind_support
            + 0.10 * front_lateral_clear
            + 0.16 * width_gain
            + 0.13 * directional_taper
            + 0.12 * balance
            + 0.08 * shaft_support
            + 0.08 * shaft_continuity
            + 0.17 * head_bulge,
            0.0,
            1.0,
        )
    )
    return float(np.clip(base_score * (0.28 + 0.72 * head_bulge), 0.0, 1.0))


def _candidate_front_ink_ratio(
    bw: np.ndarray,
    candidate: ArrowCandidate,
    A: int,
    B: int,
) -> float:
    H, W = bw.shape[:2]
    unit = np.asarray(candidate.direction, dtype=np.float32)
    norm = float(np.linalg.norm(unit))
    if norm <= 1e-6:
        return 1.0
    unit /= norm
    perp = np.asarray([-unit[1], unit[0]], dtype=np.float32)
    tip = np.asarray([float(candidate.tip[0]), float(candidate.tip[1])], dtype=np.float32)
    cell_span = float(max(H / float(A), W / float(B), 1.0))
    hits = 0
    total = 0
    for distance_cells in (0.35, 0.70, 1.05, 1.40):
        center = tip + unit * float(distance_cells * cell_span)
        for offset_cells in (-0.42, -0.24, 0.24, 0.42):
            point = center + perp * float(offset_cells * cell_span)
            hits += int(_foreground_at(bw, float(point[0]), float(point[1])))
            total += 1
    return float(hits) / max(1.0, float(total))


def _candidate_touches_image_border(candidate: ArrowCandidate, H: int, W: int, margin: int = 1) -> bool:
    x, y = float(candidate.tip[0]), float(candidate.tip[1])
    return bool(x <= margin or y <= margin or x >= W - 1 - margin or y >= H - 1 - margin)


def _candidate_with_direction(candidate: ArrowCandidate, direction: np.ndarray, score: float) -> ArrowCandidate:
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        direction = np.asarray(candidate.direction, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        direction = np.asarray([1.0, 0.0], dtype=np.float32)
    else:
        direction = direction / norm
    return ArrowCandidate(
        bbox=tuple(int(v) for v in candidate.bbox),
        tip=(float(candidate.tip[0]), float(candidate.tip[1])),
        direction=(float(direction[0]), float(direction[1])),
        score=float(score),
        area=float(candidate.area),
        component_id=int(candidate.component_id),
    )


def _refine_candidate_direction_by_geometry(
    bw: np.ndarray,
    candidate: ArrowCandidate,
    A: int,
    B: int,
) -> Tuple[ArrowCandidate, float]:
    """Re-score the candidate direction using local relative geometry."""
    base = np.asarray(candidate.direction, dtype=np.float32)
    base_norm = float(np.linalg.norm(base))
    directions: List[np.ndarray] = []
    if base_norm > 1e-6:
        base = base / base_norm
        # Direction is the core recognition target. Always compare the initial
        # proposal against its opposite; otherwise a high-scoring but reversed
        # contour/endpoint proposal can never be corrected.
        roots = [base, -base]
        for root in roots:
            root_angle = float(np.arctan2(root[1], root[0]))
            for delta in (-np.pi / 8.0, 0.0, np.pi / 8.0):
                angle = root_angle + float(delta)
                directions.append(np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32))
    else:
        directions.append(np.asarray([1.0, 0.0], dtype=np.float32))

    best_direction = base if base_norm > 1e-6 else np.asarray([1.0, 0.0], dtype=np.float32)
    best_geo = -1.0
    seen: List[np.ndarray] = []
    for direction in directions:
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            continue
        unit = direction / norm
        if any(float(np.dot(unit, existing)) > 0.985 for existing in seen):
            continue
        seen.append(unit)
        probe = _candidate_with_direction(candidate, unit, candidate.score)
        geo = _candidate_local_geometry_score(bw, probe, A, B)
        if geo > best_geo:
            best_geo = float(geo)
            best_direction = unit

    combined_score = float(np.clip(0.58 * float(candidate.score) + 0.42 * max(0.0, best_geo), 0.0, 1.0))
    return _candidate_with_direction(candidate, best_direction, combined_score), float(max(0.0, best_geo))


def _interpolate_short_profile_gaps(profile: np.ndarray, max_gap: int = 3) -> np.ndarray:
    result = np.asarray(profile, dtype=np.float32).copy()
    nonzero = np.flatnonzero(result > 0)
    for left, right in zip(nonzero[:-1], nonzero[1:]):
        gap = int(right - left - 1)
        if 0 < gap <= int(max_gap):
            result[left : right + 1] = np.linspace(result[left], result[right], right - left + 1)
    return result


def _arrow_width_peak_evidence(
    bw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, List[Tuple[float, int, int]]]:
    binary = (bw > 0).astype(np.uint8)
    dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    skel = skeletonize(binary.astype(bool)).astype(np.uint8)
    radii = dist_map[skel > 0]
    radii = radii[radii > 0]
    if radii.size == 0:
        return dist_map, skel, 1.0, []

    stroke_radius = float(max(1.0, np.quantile(radii, 0.20)))
    kernel_radius = max(3, int(round(stroke_radius * 4.0)))
    kernel_size = kernel_radius * 2 + 1
    local_max = cv2.dilate(dist_map, np.ones((kernel_size, kernel_size), dtype=np.uint8))
    peak_mask = (
        (dist_map >= local_max - 1e-5)
        & (dist_map >= max(stroke_radius * 1.65, stroke_radius + 0.75))
    ).astype(np.uint8)

    # A distance maximum need not lie on the thinning skeleton of an asymmetric
    # triangle.  Collapse each regional-maximum plateau to its strongest pixel.
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(peak_mask, connectivity=8)
    raw: List[Tuple[float, int, int]] = []
    for component_id in range(1, int(count)):
        ys, xs = np.nonzero(labels == component_id)
        if xs.size == 0:
            continue
        best_index = int(np.argmax(dist_map[ys, xs]))
        x = int(xs[best_index])
        y = int(ys[best_index])
        raw.append((float(dist_map[y, x]), x, y))
    raw.sort(reverse=True)
    peaks: List[Tuple[float, int, int]] = []
    for radius, x, y in raw:
        if any(
            (x - existing_x) ** 2 + (y - existing_y) ** 2
            <= max(stroke_radius * 2.5, min(radius, existing_radius) * 1.35) ** 2
            for existing_radius, existing_x, existing_y in peaks
        ):
            continue
        peaks.append((radius, x, y))
    return dist_map, skel, stroke_radius, peaks


def _eroded_arrow_core_evidence(
    bw: np.ndarray,
    stroke_radius: float,
) -> List[ArrowCoreEvidence]:
    binary = (bw > 0).astype(np.uint8)
    kernel_radius = max(1, int(np.ceil(float(stroke_radius) * 1.05)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_radius * 2 + 1, kernel_radius * 2 + 1),
    )
    core = cv2.erode(binary, kernel, iterations=1)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(core, connectivity=8)
    min_area = int(max(8, round(stroke_radius * stroke_radius * 1.20)))
    max_area = int(max(min_area + 1, round(stroke_radius * stroke_radius * 40.0)))
    evidence: List[ArrowCoreEvidence] = []
    for component_id in range(1, int(count)):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        ys, xs = np.nonzero(labels == component_id)
        if xs.size < 4:
            continue
        points = np.stack((xs, ys), axis=1).astype(np.float64)
        center = np.mean(points, axis=0)
        centered = points - center
        covariance = (centered.T @ centered) / max(1, points.shape[0])
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        minor = float(np.min(eigenvalues))
        major = float(np.max(eigenvalues))
        if minor <= 1e-6:
            continue
        axis_ratio = major / minor
        if axis_ratio < 1.25 or axis_ratio > 8.0:
            continue
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        projection = centered @ axis
        negative_extent = abs(float(np.min(projection)))
        positive_extent = float(np.max(projection))
        total_extent = positive_extent + negative_extent
        if total_extent <= 1e-6:
            continue
        asymmetry = abs(positive_extent - negative_extent) / total_extent
        if asymmetry < 0.10:
            continue
        sign = 1.0 if positive_extent >= negative_extent else -1.0
        direction = axis * sign
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        apex_extent = max(positive_extent, negative_extent)
        tip = center + direction * apex_extent
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        evidence.append(
            ArrowCoreEvidence(
                center=(float(center[0]), float(center[1])),
                tip=(float(tip[0]), float(tip[1])),
                direction=(float(direction[0]), float(direction[1])),
                asymmetry=float(asymmetry),
                axis_ratio=float(axis_ratio),
                area=int(area),
                bbox=(x, y, x + width, y + height),
            )
        )
    return evidence


def _width_profile_direction_at_peak(
    dist_map: np.ndarray,
    skel: np.ndarray,
    peak_x: int,
    peak_y: int,
    stroke_radius: float,
) -> Tuple[np.ndarray, float, float, float, float, np.ndarray] | None:
    del skel
    H, W = dist_map.shape[:2]
    if not (0 <= int(peak_x) < W and 0 <= int(peak_y) < H):
        return None
    peak_radius = float(dist_map[int(peak_y), int(peak_x)])
    if peak_radius < max(float(stroke_radius) * 1.20, float(stroke_radius) + 0.35):
        return None
    binary = (dist_map > 0).astype(np.uint8)

    def sample_binary(points: np.ndarray) -> np.ndarray:
        xs = np.rint(points[..., 0]).astype(np.int32)
        ys = np.rint(points[..., 1]).astype(np.int32)
        valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        values = np.zeros(xs.shape, dtype=np.uint8)
        values[valid] = binary[ys[valid], xs[valid]]
        return values

    center = np.asarray([float(peak_x), float(peak_y)], dtype=np.float32)
    axis_start = float(max(peak_radius * 1.15, stroke_radius * 2.5))
    axis_end = float(max(axis_start + 4.0, peak_radius * 4.0, stroke_radius * 7.0))
    axis_offsets = np.linspace(axis_start, axis_end, 18, dtype=np.float32)
    lateral_span = float(max(2.0, stroke_radius * 1.3))
    lateral_offsets = np.linspace(-lateral_span, lateral_span, 7, dtype=np.float32)

    best_axis: np.ndarray | None = None
    best_axis_score = -1.0
    best_side_support = (0.0, 0.0)
    for angle_index in range(36):
        angle = float(np.pi * angle_index / 36.0)
        unit = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
        perpendicular = np.asarray([-unit[1], unit[0]], dtype=np.float32)
        supports: List[float] = []
        for sign in (-1.0, 1.0):
            points = (
                center[None, None, :]
                + sign * axis_offsets[:, None, None] * unit[None, None, :]
                + lateral_offsets[None, :, None] * perpendicular[None, None, :]
            )
            values = sample_binary(points)
            supports.append(float(np.mean(np.max(values, axis=1) > 0)))
        axis_score = min(supports) + 0.30 * max(supports)
        if axis_score > best_axis_score:
            best_axis_score = float(axis_score)
            best_axis = unit
            best_side_support = (float(supports[0]), float(supports[1]))

    if best_axis is None:
        return None
    axis_support = float(min(best_side_support))
    if axis_support < 0.20 and max(best_side_support) < 0.55:
        return None

    perpendicular = np.asarray([-best_axis[1], best_axis[0]], dtype=np.float32)
    profile_end = float(max(peak_radius * 3.6, stroke_radius * 7.0))
    profile_offsets = np.linspace(max(0.5, stroke_radius * 0.20), profile_end, 40, dtype=np.float32)
    profile_lateral = float(max(peak_radius * 1.8, stroke_radius * 4.0))
    profile_lateral_offsets = np.linspace(-profile_lateral, profile_lateral, 61, dtype=np.float32)
    profiles: List[np.ndarray] = []
    for sign in (-1.0, 1.0):
        points = (
            center[None, None, :]
            + sign * profile_offsets[:, None, None] * best_axis[None, None, :]
            + profile_lateral_offsets[None, :, None] * perpendicular[None, None, :]
        )
        profiles.append(np.sum(sample_binary(points) > 0, axis=1).astype(np.float32))

    negative_profile, positive_profile = profiles
    negative_peak_index = int(np.argmax(negative_profile))
    positive_peak_index = int(np.argmax(positive_profile))
    first_count = min(4, int(negative_profile.size))
    negative_first = float(np.mean(negative_profile[:first_count]))
    positive_first = float(np.mean(positive_profile[:first_count]))
    negative_mass = float(np.sum(negative_profile))
    positive_mass = float(np.sum(positive_profile))

    if negative_peak_index != positive_peak_index:
        direction_sign = -1.0 if negative_peak_index < positive_peak_index else 1.0
    elif abs(negative_first - positive_first) > 1e-6:
        direction_sign = -1.0 if negative_first < positive_first else 1.0
    else:
        direction_sign = -1.0 if negative_mass < positive_mass else 1.0
    direction = best_axis * float(direction_sign)
    chosen = negative_profile if direction_sign < 0 else positive_profile
    other = positive_profile if direction_sign < 0 else negative_profile

    peak_index_gap = abs(negative_peak_index - positive_peak_index) / max(1.0, float(chosen.size - 1))
    first_gap = abs(negative_first - positive_first) / max(negative_first + positive_first, 1.0)
    mass_gap = abs(negative_mass - positive_mass) / max(negative_mass + positive_mass, 1.0)
    asymmetry = float(
        np.clip(2.8 * peak_index_gap + 1.8 * first_gap + 1.2 * mass_gap, 0.0, 1.0)
    )

    far_values = np.concatenate((negative_profile[-8:], positive_profile[-8:]))
    baseline_width = float(max(1.0, np.median(far_values)))
    local_peak_width = float(max(np.max(negative_profile[:12]), np.max(positive_profile[:12])))
    profile_ratio = (local_peak_width - baseline_width) / baseline_width
    distance_ratio = (peak_radius - float(stroke_radius)) / max(float(stroke_radius), 0.5)
    radius_ratio = float(max(profile_ratio, distance_ratio))
    differences = np.diff(chosen[:20])
    monotonicity = float(np.mean(differences <= 0.0)) if differences.size else 0.0
    score = float(
        np.clip(
            0.34 * min(1.0, best_axis_score / 1.30)
            + 0.26 * np.clip(radius_ratio / 1.20, 0.0, 1.0)
            + 0.22 * asymmetry
            + 0.18 * monotonicity,
            0.0,
            1.0,
        )
    )

    chosen_baseline = float(max(1.0, np.median(chosen[-8:])))
    chosen_peak = float(max(np.max(chosen[:12]), chosen_baseline))
    taper_threshold = chosen_baseline + 0.18 * (chosen_peak - chosen_baseline)
    taper_index = 0
    for index in range(1, max(1, int(chosen.size) - 1)):
        if chosen[index] <= taper_threshold and chosen[index + 1] <= taper_threshold:
            taper_index = index
            break
    if taper_index <= 0:
        taper_distance = float(min(profile_end * 0.72, peak_radius * 2.2))
    else:
        taper_distance = float(profile_offsets[taper_index])
    tip_offset = direction * taper_distance
    return direction, score, axis_support, radius_ratio, asymmetry, tip_offset.astype(np.float32)


def _refine_candidates_by_width_profile(
    bw: np.ndarray,
    candidates: Sequence[ArrowCandidate],
) -> List[ArrowCandidate]:
    dist_map, skel, stroke_radius, peaks = _arrow_width_peak_evidence(bw)
    if not peaks:
        return list(candidates)

    evidence_cache: Dict[
        Tuple[int, int],
        Tuple[np.ndarray, float, float, float, float, np.ndarray] | None,
    ] = {}
    peak_radius_by_key = {(int(x), int(y)): float(radius) for radius, x, y in peaks}

    def is_isolated_peak(key: Tuple[int, int]) -> bool:
        radius = peak_radius_by_key.get(key)
        if radius is None:
            return False
        nearest = min(
            (
                float(np.hypot(key[0] - other_key[0], key[1] - other_key[1]))
                for other_key in peak_radius_by_key
                if other_key != key
            ),
            default=float("inf"),
        )
        return nearest >= max(float(stroke_radius) * 7.0, float(radius) * 3.0)

    refined: List[ArrowCandidate] = []
    for candidate in candidates:
        bbox_width = max(1, int(candidate.bbox[2]) - int(candidate.bbox[0]))
        bbox_height = max(1, int(candidate.bbox[3]) - int(candidate.bbox[1]))
        match_radius = float(max(stroke_radius * 8.0, np.hypot(bbox_width, bbox_height) * 0.72))
        nearby = sorted(
            [
                (float((x - candidate.tip[0]) ** 2 + (y - candidate.tip[1]) ** 2), x, y, False)
                for _radius, x, y in peaks
                if (x - candidate.tip[0]) ** 2 + (y - candidate.tip[1]) ** 2 <= match_radius ** 2
            ],
            key=lambda item: item[0],
        )[:3]
        nearby.insert(
            0,
            (
                0.0,
                int(round(candidate.tip[0])),
                int(round(candidate.tip[1])),
                True,
            ),
        )
        best_evidence: Tuple[np.ndarray, float, float, float, float, np.ndarray] | None = None
        best_quality = -1.0
        best_evidence_isolated = False
        has_close_width_peak = any(
            not item[3]
            and item[0] <= (stroke_radius * 4.5) ** 2
            for item in nearby
        )
        for distance_sq, x, y, is_direct in nearby:
            key = (int(x), int(y))
            if key not in evidence_cache:
                evidence_cache[key] = _width_profile_direction_at_peak(
                    dist_map,
                    skel,
                    int(x),
                    int(y),
                    stroke_radius,
                )
            evidence = evidence_cache[key]
            if evidence is None:
                continue
            quality = (
                float(evidence[1])
                - 0.12 * float(np.sqrt(distance_sq) / max(match_radius, 1.0))
                - (0.15 if is_direct and has_close_width_peak else 0.0)
            )
            if quality > best_quality:
                best_quality = quality
                best_evidence = evidence
                best_evidence_isolated = bool(not is_direct and is_isolated_peak(key))

        if best_evidence is None:
            refined.append(candidate)
            continue
        direction, score, axis_support, radius_ratio, asymmetry, _tip_offset = best_evidence
        if score < 0.62 or axis_support < 0.45 or radius_ratio < 0.30:
            refined.append(candidate)
            continue

        original = np.asarray(candidate.direction, dtype=np.float32)
        original_norm = float(np.linalg.norm(original))
        if original_norm > 1e-6:
            original /= original_norm
        direction = np.asarray(direction, dtype=np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        signed_alignment = float(np.dot(original, direction)) if original_norm > 1e-6 else 1.0
        unsigned_alignment = abs(float(np.dot(original, direction))) if original_norm > 1e-6 else 1.0
        strong_sign_evidence = bool(
            score >= 0.70
            and axis_support >= 0.65
            and radius_ratio >= 0.60
            and asymmetry >= 0.18
        )
        if not best_evidence_isolated and signed_alignment < 0.80:
            refined.append(candidate)
            continue
        if signed_alignment < 0.0 and (
            not strong_sign_evidence or not best_evidence_isolated
        ):
            refined.append(candidate)
            continue
        if unsigned_alignment < 0.35 and not (
            score >= 0.70
            and axis_support >= 0.60
            and asymmetry >= 0.25
            and best_evidence_isolated
        ):
            refined.append(candidate)
            continue

        # Weak taper asymmetry determines only the shaft axis, not its sign.
        if asymmetry < 0.18 and original_norm > 1e-6 and float(np.dot(original, direction)) < 0.0:
            direction = -direction
        combined_score = float(np.clip(0.85 * candidate.score + 0.15 * score, 0.0, 1.0))
        refined.append(_candidate_with_direction(candidate, direction, combined_score))

    component_labels = _label_binary_components(bw)
    recovered: List[ArrowCandidate] = []
    for peak_radius, peak_x, peak_y in peaks:
        key = (int(peak_x), int(peak_y))
        if key not in evidence_cache:
            evidence_cache[key] = _width_profile_direction_at_peak(
                dist_map,
                skel,
                int(peak_x),
                int(peak_y),
                stroke_radius,
            )
        evidence = evidence_cache[key]
        if evidence is None:
            continue
        direction, score, axis_support, radius_ratio, asymmetry, tip_offset = evidence
        if (
            score < 0.72
            or axis_support < 0.65
            or radius_ratio < 0.60
            or asymmetry < 0.18
        ):
            continue
        tip_x = float(peak_x + float(tip_offset[0]))
        tip_y = float(peak_y + float(tip_offset[1]))
        duplicate_radius = float(max(stroke_radius * 6.0, peak_radius * 2.2))
        if any(
            (tip_x - candidate.tip[0]) ** 2 + (tip_y - candidate.tip[1]) ** 2 <= duplicate_radius ** 2
            for candidate in refined
        ):
            continue
        component_id = _component_id_near_tip(
            component_labels,
            float(peak_x),
            float(peak_y),
            radius=max(3, int(round(stroke_radius * 2.0))),
        )
        if component_id <= 0:
            continue
        margin = int(max(3, round(stroke_radius * 2.0), round(peak_radius * 1.75)))
        x1 = max(0, int(np.floor(min(float(peak_x), tip_x))) - margin)
        y1 = max(0, int(np.floor(min(float(peak_y), tip_y))) - margin)
        x2 = min(bw.shape[1], int(np.ceil(max(float(peak_x), tip_x))) + margin + 1)
        y2 = min(bw.shape[0], int(np.ceil(max(float(peak_y), tip_y))) + margin + 1)
        recovered.append(
            ArrowCandidate(
                bbox=(x1, y1, x2, y2),
                tip=(tip_x, tip_y),
                direction=(float(direction[0]), float(direction[1])),
                score=float(np.clip(score, 0.0, 1.0)),
                area=float(max(1.0, np.pi * peak_radius * peak_radius)),
                component_id=int(component_id),
            )
        )

    merged = [*refined, *recovered]
    core_evidence = _eroded_arrow_core_evidence(bw, stroke_radius)
    assignment_edges: List[Tuple[float, int, int]] = []
    expansion = float(max(4.0, stroke_radius * 3.5))
    for candidate_index, candidate in enumerate(merged):
        if float(candidate.score) >= 0.80:
            continue
        candidate_direction = np.asarray(candidate.direction, dtype=np.float32)
        candidate_norm = float(np.linalg.norm(candidate_direction))
        if candidate_norm <= 1e-6:
            continue
        candidate_direction /= candidate_norm
        x1, y1, x2, y2 = (float(value) for value in candidate.bbox)
        for evidence_index, item in enumerate(core_evidence):
            core_direction = np.asarray(item.direction, dtype=np.float32)
            alignment = float(np.dot(candidate_direction, core_direction))
            if alignment < 0.70:
                continue
            points = (item.center, item.tip)
            if not any(
                x1 - expansion <= point[0] <= x2 + expansion
                and y1 - expansion <= point[1] <= y2 + expansion
                for point in points
            ):
                continue
            distance = min(
                float(np.hypot(candidate.tip[0] - point[0], candidate.tip[1] - point[1]))
                for point in points
            )
            cost = (
                distance / max(expansion * 3.0, 1.0)
                + 0.35 * (1.0 - alignment)
                - 0.45 * float(item.asymmetry)
            )
            assignment_edges.append((float(cost), int(candidate_index), int(evidence_index)))

    assigned_candidates: set[int] = set()
    assigned_evidence: set[int] = set()
    for _cost, candidate_index, evidence_index in sorted(assignment_edges):
        if candidate_index in assigned_candidates or evidence_index in assigned_evidence:
            continue
        candidate = merged[candidate_index]
        item = core_evidence[evidence_index]
        core_direction = np.asarray(item.direction, dtype=np.float32)
        core_score = float(np.clip(0.62 + 1.5 * item.asymmetry, 0.0, 0.95))
        combined_score = float(np.clip(0.88 * candidate.score + 0.12 * core_score, 0.0, 1.0))
        merged[candidate_index] = _candidate_with_direction(candidate, core_direction, combined_score)
        assigned_candidates.add(candidate_index)
        assigned_evidence.add(evidence_index)

    merged.sort(key=lambda item: item.score, reverse=True)
    filtered: List[ArrowCandidate] = []
    margin = int(max(3, round(stroke_radius * 6.0)))
    for candidate in merged:
        if any(
            candidate.component_id == existing.component_id
            and _candidate_similarity(candidate, existing, margin)
            for existing in filtered
        ):
            continue
        filtered.append(candidate)
    return filtered


def _candidate_similarity(a: ArrowCandidate, b: ArrowCandidate, margin: int) -> bool:
    tip_gap_sq = (a.tip[0] - b.tip[0]) ** 2 + (a.tip[1] - b.tip[1]) ** 2
    near_tip = tip_gap_sq <= float(max(3, margin) ** 2)
    overlap = _bbox_iou(a.bbox, b.bbox)
    direction_dot = _candidate_direction_dot(a, b)
    if direction_dot < -0.55:
        return False
    same_direction = direction_dot > 0.70
    return bool((near_tip and (same_direction or overlap > 0.18)) or overlap > 0.52)


def _component_id_near_tip(component_labels: np.ndarray, tip_x: float, tip_y: float, radius: int = 4) -> int:
    H, W = component_labels.shape[:2]
    x = int(round(float(tip_x)))
    y = int(round(float(tip_y)))
    if 0 <= x < W and 0 <= y < H and int(component_labels[y, x]) > 0:
        return int(component_labels[y, x])
    x1 = max(0, x - int(radius))
    y1 = max(0, y - int(radius))
    x2 = min(W, x + int(radius) + 1)
    y2 = min(H, y + int(radius) + 1)
    crop = component_labels[y1:y2, x1:x2]
    ids, counts = np.unique(crop[crop > 0], return_counts=True)
    if ids.size == 0:
        return 0
    return int(ids[int(np.argmax(counts))])


def _draw_arrow_template(unit: np.ndarray, cell_span: float, scale: float) -> Tuple[np.ndarray, Tuple[float, float], float]:
    head_len = float(max(7.0, cell_span * 0.92 * scale))
    head_width = float(max(6.0, head_len * 0.86))
    shaft_len = float(max(5.0, cell_span * 0.58 * scale))
    shaft_width = int(max(2, round(cell_span * 0.10 * scale)))
    size = int(max(21, round((head_len + shaft_len + head_width) * 2.25)))
    if size % 2 == 0:
        size += 1
    canvas = np.zeros((size, size), dtype=np.uint8)
    center = np.asarray([size / 2.0, size / 2.0], dtype=np.float32)
    tip = center + unit.astype(np.float32) * float(head_len * 0.48)
    base = tip - unit.astype(np.float32) * head_len
    perp = np.asarray([-unit[1], unit[0]], dtype=np.float32)
    left = base + perp * (head_width * 0.50)
    right = base - perp * (head_width * 0.50)
    tail = base - unit.astype(np.float32) * shaft_len
    polygon = np.asarray([tip, left, right], dtype=np.int32)
    cv2.fillConvexPoly(canvas, polygon, 255)
    cv2.line(
        canvas,
        (int(round(tail[0])), int(round(tail[1]))),
        (int(round(base[0])), int(round(base[1]))),
        255,
        thickness=shaft_width,
        lineType=cv2.LINE_AA,
    )
    return canvas, (float(tip[0]), float(tip[1])), float(np.count_nonzero(canvas > 0))


def _template_arrowhead_candidates(
    bw: np.ndarray,
    A: int,
    B: int,
    component_labels: np.ndarray,
    margin: int,
    head_score_thr: float,
) -> List[ArrowCandidate]:
    """Recover arrowheads inside one large connected component using relative templates."""
    H, W = bw.shape[:2]
    cell_span = float(max(H / float(A), W / float(B), 1.0))
    image = (bw > 0).astype(np.float32)
    raw: List[ArrowCandidate] = []
    angle_count = 16
    scales = (0.78, 1.02, 1.28)
    for scale in scales:
        for angle_index in range(angle_count):
            angle = float(2.0 * np.pi * angle_index / angle_count)
            unit = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
            template, tip_offset, template_area = _draw_arrow_template(unit, cell_span, scale)
            th, tw = template.shape[:2]
            if th >= H or tw >= W or template_area <= 0:
                continue
            templ = (template > 0).astype(np.float32)
            response = cv2.matchTemplate(image, templ, cv2.TM_CCORR_NORMED)
            if response.size == 0:
                continue
            threshold = max(0.53, float(head_score_thr) + 0.38)
            local_max = cv2.dilate(response, np.ones((5, 5), dtype=np.float32))
            ys, xs = np.nonzero((response >= threshold) & (response >= local_max - 1e-6))
            if ys.size == 0:
                continue
            order = np.argsort(response[ys, xs])[::-1][:80]
            for order_idx in order:
                y = int(ys[int(order_idx)])
                x = int(xs[int(order_idx)])
                match_score = float(response[y, x])
                tip_x = float(x + tip_offset[0])
                tip_y = float(y + tip_offset[1])
                component_id = _component_id_near_tip(component_labels, tip_x, tip_y, radius=max(3, int(round(cell_span * 0.24))))
                if component_id <= 0:
                    continue
                radius = int(max(3, round(float(margin) * 1.35)))
                candidate = ArrowCandidate(
                    bbox=(
                        max(0, int(round(tip_x)) - radius),
                        max(0, int(round(tip_y)) - radius),
                        min(W, int(round(tip_x)) + radius + 1),
                        min(H, int(round(tip_y)) + radius + 1),
                    ),
                    tip=(tip_x, tip_y),
                    direction=(float(unit[0]), float(unit[1])),
                    score=float(np.clip(0.34 + 0.66 * match_score, 0.0, 1.0)),
                    area=float(template_area),
                    component_id=int(component_id),
                )
                refined, geometry_score = _refine_candidate_direction_by_geometry(bw, candidate, A, B)
                if geometry_score < 0.24:
                    continue
                if _candidate_touches_image_border(refined, H, W, margin=1):
                    continue
                if _candidate_front_ink_ratio(bw, refined, A, B) > 0.58:
                    continue
                raw.append(refined)

    raw.sort(key=lambda item: item.score, reverse=True)
    filtered: List[ArrowCandidate] = []
    max_candidates = max(16, int(np.sqrt(max(H * W, 1)) // 10))
    for candidate in raw:
        if any(candidate.component_id == existing.component_id and _candidate_similarity(candidate, existing, margin) for existing in filtered):
            continue
        if any(
            (candidate.tip[0] - existing.tip[0]) ** 2 + (candidate.tip[1] - existing.tip[1]) ** 2
            <= float((cell_span * 0.58) ** 2)
            and _candidate_direction_dot(candidate, existing) > 0.20
            for existing in filtered
        ):
            continue
        filtered.append(candidate)
        if len(filtered) >= max_candidates:
            break
    return filtered


def detect_arrow_candidates(
    bw: np.ndarray,
    A: int,
    B: int,
    component_labels: np.ndarray | None = None,
    head_score_thr: float = 0.15,
) -> List[ArrowCandidate]:
    """Detect arrow candidates from connected binary objects."""
    H, W = bw.shape[:2]
    cell_area = _cell_area(H, W, A, B)
    if component_labels is None:
        component_labels = _label_binary_components(bw)

    skel = skeletonize((bw > 0).astype(bool)).astype(np.uint8)
    adj = _build_skeleton_adjacency(skel)
    dist_map = cv2.distanceTransform((bw > 0).astype(np.uint8), cv2.DIST_L2, 3)
    component_nodes: DefaultDict[int, List[Pixel]] = defaultdict(list)
    for node in adj:
        component_id = int(component_labels[node[0], node[1]])
        if component_id > 0:
            component_nodes[component_id].append(node)

    def endpoint_probe_path(component_set: set[Pixel], start: Pixel, max_len: int = 12) -> List[Pixel]:
        path = [start]
        prev: Pixel | None = None
        current = start
        for _ in range(max_len - 1):
            neighbors = [nb for nb in adj.get(current, []) if nb in component_set and nb != prev]
            if len(neighbors) != 1:
                break
            nxt = neighbors[0]
            path.append(nxt)
            prev, current = current, nxt
            if len(adj.get(current, [])) != 2 and len(path) >= 4:
                break
        return path

    def endpoint_to_branch_path(component_set: set[Pixel], start: Pixel, max_len: int) -> Tuple[List[Pixel], bool]:
        path = [start]
        prev: Pixel | None = None
        current = start
        reached_branch = False
        for _ in range(max_len - 1):
            neighbors = [nb for nb in adj.get(current, []) if nb in component_set and nb != prev]
            if len(neighbors) != 1:
                break
            nxt = neighbors[0]
            path.append(nxt)
            prev, current = current, nxt
            if len(adj.get(current, [])) > 2:
                reached_branch = True
                break
        return path, reached_branch

    def merged_head_candidate(component_id: int, path: Sequence[Pixel], margin: int) -> ArrowCandidate | None:
        if len(path) < 4:
            return None
        tail = path[0]
        head = path[-1]
        vec = np.array([head[1] - tail[1], head[0] - tail[0]], dtype=np.float32)
        length = float(np.linalg.norm(vec))
        min_length = float(max(H / float(A), W / float(B)) * 1.8)
        if length < max(18.0, min_length):
            return None
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-6:
            return None
        direction = vec / norm
        head_probe = list(path[max(0, len(path) - 8):])
        xs = [int(p[1]) for p in head_probe]
        ys = [int(p[0]) for p in head_probe]
        head_margin = int(max(2, round(float(margin) * 1.65)))
        length_score = float(np.clip(length / max(4.0 * max(H / float(A), W / float(B)), 1.0), 0.0, 1.0))
        score = float(0.34 + 0.36 * length_score)
        return ArrowCandidate(
            bbox=(
                max(0, min(xs) - head_margin),
                max(0, min(ys) - head_margin),
                min(W, max(xs) + head_margin + 1),
                min(H, max(ys) + head_margin + 1),
            ),
            tip=(float(head[1]), float(head[0])),
            direction=(float(direction[0]), float(direction[1])),
            score=score,
            area=float(max(1.0, length)),
            component_id=int(component_id),
        )

    def endpoint_candidate(component_id: int, path: Sequence[Pixel], margin: int) -> ArrowCandidate | None:
        if len(path) < 3:
            return None
        widths = [float(dist_map[y, x]) for y, x in path]
        tip_width = widths[0]
        inner_width = float(np.mean(widths[1:]))
        max_width = float(np.max(widths[1:]))
        if inner_width <= 0:
            return None

        taper_score = float(np.clip((inner_width - tip_width) / max(inner_width, 0.75), 0.0, 1.0))
        growth_score = float(np.clip((max_width - tip_width) / 2.5, 0.0, 1.0))
        length_score = float(np.clip((len(path) - 2) / 5.0, 0.0, 1.0))
        score = 0.16 + 0.46 * taper_score + 0.26 * growth_score + 0.12 * length_score
        if score < max(0.12, head_score_thr * 0.78):
            return None

        anchor = path[min(len(path) - 1, 3)]
        direction = np.array([path[0][1] - anchor[1], path[0][0] - anchor[0]], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            return None
        direction /= norm

        probe = list(path[: min(len(path), 6)])
        xs = [int(p[1]) for p in probe]
        ys = [int(p[0]) for p in probe]
        bbox = (
            max(0, min(xs) - margin),
            max(0, min(ys) - margin),
            min(W, max(xs) + margin + 1),
            min(H, max(ys) + margin + 1),
        )
        return ArrowCandidate(
            bbox=bbox,
            tip=(float(path[0][1]), float(path[0][0])),
            direction=(float(direction[0]), float(direction[1])),
            score=float(score),
            area=float(max(1.0, inner_width * len(path))),
            component_id=component_id,
        )

    def endpoint_cluster_candidates(
        raw_candidates: Sequence[ArrowCandidate],
        existing_candidates: Sequence[ArrowCandidate],
        max_keep: int,
    ) -> List[ArrowCandidate]:
        """Recover embedded arrow heads from endpoint groups in merged components."""
        if not raw_candidates or max_keep <= 0:
            return []

        cell_span = float(max(H / float(A), W / float(B), 1.0))
        cluster_gap = float(max(3.0, cell_span * 0.82))
        cluster_gap_sq = cluster_gap * cluster_gap
        min_cluster_score = float(max(0.58, head_score_thr * 3.6))
        items = [candidate for candidate in raw_candidates if float(candidate.score) >= min_cluster_score]
        if not items:
            return []

        used = [False for _ in items]
        clusters: List[List[ArrowCandidate]] = []
        for idx, candidate in enumerate(items):
            if used[idx]:
                continue
            queue = [idx]
            used[idx] = True
            cluster: List[ArrowCandidate] = []
            while queue:
                cur_idx = queue.pop()
                current = items[cur_idx]
                cluster.append(current)
                for other_idx, other in enumerate(items):
                    if used[other_idx]:
                        continue
                    dist_sq = (current.tip[0] - other.tip[0]) ** 2 + (current.tip[1] - other.tip[1]) ** 2
                    if dist_sq <= cluster_gap_sq:
                        used[other_idx] = True
                        queue.append(other_idx)
            clusters.append(cluster)

        recovered: List[ArrowCandidate] = []

        def unit_direction(candidate: ArrowCandidate) -> np.ndarray | None:
            vec = np.asarray(candidate.direction, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm <= 1e-6:
                return None
            return vec / norm

        def local_contour_tip(cluster_items: Sequence[ArrowCandidate]) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]] | None:
            tip_xs = [float(item.tip[0]) for item in cluster_items]
            tip_ys = [float(item.tip[1]) for item in cluster_items]
            crop_margin = int(max(3, round(cell_span * 0.85)))
            x1 = int(max(0, min(tip_xs) - crop_margin))
            y1 = int(max(0, min(tip_ys) - crop_margin))
            x2 = int(min(W, max(tip_xs) + crop_margin + 1))
            y2 = int(min(H, max(tip_ys) + crop_margin + 1))
            if x2 <= x1 + 2 or y2 <= y1 + 2:
                return None
            crop = (bw[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            contour = max(contours, key=cv2.contourArea)
            if float(cv2.contourArea(contour)) < max(8.0, cell_area * 0.04):
                return None
            tip, direction = _tip_from_contour(contour)
            axis = np.asarray(direction, dtype=np.float32)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1e-6:
                return None
            tip_global = np.asarray([float(tip[0] + x1), float(tip[1] + y1)], dtype=np.float32)
            return tip_global, axis / axis_norm, (x1, y1, x2, y2)

        for cluster in clusters:
            cluster = sorted(cluster, key=lambda item: item.score, reverse=True)
            weighted = np.zeros(2, dtype=np.float32)
            total_weight = 0.0
            usable: List[Tuple[ArrowCandidate, np.ndarray]] = []
            for candidate in cluster:
                direction = unit_direction(candidate)
                if direction is None:
                    continue
                weight = float(max(candidate.score, 0.05))
                weighted += direction * weight
                total_weight += weight
                usable.append((candidate, direction))

            if not usable or total_weight <= 1e-6:
                continue

            avg_norm = float(np.linalg.norm(weighted) / max(total_weight, 1e-6))
            axis: np.ndarray | None = None
            aligned: List[ArrowCandidate] = []
            contour_result = local_contour_tip(cluster)
            contour_bbox: Tuple[int, int, int, int] | None = None
            contour_tip: np.ndarray | None = None
            if contour_result is not None:
                contour_tip, axis, contour_bbox = contour_result
                aligned = [candidate for candidate, _direction in usable]

            if not aligned and len(usable) >= 2 and avg_norm >= 0.30:
                axis = weighted / max(float(np.linalg.norm(weighted)), 1e-6)
                aligned = [
                    candidate
                    for candidate, direction in usable
                    if float(np.dot(direction, axis)) >= 0.10
                ]
                if len(aligned) < 2:
                    aligned = []

            if not aligned:
                strongest = usable[0][0]
                best_score = float(strongest.score)
                best_endpoint_score = float(items[0].score)
                if best_score < max(0.80, best_endpoint_score * 0.985):
                    continue
                axis = usable[0][1]
                aligned = [strongest]

            if axis is None:
                continue

            if contour_tip is not None:
                tip_xy = contour_tip
                tip_component = int(aligned[0].component_id)
            else:
                pts = np.asarray([[float(item.tip[0]), float(item.tip[1])] for item in aligned], dtype=np.float32)
                center = np.mean(pts, axis=0, dtype=np.float32)
                tip_candidate = max(
                    aligned,
                    key=lambda item: float(
                        np.dot(np.asarray([item.tip[0], item.tip[1]], dtype=np.float32) - center, axis)
                    )
                    + 4.0 * float(item.score),
                )
                tip_xy = np.asarray([float(tip_candidate.tip[0]), float(tip_candidate.tip[1])], dtype=np.float32)
                tip_component = int(tip_candidate.component_id)

            if contour_bbox is not None:
                x1, y1, x2, y2 = contour_bbox
            else:
                x1 = int(max(0, min(item.bbox[0] for item in aligned)))
                y1 = int(max(0, min(item.bbox[1] for item in aligned)))
                x2 = int(min(W, max(item.bbox[2] for item in aligned)))
                y2 = int(min(H, max(item.bbox[3] for item in aligned)))
            score = float(np.clip(max(item.score for item in aligned) * 0.72 + avg_norm * 0.28, 0.05, 0.92))
            recovered.append(
                ArrowCandidate(
                    bbox=(x1, y1, x2, y2),
                    tip=(float(tip_xy[0]), float(tip_xy[1])),
                    direction=(float(axis[0]), float(axis[1])),
                    score=score,
                    area=float(sum(max(1.0, item.area) for item in aligned)),
                    component_id=int(tip_component),
                )
            )

        recovered.sort(key=lambda item: item.score, reverse=True)
        filtered: List[ArrowCandidate] = []
        dedupe_gap_sq = float((cell_span * 0.72) ** 2)
        for candidate in recovered:
            near_existing = any(
                (candidate.tip[0] - existing.tip[0]) ** 2 + (candidate.tip[1] - existing.tip[1]) ** 2 <= dedupe_gap_sq
                and _candidate_direction_dot(candidate, existing) > 0.45
                for existing in [*existing_candidates, *filtered]
            )
            if near_existing:
                continue
            filtered.append(candidate)
            if len(filtered) >= max_keep:
                break
        return filtered

    def branch_head_candidates(
        component_id: int,
        component_nodes: Sequence[Pixel],
        endpoints: Sequence[Pixel],
        branch_nodes: Sequence[Pixel],
        margin: int,
    ) -> List[ArrowCandidate]:
        """Recover heads whose actual tips are merged at a skeleton junction."""
        if not endpoints or not branch_nodes:
            return []
        cell_span = float(max(H / float(A), W / float(B), 1.0))
        min_length = float(cell_span * 2.4)
        max_branch_count = 36
        cluster_gap = float(max(2.0, cell_span * 0.42))
        cluster_gap_sq = cluster_gap * cluster_gap

        branch_order = sorted(
            branch_nodes,
            key=lambda node: (len(adj.get(node, [])), float(dist_map[node[0], node[1]])),
            reverse=True,
        )
        cluster_heads: List[Pixel] = []
        for node in branch_order:
            if any((node[0] - head[0]) ** 2 + (node[1] - head[1]) ** 2 <= cluster_gap_sq for head in cluster_heads):
                continue
            cluster_heads.append(node)
            if len(cluster_heads) >= max_branch_count:
                break

        recovered: List[ArrowCandidate] = []
        component_list = list(component_nodes)
        for head_node in cluster_heads:
            dist, parent = _bfs_within_component(adj, component_list, head_node)
            options: List[Tuple[float, ArrowCandidate]] = []
            for endpoint in endpoints:
                if endpoint not in dist:
                    continue
                length = float(dist[endpoint])
                if length < min_length:
                    continue
                tail_to_head = _path_from_parent(parent, endpoint)
                if len(tail_to_head) < 4:
                    continue
                direction = np.asarray(
                    [float(head_node[1] - endpoint[1]), float(head_node[0] - endpoint[0])],
                    dtype=np.float32,
                )
                norm = float(np.linalg.norm(direction))
                if norm <= 1e-6:
                    continue
                direction /= norm
                align = _path_alignment_near_head(tail_to_head, (float(direction[0]), float(direction[1])), probe_len=7)
                if align < 0.55:
                    continue
                if len(endpoints) <= 3 and branch_count <= 6:
                    endpoint_dirs: List[np.ndarray] = []
                    for other in endpoints:
                        if other == endpoint:
                            continue
                        other_vec = np.asarray(
                            [float(head_node[1] - other[1]), float(head_node[0] - other[0])],
                            dtype=np.float32,
                        )
                        other_norm = float(np.linalg.norm(other_vec))
                        if other_norm <= 1e-6:
                            continue
                        endpoint_dirs.append(other_vec / other_norm)
                    if endpoint_dirs and all(abs(float(np.dot(direction, other_dir))) < 0.35 for other_dir in endpoint_dirs):
                        continue
                tail_head_score = _endpoint_head_score(tail_to_head, dist_map, "start")
                length_score = float(np.clip(length / max(cell_span * 5.0, 1.0), 0.0, 1.0))
                width_score = float(np.clip(float(dist_map[head_node[0], head_node[1]]) / max(cell_span * 0.30, 1.0), 0.0, 1.0))
                score = float(0.34 + 0.28 * max(0.0, align) + 0.22 * length_score + 0.16 * width_score - 0.10 * tail_head_score)
                xs = [int(endpoint[1]), int(head_node[1])]
                ys = [int(endpoint[0]), int(head_node[0])]
                head_margin = int(max(2, round(float(margin) * 1.45)))
                options.append(
                    (
                        score,
                        ArrowCandidate(
                            bbox=(
                                max(0, min(xs) - head_margin),
                                max(0, min(ys) - head_margin),
                                min(W, max(xs) + head_margin + 1),
                                min(H, max(ys) + head_margin + 1),
                            ),
                            tip=(float(head_node[1]), float(head_node[0])),
                            direction=(float(direction[0]), float(direction[1])),
                            score=score,
                            area=float(max(1.0, length)),
                            component_id=int(component_id),
                        ),
                    )
                )
            options.sort(key=lambda item: item[0], reverse=True)
            for _score, candidate in options[:2]:
                recovered.append(candidate)

        recovered.sort(key=lambda item: item.score, reverse=True)
        filtered: List[ArrowCandidate] = []
        for candidate in recovered:
            if any(_candidate_similarity(candidate, existing, margin) for existing in filtered):
                continue
            filtered.append(candidate)
            if len(filtered) >= 8:
                break
        return filtered

    def open_endpoint_fallback_candidates(
        component_id: int,
        endpoints: Sequence[Pixel],
        component_set: set[Pixel],
        margin: int,
        max_keep: int,
    ) -> List[ArrowCandidate]:
        recovered: List[ArrowCandidate] = []
        for endpoint in endpoints:
            path = endpoint_probe_path(component_set, endpoint, max_len=max(6, int(round(max(A, B) * 0.10))))
            candidate = endpoint_candidate(component_id, path, margin=max(2, margin))
            if candidate is None:
                continue
            if float(candidate.score) < 0.30:
                continue
            if _candidate_touches_image_border(candidate, H, W, margin=1):
                continue
            recovered.append(candidate)
        recovered.sort(key=lambda item: item.score, reverse=True)
        filtered: List[ArrowCandidate] = []
        for candidate in recovered:
            if any(_candidate_similarity(candidate, existing, margin) for existing in filtered):
                continue
            filtered.append(candidate)
            if len(filtered) >= max(1, int(max_keep)):
                break
        return filtered

    candidates: List[ArrowCandidate] = []
    margin = int(round(max(H / float(A), W / float(B)) * 0.75))
    template_candidates = _template_arrowhead_candidates(
        bw,
        A,
        B,
        component_labels,
        margin=margin,
        head_score_thr=head_score_thr,
    )
    template_candidates_by_component: DefaultDict[int, List[ArrowCandidate]] = defaultdict(list)
    for candidate in template_candidates:
        template_candidates_by_component[int(candidate.component_id)].append(candidate)
    component_count = int(component_labels.max())
    for component_id in range(1, component_count + 1):
        mask = np.where(component_labels == component_id, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nodes = component_nodes.get(component_id, [])
        if not nodes:
            continue
        component_set = set(nodes)
        endpoints = [node for node in nodes if len(adj.get(node, [])) <= 1]
        branch_count = sum(1 for node in nodes if len(adj.get(node, [])) > 2)
        branch_nodes = [node for node in nodes if len(adj.get(node, [])) > 2]
        simple_component = branch_count == 0 and len(endpoints) <= 2
        simple_arrow_like = len(endpoints) <= 3 and branch_count <= 4
        contour_candidates: List[ArrowCandidate] = []
        merged_head_candidates: List[ArrowCandidate] = []
        endpoint_candidates: List[ArrowCandidate] = []
        branch_candidates: List[ArrowCandidate] = []
        template_component_candidates = template_candidates_by_component.get(component_id, [])

        if contours:
            contour = max(contours, key=cv2.contourArea)
            contour_keep = 1 if simple_arrow_like else min(4, max(2, len(endpoints) // 2))
            if simple_arrow_like:
                contour_candidates.extend(
                    _contour_tip_candidates(
                        contour,
                        cell_area,
                        margin=margin,
                        component_id=component_id,
                        branch_nodes=branch_nodes,
                        H=H,
                        W=W,
                        max_tips=contour_keep,
                    )
                )
            else:
                bisector_keep = min(12, max(4, len(endpoints)))
                if branch_count >= 6:
                    contour_candidates.extend(
                        _contour_bisector_candidates(
                            contour,
                            bw,
                            A,
                            B,
                            margin=margin,
                            component_id=component_id,
                            max_tips=bisector_keep,
                        )
                    )
                if not contour_candidates:
                    contour_candidates.extend(
                        _contour_tip_candidates(
                            contour,
                            cell_area,
                            margin=margin,
                            component_id=component_id,
                            branch_nodes=branch_nodes,
                            H=H,
                            W=W,
                            max_tips=contour_keep,
                        )
                    )
            score = -1.0
            if score >= head_score_thr:
                x, y, w, h = cv2.boundingRect(contour)
                tip, direction = _tip_from_contour(contour)
                bbox = (
                    max(0, x - margin),
                    max(0, y - margin),
                    min(W, x + w + margin),
                    min(H, y + h + margin),
                )
                contour_candidates.append(
                    ArrowCandidate(
                        bbox=bbox,
                        tip=tip,
                        direction=direction,
                        score=score,
                        area=float(cv2.contourArea(contour)),
                        component_id=component_id,
                    )
                )

        # Endpoint-to-branch paths alone are too ambiguous: ordinary grid lines and
        # pipe junctions produce the same skeleton pattern. Keep merged candidates
        # disabled unless a contour-derived head has already provided real evidence.

        endpoint_graph_allowed = (
            branch_count > 0
            and len(endpoints) >= 6
            and branch_count <= max(8, len(endpoints) * 5)
        )
        endpoint_only_allowed = (
            not contour_candidates
            and not merged_head_candidates
            and endpoint_graph_allowed
        )
        if contour_candidates or merged_head_candidates or endpoint_only_allowed:
            for endpoint in endpoints:
                path = endpoint_probe_path(component_set, endpoint, max_len=max(6, int(round(max(A, B) * 0.08))))
                candidate = endpoint_candidate(component_id, path, margin=max(2, margin))
                if candidate is not None:
                    endpoint_candidates.append(candidate)

        if False and branch_count <= 24 and len(endpoints) <= 12:
            branch_candidates.extend(
                branch_head_candidates(
                    component_id,
                    nodes,
                    endpoints,
                    branch_nodes,
                    margin=max(2, margin),
                )
            )

        contour_candidates.sort(key=lambda item: item.score, reverse=True)
        merged_head_candidates.sort(key=lambda item: item.score, reverse=True)
        endpoint_candidates.sort(key=lambda item: item.score, reverse=True)
        branch_candidates.sort(key=lambda item: item.score, reverse=True)

        if branch_candidates:
            branch_gap = float(max(margin * 1.35, max(H / float(A), W / float(B)) * 0.90))
            branch_gap_sq = branch_gap * branch_gap

            def far_from_branch_head(candidate: ArrowCandidate) -> bool:
                return not any(
                    (candidate.tip[0] - branch.tip[0]) ** 2 + (candidate.tip[1] - branch.tip[1]) ** 2 <= branch_gap_sq
                    for branch in branch_candidates
                )

            contour_candidates = [candidate for candidate in contour_candidates if far_from_branch_head(candidate)]
            endpoint_candidates = [candidate for candidate in endpoint_candidates if far_from_branch_head(candidate)]
            merged_head_candidates = [candidate for candidate in merged_head_candidates if far_from_branch_head(candidate)]

        component_candidates: List[ArrowCandidate] = []
        if template_component_candidates and not simple_arrow_like and branch_count >= 6 and len(endpoints) >= 8:
            keep_limit = 1 if simple_arrow_like else min(16, max(6, len(endpoints) // 2))
            component_candidates.extend(template_component_candidates[:keep_limit])
        if contour_candidates and simple_arrow_like:
            # A single arrow object with a visible head should trust the contour-derived
            # head more than the opposite tail endpoint, which often looks deceptively sharp.
            component_candidates.extend(contour_candidates[:1])
        else:
            if branch_candidates:
                component_candidates.extend(branch_candidates[: min(4, len(branch_candidates))])
            if contour_candidates:
                contour_keep = min(len(contour_candidates), 10 if branch_count > 4 else 2)
                component_candidates.extend(contour_candidates[:contour_keep])
            if merged_head_candidates:
                component_candidates.extend(merged_head_candidates[: min(4, len(merged_head_candidates))])
            if endpoint_candidates:
                if (len(contour_candidates) >= 2 or merged_head_candidates) and branch_count > 4:
                    keep_limit = min(max(2, len(endpoints) // 4), 8)
                    clustered_endpoints = endpoint_cluster_candidates(
                        endpoint_candidates,
                        component_candidates,
                        max_keep=keep_limit,
                    )
                    component_candidates.extend(clustered_endpoints)
                    if not clustered_endpoints and endpoint_graph_allowed:
                        best_endpoint = float(endpoint_candidates[0].score)
                        fallback_keep = min(max(2, len(endpoints) // 3), 8)
                        component_candidates.extend(
                            [
                                cand
                                for cand in endpoint_candidates
                                if float(cand.score) >= max(head_score_thr, best_endpoint * 0.72)
                            ][:fallback_keep]
                        )
                elif component_candidates:
                    best_endpoint = float(endpoint_candidates[0].score)
                    keep_limit = 2 if branch_count <= 1 else 3
                    filtered_endpoints = [
                        cand
                        for cand in endpoint_candidates
                        if float(cand.score) >= max(head_score_thr, best_endpoint * 0.78)
                    ]
                    component_candidates.extend(filtered_endpoints[:keep_limit])
                else:
                    keep_limit = 1 if simple_component else min(3, max(2, (len(endpoints) // 2) + (1 if branch_count > 0 else 0)))
                    component_candidates.extend(endpoint_candidates[:keep_limit])

        refined_component_candidates: List[ArrowCandidate] = []
        for candidate in component_candidates:
            refined, geometry_score = _refine_candidate_direction_by_geometry(bw, candidate, A, B)
            min_geometry = 0.20 if simple_arrow_like else 0.28
            if geometry_score < min_geometry:
                continue
            if _candidate_touches_image_border(refined, H, W, margin=1):
                continue
            front_ink_ratio = _candidate_front_ink_ratio(bw, refined, A, B)
            if front_ink_ratio > 0.50 or (front_ink_ratio > 0.42 and float(refined.score) < 0.86):
                continue
            refined_component_candidates.append(refined)

        if not refined_component_candidates and endpoint_graph_allowed and endpoint_candidates:
            for candidate in endpoint_candidates[: min(8, max(2, len(endpoints) // 3))]:
                refined, geometry_score = _refine_candidate_direction_by_geometry(bw, candidate, A, B)
                if geometry_score < 0.20:
                    continue
                if _candidate_touches_image_border(refined, H, W, margin=1):
                    continue
                if _candidate_front_ink_ratio(bw, refined, A, B) > 0.58:
                    continue
                refined_component_candidates.append(refined)

        refined_component_candidates.sort(key=lambda item: item.score, reverse=True)
        if len(refined_component_candidates) >= 3:
            best_component_score = float(refined_component_candidates[0].score)
            score_floor = 0.62 if branch_count >= 6 and len(endpoints) >= 8 else 0.70
            relative_floor = 0.64 if branch_count >= 6 and len(endpoints) >= 8 else 0.78
            refined_component_candidates = [
                candidate
                for candidate in refined_component_candidates
                if float(candidate.score) >= max(score_floor, best_component_score * relative_floor)
            ]
        if branch_count >= 6 and 4 <= len(endpoints) <= 14 and len(refined_component_candidates) <= 4:
            vertical_candidates = [
                candidate
                for candidate in refined_component_candidates
                if _direction_from_vector(candidate.direction) in (2, 4)
            ]
            cell_span_local = float(max(H / float(A), W / float(B), 1.0))
            vertical_pair: Tuple[ArrowCandidate, ArrowCandidate] | None = None
            for idx, first in enumerate(vertical_candidates):
                for second in vertical_candidates[idx + 1:]:
                    if {_direction_from_vector(first.direction), _direction_from_vector(second.direction)} != {2, 4}:
                        continue
                    tip_gap_x = abs(float(first.tip[0]) - float(second.tip[0]))
                    tip_gap_y = abs(float(first.tip[1]) - float(second.tip[1]))
                    if tip_gap_x <= cell_span_local * 1.1 and tip_gap_y <= cell_span_local * 2.8:
                        vertical_pair = (first, second)
                        break
                if vertical_pair is not None:
                    break

            if vertical_pair is not None:
                first, second = vertical_pair
                dist_map_local = cv2.distanceTransform((bw > 0).astype(np.uint8), cv2.DIST_L2, 3)
                branch_order = sorted(
                    branch_nodes,
                    key=lambda node: (len(adj.get(node, [])), float(dist_map_local[node[0], node[1]])),
                    reverse=True,
                )
                if branch_order:
                    head_node = branch_order[0]
                    left_endpoint = min(endpoints, key=lambda node: node[1])
                    right_endpoint = max(endpoints, key=lambda node: node[1])
                    left_len = abs(head_node[1] - left_endpoint[1])
                    right_len = abs(right_endpoint[1] - head_node[1])
                    if left_len >= cell_span_local * 2.0 and right_len >= cell_span_local * 2.0:
                        radius = int(max(3, round(float(margin) * 1.65)))
                        x1 = max(0, int(head_node[1]) - radius)
                        y1 = max(0, int(head_node[0]) - radius)
                        x2 = min(W, int(head_node[1]) + radius + 1)
                        y2 = min(H, int(head_node[0]) + radius + 1)
                        score = float(max(float(first.score), float(second.score)) + 0.02)
                        refined_component_candidates = [
                            ArrowCandidate(
                                bbox=(x1, y1, x2, y2),
                                tip=(float(head_node[1]), float(head_node[0])),
                                direction=(1.0, 0.0),
                                score=score,
                                area=float(max(1.0, left_len)),
                                component_id=int(component_id),
                            ),
                            ArrowCandidate(
                                bbox=(x1, y1, x2, y2),
                                tip=(float(head_node[1]), float(head_node[0])),
                                direction=(-1.0, 0.0),
                                score=score,
                                area=float(max(1.0, right_len)),
                                component_id=int(component_id),
                            ),
                        ]
        component_filtered: List[ArrowCandidate] = []
        for candidate in refined_component_candidates:
            if any(_candidate_similarity(candidate, existing, margin) for existing in component_filtered):
                continue
            component_filtered.append(candidate)

        if not component_filtered and endpoint_graph_allowed:
            component_filtered.extend(
                open_endpoint_fallback_candidates(
                    component_id,
                    endpoints,
                    component_set,
                    margin=max(2, margin),
                    max_keep=min(8, max(2, len(endpoints) // 3)),
                )
            )

        candidates.extend(component_filtered)

    # Prefer stronger candidates and suppress near-duplicates.
    candidates.sort(key=lambda item: item.score, reverse=True)
    filtered: List[ArrowCandidate] = []
    for candidate in candidates:
        if any(candidate.component_id == existing.component_id and _candidate_similarity(candidate, existing, margin) for existing in filtered):
            continue
        if any(_bbox_iou(candidate.bbox, existing.bbox) > 0.70 and _candidate_direction_dot(candidate, existing) > 0.70 for existing in filtered):
            continue
        filtered.append(candidate)
    return filtered


def _candidate_detection_rows(A: int) -> Tuple[int, ...]:
    del A
    # Arrow objects exist in image space.  The requested output matrix size
    # must not change how many arrowheads are detected.
    return (20, 30, 50)


def _needs_multiscale_candidate_recovery(
    bw: np.ndarray,
    component_labels: np.ndarray,
    base_candidate_count: int,
) -> bool:
    skel = skeletonize((bw > 0).astype(bool)).astype(np.uint8)
    adj = _build_skeleton_adjacency(skel)
    if not adj:
        return False
    component_stats: DefaultDict[int, List[int]] = defaultdict(lambda: [0, 0, 0])
    for node, neighbors in adj.items():
        component_id = int(component_labels[node[0], node[1]])
        if component_id <= 0:
            continue
        component_stats[component_id][0] += 1
        if len(neighbors) <= 1:
            component_stats[component_id][1] += 1
        if len(neighbors) > 2:
            component_stats[component_id][2] += 1
    if not component_stats:
        return False
    max_endpoints = max(values[1] for values in component_stats.values())
    max_branches = max(values[2] for values in component_stats.values())
    if max_endpoints < 8 and max_branches < 8:
        return False
    if int(base_candidate_count) < 2:
        return True
    if max_endpoints >= 24 and max_branches >= 60:
        expected_min = min(48, max(8, int(round(max_endpoints * 0.50))))
        return int(base_candidate_count) < expected_min
    return False


def _detect_arrow_candidates_multiscale(
    bw: np.ndarray,
    A: int,
    B: int,
    component_labels: np.ndarray | None = None,
    head_score_thr: float = 0.15,
) -> Tuple[List[ArrowCandidate], Tuple[int, ...], Tuple[int, ...]]:
    """Detect arrow heads at several relative grid scales and merge in pixel space."""
    H, W = bw.shape[:2]
    if component_labels is None:
        component_labels = _label_binary_components(bw)
    del B
    rows = _candidate_detection_rows(A)
    tagged_source: List[Tuple[int, ArrowCandidate]] = []
    counts: List[int] = []
    for row_count in rows:
        col_count = derive_grid_B(row_count, W, H)
        scale_candidates = detect_arrow_candidates(
            bw,
            row_count,
            col_count,
            component_labels=component_labels,
            head_score_thr=head_score_thr,
        )
        counts.append(len(scale_candidates))
        if not scale_candidates:
            continue
        local_cell_span = float(max(H / float(row_count), W / float(col_count), 1.0))
        protected_indices: set[int] = set()
        for first_index, first in enumerate(scale_candidates):
            for second_index in range(first_index + 1, len(scale_candidates)):
                second = scale_candidates[second_index]
                tip_distance_sq = (
                    (first.tip[0] - second.tip[0]) ** 2
                    + (first.tip[1] - second.tip[1]) ** 2
                )
                if (
                    _candidate_direction_dot(first, second) <= -0.75
                    and tip_distance_sq <= (local_cell_span * 0.45) ** 2
                    and _bbox_iou(first.bbox, second.bbox) >= 0.20
                ):
                    protected_indices.update((first_index, second_index))
        for candidate_index, candidate in enumerate(scale_candidates):
            tagged_source.append(
                (
                    int(row_count),
                    ArrowCandidate(
                    bbox=tuple(int(v) for v in candidate.bbox),
                    tip=(float(candidate.tip[0]), float(candidate.tip[1])),
                    direction=(float(candidate.direction[0]), float(candidate.direction[1])),
                    score=float(
                        np.clip(
                            max(float(candidate.score), 0.96 if candidate_index in protected_indices else 0.0),
                            0.0,
                            1.0,
                        )
                    ),
                    area=float(candidate.area),
                    component_id=int(candidate.component_id),
                    ),
                )
            )

    if not tagged_source:
        return [], rows, tuple(counts)

    canonical_rows = 30
    canonical_cols = derive_grid_B(canonical_rows, W, H)
    cell_span = float(max(H / float(canonical_rows), W / float(canonical_cols), 1.0))
    margin = int(round(cell_span * 0.75))
    max_keep = int(min(160, max(24, round(np.sqrt(max(H * W, 1)) / 11.0))))
    clusters: List[Tuple[ArrowCandidate, set[int]]] = []
    for row_count, candidate in sorted(tagged_source, key=lambda item: item[1].score, reverse=True):
        matched_index: int | None = None
        for index, (existing, _support_rows) in enumerate(clusters):
            if candidate.component_id == existing.component_id and _candidate_similarity(candidate, existing, margin):
                matched_index = index
                break
        if matched_index is None:
            clusters.append((candidate, {int(row_count)}))
            continue
        existing, support_rows = clusters[matched_index]
        support_rows.add(int(row_count))
        if candidate.score > existing.score:
            clusters[matched_index] = (candidate, support_rows)

    consensus = [(candidate, len(support_rows)) for candidate, support_rows in clusters]
    consensus.sort(key=lambda item: (int(item[1] >= 2), item[0].score), reverse=True)

    protected_pair_indices: set[int] = set()
    for first_index, (first, _first_support) in enumerate(consensus):
        for second_index in range(first_index + 1, len(consensus)):
            second = consensus[second_index][0]
            tip_distance_sq = (
                (first.tip[0] - second.tip[0]) ** 2
                + (first.tip[1] - second.tip[1]) ** 2
            )
            if (
                first.score >= 0.94
                and second.score >= 0.94
                and _candidate_direction_dot(first, second) <= -0.75
                and tip_distance_sq <= (cell_span * 0.45) ** 2
                and _bbox_iou(first.bbox, second.bbox) >= 0.20
            ):
                protected_pair_indices.update((first_index, second_index))

    def inside_supported_head_cone(strong: ArrowCandidate, weak: ArrowCandidate) -> bool:
        direction = np.asarray(strong.direction, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            return False
        direction /= norm
        rel = np.asarray(
            [float(weak.tip[0] - strong.tip[0]), float(weak.tip[1] - strong.tip[1])],
            dtype=np.float32,
        )
        behind = -float(np.dot(rel, direction))
        lateral = abs(float(rel[0] * direction[1] - rel[1] * direction[0]))
        head_span = float(
            max(
                cell_span * 1.2,
                strong.bbox[2] - strong.bbox[0],
                strong.bbox[3] - strong.bbox[1],
            )
        )
        return bool(0.0 <= behind <= head_span * 2.5 and lateral <= head_span * 1.4)

    filtered_with_support: List[Tuple[ArrowCandidate, int]] = []
    for consensus_index, (candidate, support_count) in enumerate(consensus):
        wing_of_protected_pair = any(
            consensus_index not in protected_pair_indices
            and candidate.score <= consensus[protected_index][0].score - 0.04
            and (candidate.tip[0] - consensus[protected_index][0].tip[0]) ** 2
            + (candidate.tip[1] - consensus[protected_index][0].tip[1]) ** 2
            <= max(
                cell_span * 1.25,
                consensus[protected_index][0].bbox[2] - consensus[protected_index][0].bbox[0],
                consensus[protected_index][0].bbox[3] - consensus[protected_index][0].bbox[1],
            ) ** 2
            for protected_index in protected_pair_indices
        )
        if wing_of_protected_pair:
            continue
        wing_of_supported_head = any(
            support_count == 1
            and existing_support >= 2
            and candidate.score <= existing.score - 0.04
            and _candidate_direction_dot(candidate, existing) < 0.45
            and inside_supported_head_cone(existing, candidate)
            for existing, existing_support in filtered_with_support
        )
        if wing_of_supported_head:
            continue
        filtered = [existing for existing, _support in filtered_with_support]
        if any(candidate.component_id == existing.component_id and _candidate_similarity(candidate, existing, margin) for existing in filtered):
            continue
        if any(
            (candidate.tip[0] - existing.tip[0]) ** 2 + (candidate.tip[1] - existing.tip[1]) ** 2
            <= float((cell_span * 0.42) ** 2)
            and _candidate_direction_dot(candidate, existing) > 0.10
            for existing in filtered
        ):
            continue
        if any(_bbox_iou(candidate.bbox, existing.bbox) > 0.78 and _candidate_direction_dot(candidate, existing) > 0.55 for existing in filtered):
            continue
        filtered_with_support.append((candidate, int(support_count)))
        if len(filtered_with_support) >= max_keep:
            break
    return [candidate for candidate, _support in filtered_with_support], rows, tuple(counts)


def detect_arrow_bboxes(
    bw: np.ndarray,
    A: int,
    B: int,
    occ: np.ndarray | None = None,
    head_score_thr: float = 0.15,
) -> List[Tuple[int, int, int, int]]:
    """Backward-compatible wrapper returning only bounding boxes."""
    del occ
    return [candidate.bbox for candidate in detect_arrow_candidates(bw, A, B, head_score_thr=head_score_thr)]


def _build_skeleton_adjacency(skel: np.ndarray) -> Dict[Pixel, List[Pixel]]:
    pixels = np.argwhere(skel > 0)
    pixel_set = {tuple(int(v) for v in p) for p in pixels}
    adj: Dict[Pixel, List[Pixel]] = {}
    for y, x in pixel_set:
        neighbors: List[Pixel] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nxt = (y + dy, x + dx)
                if nxt in pixel_set:
                    neighbors.append(nxt)
        adj[(y, x)] = neighbors
    return adj


def _connected_components(adj: Dict[Pixel, List[Pixel]]) -> Tuple[List[List[Pixel]], Dict[Pixel, int]]:
    visited: set[Pixel] = set()
    components: List[List[Pixel]] = []
    pixel_to_component: Dict[Pixel, int] = {}

    for node in adj:
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        component: List[Pixel] = []
        while stack:
            current = stack.pop()
            component.append(current)
            pixel_to_component[current] = len(components)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components, pixel_to_component


def _nearest_node(nodes: Iterable[Pixel], target_xy: Tuple[float, float]) -> Pixel | None:
    tx, ty = target_xy
    best_node: Pixel | None = None
    best_dist = float("inf")
    for y, x in nodes:
        dist = (x - tx) ** 2 + (y - ty) ** 2
        if dist < best_dist:
            best_dist = dist
            best_node = (y, x)
    return best_node


def _bfs_within_component(
    adj: Dict[Pixel, List[Pixel]],
    component_nodes: Sequence[Pixel] | set[Pixel],
    root: Pixel,
) -> Tuple[Dict[Pixel, int], Dict[Pixel, Pixel | None]]:
    component_set = component_nodes if isinstance(component_nodes, set) else set(component_nodes)
    dist: Dict[Pixel, int] = {root: 0}
    parent: Dict[Pixel, Pixel | None] = {root: None}
    queue: deque[Pixel] = deque([root])

    while queue:
        current = queue.popleft()
        for neighbor in adj[current]:
            if neighbor not in component_set or neighbor in dist:
                continue
            dist[neighbor] = dist[current] + 1
            parent[neighbor] = current
            queue.append(neighbor)
    return dist, parent


def _path_from_parent(parent: Dict[Pixel, Pixel | None], target: Pixel) -> List[Pixel]:
    path: List[Pixel] = []
    current: Pixel | None = target
    while current is not None:
        path.append(current)
        current = parent.get(current)
    return path


def _dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1])


def _direction_from_vector(vec: Tuple[float, float]) -> DirectionCode:
    dx, dy = float(vec[0]), float(vec[1])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0
    if abs(dx) >= abs(dy):
        return 1 if dx > 0 else 3
    return 2 if dy > 0 else 4


def _is_axis_dominant(vec: Tuple[float, float], threshold: float = 1.8) -> bool:
    dx = abs(float(vec[0]))
    dy = abs(float(vec[1]))
    if dx <= 1e-6 or dy <= 1e-6:
        return True
    ratio = max(dx, dy) / max(min(dx, dy), 1e-6)
    return ratio >= float(threshold)


def _pick_component_head_node(
    component_nodes: Sequence[Pixel],
    candidate: ArrowCandidate,
    max_dist_sq: float,
) -> Pixel | None:
    bx1, by1, bx2, by2 = candidate.bbox
    inside = [node for node in component_nodes if bx1 <= node[1] < bx2 and by1 <= node[0] < by2]
    if inside:
        return _nearest_node(inside, candidate.tip)
    nearest = _nearest_node(component_nodes, candidate.tip)
    if nearest is None:
        return None
    dist_sq = (nearest[1] - candidate.tip[0]) ** 2 + (nearest[0] - candidate.tip[1]) ** 2
    if dist_sq > max_dist_sq:
        return None
    return nearest


def _endpoint_head_score(
    path: Sequence[Pixel],
    dist_map: np.ndarray,
    endpoint: str,
    probe_len: int = 8,
) -> float:
    if len(path) < 3:
        return 0.0
    if endpoint not in {"start", "end"}:
        raise ValueError("endpoint must be 'start' or 'end'")
    seq = list(path if endpoint == "start" else reversed(path))
    probe = seq[: min(len(seq), max(3, int(probe_len)))]
    widths = [float(dist_map[y, x]) for y, x in probe]
    if len(widths) < 2:
        return 0.0
    tip_width = widths[0]
    inner_width = float(np.mean(widths[1:]))
    max_width = float(np.max(widths[1:]))
    taper = float(np.clip((inner_width - tip_width) / max(inner_width, 0.75), 0.0, 1.0))
    growth = float(np.clip((max_width - tip_width) / 2.5, 0.0, 1.0))
    length = float(np.clip((len(probe) - 2) / 6.0, 0.0, 1.0))
    return 0.60 * taper + 0.25 * growth + 0.15 * length


def _pixel_to_cell(y: int, x: int, A: int, B: int, H: int, W: int) -> Cell:
    row = int(np.clip(np.floor(y / max(H / float(A), 1e-6)), 0, A - 1))
    col = int(np.clip(np.floor(x / max(W / float(B), 1e-6)), 0, B - 1))
    return row, col


def _expand_diagonal_steps(cells: List[Cell]) -> List[Cell]:
    if not cells:
        return []

    expanded: List[Cell] = [cells[0]]
    for nxt in cells[1:]:
        cur = expanded[-1]
        dr = nxt[0] - cur[0]
        dc = nxt[1] - cur[1]
        if abs(dr) <= 1 and abs(dc) <= 1 and dr != 0 and dc != 0:
            expanded.append((cur[0], cur[1] + dc))
        elif abs(dr) > 1 or abs(dc) > 1:
            # Walk horizontally first, then vertically. This matches the requested matrix pattern.
            step_c = 1 if dc > 0 else -1
            step_r = 1 if dr > 0 else -1
            for col in range(cur[1] + step_c, nxt[1] + step_c, step_c) if dc != 0 else []:
                expanded.append((cur[0], col))
            row_start = expanded[-1][0]
            col_last = expanded[-1][1]
            for row in range(row_start + step_r, nxt[0] + step_r, step_r) if dr != 0 else []:
                expanded.append((row, col_last))
            continue
        expanded.append(nxt)

    deduped: List[Cell] = [expanded[0]]
    for cell in expanded[1:]:
        if cell != deduped[-1]:
            deduped.append(cell)
    return deduped


def _cell_path_from_pixel_path(
    path: Sequence[Pixel],
    A: int,
    B: int,
    H: int,
    W: int,
    expand_diagonal: bool = True,
) -> List[Cell]:
    cells: List[Cell] = []
    for y, x in path:
        cell = _pixel_to_cell(y, x, A, B, H, W)
        if not cells or cell != cells[-1]:
            cells.append(cell)
    if expand_diagonal:
        return _expand_diagonal_steps(cells)
    return cells


def _is_locally_supported_path(candidate_path: Sequence[Cell], raw_path: Sequence[Cell], tolerance: int = 1) -> bool:
    raw_cells = list(raw_path)
    if not candidate_path or not raw_cells:
        return False
    for cell in candidate_path:
        if min(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) for other in raw_cells) > tolerance:
            return False
    return True


def _grid_traversal_cells(
    start_pixel: Pixel,
    end_pixel: Pixel,
    A: int,
    B: int,
    H: int,
    W: int,
) -> List[Cell]:
    """Trace a continuous centerline through the grid.

    This is more stable for diagonal arrows than relying on local skeleton wiggles.
    """
    cell_h = max(H / float(A), 1e-6)
    cell_w = max(W / float(B), 1e-6)

    y0, x0 = start_pixel
    y1, x1 = end_pixel
    gy0 = float(y0) / cell_h
    gx0 = float(x0) / cell_w
    gy1 = float(y1) / cell_h
    gx1 = float(x1) / cell_w

    row = int(np.clip(np.floor(gy0), 0, A - 1))
    col = int(np.clip(np.floor(gx0), 0, B - 1))
    end_row = int(np.clip(np.floor(gy1), 0, A - 1))
    end_col = int(np.clip(np.floor(gx1), 0, B - 1))

    drow = gy1 - gy0
    dcol = gx1 - gx0

    if abs(drow) < 1e-9 and abs(dcol) < 1e-9:
        return [(row, col)]

    step_row = 1 if drow > 0 else -1 if drow < 0 else 0
    step_col = 1 if dcol > 0 else -1 if dcol < 0 else 0

    if step_col != 0:
        next_col_boundary = (col + 1) if step_col > 0 else col
        t_max_col = (next_col_boundary - gx0) / dcol
        t_delta_col = 1.0 / abs(dcol)
    else:
        t_max_col = float("inf")
        t_delta_col = float("inf")

    if step_row != 0:
        next_row_boundary = (row + 1) if step_row > 0 else row
        t_max_row = (next_row_boundary - gy0) / drow
        t_delta_row = 1.0 / abs(drow)
    else:
        t_max_row = float("inf")
        t_delta_row = float("inf")

    cells: List[Cell] = [(row, col)]
    max_steps = A * B * 4
    for _ in range(max_steps):
        if (row, col) == (end_row, end_col):
            break
        if t_max_col < t_max_row:
            col += step_col
            t_max_col += t_delta_col
            if 0 <= row < A and 0 <= col < B and (row, col) != cells[-1]:
                cells.append((row, col))
        elif t_max_row < t_max_col:
            row += step_row
            t_max_row += t_delta_row
            if 0 <= row < A and 0 <= col < B and (row, col) != cells[-1]:
                cells.append((row, col))
        else:
            col += step_col
            if 0 <= row < A and 0 <= col < B and (row, col) != cells[-1]:
                cells.append((row, col))
            row += step_row
            t_max_col += t_delta_col
            t_max_row += t_delta_row
            if 0 <= row < A and 0 <= col < B and (row, col) != cells[-1]:
                cells.append((row, col))
    return _expand_diagonal_steps(cells)


def _direction_from_step(src: Cell, dst: Cell) -> DirectionCode:
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    if abs(dc) >= abs(dr) and dc > 0:
        return 1
    if abs(dr) > abs(dc) and dr > 0:
        return 2
    if abs(dc) >= abs(dr) and dc < 0:
        return 3
    if abs(dr) > abs(dc) and dr < 0:
        return 4
    return 0


def _vote_path(
    cell_path: Sequence[Cell],
    votes: DefaultDict[Cell, DefaultDict[DirectionCode, float]],
    weight: float = 3.0,
    terminal_weight: float = 1.5,
    transition_votes: DefaultDict[Tuple[Cell, Cell], float] | None = None,
) -> None:
    if len(cell_path) < 2:
        return
    last_direction = 0
    for src, dst in zip(cell_path[:-1], cell_path[1:]):
        direction = _direction_from_step(src, dst)
        if direction == 0:
            continue
        votes[src][direction] += weight
        if transition_votes is not None:
            transition_votes[(src, dst)] += float(weight)
        last_direction = direction
    if last_direction != 0:
        votes[cell_path[-1]][last_direction] += terminal_weight


def _direction_from_object_step(src: Cell, dst: Cell, object_direction: DirectionCode) -> DirectionCode:
    step_direction = _direction_from_step(src, dst)
    if step_direction == 0 or int(object_direction) not in _DIRECTION_CODE_TO_CELL_STEP:
        return step_direction
    step_vec = _DIRECTION_CODE_TO_CELL_STEP.get(int(step_direction))
    object_vec = _DIRECTION_CODE_TO_CELL_STEP.get(int(object_direction))
    if step_vec is None or object_vec is None:
        return step_direction
    if int(step_vec[0]) * int(object_vec[0]) + int(step_vec[1]) * int(object_vec[1]) < 0:
        return int(object_direction)
    return step_direction


def _vote_object_path(
    object_path: ArrowObjectPath,
    votes: DefaultDict[Cell, DefaultDict[DirectionCode, float]],
    weight: float = 3.0,
    terminal_weight: float = 1.5,
    transition_votes: DefaultDict[Tuple[Cell, Cell], float] | None = None,
) -> None:
    cell_path = tuple(object_path.vote_cell_path)
    if len(cell_path) < 2:
        return
    object_direction = int(object_path.direction_code)
    last_direction = 0
    for src, dst in zip(cell_path[:-1], cell_path[1:]):
        raw_direction = _direction_from_step(src, dst)
        direction = _direction_from_object_step(src, dst, object_direction)
        if direction == 0:
            continue
        votes[src][direction] += weight
        if transition_votes is not None and direction == raw_direction:
            transition_votes[(src, dst)] += float(weight)
        last_direction = direction
    if last_direction != 0:
        votes[cell_path[-1]][last_direction] += terminal_weight


def rasterize_arrow_object_paths(
    object_paths: Sequence[ArrowObjectPath],
    A: int,
    B: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize extracted arrow objects before cell-level fallback is applied."""
    matrix = np.zeros((A, B), dtype=np.uint8)
    support = np.zeros((A, B), dtype=np.uint8)
    overlap = np.zeros((A, B), dtype=np.int32)
    votes = np.zeros((A, B, 5), dtype=np.float32)

    for object_path in object_paths:
        touched: set[Cell] = set()
        for cell in tuple(object_path.raw_cell_path) + tuple(object_path.vote_cell_path):
            row, col = int(cell[0]), int(cell[1])
            if 0 <= row < A and 0 <= col < B:
                touched.add((row, col))
        for row, col in touched:
            support[row, col] = 1
            overlap[row, col] += 1

        cell_path = tuple(object_path.vote_cell_path) if len(object_path.vote_cell_path) >= 2 else tuple(object_path.raw_cell_path)
        if len(cell_path) < 2:
            continue
        object_direction = int(object_path.direction_code)
        weight = float(np.clip(1.0 + 4.2 * float(object_path.score), 0.25, 8.0))
        last_direction = 0
        for src, dst in zip(cell_path[:-1], cell_path[1:]):
            direction = _direction_from_object_step(src, dst, object_direction)
            if direction == 0:
                continue
            row, col = int(src[0]), int(src[1])
            if 0 <= row < A and 0 <= col < B:
                votes[row, col, direction] += weight
            last_direction = direction
        if last_direction != 0:
            row, col = int(cell_path[-1][0]), int(cell_path[-1][1])
            if 0 <= row < A and 0 <= col < B:
                votes[row, col, last_direction] += 0.5 * weight

    positive = np.sum(votes[:, :, 1:5], axis=2) > 0
    if np.any(positive):
        matrix[positive] = np.argmax(votes[:, :, 1:5][positive], axis=1).astype(np.uint8) + 1
    return matrix.astype(np.uint8), support.astype(np.uint8), overlap.astype(np.int32)


def _vote_confidence(cell_votes: Dict[DirectionCode, float]) -> float:
    if not cell_votes:
        return 0.0
    ordered = sorted((float(score), int(direction)) for direction, score in cell_votes.items() if score > 0)
    if not ordered:
        return 0.0
    best = ordered[-1][0]
    second = ordered[-2][0] if len(ordered) >= 2 else 0.0
    total = float(sum(score for score, _direction in ordered))
    if total <= 1e-6:
        return 0.0
    strength = float(np.clip(total / 4.0, 0.0, 1.0))
    dominance = best / total
    margin = (best - second) / max(best, 1e-6)
    return float(np.clip(strength * (0.6 * dominance + 0.4 * margin), 0.0, 1.0))


def _compute_cell_axis_field(
    skel: np.ndarray,
    A: int,
    B: int,
    window_radius: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the local, sign-agnostic axis of ink inside each cell neighborhood."""
    H, W = skel.shape[:2]
    cell_h = H / float(A)
    cell_w = W / float(B)
    axis_x = np.zeros((A, B), dtype=np.float32)
    axis_y = np.zeros((A, B), dtype=np.float32)
    axis_conf = np.zeros((A, B), dtype=np.float32)

    for i in range(A):
        row0 = max(0, i - window_radius)
        row1 = min(A, i + window_radius + 1)
        y1 = int(round(row0 * cell_h))
        y2 = int(round(row1 * cell_h))
        y1 = max(0, min(H - 1, y1))
        y2 = max(y1 + 1, min(H, y2))
        for j in range(B):
            col0 = max(0, j - window_radius)
            col1 = min(B, j + window_radius + 1)
            x1 = int(round(col0 * cell_w))
            x2 = int(round(col1 * cell_w))
            x1 = max(0, min(W - 1, x1))
            x2 = max(x1 + 1, min(W, x2))

            ys, xs = np.nonzero(skel[y1:y2, x1:x2] > 0)
            if xs.size < 5:
                continue

            pts = np.column_stack([xs.astype(np.float32) + float(x1), ys.astype(np.float32) + float(y1)])
            center = np.mean(pts, axis=0, dtype=np.float32)
            shifted = pts - center
            if shifted.shape[0] < 2:
                continue

            cov = np.cov(shifted.T)
            if not isinstance(cov, np.ndarray) or cov.shape != (2, 2):
                continue
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)
            major = eigvecs[:, int(order[-1])].astype(np.float32)
            l1 = float(max(eigvals[int(order[-1])], 0.0))
            l2 = float(max(eigvals[int(order[-2])], 0.0))
            if l1 <= 1e-6:
                continue
            conf = float(np.clip((l1 - l2) / max(l1 + l2, 1e-6), 0.0, 1.0))
            if conf <= 0.03:
                continue

            norm = float(np.linalg.norm(major))
            if norm <= 1e-6:
                continue
            major /= norm
            # Fix the sign for stability only; candidate matching still uses abs(dot).
            if major[0] < 0 or (abs(float(major[0])) < 1e-6 and major[1] < 0):
                major = -major
            axis_x[i, j] = float(major[0])
            axis_y[i, j] = float(major[1])
            axis_conf[i, j] = np.float32(conf)
    return axis_x, axis_y, axis_conf


def _add_candidate_guided_votes(
    candidates: Sequence[ArrowCandidate],
    occ: np.ndarray,
    path_occ: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    axis_conf: np.ndarray,
    A: int,
    B: int,
    H: int,
    W: int,
    votes: DefaultDict[Cell, DefaultDict[DirectionCode, float]],
) -> None:
    """Use local axis agreement to separate overlapping arrow candidates at the cell level."""
    if not candidates:
        return

    cell_h = H / float(A)
    cell_w = W / float(B)
    cell_scale = max(1.0, float(np.hypot(cell_h, cell_w)))
    cached: List[Tuple[ArrowCandidate, np.ndarray]] = []
    for candidate in candidates:
        vec = np.asarray(candidate.direction, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-6:
            continue
        cached.append((candidate, vec / norm))
    if not cached:
        return

    support = np.maximum(occ.astype(np.uint8), path_occ.astype(np.uint8))
    for i in range(A):
        cy = (i + 0.5) * cell_h
        for j in range(B):
            if int(support[i, j]) != 1:
                continue
            center = np.array([(j + 0.5) * cell_w, cy], dtype=np.float32)
            best_direction = 0
            best_score = 0.0

            for candidate, axis in cached:
                to_tip = np.array([float(candidate.tip[0]) - float(center[0]), float(candidate.tip[1]) - float(center[1])], dtype=np.float32)
                dist = float(np.linalg.norm(to_tip))
                if dist <= 1e-6:
                    continue
                unit_to_tip = to_tip / dist
                sign_align = float(np.dot(unit_to_tip, axis))
                if sign_align <= 0.10:
                    continue

                lateral = abs(float((center[0] - candidate.tip[0]) * axis[1] - (center[1] - candidate.tip[1]) * axis[0]))
                lateral_score = float(np.exp(-((lateral / max(cell_scale * 1.8, 1e-6)) ** 2)))

                proj = float((center[0] - candidate.tip[0]) * axis[0] + (center[1] - candidate.tip[1]) * axis[1])
                if proj > cell_scale * 1.2:
                    continue
                ahead_score = 1.0 - float(np.clip(proj / max(cell_scale * 2.5, 1e-6), 0.0, 1.0))

                axis_score = 0.45
                local_conf = float(axis_conf[i, j])
                if local_conf > 0.08:
                    local_axis = np.array([float(axis_x[i, j]), float(axis_y[i, j])], dtype=np.float32)
                    align = abs(float(np.dot(local_axis, axis)))
                    if align < 0.35:
                        continue
                    axis_score = 0.25 + 0.75 * align

                score = (
                    (0.30 + 0.70 * float(max(candidate.score, 0.0)))
                    * sign_align
                    * lateral_score
                    * ahead_score
                    * axis_score
                )
                if int(path_occ[i, j]) == 1:
                    score *= 1.10

                if score > best_score:
                    best_score = score
                    best_direction = _direction_from_vector((float(to_tip[0]), float(to_tip[1])))

            if best_direction != 0 and best_score >= 0.03:
                votes[(i, j)][best_direction] += float(0.7 + 2.2 * best_score)


def _fallback_orientation_votes(
    adj: Dict[Pixel, List[Pixel]],
    A: int,
    B: int,
    H: int,
    W: int,
    votes: DefaultDict[Cell, DefaultDict[DirectionCode, float]],
    allowed_cells: set[Cell] | None = None,
) -> None:
    for (y, x), neighbors in adj.items():
        if not neighbors:
            continue
        dx = float(sum(nb[1] - x for nb in neighbors))
        dy = float(sum(nb[0] - y for nb in neighbors))
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        if abs(dx) >= abs(dy):
            direction = 1 if dx > 0 else 3
        else:
            direction = 2 if dy > 0 else 4
        cell = _pixel_to_cell(y, x, A, B, H, W)
        if allowed_cells is not None and cell not in allowed_cells:
            continue
        if cell in votes:
            continue
        votes[cell][direction] += 0.5


def build_support_mask_from_binary(bw: np.ndarray, A: int, B: int, min_pixels: int = 1) -> np.ndarray:
    """Return cells touched by any meaningful amount of foreground ink."""
    H, W = bw.shape[:2]
    cell_h = H / float(A)
    cell_w = W / float(B)
    mask = np.zeros((A, B), dtype=np.uint8)
    for i in range(A):
        y1 = int(round(i * cell_h))
        y2 = int(round((i + 1) * cell_h))
        y1 = max(0, min(H - 1, y1))
        y2 = max(y1 + 1, min(H, y2))
        for j in range(B):
            x1 = int(round(j * cell_w))
            x2 = int(round((j + 1) * cell_w))
            x1 = max(0, min(W - 1, x1))
            x2 = max(x1 + 1, min(W, x2))
            if int(np.count_nonzero(bw[y1:y2, x1:x2] > 0)) >= min_pixels:
                mask[i, j] = 1
    return mask


def _neighbors4(cell: Cell, A: int, B: int) -> List[Cell]:
    row, col = cell
    neighbors: List[Cell] = []
    if row > 0:
        neighbors.append((row - 1, col))
    if row + 1 < A:
        neighbors.append((row + 1, col))
    if col > 0:
        neighbors.append((row, col - 1))
    if col + 1 < B:
        neighbors.append((row, col + 1))
    return neighbors


def _rasterize_skeleton_support(
    adj: Dict[Pixel, List[Pixel]],
    A: int,
    B: int,
    H: int,
    W: int,
) -> np.ndarray:
    """Rasterize a pixel skeleton as a deterministic 4-connected supercover."""
    support = np.zeros((A, B), dtype=np.uint8)
    cell_by_node: Dict[Pixel, Cell] = {
        node: _pixel_to_cell(node[0], node[1], A, B, H, W)
        for node in adj
    }
    for node, cell in cell_by_node.items():
        support[cell] = 1
        for neighbor in adj.get(node, []):
            if node >= neighbor:
                continue
            next_cell = cell_by_node.get(neighbor)
            if next_cell is None or next_cell == cell:
                continue
            for row, col in _expand_diagonal_steps([cell, next_cell]):
                if 0 <= row < A and 0 <= col < B:
                    support[row, col] = 1
    return support.astype(np.uint8)


def _propagate_directions_on_unbranched_centerline(
    matrix: np.ndarray,
    centerline_support: np.ndarray,
    confidence: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Fill zero runs locally, stopping at ambiguous grid branches.

    Only the initial detector output is used as an anchor.  Proposals are
    accumulated before any fill is applied, so iteration order cannot make a
    newly filled cell spread farther in the same pass.
    """
    base = np.asarray(matrix, dtype=np.uint8)
    support = (np.asarray(centerline_support) > 0).astype(np.uint8)
    if base.shape != support.shape:
        raise ValueError("matrix and centerline_support must have the same shape")
    A, B = base.shape
    proposals = np.zeros((A, B, 5), dtype=np.float32)
    seed_cells = [tuple(int(v) for v in row) for row in np.argwhere((base > 0) & (support > 0))]

    for seed in seed_cells:
        seed_direction = int(base[seed])
        seed_step = _DIRECTION_CODE_TO_CELL_STEP.get(seed_direction)
        if seed_step is None:
            continue
        seed_vec = np.asarray([float(seed_step[1]), float(seed_step[0])], dtype=np.float32)
        for first in _neighbors4(seed, A, B):
            if int(support[first]) != 1 or int(base[first]) != 0:
                continue
            initial = np.asarray(
                [float(first[1] - seed[1]), float(first[0] - seed[0])],
                dtype=np.float32,
            )
            initial_norm = float(np.linalg.norm(initial))
            if initial_norm <= 1e-6:
                continue
            initial /= initial_norm
            signed_alignment = float(np.dot(initial, seed_vec))
            if abs(signed_alignment) < 0.55:
                continue
            downstream = signed_alignment > 0.0

            path: List[Cell] = [seed, first]
            previous = seed
            current = first
            heading = initial.copy()
            visited: set[Cell] = {seed, first}
            encountered_label: Cell | None = None
            for _ in range(max(1, A * B)):
                if int(base[current]) != 0:
                    encountered_label = current
                    break
                options: List[Tuple[float, Cell, np.ndarray]] = []
                for neighbor in _neighbors4(current, A, B):
                    if neighbor == previous or neighbor in visited or int(support[neighbor]) != 1:
                        continue
                    step = np.asarray(
                        [float(neighbor[1] - current[1]), float(neighbor[0] - current[0])],
                        dtype=np.float32,
                    )
                    step_norm = float(np.linalg.norm(step))
                    if step_norm <= 1e-6:
                        continue
                    step /= step_norm
                    options.append((float(np.dot(step, heading)), neighbor, step))
                if not options:
                    break
                options.sort(key=lambda item: item[0], reverse=True)
                best_score, next_cell, next_step = options[0]
                if best_score < -0.10:
                    break
                if len(options) > 1 and options[1][0] >= best_score - 0.20:
                    break
                path.append(next_cell)
                visited.add(next_cell)
                previous, current = current, next_cell
                blended = heading * 0.65 + next_step * 0.35
                blended_norm = float(np.linalg.norm(blended))
                heading = blended / blended_norm if blended_norm > 1e-6 else next_step
                if int(base[current]) != 0:
                    encountered_label = current
                    break

            if len(path) < 2:
                continue
            if encountered_label is not None:
                terminal_direction = int(base[encountered_label])
                terminal_step = _DIRECTION_CODE_TO_CELL_STEP.get(terminal_direction)
                if terminal_step is not None and len(path) >= 2:
                    flow_src, flow_dst = (path[-2], path[-1]) if downstream else (path[-1], path[-2])
                    flow_direction = _direction_from_step(flow_src, flow_dst)
                    flow_vec = _DIRECTION_CODE_TO_CELL_STEP.get(flow_direction)
                    if flow_vec is not None:
                        agreement = int(flow_vec[0]) * int(terminal_step[0]) + int(flow_vec[1]) * int(terminal_step[1])
                        if agreement < 0:
                            continue

            zero_path = [cell for cell in path[1:] if int(base[cell]) == 0]
            if not zero_path:
                continue
            if downstream:
                ordered = path
                last_direction = seed_direction
                for index in range(1, len(ordered)):
                    cell = ordered[index]
                    if int(base[cell]) != 0:
                        continue
                    if index + 1 < len(ordered):
                        direction = _direction_from_step(cell, ordered[index + 1])
                        if direction != 0:
                            last_direction = direction
                    proposals[cell[0], cell[1], last_direction] += 1.0
            else:
                for index in range(1, len(path)):
                    cell = path[index]
                    if int(base[cell]) != 0:
                        continue
                    direction = _direction_from_step(cell, path[index - 1])
                    if direction != 0:
                        proposals[cell[0], cell[1], direction] += 1.0

    filled = base.copy()
    confidence_out = (
        np.asarray(confidence, dtype=np.float32).copy()
        if confidence is not None and np.asarray(confidence).shape == base.shape
        else np.zeros(base.shape, dtype=np.float32)
    )
    fill_count = 0
    for row, col in np.argwhere((base == 0) & (support > 0)):
        direction_votes = proposals[int(row), int(col), 1:5]
        total = float(np.sum(direction_votes))
        if total <= 0.0:
            continue
        order = np.argsort(direction_votes)[::-1]
        best_index = int(order[0])
        best = float(direction_votes[best_index])
        second = float(direction_votes[int(order[1])]) if order.size > 1 else 0.0
        if best <= second or best / total < 0.67:
            continue
        filled[int(row), int(col)] = np.uint8(best_index + 1)
        confidence_out[int(row), int(col)] = np.float32(min(0.64, 0.44 + 0.08 * best))
        fill_count += 1
    return filled.astype(np.uint8), confidence_out.astype(np.float32), int(fill_count)


def _augment_support_mask(mask: np.ndarray) -> np.ndarray:
    """Bridge one-cell holes conservatively without overwriting labels."""
    A, B = mask.shape
    augmented = mask.astype(np.uint8).copy()
    for i in range(A):
        for j in range(B):
            if augmented[i, j] == 1:
                continue
            up = i > 0 and mask[i - 1, j] == 1
            down = i + 1 < A and mask[i + 1, j] == 1
            left = j > 0 and mask[i, j - 1] == 1
            right = j + 1 < B and mask[i, j + 1] == 1
            if (up and down) or (left and right):
                augmented[i, j] = 1
                continue
            if (up and left) or (up and right) or (down and left) or (down and right):
                augmented[i, j] = 1
    return augmented


def build_component_cell_map(
    bw: np.ndarray,
    A: int,
    B: int,
    ink_thr: float = DEFAULT_INK_THRESHOLD,
    min_pixels: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map each occupied cell to its dominant connected foreground component."""
    labels = _label_binary_components(bw)
    H, W = bw.shape[:2]
    cell_h = H / float(A)
    cell_w = W / float(B)
    component_ids = np.zeros((A, B), dtype=np.int32)
    counts = np.zeros((A, B), dtype=np.int32)

    for i in range(A):
        y1 = int(round(i * cell_h))
        y2 = int(round((i + 1) * cell_h))
        y1 = max(0, min(H - 1, y1))
        y2 = max(y1 + 1, min(H, y2))
        for j in range(B):
            x1 = int(round(j * cell_w))
            x2 = int(round((j + 1) * cell_w))
            x1 = max(0, min(W - 1, x1))
            x2 = max(x1 + 1, min(W, x2))
            cell_labels = labels[y1:y2, x1:x2]
            foreground = cell_labels[cell_labels > 0]
            if foreground.size == 0:
                continue
            ids, id_counts = np.unique(foreground, return_counts=True)
            idx = int(np.argmax(id_counts))
            best_id = int(ids[idx])
            best_count = int(id_counts[idx])
            area = int(cell_labels.size)
            threshold = max(int(min_pixels), int(round(area * max(0.004, float(ink_thr)))))
            if best_count >= threshold:
                component_ids[i, j] = best_id
                counts[i, j] = best_count
    return component_ids, counts


def _augment_component_ids(component_ids: np.ndarray) -> np.ndarray:
    """Fill only one-cell holes whose neighboring support belongs to the same object."""
    A, B = component_ids.shape
    augmented = component_ids.astype(np.int32).copy()
    orthogonal_pairs = [
        ((-1, 0), (1, 0)),
        ((0, -1), (0, 1)),
        ((-1, 0), (0, -1)),
        ((-1, 0), (0, 1)),
        ((1, 0), (0, -1)),
        ((1, 0), (0, 1)),
    ]
    diagonal_pairs = [
        ((-1, -1), (1, 1)),
        ((-1, 1), (1, -1)),
    ]

    for i in range(A):
        for j in range(B):
            if augmented[i, j] != 0:
                continue

            chosen = 0
            for (d1, d2) in orthogonal_pairs + diagonal_pairs:
                r1, c1 = i + d1[0], j + d1[1]
                r2, c2 = i + d2[0], j + d2[1]
                if not (0 <= r1 < A and 0 <= c1 < B and 0 <= r2 < A and 0 <= c2 < B):
                    continue
                id1 = int(component_ids[r1, c1])
                id2 = int(component_ids[r2, c2])
                if id1 != 0 and id1 == id2:
                    chosen = id1
                    break
            if chosen != 0:
                augmented[i, j] = chosen
    return augmented


def _strict_seed_bridge_mask(seed_mask: np.ndarray, component_ids: np.ndarray, bridge_ids: np.ndarray) -> np.ndarray:
    """Allow only one-cell bridges backed by seeded cells from the same object."""
    A, B = seed_mask.shape
    merged_ids = np.where(component_ids > 0, component_ids, bridge_ids).astype(np.int32)
    bridge_mask = np.zeros((A, B), dtype=np.uint8)
    neighbor_pairs = [
        ((-1, 0), (1, 0)),
        ((0, -1), (0, 1)),
        ((-1, 0), (0, -1)),
        ((-1, 0), (0, 1)),
        ((1, 0), (0, -1)),
        ((1, 0), (0, 1)),
        ((-1, -1), (1, 1)),
        ((-1, 1), (1, -1)),
    ]

    for i in range(A):
        for j in range(B):
            if int(seed_mask[i, j]) == 1:
                continue
            for (d1, d2) in neighbor_pairs:
                r1, c1 = i + d1[0], j + d1[1]
                r2, c2 = i + d2[0], j + d2[1]
                if not (0 <= r1 < A and 0 <= c1 < B and 0 <= r2 < A and 0 <= c2 < B):
                    continue
                if int(seed_mask[r1, c1]) != 1 or int(seed_mask[r2, c2]) != 1:
                    continue
                id1 = int(merged_ids[r1, c1])
                id2 = int(merged_ids[r2, c2])
                if id1 != 0 and id1 == id2:
                    bridge_mask[i, j] = 1
                    break
    return bridge_mask.astype(np.uint8)


def _head_stage_masks_from_candidates(
    candidates: Sequence[ArrowCandidate],
    A: int,
    B: int,
    H: int,
    W: int,
) -> Tuple[np.ndarray, np.ndarray, List[Cell]]:
    tip_mask = np.zeros((A, B), dtype=np.uint8)
    halo_mask = np.zeros((A, B), dtype=np.uint8)
    tip_cells: List[Cell] = []
    seen: set[Cell] = set()
    for candidate in candidates:
        cell = _pixel_to_cell(
            int(round(candidate.tip[1])),
            int(round(candidate.tip[0])),
            A,
            B,
            H,
            W,
        )
        if cell not in seen:
            seen.add(cell)
            tip_cells.append(cell)
        tip_mask[cell] = 1
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                row = cell[0] + di
                col = cell[1] + dj
                if 0 <= row < A and 0 <= col < B:
                    halo_mask[row, col] = 1
    return tip_mask, halo_mask, tip_cells


def build_head_halo_mask(
    bw: np.ndarray,
    A: int,
    B: int,
    head_score_thr: float = 0.15,
) -> np.ndarray:
    """Allow at most a one-cell correction around detected arrow heads."""
    H, W = bw.shape[:2]
    component_labels = _label_binary_components(bw)
    candidates = detect_arrow_candidates(
        bw,
        A,
        B,
        component_labels=component_labels,
        head_score_thr=head_score_thr,
    )
    _tip_mask, halo_mask, _tip_cells = _head_stage_masks_from_candidates(candidates, A, B, H, W)
    return halo_mask


def build_head_stage_data(
    bw: np.ndarray,
    A: int,
    B: int,
    head_score_thr: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, List[Cell]]:
    """Return head tip cells and their immediate one-cell halo."""
    H, W = bw.shape[:2]
    component_labels = _label_binary_components(bw)
    candidates = detect_arrow_candidates(
        bw,
        A,
        B,
        component_labels=component_labels,
        head_score_thr=head_score_thr,
    )
    return _head_stage_masks_from_candidates(candidates, A, B, H, W)


def build_guidance_support_mask(
    bw: np.ndarray,
    A: int,
    B: int,
    occ: np.ndarray | None = None,
    matrix: np.ndarray | None = None,
    path_occ: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    ink_thr: float = DEFAULT_INK_THRESHOLD,
    low_conf_threshold: float = 0.55,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a conservative support map that never invents long empty corridors."""
    component_ids, counts = build_component_cell_map(
        bw,
        A,
        B,
        ink_thr=max(DEFAULT_INK_THRESHOLD, float(ink_thr)),
        min_pixels=2,
    )
    component_ids = np.where(counts > 0, component_ids, 0).astype(np.int32)
    bridge_ids = _augment_component_ids(component_ids)
    base_support = (component_ids > 0).astype(np.uint8)
    matrix_seed = (matrix > 0).astype(np.uint8) if matrix is not None else np.zeros((A, B), dtype=np.uint8)
    path_seed = (path_occ > 0).astype(np.uint8) if path_occ is not None else np.zeros((A, B), dtype=np.uint8)
    occ_seed = (occ > 0).astype(np.uint8) if occ is not None else np.zeros((A, B), dtype=np.uint8)

    # Route BFS primarily through cells that were already traced as path cells or
    # are explicitly labeled in the matrix. Falling back to broad occupancy is only
    # allowed when we have no directional seed at all.
    seed_mask = np.maximum(matrix_seed, path_seed).astype(np.uint8)
    if int(np.count_nonzero(seed_mask)) == 0:
        seed_mask = (occ_seed & base_support).astype(np.uint8)
    seed_mask = np.maximum(seed_mask, matrix_seed).astype(np.uint8)

    strict_bridge_mask = _strict_seed_bridge_mask(seed_mask, component_ids, bridge_ids)
    if matrix is None:
        fillable_mask = strict_bridge_mask.astype(np.uint8)
    else:
        zero_mask = (matrix == 0).astype(np.uint8)
        fillable_mask = (strict_bridge_mask & zero_mask).astype(np.uint8)

    support_mask = np.maximum(seed_mask, fillable_mask).astype(np.uint8)

    return support_mask.astype(np.uint8), fillable_mask.astype(np.uint8), bridge_ids.astype(np.int32)


def _path_alignment_near_head(path_tail_to_head: Sequence[Pixel], direction: Tuple[float, float], probe_len: int = 4) -> float:
    if len(path_tail_to_head) < 2:
        return 0.0
    head = path_tail_to_head[-1]
    anchor = path_tail_to_head[max(0, len(path_tail_to_head) - 1 - max(1, int(probe_len)))]
    vec = np.array([head[1] - anchor[1], head[0] - anchor[0]], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return 0.0
    ref = np.asarray(direction, dtype=np.float32)
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm <= 1e-6:
        return 0.0
    return float(np.clip(np.dot(vec / norm, ref / ref_norm), -1.0, 1.0))


def _trace_local_tail_to_head_path(
    head_node: Pixel,
    component_nodes: Sequence[Pixel] | set[Pixel],
    adj: Dict[Pixel, List[Pixel]],
    direction: Tuple[float, float],
    max_pixels: float,
) -> List[Pixel]:
    """Trace only the local shaft behind one arrow head.

    Complex drawings often contain many arrow heads in one connected ink object.
    In that case a global BFS endpoint is not the tail of every arrow. This
    routine walks backward from the head along the opposite of the candidate
    direction and returns a local tail-to-head path.
    """
    axis = np.asarray(direction, dtype=np.float32)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-6:
        return []
    backward_axis = -(axis / axis_norm)
    component_set = component_nodes if isinstance(component_nodes, set) else set(component_nodes)
    if head_node not in component_set:
        return []

    head_to_tail: List[Pixel] = [head_node]
    visited: set[Pixel] = {head_node}
    prev: Pixel | None = None
    current = head_node
    travelled = 0.0

    for _ in range(max(8, int(round(max_pixels * 2.0)))):
        choices: List[Tuple[float, float, Pixel, float]] = []
        for neighbor in adj.get(current, []):
            if neighbor == prev or neighbor in visited or neighbor not in component_set:
                continue
            step = np.asarray([float(neighbor[1] - current[1]), float(neighbor[0] - current[0])], dtype=np.float32)
            step_norm = float(np.linalg.norm(step))
            if step_norm <= 1e-6:
                continue
            step_unit = step / step_norm
            align = float(np.dot(step_unit, backward_axis))
            lateral = abs(float(step_unit[0] * backward_axis[1] - step_unit[1] * backward_axis[0]))
            if align < -0.20:
                continue
            choices.append((align, -lateral, neighbor, step_norm))

        if not choices:
            break
        choices.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_align, _best_lateral, next_node, step_norm = choices[0]
        if best_align < 0.05 and len(head_to_tail) >= 4:
            break

        head_to_tail.append(next_node)
        visited.add(next_node)
        travelled += float(step_norm)
        prev, current = current, next_node
        if travelled >= max_pixels and len(head_to_tail) >= 4:
            break

    if len(head_to_tail) < 2:
        return []
    tail_to_head = list(reversed(head_to_tail))
    if _path_alignment_near_head(tail_to_head, direction) < -0.05:
        return []
    return tail_to_head


def _trace_skeleton_ray(
    start: Pixel,
    component_set: set[Pixel],
    adj: Dict[Pixel, List[Pixel]],
    initial_axis: np.ndarray,
    max_pixels: float,
) -> List[Pixel]:
    """Follow one locally continuous pipe branch from an arrow anchor."""
    axis = np.asarray(initial_axis, dtype=np.float32)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-6 or start not in component_set:
        return []
    axis /= axis_norm
    heading = axis.copy()
    path: List[Pixel] = [start]
    visited: set[Pixel] = {start}
    previous: Pixel | None = None
    current = start
    travelled = 0.0
    max_steps = max(12, int(round(max_pixels * 2.5)))

    for _ in range(max_steps):
        choices: List[Tuple[float, float, float, Pixel, np.ndarray]] = []
        for neighbor in adj.get(current, []):
            if neighbor == previous or neighbor in visited or neighbor not in component_set:
                continue
            step = np.asarray(
                [float(neighbor[1] - current[1]), float(neighbor[0] - current[0])],
                dtype=np.float32,
            )
            step_norm = float(np.linalg.norm(step))
            if step_norm <= 1e-6:
                continue
            step_unit = step / step_norm
            heading_align = float(np.dot(step_unit, heading))
            anchor_align = float(np.dot(step_unit, axis))
            if len(path) <= 4 and anchor_align < -0.15:
                continue
            if len(path) > 4 and heading_align < -0.30:
                continue
            rel = np.asarray(
                [float(neighbor[1] - start[1]), float(neighbor[0] - start[0])],
                dtype=np.float32,
            )
            outward = float(np.dot(rel, axis))
            anchor_weight = float(np.clip(1.0 - travelled / max(max_pixels, 1.0), 0.0, 1.0)) * 0.28
            continuity_weight = 1.0 - anchor_weight
            score = continuity_weight * heading_align + anchor_weight * anchor_align
            if outward < -2.0:
                score -= 0.35
            branch_penalty = 0.025 * float(max(0, len(adj.get(neighbor, [])) - 3))
            score -= branch_penalty
            choices.append((score, heading_align, anchor_align, neighbor, step_unit))

        if not choices:
            break
        choices.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        best_score, best_heading_align, _best_anchor_align, next_node, step_unit = choices[0]
        if best_score < 0.05 and len(path) >= 4:
            break
        if best_heading_align < 0.05 and len(choices) > 1 and len(path) >= 5:
            break

        step_length = float(
            np.hypot(float(next_node[1] - current[1]), float(next_node[0] - current[0]))
        )
        path.append(next_node)
        visited.add(next_node)
        travelled += step_length
        previous, current = current, next_node
        blended = heading * 0.72 + step_unit * 0.28
        blended_norm = float(np.linalg.norm(blended))
        heading = blended / blended_norm if blended_norm > 1e-6 else step_unit
        if travelled >= max_pixels:
            break
    return path


def _trace_candidate_centerline(
    head_node: Pixel,
    component_nodes: Sequence[Pixel] | set[Pixel],
    adj: Dict[Pixel, List[Pixel]],
    direction: Tuple[float, float],
    max_pixels: float,
) -> List[Pixel]:
    """Trace the pipe through both sides of an inline arrow head.

    The returned sequence is ordered in the arrow's flow direction.  At a
    junction each ray selects the locally straight continuation instead of
    flooding every branch.
    """
    axis = np.asarray(direction, dtype=np.float32)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-6:
        return []
    axis /= axis_norm
    component_set = component_nodes if isinstance(component_nodes, set) else set(component_nodes)
    forward = _trace_skeleton_ray(head_node, component_set, adj, axis, max_pixels)
    backward = _trace_skeleton_ray(head_node, component_set, adj, -axis, max_pixels)
    if not forward and not backward:
        return []
    combined = [*reversed(backward[1:]), *forward]
    deduped: List[Pixel] = []
    for node in combined:
        if not deduped or node != deduped[-1]:
            deduped.append(node)
    return deduped


def _candidate_vector_cell_path(
    candidate: ArrowCandidate,
    A: int,
    B: int,
    H: int,
    W: int,
    length_pixels: float,
) -> List[Cell]:
    vec = np.asarray(candidate.direction, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return []
    unit = vec / norm
    tip_x = float(candidate.tip[0])
    tip_y = float(candidate.tip[1])
    tail_x = float(np.clip(tip_x - unit[0] * float(length_pixels), 0.0, max(0.0, float(W - 1))))
    tail_y = float(np.clip(tip_y - unit[1] * float(length_pixels), 0.0, max(0.0, float(H - 1))))
    tail_pixel = (int(round(tail_y)), int(round(tail_x)))
    head_pixel = (int(round(np.clip(tip_y, 0.0, max(0.0, float(H - 1))))), int(round(np.clip(tip_x, 0.0, max(0.0, float(W - 1))))))
    path = _grid_traversal_cells(tail_pixel, head_pixel, A, B, H, W)
    deduped: List[Cell] = []
    for cell in path:
        if not deduped or cell != deduped[-1]:
            deduped.append(cell)
    return deduped


def _build_candidate_object_path(
    candidate: ArrowCandidate,
    candidate_index: int,
    candidate_nodes: Sequence[Pixel],
    component_nodes: Sequence[Pixel] | set[Pixel],
    adj: Dict[Pixel, List[Pixel]],
    dist_map: np.ndarray,
    component_candidate_count: int,
    A: int,
    B: int,
    H: int,
    W: int,
) -> ArrowObjectPath | None:
    bx1, by1, bx2, by2 = candidate.bbox
    max_dist_sq = float(max(bx2 - bx1, by2 - by1, 1) ** 2) * 2.0
    head_node = _pick_component_head_node(candidate_nodes, candidate, max_dist_sq=max_dist_sq)
    if head_node is None:
        return None

    cell_span = float(max(H / float(A), W / float(B), 1.0))
    endpoint_count = sum(1 for node in component_nodes if len(adj.get(node, [])) <= 1)
    branch_count = sum(1 for node in component_nodes if len(adj.get(node, [])) > 2)
    complex_component = component_candidate_count >= 4 or endpoint_count > 10 or branch_count > 24
    if complex_component:
        local_pixel_path = _trace_local_tail_to_head_path(
            head_node,
            component_nodes,
            adj,
            candidate.direction,
            max_pixels=float(cell_span * 5.2),
        )
        if len(local_pixel_path) >= 2:
            raw_cell_path = _cell_path_from_pixel_path(local_pixel_path, A, B, H, W, expand_diagonal=False)
            vote_cell_path = _cell_path_from_pixel_path(local_pixel_path, A, B, H, W, expand_diagonal=True)
            if len(raw_cell_path) >= 2 and len(vote_cell_path) >= 2:
                return ArrowObjectPath(
                    component_id=int(candidate.component_id),
                    candidate_index=int(candidate_index),
                    bbox=tuple(int(v) for v in candidate.bbox),
                    head_cell=vote_cell_path[-1],
                    tail_cell=vote_cell_path[0],
                    raw_cell_path=tuple(raw_cell_path),
                    vote_cell_path=tuple(vote_cell_path),
                    score=float(np.clip(0.54 + 0.46 * float(max(candidate.score, 0.0)), 0.05, 1.2)),
                    direction_code=int(_direction_from_vector(candidate.direction)),
                )
        vector_path = _candidate_vector_cell_path(
            candidate,
            A,
            B,
            H,
            W,
            length_pixels=float(cell_span * 3.8),
        )
        if len(vector_path) >= 2:
            return ArrowObjectPath(
                component_id=int(candidate.component_id),
                candidate_index=int(candidate_index),
                bbox=tuple(int(v) for v in candidate.bbox),
                head_cell=vector_path[-1],
                tail_cell=vector_path[0],
                raw_cell_path=tuple(vector_path),
                vote_cell_path=tuple(vector_path),
                score=float(np.clip(0.44 + 0.56 * float(max(candidate.score, 0.0)), 0.05, 1.2)),
                direction_code=int(_direction_from_vector(candidate.direction)),
            )

    dist, parent = _bfs_within_component(adj, component_nodes, head_node)
    if not dist:
        return None

    endpoints = [
        node
        for node in component_nodes
        if node in dist and node != head_node and len(adj.get(node, [])) <= 1
    ]
    if not endpoints:
        ordered_nodes = sorted(
            (node for node in component_nodes if node in dist and node != head_node),
            key=lambda node: dist[node],
            reverse=True,
        )
        endpoints = ordered_nodes[: min(8, len(ordered_nodes))]
    if not endpoints:
        return None

    best_path: List[Pixel] | None = None
    best_score = -1.0
    candidate_weight = float(max(candidate.score, 0.0))
    direction_unit = np.asarray(candidate.direction, dtype=np.float32)
    direction_norm = float(np.linalg.norm(direction_unit))
    if direction_norm > 1e-6:
        direction_unit = direction_unit / direction_norm
    else:
        direction_unit = np.array([0.0, 0.0], dtype=np.float32)
    head_xy = np.array([float(head_node[1]), float(head_node[0])], dtype=np.float32)
    projection_norm = max(float(cell_span * 8.0), 1.0)
    lateral_norm = max(float(cell_span * 3.0), 1.0)
    for endpoint in endpoints:
        path_tail_to_head = _path_from_parent(parent, endpoint)
        if len(path_tail_to_head) < 2:
            continue
        tail_head_score = _endpoint_head_score(path_tail_to_head, dist_map, "start")
        head_head_score = _endpoint_head_score(path_tail_to_head, dist_map, "end")
        align = _path_alignment_near_head(path_tail_to_head, candidate.direction)
        if align < -0.10:
            continue
        endpoint_xy = np.array([float(endpoint[1]), float(endpoint[0])], dtype=np.float32)
        tail_to_head = head_xy - endpoint_xy
        euclidean_len = float(np.linalg.norm(tail_to_head))
        projection = float(np.dot(tail_to_head, direction_unit)) if direction_norm > 1e-6 else 0.0
        if projection < -float(cell_span * 0.5):
            continue
        straight_score = float(np.clip(projection / max(euclidean_len, 1e-6), 0.0, 1.0)) if euclidean_len > 1e-6 else 0.0
        projection_score = float(np.clip(projection / projection_norm, 0.0, 1.0))
        lateral = abs(float(tail_to_head[0] * direction_unit[1] - tail_to_head[1] * direction_unit[0])) if direction_norm > 1e-6 else 0.0
        lateral_penalty = float(np.clip(lateral / lateral_norm, 0.0, 1.0))
        length_score = float(np.clip((len(path_tail_to_head) - 2) / 12.0, 0.0, 1.0))
        branching_bonus = 0.12 if component_candidate_count > 1 and head_head_score >= tail_head_score else 0.0
        total = (
            0.32 * candidate_weight
            + 0.20 * max(0.0, head_head_score)
            + 0.16 * max(0.0, align)
            + 0.12 * length_score
            + 0.18 * projection_score
            + 0.12 * straight_score
            - 0.18 * lateral_penalty
            - 0.14 * max(0.0, tail_head_score)
            + branching_bonus
        )
        if total > best_score:
            best_score = float(total)
            best_path = path_tail_to_head

    if best_path is None:
        return None

    raw_cell_path = _cell_path_from_pixel_path(best_path, A, B, H, W, expand_diagonal=False)
    expanded_cell_path = _cell_path_from_pixel_path(best_path, A, B, H, W, expand_diagonal=True)
    if len(raw_cell_path) < 2 or len(expanded_cell_path) < 2:
        return None

    line_cell_path = _grid_traversal_cells(best_path[0], best_path[-1], A, B, H, W)
    prefer_line = (
        component_candidate_count <= 1
        and len(line_cell_path) >= 2
        and _is_locally_supported_path(line_cell_path, expanded_cell_path, tolerance=0)
        and abs(len(line_cell_path) - len(expanded_cell_path)) <= max(2, int(round(len(expanded_cell_path) * 0.35)))
    )
    vote_cell_path = line_cell_path if prefer_line else expanded_cell_path
    if len(vote_cell_path) < 2:
        return None

    return ArrowObjectPath(
        component_id=int(candidate.component_id),
        candidate_index=int(candidate_index),
        bbox=tuple(int(v) for v in candidate.bbox),
        head_cell=vote_cell_path[-1],
        tail_cell=vote_cell_path[0],
        raw_cell_path=tuple(raw_cell_path),
        vote_cell_path=tuple(vote_cell_path),
        score=float(np.clip(best_score, 0.05, 1.5)),
        direction_code=int(_direction_from_vector(candidate.direction)),
    )


_DIR_TO_VEC: Dict[int, Tuple[float, float]] = {
    1: (1.0, 0.0),
    2: (0.0, 1.0),
    3: (-1.0, 0.0),
    4: (0.0, -1.0),
}


def _resolve_cell_direction(
    cell: Cell,
    cell_votes: Dict[DirectionCode, float],
    transition_votes: Dict[Tuple[Cell, Cell], float],
    overlap_count: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    axis_conf: np.ndarray,
    A: int,
    B: int,
) -> DirectionCode:
    best_direction = 0
    best_key: Tuple[float, float, float, float] | None = None
    overlap = int(overlap_count[cell])
    for direction, vote in cell_votes.items():
        nxt = _next_cell_from_direction(cell, int(direction), A, B)
        transition = float(transition_votes.get((cell, nxt), 0.0)) if nxt is not None else 0.0
        axis_bonus = 0.0
        local_conf = float(axis_conf[cell])
        if local_conf > 0.05 and int(direction) in _DIR_TO_VEC:
            axis_vec = np.array([float(axis_x[cell]), float(axis_y[cell])], dtype=np.float32)
            dir_vec = np.array(_DIR_TO_VEC[int(direction)], dtype=np.float32)
            axis_bonus = abs(float(np.dot(axis_vec, dir_vec)))
        overlap_bonus = 0.20 * float(min(overlap, 3))
        continuity = transition * (1.35 + overlap_bonus)
        score = continuity + 0.85 * float(vote) + 0.20 * axis_bonus
        if overlap > 1 and transition <= 1e-6:
            score -= 0.35
        key = (score, transition, float(vote), axis_bonus)
        if best_key is None or key > best_key:
            best_key = key
            best_direction = int(direction)
    return int(best_direction)


def _cell_components(mask: np.ndarray) -> List[List[Cell]]:
    A, B = mask.shape
    visited = np.zeros((A, B), dtype=np.uint8)
    components: List[List[Cell]] = []
    for i in range(A):
        for j in range(B):
            if mask[i, j] != 1 or visited[i, j] == 1:
                continue
            queue = deque([(i, j)])
            visited[i, j] = 1
            component: List[Cell] = []
            while queue:
                cell = queue.popleft()
                component.append(cell)
                for nb in _neighbors4(cell, A, B):
                    if mask[nb] == 1 and visited[nb] == 0:
                        visited[nb] = 1
                        queue.append(nb)
            components.append(component)
    return components


def _shortest_path_in_component(component_set: set[Cell], start: Cell, goal: Cell, A: int, B: int) -> List[Cell]:
    parent: Dict[Cell, Cell | None] = {start: None}
    queue: deque[Cell] = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        for nb in _neighbors4(cur, A, B):
            if nb not in component_set or nb in parent:
                continue
            parent[nb] = cur
            queue.append(nb)
    if goal not in parent:
        return [start]
    path: List[Cell] = []
    cur: Cell | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _component_backbone_path(component: Sequence[Cell], A: int, B: int) -> List[Cell]:
    component_set = set(component)
    start = component[0]

    def farthest(src: Cell) -> Tuple[Cell, Dict[Cell, int]]:
        dist: Dict[Cell, int] = {src: 0}
        queue: deque[Cell] = deque([src])
        while queue:
            cur = queue.popleft()
            for nb in _neighbors4(cur, A, B):
                if nb not in component_set or nb in dist:
                    continue
                dist[nb] = dist[cur] + 1
                queue.append(nb)
        far = max(dist, key=dist.get)
        return far, dist

    u, _ = farthest(start)
    v, _ = farthest(u)
    return _shortest_path_in_component(component_set, u, v, A, B)


def _distance_map_in_component(component_set: set[Cell], start: Cell, A: int, B: int) -> Dict[Cell, int]:
    dist: Dict[Cell, int] = {start: 0}
    queue: deque[Cell] = deque([start])
    while queue:
        cur = queue.popleft()
        for nb in _neighbors4(cur, A, B):
            if nb not in component_set or nb in dist:
                continue
            dist[nb] = dist[cur] + 1
            queue.append(nb)
    return dist


def _path_match_score(matrix: np.ndarray, path: Sequence[Cell]) -> Tuple[int, int]:
    matches = 0
    mismatches = 0
    last_direction = 0
    for src, dst in zip(path[:-1], path[1:]):
        direction = _direction_from_step(src, dst)
        if direction == 0:
            continue
        current = int(matrix[src])
        if current != 0:
            if current == direction:
                matches += 1
            else:
                mismatches += 1
        last_direction = direction
    if last_direction != 0 and path:
        tail_val = int(matrix[path[-1]])
        if tail_val != 0:
            if tail_val == last_direction:
                matches += 1
            else:
                mismatches += 1
    return matches, mismatches


def _fill_zero_cells_along_path(
    matrix: np.ndarray,
    path: Sequence[Cell],
    fillable_mask: np.ndarray | None = None,
) -> int:
    filled = 0
    last_direction = 0
    for src, dst in zip(path[:-1], path[1:]):
        direction = _direction_from_step(src, dst)
        if direction == 0:
            continue
        if int(matrix[src]) == 0:
            if fillable_mask is not None and int(fillable_mask[src]) != 1:
                last_direction = direction
                continue
            matrix[src] = np.uint8(direction)
            filled += 1
        last_direction = direction
    if last_direction != 0 and path and int(matrix[path[-1]]) == 0:
        if fillable_mask is not None and int(fillable_mask[path[-1]]) != 1:
            return filled
        matrix[path[-1]] = np.uint8(last_direction)
        filled += 1
    return filled


def fill_component_gaps(
    matrix: np.ndarray,
    support_mask: np.ndarray,
    fillable_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, int]:
    """Fill only strict one-cell gaps inside each arrow object."""
    assisted = matrix.astype(np.uint8).copy()
    A, B = assisted.shape
    total_filled = 0
    for component in _cell_components(support_mask):
        component_set = set(component)
        candidate_cells = [cell for cell in component if int(assisted[cell]) == 0]
        for cell in candidate_cells:
            if fillable_mask is not None and int(fillable_mask[cell]) != 1:
                continue

            labeled_neighbors = [
                nb for nb in _neighbors4(cell, A, B)
                if nb in component_set and int(assisted[nb]) != 0
            ]
            if len(labeled_neighbors) < 2:
                continue

            best_path: List[Cell] | None = None
            best_score: Tuple[int, int, int] | None = None
            for idx, src in enumerate(labeled_neighbors):
                for dst in labeled_neighbors[idx + 1:]:
                    path = [src, cell, dst]
                    reverse = [dst, cell, src]
                    score_f = _path_match_score(assisted, path)
                    score_r = _path_match_score(assisted, reverse)
                    chosen = path
                    chosen_score = (int(score_f[0]), -int(score_f[1]), 0)
                    if score_r > score_f:
                        chosen = reverse
                        chosen_score = (int(score_r[0]), -int(score_r[1]), 0)
                    if best_score is None or chosen_score > best_score:
                        best_score = chosen_score
                        best_path = chosen

            if best_path is None:
                continue
            total_filled += _fill_zero_cells_along_path(assisted, best_path, fillable_mask=fillable_mask)
    return assisted, total_filled


def _distance_map_to_outlet(support_mask: np.ndarray, outlet: Cell) -> np.ndarray:
    A, B = support_mask.shape
    if not (0 <= outlet[0] < A and 0 <= outlet[1] < B):
        raise ValueError("outlet is outside the matrix")

    support = support_mask.astype(bool).copy()
    support[outlet] = True
    dist = np.full((A, B), -1, dtype=np.int32)
    queue: deque[Cell] = deque([outlet])
    dist[outlet] = 0
    while queue:
        cur = queue.popleft()
        for nb in _neighbors4(cur, A, B):
            if not support[nb] or dist[nb] != -1:
                continue
            dist[nb] = dist[cur] + 1
            queue.append(nb)
    return dist


def _best_downhill_neighbor(
    matrix: np.ndarray,
    dist: np.ndarray,
    cell: Cell,
    outlet: Cell,
) -> Cell | None:
    if dist[cell] <= 0:
        return None
    best_neighbor: Cell | None = None
    best_key: Tuple[int, int] | None = None
    for nb in _neighbors4(cell, int(matrix.shape[0]), int(matrix.shape[1])):
        if dist[nb] == -1 or dist[nb] >= dist[cell]:
            continue
        key = (int(dist[nb]), 0 if int(matrix[nb]) != 0 or nb == outlet else 1)
        if best_key is None or key < best_key:
            best_key = key
            best_neighbor = nb
    return best_neighbor


def _next_cell_from_direction(cell: Cell, direction: int, A: int, B: int) -> Cell | None:
    mapping = {
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
        4: (-1, 0),
    }
    step = mapping.get(int(direction))
    if step is None:
        return None
    nxt = (cell[0] + step[0], cell[1] + step[1])
    if not (0 <= nxt[0] < A and 0 <= nxt[1] < B):
        return None
    return nxt


def _path_reaches_outlet(matrix: np.ndarray, route_mask: np.ndarray, start: Cell, outlet: Cell) -> bool:
    A, B = matrix.shape
    if start != outlet and int(route_mask[start]) != 1:
        return False
    current = start
    seen: set[Cell] = set()
    for _ in range(A * B):
        if current == outlet:
            return True
        if current in seen:
            return False
        seen.add(current)
        value = int(matrix[current])
        if value == 0:
            return False
        nxt = _next_cell_from_direction(current, value, A, B)
        if nxt is None:
            return False
        if nxt != outlet and int(route_mask[nxt]) != 1:
            return False
        current = nxt
    return False


def recommend_outlet_candidates(
    matrix: np.ndarray,
    support_mask: np.ndarray | None = None,
    *,
    limit: int = 12,
) -> List[Tuple[Cell, int]]:
    """Rank natural flow terminals by the number of occupied cells draining to them."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    route = (
        (support_mask > 0).astype(np.uint8)
        if support_mask is not None and support_mask.shape == matrix.shape
        else (matrix > 0).astype(np.uint8)
    )
    route = np.maximum(route, (matrix > 0).astype(np.uint8))
    A, B = matrix.shape
    terminal_counts: DefaultDict[Cell, int] = defaultdict(int)
    terminal_cache: Dict[Cell, Cell | None] = {}

    for raw_start in np.argwhere(route > 0):
        start = (int(raw_start[0]), int(raw_start[1]))
        current = start
        path: List[Cell] = []
        seen: set[Cell] = set()
        terminal: Cell | None = None
        last_directed: Cell | None = None
        while current not in seen:
            cached = terminal_cache.get(current)
            if current in terminal_cache:
                terminal = cached
                break
            seen.add(current)
            path.append(current)
            direction = int(matrix[current])
            if direction in (1, 2, 3, 4):
                last_directed = current
            nxt = _next_cell_from_direction(current, direction, A, B)
            if nxt is None or int(route[nxt]) != 1:
                terminal = last_directed
                break
            current = nxt
        for cell in path:
            terminal_cache[cell] = terminal
        if terminal is not None:
            terminal_counts[terminal] += 1

    ranked = sorted(
        terminal_counts.items(),
        key=lambda item: (-int(item[1]), int(item[0][0]), int(item[0][1])),
    )
    return [(cell, int(count)) for cell, count in ranked[: max(1, int(limit))]]


def _forward_progress_steps(
    matrix: np.ndarray,
    route_mask: np.ndarray,
    start: Cell,
    outlet: Cell,
    max_steps: int = 4,
) -> int:
    """Count how many valid existing steps the current direction survives."""
    A, B = matrix.shape
    current = start
    seen: set[Cell] = set()
    steps = 0
    for _ in range(max_steps):
        if current == outlet:
            return max_steps
        if current in seen:
            return steps
        seen.add(current)
        value = int(matrix[current])
        if value == 0:
            return steps
        nxt = _next_cell_from_direction(current, value, A, B)
        if nxt is None:
            return steps
        if nxt != outlet and int(route_mask[nxt]) != 1:
            return steps
        steps += 1
        current = nxt
    return steps


def _route_degree(route_mask: np.ndarray, cell: Cell, outlet: Cell | None = None) -> int:
    A, B = route_mask.shape
    degree = 0
    for nb in _neighbors4(cell, A, B):
        if int(route_mask[nb]) == 1 or (outlet is not None and nb == outlet):
            degree += 1
    return degree


def _candidate_starts_for_component(
    component: Sequence[Cell],
    matrix: np.ndarray,
    route_mask: np.ndarray,
    head_tip_mask: np.ndarray,
    outlet: Cell,
) -> List[Cell]:
    candidates: List[Tuple[Tuple[int, int, int, int], Cell]] = []
    seen: set[Cell] = set()
    for cell in component:
        if int(route_mask[cell]) != 1:
            continue
        progress = _forward_progress_steps(matrix, route_mask, cell, outlet, max_steps=3)
        if progress > 1:
            continue
        priority = 0 if int(head_tip_mask[cell]) == 1 else 1 if _route_degree(route_mask, cell) <= 1 else 2
        key = (priority, progress, _route_degree(route_mask, cell), cell[0] + cell[1])
        if cell not in seen:
            candidates.append((key, cell))
            seen.add(cell)
    if not candidates:
        for cell in component:
            if int(route_mask[cell]) != 1:
                continue
            priority = 0 if int(head_tip_mask[cell]) == 1 else 3
            key = (priority, 9, _route_degree(route_mask, cell), cell[0] + cell[1])
            if cell not in seen:
                candidates.append((key, cell))
                seen.add(cell)
    candidates.sort(key=lambda item: item[0])
    return [cell for _, cell in candidates]


def _direction_change_cost(
    matrix: np.ndarray,
    cell: Cell,
    nxt: Cell,
    head_tip_mask: np.ndarray,
    head_halo_mask: np.ndarray,
) -> int:
    desired = _direction_from_step(cell, nxt)
    current = int(matrix[cell])
    if current == desired:
        return 0
    if current == 0:
        return 120
    if int(head_tip_mask[cell]) == 1:
        return 1
    if int(head_halo_mask[cell]) == 1:
        return 3
    return 7


def _find_min_cost_connection_path(
    matrix: np.ndarray,
    start: Cell,
    outlet: Cell,
    route_mask: np.ndarray,
    head_tip_mask: np.ndarray,
    head_halo_mask: np.ndarray,
    optional_zero_mask: np.ndarray | None = None,
    allow_one_zero: bool = False,
) -> Tuple[List[Cell], int] | None:
    if start == outlet:
        return [start], 0

    A, B = matrix.shape
    start_state = (start, 0)
    best_cost: Dict[Tuple[Cell, int], int] = {start_state: 0}
    parent: Dict[Tuple[Cell, int], Tuple[Cell, int] | None] = {start_state: None}
    heap: List[Tuple[int, int, Tuple[Cell, int]]] = [(0, 0, start_state)]

    while heap:
        total_cost, steps, state = heapq.heappop(heap)
        cell, zero_used = state
        if total_cost != best_cost.get(state, 10**18):
            continue
        if cell == outlet:
            path: List[Cell] = []
            cur: Tuple[Cell, int] | None = state
            while cur is not None:
                path.append(cur[0])
                cur = parent[cur]
            path.reverse()
            return path, zero_used

        for nb in _neighbors4(cell, A, B):
            entering_zero = False
            if nb != outlet and int(route_mask[nb]) != 1:
                if not allow_one_zero or zero_used >= 1:
                    continue
                if optional_zero_mask is None or int(optional_zero_mask[nb]) != 1:
                    continue
                entering_zero = True

            next_zero_used = zero_used + (1 if entering_zero else 0)
            step_cost = _direction_change_cost(matrix, cell, nb, head_tip_mask, head_halo_mask)
            if entering_zero:
                step_cost += 1000
            next_state = (nb, next_zero_used)
            next_steps = steps + 1
            next_cost = total_cost + step_cost * 100 + 1
            if next_cost < best_cost.get(next_state, 10**18):
                best_cost[next_state] = next_cost
                parent[next_state] = state
                heapq.heappush(heap, (next_cost, next_steps, next_state))
    return None


def _apply_connection_path(
    matrix: np.ndarray,
    path: Sequence[Cell],
    head_tip_mask: np.ndarray,
    head_halo_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, int]]:
    assisted = matrix.astype(np.uint8).copy()
    stats = {
        "head_fill": 0,
        "head_neighbor_fill": 0,
        "component_fill": 0,
        "zero_fill": 0,
    }
    for src, dst in zip(path[:-1], path[1:]):
        desired = _direction_from_step(src, dst)
        if desired == 0:
            continue
        current = int(assisted[src])
        if current == desired:
            continue
        assisted[src] = np.uint8(desired)
        if current == 0:
            stats["zero_fill"] += 1
        elif int(head_tip_mask[src]) == 1:
            stats["head_fill"] += 1
        elif int(head_halo_mask[src]) == 1:
            stats["head_neighbor_fill"] += 1
        else:
            stats["component_fill"] += 1
    return assisted, stats


def _apply_guided_updates(
    matrix: np.ndarray,
    dist: np.ndarray,
    target_mask: np.ndarray,
    outlet: Cell,
    overwrite_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, int]:
    assisted = matrix.astype(np.uint8).copy()
    A, B = assisted.shape
    candidates = [(int(dist[i, j]), (i, j)) for i in range(A) for j in range(B) if target_mask[i, j] == 1 and dist[i, j] > 0]
    candidates.sort(reverse=True)
    changed = 0
    for _d, cell in candidates:
        best_neighbor = _best_downhill_neighbor(assisted, dist, cell, outlet)
        if best_neighbor is None:
            continue
        desired = _direction_from_step(cell, best_neighbor)
        current = int(assisted[cell])
        may_overwrite = overwrite_mask is not None and int(overwrite_mask[cell]) == 1
        if current != 0 and not may_overwrite:
            continue
        if current == desired:
            continue
        assisted[cell] = np.uint8(desired)
        changed += 1
    return assisted, changed


def _select_stage2_neighbor_mask(
    matrix: np.ndarray,
    dist: np.ndarray,
    route_mask: np.ndarray,
    tip_cells: Sequence[Cell],
    halo_mask: np.ndarray,
    allow_mask: np.ndarray,
    outlet: Cell,
) -> np.ndarray:
    selected = np.zeros_like(allow_mask, dtype=np.uint8)
    A, B = matrix.shape
    for tip in tip_cells:
        best_cell: Cell | None = None
        best_key: Tuple[int, int, int] | None = None
        for cell in _neighbors4(tip, A, B):
            if int(halo_mask[cell]) != 1 or int(allow_mask[cell]) != 1 or dist[cell] <= 0:
                continue
            if _forward_progress_steps(matrix, route_mask, cell, outlet, max_steps=3) > 1:
                continue
            best_neighbor = _best_downhill_neighbor(matrix, dist, cell, outlet)
            if best_neighbor is None:
                continue
            desired = _direction_from_step(cell, best_neighbor)
            current = int(matrix[cell])
            # Around arrow heads, prefer redirecting an existing supported cell over
            # synthesizing a new one in empty space.
            change_cost = 0 if current != 0 and current != desired else 1 if current == 0 else 2
            key = (change_cost, int(dist[cell]), abs(cell[0] - tip[0]) + abs(cell[1] - tip[1]))
            if best_key is None or key < best_key:
                best_key = key
                best_cell = cell
        if best_cell is not None:
            selected[best_cell] = 1
    return selected.astype(np.uint8)


def guide_zero_cells_to_outlet(
    matrix: np.ndarray,
    support_mask: np.ndarray,
    outlet: Cell,
    fillable_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Guide zero cells toward the outlet without changing existing non-zero directions."""
    assisted = matrix.astype(np.uint8).copy()
    A, B = assisted.shape
    if not (0 <= outlet[0] < A and 0 <= outlet[1] < B):
        raise ValueError("outlet is outside the matrix")

    support = support_mask.astype(bool).copy()
    support[outlet] = True

    dist = np.full((A, B), -1, dtype=np.int32)
    queue: deque[Cell] = deque()
    dist[outlet] = 0
    queue.append(outlet)

    while queue:
        cur = queue.popleft()
        for nb in _neighbors4(cur, A, B):
            if not support[nb] or dist[nb] != -1:
                continue
            dist[nb] = dist[cur] + 1
            queue.append(nb)

    fill_candidates = [(int(dist[i, j]), (i, j)) for i in range(A) for j in range(B) if dist[i, j] > 0]
    fill_candidates.sort(reverse=True)

    filled = 0
    for _, cell in fill_candidates:
        if fillable_mask is not None and int(fillable_mask[cell]) != 1:
            continue
        if int(assisted[cell]) != 0:
            continue
        row, col = cell
        best_neighbor: Cell | None = None
        best_key: Tuple[int, int] | None = None
        for nb in _neighbors4(cell, A, B):
            if dist[nb] == -1 or dist[nb] >= dist[cell]:
                continue
            key = (int(dist[nb]), 0 if int(assisted[nb]) != 0 or nb == outlet else 1)
            if best_key is None or key < best_key:
                best_key = key
                best_neighbor = nb
        if best_neighbor is None:
            continue
        assisted[row, col] = np.uint8(_direction_from_step(cell, best_neighbor))
        filled += 1

    return assisted, dist, filled


def _trace_unreachable_route(matrix: np.ndarray, route_mask: np.ndarray, start: Cell, outlet: Cell) -> Tuple[List[Cell], List[Cell]]:
    A, B = matrix.shape
    current = start
    path: List[Cell] = []
    seen: Dict[Cell, int] = {}
    for _ in range(A * B):
        if current == outlet:
            return path, []
        if not (0 <= current[0] < A and 0 <= current[1] < B):
            return path, []
        if int(route_mask[current]) != 1:
            return path, []
        if current in seen:
            return path, path[seen[current]:]
        seen[current] = len(path)
        path.append(current)
        nxt = _next_cell_from_direction(current, int(matrix[current]), A, B)
        if nxt is None:
            return path, []
        if nxt != outlet and int(route_mask[nxt]) != 1:
            return path, []
        current = nxt
    return path, []


def _repair_unreachable_cells_to_outlet(
    matrix: np.ndarray,
    support_mask: np.ndarray,
    outlet: Cell,
    starts: Sequence[Cell],
) -> Tuple[np.ndarray, int]:
    """Repair only route cells that currently fail to reach the outlet."""
    assisted = matrix.astype(np.uint8).copy()
    dist = _distance_map_to_outlet(support_mask.astype(np.uint8), outlet)
    route_mask = (assisted > 0).astype(np.uint8)
    changed = 0
    seen_starts: set[Cell] = set()

    for raw_start in starts:
        start = (int(raw_start[0]), int(raw_start[1]))
        if start in seen_starts or start == outlet:
            continue
        seen_starts.add(start)
        if int(route_mask[start]) != 1:
            continue

        for _attempt in range(8):
            if _path_reaches_outlet(assisted, route_mask, start, outlet):
                break
            path, cycle = _trace_unreachable_route(assisted, route_mask, start, outlet)
            candidates = cycle if cycle else list(reversed(path))
            candidates = [cell for cell in candidates if dist[cell] > 0 and not _path_reaches_outlet(assisted, route_mask, cell, outlet)]
            candidates.sort(key=lambda cell: (int(dist[cell]), path.index(cell) if cell in path else 10**9))

            repaired = False
            for cell in candidates:
                best_neighbor = _best_downhill_neighbor(assisted, dist, cell, outlet)
                if best_neighbor is None:
                    continue
                if best_neighbor != outlet and int(route_mask[best_neighbor]) != 1:
                    continue
                desired = _direction_from_step(cell, best_neighbor)
                if desired == 0:
                    continue
                if int(assisted[cell]) != desired:
                    assisted[cell] = np.uint8(desired)
                    route_mask = (assisted > 0).astype(np.uint8)
                    changed += 1
                repaired = True
                break
            if not repaired:
                break
    return assisted.astype(np.uint8), int(changed)


def assist_direction_matrix(
    img: np.ndarray,
    matrix: np.ndarray,
    A: int,
    outlet: Cell | None = None,
    occ: np.ndarray | None = None,
    path_occ: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    low_conf_threshold: float = 0.55,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], np.ndarray]:
    """Connect toward the outlet by minimizing direction edits on the existing pipe graph."""
    if img is None:
        raise ValueError("img must not be None")
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    if int(matrix.shape[0]) != int(A):
        raise ValueError("matrix row count does not match A")
    B = int(matrix.shape[1])
    cfg = FeatureConfig(A=A, use_edges=True, use_gray=False)
    feat = build_features(img, cfg)
    matrix_mask = (matrix > 0).astype(np.uint8)
    path_mask = (path_occ > 0).astype(np.uint8) if path_occ is not None else np.zeros((A, B), dtype=np.uint8)
    occ_mask = (occ > 0).astype(np.uint8) if occ is not None else np.zeros((A, B), dtype=np.uint8)
    visual_support = np.maximum(matrix_mask, path_mask).astype(np.uint8)
    # Routing is performed on the existing occupied direction graph first, so
    # turning pipes are respected and empty cells are not promoted into new pipes.
    route_mask = matrix_mask.astype(np.uint8)
    if int(np.count_nonzero(route_mask)) == 0:
        route_mask = visual_support.astype(np.uint8)
    if int(np.count_nonzero(route_mask)) == 0:
        route_mask = occ_mask.astype(np.uint8)

    support_mask, fillable_mask, bridge_ids = build_guidance_support_mask(
        feat["bw"],
        A,
        B,
        occ=occ,
        matrix=matrix,
        path_occ=path_occ,
        confidence=confidence,
        low_conf_threshold=low_conf_threshold,
    )
    head_tip_mask, head_halo_mask, tip_cells = build_head_stage_data(feat["bw"], A, B)

    assisted = matrix.astype(np.uint8).copy()
    head_fill = 0
    head_neighbor_fill = 0
    component_fill = 0
    outlet_fill = 0
    cycle_fix = 0
    unresolved_mask = np.zeros((A, B), dtype=np.uint8)
    returned_support = np.maximum((assisted > 0).astype(np.uint8), visual_support.astype(np.uint8)).astype(np.uint8)

    if outlet is not None:
        zero_budget = 1
        ordered_starts: List[Cell] = []
        seen_starts: set[Cell] = set()
        for component in _cell_components(route_mask):
            for start in _candidate_starts_for_component(component, assisted, route_mask, head_tip_mask, outlet):
                if start not in seen_starts:
                    seen_starts.add(start)
                    ordered_starts.append(start)

        for start in ordered_starts:
            if start == outlet or int(route_mask[start]) != 1:
                continue
            if _path_reaches_outlet(assisted, route_mask, start, outlet):
                continue

            plan = _find_min_cost_connection_path(
                assisted,
                start,
                outlet,
                route_mask,
                head_tip_mask,
                head_halo_mask,
                optional_zero_mask=fillable_mask,
                allow_one_zero=False,
            )
            if plan is None and zero_budget > 0:
                plan = _find_min_cost_connection_path(
                    assisted,
                    start,
                    outlet,
                    route_mask,
                    head_tip_mask,
                    head_halo_mask,
                    optional_zero_mask=fillable_mask,
                    allow_one_zero=True,
                )
            if plan is None:
                continue

            path, zero_used = plan
            assisted, delta = _apply_connection_path(assisted, path, head_tip_mask, head_halo_mask)
            head_fill += int(delta["head_fill"])
            head_neighbor_fill += int(delta["head_neighbor_fill"])
            component_fill += int(delta["component_fill"])
            outlet_fill += int(delta["zero_fill"])
            if zero_used > 0 or int(delta["zero_fill"]) > 0:
                zero_budget = 0
            route_mask = (assisted > 0).astype(np.uint8)

        returned_support = np.maximum(route_mask, visual_support.astype(np.uint8)).astype(np.uint8)
        repair_starts: List[Cell] = []
        for component in _cell_components(route_mask):
            for start in _candidate_starts_for_component(component, assisted, route_mask, head_tip_mask, outlet):
                if start != outlet and not _path_reaches_outlet(assisted, route_mask, start, outlet):
                    repair_starts.append(start)
        assisted, cycle_fix = _repair_unreachable_cells_to_outlet(assisted, returned_support, outlet, repair_starts)
        route_mask = (assisted > 0).astype(np.uint8)
        returned_support = np.maximum(route_mask, returned_support).astype(np.uint8)

        for i in range(A):
            for j in range(B):
                cell = (i, j)
                if cell == outlet or int(route_mask[cell]) != 1:
                    continue
                if not _path_reaches_outlet(assisted, route_mask, cell, outlet):
                    unresolved_mask[cell] = 1

    stats = {
        "head_fill": int(head_fill),
        "head_neighbor_fill": int(head_neighbor_fill),
        "component_fill": int(component_fill),
        "outlet_fill": int(outlet_fill),
        "cycle_fix": int(cycle_fix),
        "eligible_fill": 0,
        "bridge_fill": int(np.count_nonzero(bridge_ids > 0)),
        "unresolved_count": int(np.count_nonzero(unresolved_mask)),
    }
    return assisted.astype(np.uint8), returned_support, stats, unresolved_mask.astype(np.uint8)


_DIR_TO_STEP: Dict[int, Tuple[int, int]] = {
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
    4: (-1, 0),
}


def _coarse_cell_for_fine(cell: Cell, src_shape: Tuple[int, int], dst_shape: Tuple[int, int]) -> Cell:
    src_a, src_b = src_shape
    dst_a, dst_b = dst_shape
    row = min(dst_a - 1, int((cell[0] * dst_a) // max(src_a, 1)))
    col = min(dst_b - 1, int((cell[1] * dst_b) // max(src_b, 1)))
    return (row, col)


def _dedupe_consecutive_cells(cells: Sequence[Cell]) -> List[Cell]:
    deduped: List[Cell] = []
    for cell in cells:
        if not deduped or deduped[-1] != cell:
            deduped.append(cell)
    return deduped


def _trace_direction_path(matrix: np.ndarray, start: Cell, outlet: Cell | None = None) -> Tuple[List[Cell], bool]:
    A, B = matrix.shape
    path: List[Cell] = [start]
    seen: set[Cell] = {start}
    current = start
    limit = max(1, A * B + 1)
    for _ in range(limit):
        if outlet is not None and current == outlet:
            return path, True
        direction = int(matrix[current])
        step = _DIR_TO_STEP.get(direction)
        if step is None:
            break
        nxt = (current[0] + step[0], current[1] + step[1])
        if not (0 <= nxt[0] < A and 0 <= nxt[1] < B):
            break
        path.append(nxt)
        if outlet is not None and nxt == outlet:
            return path, True
        if nxt in seen:
            break
        seen.add(nxt)
        current = nxt
    return path, False


def _coarse_edge_cost(
    src: Cell,
    dst: Cell,
    dir_votes: np.ndarray,
    transition_counts: Dict[Tuple[Cell, Cell], float],
    transition_totals: Dict[Cell, float],
) -> float:
    desired = _direction_from_step(src, dst)
    vote = float(dir_votes[src[0], src[1], desired]) if desired != 0 else 0.0
    vote_total = float(np.sum(dir_votes[src[0], src[1], 1:5]))
    vote_ratio = vote / vote_total if vote_total > 1e-6 else 0.0
    transition = float(transition_counts.get((src, dst), 0.0))
    transition_total = float(transition_totals.get(src, 0.0))
    transition_ratio = transition / transition_total if transition_total > 1e-6 else 0.0
    if transition_ratio <= 0.0 and vote_ratio <= 0.0:
        return 3.0
    cost = 1.75 - 1.15 * transition_ratio - 0.50 * vote_ratio
    if transition_ratio > 0.55:
        cost -= 0.20
    return float(np.clip(cost, 0.12, 3.0))


def _coarse_distance_map_to_outlet(
    support_mask: np.ndarray,
    outlet: Cell,
    dir_votes: np.ndarray,
    transition_counts: Dict[Tuple[Cell, Cell], float],
    transition_totals: Dict[Cell, float],
) -> np.ndarray:
    del dir_votes, transition_counts, transition_totals
    return _distance_map_to_outlet(support_mask.astype(np.uint8), outlet).astype(np.float32)


def _best_compressed_neighbor(
    cell: Cell,
    support_mask: np.ndarray,
    outlet: Cell | None,
    dist: np.ndarray | None,
    dir_votes: np.ndarray,
    transition_counts: Dict[Tuple[Cell, Cell], float],
    transition_totals: Dict[Cell, float],
) -> Cell | None:
    A, B = support_mask.shape
    best_neighbor: Cell | None = None
    best_key: Tuple[float, float, float, int, int] | None = None
    for nb in _neighbors4(cell, A, B):
        if outlet is not None and nb != outlet and int(support_mask[nb]) != 1:
            continue
        if outlet is None and int(support_mask[nb]) != 1:
            continue
        if dist is not None and not np.isfinite(dist[nb]):
            continue
        edge_cost = _coarse_edge_cost(cell, nb, dir_votes, transition_counts, transition_totals)
        transition = float(transition_counts.get((cell, nb), 0.0))
        desired = _direction_from_step(cell, nb)
        vote = float(dir_votes[cell[0], cell[1], desired]) if desired != 0 else 0.0
        vote_total = float(np.sum(dir_votes[cell[0], cell[1], 1:5]))
        vote_ratio = vote / vote_total if vote_total > 1e-6 else 0.0
        dist_term = float(dist[nb]) if dist is not None else 0.0
        manhattan = abs(nb[0] - outlet[0]) + abs(nb[1] - outlet[1]) if outlet is not None else 0
        key = (dist_term, edge_cost, -transition, -vote_ratio, manhattan, desired)
        if best_key is None or key < best_key:
            best_key = key
            best_neighbor = nb
    return best_neighbor


def compress_direction_matrix(
    matrix: np.ndarray,
    target_rows: int,
    target_cols: int,
    outlet: Cell | None = None,
) -> Tuple[np.ndarray, np.ndarray, Cell | None, Dict[str, np.ndarray | int | float]]:
    """Compress a large direction matrix into a smaller outlet-oriented matrix."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    src_a, src_b = int(matrix.shape[0]), int(matrix.shape[1])
    target_rows = int(target_rows)
    target_cols = int(target_cols)
    if not (2 <= target_rows <= src_a and 2 <= target_cols <= src_b):
        raise ValueError("target shape must be between 2 and the current matrix shape")

    source_shape = (src_a, src_b)
    target_shape = (target_rows, target_cols)
    coarse_outlet = _coarse_cell_for_fine(outlet, source_shape, target_shape) if outlet is not None else None

    if target_shape == source_shape:
        support = (matrix > 0).astype(np.uint8)
        confidence = np.where(matrix > 0, 1.0, 0.0).astype(np.float32)
        meta: Dict[str, np.ndarray | int | float] = {
            "confidence": confidence,
            "support_mask": support,
            "dropped_cells": 0,
            "block_threshold": 1.0,
        }
        return matrix.astype(np.uint8).copy(), support, coarse_outlet, meta

    coarse_counts = np.zeros(target_shape, dtype=np.float32)
    coarse_hits = np.zeros(target_shape, dtype=np.float32)
    dir_votes = np.zeros((target_rows, target_cols, 5), dtype=np.float32)
    transition_counts: Dict[Tuple[Cell, Cell], float] = defaultdict(float)
    transition_totals: Dict[Cell, float] = defaultdict(float)

    occupied_cells = np.argwhere(matrix > 0)
    for row, col in occupied_cells:
        coarse = _coarse_cell_for_fine((int(row), int(col)), source_shape, target_shape)
        coarse_counts[coarse] += 1.0
        value = int(matrix[int(row), int(col)])
        if 0 <= value <= 4:
            dir_votes[coarse[0], coarse[1], value] += 1.0

    for row, col in occupied_cells:
        start = (int(row), int(col))
        fine_path, reached_outlet = _trace_direction_path(matrix, start, outlet=outlet)
        weight = 1.0 if reached_outlet or outlet is None else 0.35
        coarse_path = _dedupe_consecutive_cells(
            [_coarse_cell_for_fine(cell, source_shape, target_shape) for cell in fine_path]
        )
        for cell in coarse_path:
            coarse_hits[cell] += weight
        for src, dst in zip(coarse_path[:-1], coarse_path[1:]):
            if src == dst:
                continue
            transition_counts[(src, dst)] += weight
            transition_totals[src] += weight

    mean_block_area = max(1.0, (src_a / float(target_rows)) * (src_b / float(target_cols)))
    block_threshold = float(max(1.0, mean_block_area * 0.08))
    support_mask = np.where((coarse_counts >= block_threshold) | (coarse_hits >= 0.95), 1, 0).astype(np.uint8)
    if int(np.count_nonzero(support_mask)) == 0:
        support_mask = (coarse_counts > 0).astype(np.uint8)
    if coarse_outlet is not None:
        support_mask[coarse_outlet] = 1

    dist = (
        _coarse_distance_map_to_outlet(support_mask, coarse_outlet, dir_votes, transition_counts, transition_totals)
        if coarse_outlet is not None
        else None
    )

    compressed = np.zeros(target_shape, dtype=np.uint8)
    confidence = np.zeros(target_shape, dtype=np.float32)
    dropped_cells = 0

    for row in range(target_rows):
        for col in range(target_cols):
            cell = (row, col)
            if int(support_mask[cell]) != 1:
                continue
            if coarse_outlet is not None and cell == coarse_outlet:
                confidence[cell] = 1.0
                continue
            if dist is not None and not np.isfinite(dist[cell]):
                support_mask[cell] = 0
                dropped_cells += 1
                continue
            best_neighbor = _best_compressed_neighbor(
                cell,
                support_mask,
                coarse_outlet,
                dist,
                dir_votes,
                transition_counts,
                transition_totals,
            )
            if best_neighbor is None:
                support_mask[cell] = 0
                dropped_cells += 1
                continue
            direction = _direction_from_step(cell, best_neighbor)
            compressed[cell] = np.uint8(direction)
            vote_total = float(np.sum(dir_votes[row, col, 1:5]))
            vote = float(dir_votes[row, col, direction]) if direction != 0 else 0.0
            vote_ratio = vote / vote_total if vote_total > 1e-6 else 0.0
            transition = float(transition_counts.get((cell, best_neighbor), 0.0))
            transition_total = float(transition_totals.get(cell, 0.0))
            transition_ratio = transition / transition_total if transition_total > 1e-6 else 0.0
            confidence[cell] = float(np.clip(0.25 + 0.45 * transition_ratio + 0.30 * vote_ratio, 0.0, 1.0))

    meta = {
        "confidence": confidence.astype(np.float32),
        "support_mask": support_mask.astype(np.uint8),
        "dropped_cells": int(dropped_cells),
        "block_threshold": float(block_threshold),
    }
    return compressed.astype(np.uint8), support_mask.astype(np.uint8), coarse_outlet, meta


def compute_direction_matrix_with_meta(
    img: np.ndarray,
    A: int,
    ink_thr: float = DEFAULT_INK_THRESHOLD,
    head_score_thr: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, int]], Dict[str, np.ndarray]]:
    """Compute a direction matrix from an image.

    Direction codes:
    0 = empty, 1 = east, 2 = south, 3 = west, 4 = north.
    """
    if img is None:
        raise ValueError("img must not be None")
    H, W = img.shape[:2]
    B = derive_grid_B(A, W, H)

    cfg = FeatureConfig(A=A, use_edges=True, use_gray=False)
    feat = build_features(img, cfg)
    bw = feat["bw"]
    component_labels = _label_binary_components(bw)
    arrow_detail_bw = build_arrow_detail_binary(img)
    arrow_component_labels = _label_binary_components(arrow_detail_bw)
    dist_map = cv2.distanceTransform((bw > 0).astype(np.uint8), cv2.DIST_L2, 3)
    base_support = build_support_mask_from_binary(bw, A, B, min_pixels=1)
    occ = occupancy_from_binary(bw, A, B, ink_thr=ink_thr)
    occ = np.maximum(occ, base_support).astype(np.uint8)
    candidates, detection_rows, detection_counts = _detect_arrow_candidates_multiscale(
        arrow_detail_bw,
        A,
        B,
        component_labels=arrow_component_labels,
        head_score_thr=head_score_thr,
    )
    candidates = _refine_candidates_by_width_profile(arrow_detail_bw, candidates)
    remapped_candidates: List[ArrowCandidate] = []
    component_search_radius = max(4, int(round(max(H / 30.0, W / float(derive_grid_B(30, W, H))) * 0.45)))
    for candidate in candidates:
        support_component_id = _component_id_near_tip(
            component_labels,
            candidate.tip[0],
            candidate.tip[1],
            radius=component_search_radius,
        )
        if support_component_id <= 0:
            continue
        remapped_candidates.append(
            ArrowCandidate(
                bbox=tuple(int(value) for value in candidate.bbox),
                tip=(float(candidate.tip[0]), float(candidate.tip[1])),
                direction=(float(candidate.direction[0]), float(candidate.direction[1])),
                score=float(candidate.score),
                area=float(candidate.area),
                component_id=int(support_component_id),
            )
        )
    candidates = remapped_candidates
    bboxes = [candidate.bbox for candidate in candidates]
    head_tip_mask, head_halo_mask, _tip_cells = _head_stage_masks_from_candidates(candidates, A, B, H, W)

    skel = skeletonize((bw > 0).astype(bool)).astype(np.uint8)
    axis_x, axis_y, axis_conf = _compute_cell_axis_field(skel, A, B)
    adj = _build_skeleton_adjacency(skel)
    components, pixel_to_component = _connected_components(adj)
    nodes_by_component_label: DefaultDict[int, List[Pixel]] = defaultdict(list)
    for node in adj:
        component_id = int(component_labels[node[0], node[1]])
        if component_id > 0:
            nodes_by_component_label[component_id].append(node)
    component_node_sets: Dict[int, set[Pixel]] = {
        component_id: set(nodes)
        for component_id, nodes in nodes_by_component_label.items()
    }
    component_candidate_counts: DefaultDict[int, int] = defaultdict(int)
    for candidate in candidates:
        component_candidate_counts[int(candidate.component_id)] += 1
    component_complexity: Dict[int, bool] = {}
    for component_id in component_candidate_counts:
        nodes = nodes_by_component_label.get(int(component_id), [])
        endpoint_count = sum(1 for node in nodes if len(adj.get(node, [])) <= 1)
        branch_count = sum(1 for node in nodes if len(adj.get(node, [])) > 2)
        component_complexity[int(component_id)] = (
            int(component_candidate_counts[int(component_id)]) >= 4
            or endpoint_count > 10
            or branch_count > 24
        )

    votes: DefaultDict[Cell, DefaultDict[DirectionCode, float]] = defaultdict(lambda: defaultdict(float))
    transition_votes: DefaultDict[Tuple[Cell, Cell], float] = defaultdict(float)
    path_occ = np.zeros((A, B), dtype=np.uint8)
    overlap_count = np.zeros((A, B), dtype=np.int32)
    object_paths: List[ArrowObjectPath] = []
    propagated_cell_paths: List[Tuple[Cell, ...]] = []
    candidate_touch_ids: DefaultDict[Cell, set[int]] = defaultdict(set)
    cell_span = float(max(H / float(A), W / float(B), 1.0))

    for candidate_index, candidate in enumerate(candidates):
        candidate_nodes = nodes_by_component_label.get(int(candidate.component_id), [])
        if not candidate_nodes:
            continue
        component_nodes = component_node_sets.get(int(candidate.component_id))
        if component_nodes is None:
            component_nodes = set(components[pixel_to_component[candidate_nodes[0]]])
        object_path = _build_candidate_object_path(
            candidate,
            candidate_index,
            candidate_nodes,
            component_nodes,
            adj,
            dist_map,
            int(component_candidate_counts[int(candidate.component_id)]),
            A,
            B,
            H,
            W,
        )
        if object_path is None:
            continue
        object_paths.append(object_path)

        candidate_weight = max(0.0, float(object_path.score))
        _vote_object_path(
            object_path,
            votes,
            weight=float(1.0 + 4.2 * candidate_weight),
            terminal_weight=float(0.5 + 1.8 * candidate_weight),
            transition_votes=transition_votes,
        )
        for row, col in object_path.raw_cell_path:
            if int(base_support[row, col]) == 1 or int(occ[row, col]) == 1:
                path_occ[row, col] = 1
            candidate_touch_ids[(row, col)].add(int(candidate_index))

        same_component_distances = [
            float(np.hypot(candidate.tip[0] - other.tip[0], candidate.tip[1] - other.tip[1]))
            for other_index, other in enumerate(candidates)
            if other_index != candidate_index and int(other.component_id) == int(candidate.component_id)
        ]
        head_span = float(max(candidate.bbox[2] - candidate.bbox[0], candidate.bbox[3] - candidate.bbox[1], 1))
        if same_component_distances:
            propagation_limit = float(
                np.clip(
                    min(same_component_distances) * 0.80,
                    max(head_span * 2.2, cell_span * 3.5),
                    cell_span * 16.0,
                )
            )
        else:
            propagation_limit = float(max(head_span * 3.0, cell_span * 14.0))

        bx1, by1, bx2, by2 = candidate.bbox
        max_dist_sq = float(max(bx2 - bx1, by2 - by1, 1) ** 2) * 2.0
        anchor_node = _pick_component_head_node(candidate_nodes, candidate, max_dist_sq=max_dist_sq)
        if anchor_node is not None:
            propagation_component_set = component_node_sets.get(int(candidate.component_id))
            if propagation_component_set is None:
                propagation_component_set = set(component_nodes)
            propagated_pixel_path = _trace_candidate_centerline(
                anchor_node,
                propagation_component_set,
                adj,
                candidate.direction,
                max_pixels=propagation_limit,
            )
            propagated_raw_path = _cell_path_from_pixel_path(
                propagated_pixel_path,
                A,
                B,
                H,
                W,
                expand_diagonal=False,
            )
            propagated_vote_path = _cell_path_from_pixel_path(
                propagated_pixel_path,
                A,
                B,
                H,
                W,
                expand_diagonal=True,
            )
            if len(propagated_raw_path) >= 2 and len(propagated_vote_path) >= 2:
                propagated_object = ArrowObjectPath(
                    component_id=int(candidate.component_id),
                    candidate_index=int(candidate_index),
                    bbox=tuple(int(value) for value in candidate.bbox),
                    head_cell=_pixel_to_cell(
                        int(round(candidate.tip[1])),
                        int(round(candidate.tip[0])),
                        A,
                        B,
                        H,
                        W,
                    ),
                    tail_cell=propagated_vote_path[0],
                    raw_cell_path=tuple(propagated_raw_path),
                    vote_cell_path=tuple(propagated_vote_path),
                    score=float(np.clip(candidate.score, 0.05, 1.0)),
                    direction_code=int(_direction_from_vector(candidate.direction)),
                )
                _vote_object_path(
                    propagated_object,
                    votes,
                    weight=float(0.70 + 2.0 * candidate_weight),
                    terminal_weight=float(0.35 + 0.8 * candidate_weight),
                    transition_votes=transition_votes,
                )
                propagated_cell_paths.append(tuple(propagated_raw_path))
                for row, col in set(propagated_raw_path) | set(propagated_vote_path):
                    if int(base_support[row, col]) == 1 or int(occ[row, col]) == 1:
                        path_occ[row, col] = 1
                    candidate_touch_ids[(row, col)].add(int(candidate_index))

        if bool(component_complexity.get(int(candidate.component_id), False)):
            continue
        head_dir = _direction_from_vector(candidate.direction)
        if head_dir != 0 and _is_axis_dominant(candidate.direction):
            bx1, by1, bx2, by2 = candidate.bbox
            row0 = int(np.clip(np.floor(by1 / max(H / float(A), 1e-6)), 0, A - 1))
            row1 = int(np.clip(np.ceil(by2 / max(H / float(A), 1e-6)), 1, A))
            col0 = int(np.clip(np.floor(bx1 / max(W / float(B), 1e-6)), 0, B - 1))
            col1 = int(np.clip(np.ceil(bx2 / max(W / float(B), 1e-6)), 1, B))
            for row in range(row0, row1):
                for col in range(col0, col1):
                    if base_support[row, col] == 1 or path_occ[row, col] == 1:
                        votes[(row, col)][head_dir] += float(0.45 + 1.5 * candidate_weight)

    for (row, col), candidate_ids in candidate_touch_ids.items():
        overlap_count[row, col] = int(len(candidate_ids))

    guided_candidates = [
        candidate
        for candidate in candidates
        if not bool(component_complexity.get(int(candidate.component_id), False))
    ]
    _add_candidate_guided_votes(guided_candidates, occ, path_occ, axis_x, axis_y, axis_conf, A, B, H, W, votes)

    fallback_cells: set[Cell] = set()
    for object_path in object_paths:
        fallback_cells.update(object_path.raw_cell_path)
        fallback_cells.update(object_path.vote_cell_path)
    for propagated_path in propagated_cell_paths:
        fallback_cells.update(propagated_path)
    if fallback_cells:
        _fallback_orientation_votes(adj, A, B, H, W, votes, allowed_cells=fallback_cells)
    occ = np.maximum(occ, path_occ)

    D = np.zeros((A, B), dtype=np.uint8)
    confidence = np.zeros((A, B), dtype=np.float32)
    for i in range(A):
        for j in range(B):
            if occ[i, j] != 1:
                continue
            cell_votes = votes.get((i, j))
            if not cell_votes:
                continue
            if int(overlap_count[i, j]) > 1:
                direction = _resolve_cell_direction(
                    (i, j),
                    cell_votes,
                    transition_votes,
                    overlap_count,
                    axis_x,
                    axis_y,
                    axis_conf,
                    A,
                    B,
                )
            else:
                direction = max(cell_votes.items(), key=lambda item: item[1])[0]
            D[i, j] = np.uint8(direction)
            cell_conf = _vote_confidence(cell_votes)
            if int(overlap_count[i, j]) > 1:
                cell_conf *= 0.88
            confidence[i, j] = np.float32(cell_conf)

    for candidate in candidates:
        if bool(component_complexity.get(int(candidate.component_id), False)):
            continue
        head_dir = _direction_from_vector(candidate.direction)
        if head_dir == 0 or not _is_axis_dominant(candidate.direction):
            continue
        bx1, by1, bx2, by2 = candidate.bbox
        row0 = int(np.clip(np.floor(by1 / max(H / float(A), 1e-6)), 0, A - 1))
        row1 = int(np.clip(np.ceil(by2 / max(H / float(A), 1e-6)), 1, A))
        col0 = int(np.clip(np.floor(bx1 / max(W / float(B), 1e-6)), 0, B - 1))
        col1 = int(np.clip(np.ceil(bx2 / max(W / float(B), 1e-6)), 1, B))
        for row in range(row0, row1):
            for col in range(col0, col1):
                if occ[row, col] != 1 and path_occ[row, col] != 1:
                    continue
                if int(overlap_count[row, col]) > 1 and int(D[row, col]) != 0:
                    continue
                D[row, col] = np.uint8(head_dir)
                confidence[row, col] = np.float32(max(float(confidence[row, col]), 0.78))

    centerline_support = _rasterize_skeleton_support(adj, A, B, H, W)
    D, confidence, centerline_fill_count = _propagate_directions_on_unbranched_centerline(
        D,
        centerline_support,
        confidence=confidence,
    )
    path_occ = np.maximum(
        path_occ,
        ((D > 0) & (centerline_support > 0)).astype(np.uint8),
    ).astype(np.uint8)

    support_mask, fillable_mask, bridge_ids = build_guidance_support_mask(
        bw,
        A,
        B,
        occ=occ,
        matrix=D,
        path_occ=path_occ,
        confidence=confidence,
        ink_thr=max(DEFAULT_INK_THRESHOLD, ink_thr),
    )
    support_level = np.zeros((A, B), dtype=np.uint8)
    weak_support = np.maximum(fillable_mask.astype(np.uint8), ((head_halo_mask == 1) & (base_support == 0)).astype(np.uint8))
    strong_support = np.maximum(base_support.astype(np.uint8), np.maximum(occ.astype(np.uint8), path_occ.astype(np.uint8)))
    support_level[weak_support == 1] = 1
    support_level[strong_support == 1] = 2
    object_head_cells = np.asarray([path.head_cell for path in object_paths], dtype=np.int32).reshape(-1, 2)
    object_tail_cells = np.asarray([path.tail_cell for path in object_paths], dtype=np.int32).reshape(-1, 2)
    object_scores = np.asarray([float(path.score) for path in object_paths], dtype=np.float32)
    object_directions = np.asarray(
        [
            int(path.direction_code) if int(path.direction_code) != 0 else _direction_from_step(path.tail_cell, path.head_cell)
            for path in object_paths
        ],
        dtype=np.uint8,
    )
    object_path_lengths = np.asarray([len(path.raw_cell_path) for path in object_paths], dtype=np.int32)
    object_direction_matrix, object_support_mask, object_overlap_count = rasterize_arrow_object_paths(object_paths, A, B)
    meta = {
        "confidence": confidence.astype(np.float32),
        "support_mask": support_mask.astype(np.uint8),
        "fillable_mask": fillable_mask.astype(np.uint8),
        "bridge_ids": bridge_ids.astype(np.int32),
        "path_occ": path_occ.astype(np.uint8),
        "overlap_count": overlap_count.astype(np.int32),
        "support_level": support_level.astype(np.uint8),
        "axis_x": axis_x.astype(np.float32),
        "axis_y": axis_y.astype(np.float32),
        "axis_confidence": axis_conf.astype(np.float32),
        "bw": bw.astype(np.uint8),
        "arrow_detail_bw": arrow_detail_bw.astype(np.uint8),
        "edges": feat["edges"].astype(np.uint8),
        "object_count": np.array([len(object_paths)], dtype=np.int32),
        "object_head_cells": object_head_cells,
        "object_tail_cells": object_tail_cells,
        "object_directions": object_directions,
        "object_scores": object_scores,
        "object_path_lengths": object_path_lengths,
        "object_direction_matrix": object_direction_matrix.astype(np.uint8),
        "object_support_mask": object_support_mask.astype(np.uint8),
        "object_overlap_count": object_overlap_count.astype(np.int32),
        "centerline_support_mask": centerline_support.astype(np.uint8),
        "centerline_fill_count": np.array([int(centerline_fill_count)], dtype=np.int32),
        "centerline_unassigned_count": np.array(
            [int(np.count_nonzero((centerline_support > 0) & (D == 0)))],
            dtype=np.int32,
        ),
        "propagated_path_count": np.array([len(propagated_cell_paths)], dtype=np.int32),
        "propagated_path_nonzero": np.array(
            [len({cell for path in propagated_cell_paths for cell in path})],
            dtype=np.int32,
        ),
        "candidate_detection_rows": np.asarray(detection_rows, dtype=np.int32),
        "candidate_count_by_row": np.asarray(detection_counts, dtype=np.int32),
    }
    return D, occ.astype(np.uint8), bboxes, meta


def compute_direction_matrix(
    img: np.ndarray,
    A: int,
    ink_thr: float = DEFAULT_INK_THRESHOLD,
    head_score_thr: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
    direction_matrix, occ, bboxes, _meta = compute_direction_matrix_with_meta(
        img,
        A,
        ink_thr=ink_thr,
        head_score_thr=head_score_thr,
    )
    return direction_matrix, occ, bboxes


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        raise SystemExit("usage: py -3 solution.py <image_path> <rows>")

    image_path = sys.argv[1]
    rows = int(sys.argv[2])
    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"failed to read image: {image_path}")

    direction_matrix, occupancy, arrow_boxes = compute_direction_matrix(image, rows)
    print("Direction matrix:")
    print(direction_matrix)
    print("Occupancy:")
    print(occupancy)
    print("Arrow boxes:", arrow_boxes)
