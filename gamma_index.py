# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Uses NumPy: https://github.com/numpy/numpy (BSD-3-Clause plus bundled notices).
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


_DIRECTION_DELTAS = np.asarray(
    [
        (0, 0),
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
    ],
    dtype=np.int8,
)


@dataclass(frozen=True)
class GammaResult:
    outlet: Tuple[int, int]
    contributing_cells: int
    removed_lengths_edge: np.ndarray
    removed_lengths_cell: np.ndarray
    remaining_cells: np.ndarray
    removal_order: np.ndarray
    removed_paths: Tuple[Tuple[Tuple[int, int], ...], ...]
    unique_lengths: np.ndarray
    occurrence_counts: np.ndarray
    weighted_values: np.ndarray
    gravity_center_x: float
    maximum_length: int
    gamma: float
    mean_removed_length: float
    sample_std_removed_length: float
    top10_length_ratio: float


def _validated_matrix(matrix: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
    raw = np.asarray(matrix)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError("direction matrix must be a non-empty two-dimensional array")
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("direction matrix contains a non-numeric value") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.rint(numeric)):
        raise ValueError("direction matrix values must be finite integers")
    if np.any((numeric < 0) | (numeric > 4)):
        raise ValueError("direction matrix codes must be in the range 0..4")
    return numeric.astype(np.uint8)


def _coords(index: int, rows: int) -> Tuple[int, int]:
    return int(index % rows), int(index // rows)


def _downstream_index(matrix: np.ndarray, active: np.ndarray, index: int) -> Optional[int]:
    rows, cols = matrix.shape
    row, col = _coords(index, rows)
    code = int(matrix[row, col])
    if code < 1 or code > 4:
        return None
    row2 = row + int(_DIRECTION_DELTAS[code, 0])
    col2 = col + int(_DIRECTION_DELTAS[code, 1])
    if row2 < 0 or row2 >= rows or col2 < 0 or col2 >= cols or not bool(active[row2, col2]):
        return None
    return int(row2 + col2 * rows)


def _trace_active(matrix: np.ndarray, active: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace a functional grid graph without storing an O(N^2) path table."""
    rows, cols = matrix.shape
    total = rows * cols
    distance = np.full(total, np.nan, dtype=float)
    sink = np.full(total, -1, dtype=np.int64)
    valid = np.zeros(total, dtype=bool)
    state = np.zeros(total, dtype=np.uint8)
    active_ids = np.flatnonzero(active.ravel(order="F"))

    def invalidate(trail: list[int]) -> None:
        for trail_index in trail:
            state[trail_index] = 2
            distance[trail_index] = np.nan
            sink[trail_index] = -1
            valid[trail_index] = False

    def resolve_from_child(trail: list[int], child: int) -> None:
        for parent in reversed(trail):
            state[parent] = 2
            distance[parent] = distance[child] + 1.0
            sink[parent] = sink[child]
            valid[parent] = True
            child = parent

    for raw_start in active_ids:
        start = int(raw_start)
        if state[start] == 2:
            continue
        trail: list[int] = []
        position: dict[int, int] = {}
        current = start

        while True:
            if state[current] == 2:
                if valid[current]:
                    resolve_from_child(trail, current)
                else:
                    invalidate(trail)
                break
            if state[current] == 3:
                invalidate(trail)
                break
            if current in position or state[current] == 1:
                invalidate(trail)
                break

            row, col = _coords(current, rows)
            code = int(matrix[row, col])
            position[current] = len(trail)
            trail.append(current)
            state[current] = 1
            if code < 1 or code > 4:
                invalidate(trail)
                break

            next_index = _downstream_index(matrix, active, current)
            if next_index is None:
                distance[current] = 0.0
                sink[current] = current
                valid[current] = True
                state[current] = 2
                trail.pop()
                resolve_from_child(trail, current)
                break
            current = next_index

    return distance, sink, valid


def _path_from_start(matrix: np.ndarray, active: np.ndarray, start: int) -> list[int]:
    rows, _cols = matrix.shape
    path: list[int] = []
    seen: set[int] = set()
    current = int(start)
    while True:
        if current in seen:
            raise RuntimeError("cycle encountered while reconstructing a valid flowline")
        seen.add(current)
        row, col = _coords(current, rows)
        if not bool(active[row, col]):
            break
        path.append(current)
        next_index = _downstream_index(matrix, active, current)
        if next_index is None:
            break
        current = next_index
    return path


def compute_gamma_index(
    matrix: np.ndarray | Sequence[Sequence[int]],
    outlet: Optional[Tuple[int, int]] = None,
) -> GammaResult:
    """Reproduce the supplied MATLAB beta2/center-of-gravity procedure.

    Direction codes are 1=East, 2=South, 3=West, and 4=North. Candidate
    ties follow MATLAB's column-major ``find`` order.
    """
    direction = _validated_matrix(matrix)
    rows, cols = direction.shape
    active0 = direction > 0
    distance0, sink0, valid0 = _trace_active(direction, active0)
    active0_flat = active0.ravel(order="F")
    valid_active_ids = np.flatnonzero(active0_flat & valid0)
    if valid_active_ids.size == 0:
        raise ValueError("the direction matrix has no valid downstream path")

    if outlet is None:
        unique_sinks, sink_counts = np.unique(sink0[valid_active_ids], return_counts=True)
        main_sink = int(unique_sinks[int(np.argmax(sink_counts))])
        outlet_row, outlet_col = _coords(main_sink, rows)
    else:
        outlet_row, outlet_col = int(outlet[0]), int(outlet[1])
        if not (0 <= outlet_row < rows and 0 <= outlet_col < cols):
            raise ValueError("outlet is outside the direction matrix")
        main_sink = int(outlet_row + outlet_col * rows)

    basin_flat = active0_flat & valid0 & (sink0 == main_sink)
    basin = basin_flat.reshape((rows, cols), order="F")
    contributing_cells = int(np.count_nonzero(basin))
    if contributing_cells == 0:
        raise ValueError("no valid cell drains to the selected outlet")

    active = basin.copy()
    removal_order = np.full((rows, cols), np.nan, dtype=float)
    edge_lengths: list[int] = []
    cell_lengths: list[int] = []
    remaining_cells: list[int] = []
    removed_paths: list[Tuple[Tuple[int, int], ...]] = []

    iteration = 0
    while np.any(active):
        distance, _sink, valid = _trace_active(direction, active)
        candidate_ids = np.flatnonzero(active.ravel(order="F") & valid & np.isfinite(distance))
        if candidate_ids.size == 0:
            break
        candidate_distances = distance[candidate_ids]
        start_index = int(candidate_ids[int(np.argmax(candidate_distances))])
        path_indices = _path_from_start(direction, active, start_index)
        path_indices = [index for index in path_indices if active[_coords(index, rows)]]
        if not path_indices:
            row, col = _coords(start_index, rows)
            active[row, col] = False
            continue

        iteration += 1
        path_coords = tuple(_coords(index, rows) for index in path_indices)
        for row, col in path_coords:
            removal_order[row, col] = float(iteration)
            active[row, col] = False
        removed_paths.append(path_coords)
        edge_lengths.append(int(round(float(distance[start_index]))))
        cell_lengths.append(len(path_indices))
        remaining_cells.append(int(np.count_nonzero(active)))

    if not cell_lengths:
        raise ValueError("the selected basin produced no removable flowline")

    removed_cells = np.asarray(cell_lengths, dtype=np.int64)
    unique_lengths, occurrence_counts = np.unique(removed_cells, return_counts=True)
    occurrence_counts = occurrence_counts.astype(np.int64)
    weighted_values = unique_lengths.astype(float) * occurrence_counts.astype(float)
    weighted_sum = float(np.sum(weighted_values))
    maximum_length = int(np.max(unique_lengths))
    if weighted_sum == 0.0 or maximum_length <= 0:
        gravity_center_x = float("nan")
        gamma = float("nan")
    else:
        gravity_center_x = float(np.sum(unique_lengths.astype(float) * weighted_values) / weighted_sum)
        gamma = float(gravity_center_x / float(maximum_length))

    mean_removed_length = float(np.mean(removed_cells))
    sample_std_removed_length = float(np.std(removed_cells, ddof=1)) if removed_cells.size > 1 else 0.0
    sorted_lengths = np.sort(removed_cells)[::-1]
    top_count = max(1, int(math.ceil(0.10 * sorted_lengths.size)))
    top10_length_ratio = float(np.sum(sorted_lengths[:top_count]) / np.sum(sorted_lengths))

    return GammaResult(
        outlet=(outlet_row, outlet_col),
        contributing_cells=contributing_cells,
        removed_lengths_edge=np.asarray(edge_lengths, dtype=np.int64),
        removed_lengths_cell=removed_cells,
        remaining_cells=np.asarray(remaining_cells, dtype=np.int64),
        removal_order=removal_order,
        removed_paths=tuple(removed_paths),
        unique_lengths=unique_lengths.astype(np.int64),
        occurrence_counts=occurrence_counts,
        weighted_values=weighted_values,
        gravity_center_x=gravity_center_x,
        maximum_length=maximum_length,
        gamma=gamma,
        mean_removed_length=mean_removed_length,
        sample_std_removed_length=sample_std_removed_length,
        top10_length_ratio=top10_length_ratio,
    )


def load_plena_input_matrix(path: str | Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    input_path = Path(path)
    lines = [line.strip() for line in input_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError("PLENA input is missing its header, outlet, or matrix")
    try:
        rows, cols = (int(token) for token in lines[0].split())
        outlet_row_1, outlet_col_1 = (int(token) for token in lines[1].split())
    except (TypeError, ValueError) as exc:
        raise ValueError("PLENA input header and outlet must each contain two integers") from exc
    if rows <= 0 or cols <= 0:
        raise ValueError("PLENA input dimensions must be positive")
    if len(lines) != rows + 2:
        raise ValueError(f"PLENA input declares {rows} matrix rows but contains {len(lines) - 2}")
    matrix_rows: list[list[int]] = []
    for row_index, line in enumerate(lines[2:]):
        try:
            values = [int(token) for token in line.split()]
        except ValueError as exc:
            raise ValueError(f"PLENA input row {row_index + 1} contains a non-integer value") from exc
        if len(values) != cols:
            raise ValueError(f"PLENA input row {row_index + 1} has {len(values)} values; expected {cols}")
        matrix_rows.append(values)
    outlet = (outlet_row_1 - 1, outlet_col_1 - 1)
    if not (0 <= outlet[0] < rows and 0 <= outlet[1] < cols):
        raise ValueError("PLENA input outlet is outside the declared matrix")
    return _validated_matrix(matrix_rows), outlet


def format_gamma_summary(result: GammaResult) -> str:
    lines = [
        "Gamma index (MATLAB beta2 / GravityCenterRatio)",
        "Formula: Gamma = sum(L^2 * count) / (sum(L * count) * max(L))",
        f"Outlet (1-based row, col): {result.outlet[0] + 1}, {result.outlet[1] + 1}",
        f"Contributing cells: {result.contributing_cells}",
        f"Removal iterations: {result.removed_lengths_cell.size}",
        f"Center of gravity X: {result.gravity_center_x:.10f}",
        f"Maximum removed flowline length: {result.maximum_length}",
        f"Gamma: {result.gamma:.10f}",
        f"Mean removed length: {result.mean_removed_length:.10f}",
        f"Sample standard deviation: {result.sample_std_removed_length:.10f}",
        f"Top 10% length ratio: {result.top10_length_ratio:.10f}",
        "",
        "[Removal iterations]",
        "iteration\tlength_edge\tlength_cell\tremaining_cells",
    ]
    for index, (edge, cell, remaining) in enumerate(
        zip(result.removed_lengths_edge, result.removed_lengths_cell, result.remaining_cells),
        start=1,
    ):
        lines.append(f"{index}\t{int(edge)}\t{int(cell)}\t{int(remaining)}")
    lines.extend(["", "[Figure 6 distribution]", "length\tcount\tlength_x_count"])
    for length, count, weighted in zip(result.unique_lengths, result.occurrence_counts, result.weighted_values):
        lines.append(f"{int(length)}\t{int(count)}\t{float(weighted):.10f}")
    return "\n".join(lines)


def save_gamma_outputs(input_path: str | Path, result: GammaResult) -> Tuple[Path, Path]:
    source = Path(input_path)
    summary_path = source.with_name(f"{source.stem}_gamma.txt")
    distribution_path = source.with_name(f"{source.stem}_gamma_distribution.csv")
    summary_path.write_text(format_gamma_summary(result) + "\n", encoding="utf-8")
    with distribution_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("removed_flowline_length", "occurrence_count", "length_x_count"))
        for length, count, weighted in zip(result.unique_lengths, result.occurrence_counts, result.weighted_values):
            writer.writerow((int(length), int(count), f"{float(weighted):.10f}"))
    return summary_path, distribution_path
