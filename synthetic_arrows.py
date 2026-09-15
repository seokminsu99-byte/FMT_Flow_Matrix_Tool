# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Uses OpenCV: https://github.com/opencv/opencv (Apache-2.0); opencv-python wheel notices apply.
# Uses NumPy: https://github.com/numpy/numpy (BSD-3-Clause plus bundled notices).
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from cell_training import CONTEXT_RADIUS, extract_features_and_labels


Cell = Tuple[int, int]


@dataclass
class SyntheticArrowVariant:
    image_bgr: np.ndarray
    labels: Dict[Cell, int]
    style_counts: Dict[str, int]


@dataclass
class SyntheticArrowTrainingBatch:
    X: List[List[float]]
    X_patch: List[np.ndarray]
    y: List[int]
    preview_images: List[np.ndarray]
    label_counts: Dict[int, int]
    style_counts: Dict[str, int]
    variant_count: int
    labeled_cells: int


_STYLES = ("arrowed", "filled", "open", "chevron", "block", "broken")


def _direction_unit(code: int) -> Tuple[float, float]:
    if int(code) == 1:
        return 1.0, 0.0
    if int(code) == 2:
        return 0.0, 1.0
    if int(code) == 3:
        return -1.0, 0.0
    if int(code) == 4:
        return 0.0, -1.0
    return 0.0, 0.0


def _as_point(x: float, y: float) -> Tuple[int, int]:
    return int(round(float(x))), int(round(float(y)))


def _draw_filled_head(
    img: np.ndarray,
    tip: np.ndarray,
    unit: np.ndarray,
    perp: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
    head_len: float,
    head_w: float,
    filled: bool = True,
) -> np.ndarray:
    base = tip - unit * head_len
    pts = np.asarray(
        [
            _as_point(tip[0], tip[1]),
            _as_point(base[0] + perp[0] * head_w, base[1] + perp[1] * head_w),
            _as_point(base[0] - perp[0] * head_w, base[1] - perp[1] * head_w),
        ],
        dtype=np.int32,
    )
    if filled:
        cv2.fillConvexPoly(img, pts, color, lineType=cv2.LINE_AA)
    else:
        cv2.polylines(img, [pts.reshape((-1, 1, 2))], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return base


def _draw_one_arrow(
    img: np.ndarray,
    center: np.ndarray,
    code: int,
    style: str,
    rng: np.random.Generator,
    cell_px: int,
) -> None:
    dx, dy = _direction_unit(code)
    if dx == 0.0 and dy == 0.0:
        return
    unit = np.asarray([dx, dy], dtype=np.float32)
    perp = np.asarray([-dy, dx], dtype=np.float32)
    color_value = int(rng.integers(0, 42))
    color = (color_value, color_value, color_value)
    thickness = int(rng.integers(max(1, cell_px // 14), max(2, cell_px // 7) + 1))
    tail_len = float(cell_px) * float(rng.uniform(0.24, 0.38))
    head_len = float(cell_px) * float(rng.uniform(0.20, 0.34))
    head_w = float(cell_px) * float(rng.uniform(0.14, 0.27))
    tip = center + unit * float(cell_px) * float(rng.uniform(0.24, 0.36))
    tail = center - unit * tail_len

    if style == "arrowed":
        cv2.arrowedLine(
            img,
            _as_point(tail[0], tail[1]),
            _as_point(tip[0], tip[1]),
            color,
            thickness,
            cv2.LINE_AA,
            0,
            float(rng.uniform(0.34, 0.48)),
        )
        return

    if style == "open":
        cv2.line(img, _as_point(tail[0], tail[1]), _as_point(tip[0], tip[1]), color, thickness, lineType=cv2.LINE_AA)
        base = tip - unit * head_len
        left = base + perp * head_w
        right = base - perp * head_w
        cv2.line(img, _as_point(tip[0], tip[1]), _as_point(left[0], left[1]), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, _as_point(tip[0], tip[1]), _as_point(right[0], right[1]), color, thickness, lineType=cv2.LINE_AA)
        return

    if style == "chevron":
        mid = center - unit * float(cell_px) * 0.07
        cv2.line(img, _as_point(tail[0], tail[1]), _as_point(mid[0], mid[1]), color, thickness, lineType=cv2.LINE_AA)
        base = tip - unit * head_len
        cv2.line(
            img,
            _as_point(base[0] + perp[0] * head_w, base[1] + perp[1] * head_w),
            _as_point(tip[0], tip[1]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            img,
            _as_point(base[0] - perp[0] * head_w, base[1] - perp[1] * head_w),
            _as_point(tip[0], tip[1]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        return

    if style == "block":
        base = _draw_filled_head(img, tip, unit, perp, color, thickness, head_len * 0.9, head_w * 1.05, filled=True)
        shaft_tail = tail - perp * float(rng.uniform(-0.8, 0.8))
        cv2.line(img, _as_point(shaft_tail[0], shaft_tail[1]), _as_point(base[0], base[1]), color, thickness + 1, lineType=cv2.LINE_AA)
        return

    if style == "broken":
        base = _draw_filled_head(img, tip, unit, perp, color, thickness, head_len, head_w, filled=True)
        gap = float(cell_px) * float(rng.uniform(0.05, 0.10))
        a = center - unit * gap
        b = center + unit * gap
        cv2.line(img, _as_point(tail[0], tail[1]), _as_point(a[0], a[1]), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, _as_point(b[0], b[1]), _as_point(base[0], base[1]), color, thickness, lineType=cv2.LINE_AA)
        return

    base = _draw_filled_head(img, tip, unit, perp, color, thickness, head_len, head_w, filled=True)
    cv2.line(img, _as_point(tail[0], tail[1]), _as_point(base[0], base[1]), color, thickness, lineType=cv2.LINE_AA)


def render_arrow_matrix_image(
    matrix: np.ndarray,
    *,
    cell_px: int = 24,
    variant: int = 0,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Render a direction matrix as diverse synthetic arrow drawings."""
    mat = np.asarray(matrix, dtype=np.uint8)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    rows, cols = mat.shape
    cell_px = max(12, int(cell_px))
    rng = np.random.default_rng(int(seed) + int(variant) * 1000003)
    bg = int(rng.integers(238, 256))
    img = np.full((rows * cell_px, cols * cell_px, 3), bg, dtype=np.uint8)

    if bool(rng.integers(0, 2)):
        noise = rng.normal(0.0, float(rng.uniform(1.0, 4.5)), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if bool(rng.integers(0, 3) == 0):
        grid_color = int(rng.integers(220, 242))
        for row in range(rows + 1):
            y = int(row * cell_px)
            cv2.line(img, (0, y), (cols * cell_px, y), (grid_color, grid_color, grid_color), 1, lineType=cv2.LINE_AA)
        for col in range(cols + 1):
            x = int(col * cell_px)
            cv2.line(img, (x, 0), (x, rows * cell_px), (grid_color, grid_color, grid_color), 1, lineType=cv2.LINE_AA)

    style_counts: Counter[str] = Counter()
    cells = [(int(i), int(j)) for i, j in np.argwhere(mat > 0)]
    rng.shuffle(cells)
    for row, col in cells:
        code = int(mat[row, col])
        style = _STYLES[int(rng.integers(0, len(_STYLES)))]
        jitter = float(cell_px) * float(rng.uniform(0.00, 0.11))
        center = np.asarray(
            [
                (col + 0.5) * cell_px + float(rng.uniform(-jitter, jitter)),
                (row + 0.5) * cell_px + float(rng.uniform(-jitter, jitter)),
            ],
            dtype=np.float32,
        )
        _draw_one_arrow(img, center, code, style, rng, cell_px)
        style_counts[style] += 1

    if bool(rng.integers(0, 4) == 0):
        kernel_size = int(rng.choice([2, 3]))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        if bool(rng.integers(0, 2)):
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)
    if bool(rng.integers(0, 3) == 0):
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=float(rng.uniform(0.25, 0.65)))
    return img, {style: int(count) for style, count in style_counts.items()}


def labels_from_matrix(
    matrix: np.ndarray,
    *,
    max_directional_cells: int = 5000,
    zero_ratio: float = 0.35,
    seed: int = 0,
) -> Dict[Cell, int]:
    mat = np.asarray(matrix, dtype=np.uint8)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    rng = np.random.default_rng(int(seed))
    directional = [(int(i), int(j)) for i, j in np.argwhere(mat > 0)]
    if max_directional_cells > 0 and len(directional) > int(max_directional_cells):
        selected_idx = rng.choice(len(directional), size=int(max_directional_cells), replace=False)
        directional = [directional[int(idx)] for idx in selected_idx]
    directional.sort()

    labels: Dict[Cell, int] = {(i, j): int(mat[i, j]) for i, j in directional if int(mat[i, j]) in (1, 2, 3, 4)}
    occupied = (mat > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    far_zero_mask = ((mat == 0) & (cv2.dilate(occupied, kernel, iterations=1) == 0)).astype(np.uint8)
    zero_cells = [(int(i), int(j)) for i, j in np.argwhere(far_zero_mask > 0)]
    if not zero_cells:
        zero_cells = [(int(i), int(j)) for i, j in np.argwhere(mat == 0)]
    zero_count = min(len(zero_cells), max(1, int(round(len(labels) * max(0.0, float(zero_ratio))))))
    if zero_count > 0:
        selected_zero_idx = rng.choice(len(zero_cells), size=zero_count, replace=False)
        for idx in selected_zero_idx:
            labels[zero_cells[int(idx)]] = 0
    return labels


def build_synthetic_arrow_training_batch(
    matrix: np.ndarray,
    *,
    variants: int = 8,
    cell_px: int = 24,
    max_directional_cells_per_variant: int = 5000,
    seed: int = 0,
    preview_limit: int = 4,
) -> SyntheticArrowTrainingBatch:
    mat = np.asarray(matrix, dtype=np.uint8)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    rows = int(mat.shape[0])
    variants = max(1, int(variants))
    X_all: List[List[float]] = []
    X_patch_all: List[np.ndarray] = []
    y_all: List[int] = []
    preview_images: List[np.ndarray] = []
    label_counts: Counter[int] = Counter()
    style_counts: Counter[str] = Counter()
    labeled_cells = 0

    for variant in range(variants):
        variant_seed = int(seed) + variant * 7919
        image, styles = render_arrow_matrix_image(mat, cell_px=cell_px, variant=variant, seed=seed)
        labels = labels_from_matrix(
            mat,
            max_directional_cells=max_directional_cells_per_variant,
            zero_ratio=0.35,
            seed=variant_seed,
        )
        X_new, X_patch_new, y_new = extract_features_and_labels(
            image,
            rows,
            labels,
            reference_matrix=mat,
            context_radius=CONTEXT_RADIUS,
            corrected_weight=1,
        )
        X_all.extend(X_new)
        X_patch_all.extend(X_patch_new)
        y_all.extend(y_new)
        label_counts.update(int(v) for v in y_new)
        style_counts.update(styles)
        labeled_cells += len(labels)
        if len(preview_images) < max(0, int(preview_limit)):
            preview_images.append(image)

    return SyntheticArrowTrainingBatch(
        X=X_all,
        X_patch=X_patch_all,
        y=y_all,
        preview_images=preview_images,
        label_counts={int(k): int(v) for k, v in sorted(label_counts.items())},
        style_counts={str(k): int(v) for k, v in sorted(style_counts.items())},
        variant_count=int(variants),
        labeled_cells=int(labeled_cells),
    )
