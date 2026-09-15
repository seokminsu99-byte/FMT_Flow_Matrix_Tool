# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Uses OpenCV: https://github.com/opencv/opencv (Apache-2.0); opencv-python wheel notices apply.
# Uses NumPy: https://github.com/numpy/numpy (BSD-3-Clause plus bundled notices).
# Uses scikit-image: https://github.com/scikit-image/scikit-image (BSD-3-Clause plus per-file notices).
# Uses PyTorch: https://github.com/pytorch/pytorch (BSD-3-Clause plus bundled notices).
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
from __future__ import annotations

import hashlib
import os
import shutil
import threading
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import torch.nn as nn
import torch.nn.functional as F
# Library implementation of 2-D Zhang-Suen thinning; see docs/PROVENANCE.md.
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from solution import DEFAULT_INK_THRESHOLD, FeatureConfig, build_features, compute_direction_matrix_with_meta, derive_grid_B, occupancy_from_binary

CONTEXT_RADIUS = 2
NUM_CLASSES = 5
DATASET_SCHEMA_VERSION = 5
MODEL_SCHEMA_VERSION = 2
SMALL_DATASET_THRESHOLD = 96
MIN_CLASS_SUPPORT_FOR_MLP = 6
MIN_CLASS_SUPPORT_FOR_OVERRIDE = 4
MIN_DEPLOYMENT_SUPPORT_PER_CLASS = 6
MIN_NEURAL_TRAINING_GROUPS = 8
PATCH_CHANNELS = 3
RAW_PATCH_SIZE = 16
PATCH_CONTEXT_RADIUS_CELLS = 1
PATCH_SIGNAL_THRESHOLD = 0.055
_DATASET_IO_LOCK = threading.RLock()
_TRAINING_LOCK = threading.RLock()
_DISTANCE_WORK_BYTES = 16 * 1024 * 1024
_INFERENCE_BATCH_SIZE = 64


def get_torch_runtime_info() -> Dict[str, object]:
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    device_names = [torch.cuda.get_device_name(i) for i in range(device_count)] if cuda_available else []
    return {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_names": device_names,
        "is_cpu_build": "+cpu" in str(torch.__version__) or torch.version.cuda is None,
    }


def resolve_compute_device(device_preference: str = "auto") -> Tuple[torch.device, str]:
    pref = str(device_preference or "auto").strip().lower()
    info = get_torch_runtime_info()
    if pref == "cuda":
        if info["cuda_available"]:
            return torch.device("cuda"), "cuda"
        return torch.device("cpu"), "cpu (CUDA 요청이 있었지만 현재 torch/장치에서 사용할 수 없음)"
    if pref == "cpu":
        return torch.device("cpu"), "cpu"
    if info["cuda_available"]:
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def _stable_group_id(group_key: str) -> np.int64:
    digest = hashlib.blake2b(group_key.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)
    return np.int64(value)


def _atomic_savez(path: str | Path, **arrays: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp.npz")
    try:
        np.savez(temp_path, **arrays)
        os.replace(temp_path, target)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def _atomic_save_model(checkpoint: object, path: str | Path) -> None:
    """Publish a complete checkpoint while preserving the previous file on failure."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temp_path.open("wb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _validated_training_arrays(
    X: object, X_patch: object, y: object, group_ids: object,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(X, dtype=np.float32)
    labels = np.asarray(y)
    groups = np.asarray(group_ids)
    expected_dim = expected_feature_dim(CONTEXT_RADIUS)
    if features.ndim != 2 or features.shape[1] != expected_dim:
        raise RuntimeError(f"학습 특징 모양이 맞지 않습니다: {features.shape}, 필요 열 수: {expected_dim}.")
    if labels.ndim != 1 or len(labels) != len(features) or not np.isin(labels, np.arange(NUM_CLASSES)).all():
        raise RuntimeError("학습 레이블은 특징과 같은 개수의 정수 0~4여야 합니다.")
    if groups.ndim != 1 or len(groups) != len(labels) or not np.issubdtype(groups.dtype, np.integer):
        raise RuntimeError("학습 그룹 정보는 레이블과 같은 개수의 정수여야 합니다.")
    patches = _normalize_patch_array(X_patch, len(labels))
    if not np.isfinite(features).all() or not np.isfinite(patches).all():
        raise RuntimeError("학습 특징과 패치에 NaN 또는 무한대가 포함되어 있습니다.")
    return features, patches, labels.astype(np.int64), groups.astype(np.int64)


def _load_training_snapshot(dataset_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read one consistent dataset; invalid files remain untouched for recovery."""
    with _DATASET_IO_LOCK:
        try:
            with np.load(dataset_path, allow_pickle=False) as data:
                schema = np.asarray(data.get("schema_version", []))
                if schema.size != 1 or int(schema.reshape(-1)[0]) != DATASET_SCHEMA_VERSION:
                    raise RuntimeError(f"학습 데이터 버전이 맞지 않습니다 (필요 버전: {DATASET_SCHEMA_VERSION}).")
                labels = data["y"]
                return _validated_training_arrays(
                    data["X"], data["X_patch"] if "X_patch" in data else None, labels,
                    data["group_ids"] if "group_ids" in data else np.arange(len(labels), dtype=np.int64),
                )
        except Exception as exc:
            raise RuntimeError(
                f"학습 데이터를 읽을 수 없습니다: {dataset_path}. 기존 파일은 보존했습니다. "
                f"백업 또는 호환되는 데이터를 확인한 뒤 다시 시도하세요. 원인: {exc}"
            ) from exc


def _bbox_to_cell_ranges(
    bbox: Tuple[int, int, int, int],
    A: int,
    B: int,
    H: int,
    W: int,
) -> Tuple[range, range]:
    x1, y1, x2, y2 = bbox
    row0 = int(np.clip(np.floor(y1 / max(H / float(A), 1e-6)), 0, A - 1))
    row1 = int(np.clip(np.ceil(y2 / max(H / float(A), 1e-6)), 1, A))
    col0 = int(np.clip(np.floor(x1 / max(W / float(B), 1e-6)), 0, B - 1))
    col1 = int(np.clip(np.ceil(x2 / max(W / float(B), 1e-6)), 1, B))
    return range(row0, row1), range(col0, col1)


def _compute_cell_means(image: np.ndarray, A: int, B: int) -> np.ndarray:
    H, W = image.shape[:2]
    cell_h = H / float(A)
    cell_w = W / float(B)
    out = np.zeros((A, B), dtype=np.float32)
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
            cell = image[y1:y2, x1:x2]
            if cell.size:
                out[i, j] = float(np.mean(cell > 0))
    return out


def _compute_morphological_features(
    img: np.ndarray,
    A: int,
    ink_thr: float = DEFAULT_INK_THRESHOLD,
    detection_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute grid-aligned features used by the lightweight learner."""
    H, W = img.shape[:2]
    B = derive_grid_B(A, W, H)
    cache_ok = (
        isinstance(detection_cache, dict)
        and int(detection_cache.get("A", -1)) == int(A)
    )
    if cache_ok:
        cached_meta = detection_cache.get("meta", {})
        morph_dir = np.asarray(detection_cache.get("morph_dir", np.zeros((A, B), dtype=np.uint8)), dtype=np.uint8)
        detected_occ = np.asarray(detection_cache.get("occ", np.zeros((A, B), dtype=np.uint8)), dtype=np.uint8)
        bboxes = list(detection_cache.get("bboxes", []))
        bw = np.asarray(cached_meta.get("bw", np.zeros((H, W), dtype=np.uint8)), dtype=np.uint8)
        edges = np.asarray(cached_meta.get("edges", np.zeros((H, W), dtype=np.uint8)), dtype=np.uint8)
        meta = cached_meta if isinstance(cached_meta, dict) else {}
        if bw.shape != (H, W) or edges.shape != (H, W) or morph_dir.shape != (A, B):
            cache_ok = False
    if not cache_ok:
        feature_maps = build_features(img, FeatureConfig(A=A, use_edges=True, use_gray=False))
        bw = feature_maps["bw"]
        edges = feature_maps["edges"]
        morph_dir, detected_occ, bboxes, meta = compute_direction_matrix_with_meta(img, A, ink_thr=ink_thr)
    occ = np.maximum(detected_occ, occupancy_from_binary(bw, A, B, ink_thr=ink_thr))

    mean_bw = _compute_cell_means(bw, A, B)
    mean_edges = _compute_cell_means(edges, A, B)
    skel = skeletonize((bw > 0).astype(bool)).astype(np.uint8) * 255
    mean_skeleton = _compute_cell_means(skel, A, B)

    head_hint = np.zeros((A, B), dtype=np.float32)
    for bbox in bboxes:
        rows, cols = _bbox_to_cell_ranges(bbox, A, B, H, W)
        for i in rows:
            for j in cols:
                head_hint[i, j] = 1.0

    confidence = np.asarray(meta.get("confidence", np.zeros((A, B), dtype=np.float32)), dtype=np.float32)
    return morph_dir, occ.astype(np.uint8), mean_bw, mean_edges, mean_skeleton, head_hint, confidence


def expected_patch_shape() -> Tuple[int, int, int]:
    return PATCH_CHANNELS, RAW_PATCH_SIZE, RAW_PATCH_SIZE


def _empty_patch_tensor() -> np.ndarray:
    return np.zeros(expected_patch_shape(), dtype=np.float32)


def _normalize_patch_array(X_patch: np.ndarray | None, n_samples: int) -> np.ndarray:
    if X_patch is None:
        return np.zeros((n_samples, *expected_patch_shape()), dtype=np.float32)
    arr = np.asarray(X_patch, dtype=np.float32)
    expected = expected_patch_shape()
    if arr.ndim == 2 and arr.shape[1] == int(np.prod(expected)):
        arr = arr.reshape(n_samples, *expected)
    if arr.shape != (n_samples, *expected):
        raise RuntimeError(
            f"저장된 패치 데이터 모양이 현재 코드와 맞지 않습니다. 현재 데이터: {arr.shape}, 필요 모양: {(n_samples, *expected)}"
        )
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def _sample_identity_key(feature: np.ndarray, patch: np.ndarray) -> bytes:
    feature_arr = np.asarray(feature, dtype=np.float32)
    patch_arr = np.asarray(patch, dtype=np.float32)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(feature_arr).view(np.uint8))
    digest.update(np.ascontiguousarray(patch_arr).view(np.uint8))
    return digest.digest()


def _deduplicate_training_arrays(
    X: np.ndarray,
    X_patch: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Keep the newest label for identical feature+patch samples."""
    X_arr = np.asarray(X, dtype=np.float32)
    X_patch_arr = _normalize_patch_array(X_patch, int(len(y)))
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    group_arr = np.asarray(group_ids, dtype=np.int64).reshape(-1)
    if group_arr.shape[0] != y_arr.shape[0]:
        group_arr = np.arange(y_arr.shape[0], dtype=np.int64)
    if y_arr.size == 0:
        return X_arr, X_patch_arr, y_arr, group_arr, {"removed": 0, "conflicts": 0}

    latest_index: Dict[bytes, int] = {}
    conflicts = 0
    for idx in range(int(y_arr.shape[0])):
        key = _sample_identity_key(X_arr[idx], X_patch_arr[idx])
        prev = latest_index.get(key)
        if prev is not None and int(y_arr[prev]) != int(y_arr[idx]):
            conflicts += 1
        latest_index[key] = idx

    keep_indices = np.asarray(sorted(latest_index.values()), dtype=np.int64)
    removed = int(y_arr.shape[0] - keep_indices.shape[0])
    return (
        X_arr[keep_indices].astype(np.float32),
        X_patch_arr[keep_indices].astype(np.float32),
        y_arr[keep_indices].astype(np.int64),
        group_arr[keep_indices].astype(np.int64),
        {"removed": removed, "conflicts": int(conflicts)},
    )


def _cell_pixel_bounds(i: int, j: int, A: int, B: int, H: int, W: int) -> Tuple[int, int, int, int]:
    cell_h = H / float(A)
    cell_w = W / float(B)
    y1 = int(round(i * cell_h))
    y2 = int(round((i + 1) * cell_h))
    x1 = int(round(j * cell_w))
    x2 = int(round((j + 1) * cell_w))
    y1 = max(0, min(H - 1, y1))
    y2 = max(y1 + 1, min(H, y2))
    x1 = max(0, min(W - 1, x1))
    x2 = max(x1 + 1, min(W, x2))
    return x1, y1, x2, y2


def _compute_patch_source_maps(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = FeatureConfig(use_edges=True, use_gray=False)
    feature_maps = build_features(img, cfg)
    if img.ndim == 2:
        gray = img.copy()
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_inv = 1.0 - (gray.astype(np.float32) / 255.0)
    bw = (feature_maps["bw"] > 0).astype(np.float32)
    edges = (feature_maps["edges"] > 0).astype(np.float32)
    return gray_inv.astype(np.float32), bw.astype(np.float32), edges.astype(np.float32)


def _extract_cell_patch_tensor(
    gray_inv: np.ndarray,
    bw: np.ndarray,
    edges: np.ndarray,
    i: int,
    j: int,
    A: int,
    B: int,
    patch_size: int = RAW_PATCH_SIZE,
    context_radius_cells: int = PATCH_CONTEXT_RADIUS_CELLS,
) -> np.ndarray:
    H, W = gray_inv.shape[:2]
    row0 = max(0, int(i) - int(context_radius_cells))
    row1 = min(int(A) - 1, int(i) + int(context_radius_cells))
    col0 = max(0, int(j) - int(context_radius_cells))
    col1 = min(int(B) - 1, int(j) + int(context_radius_cells))
    x1, y1, _, _ = _cell_pixel_bounds(row0, col0, A, B, H, W)
    _, _, x2, y2 = _cell_pixel_bounds(row1, col1, A, B, H, W)
    if x2 <= x1 or y2 <= y1:
        return _empty_patch_tensor()
    gray_crop = gray_inv[y1:y2, x1:x2]
    bw_crop = bw[y1:y2, x1:x2]
    edge_crop = edges[y1:y2, x1:x2]
    if gray_crop.size == 0 or bw_crop.size == 0 or edge_crop.size == 0:
        return _empty_patch_tensor()
    resized_gray = cv2.resize(gray_crop, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    resized_bw = cv2.resize(bw_crop, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    resized_edge = cv2.resize(edge_crop, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    patch = np.stack([resized_gray, resized_bw, resized_edge], axis=0).astype(np.float32)
    return np.clip(patch, 0.0, 1.0)


def _combine_struct_and_patch_features(X_struct: np.ndarray, X_patch: np.ndarray) -> np.ndarray:
    if X_struct.size == 0:
        return np.zeros((0, int(X_struct.shape[1]) + int(np.prod(expected_patch_shape()))), dtype=np.float32)
    flat_patch = X_patch.reshape(X_patch.shape[0], -1).astype(np.float32)
    centered_patch = flat_patch - 0.5
    return np.concatenate([X_struct.astype(np.float32), centered_patch.astype(np.float32)], axis=1)


def _patch_signal_strength(patch: np.ndarray) -> float:
    if patch.size == 0:
        return 0.0
    gray_strength = float(np.mean(patch[0]))
    bw_strength = float(np.mean(patch[1]))
    edge_strength = float(np.mean(patch[2]))
    return float(np.clip(0.55 * gray_strength + 0.30 * bw_strength + 0.15 * edge_strength, 0.0, 1.0))


def _one_hot_orientation(direction: int) -> List[float]:
    out = [0.0, 0.0, 0.0, 0.0]
    if 1 <= direction <= 4:
        out[direction - 1] = 1.0
    return out


def expected_feature_dim(context_radius: int = CONTEXT_RADIUS) -> int:
    radius = max(0, int(context_radius))
    local_cell_dim = len(_one_hot_orientation(0)) + 6
    global_dim = 10
    return ((2 * radius + 1) ** 2) * local_cell_dim + global_dim


def _safe_value(grid: np.ndarray, i: int, j: int, default: float = 0.0) -> float:
    if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
        return float(grid[i, j])
    return float(default)


def _expanded_training_cells(
    corrected_labels: Dict[Tuple[int, int], int],
    reference_matrix: np.ndarray | None,
    A: int,
    B: int,
    radius: int = 1,
) -> List[Tuple[int, int]]:
    cells: set[Tuple[int, int]] = set()
    for i, j in corrected_labels:
        if not (0 <= i < A and 0 <= j < B):
            continue
        cells.add((i, j))
        if reference_matrix is None:
            continue
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                row = i + di
                col = j + dj
                if 0 <= row < A and 0 <= col < B:
                    cells.add((row, col))
    return sorted(cells)


def _compute_component_ratio_map(occ: np.ndarray) -> np.ndarray:
    if occ.size == 0:
        return np.zeros_like(occ, dtype=np.float32)
    labels_count, labels = cv2.connectedComponents(occ.astype(np.uint8), connectivity=4)
    if labels_count <= 1:
        return np.zeros_like(occ, dtype=np.float32)
    counts = np.bincount(labels.reshape(-1), minlength=labels_count).astype(np.float32)
    ratio = counts[labels] / max(1.0, float(occ.shape[0] * occ.shape[1]))
    ratio[labels == 0] = 0.0
    return ratio.astype(np.float32)


def _compute_direction_run_map(morph_dir: np.ndarray, occ: np.ndarray, max_steps: int = 6) -> np.ndarray:
    A, B = morph_dir.shape
    out = np.zeros((A, B), dtype=np.float32)
    dir_to_step = {1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
    for i in range(A):
        for j in range(B):
            if int(occ[i, j]) != 1 and int(morph_dir[i, j]) == 0:
                continue
            current = (i, j)
            seen = {current}
            steps = 0
            for _ in range(max_steps):
                direction = int(morph_dir[current])
                step = dir_to_step.get(direction)
                if step is None:
                    break
                nxt = (current[0] + step[0], current[1] + step[1])
                if not (0 <= nxt[0] < A and 0 <= nxt[1] < B):
                    break
                if int(occ[nxt]) != 1:
                    break
                steps += 1
                if nxt in seen:
                    break
                seen.add(nxt)
                current = nxt
            out[i, j] = float(steps / max(1, max_steps))
    return out


def _compute_degree_maps(morph_dir: np.ndarray, occ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A, B = morph_dir.shape
    occ_degree = np.zeros((A, B), dtype=np.float32)
    incoming_degree = np.zeros((A, B), dtype=np.float32)
    dir_to_step = {1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
    for i in range(A):
        for j in range(B):
            if int(occ[i, j]) != 1:
                continue
            degree = 0
            for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                ni = i + di
                nj = j + dj
                if 0 <= ni < A and 0 <= nj < B and int(occ[ni, nj]) == 1:
                    degree += 1
                    step = dir_to_step.get(int(morph_dir[ni, nj]))
                    if step is not None and (ni + step[0], nj + step[1]) == (i, j):
                        incoming_degree[i, j] += 1.0
            occ_degree[i, j] = float(degree)
    if np.max(occ_degree) > 0:
        occ_degree /= float(np.max(occ_degree))
    if np.max(incoming_degree) > 0:
        incoming_degree /= float(np.max(incoming_degree))
    return occ_degree.astype(np.float32), incoming_degree.astype(np.float32)


def _compute_global_context_maps(
    morph_dir: np.ndarray,
    occ: np.ndarray,
    head_hint: np.ndarray,
) -> Dict[str, np.ndarray]:
    row_occ_density = np.mean(occ > 0, axis=1).astype(np.float32)
    col_occ_density = np.mean(occ > 0, axis=0).astype(np.float32)
    row_head_density = np.mean(head_hint > 0, axis=1).astype(np.float32)
    col_head_density = np.mean(head_hint > 0, axis=0).astype(np.float32)
    component_ratio = _compute_component_ratio_map(occ)
    direction_run = _compute_direction_run_map(morph_dir, occ)
    occ_degree, incoming_degree = _compute_degree_maps(morph_dir, occ)
    return {
        "row_occ_density": row_occ_density,
        "col_occ_density": col_occ_density,
        "row_head_density": row_head_density,
        "col_head_density": col_head_density,
        "component_ratio": component_ratio,
        "direction_run": direction_run,
        "occ_degree": occ_degree,
        "incoming_degree": incoming_degree,
    }


def _cell_feature_vector(
    morph_dir: np.ndarray,
    occ: np.ndarray,
    mean_bw: np.ndarray,
    mean_edges: np.ndarray,
    mean_skeleton: np.ndarray,
    head_hint: np.ndarray,
    confidence: np.ndarray,
    global_maps: Dict[str, np.ndarray],
    i: int,
    j: int,
    context_radius: int = CONTEXT_RADIUS,
) -> List[float]:
    features: List[float] = []
    for di in range(-context_radius, context_radius + 1):
        for dj in range(-context_radius, context_radius + 1):
            row = i + di
            col = j + dj
            direction = int(_safe_value(morph_dir, row, col, default=0.0))
            features.extend(_one_hot_orientation(direction))
            features.extend(
                [
                    _safe_value(occ, row, col, default=0.0),
                    _safe_value(mean_bw, row, col, default=0.0),
                    _safe_value(mean_edges, row, col, default=0.0),
                    _safe_value(mean_skeleton, row, col, default=0.0),
                    _safe_value(head_hint, row, col, default=0.0),
                    _safe_value(confidence, row, col, default=0.0),
                ]
            )
    features.extend(
        [
            float(i / max(1, morph_dir.shape[0] - 1)),
            float(j / max(1, morph_dir.shape[1] - 1)),
            float(_safe_value(global_maps["component_ratio"], i, j, default=0.0)),
            float(_safe_value(global_maps["direction_run"], i, j, default=0.0)),
            float(_safe_value(global_maps["occ_degree"], i, j, default=0.0)),
            float(_safe_value(global_maps["incoming_degree"], i, j, default=0.0)),
            float(global_maps["row_occ_density"][i]) if 0 <= i < global_maps["row_occ_density"].shape[0] else 0.0,
            float(global_maps["col_occ_density"][j]) if 0 <= j < global_maps["col_occ_density"].shape[0] else 0.0,
            float(global_maps["row_head_density"][i]) if 0 <= i < global_maps["row_head_density"].shape[0] else 0.0,
            float(global_maps["col_head_density"][j]) if 0 <= j < global_maps["col_head_density"].shape[0] else 0.0,
        ]
    )
    return features


def _select_reliable_context_cells(
    corrected_cells: List[Tuple[int, int]],
    reference_matrix: np.ndarray | None,
    morph_dir: np.ndarray,
    occ: np.ndarray,
    mean_bw: np.ndarray,
    mean_edges: np.ndarray,
    mean_skeleton: np.ndarray,
    head_hint: np.ndarray,
    confidence: np.ndarray,
    ink_thr: float,
    radius: int = 2,
) -> List[Tuple[int, int, int]]:
    """Collect a few highly reliable nearby labels to stabilize training.

    We do not relabel entire neighborhoods. Instead, we only keep cells whose current
    morphology is already very trustworthy, so the model learns both "change this cell"
    and "leave these nearby cells alone".
    """
    if reference_matrix is None:
        return []
    reference = np.asarray(reference_matrix)
    if reference.shape != morph_dir.shape:
        return []

    corrected_set = {(int(i), int(j)) for i, j in corrected_cells}
    selected: Dict[Tuple[int, int], Tuple[float, int]] = {}
    max_zero_per_anchor = 2
    max_dir_per_anchor = 2

    for anchor_i, anchor_j in corrected_cells:
        zero_candidates: List[Tuple[float, int, int, int]] = []
        dir_candidates: List[Tuple[float, int, int, int]] = []
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                if di == 0 and dj == 0:
                    continue
                row = anchor_i + di
                col = anchor_j + dj
                if not (0 <= row < morph_dir.shape[0] and 0 <= col < morph_dir.shape[1]):
                    continue
                if (row, col) in corrected_set:
                    continue

                distance = abs(di) + abs(dj)
                if distance <= 0 or distance > radius + 1:
                    continue

                label = int(reference[row, col])
                if label not in (0, 1, 2, 3, 4):
                    continue

                base_dir = int(morph_dir[row, col])
                occ_val = int(occ[row, col])
                conf = float(confidence[row, col])
                ink = float(mean_bw[row, col])
                edge = float(mean_edges[row, col])
                skeleton = float(mean_skeleton[row, col])
                head = float(head_hint[row, col])
                distance_penalty = 0.06 * float(distance)

                if label == 0:
                    if base_dir != 0 or occ_val != 0:
                        continue
                    if ink > max(ink_thr * 8.0, 0.025):
                        continue
                    if edge > 0.08 or skeleton > 0.05 or head > 0.0:
                        continue
                    score = 1.0 - (ink * 6.0 + edge * 2.0 + skeleton * 2.0 + distance_penalty)
                    zero_candidates.append((score, row, col, label))
                    continue

                if occ_val != 1 or base_dir != label:
                    continue
                if conf < 0.82:
                    continue
                if ink < max(ink_thr * 3.0, 0.012):
                    continue
                score = conf + min(0.20, ink * 0.35) + min(0.10, skeleton * 0.25) - distance_penalty
                dir_candidates.append((score, row, col, label))

        zero_candidates.sort(key=lambda item: item[0], reverse=True)
        dir_candidates.sort(key=lambda item: item[0], reverse=True)
        for score, row, col, label in zero_candidates[:max_zero_per_anchor] + dir_candidates[:max_dir_per_anchor]:
            key = (row, col)
            previous = selected.get(key)
            if previous is None or score > previous[0]:
                selected[key] = (float(score), int(label))

    ordered = sorted(selected.items(), key=lambda item: item[1][0], reverse=True)
    return [(row, col, label) for (row, col), (_score, label) in ordered]


def extract_features_and_labels(
    img: np.ndarray,
    A: int,
    corrected_labels: Dict[Tuple[int, int], int],
    reference_matrix: np.ndarray | None = None,
    context_radius: int = CONTEXT_RADIUS,
    corrected_weight: int = 1,
    ink_thr: float = DEFAULT_INK_THRESHOLD,
    detection_cache: Optional[Dict[str, Any]] = None,
    include_reliable_context: bool = False,
) -> Tuple[List[List[float]], List[np.ndarray], List[int]]:
    """Turn user edits into training samples.

    By default, only user-corrected cells are used as labels. The detector/FMT
    output is a feature source and proposal, not ground truth. Nearby automatic
    context labels remain opt-in because they can inject the detector's current
    bias into the learner.
    """
    morph_dir, occ, mean_bw, mean_edges, mean_skeleton, head_hint, confidence = _compute_morphological_features(
        img,
        A,
        ink_thr=ink_thr,
        detection_cache=detection_cache,
    )
    H, W = img.shape[:2]
    B = derive_grid_B(A, W, H)
    global_maps = _compute_global_context_maps(morph_dir, occ, head_hint)
    X_list: List[List[float]] = []
    X_patch_list: List[np.ndarray] = []
    y_list: List[int] = []
    gray_inv, bw_patch_map, edge_patch_map = _compute_patch_source_maps(img)
    cache_meta = detection_cache.get("meta", {}) if isinstance(detection_cache, dict) else {}
    has_detection_aux = isinstance(cache_meta, dict) and (
        "overlap_count" in cache_meta or "support_level" in cache_meta
    )
    overlap_count = np.asarray(cache_meta.get("overlap_count", np.zeros((A, B), dtype=np.int32)), dtype=np.int32)
    support_level = np.asarray(
        cache_meta.get("support_level", np.where(occ > 0, 2, 0).astype(np.uint8)),
        dtype=np.uint8,
    )
    corrected_cells = sorted(
        {
            (int(i), int(j))
            for i, j in corrected_labels
            if 0 <= int(i) < int(A) and 0 <= int(j) < int(B)
        }
    )
    reliable_context_cells: List[Tuple[int, int, int]] = []
    if include_reliable_context:
        reliable_context_cells = _select_reliable_context_cells(
            corrected_cells,
            reference_matrix,
            morph_dir,
            occ,
            mean_bw,
            mean_edges,
            mean_skeleton,
            head_hint,
            confidence,
            ink_thr=ink_thr,
        )
        if has_detection_aux:
            reliable_context_cells = [
                (i, j, label)
                for i, j, label in reliable_context_cells
                if int(overlap_count[i, j]) <= 1 and int(support_level[i, j]) >= 2
            ]

    for i, j in corrected_cells:
        if not (0 <= i < A and 0 <= j < B):
            continue
        label = int(corrected_labels[(i, j)])
        if label not in (0, 1, 2, 3, 4):
            continue
        feature = _cell_feature_vector(
            morph_dir,
            occ,
            mean_bw,
            mean_edges,
            mean_skeleton,
            head_hint,
            confidence,
            global_maps,
            i,
            j,
            context_radius=max(0, int(context_radius)),
        )
        repeats = max(1, int(corrected_weight)) if (i, j) in corrected_labels else 1
        hard_example = (
            int(overlap_count[i, j]) > 1
            or float(confidence[i, j]) < 0.55
            or (has_detection_aux and int(support_level[i, j]) <= 1)
            or (reference_matrix is not None and int(reference_matrix[i, j]) != int(morph_dir[i, j]))
        )
        if hard_example:
            repeats = min(4, repeats + 1 + (1 if int(overlap_count[i, j]) > 1 else 0))
        patch = _extract_cell_patch_tensor(gray_inv, bw_patch_map, edge_patch_map, i, j, A, B)
        for _ in range(repeats):
            X_list.append(feature)
            X_patch_list.append(patch.copy())
            y_list.append(label)

    for i, j, label in reliable_context_cells:
        feature = _cell_feature_vector(
            morph_dir,
            occ,
            mean_bw,
            mean_edges,
            mean_skeleton,
            head_hint,
            confidence,
            global_maps,
            i,
            j,
            context_radius=max(0, int(context_radius)),
        )
        patch = _extract_cell_patch_tensor(gray_inv, bw_patch_map, edge_patch_map, i, j, A, B)
        X_list.append(feature)
        X_patch_list.append(patch.copy())
        y_list.append(int(label))
    return X_list, X_patch_list, y_list


def append_to_dataset(
    X_new: List[List[float]],
    X_patch_new: List[np.ndarray],
    y_new: List[int],
    dataset_path: str = "cell_dataset.npz",
    group_key: Optional[str] = None,
) -> None:
    """Append new labeled cells to the on-disk dataset."""
    if not X_new or not y_new:
        return

    X_new_arr = np.asarray(X_new, dtype=np.float32)
    X_patch_new_arr = _normalize_patch_array(np.asarray(X_patch_new, dtype=np.float32), len(y_new))
    y_new_arr = np.asarray(y_new, dtype=np.int64)
    requested_group_id = _stable_group_id(group_key) if group_key else None
    expected_dim = expected_feature_dim(CONTEXT_RADIUS)
    if X_new_arr.ndim != 2 or int(X_new_arr.shape[1]) != expected_dim:
        raise RuntimeError(
            f"feature dimension mismatch while saving corrections: got {X_new_arr.shape[1]}, expected {expected_dim}"
        )

    with _DATASET_IO_LOCK:
        if os.path.exists(dataset_path):
            with np.load(dataset_path) as data:
                schema_ok = int(np.asarray(data.get("schema_version", np.array([0], dtype=np.int32))).reshape(-1)[0]) == DATASET_SCHEMA_VERSION
                dim_ok = "X" in data and "y" in data and int(np.asarray(data["X"]).shape[1]) == int(X_new_arr.shape[1])
                if schema_ok and dim_ok:
                    X = np.concatenate([data["X"], X_new_arr], axis=0)
                    existing_patch = _normalize_patch_array(data["X_patch"] if "X_patch" in data else None, len(data["y"]))
                    X_patch = np.concatenate([existing_patch, X_patch_new_arr], axis=0)
                    y = np.concatenate([data["y"], y_new_arr], axis=0)
                    existing_group_ids = np.asarray(
                        data.get("group_ids", np.arange(len(data["y"]), dtype=np.int64)),
                        dtype=np.int64,
                    ).reshape(-1)
                    if requested_group_id is None:
                        next_group = int(existing_group_ids.max()) + 1 if existing_group_ids.size else 0
                        new_group_ids = np.full((len(y_new_arr),), next_group, dtype=np.int64)
                    else:
                        new_group_ids = np.full((len(y_new_arr),), requested_group_id, dtype=np.int64)
                    group_ids = np.concatenate([existing_group_ids, new_group_ids], axis=0)
                else:
                    X = X_new_arr
                    X_patch = X_patch_new_arr
                    y = y_new_arr
                    group_ids = np.full(
                        (len(y_new_arr),),
                        requested_group_id if requested_group_id is not None else 0,
                        dtype=np.int64,
                    )
        else:
            X = X_new_arr
            X_patch = X_patch_new_arr
            y = y_new_arr
            group_ids = np.full(
                (len(y_new_arr),),
                requested_group_id if requested_group_id is not None else 0,
                dtype=np.int64,
            )
        X, X_patch, y, group_ids, _dedupe_stats = _deduplicate_training_arrays(X, X_patch, y, group_ids)
        _atomic_savez(
            dataset_path,
            X=X,
            X_patch=X_patch,
            y=y,
            group_ids=group_ids,
            schema_version=np.array([DATASET_SCHEMA_VERSION], dtype=np.int32),
            num_classes=np.array([NUM_CLASSES], dtype=np.int32),
            patch_channels=np.array([PATCH_CHANNELS], dtype=np.int32),
            patch_size=np.array([RAW_PATCH_SIZE], dtype=np.int32),
        )


def reset_training_state(
    dataset_path: str = "cell_dataset.npz",
    model_path: str = "cell_model.pt",
) -> Dict[str, bool]:
    """Reset saved dataset/model files, backing them up into bak/ when possible."""
    def _backup_target(path: Path, kind: str) -> Path:
        bak_dir = path.resolve().parent / "bak"
        bak_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = bak_dir / f"{path.stem}_{kind}_reset_{timestamp}{path.suffix}"
        if not candidate.exists():
            return candidate
        index = 1
        while True:
            candidate = bak_dir / f"{path.stem}_{kind}_reset_{timestamp}_{index}{path.suffix}"
            if not candidate.exists():
                return candidate
            index += 1

    def _disable_file(path_str: str, kind: str) -> bool:
        path = Path(path_str)
        if not path.exists():
            return False
        try:
            shutil.copy2(path, _backup_target(path, kind))
        except OSError:
            pass
        try:
            if kind == "dataset":
                feature_dim = expected_feature_dim(CONTEXT_RADIUS)
                _atomic_savez(
                    path,
                    X=np.zeros((0, feature_dim), dtype=np.float32),
                    X_patch=np.zeros((0, *expected_patch_shape()), dtype=np.float32),
                    y=np.zeros((0,), dtype=np.int64),
                    group_ids=np.zeros((0,), dtype=np.int64),
                    disabled=np.array([1], dtype=np.uint8),
                    schema_version=np.array([DATASET_SCHEMA_VERSION], dtype=np.int32),
                    num_classes=np.array([NUM_CLASSES], dtype=np.int32),
                    patch_channels=np.array([PATCH_CHANNELS], dtype=np.int32),
                    patch_size=np.array([RAW_PATCH_SIZE], dtype=np.int32),
                )
            else:
                path.write_text("DISABLED\n", encoding="utf-8")
            return True
        except OSError:
            return False

    removed = {"dataset": False, "model": False}
    removed["dataset"] = _disable_file(dataset_path, "dataset")
    removed["model"] = _disable_file(model_path, "model")
    return removed


@dataclass
class CellDataset(Dataset):
    X_struct: np.ndarray
    X_patch: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return int(self.X_struct.shape[0])

    def __getitem__(self, idx: int):
        return self.X_struct[idx], self.X_patch[idx], self.y[idx]


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.10):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


class CellOrientationNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 260,
        patch_channels: int = PATCH_CHANNELS,
        patch_size: int = RAW_PATCH_SIZE,
        hidden_dim: int = 224,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.patch_channels = int(patch_channels)
        self.patch_size = int(patch_size)
        self.struct_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.12),
            ResidualMLPBlock(hidden_dim, dropout=0.10),
            ResidualMLPBlock(hidden_dim, dropout=0.08),
        )
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(self.patch_channels, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(48, 72, kernel_size=3, padding=1),
            nn.BatchNorm2d(72),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 72, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.15),
            ResidualMLPBlock(256, dropout=0.10),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_struct: torch.Tensor, x_patch: torch.Tensor | None = None) -> torch.Tensor:
        if x_patch is None:
            x_patch = torch.zeros(
                (x_struct.shape[0], self.patch_channels, self.patch_size, self.patch_size),
                dtype=x_struct.dtype,
                device=x_struct.device,
            )
        struct_feat = self.struct_encoder(x_struct)
        patch_feat = self.patch_encoder(x_patch)
        return self.fusion(torch.cat([struct_feat, patch_feat], dim=1))


def _standardize(
    X_train: np.ndarray,
    X_other: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    X_train_std = (X_train - mean) / std
    X_other_std = None if X_other is None else (X_other - mean) / std
    return X_train_std, X_other_std, mean.astype(np.float32), std.astype(np.float32)


def _apply_standardization(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = np.where(std < 1e-6, 1.0, std)
    return ((X - mean) / safe_std).astype(np.float32)


def _counts_dict(y: np.ndarray) -> Dict[int, int]:
    if y.size == 0:
        return {}
    return {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}


def _coverage_mode(y: np.ndarray) -> Tuple[str, str]:
    if y.size == 0:
        return "restricted", "학습 샘플이 비어 있습니다."
    counts = _counts_dict(y)
    if not counts:
        return "restricted", "학습된 클래스 샘플이 없어 형태 기반만 신뢰합니다."
    min_support = min(counts.values())
    if int(y.shape[0]) < SMALL_DATASET_THRESHOLD:
        return "restricted", f"학습 샘플이 {int(y.shape[0])}개로 적어 예시 기반 보정을 우선 사용합니다."
    if len(counts) < NUM_CLASSES:
        return "restricted", f"학습된 클래스 종류가 {len(counts)}개뿐이라 보수적으로 적용합니다."
    if min_support < MIN_CLASS_SUPPORT_FOR_MLP:
        return "restricted", f"일부 클래스 샘플이 {min_support}개뿐이라 보수적으로 적용합니다."
    return "normal", ""


def _local_pipe_support_score(
    mean_bw: np.ndarray,
    mean_edges: np.ndarray,
    mean_skeleton: np.ndarray,
    head_hint: np.ndarray,
    global_maps: Dict[str, np.ndarray],
    i: int,
    j: int,
) -> float:
    return float(
        np.clip(
            0.36 * float(mean_bw[i, j])
            + 0.12 * float(mean_edges[i, j])
            + 0.18 * float(mean_skeleton[i, j])
            + 0.16 * float(head_hint[i, j])
            + 0.10 * float(global_maps["direction_run"][i, j])
            + 0.05 * float(global_maps["occ_degree"][i, j])
            + 0.03 * float(global_maps["incoming_degree"][i, j]),
            0.0,
            1.0,
        )
    )


def _has_local_directional_signal(
    mean_bw: np.ndarray,
    mean_edges: np.ndarray,
    mean_skeleton: np.ndarray,
    head_hint: np.ndarray,
    global_maps: Dict[str, np.ndarray],
    i: int,
    j: int,
) -> bool:
    return bool(
        float(head_hint[i, j]) >= 0.45
        or float(mean_skeleton[i, j]) >= 0.12
        or float(global_maps["direction_run"][i, j]) >= 0.38
        or float(global_maps["occ_degree"][i, j]) >= 0.60
        or (float(mean_bw[i, j]) >= 0.16 and float(mean_edges[i, j]) >= 0.05)
    )


def checkpoint_deployment_status(checkpoint: Dict[str, object]) -> Tuple[bool, str]:
    model_kind = str(checkpoint.get("model_kind", "mlp")).strip().lower()
    coverage_mode = str(checkpoint.get("coverage_mode", "restricted")).strip().lower()
    seen_labels_arr = np.asarray(checkpoint.get("seen_labels", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    class_counts = np.asarray(checkpoint.get("class_counts", np.zeros((NUM_CLASSES,), dtype=np.int32)), dtype=np.int32).reshape(-1)

    if model_kind != "mlp":
        return False, "배포형 모드는 사전학습된 신경망 모델만 사용합니다."
    if coverage_mode == "restricted":
        return False, "현재 모델은 소량 사용자 데이터 보정용 제한 모드입니다."

    seen_labels = {int(v) for v in seen_labels_arr.tolist()}
    missing_labels = [label for label in range(NUM_CLASSES) if label not in seen_labels]
    if missing_labels:
        return False, "현재 모델은 일부 방향 또는 빈칸 클래스를 포함하지 않아 배포형으로 사용할 수 없습니다."

    if class_counts.shape[0] < NUM_CLASSES:
        return False, "현재 모델의 클래스 통계가 손상되어 있습니다."
    weak_labels = [label for label in range(NUM_CLASSES) if int(class_counts[label]) < MIN_DEPLOYMENT_SUPPORT_PER_CLASS]
    if weak_labels:
        return False, "현재 모델은 일부 클래스 표본 수가 너무 적어 배포형으로 사용하지 않습니다."

    if "model" not in checkpoint:
        return False, "현재 모델 가중치가 없어 배포형으로 사용할 수 없습니다."
    return True, ""


def _pairwise_nearest_distances(X: np.ndarray) -> np.ndarray:
    n_samples = int(X.shape[0])
    if n_samples <= 1:
        return np.full((n_samples,), np.inf, dtype=np.float32)
    diff = X[:, None, :] - X[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float64), dtype=np.float64)
    np.fill_diagonal(dist, np.inf)
    dist[dist <= 1e-8] = np.inf
    return np.min(dist, axis=1).astype(np.float32)


def _build_reference_artifacts(
    X_std: np.ndarray,
    y: np.ndarray,
) -> Dict[str, object]:
    if X_std.size == 0 or y.size == 0:
        return {
            "reference_features": np.zeros((0, 0), dtype=np.float32),
            "reference_labels": np.zeros((0,), dtype=np.int64),
            "distance_scale": 1.0,
            "distance_limit": 1.0,
            "knn_k": 1,
            "class_counts": np.zeros((NUM_CLASSES,), dtype=np.int32),
        }
    nearest = _pairwise_nearest_distances(X_std)
    finite = nearest[np.isfinite(nearest)]
    if finite.size == 0:
        distance_scale = 1.0
        distance_limit = 1.0
    else:
        distance_scale = max(float(np.median(finite)), 1e-3)
        distance_limit = max(float(np.quantile(finite, 0.85)) * 1.35, distance_scale * 1.25, 1e-3)
    knn_k = int(np.clip(round(np.sqrt(max(1, X_std.shape[0]))), 3, 9))
    return {
        "reference_features": X_std.astype(np.float32),
        "reference_labels": y.astype(np.int64),
        "distance_scale": float(distance_scale),
        "distance_limit": float(distance_limit),
        "knn_k": int(knn_k),
        "class_counts": np.bincount(y, minlength=NUM_CLASSES).astype(np.int32),
    }


def _distance_scores(
    query: np.ndarray,
    reference_features: np.ndarray,
    distance_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if reference_features.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    diff = reference_features - query
    distances = np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float64), dtype=np.float64).astype(np.float32)
    sigma = max(float(distance_scale), 1e-3)
    weights = np.exp(-((distances.astype(np.float64) ** 2) / (2.0 * sigma * sigma))).astype(np.float32)
    return distances, weights


def _predict_knn_probabilities(
    query: np.ndarray,
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    num_classes: int,
    k: int,
    distance_scale: float,
) -> Tuple[np.ndarray, float]:
    if reference_features.size == 0 or reference_labels.size == 0:
        return np.zeros((num_classes,), dtype=np.float32), float("inf")
    distances, weights = _distance_scores(query, reference_features, distance_scale)
    if distances.size == 0:
        return np.zeros((num_classes,), dtype=np.float32), float("inf")
    order = np.argsort(distances)[: max(1, min(int(k), distances.size))]
    probs = np.zeros((num_classes,), dtype=np.float32)
    for idx in order:
        probs[int(reference_labels[idx])] += float(weights[idx])
    if float(np.sum(probs)) <= 1e-8:
        probs[int(reference_labels[order[0]])] = 1.0
    else:
        probs = probs / float(np.sum(probs))
    return probs.astype(np.float32), float(distances[order[0]])


def _stratified_split_indices(
    y: np.ndarray,
    val_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = int(y.shape[0])
    all_indices = np.arange(n_samples, dtype=np.int64)
    per_class_val: List[np.ndarray] = []

    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        if cls_idx.size <= 1:
            continue
        n_val_cls = int(round(cls_idx.size * float(val_ratio)))
        n_val_cls = max(1, n_val_cls)
        n_val_cls = min(n_val_cls, cls_idx.size - 1)
        if n_val_cls > 0:
            per_class_val.append(cls_idx[:n_val_cls])

    if per_class_val:
        val_idx = np.unique(np.concatenate(per_class_val).astype(np.int64))
    else:
        rng.shuffle(all_indices)
        n_val = max(1, int(round(n_samples * float(val_ratio))))
        n_val = min(n_val, max(1, n_samples - 1))
        val_idx = np.sort(all_indices[:n_val])

    train_mask = np.ones((n_samples,), dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.flatnonzero(train_mask)
    if train_idx.size == 0 or val_idx.size == 0:
        rng.shuffle(all_indices)
        n_val = max(1, int(round(n_samples * float(val_ratio))))
        n_val = min(n_val, max(1, n_samples - 1))
        train_idx = np.sort(all_indices[:-n_val])
        val_idx = np.sort(all_indices[-n_val:])
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def _covers_all_present_classes(y_subset: np.ndarray, required_classes: np.ndarray) -> bool:
    return set(int(v) for v in np.unique(y_subset)).issuperset(int(v) for v in required_classes)


def _choose_train_val_split(
    y: np.ndarray,
    group_ids: np.ndarray,
    val_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, str]:
    present_classes = np.unique(y)
    unique_groups = np.unique(group_ids)

    # Group-preserving validation is only safe once we have enough independent correction
    # batches to keep every direction class represented in training.
    if unique_groups.size >= 4:
        shuffled_groups = unique_groups.copy()
        rng.shuffle(shuffled_groups)
        target_val = max(1, int(round(y.shape[0] * float(val_ratio))))
        val_groups: List[int] = []
        val_count = 0
        for group in shuffled_groups:
            if len(val_groups) >= max(1, len(shuffled_groups) - 1):
                break
            val_groups.append(int(group))
            val_count += int(np.count_nonzero(group_ids == group))
            if val_count >= target_val:
                val_mask = np.isin(group_ids, np.asarray(val_groups, dtype=np.int64))
                train_idx = np.flatnonzero(~val_mask)
                val_idx = np.flatnonzero(val_mask)
                if (
                    train_idx.size > 0
                    and val_idx.size > 0
                    and _covers_all_present_classes(y[train_idx], present_classes)
                ):
                    return train_idx.astype(np.int64), val_idx.astype(np.int64), "group"

    train_idx, val_idx = _stratified_split_indices(y, val_ratio, rng)
    return train_idx, val_idx, "stratified"


def _detect_prediction_collapse(
    true_counts: Dict[int, int],
    pred_counts: Dict[int, int],
) -> str:
    if not true_counts:
        return ""
    true_classes = {int(k) for k, v in true_counts.items() if int(v) > 0}
    pred_classes = {int(k) for k, v in pred_counts.items() if int(v) > 0}
    if not pred_classes:
        return "모델이 어떤 방향도 예측하지 못했습니다."
    if len(true_classes) >= 3 and len(pred_classes) <= 1:
        only_cls = next(iter(pred_classes))
        return f"모델 예측이 방향 {only_cls}에 거의 고정되어 붕괴로 의심됩니다."
    total_pred = max(1, int(sum(int(v) for v in pred_counts.values())))
    major_cls, major_count = max(pred_counts.items(), key=lambda item: int(item[1]))
    if len(true_classes) >= 3 and float(major_count) / float(total_pred) >= 0.85:
        return f"모델 예측의 {float(major_count) / float(total_pred):.0%}가 방향 {int(major_cls)}에 집중되어 붕괴가 의심됩니다."
    missing = sorted(true_classes - pred_classes)
    if len(true_classes) >= 3 and missing:
        return "모델이 일부 방향을 전혀 예측하지 못했습니다: " + ", ".join(str(v) for v in missing)
    return ""


def _compute_eval_stats(
    model: nn.Module,
    X_struct: np.ndarray,
    X_patch: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, float]]:
    if X_struct.size == 0 or y.size == 0:
        return {}, {}, {}
    with torch.no_grad():
        logits = model(
            torch.from_numpy(X_struct).float().to(device),
            torch.from_numpy(X_patch).float().to(device),
        )
        pred = logits.argmax(dim=1).cpu().numpy()
    true_counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    pred_counts = {int(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))}
    per_class_acc: Dict[int, float] = {}
    for cls in np.unique(y):
        mask = y == cls
        per_class_acc[int(cls)] = float(np.mean(pred[mask] == y[mask])) if np.any(mask) else 0.0
    return true_counts, pred_counts, per_class_acc


def _macro_accuracy_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    scores: List[float] = []
    for cls in np.unique(y_true):
        mask = y_true == cls
        if np.any(mask):
            scores.append(float(np.mean(y_pred[mask] == y_true[mask])))
    return float(np.mean(scores)) if scores else 0.0


def _compute_knn_eval_stats(
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    num_classes: int,
    k: int,
    distance_scale: float,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, float], float]:
    if X_eval.size == 0 or y_eval.size == 0:
        return {}, {}, {}, 0.0
    preds: List[int] = []
    for row in X_eval:
        probs, _distance = _predict_knn_probabilities(
            row,
            X_ref,
            y_ref,
            num_classes=num_classes,
            k=k,
            distance_scale=distance_scale,
        )
        preds.append(int(np.argmax(probs)))
    pred = np.asarray(preds, dtype=np.int64)
    true_counts = _counts_dict(y_eval)
    pred_counts = _counts_dict(pred)
    per_class_acc: Dict[int, float] = {}
    for cls in np.unique(y_eval):
        mask = y_eval == cls
        per_class_acc[int(cls)] = float(np.mean(pred[mask] == y_eval[mask])) if np.any(mask) else 0.0
    acc = float(np.mean(pred == y_eval)) if y_eval.size else 0.0
    return true_counts, pred_counts, per_class_acc, acc


def _legacy_train_cell_model_v1(
    dataset_path: str = "cell_dataset.npz",
    model_path: str = "cell_model.pt",
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    seed: int = 42,
    device_preference: str = "auto",
) -> Dict[str, object]:
    """Legacy training path kept only for old notes. Do not call."""
    if not os.path.exists(dataset_path):
        raise RuntimeError(f"dataset file not found: {dataset_path}")

    with np.load(dataset_path) as data:
        X = np.asarray(data["X"], dtype=np.float32)
        X_patch = _normalize_patch_array(data["X_patch"] if "X_patch" in data else None, len(data["y"]))
        y = np.asarray(data["y"], dtype=np.int64)
        group_ids = np.asarray(data.get("group_ids", np.arange(len(y), dtype=np.int64)), dtype=np.int64).reshape(-1)
        schema_version = int(np.asarray(data.get("schema_version", np.array([0], dtype=np.int32))).reshape(-1)[0])
    if schema_version != DATASET_SCHEMA_VERSION:
        raise RuntimeError("저장된 학습 데이터 버전이 현재 코드와 맞지 않습니다. 설정에서 학습 초기화 후 수정 셀을 다시 저장해주세요.")
    expected_dim = expected_feature_dim(CONTEXT_RADIUS)
    if X.ndim != 2 or int(X.shape[1]) != expected_dim:
        raise RuntimeError(
            f"저장된 학습 데이터 특징 수가 현재 코드와 맞지 않습니다. 현재 데이터: {X.shape[1]}개, 필요 특징 수: {expected_dim}개. 설정에서 학습 초기화 후 수정 셀을 다시 저장해주세요."
        )
    n_samples = int(X.shape[0])
    if n_samples < 12:
        raise RuntimeError("at least 12 corrected cells are required before training")
    if group_ids.shape[0] != n_samples:
        group_ids = np.arange(n_samples, dtype=np.int64)

    rng = np.random.default_rng(seed)
    train_idx, val_idx, split_strategy = _choose_train_val_split(y, group_ids, val_ratio, rng)

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    X_train, X_val, mean, std = _standardize(X_train, X_val)

    train_class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
    sample_weights = train_class_counts.sum() / np.maximum(train_class_counts[y_train], 1.0)
    train_sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights.astype(np.float64)),
        num_samples=int(len(sample_weights)),
        replacement=True,
    )
    train_loader = DataLoader(
        CellDataset(X_train, np.zeros((len(y_train), *expected_patch_shape()), dtype=np.float32), y_train),
        batch_size=batch_size,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        CellDataset(X_val, np.zeros((len(y_val), *expected_patch_shape()), dtype=np.float32), y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    device, device_label = resolve_compute_device(device_preference)
    model = CellOrientationNet(input_dim=X.shape[1], num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history: Dict[str, object] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_state: Dict[str, torch.Tensor] | None = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    patience = 12
    epochs_without_improvement = 0

    for _epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, xb_patch, yb in train_loader:
            xb = xb.to(device).float()
            xb_patch = xb_patch.to(device).float()
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb, xb_patch)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1)
            train_correct += int((preds == yb).sum().item())
            train_total += int(yb.size(0))

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, xb_patch, yb in val_loader:
                xb = xb.to(device).float()
                xb_patch = xb_patch.to(device).float()
                yb = yb.to(device)
                logits = model(xb, xb_patch)
                loss = loss_fn(logits, yb)
                val_loss += float(loss.item()) * xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += int((preds == yb).sum().item())
                val_total += int(yb.size(0))

        history["train_loss"].append(train_loss / max(1, train_total))
        history["val_loss"].append(val_loss / max(1, val_total))
        history["train_acc"].append(train_correct / max(1, train_total))
        history["val_acc"].append(val_correct / max(1, val_total))
        current_val_acc = val_correct / max(1, val_total)
        current_val_loss = val_loss / max(1, val_total)
        improved = False
        if current_val_acc > best_val_acc + 1e-6:
            improved = True
        elif abs(current_val_acc - best_val_acc) <= 1e-6 and current_val_loss < best_val_loss - 1e-6:
            improved = True
        if improved:
            best_val_acc = float(current_val_acc)
            best_val_loss = float(current_val_loss)
            best_epoch = int(_epoch)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": int(X.shape[1]),
            "feature_mean": mean,
            "feature_std": std,
            "num_classes": int(NUM_CLASSES),
            "schema_version": int(DATASET_SCHEMA_VERSION),
            "device_hint": str(device),
            "device_label": str(device_label),
        },
        model_path,
    )
    history["device"] = str(device)
    history["device_label"] = str(device_label)
    history["num_samples"] = int(n_samples)
    history["num_classes"] = int(NUM_CLASSES)
    history["num_train_groups"] = int(len(np.unique(group_ids[train_idx])))
    history["num_val_groups"] = int(len(np.unique(group_ids[val_idx])))
    history["split_strategy"] = str(split_strategy)
    history["best_epoch"] = int(best_epoch + 1)
    train_true_counts, train_pred_counts, train_per_class_acc = _compute_eval_stats(
        model,
        X_train,
        np.zeros((len(y_train), *expected_patch_shape()), dtype=np.float32),
        y_train,
        device,
    )
    val_true_counts, val_pred_counts, val_per_class_acc = _compute_eval_stats(
        model,
        X_val,
        np.zeros((len(y_val), *expected_patch_shape()), dtype=np.float32),
        y_val,
        device,
    )
    history["train_class_counts"] = train_true_counts
    history["val_class_counts"] = val_true_counts
    history["train_pred_counts"] = train_pred_counts
    history["val_pred_counts"] = val_pred_counts
    history["train_per_class_acc"] = train_per_class_acc
    history["val_per_class_acc"] = val_per_class_acc
    collapse_warning = _detect_prediction_collapse(val_true_counts, val_pred_counts) or _detect_prediction_collapse(
        train_true_counts,
        train_pred_counts,
    )
    history["collapse_warning"] = collapse_warning
    return history


def _legacy_predict_with_model_with_meta_v1(
    img: np.ndarray,
    A: int,
    model_path: str = "cell_model.pt",
    ink_thr: float = DEFAULT_INK_THRESHOLD,
    use_threshold: float = 0.60,
    device_preference: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Legacy predictor kept only for backward reference. Do not call."""
    morph_dir, occ, mean_bw, mean_edges, mean_skeleton, head_hint, morph_confidence = _compute_morphological_features(
        img,
        A,
        ink_thr=ink_thr,
    )
    global_maps = _compute_global_context_maps(morph_dir, occ, head_hint)
    H, W = img.shape[:2]
    B = derive_grid_B(A, W, H)
    meta: Dict[str, np.ndarray] = {
        "occ": occ.astype(np.uint8),
        "morph_confidence": morph_confidence.astype(np.float32),
        "model_confidence": np.zeros((A, B), dtype=np.float32),
        "confidence": morph_confidence.astype(np.float32).copy(),
    }
    expected_dim = expected_feature_dim(CONTEXT_RADIUS)

    def _fallback(reason: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        meta["device_label"] = np.array(["형태 기반"], dtype=object)
        meta["model_notice"] = np.array([reason], dtype=object)
        return morph_dir.copy(), morph_dir.copy(), meta

    if not os.path.exists(model_path):
        return _fallback("저장된 학습 모델이 없어 형태 기반 결과만 사용했습니다.")
    try:
        model_header = Path(model_path).read_bytes()[:32]
        if model_header.startswith(b"DISABLED"):
            return _fallback("학습 모델이 초기화되어 형태 기반 결과만 사용했습니다. 수정 셀을 다시 저장한 뒤 학습해주세요.")
    except OSError:
        return _fallback("저장된 학습 모델을 읽을 수 없어 형태 기반 결과만 사용했습니다.")

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            return _fallback("저장된 학습 모델 형식이 올바르지 않아 형태 기반 결과만 사용했습니다.")
        input_dim = int(checkpoint.get("input_dim", 0))
        num_classes = int(checkpoint.get("num_classes", 0))
        schema_version = int(checkpoint.get("schema_version", 0))
        feature_mean = np.asarray(checkpoint.get("feature_mean", np.zeros((1, input_dim), dtype=np.float32)), dtype=np.float32)
        feature_std = np.asarray(checkpoint.get("feature_std", np.ones((1, input_dim), dtype=np.float32)), dtype=np.float32)
        feature_mean = feature_mean.reshape(1, -1) if feature_mean.size else np.zeros((1, input_dim), dtype=np.float32)
        feature_std = feature_std.reshape(1, -1) if feature_std.size else np.ones((1, input_dim), dtype=np.float32)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
        if schema_version != DATASET_SCHEMA_VERSION:
            return _fallback("저장된 학습 모델 버전이 현재 코드와 맞지 않아 형태 기반 결과만 사용했습니다. 설정에서 학습 초기화 후 다시 학습해주세요.")
        if num_classes != NUM_CLASSES:
            return _fallback("저장된 학습 모델 클래스 수가 현재 코드와 맞지 않아 형태 기반 결과만 사용했습니다.")
        if input_dim != expected_dim:
            return _fallback(
                f"저장된 학습 모델 특징 수({input_dim})가 현재 코드 특징 수({expected_dim})와 달라 형태 기반 결과만 사용했습니다. 설정에서 학습 초기화 후 다시 학습해주세요."
            )
        if feature_mean.shape[1] != input_dim or feature_std.shape[1] != input_dim:
            return _fallback("저장된 학습 모델 정규화 정보가 현재 모델과 맞지 않아 형태 기반 결과만 사용했습니다.")

        device, device_label = resolve_compute_device(device_preference)
        model = CellOrientationNet(input_dim=input_dim, num_classes=num_classes).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
    except Exception:
        return _fallback("저장된 학습 모델을 읽는 중 오류가 발생해 형태 기반 결과만 사용했습니다.")

    final_dir = morph_dir.copy().astype(np.uint8)
    model_confidence = np.zeros((A, B), dtype=np.float32)
    model_applied_mask = np.zeros((A, B), dtype=np.uint8)
    ambiguity_threshold = 0.72
    override_margin = 0.08
    for i in range(A):
        for j in range(B):
            patch = _extract_cell_patch_tensor(gray_inv, bw_patch_map, edge_patch_map, i, j, A, B)
            raw_support = _patch_signal_strength(patch)
            meta["model_raw_support"][i, j] = np.float32(raw_support)
            if int(occ[i, j]) != 1 and int(morph_dir[i, j]) == 0 and raw_support < PATCH_SIGNAL_THRESHOLD:
                continue
            base_direction = int(morph_dir[i, j])
            base_conf = float(morph_confidence[i, j])
            feat = np.asarray(
                _cell_feature_vector(
                    morph_dir,
                    occ,
                    mean_bw,
                    mean_edges,
                    mean_skeleton,
                    head_hint,
                    morph_confidence,
                    global_maps,
                    i,
                    j,
                    context_radius=CONTEXT_RADIUS,
                ),
                dtype=np.float32,
            )[None, :]
            feat = (feat - feature_mean) / feature_std
            x = torch.from_numpy(feat).float().to(device)
            with torch.no_grad():
                logits = model(x)
                prob = F.softmax(logits, dim=1)[0]
                top_prob, top_idx = prob.max(dim=0)
            model_confidence[i, j] = np.float32(float(top_prob))
            predicted_direction = int(top_idx.item())
            if predicted_direction == base_direction:
                continue
            is_ambiguous = base_direction == 0 or base_conf < ambiguity_threshold
            if (
                predicted_direction != 0
                and is_ambiguous
                and float(top_prob) >= float(use_threshold)
                and float(top_prob) >= base_conf + override_margin
            ):
                final_dir[i, j] = np.uint8(predicted_direction)
                model_applied_mask[i, j] = 1

    meta["model_confidence"] = model_confidence.astype(np.float32)
    meta["confidence"] = np.maximum(morph_confidence.astype(np.float32), model_confidence.astype(np.float32))
    meta["model_applied_mask"] = model_applied_mask.astype(np.uint8)
    meta["device_label"] = np.array([str(device_label)], dtype=object)
    return final_dir, morph_dir, meta


def predict_with_model(
    img: np.ndarray,
    A: int,
    model_path: str = "cell_model.pt",
    ink_thr: float = DEFAULT_INK_THRESHOLD,
    use_threshold: float = 0.60,
    device_preference: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    final_dir, morph_dir, _meta = predict_with_model_with_meta(
        img,
        A,
        model_path=model_path,
        ink_thr=ink_thr,
        use_threshold=use_threshold,
        device_preference=device_preference,
    )
    return final_dir, morph_dir


def _v2_pairwise_nearest_distances(X: np.ndarray) -> np.ndarray:
    n_samples = int(X.shape[0])
    if n_samples <= 1:
        return np.full((n_samples,), np.inf, dtype=np.float32)
    diff = X[:, None, :] - X[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float64)).astype(np.float32)
    np.fill_diagonal(dist, np.inf)
    dist[dist <= 1e-8] = np.inf
    return np.min(dist, axis=1).astype(np.float32)


def _v2_build_reference_artifacts(X_std: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    if X_std.size == 0 or y.size == 0:
        return {
            "reference_features": np.zeros((0, 0), dtype=np.float32),
            "reference_labels": np.zeros((0,), dtype=np.int64),
            "distance_scale": 1.0,
            "distance_limit": 1.0,
            "knn_k": 1,
            "class_counts": np.zeros((NUM_CLASSES,), dtype=np.int32),
        }
    nearest = _v2_pairwise_nearest_distances(X_std)
    finite = nearest[np.isfinite(nearest)]
    if finite.size == 0:
        distance_scale = 1.0
        distance_limit = 1.0
    else:
        distance_scale = max(float(np.median(finite)), 1e-3)
        distance_limit = max(float(np.quantile(finite, 0.85)) * 1.35, distance_scale * 1.25, 1e-3)
    knn_k = int(np.clip(round(np.sqrt(max(1, X_std.shape[0]))), 3, 9))
    return {
        "reference_features": X_std.astype(np.float32),
        "reference_labels": y.astype(np.int64),
        "distance_scale": float(distance_scale),
        "distance_limit": float(distance_limit),
        "knn_k": int(knn_k),
        "class_counts": np.bincount(y, minlength=NUM_CLASSES).astype(np.int32),
    }


def _v2_predict_knn_probabilities(
    query: np.ndarray,
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    num_classes: int,
    k: int,
    distance_scale: float,
) -> Tuple[np.ndarray, float]:
    if reference_features.size == 0 or reference_labels.size == 0:
        return np.zeros((num_classes,), dtype=np.float32), float("inf")
    diff = reference_features - query
    distances = np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float64)).astype(np.float32)
    if distances.size == 0:
        return np.zeros((num_classes,), dtype=np.float32), float("inf")
    sigma = max(float(distance_scale), 1e-3)
    order = np.argsort(distances)[: max(1, min(int(k), int(distances.size)))]
    probs = np.zeros((num_classes,), dtype=np.float32)
    for idx in order:
        weight = float(np.exp(-((float(distances[idx]) ** 2) / (2.0 * sigma * sigma))))
        probs[int(reference_labels[idx])] += weight
    if float(np.sum(probs)) <= 1e-8:
        probs[int(reference_labels[order[0]])] = 1.0
    else:
        probs = probs / float(np.sum(probs))
    return probs.astype(np.float32), float(distances[order[0]])


def _v2_class_support_score(class_counts: np.ndarray, direction: int) -> float:
    if direction < 0 or direction >= int(class_counts.shape[0]):
        return 0.0
    support = int(class_counts[direction])
    if support <= 0:
        return 0.0
    support_floor = 0.20 if direction == 0 else 0.25
    support_denom = max(2, MIN_CLASS_SUPPORT_FOR_OVERRIDE // 2) if direction == 0 else max(1, MIN_CLASS_SUPPORT_FOR_OVERRIDE)
    return float(np.clip(support / max(1, support_denom), support_floor, 1.0))


def _v2_query_closeness(nearest_distance: float, distance_scale: float, distance_limit: float) -> Tuple[float, bool]:
    if not np.isfinite(nearest_distance):
        return 0.0, False
    sigma = max(float(distance_scale), 1e-3)
    closeness = float(np.exp(-((float(nearest_distance) ** 2) / (2.0 * sigma * sigma))))
    distance_ok = bool(float(nearest_distance) <= max(float(distance_limit), sigma * 1.05))
    return closeness, distance_ok


def train_cell_model(
    dataset_path: str = "cell_dataset.npz",
    model_path: str = "cell_model.pt",
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    seed: int = 42,
    device_preference: str = "auto",
) -> Dict[str, object]:
    """Train the correction model.

    Small or weakly covered datasets are stored as exemplar-based checkpoints, while larger and
    diverse datasets use the neural network.
    """
    if not os.path.exists(dataset_path):
        raise RuntimeError(f"dataset file not found: {dataset_path}")

    with np.load(dataset_path) as data:
        X = np.asarray(data["X"], dtype=np.float32)
        X_patch = _normalize_patch_array(data["X_patch"] if "X_patch" in data else None, len(data["y"]))
        y = np.asarray(data["y"], dtype=np.int64)
        group_ids = np.asarray(data.get("group_ids", np.arange(len(y), dtype=np.int64)), dtype=np.int64).reshape(-1)
        schema_version = int(np.asarray(data.get("schema_version", np.array([0], dtype=np.int32))).reshape(-1)[0])
    if schema_version != DATASET_SCHEMA_VERSION:
        raise RuntimeError("저장된 학습 데이터 버전이 현재 코드와 맞지 않습니다. 설정에서 학습 초기화 후 수정 셀을 다시 저장해주세요.")
    expected_dim = expected_feature_dim(CONTEXT_RADIUS)
    if X.ndim != 2 or int(X.shape[1]) != expected_dim:
        raise RuntimeError(
            f"저장된 학습 데이터 특징 수가 현재 코드와 맞지 않습니다. 현재 데이터: {X.shape[1]}개, 필요 특징 수: {expected_dim}개. 설정에서 학습 초기화 후 수정 셀을 다시 저장해주세요."
        )

    n_samples = int(X.shape[0])
    if n_samples < 12:
        raise RuntimeError("at least 12 corrected cells are required before training")
    if group_ids.shape[0] != n_samples:
        group_ids = np.arange(n_samples, dtype=np.int64)
    X, X_patch, y, group_ids, dedupe_stats = _deduplicate_training_arrays(X, X_patch, y, group_ids)
    if int(dedupe_stats.get("removed", 0)) > 0:
        _atomic_savez(
            dataset_path,
            X=X,
            X_patch=X_patch,
            y=y,
            group_ids=group_ids,
            schema_version=np.array([DATASET_SCHEMA_VERSION], dtype=np.int32),
            num_classes=np.array([NUM_CLASSES], dtype=np.int32),
            patch_channels=np.array([PATCH_CHANNELS], dtype=np.int32),
            patch_size=np.array([RAW_PATCH_SIZE], dtype=np.int32),
        )
    n_samples = int(X.shape[0])
    if n_samples < 12:
        raise RuntimeError("at least 12 unique corrected cells are required before training")
    patch_nonzero_ratio = float(np.mean(np.sum(X_patch, axis=(1, 2, 3)) > 1e-6)) if n_samples else 0.0
    patch_warning = ""
    if patch_nonzero_ratio < 0.25:
        patch_warning = "기존 학습 데이터에 원본 이미지 패치 정보가 거의 없어 새 하이브리드 모델 효과가 제한될 수 있습니다. 수정 셀을 다시 저장해 원본 패치 샘플을 누적해 주세요."

    coverage_mode, coverage_reason = _coverage_mode(y)
    independent_group_count = int(np.unique(group_ids).size)
    if coverage_mode == "normal" and independent_group_count < MIN_NEURAL_TRAINING_GROUPS:
        coverage_mode = "restricted"
        coverage_reason = (
            f"Only {independent_group_count} independent correction groups are available; "
            "human-exemplar correction is used until more independent groups are collected."
        )
    rng = np.random.default_rng(seed)
    train_idx, val_idx, split_strategy = _choose_train_val_split(y, group_ids, val_ratio, rng)

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_patch_train = X_patch[train_idx]
    X_patch_val = X_patch[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    history: Dict[str, object] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    device = torch.device("cpu")
    device_label = "cpu"
    best_val_acc = 0.0
    best_epoch = 1
    model_kind = "knn" if coverage_mode == "restricted" else "mlp"

    if model_kind == "knn":
        X_train_std, X_val_std, _mean_train, _std_train = _standardize(X_train, X_val)
        X_train_ref = _combine_struct_and_patch_features(X_train_std, X_patch_train)
        X_val_ref = _combine_struct_and_patch_features(X_val_std, X_patch_val)
        train_artifacts = _v2_build_reference_artifacts(X_train_ref, y_train)
        train_true_counts, train_pred_counts, train_per_class_acc, train_acc = _compute_knn_eval_stats(
            X_train_ref,
            y_train,
            X_train_ref,
            y_train,
            num_classes=NUM_CLASSES,
            k=int(train_artifacts["knn_k"]),
            distance_scale=float(train_artifacts["distance_scale"]),
        )
        val_true_counts, val_pred_counts, val_per_class_acc, val_acc = _compute_knn_eval_stats(
            X_train_ref,
            y_train,
            X_val_ref,
            y_val,
            num_classes=NUM_CLASSES,
            k=int(train_artifacts["knn_k"]),
            distance_scale=float(train_artifacts["distance_scale"]),
        )
        history["train_loss"].append(0.0)
        history["val_loss"].append(0.0)
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        best_val_acc = float(val_acc)
        best_val_macro = float(np.mean(list(val_per_class_acc.values()))) if val_per_class_acc else 0.0

        mean_all = X.mean(axis=0, keepdims=True).astype(np.float32)
        std_all = X.std(axis=0, keepdims=True).astype(np.float32)
        std_all = np.where(std_all < 1e-6, 1.0, std_all).astype(np.float32)
        X_all_std = _apply_standardization(X, mean_all, std_all)
        X_all_ref = _combine_struct_and_patch_features(X_all_std, X_patch)
        final_artifacts = _v2_build_reference_artifacts(X_all_ref, y)
        collapse_warning = _detect_prediction_collapse(val_true_counts, val_pred_counts) or _detect_prediction_collapse(
            train_true_counts,
            train_pred_counts,
        )
        if coverage_mode == "restricted" and coverage_reason:
            collapse_warning = coverage_reason
        checkpoint: Dict[str, object] = {
            "model_kind": "knn",
            "model_schema_version": int(MODEL_SCHEMA_VERSION),
            "input_dim": int(X.shape[1]),
            "feature_mean": mean_all,
            "feature_std": std_all,
            "num_classes": int(NUM_CLASSES),
            "schema_version": int(DATASET_SCHEMA_VERSION),
            "device_hint": "cpu",
            "device_label": "cpu (예시 기반)",
            "coverage_mode": str(coverage_mode),
            "coverage_reason": str(coverage_reason),
            "val_best_acc": float(best_val_acc),
            "val_best_macro_acc": float(best_val_macro),
            "seen_labels": np.asarray(sorted(int(v) for v in np.unique(y)), dtype=np.int64),
            "patch_channels": int(PATCH_CHANNELS),
            "patch_size": int(RAW_PATCH_SIZE),
            "best_epoch": int(best_epoch),
            "split_strategy": str(split_strategy),
            "collapse_warning": str(collapse_warning),
            "num_samples": int(n_samples),
            "deduplicated_samples": int(n_samples),
            "deduplicated_removed": int(dedupe_stats.get("removed", 0)),
            "deduplicated_conflicts": int(dedupe_stats.get("conflicts", 0)),
            "num_train_groups": int(len(np.unique(group_ids[train_idx]))),
            "num_val_groups": int(len(np.unique(group_ids[val_idx]))),
            "patch_nonzero_ratio": float(patch_nonzero_ratio),
            "patch_warning": str(patch_warning),
            "train_class_counts": dict(train_true_counts),
            "val_class_counts": dict(val_true_counts),
            "train_pred_counts": dict(train_pred_counts),
            "val_pred_counts": dict(val_pred_counts),
            "train_per_class_acc": dict(train_per_class_acc),
            "val_per_class_acc": dict(val_per_class_acc),
        }
        checkpoint.update(final_artifacts)
        torch.save(checkpoint, model_path)
    else:
        X_train_std, X_val_std, mean, std = _standardize(X_train, X_val)
        train_class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
        sample_weights = train_class_counts.sum() / np.maximum(train_class_counts[y_train], 1.0)
        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights.astype(np.float64)),
            num_samples=int(len(sample_weights)),
            replacement=True,
        )
        train_loader = DataLoader(CellDataset(X_train_std, X_patch_train, y_train), batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(CellDataset(X_val_std, X_patch_val, y_val), batch_size=batch_size, shuffle=False)

        device, device_label = resolve_compute_device(device_preference)
        model = CellOrientationNet(
            input_dim=X.shape[1],
            patch_channels=PATCH_CHANNELS,
            patch_size=RAW_PATCH_SIZE,
            num_classes=NUM_CLASSES,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_weights = train_class_counts.sum() / np.maximum(train_class_counts, 1.0)
        positive_mask = train_class_counts > 0
        if np.any(positive_mask):
            loss_weights = loss_weights / max(1e-6, float(np.mean(loss_weights[positive_mask])))
        loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(loss_weights.astype(np.float32)).to(device))

        best_state: Dict[str, torch.Tensor] | None = None
        best_val_loss = float("inf")
        best_val_macro = -1.0
        patience = 12
        epochs_without_improvement = 0

        for _epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for xb_struct, xb_patch, yb in train_loader:
                xb_struct = xb_struct.to(device).float()
                xb_patch = xb_patch.to(device).float()
                yb = yb.to(device)

                optimizer.zero_grad()
                logits = model(xb_struct, xb_patch)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

                train_loss += float(loss.item()) * xb_struct.size(0)
                preds = logits.argmax(dim=1)
                train_correct += int((preds == yb).sum().item())
                train_total += int(yb.size(0))

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_targets_epoch: List[np.ndarray] = []
            val_preds_epoch: List[np.ndarray] = []
            with torch.no_grad():
                for xb_struct, xb_patch, yb in val_loader:
                    xb_struct = xb_struct.to(device).float()
                    xb_patch = xb_patch.to(device).float()
                    yb = yb.to(device)
                    logits = model(xb_struct, xb_patch)
                    loss = loss_fn(logits, yb)
                    val_loss += float(loss.item()) * xb_struct.size(0)
                    preds = logits.argmax(dim=1)
                    val_correct += int((preds == yb).sum().item())
                    val_total += int(yb.size(0))
                    val_targets_epoch.append(yb.detach().cpu().numpy())
                    val_preds_epoch.append(preds.detach().cpu().numpy())

            history["train_loss"].append(train_loss / max(1, train_total))
            history["val_loss"].append(val_loss / max(1, val_total))
            history["train_acc"].append(train_correct / max(1, train_total))
            history["val_acc"].append(val_correct / max(1, val_total))

            current_val_acc = val_correct / max(1, val_total)
            current_val_loss = val_loss / max(1, val_total)
            if val_targets_epoch and val_preds_epoch:
                current_val_macro = _macro_accuracy_from_arrays(
                    np.concatenate(val_targets_epoch).astype(np.int64),
                    np.concatenate(val_preds_epoch).astype(np.int64),
                )
            else:
                current_val_macro = 0.0
            improved = False
            if current_val_macro > best_val_macro + 1e-6:
                improved = True
            elif abs(current_val_macro - best_val_macro) <= 1e-6 and current_val_acc > best_val_acc + 1e-6:
                improved = True
            elif (
                abs(current_val_macro - best_val_macro) <= 1e-6
                and abs(current_val_acc - best_val_acc) <= 1e-6
                and current_val_loss < best_val_loss - 1e-6
            ):
                improved = True
            if improved:
                best_val_macro = float(current_val_macro)
                best_val_acc = float(current_val_acc)
                best_val_loss = float(current_val_loss)
                best_epoch = int(_epoch) + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        train_true_counts, train_pred_counts, train_per_class_acc = _compute_eval_stats(model, X_train_std, X_patch_train, y_train, device)
        val_true_counts, val_pred_counts, val_per_class_acc = _compute_eval_stats(model, X_val_std, X_patch_val, y_val, device)
        X_all_std = _apply_standardization(X, mean, std)
        X_all_ref = _combine_struct_and_patch_features(X_all_std, X_patch)
        final_artifacts = _v2_build_reference_artifacts(X_all_ref, y)
        collapse_warning = _detect_prediction_collapse(val_true_counts, val_pred_counts) or _detect_prediction_collapse(
            train_true_counts,
            train_pred_counts,
        )
        if coverage_mode == "restricted" and coverage_reason:
            collapse_warning = coverage_reason
        checkpoint = {
            "model_kind": "mlp",
            "model_schema_version": int(MODEL_SCHEMA_VERSION),
            "model": model.state_dict(),
            "input_dim": int(X.shape[1]),
            "feature_mean": mean,
            "feature_std": std,
            "num_classes": int(NUM_CLASSES),
            "schema_version": int(DATASET_SCHEMA_VERSION),
            "device_hint": str(device),
            "device_label": str(device_label),
            "coverage_mode": str(coverage_mode),
            "coverage_reason": str(coverage_reason),
            "val_best_acc": float(best_val_acc),
            "val_best_macro_acc": float(best_val_macro),
            "seen_labels": np.asarray(sorted(int(v) for v in np.unique(y)), dtype=np.int64),
            "patch_channels": int(PATCH_CHANNELS),
            "patch_size": int(RAW_PATCH_SIZE),
            "best_epoch": int(best_epoch),
            "split_strategy": str(split_strategy),
            "collapse_warning": str(collapse_warning),
            "num_samples": int(n_samples),
            "deduplicated_samples": int(n_samples),
            "deduplicated_removed": int(dedupe_stats.get("removed", 0)),
            "deduplicated_conflicts": int(dedupe_stats.get("conflicts", 0)),
            "num_train_groups": int(len(np.unique(group_ids[train_idx]))),
            "num_val_groups": int(len(np.unique(group_ids[val_idx]))),
            "patch_nonzero_ratio": float(patch_nonzero_ratio),
            "patch_warning": str(patch_warning),
            "train_class_counts": dict(train_true_counts),
            "val_class_counts": dict(val_true_counts),
            "train_pred_counts": dict(train_pred_counts),
            "val_pred_counts": dict(val_pred_counts),
            "train_per_class_acc": dict(train_per_class_acc),
            "val_per_class_acc": dict(val_per_class_acc),
        }
        checkpoint.update(final_artifacts)
        torch.save(checkpoint, model_path)

    history["device"] = str(device)
    history["device_label"] = str(device_label)
    history["num_samples"] = int(n_samples)
    history["deduplicated_samples"] = int(n_samples)
    history["deduplicated_removed"] = int(dedupe_stats.get("removed", 0))
    history["deduplicated_conflicts"] = int(dedupe_stats.get("conflicts", 0))
    history["num_classes"] = int(NUM_CLASSES)
    history["num_train_groups"] = int(len(np.unique(group_ids[train_idx])))
    history["num_val_groups"] = int(len(np.unique(group_ids[val_idx])))
    history["split_strategy"] = str(split_strategy)
    history["best_epoch"] = int(best_epoch)
    history["val_best_acc"] = float(best_val_acc)
    history["val_best_macro_acc"] = float(best_val_macro)
    history["model_kind"] = str(model_kind)
    history["coverage_mode"] = str(coverage_mode)
    history["coverage_reason"] = str(coverage_reason)
    history["train_class_counts"] = train_true_counts
    history["val_class_counts"] = val_true_counts
    history["train_pred_counts"] = train_pred_counts
    history["val_pred_counts"] = val_pred_counts
    history["train_per_class_acc"] = train_per_class_acc
    history["val_per_class_acc"] = val_per_class_acc
    history["patch_nonzero_ratio"] = float(patch_nonzero_ratio)
    history["patch_warning"] = str(patch_warning)
    collapse_warning = _detect_prediction_collapse(val_true_counts, val_pred_counts) or _detect_prediction_collapse(
        train_true_counts,
        train_pred_counts,
    )
    if coverage_mode == "restricted" and coverage_reason:
        collapse_warning = coverage_reason
    history["collapse_warning"] = collapse_warning
    return history


def predict_with_model_with_meta(
    img: np.ndarray,
    A: int,
    model_path: str = "cell_model.pt",
    ink_thr: float = DEFAULT_INK_THRESHOLD,
    use_threshold: float = 0.60,
    device_preference: str = "auto",
    strict_deployment_ready: bool = False,
    detection_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Predict a corrected matrix using the trained model when available."""
    H, W = img.shape[:2]
    B = derive_grid_B(A, W, H)
    working_cache = detection_cache
    if not (
        isinstance(working_cache, dict)
        and int(working_cache.get("A", -1)) == int(A)
        and isinstance(working_cache.get("meta"), dict)
    ):
        working_cache = None
    morph_dir, occ, mean_bw, mean_edges, mean_skeleton, head_hint, morph_confidence = _compute_morphological_features(
        img,
        A,
        ink_thr=ink_thr,
        detection_cache=working_cache,
    )
    cache_meta = working_cache.get("meta", {}) if isinstance(working_cache, dict) else {}
    has_detection_aux = isinstance(cache_meta, dict) and (
        "overlap_count" in cache_meta or "support_level" in cache_meta
    )
    overlap_count = np.asarray(cache_meta.get("overlap_count", np.zeros((A, B), dtype=np.int32)), dtype=np.int32)
    support_level = np.asarray(
        cache_meta.get("support_level", np.where(occ > 0, 2, 0).astype(np.uint8)),
        dtype=np.uint8,
    )
    path_occ = np.asarray(cache_meta.get("path_occ", occ.astype(np.uint8)), dtype=np.uint8)
    support_mask = np.asarray(cache_meta.get("support_mask", np.maximum(occ, path_occ).astype(np.uint8)), dtype=np.uint8)
    gray_inv, bw_patch_map, edge_patch_map = _compute_patch_source_maps(img)
    global_maps = _compute_global_context_maps(morph_dir, occ, head_hint)
    blended_confidence = morph_confidence.astype(np.float32).copy()
    meta: Dict[str, np.ndarray] = {
        "occ": occ.astype(np.uint8),
        "path_occ": path_occ.astype(np.uint8),
        "support_mask": support_mask.astype(np.uint8),
        "morph_confidence": morph_confidence.astype(np.float32),
        "model_confidence": np.zeros((A, B), dtype=np.float32),
        "model_probability": np.zeros((A, B), dtype=np.float32),
        "model_distance": np.full((A, B), np.inf, dtype=np.float32),
        "model_applied_mask": np.zeros((A, B), dtype=np.uint8),
        "model_direct_mask": np.zeros((A, B), dtype=np.uint8),
        "model_candidate_mask": np.zeros((A, B), dtype=np.uint8),
        "model_raw_support": np.zeros((A, B), dtype=np.float32),
        "model_direction": np.zeros((A, B), dtype=np.uint8),
        "overlap_count": overlap_count.astype(np.int32),
        "support_level": support_level.astype(np.uint8),
        "confidence": blended_confidence.copy(),
    }
    expected_dim = expected_feature_dim(CONTEXT_RADIUS)

    def _fallback(reason: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        meta["device_label"] = np.array(["형태 기반"], dtype=object)
        meta["model_notice"] = np.array([reason], dtype=object)
        return morph_dir.copy(), morph_dir.copy(), meta

    if not os.path.exists(model_path):
        return _fallback("저장된 학습 모델이 없어 형태 기반 결과만 사용했습니다.")
    try:
        model_header = Path(model_path).read_bytes()[:32]
        if model_header.startswith(b"DISABLED"):
            return _fallback("학습 모델이 초기화되어 형태 기반 결과만 사용했습니다. 수정 셀을 다시 저장한 뒤 학습해주세요.")
    except OSError:
        return _fallback("저장된 학습 모델을 읽을 수 없어 형태 기반 결과만 사용했습니다.")

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            return _fallback("저장된 학습 모델 형식이 올바르지 않아 형태 기반 결과만 사용했습니다.")
        input_dim = int(checkpoint.get("input_dim", 0))
        num_classes = int(checkpoint.get("num_classes", 0))
        schema_version = int(checkpoint.get("schema_version", 0))
        model_kind = str(checkpoint.get("model_kind", "mlp")).strip().lower()
        feature_mean = np.asarray(checkpoint.get("feature_mean", np.zeros((1, input_dim), dtype=np.float32)), dtype=np.float32)
        feature_std = np.asarray(checkpoint.get("feature_std", np.ones((1, input_dim), dtype=np.float32)), dtype=np.float32)
        feature_mean = feature_mean.reshape(1, -1) if feature_mean.size else np.zeros((1, input_dim), dtype=np.float32)
        feature_std = feature_std.reshape(1, -1) if feature_std.size else np.ones((1, input_dim), dtype=np.float32)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
        if schema_version != DATASET_SCHEMA_VERSION:
            return _fallback("저장된 학습 모델 버전이 현재 코드와 맞지 않아 형태 기반 결과만 사용했습니다. 설정에서 다시 학습해주세요.")
        if num_classes != NUM_CLASSES:
            return _fallback("저장된 학습 모델 클래스 수가 현재 코드와 맞지 않아 형태 기반 결과만 사용했습니다.")
        if input_dim != expected_dim:
            return _fallback(
                f"저장된 학습 모델 특징 수({input_dim})가 현재 코드 특징 수({expected_dim})와 달라 형태 기반 결과만 사용했습니다. 설정에서 다시 학습해주세요."
            )
        if feature_mean.shape[1] != input_dim or feature_std.shape[1] != input_dim:
            return _fallback("저장된 학습 모델 정규화 정보가 현재 모델과 맞지 않아 형태 기반 결과만 사용했습니다.")
        if int(checkpoint.get("patch_channels", PATCH_CHANNELS)) != PATCH_CHANNELS or int(checkpoint.get("patch_size", RAW_PATCH_SIZE)) != RAW_PATCH_SIZE:
            return _fallback("저장된 학습 모델 패치 설정이 현재 코드와 맞지 않아 형태 기반 결과만 사용했습니다. 설정에서 다시 학습해주세요.")

        ref_features = np.asarray(checkpoint.get("reference_features", np.zeros((0, input_dim), dtype=np.float32)), dtype=np.float32)
        ref_labels = np.asarray(checkpoint.get("reference_labels", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
        class_counts = np.asarray(checkpoint.get("class_counts", np.zeros((NUM_CLASSES,), dtype=np.int32)), dtype=np.int32).reshape(-1)
        expected_ref_dim = int(input_dim + np.prod(expected_patch_shape()))
        if ref_features.size and (ref_features.ndim != 2 or int(ref_features.shape[1]) != expected_ref_dim):
            return _fallback("저장된 학습 모델 기준 특징 차원이 현재 코드와 맞지 않아 형태 기반 결과만 사용했습니다. 설정에서 다시 학습해주세요.")
        distance_scale = float(checkpoint.get("distance_scale", 1.0))
        distance_limit = float(checkpoint.get("distance_limit", max(distance_scale, 1.0)))
        knn_k = int(checkpoint.get("knn_k", 5))
        patch_channels = int(checkpoint.get("patch_channels", PATCH_CHANNELS))
        patch_size = int(checkpoint.get("patch_size", RAW_PATCH_SIZE))
        coverage_mode = str(checkpoint.get("coverage_mode", "restricted")).strip().lower()
        val_macro = checkpoint.get("val_best_macro_acc")
        exemplar_fallback_notice = ""
        if model_kind == "mlp" and val_macro is not None and float(val_macro) < 0.70:
            if strict_deployment_ready:
                return _fallback("model validation macro accuracy is below 0.70.")
            if ref_features.size == 0 or ref_labels.size == 0:
                return _fallback("model validation macro accuracy is below 0.70 and no human exemplars are available.")
            model_kind = "knn"
            coverage_mode = "restricted"
            exemplar_fallback_notice = (
                f"Neural validation macro accuracy {float(val_macro):.3f} is below 0.700; "
                "human-exemplar correction was used."
            )
        coverage_reason = str(checkpoint.get("coverage_reason", "")).strip()
        patch_warning = str(checkpoint.get("patch_warning", "")).strip()
        collapse_warning = str(checkpoint.get("collapse_warning", "")).strip()
        device_label = "cpu (human exemplar fallback)" if exemplar_fallback_notice else str(checkpoint.get("device_label", "cpu"))
        if strict_deployment_ready:
            ready, readiness_reason = checkpoint_deployment_status(checkpoint)
            if not ready:
                return _fallback(readiness_reason)

        model = None
        device = torch.device("cpu")
        if model_kind != "knn":
            if "model" not in checkpoint:
                return _fallback("저장된 학습 모델 가중치가 없어 형태 기반 결과만 사용했습니다.")
            device, device_label = resolve_compute_device(device_preference)
            model = CellOrientationNet(
                input_dim=input_dim,
                patch_channels=patch_channels,
                patch_size=patch_size,
                num_classes=num_classes,
            ).to(device)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
    except Exception:
        return _fallback("저장된 학습 모델을 읽는 중 오류가 발생해 형태 기반 결과만 사용했습니다.")

    final_dir = morph_dir.copy().astype(np.uint8)
    restricted_mode = coverage_mode == "restricted"
    ambiguity_threshold = 0.55 if restricted_mode else 0.72
    override_margin = 0.12 if restricted_mode else 0.05
    prob_margin = 0.18 if restricted_mode else 0.10
    trust_threshold = 0.72 if restricted_mode else 0.60

    for i in range(A):
        for j in range(B):
            patch = _extract_cell_patch_tensor(gray_inv, bw_patch_map, edge_patch_map, i, j, A, B)
            raw_support = _patch_signal_strength(patch)
            meta["model_raw_support"][i, j] = np.float32(raw_support)
            if int(occ[i, j]) != 1 and int(morph_dir[i, j]) == 0 and raw_support < PATCH_SIGNAL_THRESHOLD:
                continue
            base_direction = int(morph_dir[i, j])
            base_conf = float(morph_confidence[i, j])
            overlap = int(overlap_count[i, j])
            support_tier = int(support_level[i, j]) if has_detection_aux else 2
            structurally_ambiguous = (
                base_direction == 0
                or base_conf < ambiguity_threshold
                or overlap > 1
                or (has_detection_aux and support_tier <= 1)
            )
            feat = np.asarray(
                _cell_feature_vector(
                    morph_dir,
                    occ,
                    mean_bw,
                    mean_edges,
                    mean_skeleton,
                    head_hint,
                    morph_confidence,
                    global_maps,
                    i,
                    j,
                    context_radius=CONTEXT_RADIUS,
                ),
                dtype=np.float32,
            )[None, :]
            feat_std = (feat - feature_mean) / feature_std
            combined_query = _combine_struct_and_patch_features(feat_std.astype(np.float32), patch[None, ...].astype(np.float32))

            if model_kind == "knn":
                probs, nearest_distance = _v2_predict_knn_probabilities(
                    combined_query.reshape(-1).astype(np.float32),
                    ref_features,
                    ref_labels,
                    num_classes=num_classes,
                    k=knn_k,
                    distance_scale=distance_scale,
                )
            else:
                with torch.no_grad():
                    logits = model(
                        torch.from_numpy(feat_std).float().to(device),
                        torch.from_numpy(patch[None, ...]).float().to(device),
                    )
                    probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)
                if ref_features.size:
                    diff = ref_features - combined_query.astype(np.float32)
                    nearest_distance = float(np.min(np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float64)).astype(np.float32)))
                else:
                    nearest_distance = float("inf")

            predicted_direction = int(np.argmax(probs))
            meta["model_direction"][i, j] = np.uint8(predicted_direction)
            if structurally_ambiguous or predicted_direction != base_direction:
                meta["model_candidate_mask"][i, j] = 1
            top_prob = float(probs[predicted_direction])
            base_prob = float(probs[base_direction]) if 0 <= base_direction < probs.shape[0] else 0.0
            closeness, distance_ok = _v2_query_closeness(nearest_distance, distance_scale, distance_limit)
            support_score = _v2_class_support_score(class_counts, predicted_direction)
            trust_conf = float(top_prob * (0.45 + 0.55 * closeness) * (0.35 + 0.65 * support_score))
            local_pipe_support = _local_pipe_support_score(
                mean_bw,
                mean_edges,
                mean_skeleton,
                head_hint,
                global_maps,
                i,
                j,
            )
            local_directional_signal = _has_local_directional_signal(
                mean_bw,
                mean_edges,
                mean_skeleton,
                head_hint,
                global_maps,
                i,
                j,
            )

            meta["model_probability"][i, j] = np.float32(top_prob)
            meta["model_confidence"][i, j] = np.float32(trust_conf)
            meta["model_distance"][i, j] = np.float32(nearest_distance)

            if predicted_direction == base_direction and distance_ok:
                blended_confidence[i, j] = np.float32(max(base_conf, min(0.98, trust_conf)))
                continue

            is_ambiguous = structurally_ambiguous
            if restricted_mode and base_direction != 0 and base_conf >= 0.45:
                is_ambiguous = overlap > 1 or (has_detection_aux and support_tier <= 1)

            nonzero_fill = base_direction == 0 and predicted_direction != 0
            direction_relabel = base_direction != 0 and predicted_direction not in (0, base_direction)
            zero_override = predicted_direction == 0
            nonzero_fill_plausible = (
                nonzero_fill
                and (
                    not has_detection_aux
                    or support_tier >= 2
                    or int(occ[i, j]) == 1
                    or overlap > 1
                    or raw_support >= (0.16 if restricted_mode else 0.13)
                )
                and (
                    local_directional_signal
                    or raw_support >= (0.20 if restricted_mode else 0.16)
                )
                and max(local_pipe_support, raw_support) >= (0.18 if restricted_mode else 0.15)
            )
            strong_direction_relabel = (
                direction_relabel
                and (overlap > 1 or (has_detection_aux and support_tier <= 1) or base_conf < ambiguity_threshold)
                and distance_ok
                and top_prob >= max(float(use_threshold), 0.80 if restricted_mode else 0.72)
                and top_prob >= base_prob + max(prob_margin, 0.18 if restricted_mode else 0.12)
                and trust_conf >= max(trust_threshold, 0.70 if restricted_mode else 0.62)
                and closeness >= (0.70 if restricted_mode else 0.58)
                and support_score >= (0.55 if restricted_mode else 0.42)
                and max(local_pipe_support, raw_support) >= 0.10
                and (
                    base_conf < ambiguity_threshold
                    or (
                        top_prob >= 0.80
                        and top_prob >= base_prob + 0.20
                        and closeness >= 0.80
                    )
                )
            )
            exceptional_direction_relabel = (
                direction_relabel
                and distance_ok
                and top_prob >= max(float(use_threshold), 0.82 if restricted_mode else 0.78)
                and top_prob >= base_prob + max(prob_margin, 0.20 if restricted_mode else 0.16)
                and trust_conf >= max(trust_threshold, 0.72 if restricted_mode else 0.64)
                and closeness >= (0.78 if restricted_mode else 0.68)
                and support_score >= (0.55 if restricted_mode else 0.45)
            )
            zero_plausible = (
                zero_override
                and base_direction != 0
                and base_conf < 0.52
                and (not has_detection_aux or support_tier <= 1)
                and float(head_hint[i, j]) < 0.5
                and float(mean_bw[i, j]) < 0.18
                and float(mean_skeleton[i, j]) < 0.16
                and float(global_maps["direction_run"][i, j]) < 0.35
                and float(global_maps["incoming_degree"][i, j]) < 0.55
                and raw_support < 0.11
            )

            direct_model_primary = (
                nonzero_fill_plausible
                and (overlap > 1 or not has_detection_aux or support_tier >= 2)
                and distance_ok
                and top_prob >= max(float(use_threshold), 0.84 if restricted_mode else 0.76)
                and top_prob >= base_prob + max(prob_margin, 0.20 if restricted_mode else 0.15)
                and trust_conf >= max(trust_threshold, 0.82 if restricted_mode else 0.72)
                and closeness >= (0.68 if restricted_mode else 0.56)
                and support_score >= (0.60 if restricted_mode else 0.45)
                and raw_support >= (0.18 if restricted_mode else 0.14)
            )

            if direct_model_primary:
                final_dir[i, j] = np.uint8(predicted_direction)
                blended_confidence[i, j] = np.float32(max(base_conf, min(0.98, trust_conf)))
                meta["model_applied_mask"][i, j] = 1
                meta["model_direct_mask"][i, j] = 1
            elif (
                nonzero_fill_plausible
                and (overlap > 1 or not has_detection_aux or support_tier >= 2)
                and distance_ok
                and top_prob >= max(float(use_threshold), 0.82 if restricted_mode else 0.74)
                and top_prob >= base_prob + max(prob_margin, 0.18 if restricted_mode else 0.14)
                and trust_conf >= max(trust_threshold, 0.80 if restricted_mode else 0.70)
                and closeness >= (0.62 if restricted_mode else 0.52)
                and support_score >= (0.60 if restricted_mode else 0.45)
            ):
                final_dir[i, j] = np.uint8(predicted_direction)
                blended_confidence[i, j] = np.float32(max(base_conf, min(0.98, trust_conf)))
                meta["model_applied_mask"][i, j] = 1
                if base_direction == 0:
                    meta["model_direct_mask"][i, j] = 1
            elif (
                base_direction == 0
                and predicted_direction == 0
                and raw_support < PATCH_SIGNAL_THRESHOLD
                and top_prob >= 0.55
            ):
                blended_confidence[i, j] = np.float32(max(base_conf, min(0.95, trust_conf)))
            elif (
                base_direction == 0
                and predicted_direction != 0
                and raw_support < PATCH_SIGNAL_THRESHOLD
            ):
                continue
            elif strong_direction_relabel or exceptional_direction_relabel:
                final_dir[i, j] = np.uint8(predicted_direction)
                blended_confidence[i, j] = np.float32(max(base_conf, min(0.98, top_prob)))
                meta["model_applied_mask"][i, j] = 1
            elif (
                not zero_override
                and not nonzero_fill
                and is_ambiguous
                and distance_ok
                and top_prob >= float(use_threshold)
                and top_prob >= base_prob + prob_margin
                and trust_conf >= trust_threshold
                and trust_conf >= base_conf + override_margin
                and support_score >= (0.65 if restricted_mode else 0.45)
            ):
                final_dir[i, j] = np.uint8(predicted_direction)
                blended_confidence[i, j] = np.float32(max(base_conf, min(0.98, trust_conf)))
                meta["model_applied_mask"][i, j] = 1
            elif (
                zero_plausible
                and distance_ok
                and top_prob >= max(float(use_threshold), 0.72)
                and top_prob >= base_prob + max(prob_margin, 0.20)
                and trust_conf >= max(trust_threshold, 0.70)
                and trust_conf >= base_conf + max(override_margin, 0.10)
                and support_score >= 0.50
            ):
                final_dir[i, j] = np.uint8(0)
                blended_confidence[i, j] = np.float32(max(base_conf, min(0.95, trust_conf)))
                meta["model_applied_mask"][i, j] = 1

    meta["confidence"] = blended_confidence.astype(np.float32)
    meta["device_label"] = np.array([str(device_label)], dtype=object)
    notice_parts = [
        text
        for text in [coverage_reason, patch_warning, collapse_warning, exemplar_fallback_notice]
        if text
    ]
    if notice_parts:
        meta["model_notice"] = np.array([" | ".join(notice_parts)], dtype=object)
    return final_dir, morph_dir, meta
