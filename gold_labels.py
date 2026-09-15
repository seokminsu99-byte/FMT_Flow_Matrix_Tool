# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Uses NumPy: https://github.com/numpy/numpy (BSD-3-Clause plus bundled notices).
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


VALID_LABELS = {0, 1, 2, 3, 4}


@dataclass(frozen=True)
class GoldLabelRecord:
    path: Path
    image_hash: str
    image_path: str
    rows: int
    cols: int
    matrix: np.ndarray
    review_mask: np.ndarray
    review_scope: str
    created_at: str


def image_array_hash(image: np.ndarray) -> str:
    arr = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("ascii", errors="ignore"))
    digest.update(str(arr.dtype).encode("ascii", errors="ignore"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _safe_name(text: str, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", text).strip("._")
    if not cleaned:
        cleaned = "image"
    return cleaned[:max_len]


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def gold_label_path(root: str | Path, image_hash: str, rows: int, cols: int, image_path: str = "") -> Path:
    stem = _safe_name(Path(image_path).stem if image_path else "image")
    short_hash = str(image_hash)[:12]
    return Path(root) / f"{stem}_A{int(rows)}_B{int(cols)}_{short_hash}.npz"


def save_gold_label(
    root: str | Path,
    image: np.ndarray,
    image_path: str,
    matrix: np.ndarray,
    prediction_matrix: Optional[np.ndarray] = None,
    review_mask: Optional[np.ndarray] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    if image is None:
        raise ValueError("image must not be None")
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    labels = set(int(v) for v in np.unique(matrix))
    invalid = sorted(labels - VALID_LABELS)
    if invalid:
        raise ValueError(f"gold matrix contains invalid labels: {invalid}")
    rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
    reviewed = (
        np.ones_like(matrix, dtype=np.uint8)
        if review_mask is None
        else (np.asarray(review_mask) > 0).astype(np.uint8)
    )
    if reviewed.shape != matrix.shape:
        raise ValueError("review_mask must have the same shape as matrix")
    image_hash = image_array_hash(image)
    path = gold_label_path(root, image_hash, rows, cols, image_path)
    matrix_to_save = np.asarray(matrix, dtype=np.uint8).copy()
    if path.exists() and int(np.count_nonzero(reviewed)) < int(reviewed.size):
        try:
            existing = load_gold_label(path)
        except Exception:
            existing = None
        if (
            existing is not None
            and existing.image_hash == image_hash
            and existing.matrix.shape == matrix_to_save.shape
            and existing.review_mask.shape == reviewed.shape
        ):
            keep_existing = (existing.review_mask > 0) & (reviewed == 0)
            matrix_to_save[keep_existing] = existing.matrix[keep_existing]
            reviewed = np.maximum(reviewed, existing.review_mask).astype(np.uint8)
    created_at = _timestamp()
    meta = dict(metadata or {})
    review_scope = str(meta.get("review_scope", "full" if review_mask is None else "partial"))
    meta.update(
        {
            "created_at": created_at,
            "image_path": str(image_path or ""),
            "image_hash": image_hash,
            "rows": rows,
            "cols": cols,
            "review_scope": review_scope,
            "reviewed_cell_count": int(np.count_nonzero(reviewed)),
        }
    )
    pred = (
        np.asarray(prediction_matrix, dtype=np.uint8)
        if prediction_matrix is not None and prediction_matrix.shape == matrix.shape
        else np.zeros_like(matrix, dtype=np.uint8)
    )
    _atomic_savez(
        path,
        matrix=matrix_to_save,
        review_mask=reviewed.astype(np.uint8),
        prediction_matrix=pred,
        image_hash=np.asarray([image_hash], dtype=object),
        image_path=np.asarray([str(image_path or "")], dtype=object),
        created_at=np.asarray([created_at], dtype=object),
        rows=np.asarray([rows], dtype=np.int32),
        cols=np.asarray([cols], dtype=np.int32),
        metadata_json=np.asarray([json.dumps(meta, ensure_ascii=False, default=str)], dtype=object),
    )
    return path


def load_gold_label(path: str | Path) -> GoldLabelRecord:
    label_path = Path(path)
    with np.load(label_path, allow_pickle=True) as data:
        matrix = np.asarray(data["matrix"], dtype=np.uint8)
        if "review_mask" in data:
            review_mask = (np.asarray(data["review_mask"]) > 0).astype(np.uint8)
        else:
            # Legacy GUI files did not distinguish reviewed cells from automatic
            # predictions and therefore cannot support a defensible evaluation.
            review_mask = np.zeros_like(matrix, dtype=np.uint8)
        metadata_text = str(
            np.asarray(data.get("metadata_json", np.asarray(["{}"])), dtype=object).reshape(-1)[0]
        )
        try:
            metadata = json.loads(metadata_text)
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        review_scope = str(
            metadata.get(
                "review_scope",
                "legacy_unverified" if "review_mask" not in data else "partial",
            )
        )
        image_hash = str(np.asarray(data.get("image_hash", np.asarray([""])), dtype=object).reshape(-1)[0])
        image_path = str(np.asarray(data.get("image_path", np.asarray([""])), dtype=object).reshape(-1)[0])
        created_at = str(np.asarray(data.get("created_at", np.asarray([""])), dtype=object).reshape(-1)[0])
        rows = int(np.asarray(data.get("rows", np.asarray([matrix.shape[0]]))).reshape(-1)[0])
        cols = int(np.asarray(data.get("cols", np.asarray([matrix.shape[1]]))).reshape(-1)[0])
    return GoldLabelRecord(
        path=label_path,
        image_hash=image_hash,
        image_path=image_path,
        rows=rows,
        cols=cols,
        matrix=matrix,
        review_mask=review_mask,
        review_scope=review_scope,
        created_at=created_at,
    )


def find_matching_gold_labels(
    root: str | Path,
    image: np.ndarray,
    rows: int,
    cols: int,
) -> list[GoldLabelRecord]:
    label_root = Path(root)
    if not label_root.exists():
        return []
    target_hash = image_array_hash(image)
    matches: list[GoldLabelRecord] = []
    for path in sorted(label_root.glob("*.npz")):
        try:
            record = load_gold_label(path)
        except Exception:
            continue
        if record.image_hash == target_hash and int(record.rows) == int(rows) and int(record.cols) == int(cols):
            matches.append(record)
    matches.sort(key=lambda item: item.created_at)
    return matches


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _f1(precision: float, recall: float) -> float:
    return float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def evaluate_prediction(
    prediction: np.ndarray,
    gold: np.ndarray,
    review_mask: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.uint8)
    truth = np.asarray(gold, dtype=np.uint8)
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch: prediction={pred.shape}, gold={truth.shape}")
    reviewed = (
        np.ones_like(truth, dtype=bool)
        if review_mask is None
        else np.asarray(review_mask).astype(bool)
    )
    if reviewed.shape != truth.shape:
        raise ValueError("review_mask must have the same shape as prediction and gold")
    pred_pipe = (pred > 0) & reviewed
    gold_pipe = (truth > 0) & reviewed
    pipe_tp = int(np.count_nonzero(pred_pipe & gold_pipe))
    pipe_fp = int(np.count_nonzero(pred_pipe & ~gold_pipe))
    pipe_fn = int(np.count_nonzero(~pred_pipe & gold_pipe))
    pipe_tn = int(np.count_nonzero((pred == 0) & (truth == 0) & reviewed))
    pipe_precision = _safe_div(pipe_tp, pipe_tp + pipe_fp)
    pipe_recall = _safe_div(pipe_tp, pipe_tp + pipe_fn)
    pipe_f1 = _f1(pipe_precision, pipe_recall)

    overlap = pred_pipe & gold_pipe
    overlap_total = int(np.count_nonzero(overlap))
    gold_total = int(np.count_nonzero(gold_pipe))
    exact_direction = int(np.count_nonzero((pred == truth) & overlap))
    strict_exact = int(np.count_nonzero((pred == truth) & gold_pipe))
    direction_accuracy_on_overlap = _safe_div(exact_direction, overlap_total)
    strict_direction_recall = _safe_div(strict_exact, gold_total)

    confusion = np.zeros((5, 5), dtype=np.int64)
    for g, p in zip(truth[reviewed].reshape(-1), pred[reviewed].reshape(-1)):
        if int(g) in VALID_LABELS and int(p) in VALID_LABELS:
            confusion[int(g), int(p)] += 1

    class_rows: list[dict[str, Any]] = []
    directional_f1s: list[float] = []
    for code in (1, 2, 3, 4):
        tp = int(confusion[code, code])
        fp = int(np.sum(confusion[:, code]) - tp)
        fn = int(np.sum(confusion[code, :]) - tp)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _f1(precision, recall)
        directional_f1s.append(f1)
        class_rows.append(
            {
                "code": code,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return {
        "shape": tuple(int(v) for v in pred.shape),
        "review": {
            "reviewed_cells": int(np.count_nonzero(reviewed)),
            "unreviewed_cells": int(reviewed.size - np.count_nonzero(reviewed)),
        },
        "pipe": {
            "tp": pipe_tp,
            "fp": pipe_fp,
            "fn": pipe_fn,
            "tn": pipe_tn,
            "precision": pipe_precision,
            "recall": pipe_recall,
            "f1": pipe_f1,
        },
        "direction": {
            "overlap_total": overlap_total,
            "gold_total": gold_total,
            "exact_on_overlap": exact_direction,
            "strict_exact": strict_exact,
            "accuracy_on_overlap": direction_accuracy_on_overlap,
            "strict_recall": strict_direction_recall,
            "macro_f1": float(np.mean(directional_f1s)) if directional_f1s else 0.0,
        },
        "classes": class_rows,
        "confusion": confusion,
    }


def format_evaluation_report(metrics: Mapping[str, Any], gold_path: str | Path | None = None) -> str:
    pipe = metrics["pipe"]
    direction = metrics["direction"]
    lines = ["Gold-label evaluation"]
    if gold_path is not None:
        lines.append(f"gold: {gold_path}")
    lines.append(f"shape: {metrics['shape']}")
    review = metrics.get("review", {})
    lines.append(
        f"reviewed cells: {int(review.get('reviewed_cells', 0))}, "
        f"unreviewed cells: {int(review.get('unreviewed_cells', 0))}"
    )
    lines.append("")
    lines.append(
        "Pipe presence: "
        f"precision={pipe['precision']:.4f}, recall={pipe['recall']:.4f}, F1={pipe['f1']:.4f} "
        f"(TP={pipe['tp']}, FP={pipe['fp']}, FN={pipe['fn']})"
    )
    lines.append(
        "Direction: "
        f"overlap_acc={direction['accuracy_on_overlap']:.4f}, strict_recall={direction['strict_recall']:.4f}, "
        f"macro_F1={direction['macro_f1']:.4f} "
        f"(exact={direction['strict_exact']}/{direction['gold_total']})"
    )
    lines.append("")
    lines.append("Per-direction F1")
    for row in metrics["classes"]:
        lines.append(
            f"{row['code']}: precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
            f"F1={row['f1']:.4f}, TP={row['tp']}, FP={row['fp']}, FN={row['fn']}"
        )
    lines.append("")
    lines.append("Confusion matrix rows=gold 0..4, cols=pred 0..4")
    confusion = np.asarray(metrics["confusion"], dtype=np.int64)
    for row in confusion:
        lines.append(" ".join(str(int(v)) for v in row))
    return "\n".join(lines)


def compact_metric_summary(metrics: Mapping[str, Any]) -> str:
    pipe = metrics["pipe"]
    direction = metrics["direction"]
    review = metrics.get("review", {})
    return (
        f"reviewed={int(review.get('reviewed_cells', 0))}, "
        f"Gold 평가: pipe F1={pipe['f1']:.3f}, "
        f"direction strict={direction['strict_recall']:.3f}, "
        f"direction macroF1={direction['macro_f1']:.3f}"
    )
