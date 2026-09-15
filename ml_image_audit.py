# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Uses OpenCV: https://github.com/opencv/opencv (Apache-2.0); opencv-python wheel notices apply.
# Uses NumPy: https://github.com/numpy/numpy (BSD-3-Clause plus bundled notices).
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

import gold_labels
import solution


DEFAULT_IMAGE_DIR = Path("images")
DEFAULT_ROWS = (20, 30, 50)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"empty or unreadable image file: {image_path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode image file: {image_path}")
    return image


def iter_image_files(root: str | Path) -> list[Path]:
    image_root = Path(root)
    if not image_root.exists():
        raise FileNotFoundError(str(image_root))
    return sorted(
        [path for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda path: path.name,
    )


def _direction_counts(values: np.ndarray) -> dict[str, int]:
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    return {f"dir_{code}": int(np.count_nonzero(flat == code)) for code in (1, 2, 3, 4)}


def _compact_metrics(prefix: str, metrics: dict[str, Any], row: dict[str, Any]) -> None:
    pipe = metrics["pipe"]
    direction = metrics["direction"]
    row[f"{prefix}_pipe_precision"] = float(pipe["precision"])
    row[f"{prefix}_pipe_recall"] = float(pipe["recall"])
    row[f"{prefix}_pipe_f1"] = float(pipe["f1"])
    row[f"{prefix}_direction_strict_recall"] = float(direction["strict_recall"])
    row[f"{prefix}_direction_overlap_acc"] = float(direction["accuracy_on_overlap"])
    row[f"{prefix}_direction_macro_f1"] = float(direction["macro_f1"])


def evaluate_image(
    image_path: str | Path,
    rows: Sequence[int] = DEFAULT_ROWS,
    gold_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    path = Path(image_path)
    image = read_image(path)
    records: list[dict[str, Any]] = []
    for row_count in rows:
        matrix, occ, boxes, meta = solution.compute_direction_matrix_with_meta(image, int(row_count))
        object_matrix = np.asarray(meta.get("object_direction_matrix", np.zeros_like(matrix)), dtype=np.uint8)
        object_support = np.asarray(meta.get("object_support_mask", np.zeros_like(matrix)), dtype=np.uint8)
        centerline_support = np.asarray(
            meta.get("centerline_support_mask", np.zeros_like(matrix)),
            dtype=np.uint8,
        )
        centerline_count = int(np.count_nonzero(centerline_support))
        centerline_assigned = int(np.count_nonzero((centerline_support > 0) & (matrix > 0)))
        detection_rows = np.asarray(meta.get("candidate_detection_rows", []), dtype=np.int32).reshape(-1).tolist()
        detection_counts = np.asarray(meta.get("candidate_count_by_row", []), dtype=np.int32).reshape(-1).tolist()
        object_directions = np.asarray(meta.get("object_directions", []), dtype=np.uint8).reshape(-1)
        record: dict[str, Any] = {
            "image": path.name,
            "image_path": str(path),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "rows": int(matrix.shape[0]),
            "cols": int(matrix.shape[1]),
            "boxes": int(len(boxes)),
            "object_count": int(np.asarray(meta.get("object_count", [0])).reshape(-1)[0]),
            "final_nonzero": int(np.count_nonzero(matrix)),
            "object_nonzero": int(np.count_nonzero(object_matrix)),
            "object_support_nonzero": int(np.count_nonzero(object_support)),
            "centerline_nonzero": centerline_count,
            "centerline_assigned": centerline_assigned,
            "centerline_coverage": float(centerline_assigned / max(1, centerline_count)),
            "centerline_fill_count": int(
                np.asarray(meta.get("centerline_fill_count", [0]), dtype=np.int32).reshape(-1)[0]
            ),
            "centerline_unassigned": int(
                np.asarray(
                    meta.get(
                        "centerline_unassigned_count",
                        [np.count_nonzero((centerline_support > 0) & (matrix == 0))],
                    ),
                    dtype=np.int32,
                ).reshape(-1)[0]
            ),
            "occupied_nonzero": int(np.count_nonzero(occ)),
            "final_density": float(np.count_nonzero(matrix) / max(1, matrix.size)),
            "object_density": float(np.count_nonzero(object_matrix) / max(1, object_matrix.size)),
            "occupied_density": float(np.count_nonzero(occ) / max(1, occ.size)),
            "detection_rows": ";".join(str(int(v)) for v in detection_rows),
            "candidate_count_by_row": ";".join(str(int(v)) for v in detection_counts),
            "has_gold": False,
        }
        record.update(_direction_counts(object_directions))
        if gold_root is not None:
            matches = gold_labels.find_matching_gold_labels(gold_root, image, matrix.shape[0], matrix.shape[1])
            if matches:
                gold = matches[-1]
                record["has_gold"] = True
                record["gold_path"] = str(gold.path)
                final_metrics = gold_labels.evaluate_prediction(
                    matrix,
                    gold.matrix,
                    review_mask=gold.review_mask,
                )
                object_metrics = gold_labels.evaluate_prediction(
                    object_matrix,
                    gold.matrix,
                    review_mask=gold.review_mask,
                )
                _compact_metrics("final", final_metrics, record)
                _compact_metrics("object", object_metrics, record)
        records.append(record)
    return records


def evaluate_image_directory(
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    rows: Sequence[int] = DEFAULT_ROWS,
    gold_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_image_files(image_dir):
        records.extend(evaluate_image(path, rows=rows, gold_root=gold_root))
    return records


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_image: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_image.setdefault(str(record["image"]), []).append(dict(record))
    object_unstable: list[dict[str, Any]] = []
    final_density_warnings: list[dict[str, Any]] = []
    centerline_coverage_warnings: list[dict[str, Any]] = []
    for image_name, items in sorted(by_image.items()):
        object_counts = np.asarray([int(item["object_count"]) for item in items], dtype=np.float64)
        final_density = np.asarray([float(item["final_density"]) for item in items], dtype=np.float64)
        centerline_coverage = np.asarray(
            [float(item.get("centerline_coverage", 0.0)) for item in items],
            dtype=np.float64,
        )
        if object_counts.size == 0:
            continue
        object_span = int(np.max(object_counts) - np.min(object_counts))
        object_ratio = float(np.max(object_counts) / max(1.0, np.min(object_counts)))
        final_density_span = float(np.max(final_density) - np.min(final_density))
        final_density_ratio = float(np.max(final_density) / max(1e-9, np.min(final_density)))
        object_item = {
            "image": image_name,
            "object_span": object_span,
            "object_ratio": object_ratio,
        }
        density_item = {
            "image": image_name,
            "final_density_span": final_density_span,
            "final_density_ratio": final_density_ratio,
        }
        if object_span >= 8 or object_ratio >= 2.0:
            object_unstable.append(
                {
                    **object_item,
                    "final_density_span": final_density_span,
                    "final_density_ratio": final_density_ratio,
                }
            )
        if final_density_ratio >= 2.0:
            final_density_warnings.append({**density_item, **object_item})
        if centerline_coverage.size and float(np.min(centerline_coverage)) < 0.70:
            centerline_coverage_warnings.append(
                {
                    "image": image_name,
                    "minimum_centerline_coverage": float(np.min(centerline_coverage)),
                    "maximum_centerline_coverage": float(np.max(centerline_coverage)),
                }
            )
    gold_records = [record for record in records if bool(record.get("has_gold"))]
    summary: dict[str, Any] = {
        "image_count": len(by_image),
        "row_runs": len(records),
        "gold_row_runs": len(gold_records),
        "object_unstable_images": object_unstable,
        "final_density_warnings": final_density_warnings,
        "centerline_coverage_warnings": centerline_coverage_warnings,
        "unstable_images": object_unstable,
    }
    if gold_records:
        for prefix in ("final", "object"):
            for metric in ("pipe_f1", "direction_strict_recall", "direction_macro_f1"):
                key = f"{prefix}_{metric}"
                values = [float(record[key]) for record in gold_records if key in record]
                if values:
                    summary[f"mean_{key}"] = float(np.mean(values))
    return summary


def write_csv(records: Sequence[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for record in records:
        for key in record.keys():
            if key not in keys:
                keys.append(key)
    with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_json(payload: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _parse_rows(text: str) -> tuple[int, ...]:
    rows = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not rows:
        raise argparse.ArgumentTypeError("rows must contain at least one integer")
    if any(row <= 0 for row in rows):
        raise argparse.ArgumentTypeError("rows must be positive")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate NFMAT image arrow recognition over a folder.")
    parser.add_argument("--image-dir", required=True, help="Explicit image folder to audit; no private default path.")
    parser.add_argument("--rows", type=_parse_rows, default=DEFAULT_ROWS, help="Comma-separated row sizes, e.g. 20,30,50.")
    parser.add_argument("--gold-root", default=None, help="Optional folder containing human gold-label .npz files.")
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--json", default=None, help="Optional JSON output path.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    records = evaluate_image_directory(args.image_dir, rows=args.rows, gold_root=args.gold_root)
    summary = summarize_records(records)
    if args.csv:
        write_csv(records, args.csv)
    if args.json:
        write_json({"summary": summary, "records": records}, args.json)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
