# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Uses NumPy: https://github.com/numpy/numpy (BSD-3-Clause plus bundled notices).
# Uses Pillow: https://github.com/python-pillow/Pillow (MIT-CMU).
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gamma_index import GammaResult


@dataclass(frozen=True)
class WidthFunctionBlock:
    k: Optional[int]
    label: str
    xi: np.ndarray
    original: np.ndarray
    mean: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    run_count: int


@dataclass(frozen=True)
class NSEPoint:
    k: int
    beta: float
    ok_total: str
    mean_nse_run: float
    nse_mean_q: float


def _to_float(text: object) -> float:
    try:
        return float(str(text).strip())
    except Exception:
        return float("nan")


def _parse_k_from_header(label: str) -> Optional[int]:
    compact = str(label).strip().replace(" ", "")
    match = re.search(r"10\^([-+]?\d+)", compact)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:^|[_-])k=?([-+]?\d+)(?:$|[_-])", compact, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _read_header(path: Path) -> Tuple[list[str], bool]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        first = next(csv.reader(fh), [])
    has_header = any(re.search(r"[A-Za-z^]", cell) for cell in first)
    return first, has_header


def load_width_function_blocks(
    path: str | Path,
    runs_per_beta: Optional[int] = None,
    default_k_start: int = -4,
) -> list[WidthFunctionBlock]:
    csv_path = Path(path)
    header, has_header = _read_header(csv_path)
    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        skip_header=1 if has_header else 0,
        invalid_raise=False,
        filling_values=np.nan,
    )
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        raise ValueError(f"empty PLENA width CSV: {csv_path}")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"PLENA width CSV needs distance, original, and run columns: {csv_path}")

    xi = np.nan_to_num(data[:, 0], nan=0.0)
    if xi.size and np.allclose(xi, 0.0):
        xi = np.arange(xi.size, dtype=float)
    original = np.nan_to_num(data[:, 1], nan=0.0)
    run_values = np.nan_to_num(data[:, 2:], nan=0.0)

    labels = header[2:] if has_header else []
    if len(labels) < run_values.shape[1]:
        labels = labels + [f"run{idx + 1}" for idx in range(len(labels), run_values.shape[1])]

    groups: list[tuple[Optional[int], int, int]] = []
    col = 0
    block_index = 0
    while col < run_values.shape[1]:
        k = _parse_k_from_header(labels[col]) if col < len(labels) else None
        start = col
        if k is not None:
            col += 1
            while col < run_values.shape[1]:
                next_k = _parse_k_from_header(labels[col]) if col < len(labels) else None
                if next_k != k:
                    break
                col += 1
        else:
            count = runs_per_beta if runs_per_beta and runs_per_beta > 0 else 100
            k = default_k_start + block_index
            col = min(run_values.shape[1], col + int(count))
        groups.append((k, start, col))
        block_index += 1

    blocks: list[WidthFunctionBlock] = []
    for k, start, end in groups:
        values = run_values[:, start:end]
        if values.size == 0:
            continue
        label = f"10^{k}" if k is not None else f"block {len(blocks) + 1}"
        blocks.append(
            WidthFunctionBlock(
                k=k,
                label=label,
                xi=xi.copy(),
                original=original.copy(),
                mean=np.mean(values, axis=1),
                minimum=np.min(values, axis=1),
                maximum=np.max(values, axis=1),
                run_count=int(values.shape[1]),
            )
        )
    return blocks


def parse_nse_points_from_text(text: str) -> list[NSEPoint]:
    points: list[NSEPoint] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("[") or stripped.lower().startswith("k"):
            continue
        parts = re.split(r"\s+", stripped)
        if len(parts) < 5 or not re.fullmatch(r"[-+]?\d+", parts[0]):
            continue
        value_index = 6 if len(parts) >= 7 else 4
        try:
            points.append(
                NSEPoint(
                    k=int(parts[0]),
                    beta=float(parts[1]),
                    ok_total=parts[2],
                    mean_nse_run=float(parts[3]),
                    nse_mean_q=float(parts[value_index]),
                )
            )
        except ValueError:
            continue
    return points


def load_nse_points(path: str | Path) -> list[NSEPoint]:
    return parse_nse_points_from_text(Path(path).read_text(encoding="utf-8", errors="replace"))


def nse_points_from_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[NSEPoint]:
    points: list[NSEPoint] = []
    for row in rows:
        try:
            points.append(
                NSEPoint(
                    k=int(str(row.get("k", "")).strip()),
                    beta=float(str(row.get("beta", "")).strip()),
                    ok_total=str(row.get("ok_total", "")).strip(),
                    mean_nse_run=float(str(row.get("mean_nse_run", "")).strip()),
                    nse_mean_q=float(str(row.get("nse_mean_q", "")).strip()),
                )
            )
        except ValueError:
            continue
    return points


def infer_nse_path_from_width_csv(path: str | Path) -> Optional[Path]:
    width_path = Path(path)
    candidates: list[Path] = []
    name = width_path.name
    if name.endswith(".txt_width_functions.csv"):
        candidates.append(width_path.with_name(name[: -len(".txt_width_functions.csv")] + "_NSE.txt"))
    if name.endswith("_width_functions.csv"):
        candidates.append(width_path.with_name(name[: -len("_width_functions.csv")] + "_NSE.txt"))
    candidates.append(width_path.with_suffix("").with_name(width_path.stem + "_NSE.txt"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(width_path.parent.glob("*_NSE.txt"))
    return matches[0] if len(matches) == 1 else None


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf", "calibri.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _finite(values: Iterable[np.ndarray | Sequence[float]]) -> np.ndarray:
    arrays = [np.asarray(value, dtype=float).ravel() for value in values]
    if not arrays:
        return np.asarray([], dtype=float)
    merged = np.concatenate(arrays)
    return merged[np.isfinite(merged)]


def _limits(values: Iterable[np.ndarray | Sequence[float]], include_zero: bool = False) -> tuple[float, float]:
    finite = _finite(values)
    if include_zero:
        finite = np.concatenate([finite, np.asarray([0.0])])
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.min(finite))
    high = float(np.max(finite))
    if math.isclose(low, high):
        pad = max(abs(low) * 0.1, 1.0)
    else:
        pad = (high - low) * 0.06
    return low - pad, high + pad


def _nice_ticks(low: float, high: float, target: int = 6) -> list[float]:
    if not (math.isfinite(low) and math.isfinite(high)) or math.isclose(low, high):
        return [low]
    span = abs(high - low)
    raw_step = span / max(target, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    step = magnitude
    for factor in (1, 2, 5, 10):
        candidate = factor * magnitude
        if raw_step <= candidate:
            step = candidate
            break
    start = math.ceil(low / step) * step
    ticks: list[float] = []
    value = start
    limit = high + step * 1e-9
    while value <= limit and len(ticks) < 32:
        ticks.append(0.0 if abs(value) < step * 1e-9 else value)
        value += step
    return ticks


def _tick_label(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if math.isclose(value, round(value), abs_tol=1e-8):
        return str(int(round(value)))
    return f"{value:.2g}"


def _draw_plot_frame(
    draw: ImageDraw.ImageDraw,
    size: tuple[int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    ylabel: str,
) -> tuple[tuple[int, int, int, int], callable]:
    width, height = size
    scale = max(0.75, min(width / 900.0, height / 650.0))
    left = int(78 * scale)
    top = int(34 * scale)
    right = width - int(32 * scale)
    bottom = height - int(72 * scale)
    axis_font = _font(max(10, int(12 * scale)))
    label_font = _font(max(12, int(15 * scale)))
    grid_color = (0, 0, 0, 36)
    axis_color = (0, 0, 0, 255)

    plot_w = max(1, right - left)
    plot_h = max(1, bottom - top)
    xmin, xmax = xlim
    ymin, ymax = ylim
    if math.isclose(xmin, xmax):
        xmax = xmin + 1.0
    if math.isclose(ymin, ymax):
        ymax = ymin + 1.0

    def project(x: float, y: float) -> tuple[float, float]:
        px = left + (float(x) - xmin) / (xmax - xmin) * plot_w
        py = bottom - (float(y) - ymin) / (ymax - ymin) * plot_h
        return px, py

    draw.rectangle((left, top, right, bottom), outline=axis_color, width=max(1, int(1.2 * scale)))

    for tick in _nice_ticks(xmin, xmax):
        px, _ = project(tick, ymin)
        draw.line((px, top, px, bottom), fill=grid_color, width=1)
        draw.line((px, bottom, px, bottom + 5 * scale), fill=axis_color, width=1)
        text = _tick_label(tick)
        bbox = draw.textbbox((0, 0), text, font=axis_font)
        draw.text((px - (bbox[2] - bbox[0]) / 2, bottom + 8 * scale), text, fill=axis_color, font=axis_font)

    for tick in _nice_ticks(ymin, ymax):
        _, py = project(xmin, tick)
        draw.line((left, py, right, py), fill=grid_color, width=1)
        draw.line((left - 5 * scale, py, left, py), fill=axis_color, width=1)
        text = _tick_label(tick)
        bbox = draw.textbbox((0, 0), text, font=axis_font)
        draw.text((left - 10 * scale - (bbox[2] - bbox[0]), py - (bbox[3] - bbox[1]) / 2), text, fill=axis_color, font=axis_font)

    bbox = draw.textbbox((0, 0), xlabel, font=label_font)
    draw.text(((left + right - (bbox[2] - bbox[0])) / 2, height - 34 * scale), xlabel, fill=axis_color, font=label_font)

    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1), (255, 255, 255, 0)))
    ylabel_box = measure.textbbox((0, 0), ylabel, font=label_font)
    ylabel_width = max(1, int(math.ceil(ylabel_box[2] - ylabel_box[0] + 6 * scale)))
    ylabel_height = max(1, int(math.ceil(ylabel_box[3] - ylabel_box[1] + 6 * scale)))
    ylabel_img = Image.new("RGBA", (ylabel_width, ylabel_height), (255, 255, 255, 0))
    ylabel_draw = ImageDraw.Draw(ylabel_img)
    ylabel_draw.text(
        (3 * scale - ylabel_box[0], 3 * scale - ylabel_box[1]),
        ylabel,
        fill=axis_color,
        font=label_font,
    )
    ylabel_img = ylabel_img.rotate(90, expand=True)
    return (left, top, right, bottom), project, ylabel_img


def _draw_polyline(draw: ImageDraw.ImageDraw, points: Sequence[tuple[float, float]], color: tuple[int, int, int, int], width: int) -> None:
    finite_points = [(float(x), float(y)) for x, y in points if math.isfinite(x) and math.isfinite(y)]
    if len(finite_points) >= 2:
        draw.line(finite_points, fill=color, width=max(1, width), joint="curve")


def render_width_function_plot(
    block: WidthFunctionBlock,
    size: tuple[int, int] = (900, 650),
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> Image.Image:
    width = max(360, int(size[0]))
    height = max(300, int(size[1]))
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    resolved_xlim = xlim if xlim is not None else _limits([block.xi], include_zero=True)
    resolved_ylim = ylim if ylim is not None else _limits([block.original, block.minimum, block.maximum, block.mean], include_zero=True)
    frame, project, ylabel_img = _draw_plot_frame(draw, (width, height), resolved_xlim, resolved_ylim, "Distance (xi)", "W(xi)")
    left, top, right, bottom = frame
    image.alpha_composite(ylabel_img, (max(2, int(left * 0.16)), int((top + bottom - ylabel_img.height) / 2)))

    lower = [project(x, y) for x, y in zip(block.xi, block.minimum)]
    upper = [project(x, y) for x, y in zip(block.xi, block.maximum)]
    polygon = lower + list(reversed(upper))
    if len(polygon) >= 3:
        draw.polygon(polygon, fill=(115, 115, 115, 135))

    scale = max(0.75, min(width / 900.0, height / 650.0))
    line_w = max(2, int(2.0 * scale))
    _draw_polyline(draw, [project(x, y) for x, y in zip(block.xi, block.mean)], (0, 0, 0, 255), line_w)
    _draw_polyline(draw, [project(x, y) for x, y in zip(block.xi, block.original)], (220, 0, 0, 255), line_w)

    legend_font = _font(max(10, int(12 * scale)))
    legend_items = [
        ("Runs min-max band", (115, 115, 115, 135)),
        ("Run average", (0, 0, 0, 255)),
        ("Original", (220, 0, 0, 255)),
    ]
    item_h = int(22 * scale)
    legend_w = int(190 * scale)
    legend_h = item_h * len(legend_items) + int(12 * scale)
    lx = right - legend_w - int(12 * scale)
    ly = top + int(12 * scale)
    draw.rectangle((lx, ly, lx + legend_w, ly + legend_h), fill=(255, 255, 255, 220), outline=(0, 0, 0, 180))
    for idx, (text, color) in enumerate(legend_items):
        y = ly + int(9 * scale) + idx * item_h
        if idx == 0:
            draw.rectangle((lx + int(10 * scale), y + int(3 * scale), lx + int(36 * scale), y + int(13 * scale)), fill=color)
        else:
            draw.line((lx + int(10 * scale), y + int(8 * scale), lx + int(38 * scale), y + int(8 * scale)), fill=color, width=line_w)
        draw.text((lx + int(46 * scale), y), text, fill=(0, 0, 0, 255), font=legend_font)

    return image.convert("RGB")


def render_nse_plot(
    points: Sequence[NSEPoint],
    size: tuple[int, int] = (720, 650),
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> Image.Image:
    if not points:
        raise ValueError("NSE points are empty")
    width = max(360, int(size[0]))
    height = max(300, int(size[1]))
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    xs = np.asarray([p.k for p in points], dtype=float)
    ys = np.asarray([p.nse_mean_q for p in points], dtype=float)
    resolved_xlim = xlim if xlim is not None else _limits([xs], include_zero=False)
    resolved_ylim = ylim if ylim is not None else _limits([ys, np.asarray([1.0])], include_zero=False)
    frame, project, ylabel_img = _draw_plot_frame(draw, (width, height), resolved_xlim, resolved_ylim, "k", "NSE")
    left, top, right, bottom = frame
    image.alpha_composite(ylabel_img, (max(2, int(left * 0.24)), int((top + bottom - ylabel_img.height) / 2)))

    scale = max(0.75, min(width / 720.0, height / 650.0))
    y1_left = project(resolved_xlim[0], 1.0)
    y1_right = project(resolved_xlim[1], 1.0)
    _draw_polyline(draw, [y1_left, y1_right], (220, 0, 0, 255), max(2, int(2 * scale)))
    radius = max(5, int(7 * scale))
    for x, y in zip(xs, ys):
        px, py = project(float(x), float(y))
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=(20, 80, 220, 255), width=max(2, int(2 * scale)))

    return image.convert("RGB")


def render_gamma_plot(
    result: GammaResult,
    size: tuple[int, int] = (900, 650),
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> Image.Image:
    if result.unique_lengths.size == 0:
        raise ValueError("gamma distribution is empty")
    width = max(420, int(size[0]))
    height = max(320, int(size[1]))
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    xs = result.unique_lengths.astype(float)
    ys = result.weighted_values.astype(float)
    resolved_xlim = xlim if xlim is not None else (min(1.0, float(np.min(xs))) - 0.5, float(np.max(xs)) + 0.5)
    max_y = max(float(np.max(ys)), 1.0)
    resolved_ylim = ylim if ylim is not None else (0.0, max_y * 1.16)
    frame, project, ylabel_img = _draw_plot_frame(
        draw,
        (width, height),
        resolved_xlim,
        resolved_ylim,
        "Removed flowline length, x",
        "Length x occurrence count, y",
    )
    left, top, right, bottom = frame
    image.alpha_composite(ylabel_img, (max(2, int(left * 0.13)), int((top + bottom - ylabel_img.height) / 2)))

    scale = max(0.75, min(width / 900.0, height / 650.0))
    baseline_y = project(resolved_xlim[0], 0.0)[1]
    bar_half_width = 0.4
    for x, y in zip(xs, ys):
        x0, y0 = project(float(x) - bar_half_width, float(y))
        x1, _ = project(float(x) + bar_half_width, 0.0)
        draw.rectangle(
            (max(left, x0), max(top, y0), min(right, x1), min(bottom, baseline_y)),
            fill=(74, 126, 187, 210),
            outline=(38, 78, 126, 255),
            width=max(1, int(scale)),
        )

    gravity_x = float(result.gravity_center_x)
    if math.isfinite(gravity_x):
        gravity_px, _ = project(gravity_x, 0.0)
        draw.line((gravity_px, top, gravity_px, bottom), fill=(210, 24, 24, 255), width=max(2, int(3 * scale)))
    max_px, _ = project(float(result.maximum_length), 0.0)
    dash = max(4, int(7 * scale))
    gap = max(3, int(5 * scale))
    y = top
    while y < bottom:
        draw.line((max_px, y, max_px, min(bottom, y + dash)), fill=(0, 0, 0, 255), width=max(1, int(2 * scale)))
        y += dash + gap

    title_font = _font(max(12, int(16 * scale)), bold=True)
    note_font = _font(max(10, int(12 * scale)))
    title = "Figure 6. Distance-weighted Distribution with Center of Gravity"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, max(2, int(4 * scale))), title, fill=(0, 0, 0, 255), font=title_font)
    gamma_text = f"Gamma = centerX / max(x) = {result.gamma:.4f}"
    draw.text((left + int(10 * scale), top + int(8 * scale)), gamma_text, fill=(0, 0, 0, 255), font=note_font)

    gravity_label = f"Gravity X = {result.gravity_center_x:.3f}"
    max_label = f"max(x) = {result.maximum_length}"
    label_y = bottom - int(24 * scale)
    if math.isfinite(gravity_x):
        gravity_box = draw.textbbox((0, 0), gravity_label, font=note_font)
        draw.text(
            (min(max(left, gravity_px - (gravity_box[2] - gravity_box[0]) / 2), right - (gravity_box[2] - gravity_box[0])), label_y),
            gravity_label,
            fill=(210, 24, 24, 255),
            font=note_font,
        )
    max_box = draw.textbbox((0, 0), max_label, font=note_font)
    draw.text(
        (min(max(left, max_px - (max_box[2] - max_box[0]) / 2), right - (max_box[2] - max_box[0])), top + int(28 * scale)),
        max_label,
        fill=(0, 0, 0, 255),
        font=note_font,
    )
    return image.convert("RGB")
