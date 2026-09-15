# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Uses NumPy: https://github.com/numpy/numpy (BSD-3-Clause plus bundled notices).
# Uses pyproj / PROJ: https://github.com/pyproj4/pyproj (MIT).
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from pyproj import CRS, Transformer
except ImportError:  # pragma: no cover - exercised by release smoke tests without dependencies.
    CRS = None  # type: ignore[assignment]
    Transformer = None  # type: ignore[assignment]


Point = Tuple[float, float]
NestedPointGroups = Sequence[Sequence[Sequence[Point]]]


class CRSDependencyError(RuntimeError):
    pass


class CRSParseError(ValueError):
    pass


def _require_pyproj() -> None:
    if CRS is None or Transformer is None:
        raise CRSDependencyError(
            "CRS support requires pyproj. Install the release dependencies before loading GIS data."
        )


@dataclass(frozen=True)
class CRSDefinition:
    wkt: str
    label: str
    name: str
    authority_name: str = ""
    authority_code: str = ""
    unit_name: str = ""
    unit_type: str = "unknown"
    unit_to_meter: float | None = None
    is_projected: bool = False
    is_geographic: bool = False

    def to_meta(self) -> Dict[str, object]:
        return {
            "crs_valid": True,
            "crs_wkt": self.wkt,
            "crs_label": self.label,
            "crs_name": self.name,
            "crs_authority": self.authority_name,
            "crs_code": self.authority_code,
            "coordinate_unit_name": self.unit_name,
            "coordinate_unit_type": self.unit_type,
            "coordinate_unit_to_meter": self.unit_to_meter,
        }


def parse_crs(value: str | CRSDefinition) -> CRSDefinition:
    if isinstance(value, CRSDefinition):
        return value
    text = str(value).strip()
    if not text:
        raise CRSParseError("CRS input is empty.")
    _require_pyproj()
    try:
        parsed = CRS.from_user_input(text)
    except Exception as exc:
        raise CRSParseError(f"Could not parse CRS: {exc}") from exc

    authority = parsed.to_authority(min_confidence=70)
    authority_name = str(authority[0]) if authority else ""
    authority_code = str(authority[1]) if authority else ""
    name = str(parsed.name or "Unnamed CRS")
    label = f"{authority_name}:{authority_code}" if authority else name

    axis = parsed.axis_info[0] if parsed.axis_info else None
    unit_name = str(axis.unit_name or "") if axis is not None else ""
    factor: float | None = None
    if axis is not None and axis.unit_conversion_factor is not None:
        try:
            candidate = float(axis.unit_conversion_factor)
            if math.isfinite(candidate) and candidate > 0.0:
                factor = candidate
        except (TypeError, ValueError):
            factor = None

    if parsed.is_geographic:
        unit_type = "degree"
        unit_to_meter = None
    elif parsed.is_projected:
        unit_to_meter = factor
        unit_type = "meter" if factor is not None and abs(factor - 1.0) <= 1e-12 else "linear"
    else:
        unit_type = "unknown"
        unit_to_meter = None

    return CRSDefinition(
        wkt=parsed.to_wkt(version="WKT2_2019", pretty=False),
        label=label,
        name=name,
        authority_name=authority_name,
        authority_code=authority_code,
        unit_name=unit_name,
        unit_type=unit_type,
        unit_to_meter=unit_to_meter,
        is_projected=bool(parsed.is_projected),
        is_geographic=bool(parsed.is_geographic),
    )


def crs_from_meta(meta: Dict[str, object] | None) -> CRSDefinition | None:
    if not meta:
        return None
    wkt = str(meta.get("crs_wkt", "") or "").strip()
    if not wkt:
        return None
    return parse_crs(wkt)


def read_shapefile_crs(shp_path: str | Path) -> CRSDefinition | None:
    prj_path = Path(shp_path).with_suffix(".prj")
    if not prj_path.exists():
        return None
    text = prj_path.read_text(encoding="utf-8-sig", errors="ignore").strip()
    if not text:
        raise CRSParseError(f"PRJ file is empty: {prj_path.name}")
    try:
        return parse_crs(text)
    except CRSParseError as exc:
        raise CRSParseError(f"Invalid PRJ file ({prj_path.name}): {exc}") from exc


def read_shapefile_crs_meta(shp_path: str | Path) -> Dict[str, object]:
    prj_path = Path(shp_path).with_suffix(".prj")
    base: Dict[str, object] = {
        "prj_available": bool(prj_path.exists()),
        "crs_valid": False,
        "crs_wkt": "",
        "crs_label": "",
        "crs_name": "",
        "crs_authority": "",
        "crs_code": "",
        "crs_source": "missing" if not prj_path.exists() else "prj",
        "crs_error": "",
        "coordinate_unit_name": "",
        "coordinate_unit_type": "unknown",
        "coordinate_unit_to_meter": None,
    }
    if not prj_path.exists():
        return base
    try:
        definition = read_shapefile_crs(shp_path)
    except (CRSDependencyError, CRSParseError) as exc:
        base["crs_error"] = str(exc)
        return base
    if definition is not None:
        base.update(definition.to_meta())
    return base


def crs_equivalent(first: CRSDefinition | str, second: CRSDefinition | str) -> bool:
    left = parse_crs(first)
    right = parse_crs(second)
    if (
        left.authority_name
        and left.authority_code
        and right.authority_name
        and right.authority_code
        and left.authority_name.upper() == right.authority_name.upper()
        and left.authority_code == right.authority_code
    ):
        return True
    _require_pyproj()
    left_crs = CRS.from_wkt(left.wkt)
    right_crs = CRS.from_wkt(right.wkt)
    return bool(left_crs.equals(right_crs, ignore_axis_order=True))


def choose_project_crs(definitions: Sequence[CRSDefinition | None]) -> CRSDefinition | None:
    available = [definition for definition in definitions if definition is not None]
    if not available:
        return None
    for definition in available:
        if definition.is_projected and definition.unit_to_meter is not None:
            return definition
    return available[0]


def transform_nested_points(
    groups: NestedPointGroups,
    source_crs: CRSDefinition | str,
    target_crs: CRSDefinition | str,
) -> List[List[List[Point]]]:
    source = parse_crs(source_crs)
    target = parse_crs(target_crs)
    if crs_equivalent(source, target):
        return [
            [[(float(point[0]), float(point[1])) for point in part] for part in group]
            for group in groups
        ]

    _require_pyproj()
    structure = [[len(part) for part in group] for group in groups]
    point_count = sum(sum(lengths) for lengths in structure)
    if point_count == 0:
        return [[[] for _part in group] for group in groups]

    x_values = np.empty(point_count, dtype=np.float64)
    y_values = np.empty(point_count, dtype=np.float64)
    cursor = 0
    for group in groups:
        for part in group:
            length = len(part)
            if length:
                coordinates = np.asarray(part, dtype=np.float64)
                if coordinates.ndim != 2 or coordinates.shape[1] < 2:
                    raise ValueError("GIS point data must contain x and y coordinates.")
                x_values[cursor: cursor + length] = coordinates[:, 0]
                y_values[cursor: cursor + length] = coordinates[:, 1]
            cursor += length

    transformer = Transformer.from_crs(
        CRS.from_wkt(source.wkt),
        CRS.from_wkt(target.wkt),
        always_xy=True,
    )
    try:
        transformed_x, transformed_y = transformer.transform(x_values, y_values, errcheck=True)
    except Exception as exc:
        raise ValueError(f"Coordinate transformation failed ({source.label} -> {target.label}): {exc}") from exc
    transformed_x = np.asarray(transformed_x, dtype=np.float64)
    transformed_y = np.asarray(transformed_y, dtype=np.float64)
    if not np.isfinite(transformed_x).all() or not np.isfinite(transformed_y).all():
        raise ValueError(f"Coordinate transformation produced non-finite values ({source.label} -> {target.label}).")

    output: List[List[List[Point]]] = []
    cursor = 0
    for lengths in structure:
        group_output: List[List[Point]] = []
        for length in lengths:
            end = cursor + length
            group_output.append(
                [
                    (float(x_value), float(y_value))
                    for x_value, y_value in zip(transformed_x[cursor:end], transformed_y[cursor:end])
                ]
            )
            cursor = end
        output.append(group_output)
    return output
