# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from crs_support import CRSDefinition, crs_equivalent, crs_from_meta
from gis_matrix import reproject_boundaries, reproject_polylines


Layer = Dict[str, object]


def source_crs_for_layer(layer: Layer) -> CRSDefinition | None:
    stored = layer.get("source_crs")
    if isinstance(stored, CRSDefinition):
        return stored
    meta = layer.get("source_meta") or layer.get("meta")
    if isinstance(meta, dict):
        return crs_from_meta(meta)
    return None


def choose_layer_project_crs(layers: Iterable[Layer]) -> CRSDefinition | None:
    normalized = list(layers)
    priorities = (
        lambda layer, crs: layer.get("kind") == "polyline" and crs.is_projected,
        lambda _layer, crs: crs.is_projected,
        lambda layer, _crs: layer.get("kind") == "polyline",
        lambda _layer, _crs: True,
    )
    available = [(layer, source_crs_for_layer(layer)) for layer in normalized]
    for predicate in priorities:
        for layer, definition in available:
            if definition is not None and predicate(layer, definition):
                return definition
    return None


def _working_meta(
    source_meta: Dict[str, object],
    source_crs: CRSDefinition | None,
    project_crs: CRSDefinition | None,
    bbox: Tuple[float, float, float, float],
    *,
    reprojected: bool,
) -> Dict[str, object]:
    meta = dict(source_meta)
    meta["source_bbox"] = list(source_meta.get("bbox", bbox))
    meta["bbox"] = [float(value) for value in bbox]
    meta["source_crs_label"] = source_crs.label if source_crs is not None else ""
    meta["source_crs_wkt"] = source_crs.wkt if source_crs is not None else ""
    meta["project_crs_label"] = project_crs.label if project_crs is not None else ""
    meta["project_crs_wkt"] = project_crs.wkt if project_crs is not None else ""
    meta["reprojected"] = bool(reprojected)
    if project_crs is not None:
        meta.update(project_crs.to_meta())
    return meta


def prepare_layer_for_project(layer: Layer, project_crs: CRSDefinition | None) -> Layer:
    prepared = dict(layer)
    source_geometry = prepared.get("source_geometry", prepared.get("geometry"))
    source_bbox_value = prepared.get("source_bbox", prepared.get("bbox"))
    source_meta_value = prepared.get("source_meta", prepared.get("meta", {}))
    if source_geometry is None or source_bbox_value is None:
        raise ValueError("Layer does not contain source geometry and bounds.")
    source_bbox = tuple(float(value) for value in source_bbox_value)
    if len(source_bbox) != 4:
        raise ValueError("Layer bounds must contain four coordinates.")
    source_meta = dict(source_meta_value) if isinstance(source_meta_value, dict) else {}
    source_crs = source_crs_for_layer(prepared)

    prepared["source_geometry"] = source_geometry
    prepared["source_bbox"] = source_bbox
    prepared["source_meta"] = source_meta
    prepared["source_crs"] = source_crs
    prepared.pop("spatial_index", None)
    prepared.pop("feature_index", None)
    prepared.pop("point_count", None)

    if project_crs is None:
        prepared["geometry"] = source_geometry
        prepared["bbox"] = source_bbox
        prepared["meta"] = _working_meta(
            source_meta,
            source_crs,
            source_crs,
            source_bbox,
            reprojected=False,
        )
        prepared["working_crs"] = source_crs
        prepared["crs_ready"] = True
        prepared["crs_status"] = (
            "native"
            if source_crs is not None
            else ("invalid" if source_meta.get("crs_error") else "unreferenced")
        )
        return prepared

    if source_crs is None:
        prepared["geometry"] = source_geometry
        prepared["bbox"] = source_bbox
        prepared["meta"] = _working_meta(
            source_meta,
            None,
            project_crs,
            source_bbox,
            reprojected=False,
        )
        prepared["working_crs"] = None
        prepared["crs_ready"] = False
        prepared["crs_status"] = "invalid" if source_meta.get("crs_error") else "unassigned"
        return prepared

    if crs_equivalent(source_crs, project_crs):
        working_geometry = source_geometry
        working_bbox = source_bbox
        was_reprojected = False
    elif prepared.get("kind") == "polyline":
        working_geometry, working_bbox = reproject_polylines(source_geometry, source_crs, project_crs)
        was_reprojected = True
    elif prepared.get("kind") == "boundary":
        working_geometry, working_bbox = reproject_boundaries(source_geometry, source_crs, project_crs)
        was_reprojected = True
    else:
        raise ValueError(f"Unsupported GIS layer kind: {prepared.get('kind')}")

    prepared["geometry"] = working_geometry
    prepared["bbox"] = tuple(float(value) for value in working_bbox)
    prepared["meta"] = _working_meta(
        source_meta,
        source_crs,
        project_crs,
        prepared["bbox"],
        reprojected=was_reprojected,
    )
    prepared["working_crs"] = project_crs
    prepared["crs_ready"] = True
    prepared["crs_status"] = "reprojected" if was_reprojected else "native"
    return prepared


def assign_layer_source_crs(
    layer: Layer,
    source_crs: CRSDefinition,
    project_crs: CRSDefinition | None,
) -> Layer:
    assigned = dict(layer)
    source_meta_value = assigned.get("source_meta", assigned.get("meta", {}))
    source_meta = dict(source_meta_value) if isinstance(source_meta_value, dict) else {}
    source_meta.update(source_crs.to_meta())
    source_meta["crs_source"] = "manual"
    source_meta["crs_error"] = ""
    assigned["source_meta"] = source_meta
    assigned["source_crs"] = source_crs
    return prepare_layer_for_project(assigned, project_crs)


def layer_crs_display(layer: Layer) -> str:
    source = source_crs_for_layer(layer)
    status = str(layer.get("crs_status", "") or "")
    working = layer.get("working_crs")
    if status == "invalid":
        return "파일 정보 오류"
    if source is None:
        return "선택 필요"
    if status == "reprojected" and isinstance(working, CRSDefinition):
        return f"{source.label} -> {working.label}"
    return source.label
