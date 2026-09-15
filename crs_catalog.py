# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Dict, Iterable, List, Sequence, Tuple

from crs_support import CRSDefinition, parse_crs, transform_nested_points


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class CRSChoice:
    key: str
    title: str
    detail: str
    definition: CRSDefinition | None


@dataclass(frozen=True)
class CRSRecommendation:
    key: str
    confidence: str
    reason: str
    score: float
    alternatives: Tuple[str, ...] = ()


_STANDARD_SPECS = (
    (
        "epsg5179",
        "국가 통합 좌표 (최근 전국 공공 GIS)",
        "전국 단위의 비교적 최근 공공자료에서 자주 사용, 서울 숫자 예: X 약 956,000 / Y 약 1,945,000",
        "EPSG:5179",
    ),
    (
        "epsg5174",
        "서울 구형 좌표 (수정 중부원점)",
        "서울의 구형 GIS 자료에서 자주 사용, 숫자 예: X 약 200,000 / Y 약 444,000",
        "EPSG:5174",
    ),
    (
        "epsg2097",
        "구형 중부원점 좌표",
        "한국측지계 1985 중부원점, 서울 숫자 예: X 약 200,000 / Y 약 444,000",
        "EPSG:2097",
    ),
    (
        "epsg5178",
        "구형 국가 통합 좌표",
        "한국측지계 1985 통합원점, 서울 숫자 예: X 약 956,000 / Y 약 1,944,000",
        "EPSG:5178",
    ),
    (
        "epsg5181",
        "중부원점 좌표 (한국측지계 2000)",
        "서울·중부권에서 사용, 서울 숫자 예: X 약 200,000 / Y 약 445,000",
        "EPSG:5181",
    ),
    (
        "epsg5186",
        "중부원점 좌표 (2010 보정)",
        "2010년 보정 축을 사용하는 서울·중부권 좌표, 숫자 예: X 약 200,000 / Y 약 545,000",
        "EPSG:5186",
    ),
    (
        "epsg4326",
        "위도·경도 좌표 (GPS 형식)",
        "경도 약 127, 위도 약 37처럼 표시되는 세계 위도·경도 좌표",
        "EPSG:4326",
    ),
)


@lru_cache(maxsize=1)
def standard_crs_choices() -> Tuple[CRSChoice, ...]:
    return tuple(
        CRSChoice(
            key=key,
            title=title,
            detail=f"{detail} | {epsg}",
            definition=parse_crs(epsg),
        )
        for key, title, detail, epsg in _STANDARD_SPECS
    )


def build_layer_crs_choices(
    detected_crs: CRSDefinition | None,
    project_crs: CRSDefinition | None,
    assigned_crs: CRSDefinition | None = None,
) -> List[CRSChoice]:
    choices: List[CRSChoice] = []
    if detected_crs is not None:
        choices.append(
            CRSChoice(
                key="detected",
                title="파일에서 확인된 좌표 사용",
                detail=f"함께 제공된 PRJ 파일의 설정을 그대로 사용 | {detected_crs.label}",
                definition=detected_crs,
            )
        )
    if detected_crs is None and assigned_crs is not None:
        choices.append(
            CRSChoice(
                key="assigned",
                title="이 레이어의 현재 선택 유지",
                detail=f"이전에 사용자가 선택한 좌표를 그대로 사용 | {assigned_crs.label}",
                definition=assigned_crs,
            )
        )
    if project_crs is not None:
        choices.append(
            CRSChoice(
                key="project",
                title="현재 지도와 같은 좌표 사용",
                detail=f"이 레이어도 현재 전체 지도와 같은 좌표라고 해석 | {project_crs.label}",
                definition=project_crs,
            )
        )
    choices.extend(standard_crs_choices())
    choices.append(
        CRSChoice(
            key="advanced",
            title="목록에 없음 (전문가 직접 입력)",
            detail="자료 제공기관이 알려 준 EPSG 코드 또는 WKT를 직접 입력",
            definition=None,
        )
    )
    return choices


def build_project_crs_choices(
    current_crs: CRSDefinition | None,
    pipe_crs: CRSDefinition | None,
) -> List[CRSChoice]:
    choices: List[CRSChoice] = []
    if pipe_crs is not None:
        choices.append(
            CRSChoice(
                key="pipe",
                title="관로 레이어 좌표에 맞춤",
                detail=f"관로의 좌표를 작업 기준으로 사용 | {pipe_crs.label}",
                definition=pipe_crs,
            )
        )
    if current_crs is not None:
        choices.append(
            CRSChoice(
                key="current",
                title="현재 전체 지도 좌표 유지",
                detail=f"현재 설정을 변경하지 않음 | {current_crs.label}",
                definition=current_crs,
            )
        )
    choices.extend(standard_crs_choices())
    choices.append(
        CRSChoice(
            key="advanced",
            title="목록에 없음 (전문가 직접 입력)",
            detail="자료 제공기관이 알려 준 EPSG 코드 또는 WKT를 직접 입력",
            definition=None,
        )
    )
    return choices


def recommend_project_crs(
    choices: Sequence[CRSChoice],
    current_crs: CRSDefinition | None,
    pipe_crs: CRSDefinition | None,
) -> CRSRecommendation | None:
    keys = {choice.key for choice in choices}
    if pipe_crs is not None and pipe_crs.is_projected and "pipe" in keys:
        return CRSRecommendation(
            key="pipe",
            confidence="high",
            reason="관로를 작업 기준으로 두면 관로 자체를 다시 변환하지 않아 거리와 방향 해석이 가장 직접적입니다.",
            score=1.0,
        )
    if current_crs is not None and current_crs.is_projected and "current" in keys:
        return CRSRecommendation(
            key="current",
            confidence="high",
            reason="현재 지도가 이미 미터 단위 평면좌표를 사용하고 있어 기존 중첩 상태를 유지할 수 있습니다.",
            score=0.96,
        )
    if current_crs is not None and "current" in keys:
        return CRSRecommendation(
            key="current",
            confidence="medium",
            reason="현재 표시 상태를 유지하는 선택입니다. 위도·경도 좌표라면 격자 셀 크기를 미터로 직접 해석할 수 없습니다.",
            score=0.70,
        )
    if pipe_crs is not None and "pipe" in keys:
        return CRSRecommendation(
            key="pipe",
            confidence="medium",
            reason="관로의 좌표를 그대로 유지하는 선택입니다. 위도·경도 좌표라면 격자 셀 크기를 미터로 직접 해석할 수 없습니다.",
            score=0.68,
        )
    if "epsg5179" in keys:
        return CRSRecommendation(
            key="epsg5179",
            confidence="low",
            reason="확인할 레이어가 없어 최근 국내 공공 GIS의 일반 후보를 제시합니다. 실제 자료의 좌표 정보가 우선입니다.",
            score=0.35,
        )
    return None


def _bbox_area(bbox: BBox) -> float:
    return max(float(bbox[2]) - float(bbox[0]), 0.0) * max(float(bbox[3]) - float(bbox[1]), 0.0)


def _bbox_is_valid(bbox: BBox) -> bool:
    try:
        xmin, ymin, xmax, ymax = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return False
    return (
        all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax))
        and xmin <= xmax
        and ymin <= ymax
        and (xmin < xmax or ymin < ymax)
    )


def _bbox_overlap_ratio(first: BBox, second: BBox) -> float:
    intersection = (
        max(min(float(first[2]), float(second[2])) - max(float(first[0]), float(second[0])), 0.0)
        * max(min(float(first[3]), float(second[3])) - max(float(first[1]), float(second[1])), 0.0)
    )
    denominator = min(_bbox_area(first), _bbox_area(second))
    if denominator <= 1e-12:
        return 0.0
    return float(max(0.0, min(intersection / denominator, 1.0)))


def _bbox_center_closeness(first: BBox, second: BBox) -> float:
    first_center = ((float(first[0]) + float(first[2])) * 0.5, (float(first[1]) + float(first[3])) * 0.5)
    second_center = ((float(second[0]) + float(second[2])) * 0.5, (float(second[1]) + float(second[3])) * 0.5)
    distance = math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])
    diagonal = max(
        math.hypot(float(second[2]) - float(second[0]), float(second[3]) - float(second[1])),
        1e-9,
    )
    return float(max(0.0, 1.0 - distance / (diagonal * 2.0)))


def transform_bbox(
    bbox: BBox,
    source_crs: CRSDefinition,
    target_crs: CRSDefinition,
) -> BBox:
    if not _bbox_is_valid(bbox):
        raise ValueError("좌표 범위가 올바르지 않습니다.")
    xmin, ymin, xmax, ymax = (float(value) for value in bbox)
    xmid = (xmin + xmax) * 0.5
    ymid = (ymin + ymax) * 0.5
    samples = [(x, y) for x in (xmin, xmid, xmax) for y in (ymin, ymid, ymax)]
    transformed = transform_nested_points([[samples]], source_crs, target_crs)[0][0]
    xs = [point[0] for point in transformed]
    ys = [point[1] for point in transformed]
    return min(xs), min(ys), max(xs), max(ys)


def _source_year(layer_name: str) -> int | None:
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", str(layer_name))
    return int(match.group(1)) if match else None


def _coordinate_signature_scores(bbox: BBox) -> Dict[str, float]:
    xmin, ymin, xmax, ymax = (float(value) for value in bbox)
    scores: Dict[str, float] = {}
    if -180.0 <= xmin <= 180.0 and -180.0 <= xmax <= 180.0 and -90.0 <= ymin <= 90.0 and -90.0 <= ymax <= 90.0:
        scores["epsg4326"] = 0.80
    if 500_000.0 <= xmin <= xmax <= 1_500_000.0 and 1_000_000.0 <= ymin <= ymax <= 3_000_000.0:
        scores["epsg5179"] = 0.58
        scores["epsg5178"] = 0.52
    if 50_000.0 <= xmin <= xmax <= 500_000.0 and 200_000.0 <= ymin <= ymax <= 800_000.0:
        scores.update(
            {
                "epsg5174": 0.42,
                "epsg2097": 0.38,
                "epsg5181": 0.34,
                "epsg5186": 0.30,
            }
        )
    return scores


def recommend_layer_crs(
    choices: Sequence[CRSChoice],
    source_bbox: BBox,
    *,
    layer_name: str = "",
    detected_crs: CRSDefinition | None = None,
    assigned_crs: CRSDefinition | None = None,
    project_crs: CRSDefinition | None = None,
    project_bbox: BBox | None = None,
) -> CRSRecommendation | None:
    if detected_crs is not None and any(choice.key == "detected" for choice in choices):
        return CRSRecommendation(
            key="detected",
            confidence="confirmed",
            reason="함께 제공된 PRJ 파일에서 좌표 정보가 확인되었습니다.",
            score=1.0,
        )
    if assigned_crs is not None and any(choice.key == "assigned" for choice in choices):
        return CRSRecommendation(
            key="assigned",
            confidence="high",
            reason="이전에 사용자가 이 레이어에 확정한 좌표 설정입니다.",
            score=0.98,
        )
    if not _bbox_is_valid(source_bbox):
        return None

    scores = _coordinate_signature_scores(source_bbox)
    overlap_by_key: Dict[str, float] = {}
    if project_crs is not None and project_bbox is not None and _bbox_is_valid(project_bbox):
        for choice in choices:
            if choice.definition is None or choice.key in {"advanced", "detected"}:
                continue
            try:
                transformed_bbox = transform_bbox(source_bbox, choice.definition, project_crs)
            except Exception:
                continue
            overlap = _bbox_overlap_ratio(transformed_bbox, project_bbox)
            closeness = _bbox_center_closeness(transformed_bbox, project_bbox)
            overlap_by_key[choice.key] = overlap
            spatial_score = overlap * 0.72 + closeness * 0.18
            scores[choice.key] = max(scores.get(choice.key, 0.0), spatial_score)

    year = _source_year(layer_name)
    if year is not None and year >= 2002:
        for key in ("epsg5179", "epsg5181", "epsg5186"):
            if key in scores:
                scores[key] = min(scores[key] + 0.04, 1.0)
    elif year is not None:
        for key in ("epsg5178", "epsg5174", "epsg2097"):
            if key in scores:
                scores[key] = min(scores[key] + 0.04, 1.0)

    valid_keys = {choice.key for choice in choices if choice.definition is not None}
    priority = {"project": 0, "epsg5179": 1, "epsg5174": 2, "epsg2097": 3, "epsg5178": 4}
    ranked = sorted(
        ((key, min(float(score), 1.0)) for key, score in scores.items() if key in valid_keys),
        key=lambda item: (-item[1], priority.get(item[0], 20), item[0]),
    )
    if not ranked or ranked[0][1] < 0.25:
        return None

    best_key, best_score = ranked[0]
    margin = best_score - (ranked[1][1] if len(ranked) > 1 else 0.0)
    confidence = "medium" if best_score >= 0.62 and margin >= 0.05 else "low"

    reasons: List[str] = []
    if best_key == "epsg4326":
        reasons.append("숫자 범위가 경도·위도 형식과 맞습니다")
    elif best_key in {"epsg5179", "epsg5178"}:
        reasons.append("숫자 범위가 국가 통합원점 형식과 맞습니다")
    elif best_key in {"epsg5174", "epsg2097", "epsg5181", "epsg5186", "project"}:
        reasons.append("숫자 범위가 서울·중부권 평면좌표 형식과 맞습니다")
    if overlap_by_key.get(best_key, 0.0) > 0.0:
        reasons.append("이 후보로 바꾼 범위가 현재 지도와 겹칩니다")
    if year is not None:
        reasons.append(f"파일명에서 {year}년을 확인했습니다")
    if not reasons:
        reasons.append("좌표 숫자 범위와 현재 지도의 위치를 비교했습니다")

    alternatives = tuple(key for key, score in ranked[1:4] if best_score - score < 0.12)
    return CRSRecommendation(
        key=best_key,
        confidence=confidence,
        reason="; ".join(reasons) + ". PRJ가 없으므로 자료 제공기관의 좌표 정보가 이 추천보다 우선합니다.",
        score=best_score,
        alternatives=alternatives,
    )


def find_choice(choices: Iterable[CRSChoice], key: str) -> CRSChoice | None:
    return next((choice for choice in choices if choice.key == str(key)), None)
