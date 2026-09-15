# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Uses OpenCV: https://github.com/opencv/opencv (Apache-2.0); opencv-python wheel notices apply.
# Uses NumPy: https://github.com/numpy/numpy (BSD-3-Clause plus bundled notices).
# Uses Pillow: https://github.com/python-pillow/Pillow (MIT-CMU).
# Uses Tkinter / Tcl-Tk: https://docs.python.org/3/library/tkinter.html (runtime licenses).
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import shutil
import sys
import time
from collections import deque
from itertools import combinations

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import colorchooser, filedialog, messagebox, scrolledtext, simpledialog, ttk

from workspace_actions import WorkspaceActions, guarded_action
from runtime_paths import user_data_root
from workspace_state import atomic_write_text, load_settings_object, validated_direction_matrix, resample_mask_centers
from gui_layout import build_workspace_ui, apply_workspace_style, refresh_workspace_layout

from cell_training import (
    CONTEXT_RADIUS,
    append_to_dataset,
    extract_features_and_labels,
    get_torch_runtime_info,
    predict_with_model_with_meta,
    reset_training_state,
    train_cell_model,
)
from crs_catalog import (
    CRSChoice,
    CRSRecommendation,
    build_layer_crs_choices,
    build_project_crs_choices,
    find_choice,
    recommend_layer_crs,
    recommend_project_crs,
)
from crs_support import CRSDefinition, CRSDependencyError, crs_equivalent, parse_crs
from gis_matrix import (
    BoundaryFeatureIndex,
    DEFAULT_END_HEIGHT_FIELD,
    DEFAULT_START_HEIGHT_FIELD,
    PolylineSpatialIndex,
    boundary_grid_mask,
    clip_polylines_to_boundary,
    combine_bboxes,
    load_gis_matrix,
    matrix_from_loaded_polylines,
    prepare_rotated_polylines,
    read_boundary_shp,
    read_polyline_shp,
    read_shp_shape_info,
    render_gis_layers,
)
from gis_project import (
    assign_layer_source_crs,
    choose_layer_project_crs,
    layer_crs_display,
    prepare_layer_for_project,
    source_crs_for_layer,
)
from gold_labels import (
    compact_metric_summary,
    evaluate_prediction,
    find_matching_gold_labels,
    format_evaluation_report,
    save_gold_label,
)
from gamma_index import (
    GammaResult,
    compute_gamma_index,
    load_plena_input_matrix,
    save_gamma_outputs,
)
from matrix_arrows import render_direction_matrix_arrows
from plena_plots import (
    infer_nse_path_from_width_csv,
    load_nse_points,
    load_width_function_blocks,
    nse_points_from_summary_rows,
    render_gamma_plot,
    render_nse_plot,
    render_width_function_plot,
)
from solution import (
    assist_direction_matrix,
    compress_direction_matrix,
    compute_direction_matrix_with_meta,
    derive_grid_B,
    direction_cells_reaching_outlets,
    fill_basin_to_network,
    outer_zero_margin_bounds,
    recommend_outlet_candidates,
)


DIR_KR = {
    0: "\ube48\uce78",
    1: "\ub3d9",
    2: "\ub0a8",
    3: "\uc11c",
    4: "\ubd81",
}

DIR_COLORS = {
    0: "#bac4cf",
    1: "#e85d5d",
    2: "#3da56d",
    3: "#3f7de8",
    4: "#af52d9",
}
MATRIX_NUMBER_COLOR = "#ff0000"
MATRIX_NUMBER_HALO = "#ffffff"
GIS_OVERVIEW_RENDER_SIZE = 4096
GIS_RENDER_CACHE_LIMIT = 2
GIS_BACKGROUND_RENDER_POINT_THRESHOLD = 25_000
GIS_BOUNDARY_RESULT_CACHE_LIMIT = 4
GIS_PIPE_RESULT_CACHE_LIMIT = 2
ZOOM_MIN = 0.125
ZOOM_MAX = 96.0
GIS_VIEWPORT_RENDER_ZOOM_THRESHOLD = 2.0
GIS_VIEWPORT_RENDER_CACHE_LIMIT = 6
GIS_VIEWPORT_RENDER_SUPERSAMPLE = 1.35
GIS_INTERACTIVE_VIEWPORT_RENDER_MAX = 1536
GIS_INTERACTIVE_OVERVIEW_DEBOUNCE_MS = 650

TXT_TITLE = "FMT - Flow Matrix Tool"
TXT_LOAD = "이미지"
TXT_LOAD_GIS = "GIS 불러오기"
TXT_GIS_SYNTH_TRAIN = "GIS 학습"
TXT_EMPTY = "빈 행렬"
TXT_DETECT = "자동인식"
TXT_APPLY_MODEL = "모델적용"
TXT_RESET_MODEL = "\ubaa8\ub378 \ucd08\uae30\ud654"
TXT_FIT_VIEW = "맞춤"
TXT_CLEAR = "초기화"
TXT_SAVE = "저장"
TXT_TRAIN = "학습"
TXT_EXPORT = "내보내기"
TXT_MATRIX_ARROWS = "행렬/화살표"
TXT_HELP = "도움말"
TXT_ROWS = "행 A"
TXT_ROTATE = "회전°"
TXT_ROTATE_APPLY = "적용"
TXT_ROTATE_RESET = "0"
TXT_LASSO = "LASSO"
TXT_APPLY_LASSO = "LASSO 적용"
TXT_SELECT_OUTLET = "출구"
TXT_BFS_ASSIST = "BFS 연결"
TXT_COMPRESS = "압축"
TXT_TRIM = "외곽 0 제거"
TXT_PLENA = "PLENA"
TXT_PLENA_PLOT = "PLENA 그림"
TXT_INFO_TAB = "\uc791\uc5c5"
TXT_SETTINGS_TAB = "\uc124\uc815"
TXT_NO_SELECTION = "\uc120\ud0dd \uc140: \uc5c6\uc74c"
TXT_NO_MATRIX = "\ud589\ub82c\uc774 \uc544\uc9c1 \uc5c6\uc2b5\ub2c8\ub2e4."
TXT_START = "이미지 또는 GIS 레이어를 불러와 NFMAT 작업을 시작하세요."
TXT_RUNTIME_TRAINING_DISABLED = "\ub7f0\ud0c0\uc784 \ud559\uc2b5 \ube44\ud65c\uc131\ud654"
TXT_RULES = (
    "\uc785\ub825 \uaddc\uce59\n"
    "- \ub9c8\uc6b0\uc2a4 \ud074\ub9ad: \uc140 \uc120\ud0dd\n"
    "- \ub9c8\uc6b0\uc2a4 \ud720: \ud655\ub300/\ucd95\uc18c\n"
    "- \uc6b0\ud074\ub9ad \ub4dc\ub798\uadf8: \ud654\uba74 \uc774\ub3d9\n"
    "- \uc22b\uc790 0/1/2/3/4: \uac12 \uc785\ub825\n"
    "- \ubc29\ud5a5\ud0a4: \uc120\ud0dd \uc140 \uc774\ub3d9\n"
    "- Delete/BackSpace: 0 \uc785\ub825\n"
    "- 1=\ub3d9, 2=\ub0a8, 3=\uc11c, 4=\ubd81"
)
TXT_HELP_BODY = (
    "1. [\uc774\ubbf8\uc9c0 \ubd88\ub7ec\uc624\uae30]\ub85c \ub3c4\uba74 \uc774\ubbf8\uc9c0\ub97c \uc5fd\ub2c8\ub2e4.\n"
    "2. \ud589 \uac1c\uc218 A\ub97c \uc785\ub825\ud558\uba74 \uc5f4 \uac1c\uc218 B\ub294 \uc774\ubbf8\uc9c0 \ube44\uc728\ub85c \uc790\ub3d9 \uacc4\uc0b0\ub429\ub2c8\ub2e4.\n"
    "3. [\uc790\ub3d9 \uc778\uc2dd \uc2e4\ud589]\uc73c\ub85c \ucd08\uae30 \ud589\ub82c\uc744 \uc5bb\uac70\ub098, [\ube48 \ud589\ub82c \uc2dc\uc791]\uc73c\ub85c \ucc98\uc74c\ubd80\ud130 \uc785\ub825\ud569\ub2c8\ub2e4.\n"
    "4. \uce94\ubc84\uc2a4\uc5d0\uc11c \uc140\uc744 \ud55c \ubc88 \ud074\ub9ad\ud55c \ub4a4 \ud0a4\ubcf4\ub4dc 0,1,2,3,4\ub85c \uc9c1\uc811 \uc785\ub825\ud569\ub2c8\ub2e4.\n"
    "5. 0=\ube48\uce78, 1=\ub3d9, 2=\ub0a8, 3=\uc11c, 4=\ubd81\uc785\ub2c8\ub2e4.\n"
    "6. \ubc29\ud5a5\ud0a4\ub85c \uc120\ud0dd \uc140\uc744 \uc774\ub3d9\ud560 \uc218 \uc788\uc2b5\ub2c8\ub2e4. \ub9c8\uc6b0\uc2a4 \ud720\ub85c \ud655\ub300/\ucd95\uc18c, \uc6b0\ud074\ub9ad \ub4dc\ub798\uadf8\ub85c \ud654\uba74 \uc774\ub3d9\uc774 \uac00\ub2a5\ud569\ub2c8\ub2e4.\n"
    "7. [\uc218\uc815 \uc800\uc7a5] \ud6c4 [\ubaa8\ub378 \ud559\uc2b5], [\ud559\uc2b5 \ubaa8\ub378 \uc801\uc6a9] \uc21c\uc73c\ub85c \uc2e4\ud589\ud558\uba74 \ubcf4\uc815 \ubaa8\ub378\uc744 \uacc4\uc18d \uac31\uc2e0\ud558\uba70 \uc4f8 \uc218 \uc788\uc2b5\ub2c8\ub2e4.\n"
    "8. [\ucd9c\uad6c \uc120\ud0dd] \ud6c4 [BFS \ubcf4\uc870 \uc5f0\uacb0]\uc744 \uc2e4\ud589\ud558\uba74, \uae30\uc874 \ud654\uc0b4\ud45c \ubc29\ud5a5\uc740 \uac70\uc758 \uac74\ub4dc\ub9ac\uc9c0 \uc54a\uace0 \ube48\uce78 \ub610\ub294 \ub04a\uae34 \uce78\uc744 outlet \ubc29\ud5d8\uc73c\ub85c \ubcf4\uc870\ud569\ub2c8\ub2e4.\n"
    "9. \uc790\ub3d9 \uc778\uc2dd\uc740 \ud654\uc0b4\ud45c \uba38\ub9ac\uc640 \uc120\ubd84\uc774 \uac19\uc740 \uc5f0\uacb0 \uac1d\uccb4\uc77c \ub54c \uadf8 \uacbd\ub85c \uc804\uccb4\ub97c \ub530\ub77c \ubc29\ud5a5\uc744 \uacc4\uc0b0\ud569\ub2c8\ub2e4.\n"
    "10. [PLENA \uc2e4\ud589] \ubc84\ud2bc\uc73c\ub85c NFMAT \uacb0\uacfc\ub97c PLENA \uc785\ub825 txt\ub85c \uc800\uc7a5\ud558\uace0, \ubca0\ud0c0 \uac12\uc744 \uc785\ub825\ud574 EXE\ub97c \ubc14\ub85c \uc2e4\ud589\ud560 \uc218 \uc788\uc2b5\ub2c8\ub2e4.\n"
    "11. [LASSO \uc120\ud0dd] \ud6c4 [LASSO \uc801\uc6a9]\uc744 \ub204\ub974\uba74 \uc120\ud0dd \uc140 \ubc16\uc758 \ud589\ub82c\uacfc PLENA \uc785\ub825\uc740 \ud56d\uc0c1 0\uc73c\ub85c \uc720\uc9c0\ub429\ub2c8\ub2e4. \uc790\ub3d9 \uc778\uc2dd/\ubaa8\ub378/BFS\ub3c4 \uc120\ud0dd \uad6c\uc5ed \uc548\uc5d0\ub9cc \uc801\uc6a9\ub429\ub2c8\ub2e4.\n"
    "12. [\ud589\ub82c \uc555\ucd95] \uc740 \ud604\uc7ac \ud589\ub82c \uacb0\uacfc\uac00 \ub098\uc628 \ub4a4\uc5d0\ub9cc \uc0ac\uc6a9\ud558\ub294 \ud6c4\ucc98\ub9ac \uae30\ub2a5\uc785\ub2c8\ub2e4. \ud589 \uac1c\uc218(A)\ub9cc \uc785\ub825\ud558\uba74 \uc5f4 \uac1c\uc218(B)\ub294 \ube44\ub840\ud558\uc5ec \uc790\ub3d9 \uacc4\uc0b0\ub429\ub2c8\ub2e4.\n"
    "13. \uc555\ucd95 \ud589\ub82c\uc740 \uc6d0\ubcf8 \uadf8\ub9ac\ub4dc\uc640 \ub2e4\ub974\ubbc0\ub85c [\uc218\uc815 \uc800\uc7a5]/[\ubaa8\ub378 \ud559\uc2b5]\uc744 \ube44\ud65c\uc131\ud654\ud569\ub2c8\ub2e4. \uc6d0\ubcf8 \uae30\uc900\uc73c\ub85c \ub2e4\uc2dc \uc790\ub3d9 \uc778\uc2dd \ub610\ub294 \ubaa8\ub378 \uc801\uc6a9\uc744 \ud55c \ub4a4 \ud559\uc2b5\ud574 \uc8fc\uc138\uc694.\n"
    "14. [GIS 불러오기]는 여러 관로·경계 SHP를 백그라운드에서 불러와 왼쪽 GIS 레이어 목록과 중앙 합성 지도에 표시합니다. 레이어 순서, 관로 선 색상·투명도·굵기, 경계 채움·테두리·굵기, 경계 이름 DBF 필드를 설정할 수 있습니다.\n"
    "15. 관로 SHP만 있는 경우 [선택 관로 → 행렬]로 선택한 PolyLine 레이어 전체를 바로 행렬로 변환합니다. 행정경계가 있으면 GIS 지도에서 경계를 클릭해 해당 경계 내부 관로를 변환하고, [LASSO → 행렬]은 직접 그린 다각형 내부 관로를 변환합니다.\n"
    "16. 별도로 불러온 침수지역 Polygon GIS를 [침수 추가/제거]하면, 선택한 여러 침수 레이어와 관로가 겹치는 셀을 행렬과 화살표 화면에서 붉은 사각형으로 표시합니다.\n"
    "17. [권장 토출부]는 현재 방향 흐름이 실제로 끝나는 Outlet 후보를 도달 관로 셀 수 순으로 제시합니다. 다중 토출부 유지 또는 단일 토출부 사용을 선택할 수 있으며, PLENA 입력은 단일 Outlet만 지원합니다.\n"
    "18. [외곽 0 제거]는 관로와 Outlet 바깥의 연속된 빈 행·열만 1셀 여유를 남겨 제거합니다. 행렬 내부의 빈 행·열은 연결 거리 보존을 위해 제거하지 않습니다.\n"
    "19. GIS의 지도 좌표는 레이어별로 확인됩니다. PRJ가 있으면 파일 설정을 권장하고, PRJ가 없으면 좌표 범위와 현재 지도 위치를 근거로 후보를 제시합니다. 왼쪽 [선택 레이어 좌표]에서 한국어 항목을 골라 확정하세요. 목록에 없는 경우에만 전문가 입력을 사용합니다."
)

RUNTIME_TRAINING_ENABLED = True
STRICT_DEPLOYMENT_MODEL = False

DEVICE_CHOICES = {
    "\uc790\ub3d9": "auto",
    "CPU": "cpu",
    "CUDA": "cuda",
}

_LEGACY_THEME_ALIASES_GARBLED_OLD = {
    "NFMAT White": "NFMAT White",
    "NFMAT Background": "NFMAT Background",
    "NFMAT ?쒕갚": "NFMAT White",
    "NFMAT \uc21c\ubc31": "NFMAT White",
    "NFMAT \uc624\uc158": "NFMAT Background",
    "NFMAT \ube14\ub8e8\ud504\ub9b0\ud2b8": "NFMAT Background",
}

LEGACY_DEVICE_ALIASES = {
    "\uc790\ub3d9": "\uc790\ub3d9",
    "?먮룞": "\uc790\ub3d9",
    "auto": "\uc790\ub3d9",
    "AUTO": "\uc790\ub3d9",
    "CPU": "CPU",
    "cpu": "CPU",
    "CUDA": "CUDA",
    "cuda": "CUDA",
}

LEGACY_THEME_ALIASES = {
    "NFMAT Glass": "NFMAT Glass",
    "NFMAT Liquid Glass": "NFMAT Glass",
    "NFMAT \uc720\ub9ac": "NFMAT Glass",
    "NFMAT White": "NFMAT White",
    "NFMAT White Theme": "NFMAT White",
    "NFMAT \uc21c\ubc31": "NFMAT White",
    "NFMAT \ubc31\uc0c9": "NFMAT White",
    "NFMAT Background": "NFMAT Background",
    "NFMAT Background Theme": "NFMAT Background",
    "NFMAT \ubc30\uacbd": "NFMAT Background",
}

_LEGACY_THEME_PRESETS_UNUSED: Dict[str, Dict[str, str]] = {
    "NFMAT 순백": {
        "ttk_theme": "clam",
        "bg": "#ffffff",
        "panel": "#ffffff",
        "accent": "#2e6bd8",
        "accent_alt": "#5b8def",
        "text": "#1b2430",
        "muted": "#6a7380",
        "canvas": "#ffffff",
        "brand_bg": "#ffffff",
        "theme_image": "",
    },
    "NFMAT \uc624\uc158": {
        "ttk_theme": "clam",
        "bg": "#eef3f6",
        "panel": "#dce8ef",
        "accent": "#1d6f8f",
        "accent_alt": "#4a8c5f",
        "text": "#173043",
        "muted": "#577385",
        "canvas": "#102533",
        "brand_bg": "#18394d",
        "theme_image": "themes/nfmat_ocean_theme.png",
    },
    "NFMAT \ube14\ub8e8\ud504\ub9b0\ud2b8": {
        "ttk_theme": "clam",
        "bg": "#eaf0f5",
        "panel": "#d8e4ef",
        "accent": "#345d85",
        "accent_alt": "#6b90b9",
        "text": "#1b2d42",
        "muted": "#61778f",
        "canvas": "#13293d",
        "brand_bg": "#1d3650",
        "theme_image": "themes/nfmat_blueprint_theme.png",
    },
}


THEME_PRESETS = {
    "NFMAT Glass": {
        "ttk_theme": "clam",
        "bg": "#ffffff",
        "panel": "#ffffff",
        "panel_alt": "#f7f7f8",
        "accent": "#20242a",
        "accent_alt": "#e8eaed",
        "text": "#1f2328",
        "muted": "#66707a",
        "canvas": "#ffffff",
        "brand_bg": "#ffffff",
        "entry_bg": "#ffffff",
        "border": "#d7dbe0",
        "shadow": "#eceff3",
        "matrix_bg": "#ffffff",
        "theme_image": "",
        "glass": "1",
    },
    "NFMAT White": {
        "ttk_theme": "clam",
        "bg": "#ffffff",
        "panel": "#ffffff",
        "panel_alt": "#f7f7f8",
        "accent": "#20242a",
        "accent_alt": "#4b5563",
        "text": "#1f2328",
        "muted": "#66707a",
        "canvas": "#ffffff",
        "brand_bg": "#ffffff",
        "entry_bg": "#ffffff",
        "border": "#d7dbe0",
        "shadow": "#eceff3",
        "matrix_bg": "#ffffff",
        "theme_image": "",
    },
    "NFMAT Background": {
        "ttk_theme": "clam",
        "bg": "#f6f1ee",
        "panel": "#fbf7f4",
        "accent": "#5d7495",
        "accent_alt": "#d8b79f",
        "text": "#2e2a27",
        "muted": "#7e726b",
        "canvas": "#f3eee9",
        "brand_bg": "#fbf7f4",
        "theme_image": "background.png",
    },
}


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _resource_root() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return _app_root()


def _resource_roots() -> List[Path]:
    roots: List[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))
    roots.append(_app_root())
    if not getattr(sys, "frozen", False):
        roots.append(Path(__file__).resolve().parent)
    deduped: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve()).lower()
        except OSError:
            key = str(root.absolute()).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def _resource_path(relative_path: str | Path) -> Path:
    if str(relative_path) == "PLENA.exe" and os.environ.get("FMT_PLENA_EXECUTABLE"):
        return Path(os.environ["FMT_PLENA_EXECUTABLE"]).expanduser().resolve()
    path = Path(relative_path)
    if path.is_absolute():
        return path
    for root in _resource_roots():
        candidate = root / path
        if candidate.exists():
            return candidate
    return _resource_root() / path


def _resource_search_text(relative_path: str | Path) -> str:
    return ", ".join(str(root / relative_path) for root in _resource_roots())


def _writable_root() -> Path:
    return user_data_root()


class CRSChoiceDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        *,
        title: str,
        prompt: str,
        choices: Sequence[CRSChoice],
        recommendation: CRSRecommendation | None,
        expert_initial: str = "",
    ) -> None:
        super().__init__(parent)
        self.result: Optional[Tuple[CRSDefinition, str]] = None
        self._choices = list(choices)
        self._recommendation = recommendation
        self._display_to_choice: Dict[str, CRSChoice] = {}

        self.title(title)
        self.transient(parent)
        self.resizable(True, False)
        self.minsize(650, 430)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        frame = ttk.Frame(self, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text=prompt, wraplength=700, justify=tk.LEFT).grid(row=0, column=0, sticky=tk.EW)

        recommendation_frame = ttk.LabelFrame(frame, text="프로그램 확인 결과", padding=(10, 8))
        recommendation_frame.grid(row=1, column=0, sticky=tk.EW, pady=(12, 10))
        recommendation_frame.columnconfigure(0, weight=1)
        recommendation_text = self._recommendation_text()
        ttk.Label(
            recommendation_frame,
            text=recommendation_text,
            wraplength=680,
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky=tk.EW)

        ttk.Label(frame, text="이 자료의 좌표 형식을 선택하세요.").grid(row=2, column=0, sticky=tk.W)
        display_values: List[str] = []
        for choice in self._choices:
            prefix = "[추천] " if recommendation is not None and choice.key == recommendation.key else ""
            display = f"{prefix}{choice.title}"
            display_values.append(display)
            self._display_to_choice[display] = choice

        self.choice_var = tk.StringVar()
        self.choice_combo = ttk.Combobox(
            frame,
            textvariable=self.choice_var,
            values=display_values,
            state="readonly",
            height=min(max(len(display_values), 5), 12),
        )
        self.choice_combo.grid(row=3, column=0, sticky=tk.EW, pady=(4, 0))
        self.choice_combo.bind("<<ComboboxSelected>>", self._on_choice_changed)

        self.detail_var = tk.StringVar()
        ttk.Label(
            frame,
            textvariable=self.detail_var,
            wraplength=700,
            justify=tk.LEFT,
            foreground="#4b5563",
        ).grid(row=4, column=0, sticky=tk.EW, pady=(6, 8))

        expert_frame = ttk.LabelFrame(frame, text="전문가 입력", padding=(10, 7))
        expert_frame.grid(row=5, column=0, sticky=tk.EW)
        expert_frame.columnconfigure(0, weight=1)
        ttk.Label(
            expert_frame,
            text="목록에 없는 좌표만 자료 제공기관의 EPSG 코드 또는 WKT를 입력합니다.",
            wraplength=680,
        ).grid(row=0, column=0, sticky=tk.W)
        self.expert_var = tk.StringVar(value=expert_initial)
        self.expert_entry = ttk.Entry(expert_frame, textvariable=self.expert_var, state=tk.DISABLED)
        self.expert_entry.grid(row=1, column=0, sticky=tk.EW, pady=(5, 0))

        ttk.Label(
            frame,
            text="일반 사용자는 추천 항목을 확인하고 선택하면 됩니다. 이 설정은 원본 SHP와 PRJ 파일을 수정하지 않습니다.",
            wraplength=700,
            justify=tk.LEFT,
        ).grid(row=6, column=0, sticky=tk.EW, pady=(10, 0))

        buttons = ttk.Frame(frame)
        buttons.grid(row=7, column=0, sticky=tk.E, pady=(14, 0))
        ttk.Button(buttons, text="취소", command=self.destroy).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="선택 적용", command=self._apply).pack(side=tk.RIGHT, padx=(0, 6))

        initial_key = recommendation.key if recommendation is not None else (self._choices[0].key if self._choices else "")
        initial_display = next(
            (display for display, choice in self._display_to_choice.items() if choice.key == initial_key),
            display_values[0] if display_values else "",
        )
        self.choice_var.set(initial_display)
        self._on_choice_changed()

        self.bind("<Escape>", lambda _event: self.destroy())
        self.bind("<Return>", lambda _event: self._apply())
        self.update_idletasks()
        try:
            parent_x = parent.winfo_rootx()
            parent_y = parent.winfo_rooty()
            parent_width = parent.winfo_width()
            parent_height = parent.winfo_height()
            x_pos = parent_x + max((parent_width - self.winfo_width()) // 2, 0)
            y_pos = parent_y + max((parent_height - self.winfo_height()) // 2, 0)
            self.geometry(f"+{x_pos}+{y_pos}")
        except tk.TclError:
            pass
        self.grab_set()
        self.choice_combo.focus_set()

    def _recommendation_text(self) -> str:
        recommendation = self._recommendation
        if recommendation is None:
            return (
                "자동으로 좁힐 근거가 부족합니다. 자료 제공기관의 설명을 확인해 선택하세요. "
                "좌표를 임의로 확정하지 않았습니다."
            )
        choice = find_choice(self._choices, recommendation.key)
        title = choice.title if choice is not None else recommendation.key
        confidence_label = {
            "confirmed": "파일 정보 확인",
            "high": "권장",
            "medium": "추천",
            "low": "후보 제안 · 확인 필요",
        }.get(recommendation.confidence, "후보 제안")
        alternatives = [
            alternative.title
            for key in recommendation.alternatives
            for alternative in [find_choice(self._choices, key)]
            if alternative is not None
        ]
        alternative_text = f"\n비슷한 후보: {', '.join(alternatives)}" if alternatives else ""
        return f"{confidence_label}: {title}\n근거: {recommendation.reason}{alternative_text}"

    def _selected_choice(self) -> CRSChoice | None:
        return self._display_to_choice.get(self.choice_var.get())

    def _on_choice_changed(self, _event: Optional[tk.Event] = None) -> None:
        choice = self._selected_choice()
        self.detail_var.set(choice.detail if choice is not None else "")
        is_advanced = choice is not None and choice.key == "advanced"
        self.expert_entry.configure(state=tk.NORMAL if is_advanced else tk.DISABLED)
        if is_advanced:
            self.expert_entry.focus_set()

    def _apply(self) -> None:
        choice = self._selected_choice()
        if choice is None:
            messagebox.showwarning("지도 좌표 선택", "좌표 형식을 하나 선택해 주세요.", parent=self)
            return
        if choice.key == "advanced":
            try:
                definition = parse_crs(self.expert_var.get())
            except Exception as exc:
                messagebox.showerror("지도 좌표 입력 오류", str(exc), parent=self)
                return
        else:
            definition = choice.definition
        if definition is None:
            messagebox.showerror("지도 좌표 선택", "선택한 좌표 정보를 읽을 수 없습니다.", parent=self)
            return

        recommendation = self._recommendation
        if (
            recommendation is not None
            and choice.key != recommendation.key
            and recommendation.confidence in {"confirmed", "high", "medium"}
        ):
            recommended_choice = find_choice(self._choices, recommendation.key)
            recommended_title = recommended_choice.title if recommended_choice is not None else recommendation.key
            if not messagebox.askyesno(
                "추천과 다른 좌표",
                f"프로그램은 '{recommended_title}'을(를) 권장합니다.\n"
                f"'{choice.title}'을(를) 적용하시겠습니까?",
                parent=self,
            ):
                return

        self.result = (definition, choice.key)
        self.destroy()


class NFMATApp(WorkspaceActions, tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self._initialize_workspace_actions()
        self.title(TXT_TITLE)
        self.geometry("1500x940")
        self.minsize(1100, 740)

        self.app_root = _app_root()
        self.resource_root = _resource_root()
        self.data_root = _writable_root()
        self.output_root = self.data_root / "output"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.gold_label_root = self.data_root / "gold_labels"
        self.gold_label_root.mkdir(parents=True, exist_ok=True)

        self.dataset_path = str(self.data_root / "cell_dataset.npz")
        self.model_path = str(self.data_root / "cell_model.pt")
        self.settings_path = str(self.data_root / "nfmat_settings.json")
        self._seed_initial_user_data()

        self.img_path: Optional[str] = None
        self.img_bgr: Optional[np.ndarray] = None
        self.img_pil: Optional[Image.Image] = None
        self.tk_img: Optional[ImageTk.PhotoImage] = None
        self.source_kind: Optional[str] = None
        self._source_image_bgr: Optional[np.ndarray] = None
        self._gis_source_path: Optional[str] = None
        self._gis_rows: Optional[int] = None
        self._gis_start_height_field: Optional[str] = None
        self._gis_end_height_field: Optional[str] = None
        self._gis_boundary_path: Optional[str] = None
        self._gis_boundary_field: Optional[str] = None
        self._gis_boundary_value: Optional[str] = None
        self._gis_active_records: Optional[list] = None
        self._gis_active_bbox: Optional[Tuple[float, float, float, float]] = None
        self._gis_active_spatial_index: Optional[PolylineSpatialIndex] = None
        self._gis_active_source_meta: Optional[Dict[str, object]] = None
        self._gis_active_render_records: Optional[list] = None
        self._gis_active_render_bbox: Optional[Tuple[float, float, float, float]] = None
        self._gis_active_render_spatial_index: Optional[PolylineSpatialIndex] = None
        self._gis_active_render_rotation: Optional[float] = None
        self._gis_matrix_render_bbox: Optional[Tuple[float, float, float, float]] = None
        self._gis_matrix_meta: Optional[Dict[str, object]] = None
        self._gis_layers: Dict[str, Dict[str, object]] = {}
        self._gis_layer_sequence = 0
        self._gis_project_crs: Optional[CRSDefinition] = None
        self._gis_pipe_layer_id: Optional[str] = None
        self._gis_boundary_layer_id: Optional[str] = None
        self._gis_flood_layer_id: Optional[str] = None
        self._gis_flood_layer_ids: set[str] = set()
        self._gis_map_bbox: Optional[Tuple[float, float, float, float]] = None
        self._gis_render_cache: Dict[Tuple[str, ...], Tuple[np.ndarray, Tuple[float, float, float, float]]] = {}
        self._gis_last_overview_image: Optional[np.ndarray] = None
        self._gis_last_overview_bbox: Optional[Tuple[float, float, float, float]] = None
        self._gis_last_overview_path: Optional[str] = None
        self._gis_viewport_render_cache: Dict[Tuple[object, ...], Image.Image] = {}
        self._gis_viewport_render_generation = 0
        self._gis_viewport_render_polling = False
        self._gis_viewport_pending_keys: set[Tuple[object, ...]] = set()
        self._gis_viewport_result_queue: queue.Queue[Tuple[int, Tuple[object, ...], object]] = queue.Queue()
        self._gis_force_viewport_render = False
        self._gis_sync_viewport_once = False
        self._gis_boundary_result_cache: Dict[Tuple[object, ...], Tuple[list, Tuple[float, float, float, float], object]] = {}
        self._gis_pipe_result_cache: Dict[Tuple[object, ...], Tuple[list, Tuple[float, float, float, float], object]] = {}
        self._gis_boundary_job_generation = 0
        self._gis_boundary_job_polling = False
        self._gis_boundary_job_queue: queue.Queue[Tuple[int, object]] = queue.Queue()
        self._gis_pipe_job_generation = 0
        self._gis_pipe_job_polling = False
        self._gis_pipe_job_queue: queue.Queue[Tuple[int, object]] = queue.Queue()
        self._gis_render_generation = 0
        self._gis_render_polling = False
        self._gis_deferred_overview_after_id: Optional[str] = None
        self._gis_render_result_queue: queue.Queue[Tuple[int, Tuple[str, ...], object, Tuple[float, float, float, float], str, bool]] = queue.Queue()
        self._gis_hover_boundary: Optional[Tuple[str, int]] = None
        self._gis_last_clicked_boundary: Optional[Tuple[str, int]] = None
        self._flood_cell_mask: Optional[np.ndarray] = None
        self._gis_boundary_overlay_cache: Dict[Tuple[object, ...], List[List[float]]] = {}
        self._gis_hover_after_id: Optional[str] = None
        self._gis_hover_canvas_point: Optional[Tuple[float, float]] = None
        self._gis_load_in_progress = False
        self._gis_load_result_queue: queue.Queue[Tuple[List[Dict[str, object]], List[str]]] = queue.Queue()
        self._gis_crs_job_generation = 0
        self._gis_crs_job_polling = False
        self._gis_crs_result_queue: queue.Queue[Tuple[int, str, object]] = queue.Queue()
        self._gis_index_job_generation = 0
        self._gis_index_job_polling = False
        self._gis_index_job_queue: queue.Queue[Tuple[int, List[Tuple[str, str, int, object]]]] = queue.Queue()
        self._gis_compatibility_confirmed: set[Tuple[str, str]] = set()
        self.matrix_arrow_view = False
        self._matrix_arrow_pil: Optional[Image.Image] = None
        self._matrix_arrow_matrix_key: Optional[str] = None
        self._matrix_arrow_pyramid: List[Tuple[float, Image.Image]] = []
        self._theme_preview_photo: Optional[ImageTk.PhotoImage] = None
        self._brand_theme_photo: Optional[ImageTk.PhotoImage] = None
        self._brand_icon_photo: Optional[ImageTk.PhotoImage] = None
        self._roi_mask_pil: Optional[Image.Image] = None
        self._roi_mask_pyramid: List[Tuple[float, Image.Image]] = []
        self._detection_cache: Optional[Dict[str, object]] = None
        self._detection_cache_key: Optional[str] = None

        self.A = 12
        self.B = 12
        self.base_matrix: Optional[np.ndarray] = None
        self.current_matrix: Optional[np.ndarray] = None
        self.base_confidence_matrix: Optional[np.ndarray] = None
        self.confidence_matrix: Optional[np.ndarray] = None
        self.base_model_applied_mask: Optional[np.ndarray] = None
        self.model_applied_mask: Optional[np.ndarray] = None
        self.occ_matrix: Optional[np.ndarray] = None
        self.path_occ_matrix: Optional[np.ndarray] = None
        self.support_mask: Optional[np.ndarray] = None
        self.unresolved_mask: Optional[np.ndarray] = None
        self.user_edit_mask: Optional[np.ndarray] = None
        self.manual_edits_since_autogen = False
        self.last_plena_sanitized_count = 0
        self.last_plena_lasso_removed_count = 0
        self.last_plena_unreachable_count = 0
        self.last_plena_outlet_reduced_count = 0
        self._last_plena_input_matrix: Optional[np.ndarray] = None
        self.compressed_grid_shape: Optional[Tuple[int, int]] = None
        self.compression_source_shape: Optional[Tuple[int, int]] = None
        self.trimmed_grid_shape: Optional[Tuple[int, int]] = None
        self.trim_source_shape: Optional[Tuple[int, int]] = None
        self.trim_bounds: Optional[Tuple[int, int, int, int]] = None
        self.arrow_boxes: list[Tuple[int, int, int, int]] = []
        self.selected_cell: Optional[Tuple[int, int]] = None
        self.outlet_cell: Optional[Tuple[int, int]] = None
        self.outlet_cells: set[Tuple[int, int]] = set()
        self.outlet_pick_mode = False
        self.lasso_mode = False
        self.roi_polygon_points: List[Tuple[float, float]] = []
        self.roi_mask: Optional[np.ndarray] = None
        self.roi_cell_mask: Optional[np.ndarray] = None
        self._lasso_preview_point: Optional[Tuple[float, float]] = None

        self._fit_scale = 1.0
        self._zoom = 1.0
        self._scale = 1.0
        self._display_size = (1, 1)
        self._display_origin = (0.0, 0.0)
        self._display_origin_initialized = False
        self._pan_start: Optional[Tuple[int, int]] = None
        self._pan_origin = (0.0, 0.0)
        self._pan_last_canvas: Optional[Tuple[int, int]] = None
        self._icon_photo: Optional[tk.PhotoImage] = None
        self._image_pyramid: List[Tuple[float, Image.Image]] = []
        self._render_after_id: Optional[str] = None
        self._render_hq_after_id: Optional[str] = None
        self._render_high_quality = True
        self._plena_process: Optional[subprocess.Popen[bytes]] = None
        self._plena_progress_window: Optional[tk.Toplevel] = None
        self._plena_progress_text: Optional[scrolledtext.ScrolledText] = None
        self._toolbar_layout_key: Optional[Tuple[Tuple[int, ...], ...]] = None

        self.status_var = tk.StringVar(value=TXT_START)
        self.rows_var = tk.IntVar(value=self.A)
        self.rotation_var = tk.DoubleVar(value=0.0)
        self.selected_var = tk.StringVar(value=TXT_NO_SELECTION)
        self.summary_var = tk.StringVar(value=TXT_NO_MATRIX)
        self.gis_cell_size_var = tk.StringVar(value="GIS 셀 크기: -")
        self.gis_slope_var = tk.StringVar(value="점유 셀 평균 경사: -")
        self.flood_region_var = tk.StringVar(value="침수지역 레이어: 미지정")
        self.gis_project_crs_var = tk.StringVar(value="전체 지도 좌표: 자동")
        self.device_choice_var = tk.StringVar(value="\uc790\ub3d9")
        self.show_fill_var = tk.BooleanVar(value=True)
        self.theme_var = tk.StringVar(value="NFMAT White")
        self.runtime_var = tk.StringVar(value="\ub7f0\ud0c0\uc784 \uc815\ubcf4\ub97c \ubd88\ub7ec\uc624\ub294 \uc911...")

        self._apply_window_icon()
        self._build_ui()
        self._load_settings()
        self._apply_theme()
        self._refresh_runtime_info()
        self.bind("<Configure>", self._on_resize)
        self.protocol("WM_DELETE_WINDOW", self._on_workspace_close)

    def _seed_initial_user_data(self) -> None:
        # Public FMT must not silently import private NFMAT5/6 training data.
        # Only explicitly placed, trusted resource files may seed this profile.
        for filename in ("cell_dataset.npz", "cell_model.pt", "nfmat_settings.json"):
            target = self.data_root / filename
            if target.exists():
                continue
            source = _resource_path(filename)
            if not source.exists():
                continue
            try:
                shutil.copy2(source, target)
            except OSError:
                pass

    def _apply_window_icon(self) -> None:
        icon_path = _resource_path("icon.png")
        if not icon_path.exists():
            return
        try:
            self._icon_photo = tk.PhotoImage(file=str(icon_path))
            self.iconphoto(True, self._icon_photo)
        except Exception:
            self._icon_photo = None

    def _load_settings(self) -> None:
        path = Path(self.settings_path)
        if not path.exists():
            return
        try:
            data = load_settings_object(path)
        except Exception:
            return
        theme = LEGACY_THEME_ALIASES.get(str(data.get("theme", self.theme_var.get())), self.theme_var.get())
        device_choice = LEGACY_DEVICE_ALIASES.get(
            str(data.get("device_choice", self.device_choice_var.get())),
            self.device_choice_var.get(),
        )
        if theme in THEME_PRESETS:
            self.theme_var.set(theme)
        if device_choice in DEVICE_CHOICES:
            self.device_choice_var.set(device_choice)
        try:
            rotation_degrees = float(data.get("rotation_degrees", self.rotation_var.get()))
        except Exception:
            rotation_degrees = 0.0
        if np.isfinite(rotation_degrees):
            self.rotation_var.set(float(np.clip(rotation_degrees, -360.0, 360.0)))

    def _save_settings(self) -> None:
        data = {
            "theme": self.theme_var.get(),
            "device_choice": self.device_choice_var.get(),
            "rotation_degrees": self._rotation_degrees(),
        }
        atomic_write_text(self.settings_path, json.dumps(data, ensure_ascii=False, indent=2))

    def _theme_palette(self) -> Dict[str, str]:
        return THEME_PRESETS.get(self.theme_var.get(), THEME_PRESETS["NFMAT White"])

    def _apply_theme(self) -> None:
        palette = self._theme_palette()
        style = ttk.Style(self)
        try:
            style.theme_use(palette.get("ttk_theme", "clam"))
        except tk.TclError:
            pass

        bg = palette["bg"]
        panel = palette["panel"]
        panel_alt = palette.get("panel_alt", panel)
        accent = palette["accent"]
        accent_alt = palette.get("accent_alt", accent)
        text = palette["text"]
        muted = palette["muted"]
        entry_bg = palette.get("entry_bg", "white")
        border = palette.get("border", panel)
        is_glass = palette.get("glass", "0") == "1"

        self.configure(background=bg)
        style.configure("Root.TFrame", background=bg)
        style.configure("TFrame", background=bg)
        style.configure(
            "GlassPanel.TFrame",
            background=panel,
            borderwidth=1 if is_glass else 0,
            relief="ridge" if is_glass else "flat",
        )
        style.configure("TPanedwindow", background=bg)
        style.configure("TLabel", background=bg, foreground=text, font=("Malgun Gothic", 9))
        style.configure(
            "TButton",
            background=panel,
            foreground=text,
            bordercolor=border,
            lightcolor=border,
            darkcolor=palette.get("shadow", panel),
            focusthickness=1,
            focuscolor=accent_alt,
            padding=(8, 7),
            relief="flat",
            font=("Malgun Gothic", 9),
        )
        style.map(
            "TButton",
            background=[("pressed", accent), ("active", accent_alt)],
            foreground=[("pressed", "white"), ("active", "#061528" if is_glass else "white")],
            relief=[("pressed", "sunken"), ("active", "ridge")],
        )
        style.configure("TNotebook", background=bg, borderwidth=0, tabmargins=(4, 4, 4, 0))
        style.configure(
            "TNotebook.Tab",
            background=panel_alt,
            foreground=muted,
            bordercolor=border,
            lightcolor=border,
            padding=(14, 8),
            font=("Malgun Gothic", 9),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", panel), ("active", "#ffffff" if is_glass else accent)],
            foreground=[("selected", text), ("active", text if is_glass else "white")],
        )
        style.configure(
            "TLabelframe",
            background=panel,
            foreground=text,
            bordercolor=border,
            lightcolor=border,
            darkcolor=palette.get("shadow", panel),
            relief="ridge" if is_glass else "groove",
            borderwidth=1,
        )
        style.configure("TLabelframe.Label", background=panel, foreground=muted, padding=(2, 1), font=("Malgun Gothic", 9, "bold"))
        style.configure("TCombobox", fieldbackground=entry_bg, background=panel, foreground=text, arrowcolor=accent)
        style.configure("TSpinbox", fieldbackground=entry_bg, background=panel, foreground=text, arrowcolor=accent)
        style.configure("Vertical.TScrollbar", background=panel_alt, troughcolor=bg, bordercolor=border, arrowcolor=accent)

        self.canvas.configure(background=palette["canvas"])
        self.matrix_text.configure(
            background=palette.get("matrix_bg", entry_bg),
            foreground=MATRIX_NUMBER_COLOR,
            insertbackground=text,
            selectbackground=accent,
            selectforeground="white",
            relief=tk.FLAT if is_glass else tk.SUNKEN,
            borderwidth=1,
            highlightthickness=1 if is_glass else 0,
            highlightbackground=border,
        )
        glass_frame_names = {"toolbar_frame", "top_row1_frame", "top_row2_frame", "top_row3_frame", "settings_outer_frame"}
        for frame_name in ("outer_frame", "toolbar_frame", "top_row1_frame", "top_row2_frame", "top_row3_frame", "layer_panel_frame", "left_frame", "right_frame", "info_top_frame", "rules_frame", "settings_outer_frame"):
            frame = getattr(self, frame_name, None)
            if frame is not None:
                frame.configure(style="GlassPanel.TFrame" if frame_name in glass_frame_names and is_glass else "Root.TFrame")
        for frame in getattr(self, "_toolbar_rows", []):
            frame.configure(style="GlassPanel.TFrame" if is_glass else "Root.TFrame")
        self._refresh_brand_panel()
        apply_workspace_style(self)

    def _refresh_brand_panel(self) -> None:
        palette = self._theme_palette()
        bg = palette["brand_bg"]
        text = palette["text"]
        muted = palette["muted"]
        border = palette.get("border", bg)
        is_glass = palette.get("glass", "0") == "1"
        theme_image_rel = palette.get("theme_image", "")
        theme_image_path = _resource_path(theme_image_rel) if theme_image_rel else None

        self.brand_frame.configure(bg=bg, highlightbackground=border, highlightthickness=1 if is_glass else 0)
        for widget in (self.brand_image_label, self.brand_icon_label, self.brand_title_label, self.brand_credit_label):
            widget.configure(bg=bg)
        self.theme_preview_label.configure(
            bg=palette["panel"],
            fg=text,
            highlightbackground=border,
            highlightthickness=1 if is_glass else 0,
        )
        self.brand_title_label.configure(fg=text)
        self.brand_credit_label.configure(fg=muted)

        self.brand_image_label.configure(image="", text="")

        if theme_image_path is not None and theme_image_path.exists():
            try:
                preview_img = Image.open(theme_image_path).convert("RGB")
                settings_img = preview_img.resize((320, 180), Image.BILINEAR)
                self._theme_preview_photo = ImageTk.PhotoImage(settings_img)
                self.theme_preview_label.configure(image=self._theme_preview_photo, text="")
            except Exception:
                self.theme_preview_label.configure(image="", text="\ud14c\ub9c8 \ubbf8\ub9ac\ubcf4\uae30")
        else:
            self.theme_preview_label.configure(image="", text="White Theme")

        icon_path = _resource_path("icon.png")
        if icon_path.exists():
            try:
                icon_img = Image.open(icon_path).convert("RGBA").resize((84, 84), Image.LANCZOS)
                self._brand_icon_photo = ImageTk.PhotoImage(icon_img)
                self.brand_icon_label.configure(image=self._brand_icon_photo, text="")
            except Exception:
                self.brand_icon_label.configure(image="")
        else:
            self.brand_icon_label.configure(image="")

    def _refresh_runtime_info(self) -> None:
        info = get_torch_runtime_info()
        device_lines = [f"- {name}" for name in info["device_names"]] if info["device_names"] else ["- \uc5c6\uc74c"]
        cuda_build = info["torch_cuda_version"] or "\uc5c6\uc74c"
        cuda_available = "\uc608" if info["cuda_available"] else "\uc544\ub2c8\uc624"
        note_lines: List[str] = []
        if bool(info.get("is_cpu_build")):
            note_lines.append("\ucc38\uace0: \ud604\uc7ac torch \ube4c\ub4dc\ub294 CPU \uc804\uc6a9\uc785\ub2c8\ub2e4.")
            note_lines.append("\ucd94\ud6c4 exe \ubc30\ud3ec \uc2dc CUDA \uc0ac\uc6a9\uc744 \uc6d0\ud558\uba74 CUDA \uc9c0\uc6d0 torch \ube4c\ub4dc\ub97c \ud3ec\ud568\ud574\uc57c \ud569\ub2c8\ub2e4.")
        if RUNTIME_TRAINING_ENABLED:
            note_lines.append("현재 빌드는 사용자 수정 저장 및 추가 학습을 계속 사용할 수 있습니다.")
        self.runtime_var.set(
            "\n".join(
                [
                    f"PyTorch: {info['torch_version']}",
                    f"CUDA \ube4c\ub4dc: {cuda_build}",
                    f"CUDA \uc0ac\uc6a9 \uac00\ub2a5: {cuda_available}",
                    f"\uac80\ucd9c\ub41c GPU \uc218: {info['device_count']}",
                    "\uc778\uc2dd\ub41c \uc7a5\uce58:",
                    *device_lines,
                    *note_lines,
                ]
            )
        )

    def _device_preference(self) -> str:
        return DEVICE_CHOICES.get(self.device_choice_var.get(), "auto")

    def apply_settings(self) -> None:
        self._apply_theme()
        self._refresh_runtime_info()
        self._save_settings()
        self.status_var.set(
            f"\uc124\uc815\uc744 \uc801\uc6a9\ud588\uc2b5\ub2c8\ub2e4. \uc7a5\uce58={self.device_choice_var.get()}, \ud14c\ub9c8={self.theme_var.get()}"
        )
        self._queue_render(high_quality=True, delay=0)

    def _build_ui(self) -> None:
        build_workspace_ui(self, device_choices=DEVICE_CHOICES, theme_names=list(THEME_PRESETS),
                           training_enabled=RUNTIME_TRAINING_ENABLED)

    def show_help(self) -> None:
        messagebox.showinfo(TXT_HELP, TXT_HELP_BODY)

    @staticmethod
    def _partition_toolbar_widths(
        widths: List[int],
        available_width: int,
        gap: int = 6,
    ) -> Tuple[Tuple[int, ...], ...]:
        count = len(widths)
        if count == 0:
            return ()
        available = max(1, int(available_width))
        best_fallback: Optional[Tuple[Tuple[int, ...], ...]] = None
        best_fallback_overflow: Optional[int] = None
        for row_count in range(1, count + 1):
            best: Optional[Tuple[Tuple[int, ...], ...]] = None
            best_score: Optional[Tuple[int, int]] = None
            for cuts in combinations(range(1, count), row_count - 1):
                boundaries = (0,) + cuts + (count,)
                rows = tuple(tuple(range(boundaries[index], boundaries[index + 1])) for index in range(row_count))
                row_widths = [
                    sum(widths[item] for item in row) + max(0, len(row) - 1) * int(gap)
                    for row in rows
                ]
                overflow = max(max(row_widths) - available, 0)
                if best_fallback_overflow is None or overflow < best_fallback_overflow:
                    best_fallback = rows
                    best_fallback_overflow = overflow
                if overflow > 0:
                    continue
                score = (max(row_widths), max(row_widths) - min(row_widths))
                if best_score is None or score < best_score:
                    best = rows
                    best_score = score
            if best is not None:
                return best
        return best_fallback or tuple((index,) for index in range(count))

    def _refresh_toolbar_layout(self, window_width: Optional[int] = None) -> None:
        refresh_workspace_layout(self, window_width)

    def _on_resize(self, event: tk.Event) -> None:
        if event.widget is self:
            self._refresh_toolbar_layout(int(event.width))
            self._queue_render(high_quality=False, delay=15)
            self._queue_render(high_quality=True, delay=120)

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._display_origin = (0.0, 0.0)
        self._display_origin_initialized = False
        self._pan_origin = (0.0, 0.0)
        self._pan_start = None
        self._pan_last_canvas = None
        self._queue_render(high_quality=True, delay=0)

    def _build_image_pyramid(self) -> None:
        self._image_pyramid = []
        if self.img_pil is None:
            return
        current = self.img_pil.copy()
        factor = 1.0
        self._image_pyramid.append((factor, current))
        while min(current.size) > 768:
            factor *= 0.5
            current = current.resize((max(1, current.size[0] // 2), max(1, current.size[1] // 2)), Image.BILINEAR)
            self._image_pyramid.append((factor, current))
        self._build_roi_mask_pyramid()

    def _build_matrix_arrow_pyramid(self) -> None:
        self._matrix_arrow_pyramid = []
        if self._matrix_arrow_pil is None:
            return
        current = self._matrix_arrow_pil.copy()
        factor = 1.0
        self._matrix_arrow_pyramid.append((factor, current))
        while min(current.size) > 768:
            factor *= 0.5
            current = current.resize((max(1, current.size[0] // 2), max(1, current.size[1] // 2)), Image.BILINEAR)
            self._matrix_arrow_pyramid.append((factor, current))

    def _matrix_arrow_key(self) -> str:
        if self.current_matrix is None:
            return ""
        arr = np.ascontiguousarray(self.current_matrix.astype(np.uint8, copy=False))
        digest = hashlib.blake2b(arr.tobytes(), digest_size=8).hexdigest()
        flood_digest = "none"
        flood_mask = vars(self).get("_flood_cell_mask")
        if isinstance(flood_mask, np.ndarray) and flood_mask.shape == arr.shape:
            flood = np.ascontiguousarray(flood_mask.astype(np.uint8, copy=False))
            flood_digest = hashlib.blake2b(flood.tobytes(), digest_size=6).hexdigest()
        area = vars(self).get("_area_fill_mask")
        area_digest = hashlib.blake2b(area.tobytes(), digest_size=6).hexdigest() if isinstance(area, np.ndarray) else "none"
        visible = self.show_fill_var.get() if vars(self).get("show_fill_var") is not None else True
        return f"{arr.shape[0]}x{arr.shape[1]}:{digest}:flood={flood_digest}:area={area_digest}:{visible}"

    def _ensure_matrix_arrow_image(self) -> Optional[Image.Image]:
        if self.current_matrix is None:
            return None
        key = self._matrix_arrow_key()
        if self._matrix_arrow_pil is not None and self._matrix_arrow_matrix_key == key:
            return self._matrix_arrow_pil
        display_matrix = self.current_matrix.copy()
        area = vars(self).get("_area_fill_mask")
        if isinstance(area, np.ndarray) and vars(self).get("show_fill_var") is not None and not self.show_fill_var.get():
            display_matrix[area > 0] = 0
        image_bgr = render_direction_matrix_arrows(
            display_matrix,
            cell_px=max(4, min(42, 4096 // max(display_matrix.shape))),
            margin=28,
            highlight_mask=vars(self).get("_flood_cell_mask"),
            area_fill_mask=area,
        )
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._matrix_arrow_pil = Image.fromarray(rgb)
        self._matrix_arrow_matrix_key = key
        self._build_matrix_arrow_pyramid()
        return self._matrix_arrow_pil

    def _active_display_image(self) -> Optional[Image.Image]:
        if self.matrix_arrow_view:
            return self._ensure_matrix_arrow_image()
        return self.img_pil

    def _build_roi_mask_pyramid(self) -> None:
        self._roi_mask_pyramid = []
        if self._roi_mask_pil is None:
            return
        if not self._image_pyramid:
            self._roi_mask_pyramid.append((1.0, self._roi_mask_pil.copy()))
            return
        for factor, image in self._image_pyramid:
            if image.size == self._roi_mask_pil.size:
                mask_img = self._roi_mask_pil.copy()
            else:
                mask_img = self._roi_mask_pil.resize(image.size, Image.NEAREST)
            self._roi_mask_pyramid.append((factor, mask_img))

    def _select_roi_mask_level(self, target_scale: float) -> Optional[Image.Image]:
        if not self._roi_mask_pyramid:
            return None
        desired = float(np.clip(target_scale, 0.125, 1.0))
        best_factor, best_image = self._roi_mask_pyramid[0]
        best_error = abs(np.log(max(best_factor, 1e-6)) - np.log(max(desired, 1e-6)))
        for factor, image in self._roi_mask_pyramid:
            error = abs(np.log(max(factor, 1e-6)) - np.log(max(desired, 1e-6)))
            if error < best_error:
                best_error = error
                best_factor, best_image = factor, image
        return best_image

    def _queue_render(self, high_quality: bool, delay: int = 0) -> None:
        if high_quality:
            if delay == 0 and self._render_after_id is not None:
                self.after_cancel(self._render_after_id)
                self._render_after_id = None
            if self._render_hq_after_id is not None:
                self.after_cancel(self._render_hq_after_id)
            self._render_hq_after_id = self.after(delay, lambda: self._render_canvas(high_quality=True))
            return
        if self._render_after_id is not None:
            self.after_cancel(self._render_after_id)
        self._render_after_id = self.after(delay, lambda: self._render_canvas(high_quality=False))

    def _select_pyramid_level_from(
        self,
        pyramid: List[Tuple[float, Image.Image]],
        fallback: Optional[Image.Image],
        target_scale: float,
    ) -> Tuple[float, Image.Image]:
        if not pyramid:
            assert fallback is not None
            return 1.0, fallback
        desired = float(np.clip(target_scale, 0.125, 1.0))
        best_factor, best_image = pyramid[0]
        best_error = abs(np.log(max(best_factor, 1e-6)) - np.log(max(desired, 1e-6)))
        for factor, image in pyramid:
            error = abs(np.log(max(factor, 1e-6)) - np.log(max(desired, 1e-6)))
            if error < best_error:
                best_error = error
                best_factor, best_image = factor, image
        return best_factor, best_image

    def _select_pyramid_level(self, target_scale: float) -> Tuple[float, Image.Image]:
        return self._select_pyramid_level_from(self._image_pyramid, self.img_pil, target_scale)

    def _select_active_pyramid_level(self, target_scale: float) -> Tuple[float, Image.Image]:
        if self.matrix_arrow_view:
            return self._select_pyramid_level_from(
                self._matrix_arrow_pyramid,
                self._matrix_arrow_pil,
                target_scale,
            )
        return self._select_pyramid_level(target_scale)

    def _image_point_from_event(self, event: tk.Event) -> Optional[Tuple[float, float]]:
        if self.matrix_arrow_view or self.img_pil is None:
            return None
        x = (float(event.x) - float(self._display_origin[0])) / max(self._scale, 1e-6)
        y = (float(event.y) - float(self._display_origin[1])) / max(self._scale, 1e-6)
        if x < 0 or y < 0 or x >= self.img_pil.size[0] or y >= self.img_pil.size[1]:
            return None
        return x, y

    def _clear_roi(self) -> None:
        self.lasso_mode = False
        self.roi_polygon_points = []
        self.roi_mask = None
        self._roi_mask_pil = None
        self._roi_mask_pyramid = []
        self.roi_cell_mask = None
        self._lasso_preview_point = None
        self._invalidate_detection_cache()

    def _roi_cell_mask_from_polygon(self) -> Optional[np.ndarray]:
        if self.img_bgr is None or len(self.roi_polygon_points) < 3:
            return None
        h, w = self.img_bgr.shape[:2]
        poly = np.round(np.asarray(self.roi_polygon_points, dtype=np.float32)).astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        self._roi_mask_pil = Image.fromarray(mask, mode="L")
        self._build_roi_mask_pyramid()
        cell_mask = resample_mask_centers(mask, (self.A, self.B))
        self.roi_mask = mask
        return cell_mask

    def _processing_image_and_roi_mask(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.img_bgr is None:
            raise RuntimeError("image is not loaded")
        if self.roi_mask is None or self.roi_cell_mask is None:
            return self.img_bgr, None
        masked = self.img_bgr.copy()
        masked[self.roi_mask == 0] = 255
        return masked, self.roi_cell_mask

    def _merge_roi_matrix(self, previous: np.ndarray, updated: np.ndarray, roi_mask: Optional[np.ndarray]) -> np.ndarray:
        result = np.asarray(updated, dtype=previous.dtype if previous is not None else updated.dtype).copy()
        if roi_mask is not None:
            if roi_mask.shape != result.shape:
                raise ValueError("LASSO 마스크와 계산 결과의 크기가 다릅니다.")
            result[roi_mask != 1] = 0
        return result

    def _invalidate_detection_cache(self) -> None:
        self._detection_cache = None
        self._detection_cache_key = None

    def _roi_signature(self) -> str:
        if self.roi_cell_mask is None:
            return "full"
        roi_bytes = np.ascontiguousarray(self.roi_cell_mask.astype(np.uint8)).tobytes()
        return hashlib.blake2b(roi_bytes, digest_size=8).hexdigest()

    def _current_detection_cache_key(self) -> Optional[str]:
        if self.img_bgr is None:
            return None
        image_token = self.img_path or f"memory:{self.img_bgr.shape[0]}x{self.img_bgr.shape[1]}"
        return f"{image_token}|A={self.A}|rot={self._rotation_degrees():.4f}|roi={self._roi_signature()}"

    def _store_detection_cache(
        self,
        cache_key: Optional[str],
        direction_matrix: np.ndarray,
        occ_matrix: np.ndarray,
        arrow_boxes: List[Tuple[int, int, int, int]],
        meta: Dict[str, np.ndarray],
    ) -> None:
        copied_meta: Dict[str, object] = {}
        for key, value in meta.items():
            if isinstance(value, np.ndarray):
                copied_meta[key] = value.copy()
            else:
                copied_meta[key] = value
        self._detection_cache = {
            "A": int(self.A),
            "morph_dir": direction_matrix.astype(np.uint8).copy(),
            "occ": occ_matrix.astype(np.uint8).copy(),
            "bboxes": [tuple(int(v) for v in box) for box in arrow_boxes],
            "meta": copied_meta,
        }
        self._detection_cache_key = cache_key

    def _current_learning_group_key(self) -> Optional[str]:
        if self.img_path:
            boundary_path = vars(self).get("_gis_boundary_path") or ""
            boundary_field = vars(self).get("_gis_boundary_field") or ""
            boundary_value = vars(self).get("_gis_boundary_value") or ""
            base_key = f"{Path(self.img_path).resolve()}|rot={self._rotation_degrees():.4f}"
            if boundary_value:
                return f"{base_key}|boundary={boundary_path}|field={boundary_field}|value={boundary_value}"
            return base_key
        return None

    def _read_image(self, path: str) -> np.ndarray:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            pil_img = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img

    def _rotation_degrees(self) -> float:
        try:
            degrees = float(self.rotation_var.get())
        except Exception:
            degrees = 0.0
        if not np.isfinite(degrees):
            degrees = 0.0
        degrees = float(np.clip(degrees, -360.0, 360.0))
        if abs(degrees) < 1e-9:
            degrees = 0.0
        return degrees

    @staticmethod
    def _rotate_bgr_image(image_bgr: np.ndarray, degrees: float) -> np.ndarray:
        angle = float(degrees)
        if image_bgr is None:
            raise ValueError("image is not loaded")
        if abs(angle) <= 1e-9:
            return image_bgr.copy()
        h, w = image_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return image_bgr.copy()
        center = (w / 2.0, h / 2.0)
        transform = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_a = abs(float(transform[0, 0]))
        sin_a = abs(float(transform[0, 1]))
        new_w = max(1, int(np.ceil(h * sin_a + w * cos_a)))
        new_h = max(1, int(np.ceil(h * cos_a + w * sin_a)))
        transform[0, 2] += (new_w / 2.0) - center[0]
        transform[1, 2] += (new_h / 2.0) - center[1]
        return cv2.warpAffine(
            image_bgr,
            transform,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    def _set_display_image_from_bgr(self, image_bgr: np.ndarray) -> None:
        self.img_bgr = image_bgr
        rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        self.img_pil = Image.fromarray(rgb)
        self._build_image_pyramid()

    @staticmethod
    def _format_meter_distance(value: float) -> str:
        distance = abs(float(value))
        if distance >= 1000.0:
            return f"{distance / 1000.0:.3g} km"
        if distance >= 10.0:
            return f"{distance:.3g} m"
        if distance >= 1.0:
            return f"{distance:.2f} m"
        if distance >= 0.01:
            return f"{distance:.3f} m"
        return f"{distance:.3g} m"

    @staticmethod
    def _format_coordinate_distance(value: float) -> str:
        distance = abs(float(value))
        if distance >= 1000.0:
            return f"{distance:.3g}"
        if distance >= 1.0:
            return f"{distance:.3f}"
        return f"{distance:.6g}"

    def _update_gis_cell_size_label(self, meta: Optional[Dict[str, object]] = None) -> None:
        if not meta:
            self.gis_cell_size_var.set("GIS 셀 크기: -")
            self.gis_slope_var.set("점유 셀 평균 경사: -")
            return
        rows = int(meta.get("rows_A", self.A) or self.A)
        cols = int(meta.get("cols_B", self.B) or self.B)
        width_m = meta.get("cell_width_m")
        height_m = meta.get("cell_height_m")
        unit_type = str(meta.get("coordinate_unit_type", "unknown") or "unknown")
        note = str(meta.get("cell_size_note", "") or "")
        if width_m is not None and height_m is not None:
            width_text = self._format_meter_distance(float(width_m))
            height_text = self._format_meter_distance(float(height_m))
            suffix = " (경위도 근사)" if unit_type == "degree" or note else ""
            self.gis_cell_size_var.set(
                f"GIS 셀 크기: 가로 {width_text} × 세로 {height_text} / 1칸 | {rows}×{cols}{suffix}"
            )
            self._update_gis_slope_label(meta)
            return
        cell_width = float(meta.get("cell_width", 0.0) or 0.0)
        cell_height = float(meta.get("cell_height", 0.0) or 0.0)
        unit_name = str(meta.get("coordinate_unit_name", "") or "").strip()
        unit_label = unit_name if unit_name else "좌표단위"
        self.gis_cell_size_var.set(
            "GIS 셀 크기: "
            f"가로 {self._format_coordinate_distance(cell_width)} × "
            f"세로 {self._format_coordinate_distance(cell_height)} {unit_label} / 1칸 | "
            f"{rows}×{cols} (m 단위 미확인)"
        )
        self._update_gis_slope_label(meta)

    def _update_gis_slope_label(self, meta: Dict[str, object]) -> None:
        if not bool(meta.get("occupied_slope_available", False)):
            note = str(meta.get("occupied_slope_note", "") or "")
            if note:
                self.gis_slope_var.set(f"점유 셀 평균 경사: 계산 불가 ({note})")
            else:
                self.gis_slope_var.set("점유 셀 평균 경사: 계산 불가 (표고 필드 없음)")
            return
        cell_count = int(meta.get("occupied_slope_cell_count", 0) or 0)
        slope_mean = meta.get("occupied_slope_mean")
        if slope_mean is None:
            self.gis_slope_var.set("점유 셀 평균 경사: 계산 불가")
            return
        if bool(meta.get("occupied_slope_metric", False)):
            percent = float(meta.get("occupied_slope_percent", float(slope_mean) * 100.0))
            per_mille = float(meta.get("occupied_slope_per_mille", float(slope_mean) * 1000.0))
            note = str(meta.get("occupied_slope_note", "") or "")
            suffix = " (경위도 근사)" if note else ""
            self.gis_slope_var.set(
                f"점유 셀 평균 경사: {percent:.3g}% ({per_mille:.3g}‰) | 점유 {cell_count}셀{suffix}"
            )
            return
        self.gis_slope_var.set(
            f"점유 셀 평균 경사: {float(slope_mean):.6g} 표고/좌표단위 | 점유 {cell_count}셀 (m 단위 미확인)"
        )

    def _invalidate_matrix_arrow_cache(self) -> None:
        self._matrix_arrow_pil = None
        self._matrix_arrow_matrix_key = None
        self._matrix_arrow_pyramid = []

    def _reset_image_matrix_state(self) -> None:
        self._clear_compressed_grid_shape()
        self._clear_trimmed_grid_state()
        self._sync_grid_shape()
        self._flood_cell_mask = None
        self._basin_cell_mask = None
        self._basin_is_lasso = False
        self._basin_ring_groups = None
        self._area_fill_mask = None
        self._area_fill_values = None
        self.base_matrix = np.zeros((self.A, self.B), dtype=np.uint8)
        self.current_matrix = self.base_matrix.copy()
        self.base_confidence_matrix = np.zeros((self.A, self.B), dtype=np.float32)
        self.confidence_matrix = self.base_confidence_matrix.copy()
        self.base_model_applied_mask = np.zeros((self.A, self.B), dtype=np.uint8)
        self.model_applied_mask = self.base_model_applied_mask.copy()
        self.occ_matrix = np.zeros((self.A, self.B), dtype=np.uint8)
        self.path_occ_matrix = np.zeros((self.A, self.B), dtype=np.uint8)
        self.support_mask = np.zeros((self.A, self.B), dtype=np.uint8)
        self.unresolved_mask = np.zeros((self.A, self.B), dtype=np.uint8)
        self.user_edit_mask = np.zeros((self.A, self.B), dtype=np.uint8)
        self.manual_edits_since_autogen = False
        self.arrow_boxes = []
        self.outlet_cell = None
        self.outlet_cells = set()
        self.outlet_pick_mode = False
        self.matrix_arrow_view = False
        if hasattr(self, "matrix_arrow_button"):
            self.matrix_arrow_button.configure(text=TXT_MATRIX_ARROWS)
        self._update_gis_cell_size_label(None)
        self._clear_roi()
        self._invalidate_detection_cache()
        self._invalidate_matrix_arrow_cache()
        self._update_flood_region_label()
        self._set_selected_cell(0, 0)
        self._record_support_baseline()

    def _clear_unrenderable_gis_state(self) -> None:
        self._gis_map_bbox = None
        self.img_bgr = None
        self.img_pil = None
        self.tk_img = None
        self.source_kind = "gis_project"
        for name in (
            "base_matrix",
            "current_matrix",
            "base_confidence_matrix",
            "confidence_matrix",
            "base_model_applied_mask",
            "model_applied_mask",
            "occ_matrix",
            "path_occ_matrix",
            "support_mask",
            "unresolved_mask",
            "user_edit_mask",
        ):
            setattr(self, name, None)
        self.arrow_boxes = []
        self.selected_cell = None
        self.outlet_cell = None
        self.outlet_cells = set()
        self.outlet_pick_mode = False
        self.matrix_arrow_view = False
        self._clear_compressed_grid_shape()
        self._clear_trimmed_grid_state()
        self._clear_roi()
        self._invalidate_detection_cache()
        self._invalidate_matrix_arrow_cache()
        self._update_gis_cell_size_label(None)
        self.canvas.delete("all")
        self._update_matrix_preview()

    def _has_rotation_reset_risk(self) -> bool:
        if self.manual_edits_since_autogen:
            return True
        if self._active_outlet_cells() or self.roi_cell_mask is not None:
            return True
        if self.user_edit_mask is not None and int(np.count_nonzero(self.user_edit_mask)) > 0:
            return True
        if self.model_applied_mask is not None and int(np.count_nonzero(self.model_applied_mask)) > 0:
            return True
        if self.source_kind == "image" and self.current_matrix is not None:
            return int(np.count_nonzero(self.current_matrix)) > 0
        if self.base_matrix is not None and self.current_matrix is not None:
            return bool(np.any(self.current_matrix != self.base_matrix))
        return False

    def _confirm_rotation_reset(self) -> bool:
        if not self._has_rotation_reset_risk():
            return True
        return messagebox.askyesno(
            "회전 적용",
            "회전을 적용하면 현재 행렬, ROI, 출구 선택, 수정값을 새 입력 기준으로 다시 만듭니다.\n계속할까요?",
            parent=self,
        )

    def _reload_image_with_rotation(self) -> None:
        if self._source_image_bgr is None:
            raise RuntimeError("원본 이미지가 없습니다.")
        if self.trim_source_shape is not None and int(self.rows_var.get()) == int(self.A):
            self.rows_var.set(int(self.trim_source_shape[0]))
        rotated = self._rotate_bgr_image(self._source_image_bgr, self._rotation_degrees())
        self._set_display_image_from_bgr(rotated)
        self._reset_image_matrix_state()
        self.reset_view()

    def _apply_gis_result_to_state(self, result) -> None:
        self._clear_trimmed_grid_state()
        result_bbox = result.meta.get("bbox") if isinstance(result.meta, dict) else None
        if isinstance(result_bbox, (list, tuple)) and len(result_bbox) == 4:
            self._gis_matrix_render_bbox = tuple(float(value) for value in result_bbox)
        self._set_display_image_from_bgr(result.image_bgr)
        if self.source_kind == "gis" and self._gis_active_records is not None:
            self._refresh_active_gis_render_geometry()
        self._clear_compressed_grid_shape()
        self.rows_var.set(int(result.matrix.shape[0]))
        self._sync_grid_shape()
        if (self.A, self.B) != result.matrix.shape:
            self.A, self.B = result.matrix.shape
            self.rows_var.set(self.A)
        self.base_matrix = result.matrix.astype(np.uint8)
        self.current_matrix = self.base_matrix.copy()
        self.base_confidence_matrix = result.confidence.astype(np.float32)
        self.confidence_matrix = self.base_confidence_matrix.copy()
        self.base_model_applied_mask = np.zeros((self.A, self.B), dtype=np.uint8)
        self.model_applied_mask = self.base_model_applied_mask.copy()
        self.occ_matrix = result.occ.astype(np.uint8)
        self.path_occ_matrix = result.path_occ.astype(np.uint8)
        self.support_mask = result.support_mask.astype(np.uint8)
        self.unresolved_mask = np.zeros((self.A, self.B), dtype=np.uint8)
        self.user_edit_mask = np.zeros((self.A, self.B), dtype=np.uint8)
        self._gis_matrix_meta = dict(result.meta)
        self.manual_edits_since_autogen = False
        self.arrow_boxes = []
        self.outlet_cell = None
        self.outlet_cells = set()
        self.outlet_pick_mode = False
        self.matrix_arrow_view = False
        if hasattr(self, "matrix_arrow_button"):
            self.matrix_arrow_button.configure(text=TXT_MATRIX_ARROWS)
        self._clear_roi()
        self._invalidate_detection_cache()
        self._rebuild_flood_cell_mask()
        self._rebuild_basin_mask()
        self._apply_workspace_region()
        self._record_support_baseline()
        self._invalidate_matrix_arrow_cache()
        self._update_gis_cell_size_label(result.meta)
        row, col = self._default_selected_cell()
        self._set_selected_cell(row, col)
        self.reset_view()

    def _reload_gis_with_rotation(self):
        if not self._gis_source_path or self._gis_rows is None:
            raise RuntimeError("GIS 원본 정보가 없습니다.")
        try:
            requested_rows = max(2, int(self.rows_var.get()))
            if self.trim_source_shape is not None and requested_rows == int(self.A):
                requested_rows = int(self.trim_source_shape[0])
            self._gis_rows = requested_rows
        except Exception:
            pass
        if self._gis_active_records is not None and self._gis_active_bbox is not None:
            return matrix_from_loaded_polylines(
                self._gis_active_records,
                rows=int(self._gis_rows),
                bbox=self._gis_active_bbox,
                source_meta=self._gis_active_source_meta,
                rotation_degrees=self._rotation_degrees(),
            )
        return load_gis_matrix(
            self._gis_source_path,
            rows=int(self._gis_rows),
            start_height_field=self._gis_start_height_field,
            end_height_field=self._gis_end_height_field,
            rotation_degrees=self._rotation_degrees(),
            boundary_shp_path=self._gis_boundary_path,
            boundary_field=self._gis_boundary_field,
            boundary_value=self._gis_boundary_value,
        )

    @guarded_action
    def apply_rotation(self) -> None:
        if self.source_kind not in {"image", "gis"}:
            self.status_var.set("회전할 이미지 또는 GIS 파일을 먼저 불러오세요.")
            return
        if not self._confirm_rotation_reset():
            return
        try:
            angle = self._rotation_degrees()
            self.rotation_var.set(angle)
            self._save_settings()
            if self.source_kind == "gis":
                result = self._reload_gis_with_rotation()
                self._apply_gis_result_to_state(result)
                nonzero = int(np.count_nonzero(self.current_matrix)) if self.current_matrix is not None else 0
                self.status_var.set(f"GIS 회전 적용 완료: {angle:.1f}° | A={self.A}, B={self.B}, 관로 셀 {nonzero}개.")
            else:
                self._reload_image_with_rotation()
                self.status_var.set(f"이미지 회전 적용 완료: {angle:.1f}° | A={self.A}, B={self.B}. 자동인식을 다시 실행하세요.")
            self._update_matrix_preview()
            self._queue_render(high_quality=True, delay=0)
            self.canvas.focus_set()
        except Exception as exc:
            messagebox.showerror("회전 적용 오류", str(exc), parent=self)

    @guarded_action
    def reset_rotation(self) -> None:
        self.rotation_var.set(0.0)
        if self.source_kind in {"image", "gis"}:
            self.apply_rotation()
        else:
            self._save_settings()
            self.status_var.set("회전 각도를 0°로 초기화했습니다.")

    def _clear_compressed_grid_shape(self) -> None:
        self.compressed_grid_shape = None
        self.compression_source_shape = None

    def _clear_trimmed_grid_state(self) -> None:
        self.trimmed_grid_shape = None
        self.trim_source_shape = None
        self.trim_bounds = None
        self._gis_matrix_render_bbox = None
        self._gis_matrix_meta = None

    def _sync_grid_shape(self) -> None:
        if self.img_bgr is None:
            return
        prev_shape = (self.A, self.B)
        if self.compressed_grid_shape is not None:
            self.A, self.B = self.compressed_grid_shape
            self.rows_var.set(self.A)
        elif self.trimmed_grid_shape is not None:
            self.A, self.B = self.trimmed_grid_shape
            self.rows_var.set(self.A)
        elif vars(self).get("source_kind") == "gis" and isinstance(vars(self).get("current_matrix"), np.ndarray) and int(self.rows_var.get()) == self.current_matrix.shape[0]:
            self.A, self.B = self.current_matrix.shape
        else:
            self.A = max(2, int(self.rows_var.get()))
            h, w = self.img_bgr.shape[:2]
            self.B = derive_grid_B(self.A, w, h)
        if self.roi_polygon_points and prev_shape != (self.A, self.B):
            self.roi_cell_mask = self._roi_cell_mask_from_polygon()

    def _ensure_grid(self) -> bool:
        # A row-count edit must not silently replace a GIS network with zeros.
        # Only explicit vector regridding may commit the new GIS resolution.
        if vars(self).get("source_kind") == "gis" and isinstance(vars(self).get("current_matrix"), np.ndarray):
            try:
                requested_rows = int(self.rows_var.get())
            except Exception:
                self.status_var.set("행 수를 정수로 입력한 뒤 [GIS 방향 보정]을 눌러 주세요.")
                return False
            if requested_rows != self.current_matrix.shape[0]:
                self._on_requested_rows_changed()
                return False
        if self.img_bgr is None:
            messagebox.showwarning("\uc774\ubbf8\uc9c0 \uc5c6\uc74c", "\uba3c\uc800 \ub3c4\uba74 \uc774\ubbf8\uc9c0\ub97c \ubd88\ub7ec\uc640 \uc8fc\uc138\uc694.")
            return False
        self._sync_grid_shape()
        if self.base_matrix is None or self.base_matrix.shape != (self.A, self.B):
            self.base_matrix = np.zeros((self.A, self.B), dtype=np.uint8)
        if self.current_matrix is None or self.current_matrix.shape != (self.A, self.B):
            self.current_matrix = self.base_matrix.copy()
        if self.base_confidence_matrix is None or self.base_confidence_matrix.shape != (self.A, self.B):
            self.base_confidence_matrix = np.zeros((self.A, self.B), dtype=np.float32)
        if self.confidence_matrix is None or self.confidence_matrix.shape != (self.A, self.B):
            self.confidence_matrix = self.base_confidence_matrix.copy()
        if self.base_model_applied_mask is None or self.base_model_applied_mask.shape != (self.A, self.B):
            self.base_model_applied_mask = np.zeros((self.A, self.B), dtype=np.uint8)
        if self.model_applied_mask is None or self.model_applied_mask.shape != (self.A, self.B):
            self.model_applied_mask = self.base_model_applied_mask.copy()
        if self.occ_matrix is None or self.occ_matrix.shape != (self.A, self.B):
            self.occ_matrix = np.zeros((self.A, self.B), dtype=np.uint8)
        if self.path_occ_matrix is None or self.path_occ_matrix.shape != (self.A, self.B):
            self.path_occ_matrix = np.zeros((self.A, self.B), dtype=np.uint8)
        if self.support_mask is None or self.support_mask.shape != (self.A, self.B):
            self.support_mask = np.zeros((self.A, self.B), dtype=np.uint8)
        if self.unresolved_mask is None or self.unresolved_mask.shape != (self.A, self.B):
            self.unresolved_mask = np.zeros((self.A, self.B), dtype=np.uint8)
        if self.user_edit_mask is None or self.user_edit_mask.shape != (self.A, self.B):
            self.user_edit_mask = np.zeros((self.A, self.B), dtype=np.uint8)
        flood_mask = vars(self).get("_flood_cell_mask")
        if isinstance(flood_mask, np.ndarray) and flood_mask.shape != (self.A, self.B):
            self._flood_cell_mask = None
            self._update_flood_region_label()
        if self.outlet_cell is not None:
            oi, oj = self.outlet_cell
            if not (0 <= oi < self.A and 0 <= oj < self.B):
                self.outlet_cell = None
        self.outlet_cells = self._active_outlet_cells()
        basin = vars(self).get("_basin_cell_mask")
        if isinstance(basin, np.ndarray) and basin.shape != (self.A, self.B):
            if vars(self).get("_basin_ring_groups"):
                self._rebuild_basin_mask()
            else:
                self._basin_cell_mask = resample_mask_centers(basin, (self.A, self.B))
        if self.selected_cell is None:
            self.selected_cell = (0, 0)
        return True

    def _reset_manual_edit_state(self) -> None:
        self.manual_edits_since_autogen = False
        if self.current_matrix is None:
            self.user_edit_mask = None
            return
        self.user_edit_mask = np.zeros_like(self.current_matrix, dtype=np.uint8)

    def _is_compressed_mode(self) -> bool:
        return self.compressed_grid_shape is not None

    def _matrix_extent_masks(self) -> List[np.ndarray]:
        if self.current_matrix is None:
            return []
        masks: List[np.ndarray] = []
        for name in (
            "occ_matrix",
            "path_occ_matrix",
            "support_mask",
            "unresolved_mask",
            "user_edit_mask",
            "model_applied_mask",
            "_flood_cell_mask",
        ):
            value = vars(self).get(name)
            if isinstance(value, np.ndarray) and value.shape == self.current_matrix.shape:
                masks.append(value)
        region = self._effective_basin_mask()
        if region is not None:
            masks.append(region)
        return masks

    def _current_trim_bounds(self, padding: int = 1) -> Optional[Tuple[int, int, int, int]]:
        if self.current_matrix is None or self.current_matrix.size == 0:
            return None
        return outer_zero_margin_bounds(
            self.current_matrix,
            padding=padding,
            support_masks=self._matrix_extent_masks(),
            protected_cells=sorted(self._active_outlet_cells()),
        )

    def _warn_learning_locked(self) -> None:
        messagebox.showinfo(
            "압축 행렬",
            "행렬 압축 이후에는 셀 학습을 비활성화합니다.\n원본 격자 기준으로 다시 자동 인식 또는 학습 모델 적용을 실행한 뒤 수정/학습해 주세요.",
            parent=self,
        )

    def _update_action_states(self) -> None:
        self._sync_workspace_task_controls()
        if vars(self).get("_task_busy") or vars(self).get("_plena_starting"):
            return
        trim_bounds = self._current_trim_bounds(padding=1)
        can_trim = (
            trim_bounds is not None
            and trim_bounds != (0, int(self.A), 0, int(self.B))
        )
        can_compress = (
            self.current_matrix is not None
            and self.current_matrix.size > 0
            and int(np.count_nonzero(self.current_matrix)) > 0
            and (self.A > 2 or self.B > 2)
        )
        self.trim_button.configure(state=tk.NORMAL if can_trim else tk.DISABLED)
        self.compress_button.configure(state=tk.NORMAL if can_compress else tk.DISABLED)
        learning_enabled = (
            RUNTIME_TRAINING_ENABLED
            and not self._is_compressed_mode()
            and vars(self).get("source_kind") != "gis_project"
        )
        self.save_button.configure(state=tk.NORMAL if learning_enabled else tk.DISABLED)
        self.train_button.configure(state=tk.NORMAL if learning_enabled else tk.DISABLED)

    def _default_selected_cell(self) -> Tuple[int, int]:
        if self.current_matrix is None:
            return (0, 0)
        nonzero = np.argwhere(self.current_matrix > 0)
        if len(nonzero) > 0:
            return tuple(int(v) for v in nonzero[0])
        occ = np.argwhere(self.occ_matrix > 0) if self.occ_matrix is not None else []
        if len(occ) > 0:
            return tuple(int(v) for v in occ[0])
        edit_mask = self._active_edit_mask()
        if edit_mask is not None:
            roi_cells = np.argwhere(edit_mask > 0)
            if len(roi_cells) > 0:
                return tuple(int(v) for v in roi_cells[0])
        return (0, 0)

    def _set_selected_cell(self, row: int, col: int) -> None:
        if self.current_matrix is None:
            self.selected_cell = None
            self.selected_var.set(TXT_NO_SELECTION)
            return
        row = int(np.clip(row, 0, self.A - 1))
        col = int(np.clip(col, 0, self.B - 1))
        self.selected_cell = (row, col)
        value = int(self.current_matrix[row, col])
        self.selected_var.set(f"\uc120\ud0dd \uc140: ({row}, {col}) | \ud604\uc7ac\uac12: {value} ({DIR_KR[value]})")

    @guarded_action
    def load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="\ub3c4\uba74 \uc774\ubbf8\uc9c0 \uc120\ud0dd",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            original_bgr = self._read_image(path)
            self.img_path = path
            self.source_kind = "image"
            self._source_image_bgr = original_bgr.copy()
            self._gis_source_path = None
            self._gis_rows = None
            self._gis_start_height_field = None
            self._gis_end_height_field = None
            self._gis_boundary_path = None
            self._gis_boundary_field = None
            self._gis_boundary_value = None
            self._set_display_image_from_bgr(self._rotate_bgr_image(original_bgr, self._rotation_degrees()))
            self._reset_image_matrix_state()
            self.reset_view()
            self.status_var.set(
                f"\uc774\ubbf8\uc9c0 \ub85c\ub4dc \uc644\ub8cc: {os.path.basename(path)} | A={self.A}, B={self.B}, \ud68c\uc804={self._rotation_degrees():.1f}\u00b0. \uc790\ub3d9 \uc778\uc2dd\uc744 \uc2e4\ud589\ud558\uac70\ub098 \ud0a4\ubcf4\ub4dc\ub85c \uc9c1\uc811 \uc785\ub825\ud558\uc138\uc694."
            )
            self._queue_render(high_quality=True, delay=0)
            self._update_matrix_preview()
            self.canvas.focus_set()
        except Exception as exc:
            messagebox.showerror("\ubd88\ub7ec\uc624\uae30 \uc624\ub958", str(exc))

    def _ask_gis_height_fields(self, fields: List[str]) -> Optional[Tuple[Optional[str], Optional[str]]]:
        if not fields:
            return None, None
        no_field = "(사용 안 함)"
        options = [no_field] + list(fields)
        default_start = DEFAULT_START_HEIGHT_FIELD if DEFAULT_START_HEIGHT_FIELD in fields else no_field
        default_end = DEFAULT_END_HEIGHT_FIELD if DEFAULT_END_HEIGHT_FIELD in fields else no_field
        result: Dict[str, Optional[Tuple[Optional[str], Optional[str]]]] = {"value": None}

        window = tk.Toplevel(self)
        window.title("GIS 표고 필드 선택")
        window.resizable(False, False)
        if self._icon_photo is not None:
            try:
                window.iconphoto(True, self._icon_photo)
            except Exception:
                pass
        frame = ttk.Frame(window, padding=14)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="시작 표고").grid(row=0, column=0, sticky=tk.W, padx=(0, 8), pady=6)
        start_var = tk.StringVar(value=default_start)
        start_combo = ttk.Combobox(frame, textvariable=start_var, values=options, state="readonly", width=24)
        start_combo.grid(row=0, column=1, sticky=tk.W, pady=6)
        ttk.Label(frame, text="끝 표고").grid(row=1, column=0, sticky=tk.W, padx=(0, 8), pady=6)
        end_var = tk.StringVar(value=default_end)
        end_combo = ttk.Combobox(frame, textvariable=end_var, values=options, state="readonly", width=24)
        end_combo.grid(row=1, column=1, sticky=tk.W, pady=6)
        ttk.Label(frame, text="표고 필드를 사용하지 않으면 SHP 좌표 순서대로 방향을 생성합니다.").grid(
            row=2,
            column=0,
            columnspan=2,
            sticky=tk.W,
            pady=(4, 10),
        )
        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=2, sticky=tk.E)

        def normalize(value: str) -> Optional[str]:
            return None if value == no_field else value

        def on_ok() -> None:
            result["value"] = (normalize(start_var.get()), normalize(end_var.get()))
            window.destroy()

        def on_cancel() -> None:
            result["value"] = None
            window.destroy()

        ttk.Button(buttons, text="확인", command=on_ok).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(buttons, text="취소", command=on_cancel).pack(side=tk.LEFT)
        window.protocol("WM_DELETE_WINDOW", on_cancel)
        window.transient(self)
        window.grab_set()
        start_combo.focus_set()
        self.wait_window(window)
        return result["value"]

    def _ask_choice(self, title: str, label: str, options: List[str], initial: Optional[str] = None) -> Optional[str]:
        values = [str(value) for value in options if str(value)]
        if not values:
            return None
        result: Dict[str, Optional[str]] = {"value": None}
        window = tk.Toplevel(self)
        window.title(title)
        window.resizable(False, False)
        frame = ttk.Frame(window, padding=14)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text=label).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 6))
        selected = tk.StringVar(value=initial if initial in values else values[0])
        combo = ttk.Combobox(frame, textvariable=selected, values=values, state="readonly", width=44)
        combo.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(0, 12))

        def on_ok() -> None:
            result["value"] = selected.get()
            window.destroy()

        ttk.Button(frame, text="확인", command=on_ok).grid(row=2, column=0, sticky=tk.E, padx=(0, 6))
        ttk.Button(frame, text="취소", command=window.destroy).grid(row=2, column=1, sticky=tk.W)
        window.protocol("WM_DELETE_WINDOW", window.destroy)
        window.transient(self)
        window.grab_set()
        combo.focus_set()
        self.wait_window(window)
        return result["value"]

    def _ask_gis_rows(self) -> Optional[int]:
        try:
            current_rows = int(self.rows_var.get())
        except Exception:
            current_rows = int(self.A)
        initial_rows = 50 if self.img_bgr is None and current_rows == 12 else max(2, current_rows)
        return simpledialog.askinteger(
            "GIS 격자 행 개수",
            "GIS 관로를 변환할 행 개수 A를 입력해 주세요.",
            initialvalue=initial_rows,
            minvalue=2,
            maxvalue=300,
            parent=self,
        )

    def _selected_gis_layer_id(self) -> Optional[str]:
        if not hasattr(self, "layer_tree"):
            return None
        selected = self.layer_tree.selection()
        return str(selected[0]) if selected else None

    def _active_flood_layer_ids(self) -> set[str]:
        layer_ids = set(vars(self).get("_gis_flood_layer_ids") or set())
        legacy_id = vars(self).get("_gis_flood_layer_id")
        if legacy_id:
            layer_ids.add(str(legacy_id))
        return layer_ids

    def _sync_legacy_flood_layer_id(self) -> None:
        active = set(vars(self).get("_gis_flood_layer_ids") or set())
        self._gis_flood_layer_ids = active
        self._gis_flood_layer_id = next(iter(sorted(active)), None)

    def _gis_layer_role(self, layer_id: str) -> str:
        roles: List[str] = []
        if layer_id == vars(self).get("_gis_pipe_layer_id"):
            roles.append("관로")
        if layer_id == vars(self).get("_gis_boundary_layer_id"):
            roles.append("경계")
        if layer_id in self._active_flood_layer_ids():
            roles.append("침수")
        return "/".join(roles)

    @staticmethod
    def _apply_non_role_default_opacity(style: Dict[str, object], kind: str, is_role: bool) -> Dict[str, object]:
        normalized = dict(style)
        if is_role:
            return normalized
        if kind == "boundary":
            normalized["fill_opacity"] = 0.5
            normalized["outline_opacity"] = 0.5
        else:
            normalized["opacity"] = 0.5
        return normalized

    @staticmethod
    def _looks_like_flood_layer_name(name: str) -> bool:
        normalized = str(name).strip().lower()
        return any(token in normalized for token in ("침수", "범람", "flood", "inundation", "flooding"))

    def _sync_default_role_opacities(self) -> None:
        flood_layer_ids = self._active_flood_layer_ids()
        for index, (layer_id, layer) in enumerate(self._gis_layers.items()):
            if bool(layer.get("style_user_modified", False)):
                continue
            is_role = layer_id in {
                vars(self).get("_gis_pipe_layer_id"),
                vars(self).get("_gis_boundary_layer_id"),
            } or layer_id in flood_layer_ids
            default_style = (
                self._default_flood_layer_style()
                if layer_id in flood_layer_ids and layer.get("kind") == "boundary"
                else self._default_gis_layer_style(str(layer["kind"]), index)
            )
            layer["style"] = self._apply_non_role_default_opacity(default_style, str(layer["kind"]), is_role)

    @staticmethod
    def _default_gis_layer_style(kind: str, index: int) -> Dict[str, object]:
        polyline_colors = [(35, 85, 210), (65, 150, 55), (150, 70, 170), (215, 130, 45)]
        boundary_colors = [(210, 145, 85), (95, 165, 110), (70, 130, 175), (175, 95, 145)]
        palette = polyline_colors if kind == "polyline" else boundary_colors
        color = tuple(int(value) for value in palette[index % len(palette)])
        if kind == "boundary":
            return {
                "fill_color": color,
                "fill_opacity": 0.22,
                "outline_color": tuple(max(0, int(value) - 55) for value in color),
                "outline_opacity": 0.95,
                "line_width": 1.0,
            }
        return {"color": color, "opacity": 0.95, "line_width": 1.0}

    @staticmethod
    def _default_flood_layer_style() -> Dict[str, object]:
        return {
            "fill_color": (255, 150, 80),
            "fill_opacity": 0.38,
            "outline_color": (255, 75, 25),
            "outline_opacity": 0.98,
            "line_width": 1.2,
        }

    @classmethod
    def _normalized_gis_layer_style(cls, layer: Dict[str, object], index: int = 0) -> Dict[str, object]:
        kind = str(layer.get("kind", ""))
        defaults = cls._default_gis_layer_style(kind, index)
        style = dict(layer.get("style") or {})
        if kind == "boundary":
            legacy_color = style.get("color")
            legacy_opacity = style.get("opacity")
            return {
                "fill_color": tuple(style.get("fill_color", legacy_color or defaults["fill_color"])),
                "fill_opacity": float(style.get("fill_opacity", legacy_opacity if legacy_opacity is not None else defaults["fill_opacity"])),
                "outline_color": tuple(style.get("outline_color", legacy_color or defaults["outline_color"])),
                "outline_opacity": float(style.get("outline_opacity", defaults["outline_opacity"])),
                "line_width": float(style.get("line_width", defaults["line_width"])),
            }
        return {
            "color": tuple(style.get("color", defaults["color"])),
            "opacity": float(style.get("opacity", defaults["opacity"])),
            "line_width": float(style.get("line_width", defaults["line_width"])),
        }

    @staticmethod
    def _gis_style_color(layer: Dict[str, object], style: Dict[str, object]) -> Tuple[int, int, int]:
        key = "outline_color" if layer.get("kind") == "boundary" else "color"
        return tuple(int(value) for value in style[key])

    @staticmethod
    def _bgr_to_hex(color: Tuple[int, int, int]) -> str:
        blue, green, red = (int(np.clip(value, 0, 255)) for value in color)
        return f"#{red:02x}{green:02x}{blue:02x}"

    @staticmethod
    def _hex_to_bgr(color: str) -> Tuple[int, int, int]:
        value = str(color).lstrip("#")
        if len(value) != 6:
            raise ValueError("색상 값이 올바르지 않습니다.")
        return int(value[4:6], 16), int(value[2:4], 16), int(value[0:2], 16)

    @staticmethod
    def _gis_layer_crs_ready(layer: Dict[str, object]) -> bool:
        return bool(layer.get("crs_ready", True))

    def _gis_layer_edit_allowed(self) -> bool:
        if bool(vars(self).get("_gis_crs_job_polling", False)):
            self.status_var.set("지도 좌표 변환이 끝난 뒤 레이어 설정을 변경해 주세요.")
            return False
        return True

    def _gis_project_reference_bbox(self, exclude_layer_id: Optional[str] = None) -> Optional[Tuple[float, float, float, float]]:
        bboxes: List[Tuple[float, float, float, float]] = []
        for layer_id, layer in self._gis_layers.items():
            if layer_id == exclude_layer_id or not self._gis_layer_crs_ready(layer):
                continue
            try:
                bbox = tuple(float(value) for value in layer["bbox"])
            except (KeyError, TypeError, ValueError):
                continue
            if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
                continue
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                continue
            bboxes.append(bbox)
        if bboxes:
            return tuple(float(value) for value in combine_bboxes(bboxes))
        map_bbox = vars(self).get("_gis_map_bbox")
        if isinstance(map_bbox, tuple) and len(map_bbox) == 4:
            return tuple(float(value) for value in map_bbox)
        return None

    def _update_project_crs_label(self) -> None:
        label_var = vars(self).get("gis_project_crs_var")
        if label_var is None:
            return
        project_crs = vars(self).get("_gis_project_crs")
        if isinstance(project_crs, CRSDefinition):
            unit = f" / {project_crs.unit_name}" if project_crs.unit_name else ""
            label_var.set(f"전체 지도 좌표: {project_crs.label}{unit}")
        else:
            label_var.set("전체 지도 좌표: 자동/미지정")

    def _refresh_gis_layer_tree(self) -> None:
        if not hasattr(self, "layer_tree"):
            return
        selected = self._selected_gis_layer_id()
        for item in self.layer_tree.get_children():
            self.layer_tree.delete(item)
        for index, (layer_id, layer) in enumerate(self._gis_layers.items()):
            style = self._normalized_gis_layer_style(layer, index)
            style_tag = f"style_{layer_id}"
            self.layer_tree.tag_configure(style_tag, foreground=self._bgr_to_hex(self._gis_style_color(layer, style)))
            self.layer_tree.insert(
                "",
                tk.END,
                iid=layer_id,
                text=str(layer["name"]),
                tags=(style_tag,),
                values=(
                    (
                        "좌표 필요"
                        if bool(layer.get("visible", True)) and not self._gis_layer_crs_ready(layer)
                        else ("켜짐" if bool(layer.get("visible", True)) else "꺼짐")
                    ),
                    "관로" if layer.get("kind") == "polyline" else "경계",
                    self._gis_layer_role(layer_id),
                    layer_crs_display(layer),
                ),
            )
        if selected in self._gis_layers:
            self.layer_tree.selection_set(selected)
        self._update_project_crs_label()

    def _update_flood_region_label(self) -> None:
        cell_mask = vars(self).get("_flood_cell_mask")
        cell_count = int(np.count_nonzero(cell_mask)) if isinstance(cell_mask, np.ndarray) else 0
        label_var = vars(self).get("flood_region_var")
        if label_var is None:
            return
        layers = [
            self._gis_layers[layer_id]
            for layer_id in sorted(self._active_flood_layer_ids())
            if layer_id in self._gis_layers
        ]
        if not layers:
            label_var.set("침수지역 레이어: 미지정")
            return
        ready_layers = [layer for layer in layers if self._gis_layer_crs_ready(layer)]
        unresolved = len(layers) - len(ready_layers)
        feature_count = sum(len(layer.get("geometry", [])) for layer in ready_layers)
        suffix = f" / 중첩 관로 셀 {cell_count}개" if cell_count else ""
        crs_suffix = f" / 좌표 선택 필요 {unresolved}개" if unresolved else ""
        label_var.set(f"침수지역: {len(layers)}개 레이어 / {feature_count}개 영역{suffix}{crs_suffix}")

    def refresh_flood_overlap(self) -> None:
        if not self._active_flood_layer_ids():
            messagebox.showwarning("침수지역 레이어", "별도로 불러온 Polygon 침수지역 레이어를 하나 이상 지정해 주세요.", parent=self)
            return
        if not any(
            self._gis_layer_crs_ready(self._gis_layers[layer_id])
            for layer_id in self._active_flood_layer_ids()
            if layer_id in self._gis_layers
        ):
            messagebox.showwarning("침수지역 지도 좌표 필요", "침수지역 레이어의 지도 좌표를 먼저 선택해 주세요.", parent=self)
            return
        if self.source_kind != "gis" or self.current_matrix is None:
            self._update_flood_region_label()
            self.status_var.set("침수지역 레이어가 지정되었습니다. 경계 Clip 또는 LASSO 행렬 생성 후 중첩 관로 셀이 자동 표시됩니다.")
            return
        self._rebuild_flood_cell_mask()
        self._invalidate_matrix_arrow_cache()
        self._queue_render(high_quality=True, delay=0)
        self._update_matrix_preview()
        count = int(np.count_nonzero(self._flood_cell_mask)) if isinstance(self._flood_cell_mask, np.ndarray) else 0
        self.status_var.set(f"침수지역 중첩 갱신 완료: 관로 셀 {count}개")

    def clear_flood_layer(self) -> None:
        self._gis_flood_layer_ids = set()
        self._gis_flood_layer_id = None
        self._flood_cell_mask = None
        self._sync_default_role_opacities()
        self._invalidate_gis_render_cache()
        self._invalidate_matrix_arrow_cache()
        self._update_flood_region_label()
        self._refresh_gis_layer_tree()
        if self.source_kind == "gis_project":
            self._render_gis_project(reset_matrix=False)
        elif self.current_matrix is not None:
            self._queue_render(high_quality=True, delay=0)
        self._update_matrix_preview()
        self.status_var.set("모든 침수지역 레이어 지정을 해제했습니다.")

    def _rebuild_flood_cell_mask(self) -> None:
        self._flood_cell_mask = None
        bbox = vars(self).get("_gis_active_bbox")
        layers = [
            self._gis_layers[layer_id]
            for layer_id in sorted(self._active_flood_layer_ids())
            if layer_id in self._gis_layers and self._gis_layer_crs_ready(self._gis_layers[layer_id])
        ]
        if not layers or bbox is None or self.current_matrix is None:
            self._update_flood_region_label()
            return
        ring_groups = [
            [ring for ring in boundary.rings if len(ring) >= 3]
            for layer in layers
            for boundary in layer["geometry"]
        ]
        ring_groups = [rings for rings in ring_groups if rings]
        if ring_groups:
            trim_source_shape = vars(self).get("trim_source_shape")
            trim_bounds = vars(self).get("trim_bounds")
            if trim_source_shape is not None and trim_bounds is not None:
                source_rows, source_cols = (int(value) for value in trim_source_shape)
                row_start, row_stop, col_start, col_stop = (int(value) for value in trim_bounds)
                full_region_mask = boundary_grid_mask(
                    ring_groups,
                    source_rows,
                    source_cols,
                    tuple(bbox),
                    rotation_degrees=self._rotation_degrees(),
                )
                region_mask = full_region_mask[row_start:row_stop, col_start:col_stop].copy()
                if region_mask.shape != self.current_matrix.shape:
                    region_mask = self._resize_binary_mask_any(region_mask, int(self.A), int(self.B))
            else:
                region_mask = boundary_grid_mask(
                    ring_groups,
                    int(self.A),
                    int(self.B),
                    tuple(bbox),
                    rotation_degrees=self._rotation_degrees(),
                )
            support_value = vars(self).get("support_mask")
            support = (
                support_value
                if isinstance(support_value, np.ndarray) and support_value.shape == self.current_matrix.shape
                else (self.current_matrix > 0).astype(np.uint8)
            )
            self._flood_cell_mask = (region_mask & (support > 0)).astype(np.uint8)
        self._update_flood_region_label()

    @staticmethod
    def _resize_binary_mask_any(mask: np.ndarray, rows: int, cols: int) -> np.ndarray:
        source = np.asarray(mask, dtype=np.uint8)
        if source.ndim != 2:
            raise ValueError("binary mask must be 2-dimensional")
        resized = cv2.resize(source.astype(np.float32), (max(1, int(cols)), max(1, int(rows))), interpolation=cv2.INTER_AREA)
        return (resized > 0).astype(np.uint8)

    def _move_selected_gis_layer(self, direction: int) -> None:
        if not self._gis_layer_edit_allowed():
            return
        layer_id = self._selected_gis_layer_id()
        keys = list(self._gis_layers.keys())
        if layer_id not in keys:
            return
        current = keys.index(layer_id)
        target = int(np.clip(current + int(direction), 0, len(keys) - 1))
        if target == current:
            return
        if not self._confirm_gis_map_reset():
            return
        keys.insert(target, keys.pop(current))
        self._gis_layers = {key: self._gis_layers[key] for key in keys}
        self._invalidate_gis_render_cache()
        self._refresh_gis_layer_tree()
        self.layer_tree.selection_set(layer_id)
        if self.source_kind == "gis_project":
            self._render_gis_project(reset_matrix=False)

    def configure_selected_layer_style(self) -> None:
        if not self._gis_layer_edit_allowed():
            return
        layer_id = self._selected_gis_layer_id()
        layer = self._gis_layers.get(layer_id or "")
        if layer is None:
            messagebox.showwarning("레이어 선택", "색상과 투명도를 조정할 GIS 레이어를 선택해 주세요.", parent=self)
            return
        style = self._normalized_gis_layer_style(layer)
        is_boundary = layer.get("kind") == "boundary"
        colors: Dict[str, str] = {}
        if is_boundary:
            colors["fill_color"] = self._bgr_to_hex(tuple(style["fill_color"]))
            colors["outline_color"] = self._bgr_to_hex(tuple(style["outline_color"]))
        else:
            colors["color"] = self._bgr_to_hex(tuple(style["color"]))
        fill_opacity_var = tk.DoubleVar(value=float(style.get("fill_opacity", 1.0)) * 100.0)
        outline_opacity_var = tk.DoubleVar(value=float(style.get("outline_opacity", 1.0)) * 100.0)
        opacity_var = tk.DoubleVar(value=float(style.get("opacity", 1.0)) * 100.0)
        line_width_var = tk.DoubleVar(value=float(style.get("line_width", 1.8)))

        window = tk.Toplevel(self)
        window.title(f"레이어 스타일 - {layer['name']}")
        window.resizable(False, False)
        panel = ttk.Frame(window, padding=14)
        panel.pack(fill=tk.BOTH, expand=True)

        def add_color_row(row: int, label: str, key: str) -> None:
            ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w", pady=(0, 8))
            swatch = tk.Label(panel, width=6, height=1, background=colors[key], relief=tk.SOLID, borderwidth=1)
            swatch.grid(row=row, column=1, sticky="w", padx=(8, 8), pady=(0, 8))

            def choose_color() -> None:
                _rgb, color = colorchooser.askcolor(color=colors[key], parent=window, title=label)
                if color:
                    colors[key] = str(color)
                    swatch.configure(background=color)

            ttk.Button(panel, text="선택", command=choose_color).grid(row=row, column=2, sticky="ew", pady=(0, 8))

        def add_opacity_row(row: int, label: str, variable: tk.DoubleVar) -> None:
            ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w", pady=(0, 8))
            ttk.Scale(panel, from_=0.0, to=100.0, variable=variable, orient=tk.HORIZONTAL, length=220).grid(
                row=row,
                column=1,
                sticky="ew",
                padx=(8, 8),
                pady=(0, 8),
            )
            value_label = ttk.Label(panel, width=5, anchor=tk.E)
            value_label.grid(row=row, column=2, sticky="e", pady=(0, 8))

            def update_value(*_args: object) -> None:
                value_label.configure(text=f"{int(round(variable.get()))}%")

            variable.trace_add("write", update_value)
            update_value()

        row = 0
        if is_boundary:
            add_color_row(row, "채움색", "fill_color")
            row += 1
            add_opacity_row(row, "채움 불투명도", fill_opacity_var)
            row += 1
            add_color_row(row, "테두리색", "outline_color")
            row += 1
            add_opacity_row(row, "테두리 불투명도", outline_opacity_var)
            row += 1
        else:
            add_color_row(row, "선 색상", "color")
            row += 1
            add_opacity_row(row, "선 불투명도", opacity_var)
            row += 1
        ttk.Label(panel, text="선 굵기").grid(row=row, column=0, sticky="w", pady=(0, 8))
        ttk.Spinbox(
            panel,
            from_=0.5,
            to=2.0,
            increment=0.1,
            textvariable=line_width_var,
            width=8,
            format="%.1f",
        ).grid(row=row, column=1, sticky="w", padx=(8, 8), pady=(0, 8))
        ttk.Label(panel, text="확대·축소 시 탄력 적용").grid(row=row, column=2, sticky="e", pady=(0, 8))
        row += 1
        buttons = ttk.Frame(panel)
        buttons.grid(row=row, column=0, columnspan=3, sticky="e")

        def apply_style() -> None:
            if is_boundary:
                layer["style"] = {
                    "fill_color": self._hex_to_bgr(colors["fill_color"]),
                    "fill_opacity": float(np.clip(fill_opacity_var.get() / 100.0, 0.0, 1.0)),
                    "outline_color": self._hex_to_bgr(colors["outline_color"]),
                    "outline_opacity": float(np.clip(outline_opacity_var.get() / 100.0, 0.0, 1.0)),
                    "line_width": float(np.clip(line_width_var.get(), 0.5, 2.0)),
                }
            else:
                layer["style"] = {
                    "color": self._hex_to_bgr(colors["color"]),
                    "opacity": float(np.clip(opacity_var.get() / 100.0, 0.0, 1.0)),
                    "line_width": float(np.clip(line_width_var.get(), 0.5, 2.0)),
                }
            layer["style_user_modified"] = True
            self._invalidate_gis_render_cache()
            self._refresh_gis_layer_tree()
            if self.source_kind == "gis_project":
                self._render_gis_project(reset_matrix=False)
            self.status_var.set(f"{layer['name']} 스타일 적용: 선 굵기 {line_width_var.get():.1f}")
            window.destroy()

        ttk.Button(buttons, text="취소", command=window.destroy).pack(side=tk.LEFT)
        ttk.Button(buttons, text="적용", command=apply_style).pack(side=tk.LEFT, padx=(6, 0))
        window.transient(self)
        window.grab_set()

    def configure_boundary_label_field(self) -> None:
        if not self._gis_layer_edit_allowed():
            return
        layer_id = self._selected_gis_layer_id()
        layer = self._gis_layers.get(layer_id or "")
        if layer is None or layer.get("kind") != "boundary":
            layer_id = self._gis_boundary_layer_id or ""
            layer = self._gis_layers.get(layer_id)
        if layer is None or layer.get("kind") != "boundary":
            messagebox.showwarning("경계 이름", "경계 이름 필드를 설정할 Polygon 레이어를 선택해 주세요.", parent=self)
            return
        fields = [str(field) for field in layer["meta"].get("fields", [])]
        if not fields:
            messagebox.showwarning("경계 이름", "선택한 경계 레이어에 DBF 속성 필드가 없습니다.", parent=self)
            return
        boundaries = list(layer["geometry"])
        window = tk.Toplevel(self)
        window.title(f"경계 이름 필드 - {layer['name']}")
        window.geometry("470x380")
        panel = ttk.Frame(window, padding=12)
        panel.pack(fill=tk.BOTH, expand=True)
        ttk.Label(panel, text="지도 좌측 상단과 hover에 표시할 DBF 필드를 선택하세요.").pack(fill=tk.X, pady=(0, 8))
        table = ttk.Treeview(panel, columns=("sample",), show="tree headings", selectmode="browse")
        table.heading("#0", text="필드")
        table.heading("sample", text="예시 값")
        table.column("#0", width=160, minwidth=120)
        table.column("sample", width=260, minwidth=160)
        table.pack(fill=tk.BOTH, expand=True)
        table.insert("", tk.END, iid="__auto__", text="자동 선택", values=("일반적인 행정구역 이름 필드 우선",))
        for index, field in enumerate(fields):
            sample = next(
                (
                    str(boundary.attributes.get(field, "") or "").strip()
                    for boundary in boundaries
                    if str(boundary.attributes.get(field, "") or "").strip()
                ),
                "",
            )
            table.insert("", tk.END, iid=f"field_{index}", text=field, values=(sample,))
        current = str(layer.get("label_field", "") or "")
        current_iid = "__auto__"
        if current in fields:
            current_iid = f"field_{fields.index(current)}"
        table.selection_set(current_iid)
        table.see(current_iid)
        buttons = ttk.Frame(panel)
        buttons.pack(fill=tk.X, pady=(10, 0))

        def apply_field() -> None:
            selected = table.selection()
            if not selected:
                return
            iid = str(selected[0])
            layer["label_field"] = "" if iid == "__auto__" else fields[int(iid.split("_", 1)[1])]
            label = layer["label_field"] or "자동 선택"
            self._refresh_gis_boundary_overlay()
            self.status_var.set(f"{layer['name']} 경계 이름 필드: {label}")
            window.destroy()

        ttk.Button(buttons, text="취소", command=window.destroy).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="적용", command=apply_field).pack(side=tk.RIGHT, padx=(0, 6))
        table.bind("<Double-Button-1>", lambda _event: apply_field())
        window.transient(self)
        window.grab_set()
        table.focus_set()

    def _invalidate_gis_render_cache(self) -> None:
        deferred_overview_after_id = vars(self).get("_gis_deferred_overview_after_id")
        if deferred_overview_after_id is not None:
            try:
                self.after_cancel(deferred_overview_after_id)
            except Exception:
                pass
            self._gis_deferred_overview_after_id = None
        self._gis_render_generation += 1
        self._gis_render_polling = False
        self._gis_boundary_job_generation += 1
        self._gis_boundary_job_polling = False
        self._gis_render_cache.clear()
        self._gis_last_overview_image = None
        self._gis_last_overview_bbox = None
        self._gis_last_overview_path = None
        self._gis_viewport_render_cache.clear()
        self._gis_viewport_render_generation += 1
        self._gis_viewport_render_polling = False
        self._gis_viewport_pending_keys.clear()
        if vars(self).get("source_kind") == "gis_project" and isinstance(vars(self).get("img_pil"), Image.Image):
            self._gis_force_viewport_render = True
            self._gis_sync_viewport_once = True
        self._gis_boundary_overlay_cache.clear()
        self._gis_hover_boundary = None

    def _invalidate_gis_analysis_cache(self) -> None:
        self._gis_boundary_job_generation += 1
        self._gis_boundary_job_polling = False
        self._gis_pipe_job_generation += 1
        self._gis_pipe_job_polling = False
        self._gis_boundary_result_cache.clear()
        self._gis_pipe_result_cache.clear()

    def _gis_overview_line_scale(self, render_size: int = GIS_OVERVIEW_RENDER_SIZE) -> float:
        if not hasattr(self, "canvas"):
            return 1.0
        viewport = max(int(self.canvas.winfo_width()), int(self.canvas.winfo_height()), 300)
        resolution_scale = math.sqrt(max(float(render_size) / float(viewport), 1.0))
        return float(np.clip(resolution_scale, 1.0, 1.45))

    @staticmethod
    def _gis_render_line_width(style_width: float, line_scale: float) -> float:
        return float(np.clip(float(style_width) * float(line_scale), 0.5, 2.0))

    @staticmethod
    def _bbox_intersects(first: Tuple[float, float, float, float], second: Tuple[float, float, float, float]) -> bool:
        return not (
            float(first[2]) < float(second[0])
            or float(second[2]) < float(first[0])
            or float(first[3]) < float(second[1])
            or float(second[3]) < float(first[1])
        )

    @staticmethod
    def _parts_bbox(parts: List[List[Tuple[float, float]]]) -> Tuple[float, float, float, float]:
        xs = [float(point[0]) for part in parts for point in part]
        ys = [float(point[1]) for part in parts for point in part]
        if not xs or not ys:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def _gis_layer_point_count(layer: Dict[str, object]) -> int:
        cached = layer.get("point_count")
        if cached is not None:
            return int(cached)
        if layer.get("kind") == "polyline":
            count = sum(len(part) for record in layer["geometry"] for part in record.parts)
        else:
            count = sum(len(ring) for boundary in layer["geometry"] for ring in boundary.rings)
        layer["point_count"] = int(count)
        return int(count)

    @staticmethod
    def _ensure_polyline_spatial_index(layer: Dict[str, object]) -> PolylineSpatialIndex:
        spatial_index = layer.get("spatial_index")
        if isinstance(spatial_index, PolylineSpatialIndex):
            return spatial_index
        spatial_index = PolylineSpatialIndex.build(layer["geometry"])
        layer["spatial_index"] = spatial_index
        return spatial_index

    @staticmethod
    def _ensure_boundary_feature_index(layer: Dict[str, object]) -> BoundaryFeatureIndex:
        feature_index = layer.get("feature_index")
        if isinstance(feature_index, BoundaryFeatureIndex):
            return feature_index
        feature_index = BoundaryFeatureIndex.build(layer["geometry"])
        layer["feature_index"] = feature_index
        return feature_index

    def _refresh_active_gis_render_geometry(self) -> None:
        active_records = vars(self).get("_gis_active_records")
        active_bbox = vars(self).get("_gis_active_bbox")
        if active_records is None or active_bbox is None:
            self._gis_active_render_records = None
            self._gis_active_render_bbox = None
            self._gis_active_render_spatial_index = None
            self._gis_active_render_rotation = None
            return
        angle = self._rotation_degrees()
        if (
            vars(self).get("_gis_active_render_records") is not None
            and vars(self).get("_gis_active_render_bbox") is not None
            and vars(self).get("_gis_active_render_rotation") is not None
            and abs(float(vars(self).get("_gis_active_render_rotation")) - float(angle)) <= 1e-9
        ):
            return
        render_records, render_bbox = prepare_rotated_polylines(active_records, active_bbox, angle)
        render_bbox = tuple(float(value) for value in render_bbox)
        self._gis_active_render_records = list(render_records)
        self._gis_active_render_bbox = render_bbox
        self._gis_active_render_rotation = float(angle)
        try:
            self._gis_active_render_spatial_index = PolylineSpatialIndex.build(self._gis_active_render_records)
        except Exception:
            self._gis_active_render_spatial_index = None
        viewport_cache = vars(self).setdefault("_gis_viewport_render_cache", {})
        if isinstance(viewport_cache, dict):
            viewport_cache.clear()
        self._gis_viewport_render_generation = int(vars(self).get("_gis_viewport_render_generation", 0)) + 1
        pending_keys = vars(self).setdefault("_gis_viewport_pending_keys", set())
        if isinstance(pending_keys, set):
            pending_keys.clear()

    def _schedule_gis_index_build(self, layer_ids: Optional[List[str]] = None) -> None:
        if not self._gis_layers:
            return
        ids = layer_ids if layer_ids is not None else list(self._gis_layers.keys())
        tasks: List[Tuple[str, str, int, object]] = []
        for layer_id in ids:
            layer = self._gis_layers.get(layer_id)
            if layer is None or not self._gis_layer_crs_ready(layer):
                continue
            geometry = layer.get("geometry")
            if layer.get("kind") == "polyline" and not isinstance(layer.get("spatial_index"), PolylineSpatialIndex):
                tasks.append((str(layer_id), "polyline", int(id(geometry)), geometry))
            elif layer.get("kind") == "boundary" and not isinstance(layer.get("feature_index"), BoundaryFeatureIndex):
                tasks.append((str(layer_id), "boundary", int(id(geometry)), geometry))
        if not tasks:
            return
        self._gis_index_job_generation += 1
        generation = self._gis_index_job_generation
        self._gis_index_job_polling = True
        threading.Thread(target=self._gis_index_worker, args=(generation, tasks), daemon=True).start()
        self.after(60, self._poll_gis_index_result)

    def _gis_index_worker(self, generation: int, tasks: List[Tuple[str, str, int, object]]) -> None:
        results: List[Tuple[str, str, int, object]] = []
        for layer_id, kind, geometry_id, geometry in tasks:
            try:
                if kind == "polyline":
                    index: object = PolylineSpatialIndex.build(geometry)
                else:
                    index = BoundaryFeatureIndex.build(geometry)
            except Exception as exc:
                index = exc
            results.append((layer_id, kind, geometry_id, index))
        self._gis_index_job_queue.put((generation, results))

    def _poll_gis_index_result(self) -> None:
        current: Optional[List[Tuple[str, str, int, object]]] = None
        while True:
            try:
                generation, result = self._gis_index_job_queue.get_nowait()
                if generation == self._gis_index_job_generation:
                    current = result
            except queue.Empty:
                break
        if current is None:
            if self._gis_index_job_polling:
                self.after(60, self._poll_gis_index_result)
            return
        built = 0
        for layer_id, kind, geometry_id, index in current:
            layer = self._gis_layers.get(layer_id)
            if layer is None or int(id(layer.get("geometry"))) != int(geometry_id) or isinstance(index, Exception):
                continue
            if kind == "polyline" and isinstance(index, PolylineSpatialIndex):
                layer["spatial_index"] = index
                built += 1
            elif kind == "boundary" and isinstance(index, BoundaryFeatureIndex):
                layer["feature_index"] = index
                built += 1
        self._gis_index_job_polling = False
        if built:
            self.status_var.set(f"GIS 공간 인덱스 준비 완료: {built}개 레이어")

    def _apply_gis_project_image(
        self,
        image: np.ndarray,
        bbox: Tuple[float, float, float, float],
        last_path: str,
        reset_matrix: bool,
    ) -> None:
        next_bbox = tuple(float(value) for value in bbox)
        previous_source_kind = str(vars(self).get("source_kind") or "")
        previous_bbox = vars(self).get("_gis_map_bbox")
        previous_img = vars(self).get("img_pil")
        previous_size = previous_img.size if isinstance(previous_img, Image.Image) else None
        previous_zoom = float(vars(self).get("_zoom", 1.0))
        previous_origin = tuple(vars(self).get("_display_origin", (0.0, 0.0)))
        previous_origin_initialized = bool(vars(self).get("_display_origin_initialized", False))
        preserve_view = (
            not reset_matrix
            and previous_source_kind == "gis_project"
            and previous_bbox == next_bbox
            and previous_size is not None
        )
        if vars(self).get("_gis_map_bbox") != next_bbox or vars(self).get("img_path") != str(last_path):
            viewport_cache = vars(self).get("_gis_viewport_render_cache")
            if isinstance(viewport_cache, dict):
                viewport_cache.clear()
            else:
                self._gis_viewport_render_cache = {}
        self._gis_last_overview_image = image
        self._gis_last_overview_bbox = next_bbox
        self._gis_last_overview_path = str(last_path)
        self._gis_map_bbox = next_bbox
        self.source_kind = "gis_project"
        self.img_path = str(last_path)
        self._source_image_bgr = None
        self._set_display_image_from_bgr(image)
        if reset_matrix:
            self._gis_active_records = None
            self._gis_active_bbox = None
            self._gis_active_spatial_index = None
            self._gis_active_source_meta = None
            self._gis_active_render_records = None
            self._gis_active_render_bbox = None
            self._gis_active_render_spatial_index = None
            self._gis_active_render_rotation = None
            self._reset_image_matrix_state()
        if preserve_view and self.img_pil is not None and self.img_pil.size == previous_size:
            self._zoom = previous_zoom
            self._display_origin = previous_origin
            self._display_origin_initialized = previous_origin_initialized
            self._queue_render(high_quality=False, delay=0)
            self._queue_render(high_quality=True, delay=80)
        else:
            self.reset_view()
        self._update_matrix_preview()

    def _apply_last_gis_overview_if_available(self, reset_matrix: bool) -> bool:
        image = vars(self).get("_gis_last_overview_image")
        bbox = vars(self).get("_gis_last_overview_bbox")
        path = vars(self).get("_gis_last_overview_path")
        if image is None or bbox is None or path is None:
            return False
        self._apply_gis_project_image(image, tuple(bbox), str(path), reset_matrix)
        return True

    def _world_bbox_from_image_rect(
        self,
        image_rect: Tuple[float, float, float, float],
    ) -> Optional[Tuple[float, float, float, float]]:
        return self._world_bbox_from_image_rect_for_bbox(image_rect, self._gis_map_bbox)

    def _world_bbox_from_image_rect_for_bbox(
        self,
        image_rect: Tuple[float, float, float, float],
        source_bbox: Optional[Tuple[float, float, float, float]],
    ) -> Optional[Tuple[float, float, float, float]]:
        if source_bbox is None or self.img_pil is None:
            return None
        xmin, ymin, xmax, ymax = source_bbox
        # image_rect uses PIL crop edge coordinates in [0, width] x [0, height].
        width = max(float(self.img_pil.size[0]), 1.0)
        height = max(float(self.img_pil.size[1]), 1.0)
        ix0, iy0, ix1, iy1 = (float(value) for value in image_rect)
        wx0 = xmin + ix0 / width * (xmax - xmin)
        wx1 = xmin + ix1 / width * (xmax - xmin)
        wy0 = ymax - iy0 / height * (ymax - ymin)
        wy1 = ymax - iy1 / height * (ymax - ymin)
        return (min(wx0, wx1), min(wy0, wy1), max(wx0, wx1), max(wy0, wy1))

    def _gis_boundary_candidates_for_bbox(
        self,
        layer: Dict[str, object],
        bbox: Tuple[float, float, float, float],
    ) -> List[object]:
        boundaries = list(layer["geometry"])
        feature_index = self._ensure_boundary_feature_index(layer)
        return [
            boundaries[index]
            for index, feature_bbox in enumerate(feature_index.bboxes)
            if self._bbox_intersects(tuple(feature_bbox), bbox)
        ]

    def _gis_viewport_layer_stack(
        self,
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[Tuple[str, tuple], ...]:
        visible_items = [
            (layer_id, layer)
            for layer_id, layer in self._gis_layers.items()
            if bool(layer.get("visible", True)) and self._gis_layer_crs_ready(layer)
        ]
        layer_stack: List[Tuple[str, tuple]] = []
        for index, (_layer_id, layer) in reversed(list(enumerate(visible_items))):
            style = self._normalized_gis_layer_style(layer, index)
            if layer["kind"] == "polyline":
                spatial_index = layer.get("spatial_index")
                if isinstance(spatial_index, PolylineSpatialIndex):
                    geometry = spatial_index.query(bbox)
                else:
                    geometry = [
                        record
                        for record in layer["geometry"]
                        if self._bbox_intersects(self._parts_bbox(record.parts), bbox)
                    ]
                if not geometry:
                    continue
                item = (
                    geometry,
                    tuple(int(value) for value in style["color"]),
                    float(style["opacity"]),
                    self._gis_render_line_width(float(style["line_width"]), 1.0),
                )
            else:
                geometry = self._gis_boundary_candidates_for_bbox(layer, bbox)
                if not geometry:
                    continue
                item = (
                    geometry,
                    tuple(int(value) for value in style["fill_color"]),
                    float(style["fill_opacity"]),
                    tuple(int(value) for value in style["outline_color"]),
                    float(style["outline_opacity"]),
                    self._gis_render_line_width(float(style["line_width"]), 1.0),
                )
            layer_stack.append((str(layer["kind"]), item))
        return tuple(layer_stack)

    def _active_gis_viewport_layer_stack(
        self,
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[Tuple[str, tuple], ...]:
        if vars(self).get("source_kind") != "gis" or not self._gis_active_records:
            return tuple()
        self._refresh_active_gis_render_geometry()
        render_records = vars(self).get("_gis_active_render_records") or self._gis_active_records
        render_bbox = vars(self).get("_gis_active_render_bbox") or self._gis_active_bbox
        if render_bbox is None:
            return tuple()
        spatial_index = vars(self).get("_gis_active_render_spatial_index")
        if isinstance(spatial_index, PolylineSpatialIndex):
            geometry = spatial_index.query(bbox)
        else:
            geometry = [
                record
                for record in render_records
                if self._bbox_intersects(self._parts_bbox(record.parts), bbox)
            ]
        if not geometry:
            return tuple()
        line_width = float(np.clip(1.5 / math.sqrt(max(float(self._zoom), 1.0)), 0.55, 1.5))
        return (("polyline", (geometry, (0, 0, 0), 1.0, line_width)),)

    def _gis_viewport_cache_key(
        self,
        bbox: Tuple[float, float, float, float],
        dest_size: Tuple[int, int],
        *,
        source_kind: Optional[str] = None,
    ) -> Tuple[object, ...]:
        kind = source_kind or str(vars(self).get("source_kind") or "")
        if kind == "gis_project":
            visible_items = [
                (layer_id, layer)
                for layer_id, layer in self._gis_layers.items()
                if bool(layer.get("visible", True)) and self._gis_layer_crs_ready(layer)
            ]
            style_key = tuple(
                f"{layer_id}:{repr(sorted(self._normalized_gis_layer_style(layer, index).items()))}"
                for index, (layer_id, layer) in enumerate(visible_items)
            )
        elif kind == "gis":
            style_key = (
                "active_gis",
                tuple(
                    round(float(value), 6)
                    for value in (
                        vars(self).get("_gis_matrix_render_bbox")
                        or vars(self).get("_gis_active_render_bbox")
                        or vars(self).get("_gis_active_bbox")
                        or (0.0, 0.0, 0.0, 0.0)
                    )
                ),
                round(float(self._rotation_degrees()), 6),
                int(id(vars(self).get("_gis_active_render_records") or vars(self).get("_gis_active_records"))),
            )
        else:
            style_key = tuple()
        return (
            "gis_viewport",
            kind,
            tuple(round(float(value), 6) for value in bbox),
            int(dest_size[0]),
            int(dest_size[1]),
            style_key,
        )

    def _schedule_gis_viewport_render(
        self,
        cache_key: Tuple[object, ...],
        layer_stack: Tuple[Tuple[str, tuple], ...],
        bbox: Tuple[float, float, float, float],
        dest_size: Tuple[int, int],
    ) -> None:
        if cache_key in self._gis_viewport_pending_keys:
            return
        self._gis_viewport_pending_keys.add(cache_key)
        generation = self._gis_viewport_render_generation
        dest_w, dest_h = max(1, int(dest_size[0])), max(1, int(dest_size[1]))
        max_size = int(np.clip(max(dest_w, dest_h) * GIS_VIEWPORT_RENDER_SUPERSAMPLE, 384, 3072))
        threading.Thread(
            target=self._render_gis_viewport_worker,
            args=(generation, cache_key, layer_stack, tuple(bbox), (dest_w, dest_h), max_size),
            daemon=True,
        ).start()
        if not self._gis_viewport_render_polling:
            self._gis_viewport_render_polling = True
            self.after(30, self._poll_gis_viewport_render_result)

    def _render_gis_viewport_worker(
        self,
        generation: int,
        cache_key: Tuple[object, ...],
        layer_stack: Tuple[Tuple[str, tuple], ...],
        bbox: Tuple[float, float, float, float],
        dest_size: Tuple[int, int],
        max_size: int,
    ) -> None:
        try:
            render_size = self._gis_viewport_render_size(dest_size, max_size)
            image_bgr = render_gis_layers(
                [],
                [],
                bbox,
                max_size=max_size,
                layer_stack=layer_stack,
                output_size=render_size,
            )
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            rendered = Image.fromarray(image_rgb)
            if rendered.size != dest_size:
                rendered = rendered.resize(dest_size, Image.LANCZOS)
            result: object = rendered
        except Exception as exc:
            result = exc
        self._gis_viewport_result_queue.put((generation, cache_key, result))

    @staticmethod
    def _render_gis_viewport_snapshot(
        layer_stack: Tuple[Tuple[str, tuple], ...],
        bbox: Tuple[float, float, float, float],
        dest_size: Tuple[int, int],
        max_size: int,
    ) -> Image.Image:
        render_size = NFMATApp._gis_viewport_render_size(dest_size, max_size)
        image_bgr = render_gis_layers(
            [],
            [],
            bbox,
            max_size=max_size,
            layer_stack=layer_stack,
            output_size=render_size,
        )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rendered = Image.fromarray(image_rgb)
        if rendered.size != dest_size:
            rendered = rendered.resize(dest_size, Image.LANCZOS)
        return rendered

    @staticmethod
    def _gis_viewport_render_size(dest_size: Tuple[int, int], max_size: int) -> Tuple[int, int]:
        dest_w, dest_h = max(1, int(dest_size[0])), max(1, int(dest_size[1]))
        longest = max(dest_w, dest_h)
        if longest <= 0:
            return dest_w, dest_h
        scale = float(np.clip(float(max_size) / float(longest), 1.0, GIS_VIEWPORT_RENDER_SUPERSAMPLE))
        render_w = max(1, int(round(dest_w * scale)))
        render_h = max(1, int(round(dest_h * scale)))
        return render_w, render_h

    def _poll_gis_viewport_render_result(self) -> None:
        rendered_any = False
        while True:
            try:
                generation, cache_key, result = self._gis_viewport_result_queue.get_nowait()
            except queue.Empty:
                break
            self._gis_viewport_pending_keys.discard(cache_key)
            if generation != self._gis_viewport_render_generation:
                continue
            if isinstance(result, Image.Image):
                self._cache_gis_viewport_image(cache_key, result)
                rendered_any = True
        if rendered_any:
            self._queue_render(high_quality=True, delay=0)
        if self._gis_viewport_pending_keys:
            self.after(30, self._poll_gis_viewport_render_result)
        else:
            self._gis_viewport_render_polling = False

    def _cache_gis_viewport_image(self, cache_key: Tuple[object, ...], image: Image.Image) -> None:
        self._gis_viewport_render_cache[cache_key] = image
        while len(self._gis_viewport_render_cache) > GIS_VIEWPORT_RENDER_CACHE_LIMIT:
            self._gis_viewport_render_cache.pop(next(iter(self._gis_viewport_render_cache)))

    def _render_gis_viewport_image(
        self,
        image_rect: Tuple[float, float, float, float],
        dest_size: Tuple[int, int],
    ) -> Optional[Image.Image]:
        source_kind = str(vars(self).get("source_kind") or "")
        force_viewport = source_kind == "gis_project" and bool(vars(self).get("_gis_force_viewport_render", False))
        if self.matrix_arrow_view or self.img_pil is None:
            return None
        if float(self._zoom) < GIS_VIEWPORT_RENDER_ZOOM_THRESHOLD and not force_viewport:
            return None
        dest_w, dest_h = max(1, int(dest_size[0])), max(1, int(dest_size[1]))
        if dest_w <= 8 or dest_h <= 8:
            return None
        if source_kind == "gis_project":
            source_bbox = self._gis_map_bbox
        elif source_kind == "gis":
            self._refresh_active_gis_render_geometry()
            source_bbox = (
                vars(self).get("_gis_matrix_render_bbox")
                or self._gis_active_render_bbox
                or self._gis_active_bbox
            )
        else:
            return None
        bbox = self._world_bbox_from_image_rect_for_bbox(image_rect, source_bbox)
        if bbox is None:
            return None
        cache_key = self._gis_viewport_cache_key(bbox, (dest_w, dest_h), source_kind=source_kind)
        cached = self._gis_viewport_render_cache.get(cache_key)
        if cached is not None:
            return cached
        if source_kind == "gis_project":
            layer_stack = self._gis_viewport_layer_stack(bbox)
        else:
            layer_stack = self._active_gis_viewport_layer_stack(bbox)
        if not layer_stack:
            if force_viewport:
                blank = Image.new("RGB", (dest_w, dest_h), (250, 250, 250))
                self._cache_gis_viewport_image(cache_key, blank)
                self._gis_sync_viewport_once = False
                return blank
            return None
        if force_viewport and bool(vars(self).get("_gis_sync_viewport_once", False)):
            max_size = int(np.clip(max(dest_w, dest_h), 384, GIS_INTERACTIVE_VIEWPORT_RENDER_MAX))
            self._gis_sync_viewport_once = False
            try:
                rendered = self._render_gis_viewport_snapshot(layer_stack, bbox, (dest_w, dest_h), max_size)
            except Exception:
                return None
            self._cache_gis_viewport_image(cache_key, rendered)
            return rendered
        if "_gis_viewport_pending_keys" not in vars(self) or "_gis_viewport_result_queue" not in vars(self):
            max_size = int(np.clip(max(dest_w, dest_h) * GIS_VIEWPORT_RENDER_SUPERSAMPLE, 384, 3072))
            return self._render_gis_viewport_snapshot(layer_stack, bbox, (dest_w, dest_h), max_size)
        self._schedule_gis_viewport_render(cache_key, layer_stack, bbox, (dest_w, dest_h))
        return None

    def _render_gis_project_worker(
        self,
        generation: int,
        render_key: Tuple[str, ...],
        layer_stack: List[Tuple[str, tuple]],
        bbox: Tuple[float, float, float, float],
        last_path: str,
        reset_matrix: bool,
    ) -> None:
        try:
            image: object = render_gis_layers([], [], bbox, max_size=GIS_OVERVIEW_RENDER_SIZE, layer_stack=layer_stack)
        except Exception as exc:
            image = exc
        self._gis_render_result_queue.put((generation, render_key, image, bbox, last_path, reset_matrix))

    def _poll_gis_render_result(self) -> None:
        current = None
        while True:
            try:
                result = self._gis_render_result_queue.get_nowait()
                if result[0] == self._gis_render_generation:
                    current = result
            except queue.Empty:
                break
        if current is None:
            if self._gis_render_polling:
                self.after(30, self._poll_gis_render_result)
            return
        generation, render_key, image, bbox, last_path, reset_matrix = current
        if generation == self._gis_render_generation:
            if isinstance(image, Exception):
                self._gis_render_polling = False
                messagebox.showerror("GIS 지도 렌더링 오류", str(image), parent=self)
                return
            assert isinstance(image, np.ndarray)
            self._gis_render_cache[render_key] = (image, bbox)
            while len(self._gis_render_cache) > GIS_RENDER_CACHE_LIMIT:
                self._gis_render_cache.pop(next(iter(self._gis_render_cache)))
            self._apply_gis_project_image(image, bbox, last_path, reset_matrix)
            self.status_var.set("GIS 전체 지도 렌더링 완료. 행정경계를 클릭하거나 LASSO를 사용하세요.")
            self._gis_render_polling = False
            self._gis_force_viewport_render = False
            self._gis_sync_viewport_once = False
        elif self._gis_render_polling:
            self.after(30, self._poll_gis_render_result)

    def _render_gis_project(self, *, reset_matrix: bool = True) -> None:
        visible_items = [
            (layer_id, layer)
            for layer_id, layer in self._gis_layers.items()
            if bool(layer.get("visible", True)) and self._gis_layer_crs_ready(layer)
        ]
        visible_layers = [layer for _layer_id, layer in visible_items]
        if not visible_layers:
            if vars(self).get("source_kind") == "gis_project" and isinstance(vars(self).get("img_pil"), Image.Image):
                self._gis_force_viewport_render = True
                self._gis_sync_viewport_once = True
                self._queue_render(high_quality=True, delay=0)
            self.status_var.set("표시할 GIS 레이어가 없습니다.")
            return
        line_scale = self._gis_overview_line_scale()
        render_key = (f"render_size={GIS_OVERVIEW_RENDER_SIZE}", f"line_scale={line_scale:.2f}") + tuple(
            f"{layer_id}:{repr(sorted(self._normalized_gis_layer_style(layer, index).items()))}"
            for index, (layer_id, layer) in enumerate(visible_items)
        )
        cached = self._gis_render_cache.get(render_key)
        if cached is not None:
            image, bbox = cached
            self._apply_gis_project_image(image, tuple(bbox), str(visible_layers[-1]["path"]), reset_matrix)
            return
        bbox = tuple(combine_bboxes([tuple(layer["bbox"]) for layer in visible_layers]))
        layer_stack: List[Tuple[str, tuple]] = []
        for index, layer in reversed(list(enumerate(visible_layers))):
            style = self._normalized_gis_layer_style(layer, index)
            layer_point_count = self._gis_layer_point_count(layer)
            if layer["kind"] == "polyline":
                item = (
                    layer["geometry"],
                    tuple(int(value) for value in style["color"]),
                    float(style["opacity"]),
                    self._gis_render_line_width(float(style["line_width"]), line_scale),
                    int(layer_point_count),
                )
            else:
                item = (
                    layer["geometry"],
                    tuple(int(value) for value in style["fill_color"]),
                    float(style["fill_opacity"]),
                    tuple(int(value) for value in style["outline_color"]),
                    float(style["outline_opacity"]),
                    self._gis_render_line_width(float(style["line_width"]), line_scale),
                    int(layer_point_count),
                )
            layer_stack.append((str(layer["kind"]), item))
        point_count = sum(self._gis_layer_point_count(layer) for layer in visible_layers)
        last_path = str(visible_layers[-1]["path"])
        if point_count >= GIS_BACKGROUND_RENDER_POINT_THRESHOLD:
            self._gis_render_generation += 1
            generation = self._gis_render_generation
            self._gis_render_polling = True
            self.source_kind = "gis_project"
            self._apply_last_gis_overview_if_available(reset_matrix)
            if not reset_matrix and isinstance(vars(self).get("img_pil"), Image.Image):
                self._gis_force_viewport_render = True
                self._gis_sync_viewport_once = True
                self._queue_render(high_quality=True, delay=0)
            self.status_var.set(f"대규모 GIS 지도 렌더링 중... 좌표 {point_count:,}개")
            def start_overview_render() -> None:
                self._gis_deferred_overview_after_id = None
                threading.Thread(
                    target=self._render_gis_project_worker,
                    args=(generation, render_key, layer_stack, bbox, last_path, reset_matrix),
                    daemon=True,
                ).start()
                self.after(30, self._poll_gis_render_result)

            if not reset_matrix:
                self._gis_deferred_overview_after_id = self.after(
                    GIS_INTERACTIVE_OVERVIEW_DEBOUNCE_MS,
                    start_overview_render,
                )
            else:
                start_overview_render()
            return
        image = render_gis_layers([], [], bbox, max_size=GIS_OVERVIEW_RENDER_SIZE, layer_stack=layer_stack)
        self._gis_render_cache[render_key] = (image, bbox)
        while len(self._gis_render_cache) > GIS_RENDER_CACHE_LIMIT:
            self._gis_render_cache.pop(next(iter(self._gis_render_cache)))
        self._apply_gis_project_image(image, bbox, last_path, reset_matrix)

    def _confirm_gis_map_reset(self) -> bool:
        if self.source_kind != "gis" or not self._has_rotation_reset_risk():
            return True
        return messagebox.askyesno(
            "전체 GIS 지도",
            "현재 행렬의 수정값, ROI 또는 출구 선택이 초기화됩니다.\n전체 GIS 지도로 돌아갈까요?",
            parent=self,
        )

    def show_boundary_click_map(self) -> None:
        self.return_to_full_gis_map()

    @guarded_action
    def return_to_full_gis_map(self) -> bool:
        if not self._gis_layers:
            messagebox.showwarning("GIS 레이어 없음", "먼저 관로 GIS 레이어를 불러와 주세요.", parent=self)
            return False
        if self._gis_layers.get(self._gis_pipe_layer_id or "") is None:
            messagebox.showwarning("관로 레이어 없음", "왼쪽 레이어에서 관로 레이어를 지정해 주세요.", parent=self)
            return False
        if not self._confirm_gis_map_reset():
            return False
        self._gis_render_generation += 1
        self._gis_render_polling = False
        self._gis_hover_boundary = None
        self._gis_last_clicked_boundary = None
        self._clear_roi()
        self._render_gis_project()
        if not self._gis_render_polling:
            if self._gis_layers.get(self._gis_boundary_layer_id or "") is None:
                self.status_var.set("관로만 불러온 상태입니다. [선택 관로 → 행렬]로 전체 관로를 행렬로 변환할 수 있습니다.")
            else:
                self.status_var.set("행정경계를 지도에서 클릭하면 해당 구역 관로를 즉시 행렬로 변환합니다.")
        self.canvas.focus_set()
        return True

    @guarded_action
    def add_gis_layers(self) -> None:
        if self._gis_load_in_progress:
            self.status_var.set("GIS 레이어를 불러오는 중입니다.")
            return
        if bool(vars(self).get("_gis_crs_job_polling", False)):
            self.status_var.set("지도 좌표를 변환하는 중입니다.")
            return
        if not self._confirm_gis_map_reset():
            return
        paths = filedialog.askopenfilenames(
            title="GIS SHP 레이어 불러오기",
            filetypes=[("Shapefile", "*.shp"), ("All files", "*.*")],
            parent=self,
        )
        if not paths:
            return
        self._gis_load_in_progress = True
        self.status_var.set(f"GIS 레이어 {len(paths)}개를 백그라운드에서 불러오는 중...")
        threading.Thread(
            target=self._load_gis_layers_worker,
            args=(tuple(str(path) for path in paths), vars(self).get("_gis_project_crs")),
            daemon=True,
        ).start()
        self.after(30, self._poll_gis_load_result)

    def _load_gis_layers_worker(
        self,
        paths: Tuple[str, ...],
        project_crs: Optional[CRSDefinition] = None,
    ) -> None:
        raw_layers: List[Dict[str, object]] = []
        errors: List[str] = []
        for path in paths:
            try:
                shape_info = read_shp_shape_info(path)
                kind = str(shape_info.get("kind", "unsupported"))
                if kind == "polyline":
                    geometry, bbox, meta = read_polyline_shp(path)
                elif kind == "boundary":
                    geometry, bbox, meta = read_boundary_shp(path)
                else:
                    raise ValueError(
                        f"Unsupported SHP shape_type={shape_info.get('shape_type')} "
                        f"({shape_info.get('shape_type_name')})"
                    )
                layer: Dict[str, object] = {
                    "name": Path(path).stem,
                    "path": path,
                    "kind": kind,
                    "geometry": geometry,
                    "bbox": tuple(float(value) for value in bbox),
                    "meta": dict(meta),
                    "visible": True,
                    "label_field": "",
                }
                raw_layers.append(layer)
            except Exception as exc:
                errors.append(f"{Path(path).name}: {exc}")
        target_crs = project_crs or choose_layer_project_crs(raw_layers)
        loaded: List[Dict[str, object]] = []
        for layer in raw_layers:
            try:
                prepared = prepare_layer_for_project(layer, target_crs)
                prepared["project_crs_candidate"] = target_crs
                loaded.append(prepared)
            except Exception as exc:
                errors.append(f"{layer['name']}: 지도 좌표 변환 실패: {exc}")
        self._gis_load_result_queue.put((loaded, errors))

    def _poll_gis_load_result(self) -> None:
        try:
            loaded, errors = self._gis_load_result_queue.get_nowait()
        except queue.Empty:
            if self._gis_load_in_progress:
                self.after(30, self._poll_gis_load_result)
            return
        self._finish_add_gis_layers(loaded, errors)

    def _finish_add_gis_layers(self, loaded: List[Dict[str, object]], errors: List[str]) -> None:
        self._gis_load_in_progress = False
        if loaded:
            self._invalidate_gis_analysis_cache()
        project_candidate = next(
            (
                layer.get("project_crs_candidate")
                for layer in loaded
                if isinstance(layer.get("project_crs_candidate"), CRSDefinition)
            ),
            None,
        )
        if vars(self).get("_gis_project_crs") is None and isinstance(project_candidate, CRSDefinition):
            self._gis_project_crs = project_candidate
            for existing_id, existing_layer in list(self._gis_layers.items()):
                try:
                    self._gis_layers[existing_id] = prepare_layer_for_project(existing_layer, project_candidate)
                except Exception as exc:
                    errors.append(f"{existing_layer.get('name', existing_id)}: 지도 좌표 변환 실패: {exc}")
        loaded_layer_ids: List[str] = []
        for layer in loaded:
            layer.pop("project_crs_candidate", None)
            self._gis_layer_sequence += 1
            layer_id = f"layer_{self._gis_layer_sequence}"
            loaded_layer_ids.append(layer_id)
            is_role = False
            if layer["kind"] == "polyline" and self._gis_pipe_layer_id is None:
                self._gis_pipe_layer_id = layer_id
                is_role = True
            is_flood_candidate = (
                layer["kind"] == "boundary"
                and self._looks_like_flood_layer_name(str(layer.get("name", "")))
            )
            if is_flood_candidate:
                self._gis_flood_layer_ids.add(layer_id)
                self._sync_legacy_flood_layer_id()
                is_role = True
            elif layer["kind"] == "boundary" and self._gis_boundary_layer_id is None:
                self._gis_boundary_layer_id = layer_id
                is_role = True
            default_style = (
                self._default_flood_layer_style()
                if layer_id in self._active_flood_layer_ids()
                else self._default_gis_layer_style(str(layer["kind"]), self._gis_layer_sequence - 1)
            )
            layer["style"] = self._apply_non_role_default_opacity(default_style, str(layer["kind"]), is_role)
            layer["style_user_modified"] = False
            self._gis_layers[layer_id] = layer
        pipe_layer = self._gis_layers.get(self._gis_pipe_layer_id or "")
        if pipe_layer is not None:
            rejected: set[str] = set()
            for flood_layer_id in self._active_flood_layer_ids():
                flood_layer = self._gis_layers.get(flood_layer_id)
                if flood_layer is not None and not self._confirm_pipe_polygon_compatibility(pipe_layer, flood_layer, "침수지역 지도 좌표 확인"):
                    rejected.add(flood_layer_id)
            if rejected:
                self._gis_flood_layer_ids.difference_update(rejected)
                self._sync_legacy_flood_layer_id()
                self._flood_cell_mask = None
                self._sync_default_role_opacities()
        self._invalidate_gis_render_cache()
        self._refresh_gis_layer_tree()
        self._update_flood_region_label()
        unresolved_loaded_ids = [
            layer_id
            for layer_id in loaded_layer_ids
            if layer_id in self._gis_layers and not self._gis_layer_crs_ready(self._gis_layers[layer_id])
        ]
        if unresolved_loaded_ids:
            self.layer_tree.selection_set(unresolved_loaded_ids[0])
            self.layer_tree.see(unresolved_loaded_ids[0])
        if loaded:
            preserve_gis_view = vars(self).get("source_kind") == "gis_project" and isinstance(vars(self).get("img_pil"), Image.Image)
            self._render_gis_project(reset_matrix=not preserve_gis_view)
            self._schedule_gis_index_build(loaded_layer_ids)
            coordinate_note = (
                f" 지도 좌표 선택 필요 {len(unresolved_loaded_ids)}개: 왼쪽 [선택 레이어 좌표]에서 추천 항목을 확인하세요."
                if unresolved_loaded_ids
                else ""
            )
            if self._gis_render_polling:
                self.status_var.set(
                    f"GIS 레이어 {len(loaded)}개 로드 완료. 대규모 전체 지도를 렌더링하는 중...{coordinate_note}"
                )
            else:
                self.status_var.set(
                    f"GIS 레이어 {len(loaded)}개를 불러왔습니다.{coordinate_note or ' 행정경계를 클릭하면 해당 구역을 행렬로 변환합니다.'}"
                )
        elif not errors:
            self.status_var.set("불러온 GIS 레이어가 없습니다.")
        if errors:
            messagebox.showwarning("일부 GIS 불러오기 실패", "\n".join(errors[:8]), parent=self)

    @staticmethod
    def _crs_dialog_initial_value(definition: Optional[CRSDefinition]) -> str:
        if definition is None:
            return ""
        if definition.authority_name and definition.authority_code:
            return f"{definition.authority_name}:{definition.authority_code}"
        return definition.wkt

    def _show_crs_dependency_error(self, error: CRSDependencyError) -> None:
        install_command = f'"{sys.executable}" -m pip install pyproj'
        messagebox.showerror(
            "지도 좌표 기능 준비 필요",
            "현재 프로그램을 실행한 Python 환경에 pyproj가 없습니다.\n\n"
            f"실행 환경:\n{sys.executable}\n\n"
            f"아래 명령으로 해당 환경에 설치한 뒤 프로그램을 다시 시작하세요.\n{install_command}\n\n"
            f"상세 정보: {error}",
            parent=self,
        )
        self.status_var.set("지도 좌표 기능에 필요한 pyproj가 설치되지 않았습니다.")

    def configure_selected_layer_crs(self) -> None:
        layer_id = self._selected_gis_layer_id()
        layer = self._gis_layers.get(layer_id or "")
        if layer is None:
            messagebox.showwarning("선택 레이어 좌표", "지도 좌표를 정할 GIS 레이어를 선택해 주세요.", parent=self)
            return
        if self._gis_load_in_progress or bool(vars(self).get("_gis_crs_job_polling", False)):
            self.status_var.set("GIS 로드 또는 지도 좌표 변환이 끝난 뒤 다시 시도해 주세요.")
            return
        current = source_crs_for_layer(layer)
        source_bbox = tuple(float(value) for value in layer.get("source_bbox", layer.get("bbox", (0, 0, 0, 0))))
        project = vars(self).get("_gis_project_crs")
        project_note = project.label if isinstance(project, CRSDefinition) else "미지정"
        source_meta_value = layer.get("source_meta", layer.get("meta", {}))
        source_meta = source_meta_value if isinstance(source_meta_value, dict) else {}
        source_kind = str(source_meta.get("crs_source", "") or "")
        detected_crs = current if source_kind == "prj" else None
        assigned_crs = current if source_kind == "manual" else None
        try:
            choices = build_layer_crs_choices(
                detected_crs,
                project if isinstance(project, CRSDefinition) else None,
                assigned_crs,
            )
            recommendation = recommend_layer_crs(
                choices,
                source_bbox,
                layer_name=str(layer.get("name", "")),
                detected_crs=detected_crs,
                assigned_crs=assigned_crs,
                project_crs=project if isinstance(project, CRSDefinition) else None,
                project_bbox=self._gis_project_reference_bbox(str(layer_id)),
            )
        except CRSDependencyError as exc:
            self._show_crs_dependency_error(exc)
            return
        dialog = CRSChoiceDialog(
            self,
            title="선택 레이어 지도 좌표",
            prompt=(
                f"레이어: {layer.get('name', '')}\n"
                f"원본 숫자 범위: X {source_bbox[0]:.3f} ~ {source_bbox[2]:.3f}, "
                f"Y {source_bbox[1]:.3f} ~ {source_bbox[3]:.3f}\n"
                f"현재 전체 지도 좌표: {project_note}"
            ),
            choices=choices,
            recommendation=recommendation,
            expert_initial=self._crs_dialog_initial_value(current),
        )
        self.wait_window(dialog)
        if dialog.result is None:
            return
        source_crs, _selection_key = dialog.result
        project_crs = vars(self).get("_gis_project_crs")
        if (
            isinstance(current, CRSDefinition)
            and isinstance(project_crs, CRSDefinition)
            and self._gis_layer_crs_ready(layer)
            and crs_equivalent(current, source_crs)
        ):
            self.status_var.set(f"{layer['name']}: 이미 같은 지도 좌표를 사용 중입니다.")
            return
        establish_project = not isinstance(project_crs, CRSDefinition)
        if not isinstance(project_crs, CRSDefinition):
            project_crs = source_crs
        if not self._confirm_gis_map_reset():
            return
        payload = {
            "layer_id": str(layer_id),
            "layer": dict(layer),
            "source_crs": source_crs,
            "project_crs": project_crs,
            "establish_project": establish_project,
            "layers": [(existing_id, dict(existing)) for existing_id, existing in self._gis_layers.items()],
        }
        self._start_gis_crs_job("layer", payload, f"{layer['name']} 지도 좌표를 적용하는 중...")

    def configure_project_crs(self) -> None:
        if self._gis_load_in_progress or bool(vars(self).get("_gis_crs_job_polling", False)):
            self.status_var.set("GIS 로드 또는 지도 좌표 변환이 끝난 뒤 다시 시도해 주세요.")
            return
        current = vars(self).get("_gis_project_crs")
        pipe_layer = self._gis_layers.get(self._gis_pipe_layer_id or "")
        pipe_crs = source_crs_for_layer(pipe_layer) if pipe_layer is not None else None
        current_definition = current if isinstance(current, CRSDefinition) else None
        try:
            choices = build_project_crs_choices(current_definition, pipe_crs)
            recommendation = recommend_project_crs(choices, current_definition, pipe_crs)
        except CRSDependencyError as exc:
            self._show_crs_dependency_error(exc)
            return
        dialog = CRSChoiceDialog(
            self,
            title="전체 지도 좌표",
            prompt=(
                "모든 GIS 레이어를 겹쳐 표시하고 Clip·격자화할 기준 좌표를 선택하세요.\n"
                "거리와 셀 크기를 미터로 계산하려면 평면좌표 항목을 사용합니다."
            ),
            choices=choices,
            recommendation=recommendation,
            expert_initial=self._crs_dialog_initial_value(current_definition),
        )
        self.wait_window(dialog)
        if dialog.result is None:
            return
        project_crs, _selection_key = dialog.result
        if current_definition is not None and crs_equivalent(current_definition, project_crs):
            self.status_var.set(f"전체 지도는 이미 {project_crs.label} 좌표를 사용 중입니다.")
            return
        if project_crs.is_geographic and not messagebox.askyesno(
            "위도·경도 좌표 확인",
            "위도·경도 좌표에서는 격자 셀 크기를 미터로 직접 계산할 수 없습니다. 그래도 적용할까요?",
            parent=self,
        ):
            return
        if not self._gis_layers:
            self._gis_project_crs = project_crs
            self._update_project_crs_label()
            self.status_var.set(f"전체 지도 좌표 설정: {project_crs.label}")
            return
        if not self._confirm_gis_map_reset():
            return
        payload = {
            "layers": [(layer_id, dict(layer)) for layer_id, layer in self._gis_layers.items()],
            "project_crs": project_crs,
        }
        self._start_gis_crs_job("project", payload, f"모든 레이어를 {project_crs.label} 좌표로 변환하는 중...")

    def _start_gis_crs_job(self, action: str, payload: Dict[str, object], status: str) -> None:
        self._gis_crs_job_generation += 1
        generation = self._gis_crs_job_generation
        self._gis_crs_job_polling = True
        self.status_var.set(status)
        threading.Thread(
            target=self._gis_crs_worker,
            args=(generation, str(action), payload),
            daemon=True,
        ).start()
        self.after(30, self._poll_gis_crs_result)

    def _gis_crs_worker(self, generation: int, action: str, payload: Dict[str, object]) -> None:
        try:
            project_crs = payload["project_crs"]
            if not isinstance(project_crs, CRSDefinition):
                raise ValueError("전체 지도 좌표가 지정되지 않았습니다.")
            if action == "layer":
                source_crs = payload["source_crs"]
                if not isinstance(source_crs, CRSDefinition):
                    raise ValueError("레이어의 지도 좌표가 지정되지 않았습니다.")
                result: object = {
                    "layer_id": str(payload["layer_id"]),
                    "layer": assign_layer_source_crs(
                        dict(payload["layer"]),
                        source_crs,
                        project_crs,
                    ),
                    "project_crs": project_crs,
                }
                if bool(payload.get("establish_project", False)):
                    assigned_layer = dict(result["layer"])
                    result["layers"] = [
                        (
                            str(layer_id),
                            assigned_layer
                            if str(layer_id) == str(payload["layer_id"])
                            else prepare_layer_for_project(dict(layer), project_crs),
                        )
                        for layer_id, layer in payload["layers"]
                    ]
            elif action == "project":
                prepared_layers = [
                    (str(layer_id), prepare_layer_for_project(dict(layer), project_crs))
                    for layer_id, layer in payload["layers"]
                ]
                result = {"layers": prepared_layers, "project_crs": project_crs}
            else:
                raise ValueError(f"Unknown CRS job: {action}")
        except Exception as exc:
            result = exc
        self._gis_crs_result_queue.put((generation, action, result))

    def _poll_gis_crs_result(self) -> None:
        current = None
        while True:
            try:
                candidate = self._gis_crs_result_queue.get_nowait()
                if candidate[0] == self._gis_crs_job_generation:
                    current = candidate
            except queue.Empty:
                break
        if current is None:
            if self._gis_crs_job_polling:
                self.after(30, self._poll_gis_crs_result)
            return
        _generation, action, result = current
        self._gis_crs_job_polling = False
        if isinstance(result, Exception):
            messagebox.showerror("GIS 지도 좌표 변환 오류", str(result), parent=self)
            self.status_var.set("GIS 지도 좌표 변환에 실패했습니다. 기존 레이어는 변경되지 않았습니다.")
            return
        assert isinstance(result, dict)
        project_crs = result.get("project_crs")
        if not isinstance(project_crs, CRSDefinition):
            messagebox.showerror("GIS 지도 좌표 변환 오류", "변환 결과에 전체 지도 좌표가 없습니다.", parent=self)
            return
        if action == "layer":
            layer_id = str(result["layer_id"])
            if layer_id not in self._gis_layers:
                self.status_var.set("지도 좌표를 적용할 레이어가 이미 제거되었습니다.")
                return
            if "layers" in result:
                self._gis_layers = {
                    str(existing_id): dict(layer)
                    for existing_id, layer in result["layers"]
                    if str(existing_id) in self._gis_layers
                }
            else:
                self._gis_layers[layer_id] = dict(result["layer"])
        else:
            self._gis_layers = {
                str(layer_id): dict(layer)
                for layer_id, layer in result.get("layers", [])
                if str(layer_id) in self._gis_layers
            }
        self._gis_project_crs = project_crs
        self._gis_compatibility_confirmed.clear()
        self._flood_cell_mask = None
        self._gis_active_records = None
        self._gis_active_bbox = None
        self._gis_active_spatial_index = None
        self._gis_active_source_meta = None
        self._invalidate_gis_analysis_cache()
        self._invalidate_gis_render_cache()
        self._refresh_gis_layer_tree()
        self._update_flood_region_label()
        if any(
            bool(layer.get("visible", True)) and self._gis_layer_crs_ready(layer)
            for layer in self._gis_layers.values()
        ):
            self._render_gis_project(reset_matrix=True)
            self._schedule_gis_index_build()
        else:
            self._clear_unrenderable_gis_state()
        unresolved = sum(not self._gis_layer_crs_ready(layer) for layer in self._gis_layers.values())
        suffix = f" / 지도 좌표 선택 필요 {unresolved}개" if unresolved else ""
        self.status_var.set(f"전체 지도 좌표 적용 완료: {project_crs.label}{suffix}")

    def _on_gis_layer_tree_double_click(self, event: tk.Event) -> str:
        item_id = self.layer_tree.identify_row(event.y)
        if item_id and item_id in self._gis_layers:
            self.layer_tree.selection_set(item_id)
            self.layer_tree.focus(item_id)
        if self.layer_tree.identify_column(event.x) == "#4":
            self.configure_selected_layer_crs()
        else:
            self._toggle_selected_layer_visibility()
        return "break"

    def _toggle_selected_layer_visibility(self, _event: Optional[tk.Event] = None) -> None:
        if not self._gis_layer_edit_allowed():
            return
        layer_id = self._selected_gis_layer_id()
        if layer_id is None or layer_id not in self._gis_layers:
            return
        if not self._confirm_gis_map_reset():
            return
        layer = self._gis_layers[layer_id]
        layer["visible"] = not bool(layer.get("visible", True))
        self._invalidate_gis_render_cache()
        self._refresh_gis_layer_tree()
        if self.source_kind == "gis_project":
            self._render_gis_project(reset_matrix=False)
        else:
            self._render_gis_project()

    def remove_selected_gis_layer(self) -> None:
        if not self._gis_layer_edit_allowed():
            return
        layer_id = self._selected_gis_layer_id()
        if layer_id is None or layer_id not in self._gis_layers:
            return
        if not self._confirm_gis_map_reset():
            return
        self._invalidate_gis_analysis_cache()
        del self._gis_layers[layer_id]
        self._gis_flood_layer_ids.discard(layer_id)
        self._sync_legacy_flood_layer_id()
        self._flood_cell_mask = None
        self._update_flood_region_label()
        if self._gis_pipe_layer_id == layer_id:
            self._gis_pipe_layer_id = next(
                (key for key, layer in self._gis_layers.items() if layer["kind"] == "polyline"),
                None,
            )
            self._flood_cell_mask = None
        if self._gis_boundary_layer_id == layer_id:
            self._gis_boundary_layer_id = next(
                (
                    key
                    for key, layer in self._gis_layers.items()
                    if layer["kind"] == "boundary" and key not in self._active_flood_layer_ids()
                ),
                None,
            )
        self._sync_default_role_opacities()
        self._update_flood_region_label()
        self._invalidate_gis_render_cache()
        self._refresh_gis_layer_tree()
        if self._gis_layers:
            self._render_gis_project()
        else:
            self._gis_map_bbox = None
            self.img_bgr = None
            self.img_pil = None
            self.tk_img = None
            self.current_matrix = None
            self.base_matrix = None
            self.canvas.delete("all")
            self._update_matrix_preview()
            self.status_var.set("GIS 레이어가 모두 제거되었습니다.")

    def set_selected_pipe_layer(self) -> None:
        if not self._gis_layer_edit_allowed():
            return
        layer_id = self._selected_gis_layer_id()
        layer = self._gis_layers.get(layer_id or "")
        if layer is None or layer["kind"] != "polyline":
            messagebox.showwarning("관로 레이어", "PolyLine 계열 GIS 레이어를 선택해 주세요.", parent=self)
            return
        if not self._gis_layer_crs_ready(layer):
            messagebox.showwarning("관로 지도 좌표 필요", "이 레이어의 지도 좌표를 먼저 선택해 주세요.", parent=self)
            return
        for flood_layer_id in self._active_flood_layer_ids():
            flood_layer = self._gis_layers.get(flood_layer_id)
            if flood_layer is not None and not self._confirm_pipe_polygon_compatibility(layer, flood_layer, "침수지역 지도 좌표 확인"):
                return
        if self._gis_pipe_layer_id != layer_id:
            self._invalidate_gis_analysis_cache()
        self._gis_pipe_layer_id = layer_id
        self._sync_default_role_opacities()
        self._invalidate_gis_render_cache()
        self._refresh_gis_layer_tree()
        if self.source_kind == "gis_project":
            self._render_gis_project(reset_matrix=False)
        self.status_var.set(f"관로 레이어 지정: {layer['name']}")

    def set_selected_boundary_layer(self) -> None:
        if not self._gis_layer_edit_allowed():
            return
        layer_id = self._selected_gis_layer_id()
        layer = self._gis_layers.get(layer_id or "")
        if layer is None or layer["kind"] != "boundary":
            messagebox.showwarning("경계 레이어", "Polygon 계열 GIS 레이어를 선택해 주세요.", parent=self)
            return
        if not self._gis_layer_crs_ready(layer):
            messagebox.showwarning("경계 지도 좌표 필요", "이 레이어의 지도 좌표를 먼저 선택해 주세요.", parent=self)
            return
        if layer_id in self._active_flood_layer_ids():
            messagebox.showwarning("경계 레이어", "침수지역과 행정경계는 서로 다른 Polygon GIS 레이어로 지정해 주세요.", parent=self)
            return
        self._gis_boundary_layer_id = layer_id
        self._gis_hover_boundary = None
        self._sync_default_role_opacities()
        self._invalidate_gis_render_cache()
        self._refresh_gis_layer_tree()
        if self.source_kind == "gis_project":
            self._render_gis_project(reset_matrix=False)
        self.status_var.set(f"경계 레이어 지정: {layer['name']}")

    def _confirm_pipe_polygon_compatibility(
        self,
        pipe_layer: Dict[str, object],
        polygon_layer: Dict[str, object],
        title: str,
    ) -> bool:
        if not self._gis_layer_crs_ready(pipe_layer) or not self._gis_layer_crs_ready(polygon_layer):
            messagebox.showerror(
                title,
                "지도 좌표가 선택되지 않은 레이어가 있습니다. 왼쪽의 [선택 레이어 좌표]에서 추천 항목을 확인해 주세요.",
                parent=self,
            )
            return False
        pipe_crs = pipe_layer.get("working_crs")
        polygon_crs = polygon_layer.get("working_crs")
        if not isinstance(pipe_crs, CRSDefinition) or not isinstance(polygon_crs, CRSDefinition):
            messagebox.showerror(
                title,
                "여러 GIS 레이어를 함께 분석하려면 각 레이어의 지도 좌표를 선택해야 합니다.",
                parent=self,
            )
            return False
        try:
            same_crs = crs_equivalent(pipe_crs, polygon_crs)
        except Exception as exc:
            messagebox.showerror(title, str(exc), parent=self)
            return False
        if not same_crs:
            messagebox.showerror(
                title,
                f"내부 지도 좌표 불일치: {pipe_crs.label} / {polygon_crs.label}. 전체 지도 좌표를 다시 적용해 주세요.",
                parent=self,
            )
            return False
        if not self._bbox_intersects(tuple(pipe_layer["bbox"]), tuple(polygon_layer["bbox"])):
            messagebox.showerror(title, "관로와 Polygon 레이어의 작업 좌표 범위가 겹치지 않습니다.", parent=self)
            return False
        return True

    def set_selected_flood_layer(self) -> None:
        if not self._gis_layer_edit_allowed():
            return
        layer_id = self._selected_gis_layer_id()
        layer = self._gis_layers.get(layer_id or "")
        if layer is None or layer.get("kind") != "boundary":
            messagebox.showwarning("침수지역 레이어", "별도로 불러온 Polygon 침수지역 GIS 레이어를 선택해 주세요.", parent=self)
            return
        if not self._gis_layer_crs_ready(layer):
            messagebox.showwarning("침수지역 지도 좌표 필요", "이 레이어의 지도 좌표를 먼저 선택해 주세요.", parent=self)
            return
        if layer_id == self._gis_boundary_layer_id:
            messagebox.showwarning("침수지역 레이어", "행정경계가 아닌 별도의 침수지역 Polygon GIS 레이어를 선택해 주세요.", parent=self)
            return
        pipe_layer = self._gis_layers.get(self._gis_pipe_layer_id or "")
        if pipe_layer is not None and not self._confirm_pipe_polygon_compatibility(pipe_layer, layer, "침수지역 지도 좌표 확인"):
            return
        active = self._active_flood_layer_ids()
        if layer_id in active:
            active.remove(layer_id)
            action = "제거"
        else:
            active.add(layer_id)
            action = "추가"
        self._gis_flood_layer_ids = active
        self._sync_legacy_flood_layer_id()
        self._flood_cell_mask = None
        self._sync_default_role_opacities()
        self._invalidate_gis_render_cache()
        self._refresh_gis_layer_tree()
        self._update_flood_region_label()
        if self.source_kind == "gis_project":
            self._render_gis_project(reset_matrix=False)
        elif self.source_kind == "gis" and self.current_matrix is not None:
            if self._active_flood_layer_ids():
                self.refresh_flood_overlap()
            else:
                self._invalidate_matrix_arrow_cache()
                self._queue_render(high_quality=True, delay=0)
                self._update_matrix_preview()
        self.status_var.set(f"침수지역 레이어 {action}: {layer['name']} / 지정된 모든 침수 Polygon과 관로 중첩을 판정합니다.")

    def configure_pipe_height_fields(self) -> None:
        layer = self._gis_layers.get(self._gis_pipe_layer_id or "")
        if layer is None:
            messagebox.showwarning("관로 레이어 없음", "왼쪽 레이어에서 관로 레이어를 지정해 주세요.", parent=self)
            return
        height_fields = self._ask_gis_height_fields(list(layer["meta"].get("fields", [])))
        if height_fields is None:
            return
        start_height_field, end_height_field = height_fields
        try:
            geometry, bbox, meta = read_polyline_shp(
                str(layer["path"]),
                start_height_field=start_height_field,
                end_height_field=end_height_field,
            )
            layer["geometry"] = geometry
            layer["bbox"] = tuple(float(value) for value in bbox)
            layer["meta"] = dict(meta)
            layer["spatial_index"] = PolylineSpatialIndex.build(geometry)
            self._invalidate_gis_analysis_cache()
            self._invalidate_gis_render_cache()
            if self.source_kind == "gis_project":
                self._render_gis_project(reset_matrix=False)
            start_label = start_height_field or "사용 안 함"
            end_label = end_height_field or "사용 안 함"
            self.status_var.set(f"관로 표고 필드 적용: 시작={start_label}, 끝={end_label}")
        except Exception as exc:
            messagebox.showerror("관로 표고 필드 오류", str(exc), parent=self)

    @staticmethod
    def _rings_bbox(rings: List[List[Tuple[float, float]]]) -> Tuple[float, float, float, float]:
        points = [point for ring in rings for point in ring]
        if not points:
            raise ValueError("선택된 경계에 좌표가 없습니다.")
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        return min(xs), min(ys), max(xs), max(ys)

    def _selected_or_active_pipe_layer(self) -> Optional[Tuple[str, Dict[str, object]]]:
        selected_id = self._selected_gis_layer_id()
        selected_layer = self._gis_layers.get(selected_id or "")
        if selected_layer is not None and selected_layer.get("kind") == "polyline":
            return str(selected_id), selected_layer
        active_layer = self._gis_layers.get(self._gis_pipe_layer_id or "")
        if active_layer is not None and active_layer.get("kind") == "polyline":
            return str(self._gis_pipe_layer_id), active_layer
        return None

    def _pipe_layer_matrix_cache_key(
        self,
        layer_id: str,
        pipe_layer: Dict[str, object],
        rows: int,
        start_height_field: Optional[str],
        end_height_field: Optional[str],
    ) -> Tuple[object, ...]:
        return (
            "pipe-full",
            str(layer_id),
            int(id(pipe_layer.get("geometry"))),
            int(rows),
            round(float(self._rotation_degrees()), 6),
            str(start_height_field or ""),
            str(end_height_field or ""),
        )

    def _pipe_layer_matrix_worker(self, generation: int, payload: Dict[str, object]) -> None:
        try:
            result = matrix_from_loaded_polylines(
                payload["records"],
                rows=int(payload["rows"]),
                bbox=payload["bbox"],
                source_meta=payload["source_meta"],
                rotation_degrees=float(payload["rotation_degrees"]),
            )
            payload["prepared_result"] = result
            output: object = payload
        except Exception as exc:
            output = exc
        self._gis_pipe_job_queue.put((generation, output))

    def _poll_pipe_layer_matrix_result(self) -> None:
        current: object | None = None
        while True:
            try:
                generation, result = self._gis_pipe_job_queue.get_nowait()
                if generation == self._gis_pipe_job_generation:
                    current = result
            except queue.Empty:
                break
        if current is None:
            if self._gis_pipe_job_polling:
                self.after(30, self._poll_pipe_layer_matrix_result)
            return
        self._gis_pipe_job_polling = False
        if isinstance(current, Exception):
            messagebox.showerror("관로 전체 변환 오류", str(current), parent=self)
            return
        payload = current
        assert isinstance(payload, dict)
        cached = (payload["records"], payload["bbox"], payload["prepared_result"])
        self._gis_pipe_result_cache[payload["cache_key"]] = cached
        while len(self._gis_pipe_result_cache) > GIS_PIPE_RESULT_CACHE_LIMIT:
            self._gis_pipe_result_cache.pop(next(iter(self._gis_pipe_result_cache)))
        self._finish_pipe_layer_matrix_selection(payload, cached, cache_note="")

    def _finish_pipe_layer_matrix_selection(
        self,
        payload: Dict[str, object],
        cached: Tuple[list, Tuple[float, float, float, float], object],
        *,
        cache_note: str,
    ) -> None:
        records, bbox, prepared_result = cached
        pipe_layer = payload["pipe_layer"]
        self._gis_pipe_layer_id = str(payload["layer_id"])
        self._gis_boundary_path = None
        self._gis_boundary_field = None
        self._gis_boundary_value = "전체 관로"
        self._basin_ring_groups = None
        self._basin_cell_mask = None
        self._basin_is_lasso = False
        self._gis_last_clicked_boundary = None
        self._sync_default_role_opacities()
        self._refresh_gis_layer_tree()
        self._apply_clipped_records_to_matrix(
            records,
            bbox,
            payload["source_meta"],
            pipe_layer,
            int(payload["rows"]),
            payload["start_height_field"],
            payload["end_height_field"],
            f"선택 관로 전체 ({pipe_layer['name']})",
            prepared_result=prepared_result,
        )
        elapsed = time.perf_counter() - float(payload["started"])
        current_matrix = vars(self).get("current_matrix")
        nonzero = int(np.count_nonzero(current_matrix)) if isinstance(current_matrix, np.ndarray) else 0
        self.status_var.set(
            f"{pipe_layer['name']} 전체 관로 변환 완료: {elapsed:.2f}초{cache_note} | {payload['rows']}행 | 관로 셀 {nonzero}개"
        )

    @guarded_action
    def convert_selected_pipe_layer_to_matrix(self) -> None:
        selected = self._selected_or_active_pipe_layer()
        if selected is None:
            messagebox.showwarning("관로 레이어 없음", "PolyLine 관로 SHP 레이어를 선택하거나 관로 레이어로 지정해 주세요.", parent=self)
            return
        layer_id, pipe_layer = selected
        if not self._gis_layer_crs_ready(pipe_layer):
            messagebox.showwarning("관로 지도 좌표 필요", "선택한 관로 레이어의 지도 좌표를 먼저 선택해 주세요.", parent=self)
            return
        if not pipe_layer.get("geometry"):
            messagebox.showwarning("관로 레이어 비어 있음", "선택한 관로 레이어에 변환할 PolyLine 레코드가 없습니다.", parent=self)
            return
        for flood_layer_id in self._active_flood_layer_ids():
            flood_layer = self._gis_layers.get(flood_layer_id)
            if flood_layer is not None and not self._confirm_pipe_polygon_compatibility(pipe_layer, flood_layer, "침수지역 지도 좌표 확인"):
                return
        source_meta = dict(pipe_layer["meta"])
        start_height_field = str(source_meta.get("height_start_field", "") or "") or None
        end_height_field = str(source_meta.get("height_end_field", "") or "") or None
        rows = max(2, int(self.rows_var.get()))
        records = list(pipe_layer["geometry"])
        bbox = tuple(float(value) for value in pipe_layer["bbox"])
        cache_key = self._pipe_layer_matrix_cache_key(layer_id, pipe_layer, rows, start_height_field, end_height_field)
        payload: Dict[str, object] = {
            "cache_key": cache_key,
            "layer_id": layer_id,
            "pipe_layer": pipe_layer,
            "records": records,
            "bbox": bbox,
            "source_meta": source_meta,
            "rows": int(rows),
            "rotation_degrees": self._rotation_degrees(),
            "start_height_field": start_height_field,
            "end_height_field": end_height_field,
            "started": time.perf_counter(),
        }
        cached = self._gis_pipe_result_cache.get(cache_key)
        if cached is not None:
            self._finish_pipe_layer_matrix_selection(payload, cached, cache_note=" / 캐시")
            return
        self._gis_pipe_job_generation += 1
        generation = self._gis_pipe_job_generation
        self._gis_pipe_job_polling = True
        self.status_var.set(f"{pipe_layer['name']} 전체 관로를 행렬로 변환하는 중...")
        threading.Thread(target=self._pipe_layer_matrix_worker, args=(generation, payload), daemon=True).start()
        self.after(30, self._poll_pipe_layer_matrix_result)

    def _load_selected_pipe_records(self) -> Optional[Tuple[Dict[str, object], list, str | None, str | None]]:
        layer = self._gis_layers.get(self._gis_pipe_layer_id or "")
        if layer is None:
            messagebox.showwarning("관로 레이어 없음", "왼쪽 레이어에서 관로 레이어를 지정해 주세요.", parent=self)
            return None
        if not self._gis_layer_crs_ready(layer):
            messagebox.showwarning("관로 지도 좌표 필요", "선택한 관로 레이어의 지도 좌표를 먼저 선택해 주세요.", parent=self)
            return None
        source_meta = dict(layer["meta"])
        start_height_field = str(source_meta.get("height_start_field", "") or "") or None
        end_height_field = str(source_meta.get("height_end_field", "") or "") or None
        return source_meta, list(layer["geometry"]), start_height_field, end_height_field

    def _apply_clipped_records_to_matrix(
        self,
        records: list,
        bbox: Tuple[float, float, float, float],
        source_meta: Dict[str, object],
        pipe_layer: Dict[str, object],
        rows: int,
        start_height_field: Optional[str],
        end_height_field: Optional[str],
        clip_label: str,
        prepared_result: object | None = None,
    ) -> None:
        if not records:
            raise ValueError("선택한 구역 내부에서 관로를 찾지 못했습니다.")
        self._gis_active_records = list(records)
        self._gis_active_bbox = tuple(float(value) for value in bbox)
        self._gis_active_render_records = None
        self._gis_active_render_bbox = None
        self._gis_active_render_spatial_index = None
        self._gis_active_render_rotation = None
        try:
            self._gis_active_spatial_index = PolylineSpatialIndex.build(self._gis_active_records)
        except Exception:
            self._gis_active_spatial_index = None
        self._gis_active_source_meta = dict(source_meta)
        result = prepared_result or matrix_from_loaded_polylines(
            records,
            rows=int(rows),
            bbox=bbox,
            source_meta=source_meta,
            rotation_degrees=self._rotation_degrees(),
        )
        self.img_path = str(pipe_layer["path"])
        self.source_kind = "gis"
        self._source_image_bgr = None
        self._gis_source_path = str(pipe_layer["path"])
        self._gis_rows = int(rows)
        self._gis_start_height_field = start_height_field
        self._gis_end_height_field = end_height_field
        self._apply_gis_result_to_state(result)
        self.status_var.set(
            f"{clip_label} 완료: A={self.A}, B={self.B}, 관로 레코드 {len(records)}개, 관로 셀 {int(np.count_nonzero(self.current_matrix))}개."
        )
        self._queue_render(high_quality=True, delay=0)
        self._update_matrix_preview()

    def _world_point_from_event(self, event: tk.Event) -> Optional[Tuple[float, float]]:
        return self._world_point_from_canvas_xy(float(event.x), float(event.y))

    def _world_point_from_canvas_xy(self, canvas_x: float, canvas_y: float) -> Optional[Tuple[float, float]]:
        if self.matrix_arrow_view or self.img_pil is None:
            return None
        image_x = (float(canvas_x) - float(self._display_origin[0])) / max(self._scale, 1e-6)
        image_y = (float(canvas_y) - float(self._display_origin[1])) / max(self._scale, 1e-6)
        if image_x < 0 or image_y < 0 or image_x >= self.img_pil.size[0] or image_y >= self.img_pil.size[1]:
            return None
        image_point = (image_x, image_y)
        if image_point is None or self._gis_map_bbox is None or self.img_pil is None:
            return None
        xmin, ymin, xmax, ymax = self._gis_map_bbox
        width = max(float(self.img_pil.size[0] - 1), 1.0)
        height = max(float(self.img_pil.size[1] - 1), 1.0)
        return (
            xmin + float(image_point[0]) / width * (xmax - xmin),
            ymax - float(image_point[1]) / height * (ymax - ymin),
        )

    @staticmethod
    def _boundary_feature_label(
        boundary: object,
        feature_index: int,
        preferred_field: str | None = None,
    ) -> Tuple[str, str]:
        attributes = dict(getattr(boundary, "attributes", {}) or {})
        if preferred_field:
            value = str(attributes.get(preferred_field, "") or "").strip()
            if value:
                return str(preferred_field), value
        preferred = ("NAME", "DISTRICT", "ADM_NM", "SIG_KOR_NM", "EMD_KOR_NM", "CTP_KOR_NM")
        for field in preferred:
            value = str(attributes.get(field, "") or "").strip()
            if value:
                return field, value
        for field, raw_value in attributes.items():
            value = str(raw_value or "").strip()
            if value:
                return str(field), value
        return "", f"feature_{feature_index + 1}"

    def _boundary_at_world_point(self, point: Tuple[float, float]) -> Optional[Tuple[str, int]]:
        layer_id = self._gis_boundary_layer_id or ""
        layer = self._gis_layers.get(layer_id)
        if layer is None or layer.get("kind") != "boundary" or not bool(layer.get("visible", True)):
            return None
        feature_index = self._ensure_boundary_feature_index(layer)
        matched = feature_index.find_containing(point)
        return (layer_id, matched) if matched is not None else None

    def _boundary_matrix_worker(
        self,
        generation: int,
        payload: Dict[str, object],
    ) -> None:
        try:
            clipped = clip_polylines_to_boundary(
                payload["pipe_records"],
                payload["rings"],
                spatial_index=payload["spatial_index"],
            )
            if not clipped:
                raise ValueError("선택한 구역 안에서 관로를 찾지 못했습니다.")
            clipped_bbox = self._rings_bbox(payload["rings"])
            prepared_result = matrix_from_loaded_polylines(
                clipped,
                rows=int(payload["rows"]),
                bbox=clipped_bbox,
                source_meta=payload["source_meta"],
                rotation_degrees=float(payload["rotation_degrees"]),
            )
            payload["clipped"] = clipped
            payload["clipped_bbox"] = clipped_bbox
            payload["prepared_result"] = prepared_result
            result: object = payload
        except Exception as exc:
            result = exc
        self._gis_boundary_job_queue.put((generation, result))

    def _poll_boundary_matrix_result(self) -> None:
        current: object | None = None
        while True:
            try:
                generation, result = self._gis_boundary_job_queue.get_nowait()
                if generation == self._gis_boundary_job_generation:
                    current = result
            except queue.Empty:
                break
        if current is None:
            if self._gis_boundary_job_polling:
                self.after(30, self._poll_boundary_matrix_result)
            return
        self._gis_boundary_job_polling = False
        if isinstance(current, Exception):
            messagebox.showerror("경계 클릭 변환 오류", str(current), parent=self)
            return
        payload = current
        assert isinstance(payload, dict)
        cache_key = payload["cache_key"]
        cached = (payload["clipped"], payload["clipped_bbox"], payload["prepared_result"])
        self._gis_boundary_result_cache[cache_key] = cached
        while len(self._gis_boundary_result_cache) > GIS_BOUNDARY_RESULT_CACHE_LIMIT:
            self._gis_boundary_result_cache.pop(next(iter(self._gis_boundary_result_cache)))
        self._finish_boundary_matrix_selection(payload, cached, cache_note="")

    def _finish_boundary_matrix_selection(
        self,
        payload: Dict[str, object],
        cached: Tuple[list, Tuple[float, float, float, float], object],
        *,
        cache_note: str,
    ) -> None:
        clipped, clipped_bbox, prepared_result = cached
        self._basin_ring_groups = [payload["rings"]] if payload.get("rings") else None
        self._basin_is_lasso = False
        boundary_layer = payload["boundary_layer"]
        pipe_layer = payload["pipe_layer"]
        self._gis_boundary_path = str(boundary_layer["path"])
        self._gis_boundary_field = str(payload["field"]) or None
        self._gis_boundary_value = str(payload["value"])
        self._gis_last_clicked_boundary = (str(payload["layer_id"]), int(payload["feature_index"]))
        self._apply_clipped_records_to_matrix(
            clipped,
            clipped_bbox,
            payload["source_meta"],
            pipe_layer,
            int(payload["rows"]),
            payload["start_height_field"],
            payload["end_height_field"],
            f"경계 클릭 ({payload['value']})",
            prepared_result=prepared_result,
        )
        elapsed = time.perf_counter() - float(payload["started"])
        current_matrix = vars(self).get("current_matrix")
        nonzero = int(np.count_nonzero(current_matrix)) if isinstance(current_matrix, np.ndarray) else 0
        self.status_var.set(f"{payload['value']} 변환 완료: {elapsed:.2f}초{cache_note} | {payload['rows']}행 | 관로 셀 {nonzero}개")

    def _clip_boundary_feature_to_matrix(self, layer_id: str, feature_index: int) -> None:
        pipe_layer = self._gis_layers.get(self._gis_pipe_layer_id or "")
        boundary_layer = self._gis_layers.get(layer_id)
        if pipe_layer is None or boundary_layer is None:
            raise ValueError("관로 레이어와 경계 레이어를 각각 지정해 주세요.")
        boundaries = list(boundary_layer["geometry"])
        if not (0 <= int(feature_index) < len(boundaries)):
            raise ValueError("클릭한 행정경계를 찾을 수 없습니다.")
        boundary = boundaries[int(feature_index)]
        rings = [ring for ring in boundary.rings if len(ring) >= 3]
        field, value = self._boundary_feature_label(
            boundary,
            int(feature_index),
            str(boundary_layer.get("label_field", "") or "") or None,
        )
        rows = max(2, int(self.rows_var.get()))
        loaded = self._load_selected_pipe_records()
        if loaded is None:
            return
        source_meta, pipe_records, start_height_field, end_height_field = loaded
        if not self._confirm_pipe_polygon_compatibility(pipe_layer, boundary_layer, "GIS 지도 좌표 확인"):
            return
        spatial_index = self._ensure_polyline_spatial_index(pipe_layer)
        started = time.perf_counter()
        self.status_var.set(f"{value} 관로를 Clip하고 행렬로 변환하는 중...")
        self.update_idletasks()
        cache_key = (
            str(self._gis_pipe_layer_id),
            int(id(pipe_layer.get("geometry"))),
            str(layer_id),
            int(feature_index),
            int(rows),
            round(float(self._rotation_degrees()), 6),
            str(start_height_field or ""),
            str(end_height_field or ""),
        )
        cached = self._gis_boundary_result_cache.get(cache_key)
        payload: Dict[str, object] = {
            "cache_key": cache_key,
            "pipe_records": pipe_records,
            "rings": rings,
            "spatial_index": spatial_index,
            "source_meta": source_meta,
            "rotation_degrees": self._rotation_degrees(),
            "rows": int(rows),
            "boundary_layer": boundary_layer,
            "pipe_layer": pipe_layer,
            "field": field,
            "value": value,
            "layer_id": layer_id,
            "feature_index": int(feature_index),
            "start_height_field": start_height_field,
            "end_height_field": end_height_field,
            "started": started,
        }
        if cached is not None:
            self._finish_boundary_matrix_selection(payload, cached, cache_note=" / 캐시")
            return
        self._gis_boundary_job_generation += 1
        generation = self._gis_boundary_job_generation
        self._gis_boundary_job_polling = True
        threading.Thread(target=self._boundary_matrix_worker, args=(generation, payload), daemon=True).start()
        self.after(30, self._poll_boundary_matrix_result)

    def clip_selected_boundary_to_matrix(self) -> None:
        self.show_boundary_click_map()

    def _lasso_world_rings(self) -> List[List[Tuple[float, float]]]:
        if self._gis_map_bbox is None or self.img_pil is None or len(self.roi_polygon_points) < 3:
            raise ValueError("GIS 합성 화면에서 LASSO 구역을 먼저 선택해 주세요.")
        xmin, ymin, xmax, ymax = self._gis_map_bbox
        width = max(float(self.img_pil.size[0] - 1), 1.0)
        height = max(float(self.img_pil.size[1] - 1), 1.0)
        ring = [
            (
                xmin + float(point[0]) / width * (xmax - xmin),
                ymax - float(point[1]) / height * (ymax - ymin),
            )
            for point in self.roi_polygon_points
        ]
        return [ring]

    @guarded_action
    def clip_lasso_to_matrix(self) -> None:
        pipe_layer = self._gis_layers.get(self._gis_pipe_layer_id or "")
        if pipe_layer is None:
            messagebox.showwarning("LASSO Clip", "왼쪽 레이어에서 관로 레이어를 지정해 주세요.", parent=self)
            return
        try:
            rings = self._lasso_world_rings()
            rows = int(self.rows_var.get())
            if not 2 <= rows <= 300:
                raise ValueError("행 수는 2~300의 정수로 입력해 주세요.")
        except Exception as exc:
            messagebox.showwarning("LASSO Clip", str(exc), parent=self)
            return
        loaded = self._load_selected_pipe_records()
        if loaded is None:
            return
        source_meta, pipe_records, start_height_field, end_height_field = loaded
        try:
            spatial_index = self._ensure_polyline_spatial_index(pipe_layer)
            clipped = clip_polylines_to_boundary(pipe_records, rings, spatial_index=spatial_index)
            self._gis_boundary_path = None
            self._gis_boundary_field = None
            self._gis_boundary_value = "LASSO"
            self._basin_ring_groups = [rings]
            self._basin_is_lasso = True
            self._apply_clipped_records_to_matrix(
                clipped,
                self._rings_bbox(rings),
                source_meta,
                pipe_layer,
                int(rows),
                start_height_field,
                end_height_field,
                "LASSO Clip",
            )
        except Exception as exc:
            messagebox.showerror("LASSO Clip 오류", str(exc), parent=self)

    def generate_gis_arrow_training(self) -> None:
        self.status_var.set("GIS 화살표 학습은 현재 더미 기능입니다. 추후 활성화 예정입니다.")
        messagebox.showinfo(TXT_GIS_SYNTH_TRAIN, "GIS 합성 화살표 학습은 추후 활성화 예정입니다.", parent=self)

    @guarded_action
    def start_empty_grid(self) -> None:
        self._clear_compressed_grid_shape()
        if not self._ensure_grid():
            return
        self._invalidate_detection_cache()
        self.base_matrix = np.zeros((self.A, self.B), dtype=np.uint8)
        self.current_matrix = self.base_matrix.copy()
        self.base_confidence_matrix = np.zeros((self.A, self.B), dtype=np.float32)
        self.confidence_matrix = self.base_confidence_matrix.copy()
        self.base_model_applied_mask = np.zeros((self.A, self.B), dtype=np.uint8)
        self.model_applied_mask = self.base_model_applied_mask.copy()
        self.occ_matrix = np.zeros((self.A, self.B), dtype=np.uint8)
        self.path_occ_matrix = np.zeros((self.A, self.B), dtype=np.uint8)
        self.support_mask = np.zeros((self.A, self.B), dtype=np.uint8)
        self.unresolved_mask = np.zeros((self.A, self.B), dtype=np.uint8)
        self.user_edit_mask = np.zeros((self.A, self.B), dtype=np.uint8)
        self.manual_edits_since_autogen = False
        self.arrow_boxes = []
        self.outlet_cell = None
        self.outlet_cells = set()
        self.outlet_pick_mode = False
        self.lasso_mode = False
        self._lasso_preview_point = None
        self.matrix_arrow_view = False
        if "matrix_arrow_button" in vars(self):
            self.matrix_arrow_button.configure(text=TXT_MATRIX_ARROWS)
        self._invalidate_matrix_arrow_cache()
        row, col = self._default_selected_cell()
        self._set_selected_cell(row, col)
        self.status_var.set(f"\ube48 \ud589\ub82c\uc744 \uc900\ube44\ud588\uc2b5\ub2c8\ub2e4. A={self.A}, B={self.B}. \uc140\uc744 \uc120\ud0dd\ud55c \ub4a4 \uc22b\uc790\ud0a4\ub85c \uc785\ub825\ud574 \uc8fc\uc138\uc694.")
        self._record_support_baseline()
        self._queue_render(high_quality=True, delay=0)
        self._update_matrix_preview()
        self.canvas.focus_set()

    @guarded_action
    def run_detection(self) -> None:
        if vars(self).get("source_kind") == "gis_project":
            messagebox.showinfo("행렬 생성 필요", "먼저 유역을 선택해 행렬을 생성해 주세요.", parent=self)
            return
        self._clear_compressed_grid_shape()
        if not self._ensure_grid():
            return
        processing_img, roi_mask = self._processing_image_and_roi_mask()
        processing_img = processing_img.copy()
        rows = int(self.A)
        cache_key = self._current_detection_cache_key()
        def finish(result):
            direction_matrix, occ_matrix, arrow_boxes, meta = result
            self._store_detection_cache(cache_key, direction_matrix, occ_matrix, arrow_boxes, meta)
            previous_matrix = self.current_matrix.copy() if self.current_matrix is not None else np.zeros((self.A, self.B), dtype=np.uint8)
            previous_conf = self.confidence_matrix.copy() if self.confidence_matrix is not None else np.zeros((self.A, self.B), dtype=np.float32)
            previous_occ = self.occ_matrix.copy() if self.occ_matrix is not None else np.zeros((self.A, self.B), dtype=np.uint8)
            previous_path_occ = self.path_occ_matrix.copy() if self.path_occ_matrix is not None else np.zeros((self.A, self.B), dtype=np.uint8)
            previous_support = self.support_mask.copy() if self.support_mask is not None else np.zeros((self.A, self.B), dtype=np.uint8)
            previous_model_mask = self.model_applied_mask.copy() if self.model_applied_mask is not None else np.zeros((self.A, self.B), dtype=np.uint8)
            direction_matrix = self._merge_roi_matrix(previous_matrix, direction_matrix.astype(np.uint8), roi_mask)
            conf_matrix = self._merge_roi_matrix(previous_conf, np.asarray(meta.get("confidence", np.zeros((self.A, self.B))), dtype=np.float32), roi_mask)
            occ_matrix = self._merge_roi_matrix(previous_occ, occ_matrix.astype(np.uint8), roi_mask)
            path_occ_matrix = self._merge_roi_matrix(previous_path_occ, np.asarray(meta.get("path_occ", occ_matrix), dtype=np.uint8), roi_mask)
            support_matrix = self._merge_roi_matrix(previous_support, np.asarray(meta.get("support_mask", occ_matrix), dtype=np.uint8), roi_mask)
            cleared_model_mask = self._merge_roi_matrix(previous_model_mask, np.zeros((self.A, self.B), dtype=np.uint8), roi_mask)
            self.base_matrix = direction_matrix.astype(np.uint8)
            self.current_matrix = self.base_matrix.copy()
            self.base_confidence_matrix = conf_matrix.astype(np.float32)
            self.confidence_matrix = self.base_confidence_matrix.copy()
            self.base_model_applied_mask = cleared_model_mask.astype(np.uint8)
            self.model_applied_mask = self.base_model_applied_mask.copy()
            self.occ_matrix = occ_matrix.astype(np.uint8)
            self.path_occ_matrix = path_occ_matrix.astype(np.uint8)
            self.support_mask = support_matrix.astype(np.uint8)
            self.unresolved_mask = np.zeros((self.A, self.B), dtype=np.uint8)
            self._reset_manual_edit_state()
            self._apply_workspace_region()
            self._record_support_baseline()
            self.arrow_boxes = list(arrow_boxes)
            row, col = self._default_selected_cell()
            self._set_selected_cell(row, col)
            nonzero = int(np.count_nonzero(self.current_matrix))
            low_conf = int(np.count_nonzero((self.current_matrix > 0) & (self.confidence_matrix < 0.55)))
            overlap_cells = int(np.count_nonzero(np.asarray(meta.get("overlap_count", np.zeros((self.A, self.B))), dtype=np.int32) > 1))
            roi_text = " | ROI \uc801\uc6a9" if roi_mask is not None else ""
            status_text = (
                f"\uc790\ub3d9 \uc778\uc2dd \uc644\ub8cc{roi_text}: \ubc29\ud5a5 \uc140 {nonzero}\uac1c, "
                f"\uc800\uc2e0\ub8b0 \uc140 {low_conf}\uac1c, \uacb9\uce68 \uc758\uc2ec \uc140 {overlap_cells}\uac1c, "
                f"\ud654\uc0b4\ud45c \uac1d\uccb4 \ud6c4\ubcf4 {len(self.arrow_boxes)}\uac1c."
            )
            self.status_var.set(self._append_gold_summary_to_status(status_text))
            self._queue_render(high_quality=True, delay=0)
            self._update_matrix_preview()
            self.canvas.focus_set()
        self._start_compute_task("자동 인식", lambda cancel: compute_direction_matrix_with_meta(processing_img, rows), finish)

    @guarded_action
    def apply_model(self) -> None:
        if vars(self).get("source_kind") == "gis_project":
            messagebox.showinfo("행렬 생성 필요", "먼저 유역을 선택해 행렬을 생성해 주세요.", parent=self)
            return
        self._clear_compressed_grid_shape()
        if not self._ensure_grid():
            return
        processing_img, roi_mask = self._processing_image_and_roi_mask()
        processing_img = processing_img.copy()
        rows, model_path, device = int(self.A), self.model_path, self._device_preference()
        cache_key = self._current_detection_cache_key()
        detection_cache = self._detection_cache if cache_key and cache_key == self._detection_cache_key else None
        def compute(cancel):
            return predict_with_model_with_meta(processing_img, rows, model_path=model_path,
                device_preference=device, strict_deployment_ready=STRICT_DEPLOYMENT_MODEL, detection_cache=detection_cache)
        def finish(result):
            predicted, morph, meta = result
            previous_matrix = self.current_matrix.copy() if self.current_matrix is not None else np.zeros((self.A, self.B), dtype=np.uint8)
            previous_conf = self.confidence_matrix.copy() if self.confidence_matrix is not None else np.zeros((self.A, self.B), dtype=np.float32)
            previous_occ = self.occ_matrix.copy() if self.occ_matrix is not None else np.zeros((self.A, self.B), dtype=np.uint8)
            previous_path_occ = self.path_occ_matrix.copy() if self.path_occ_matrix is not None else np.zeros((self.A, self.B), dtype=np.uint8)
            previous_support = self.support_mask.copy() if self.support_mask is not None else np.zeros((self.A, self.B), dtype=np.uint8)
            previous_model_mask = self.model_applied_mask.copy() if self.model_applied_mask is not None else np.zeros((self.A, self.B), dtype=np.uint8)
            predicted = self._merge_roi_matrix(previous_matrix, predicted.astype(np.uint8), roi_mask)
            model_conf = self._merge_roi_matrix(previous_conf, np.asarray(meta.get("confidence", np.zeros((self.A, self.B))), dtype=np.float32), roi_mask)
            occ = np.asarray(meta.get("occ", np.zeros((self.A, self.B))), dtype=np.uint8)
            occ = self._merge_roi_matrix(previous_occ, np.maximum(occ, np.maximum((predicted > 0).astype(np.uint8), (morph > 0).astype(np.uint8))), roi_mask)
            path_occ = self._merge_roi_matrix(
                previous_path_occ,
                np.asarray(meta.get("path_occ", np.maximum((predicted > 0).astype(np.uint8), (morph > 0).astype(np.uint8))), dtype=np.uint8),
                roi_mask,
            )
            support_mask = self._merge_roi_matrix(
                previous_support,
                np.asarray(meta.get("support_mask", np.maximum((predicted > 0).astype(np.uint8), (morph > 0).astype(np.uint8))), dtype=np.uint8),
                roi_mask,
            )
            model_applied = self._merge_roi_matrix(
                previous_model_mask,
                np.asarray(meta.get("model_applied_mask", np.zeros((self.A, self.B))), dtype=np.uint8),
                roi_mask,
            ).astype(np.uint8)
            model_zero_mask = (model_applied == 1) & (predicted == 0)
            occ[model_zero_mask] = 0
            path_occ[model_zero_mask] = 0
            self.base_matrix = predicted.astype(np.uint8)
            self.current_matrix = self.base_matrix.copy()
            self.base_confidence_matrix = model_conf.astype(np.float32)
            self.confidence_matrix = self.base_confidence_matrix.copy()
            self.base_model_applied_mask = model_applied.copy()
            self.model_applied_mask = self.base_model_applied_mask.copy()
            self.occ_matrix = occ.astype(np.uint8)
            self.path_occ_matrix = path_occ.astype(np.uint8)
            self.support_mask = np.maximum((self.current_matrix > 0).astype(np.uint8), support_mask.astype(np.uint8))
            self.unresolved_mask = np.zeros((self.A, self.B), dtype=np.uint8)
            self._reset_manual_edit_state()
            self._apply_workspace_region()
            self._record_support_baseline()
            row, col = self._default_selected_cell()
            self._set_selected_cell(row, col)
            device_label_arr = meta.get("device_label")
            notice_arr = meta.get("model_notice")
            if isinstance(device_label_arr, np.ndarray) and device_label_arr.size:
                device_text = str(device_label_arr.reshape(-1)[0])
            else:
                device_text = self._device_preference()
            if isinstance(notice_arr, np.ndarray) and notice_arr.size:
                model_notice = str(notice_arr.reshape(-1)[0])
            else:
                model_notice = ""
            applied_cells = int(np.count_nonzero(self.model_applied_mask)) if self.model_applied_mask is not None else 0
            candidate_cells = int(np.count_nonzero(np.asarray(meta.get("model_candidate_mask", np.zeros((self.A, self.B))), dtype=np.uint8)))
            roi_text = " | ROI \uc801\uc6a9" if roi_mask is not None else ""
            if model_notice and "형태 기반" in device_text:
                status_text = f"\ud615\ud0dc \uae30\ubc18 \uacb0\uacfc \uc801\uc6a9{roi_text}. {model_notice}"
            elif model_notice:
                status_text = (
                    f"\ud559\uc2b5 \ubaa8\ub378 \uc801\uc6a9 \uc644\ub8cc{roi_text}. \uc2e4\uc81c \uc7a5\uce58: {device_text}. "
                    f"\ubaa8\ub378 \uac80\ud1a0 \uc140 {candidate_cells}\uac1c, \uac1c\uc785 \uc140 {applied_cells}\uac1c. {model_notice}"
                )
            else:
                status_text = (
                    f"\ud559\uc2b5 \ubaa8\ub378 \uc801\uc6a9 \uc644\ub8cc{roi_text}. \uc2e4\uc81c \uc7a5\uce58: {device_text}. "
                    f"\ubaa8\ub378 \uac80\ud1a0 \uc140 {candidate_cells}\uac1c, \uac1c\uc785 \uc140 {applied_cells}\uac1c."
                )
            self.status_var.set(self._append_gold_summary_to_status(status_text))
            self._queue_render(high_quality=True, delay=0)
            self._update_matrix_preview()
            self.canvas.focus_set()
        self._start_compute_task("모델 적용", compute, finish)

    @guarded_action
    def reset_model_state(self) -> None:
        if not RUNTIME_TRAINING_ENABLED:
            messagebox.showinfo("\ubaa8\ub378 \ucd08\uae30\ud654", "\ud604\uc7ac \ube4c\ub4dc\ub294 \uc0ac\uc6a9\uc790 \ub204\uc801 \ud559\uc2b5\uc744 \uc0ac\uc6a9\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4.", parent=self)
            return
        if not messagebox.askyesno(
            "\ud559\uc2b5 \ucd08\uae30\ud654",
            "\uc800\uc7a5\ub41c \ud559\uc2b5 \ub370\uc774\ud130\uc14b(cell_dataset.npz)\uacfc \ubaa8\ub378(cell_model.pt)\uc744 \ucd08\uae30\ud654\ud560\uae4c\uc694?\n\uac00\ub2a5\ud558\uba74 \uae30\uc874 \ud30c\uc77c\uc740 bak \ud3f4\ub354\uc5d0 \ubc31\uc5c5\ud569\ub2c8\ub2e4.",
            parent=self,
        ):
            return
        try:
            removed = reset_training_state(dataset_path=self.dataset_path, model_path=self.model_path)
            removed_count = int(sum(1 for value in removed.values() if value))
            self.status_var.set(f"\ud559\uc2b5 \ucd08\uae30\ud654 \uc644\ub8cc: \uc815\ub9ac/\ube44\ud65c\uc131\ud654\ub41c \ud30c\uc77c {removed_count}\uac1c.")
            dataset_text = "\uc608" if removed["dataset"] else "\uc544\ub2c8\uc624"
            model_text = "\uc608" if removed["model"] else "\uc544\ub2c8\uc624"
            messagebox.showinfo(
                "\ud559\uc2b5 \ucd08\uae30\ud654",
                f"\ub370\uc774\ud130\uc14b \ucd08\uae30\ud654: {dataset_text}\n\ubaa8\ub378 \ucd08\uae30\ud654: {model_text}",
                parent=self,
            )
            self._update_matrix_preview()
        except Exception as exc:
            messagebox.showerror("\ud559\uc2b5 \ucd08\uae30\ud654 \uc624\ub958", str(exc))

    @guarded_action
    def toggle_lasso_mode(self) -> None:
        if self.matrix_arrow_view:
            self.status_var.set("화살표 보기 모드에서는 LASSO를 사용할 수 없습니다. [도면 보기]로 전환하세요.")
            return
        if self.img_pil is None:
            messagebox.showwarning("LASSO", "\uba3c\uc800 \uc774\ubbf8\uc9c0\ub97c \ubd88\ub7ec\uc640 \uc8fc\uc138\uc694.")
            return
        if self.lasso_mode:
            self._clear_roi()
            self.status_var.set("LASSO \uc120\ud0dd\uc744 \ucde8\uc18c\ud588\uc2b5\ub2c8\ub2e4.")
            self._queue_render(high_quality=True, delay=0)
            return
        self.outlet_pick_mode = False
        self.lasso_mode = True
        self.roi_polygon_points = []
        self.roi_mask = None
        self.roi_cell_mask = None
        self._lasso_preview_point = None
        self.status_var.set("LASSO \uc120\ud0dd \ubaa8\ub4dc: \uce94\ubc84\uc2a4\uc5d0 \uc810\uc744 \ucc0d\uace0, \ub354\ube14\ud074\ub9ad\uc73c\ub85c \ud655\uc815\ud558\uc138\uc694. ESC\ub85c \ucde8\uc18c\ud569\ub2c8\ub2e4.")
        self.canvas.focus_set()
        self._queue_render(high_quality=True, delay=0)

    def _cell_is_inside_active_roi(self, cell: Tuple[int, int]) -> bool:
        if self.current_matrix is None:
            return False
        row, col = cell
        if not (0 <= row < self.current_matrix.shape[0] and 0 <= col < self.current_matrix.shape[1]):
            return False
        mask = self._active_edit_mask()
        return mask is None or bool(mask[row, col])

    @guarded_action
    def apply_lasso_to_matrix(self) -> None:
        if vars(self).get("source_kind") == "gis_project":
            self.clip_lasso_to_matrix()
            return
        if self.current_matrix is None or self.roi_cell_mask is None:
            messagebox.showwarning("LASSO 적용", "먼저 LASSO 구역을 선택하고 확정해 주세요.", parent=self)
            return
        if self.roi_cell_mask.shape != self.current_matrix.shape:
            messagebox.showerror("LASSO 적용", "LASSO 구역과 현재 행렬 크기가 일치하지 않습니다.", parent=self)
            return
        keep = self.roi_cell_mask.astype(bool)
        removed = int(np.count_nonzero((self.current_matrix > 0) & ~keep))
        for name in (
            "base_matrix",
            "current_matrix",
            "base_confidence_matrix",
            "confidence_matrix",
            "base_model_applied_mask",
            "model_applied_mask",
            "occ_matrix",
            "path_occ_matrix",
            "support_mask",
            "unresolved_mask",
            "user_edit_mask",
        ):
            value = vars(self).get(name)
            if isinstance(value, np.ndarray) and value.shape == keep.shape:
                value[~keep] = 0
        flood_mask = vars(self).get("_flood_cell_mask")
        if isinstance(flood_mask, np.ndarray) and flood_mask.shape == keep.shape:
            flood_mask[~keep] = 0
            self._update_flood_region_label()
        kept_outlets = {cell for cell in self._active_outlet_cells() if bool(keep[cell])}
        self._set_outlet_cells(kept_outlets, primary=self.outlet_cell)
        if self.selected_cell is None or not self._cell_is_inside_active_roi(self.selected_cell):
            row, col = self._default_selected_cell()
            self._set_selected_cell(row, col)
        self.manual_edits_since_autogen = False
        self._apply_workspace_region()
        self._invalidate_detection_cache()
        self._invalidate_matrix_arrow_cache()
        self.status_var.set(
            f"LASSO 행렬 적용 완료: 선택 구역 밖 관로 셀 {removed}개를 0으로 처리했습니다. "
            "PLENA 입력에서도 LASSO 밖은 다시 0으로 강제됩니다."
        )
        self._queue_render(high_quality=True, delay=0)
        self._update_matrix_preview()
        self.canvas.focus_set()

    @guarded_action
    def enable_outlet_pick_mode(self) -> None:
        if not self._ensure_grid():
            return
        self.lasso_mode = False
        self.outlet_pick_mode = True
        self.status_var.set("\uce94\ubc84\uc2a4\uc5d0\uc11c outlet \uc140\uc744 \ud074\ub9ad\ud574 \uc8fc\uc138\uc694. \uae30\uc874 \ubc29\ud5a5\uac12\uc740 \ubc14\ub00c\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4.")
        self.canvas.focus_set()

    def _active_outlet_cells(self) -> set[Tuple[int, int]]:
        cells = set(vars(self).get("outlet_cells") or set())
        if self.outlet_cell is not None:
            cells.add(tuple(self.outlet_cell))
        if self.current_matrix is None:
            return cells
        return {
            (int(row), int(col))
            for row, col in cells
            if 0 <= int(row) < self.current_matrix.shape[0] and 0 <= int(col) < self.current_matrix.shape[1]
        }

    def _set_outlet_cells(
        self,
        cells: set[Tuple[int, int]],
        *,
        primary: Tuple[int, int] | None = None,
    ) -> None:
        normalized = {(int(row), int(col)) for row, col in cells}
        self.outlet_cells = normalized
        self.outlet_cell = tuple(primary) if primary in normalized else next(iter(sorted(normalized)), None)

    @guarded_action
    def recommend_outlets(self) -> None:
        if not self._ensure_grid() or self.current_matrix is None:
            return
        ranked = recommend_outlet_candidates(self.current_matrix, self.support_mask, limit=12)
        if not ranked:
            messagebox.showwarning(
                "권장 토출부",
                "현재 방향 흐름에서 확인 가능한 자연 종료점이 없습니다. 순환 관로나 단절 방향을 먼저 확인해 주세요.",
                parent=self,
            )
            return
        if len(ranked) == 1:
            self._set_outlet_cells({ranked[0][0]}, primary=ranked[0][0])
            self.status_var.set(f"권장 토출부를 {ranked[0][0]}로 설정했습니다. Outlet 도달 관로 셀 {ranked[0][1]}개")
            self._queue_render(high_quality=True, delay=0)
            self._update_matrix_preview()
            return

        window = tk.Toplevel(self)
        window.title("권장 토출부")
        window.geometry("430x390")
        window.transient(self)
        frame = ttk.Frame(window, padding=14)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="현재 방향 흐름이 끝나는 토출부(Outlet) 후보", font=("Malgun Gothic", 10, "bold")).pack(anchor=tk.W)
        listbox = tk.Listbox(frame, exportselection=False, height=min(12, len(ranked)))
        for index, (cell, count) in enumerate(ranked, start=1):
            listbox.insert(tk.END, f"{index}. {cell}  |  Outlet 도달 관로 {count}셀")
        listbox.selection_set(0)
        listbox.pack(fill=tk.BOTH, expand=True, pady=(8, 10))
        mode = tk.StringVar(value="multiple")
        ttk.Radiobutton(frame, text="다중 토출부 유지", variable=mode, value="multiple").pack(anchor=tk.W)
        ttk.Radiobutton(frame, text="목록에서 선택한 토출부 하나만 사용", variable=mode, value="single").pack(anchor=tk.W, pady=(3, 0))

        def apply_recommendation() -> None:
            selected_index = int(listbox.curselection()[0]) if listbox.curselection() else 0
            primary = ranked[selected_index][0]
            cells = {cell for cell, _count in ranked} if mode.get() == "multiple" else {primary}
            self._set_outlet_cells(cells, primary=primary)
            window.destroy()
            self.status_var.set(f"권장 토출부 적용: {len(cells)}개 / 대표 Outlet {primary}")
            self._queue_render(high_quality=True, delay=0)
            self._update_matrix_preview()

        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(buttons, text="취소", command=window.destroy).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="적용", command=apply_recommendation).pack(side=tk.RIGHT, padx=(0, 6))
        window.grab_set()
        listbox.focus_set()

    @staticmethod
    def _partition_support_by_outlets(
        support_mask: np.ndarray,
        outlets: List[Tuple[int, int]],
    ) -> np.ndarray:
        labels = np.full(support_mask.shape, -1, dtype=np.int32)
        queue_cells: deque[Tuple[int, int, int]] = deque()
        rows, cols = support_mask.shape
        for label, (row, col) in enumerate(outlets):
            if 0 <= row < rows and 0 <= col < cols:
                labels[row, col] = label
                queue_cells.append((row, col, label))
        while queue_cells:
            row, col, label = queue_cells.popleft()
            for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + drow, col + dcol
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if int(support_mask[nr, nc]) != 1 or int(labels[nr, nc]) != -1:
                    continue
                labels[nr, nc] = label
                queue_cells.append((nr, nc, label))
        return labels

    def _assist_direction_matrix_for_active_outlets(
        self,
        processing_img: np.ndarray,
        *,
        matrix: Optional[np.ndarray] = None,
        occ: Optional[np.ndarray] = None,
        path_occ: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None,
        support_mask: Optional[np.ndarray] = None,
        outlets: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], np.ndarray]:
        # Explicit snapshots let a worker route pipes without temporarily
        # removing the displayed area drainage from the live workspace.
        matrix = self.current_matrix if matrix is None else matrix
        occ = self.occ_matrix if occ is None else occ
        path_occ = self.path_occ_matrix if path_occ is None else path_occ
        confidence = self.confidence_matrix if confidence is None else confidence
        support_mask = self.support_mask if support_mask is None else support_mask
        outlets = sorted(self._active_outlet_cells() if outlets is None else outlets)
        rows = int(matrix.shape[0])
        if len(outlets) <= 1:
            return assist_direction_matrix(
                processing_img,
                matrix,
                rows,
                outlet=outlets[0] if outlets else None,
                occ=occ,
                path_occ=path_occ,
                confidence=confidence,
            )
        route = np.maximum((matrix > 0).astype(np.uint8), support_mask.astype(np.uint8))
        labels = self._partition_support_by_outlets(route, outlets)
        assisted = matrix.astype(np.uint8).copy()
        returned_support = route.copy()
        unresolved = ((labels < 0) & (route > 0)).astype(np.uint8)
        stats: Dict[str, int] = {
            "head_fill": 0,
            "head_neighbor_fill": 0,
            "component_fill": 0,
            "outlet_fill": 0,
            "cycle_fix": 0,
            "unresolved_count": 0,
        }
        for label, outlet in enumerate(outlets):
            partition = labels == label
            if not np.any(partition):
                continue
            part_matrix = np.where(partition, matrix, 0).astype(np.uint8)
            part_occ = np.where(partition, occ, 0).astype(np.uint8)
            part_path = np.where(partition, path_occ, 0).astype(np.uint8)
            matrix_confidence = (
                confidence
                if isinstance(confidence, np.ndarray) and confidence.shape == matrix.shape
                else np.zeros_like(matrix, dtype=np.float32)
            )
            part_confidence = np.where(partition, matrix_confidence, 1.0).astype(np.float32)
            part_assisted, part_support, part_stats, part_unresolved = assist_direction_matrix(
                processing_img,
                part_matrix,
                rows,
                outlet=outlet,
                occ=part_occ,
                path_occ=part_path,
                confidence=part_confidence,
            )
            assisted[partition] = part_assisted[partition]
            returned_support[partition] = part_support[partition]
            unresolved[partition] = part_unresolved[partition]
            for key in stats:
                if key != "unresolved_count":
                    stats[key] += int(part_stats.get(key, 0))
        stats["unresolved_count"] = int(np.count_nonzero(unresolved))
        return assisted, returned_support, stats, unresolved

    @guarded_action
    def apply_bfs_assist(self) -> None:
        if vars(self).get("source_kind") == "gis_project":
            messagebox.showinfo("GIS Clip 필요", "먼저 행정경계를 클릭하거나 LASSO로 행렬을 생성해 주세요.", parent=self)
            return
        if not self._ensure_grid():
            return
        if self.outlet_cell is None:
            messagebox.showwarning("\ucd9c\uad6c \uc5c6\uc74c", "\uba3c\uc800 [\ucd9c\uad6c \uc120\ud0dd]\uc73c\ub85c outlet \uc140\uc744 \uc9c0\uc815\ud574 \uc8fc\uc138\uc694.")
            return
        processing_img, _roi_mask = self._processing_image_and_roi_mask()
        processing_img = processing_img.copy()
        previous_matrix = self.current_matrix.copy()
        previous_fill = vars(self).get("_area_fill_mask")
        has_fill = self._has_area_fill()
        pipe_matrix = previous_matrix.copy()
        if has_fill:
            if previous_fill.shape != pipe_matrix.shape:
                messagebox.showwarning("BFS 보조 연결", "면적 배수 마스크와 행렬 크기가 다릅니다. 채우기를 되돌려 주세요.", parent=self)
                return
            pipe_matrix[previous_fill > 0] = 0
        basin_mask = self._effective_basin_mask() if has_fill else None
        if has_fill and basin_mask is None:
            messagebox.showwarning("BFS 보조 연결", "면적 배수를 다시 채울 유역 경계가 없습니다.", parent=self)
            return
        basin_mask = basin_mask.copy() if basin_mask is not None else None
        region_mask = self._active_edit_mask()
        region_mask = region_mask.copy() if region_mask is not None else None
        active_outlets = sorted(self._active_outlet_cells())
        previous_occ = self.occ_matrix.astype(np.uint8).copy()
        previous_path = self.path_occ_matrix.astype(np.uint8).copy()
        previous_support = self.support_mask.astype(np.uint8).copy()
        previous_unresolved = self.unresolved_mask.astype(np.uint8).copy()
        previous_confidence = (
            self.confidence_matrix.astype(np.float32).copy()
            if isinstance(self.confidence_matrix, np.ndarray)
            else np.zeros_like(pipe_matrix, dtype=np.float32)
        )
        previous_model = vars(self).get("model_applied_mask")
        previous_model = previous_model.copy() if isinstance(previous_model, np.ndarray) else None
        def finish(result):
            assisted, support_mask, stats, unresolved_mask = result
            assisted = validated_direction_matrix(assisted)
            support_mask = np.asarray(support_mask, dtype=np.uint8).copy()
            unresolved_mask = np.asarray(unresolved_mask, dtype=np.uint8).copy()
            if any(value.shape != pipe_matrix.shape for value in (assisted, support_mask, unresolved_mask)):
                raise ValueError("BFS 결과와 현재 관망의 행렬 크기가 다릅니다.")
            stats = dict(stats)
            if region_mask is not None:
                assisted = self._merge_roi_matrix(pipe_matrix, assisted, region_mask)
                support_mask = self._merge_roi_matrix(previous_support, support_mask, region_mask)
                unresolved_mask = self._merge_roi_matrix(previous_unresolved, unresolved_mask, region_mask)
                outside_roi = region_mask != 1
                assisted[outside_roi] = 0
                support_mask[outside_roi] = 0
                unresolved_mask[outside_roi] = 0

            # Attach area drainage before dropping disconnected pipes. Otherwise
            # a removed nearby pipe would incorrectly redirect its catchment to
            # a farther outlet-connected pipe.
            combined = assisted.copy()
            refill_mask = None
            if has_fill:
                network_mask = region_mask if region_mask is not None else np.ones_like(assisted, dtype=np.uint8)
                if np.any((assisted > 0) & (network_mask > 0)):
                    combined, refill_mask, _fill_stats = fill_basin_to_network(
                        assisted, basin_mask, network_mask=network_mask,
                    )
                else:
                    refill_mask = np.zeros_like(assisted, dtype=np.uint8)
            reaches_outlet = direction_cells_reaching_outlets(combined, active_outlets)
            unreachable_nonzero = (combined > 0) & (reaches_outlet == 0)
            unreachable_zeroed = int(np.count_nonzero(unreachable_nonzero))
            unreachable_pipes = unreachable_nonzero & (assisted > 0)
            combined[unreachable_nonzero] = 0
            assisted[unreachable_nonzero] = 0
            unresolved_mask = np.maximum(
                unresolved_mask,
                unreachable_pipes.astype(np.uint8),
            )
            stats["unreachable_zeroed"] = unreachable_zeroed
            stats["unresolved_count"] = int(np.count_nonzero(unresolved_mask))
            # Only actual BFS pipe additions become recognition/support state.
            # Refilled area remains independently removable by the fill undo.
            newly_filled = ((pipe_matrix == 0) & (assisted > 0)).astype(np.float32)
            next_confidence = np.maximum(previous_confidence, newly_filled * 0.55)
            next_confidence[unreachable_nonzero] = 0.0
            next_model = previous_model.copy() if previous_model is not None else None
            if next_model is not None and next_model.shape == assisted.shape:
                next_model[unreachable_nonzero] = 0
            next_occ = np.maximum(previous_occ, (assisted > 0).astype(np.uint8))
            next_path = np.maximum(previous_path, (assisted > 0).astype(np.uint8))
            if refill_mask is not None:
                refill_mask = refill_mask.astype(np.uint8).copy()
                refill_mask[unreachable_nonzero] = 0
                refill_values = np.where(refill_mask > 0, combined, 0).astype(np.uint8)

            # Commit only after BFS, refill, and validation have all succeeded.
            self.current_matrix = combined
            self.confidence_matrix = next_confidence
            if next_model is not None:
                self.model_applied_mask = next_model
            self.occ_matrix = next_occ
            self.path_occ_matrix = next_path
            self.support_mask = support_mask
            self.unresolved_mask = unresolved_mask
            if refill_mask is not None:
                self._area_fill_mask = refill_mask
                self._area_fill_values = refill_values
            self._invalidate_matrix_arrow_cache()
            row, col = self.selected_cell if self.selected_cell is not None else self.outlet_cell
            self._set_selected_cell(row, col)
            self.status_var.set(
                f"BFS \ubcf4\uc870 \uc5f0\uacb0 \uc644\ub8cc: \uba38\ub9ac {stats.get('head_fill', 0)}\uce78, \uba38\ub9ac \uc778\uc811 {stats.get('head_neighbor_fill', 0)}\uce78, "
                f"\ubab8\ud1b5 {stats['component_fill']}\uce78, outlet \uc720\ub3c4 {stats['outlet_fill']}\uce78, "
                f"\ub9f4\ub3cc\uc774 \ubcf4\uc815 {stats.get('cycle_fix', 0)}\uce78, Outlet \ubbf8\ub3c4\ub2ec 0 \ucc98\ub9ac {unreachable_zeroed}\uce78, "
                f"\ubbf8\ud574\uacb0 \ud45c\uc2dc {stats.get('unresolved_count', 0)}\uce78, "
                f"출구 {len(self._active_outlet_cells())}개."
            )
            self._apply_workspace_region()
            self._queue_render(high_quality=True, delay=0)
            self._update_matrix_preview()
            self.canvas.focus_set()
        self._start_compute_task("BFS 관망 보정",
            lambda cancel: self._assist_direction_matrix_for_active_outlets(
                processing_img, matrix=pipe_matrix, occ=previous_occ,
                path_occ=previous_path, confidence=previous_confidence,
                support_mask=previous_support, outlets=active_outlets,
            ), finish)

    @guarded_action
    def clear_edits(self) -> None:
        if self.base_matrix is None:
            return
        self.current_matrix = self.base_matrix.copy()
        self._restore_support_baseline()
        if self.base_confidence_matrix is not None:
            self.confidence_matrix = self.base_confidence_matrix.copy()
        if self.base_model_applied_mask is not None:
            self.model_applied_mask = self.base_model_applied_mask.copy()
        if self.path_occ_matrix is not None:
            self.path_occ_matrix = np.maximum(self.path_occ_matrix.astype(np.uint8), (self.current_matrix > 0).astype(np.uint8))
        if self.unresolved_mask is not None:
            self.unresolved_mask = np.zeros_like(self.unresolved_mask, dtype=np.uint8)
        self._reset_manual_edit_state()
        row, col = self.selected_cell if self.selected_cell is not None else self._default_selected_cell()
        self._set_selected_cell(row, col)
        self.status_var.set("\uc218\uc815\uac12\uc744 \ub9c8\uc9c0\ub9c9 \uae30\ubcf8 \uc0c1\ud0dc\ub85c \ub418\ub3cc\ub838\uc2b5\ub2c8\ub2e4.")
        self.display_image()
        self._update_matrix_preview()
        self.canvas.focus_set()

    def _latest_gold_label_record(self):
        if self.img_bgr is None or self.current_matrix is None:
            return None
        matches = find_matching_gold_labels(
            self.gold_label_root,
            self.img_bgr,
            int(self.current_matrix.shape[0]),
            int(self.current_matrix.shape[1]),
        )
        return matches[-1] if matches else None

    def _gold_evaluation_for_matrix(self, matrix: np.ndarray) -> Tuple[Optional[object], Optional[Dict[str, object]]]:
        record = self._latest_gold_label_record()
        if record is None:
            return None, None
        metrics = evaluate_prediction(matrix, record.matrix, review_mask=record.review_mask)
        return record, metrics

    def _append_gold_summary_to_status(self, text: str) -> str:
        if self.current_matrix is None:
            return text
        try:
            _record, metrics = self._gold_evaluation_for_matrix(self.current_matrix)
        except Exception:
            return text
        if metrics is None:
            return text
        return f"{text} | {compact_metric_summary(metrics)}"

    def _save_current_gold_label(
        self,
        corrections: Optional[Dict[Tuple[int, int], int]] = None,
        full_review: bool = False,
    ) -> Optional[Path]:
        if self.img_bgr is None or self.current_matrix is None:
            return None
        if vars(self).get("source_kind") == "gis_project":
            return None
        review_mask = np.zeros_like(self.current_matrix, dtype=np.uint8)
        if full_review:
            review_mask[:, :] = 1
        else:
            for row, col in (corrections or {}):
                if 0 <= int(row) < review_mask.shape[0] and 0 <= int(col) < review_mask.shape[1]:
                    review_mask[int(row), int(col)] = 1
        if int(np.count_nonzero(review_mask)) == 0:
            return None
        metadata = {
            "source_kind": str(vars(self).get("source_kind") or ""),
            "rows": int(self.current_matrix.shape[0]),
            "cols": int(self.current_matrix.shape[1]),
            "correction_count": int(len(corrections or {})),
            "zero_correction_count": int(sum(1 for value in (corrections or {}).values() if int(value) == 0)),
            "review_scope": "full" if full_review else "corrected_cells_only",
        }
        prediction = self.base_matrix if self.base_matrix is not None else None
        return save_gold_label(
            self.gold_label_root,
            self.img_bgr,
            self.img_path or "",
            self.current_matrix,
            prediction_matrix=prediction,
            review_mask=review_mask,
            metadata=metadata,
        )

    @guarded_action
    def confirm_current_matrix_as_gold(self) -> None:
        if self.current_matrix is None or self.img_bgr is None:
            messagebox.showwarning("Gold 확정", "먼저 행렬을 생성하고 검토해 주십시오.", parent=self)
            return
        if vars(self).get("source_kind") == "gis_project":
            messagebox.showinfo("Gold 확정", "GIS 프로젝트 레이어는 이미지 gold-label로 저장하지 않습니다.", parent=self)
            return
        confirmed = messagebox.askyesno(
            "Gold 확정",
            "모든 셀의 관로 존재와 방향을 검토한 경우에만 확정하십시오.\n"
            "현재 행렬 전체를 gold-label로 저장하시겠습니까?",
            parent=self,
        )
        if not confirmed:
            return
        try:
            path = self._save_current_gold_label({}, full_review=True)
        except Exception as exc:
            messagebox.showerror("Gold 확정 오류", str(exc), parent=self)
            return
        if path is not None:
            self.status_var.set(f"전체 검토 gold-label 저장 완료: {path.name}")

    def evaluate_current_matrix_against_gold(self) -> None:
        if self.current_matrix is None:
            messagebox.showwarning("Gold 평가", "먼저 행렬을 생성해 주십시오.", parent=self)
            return
        try:
            record, metrics = self._gold_evaluation_for_matrix(self.current_matrix)
        except Exception as exc:
            messagebox.showerror("Gold 평가 오류", str(exc), parent=self)
            return
        if record is None or metrics is None:
            messagebox.showinfo(
                "Gold 평가",
                "현재 이미지/행 크기와 일치하는 gold-label이 없습니다.\n"
                "[수정 저장]은 수정한 셀만 검토값으로 기록하며, 전체 평가는 [Gold 확정]이 필요합니다.",
                parent=self,
            )
            return
        report = format_evaluation_report(metrics, record.path)
        self.status_var.set(compact_metric_summary(metrics))
        self._show_text_window("Gold-label 평가", report)

    def _collect_corrections(self) -> Dict[Tuple[int, int], int]:
        corrections: Dict[Tuple[int, int], int] = {}
        if self.base_matrix is None or self.current_matrix is None:
            return corrections
        rows, cols = self.current_matrix.shape
        edit_mask = (
            self.user_edit_mask
            if self.user_edit_mask is not None and self.user_edit_mask.shape == self.current_matrix.shape
            else np.zeros_like(self.current_matrix, dtype=np.uint8)
        )
        for i in range(rows):
            for j in range(cols):
                current = int(self.current_matrix[i, j])
                base = int(self.base_matrix[i, j])
                if (int(edit_mask[i, j]) == 1 or current != base) and current in (0, 1, 2, 3, 4):
                    corrections[(i, j)] = current
        return corrections

    @guarded_action
    def save_corrections(self) -> None:
        if vars(self).get("source_kind") == "gis_project":
            messagebox.showinfo("GIS Clip 필요", "GIS 레이어 확인 화면은 학습 자료로 저장하지 않습니다. 먼저 Clip하여 행렬을 생성해 주세요.", parent=self)
            return
        if not RUNTIME_TRAINING_ENABLED:
            messagebox.showinfo(TXT_SAVE, "\ud604\uc7ac \ube4c\ub4dc\ub294 \ubc30\ud3ec\ud615 \ubaa8\ub4dc\uc774\ubbc0\ub85c \uc0ac\uc6a9\uc790 \ub204\uc801 \ud559\uc2b5\uc744 \uc800\uc7a5\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4.", parent=self)
            return
        if not self._ensure_grid():
            return
        if self._is_compressed_mode():
            self._warn_learning_locked()
            return
        corrections = self._collect_corrections()
        if not corrections:
            gold_path = self._save_current_gold_label({})
            self.base_matrix = self.current_matrix.copy()
            if self.confidence_matrix is not None:
                self.base_confidence_matrix = self.confidence_matrix.copy()
            if self.model_applied_mask is not None:
                self.base_model_applied_mask = self.model_applied_mask.copy()
            self._reset_manual_edit_state()
            gold_text = f" Gold-label: {gold_path.name}" if gold_path is not None else ""
            self.status_var.set(
                "\ubc29\ud5a5 \uc218\uc815\uac12\uc740 \uc5c6\uc5c8\uc2b5\ub2c8\ub2e4. "
                f"\ud604\uc7ac \ud589\ub82c\uc744 \uae30\ubcf8 \uc0c1\ud0dc\ub85c \uac31\uc2e0\ud588\uc2b5\ub2c8\ub2e4.{gold_text}"
            )
            self._update_matrix_preview()
            return
        try:
            processing_cache = (
                self._detection_cache
                if self.roi_cell_mask is None and self._current_detection_cache_key() == self._detection_cache_key
                else None
            )
            X_new, X_patch_new, y_new = extract_features_and_labels(
                self.img_bgr,
                self.A,
                corrections,
                reference_matrix=self.current_matrix,
                context_radius=CONTEXT_RADIUS,
                detection_cache=processing_cache,
            )
            append_to_dataset(
                X_new,
                X_patch_new,
                y_new,
                dataset_path=self.dataset_path,
                group_key=self._current_learning_group_key(),
            )
            gold_path = self._save_current_gold_label(corrections)
            self.base_matrix = self.current_matrix.copy()
            if self.confidence_matrix is not None:
                self.base_confidence_matrix = self.confidence_matrix.copy()
            if self.model_applied_mask is not None:
                self.base_model_applied_mask = self.model_applied_mask.copy()
            zero_cells = int(sum(1 for value in corrections.values() if int(value) == 0))
            self._reset_manual_edit_state()
            gold_text = f" Gold-label: {gold_path.name}" if gold_path is not None else ""
            self.status_var.set(
                f"\uc218\uc815 \uc140 {len(corrections)}\uac1c(\uba85\uc2dc\uc801 0 \uc785\ub825 {zero_cells}\uac1c)\ub97c "
                f"\ud559\uc2b5 \uc0d8\ud50c {len(y_new)}\uac1c\ub85c \uc800\uc7a5\ud588\uc2b5\ub2c8\ub2e4.{gold_text}"
            )
            self.display_image()
            self._update_matrix_preview()
        except Exception as exc:
            messagebox.showerror("\uc800\uc7a5 \uc624\ub958", str(exc))

    @guarded_action
    def train_model(self) -> None:
        if not RUNTIME_TRAINING_ENABLED or self._is_compressed_mode() or vars(self).get("source_kind") == "gis_project":
            messagebox.showinfo("모델 학습", "학습 가능한 원본 격자에서 실행해 주세요.", parent=self)
            return
        dataset, model, device = self.dataset_path, self.model_path, self._device_preference()
        def finish(history):
            self._on_training_finished(
                float(history.get("val_best_acc", 0.0)), history.get("device_label", "cpu"),
                int(history.get("num_samples", 0)), int(history.get("num_val_groups", 0)),
                str(history.get("split_strategy", "unknown")), int(history.get("best_epoch", 0)),
                str(history.get("model_kind", "unknown")), str(history.get("coverage_reason", "")),
                str(history.get("collapse_warning", "")), str(history.get("patch_warning", "")))
        self._start_compute_task("모델 학습 · 완료 후 안전 저장",
            lambda cancel: train_cell_model(dataset_path=dataset, model_path=model, device_preference=device),
            finish, cancellable=False)

    def _on_training_finished(
        self,
        val_acc: float,
        device: str,
        num_samples: int,
        num_val_groups: int,
        split_strategy: str,
        best_epoch: int,
        model_kind: str,
        coverage_reason: str,
        collapse_warning: str,
        patch_warning: str,
    ) -> None:
        split_text = "\uadf8\ub8f9 \ubd84\ud560" if split_strategy == "group" else "\uacc4\uce35 \ubd84\ud560"
        model_text = "\uc608\uc2dc \uae30\ubc18" if model_kind == "knn" else "\uc2e0\uacbd\ub9dd"
        self.status_var.set(
            f"\ubaa8\ub378 \ud559\uc2b5 \uc644\ub8cc. \ubc29\uc2dd: {model_text}, \uc7a5\uce58: {device}, \uc0d8\ud50c: {num_samples}\uac1c, \uac80\uc99d \uadf8\ub8f9: {num_val_groups}\uac1c, \ubd84\ud560: {split_text}, \ucd5c\uc801 epoch: {best_epoch}, \ucd5c\uace0 \uac80\uc99d \uc815\ud655\ub3c4: {val_acc:.3f}"
        )
        warning_text = collapse_warning or coverage_reason or patch_warning
        if warning_text:
            messagebox.showwarning("\ud559\uc2b5 \uacbd\uace0", warning_text, parent=self)

    @staticmethod
    def _grid_crop_pixel_bounds(
        image_shape: Tuple[int, int],
        grid_shape: Tuple[int, int],
        bounds: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        image_rows, image_cols = (int(value) for value in image_shape)
        grid_rows, grid_cols = (int(value) for value in grid_shape)
        row_start, row_stop, col_start, col_stop = (int(value) for value in bounds)
        if image_rows <= 0 or image_cols <= 0 or grid_rows <= 0 or grid_cols <= 0:
            raise ValueError("image and grid shapes must be positive")
        if not (0 <= row_start < row_stop <= grid_rows and 0 <= col_start < col_stop <= grid_cols):
            raise ValueError("grid crop bounds are outside the current grid")
        y_start = int(round(row_start * image_rows / float(grid_rows)))
        y_stop = int(round(row_stop * image_rows / float(grid_rows)))
        x_start = int(round(col_start * image_cols / float(grid_cols)))
        x_stop = int(round(col_stop * image_cols / float(grid_cols)))
        y_start = int(np.clip(y_start, 0, image_rows - 1))
        y_stop = int(np.clip(y_stop, y_start + 1, image_rows))
        x_start = int(np.clip(x_start, 0, image_cols - 1))
        x_stop = int(np.clip(x_stop, x_start + 1, image_cols))
        return y_start, y_stop, x_start, x_stop

    @staticmethod
    def _world_bbox_for_grid_bounds(
        bbox: Tuple[float, float, float, float],
        grid_shape: Tuple[int, int],
        bounds: Tuple[int, int, int, int],
    ) -> Tuple[float, float, float, float]:
        xmin, ymin, xmax, ymax = (float(value) for value in bbox)
        rows, cols = (int(value) for value in grid_shape)
        row_start, row_stop, col_start, col_stop = (int(value) for value in bounds)
        if rows <= 0 or cols <= 0:
            raise ValueError("grid shape must be positive")
        if not (0 <= row_start < row_stop <= rows and 0 <= col_start < col_stop <= cols):
            raise ValueError("grid crop bounds are outside the current grid")
        x_span = xmax - xmin
        y_span = ymax - ymin
        cropped_xmin = xmin + col_start / float(cols) * x_span
        cropped_xmax = xmin + col_stop / float(cols) * x_span
        cropped_ymax = ymax - row_start / float(rows) * y_span
        cropped_ymin = ymax - row_stop / float(rows) * y_span
        return cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax

    @guarded_action
    def trim_matrix_margins(self) -> None:
        if vars(self).get("source_kind") == "gis_project":
            messagebox.showinfo(
                TXT_TRIM,
                "먼저 행정경계를 클릭하거나 LASSO로 행렬을 생성해 주세요.",
                parent=self,
            )
            return
        if not self._ensure_grid() or self.current_matrix is None or self.img_bgr is None:
            return

        bounds = self._current_trim_bounds(padding=1)
        if bounds is None:
            messagebox.showinfo(TXT_TRIM, "보존할 관로, 보조 셀 또는 Outlet이 없습니다.", parent=self)
            return
        old_shape = (int(self.A), int(self.B))
        if bounds == (0, old_shape[0], 0, old_shape[1]):
            messagebox.showinfo(
                TXT_TRIM,
                "제거할 외곽 0 여백이 없습니다. 행렬 내부의 빈 행·열은 보존합니다.",
                parent=self,
            )
            return
        row_start, row_stop, col_start, col_stop = bounds
        new_shape = (row_stop - row_start, col_stop - col_start)
        if not messagebox.askyesno(
            TXT_TRIM,
            f"현재 행렬 {old_shape[0]}x{old_shape[1]}을 {new_shape[0]}x{new_shape[1]}로 줄일까요?\n"
            "관로 바깥쪽의 연속된 빈 행·열만 제거하고 1셀 여유를 남깁니다.\n"
            "행렬 내부의 0과 방향값은 그대로 유지됩니다.",
            parent=self,
        ):
            return

        try:
            array_names = (
                "base_matrix",
                "current_matrix",
                "base_confidence_matrix",
                "confidence_matrix",
                "base_model_applied_mask",
                "model_applied_mask",
                "occ_matrix",
                "path_occ_matrix",
                "support_mask",
                "unresolved_mask",
                "user_edit_mask",
                "_flood_cell_mask",
                "roi_cell_mask",
                "_basin_cell_mask",
                "_area_fill_mask",
                "_area_fill_values",
            )
            cropped_arrays: Dict[str, Optional[np.ndarray]] = {}
            for name in array_names:
                value = vars(self).get(name)
                if value is None:
                    cropped_arrays[name] = None
                    continue
                if not isinstance(value, np.ndarray) or value.shape != old_shape:
                    raise RuntimeError(f"{name} shape does not match current grid {old_shape}")
                cropped_arrays[name] = value[row_start:row_stop, col_start:col_stop].copy()

            image_rows, image_cols = self.img_bgr.shape[:2]
            y_start, y_stop, x_start, x_stop = self._grid_crop_pixel_bounds(
                (image_rows, image_cols), old_shape, bounds
            )
            cropped_image = self.img_bgr[y_start:y_stop, x_start:x_stop].copy()
            cropped_roi_mask: Optional[np.ndarray] = None
            if self.roi_mask is not None:
                if self.roi_mask.shape != (image_rows, image_cols):
                    raise RuntimeError("ROI pixel mask shape does not match the displayed image")
                cropped_roi_mask = self.roi_mask[y_start:y_stop, x_start:x_stop].copy()

            previous_outlets = self._active_outlet_cells()
            shifted_outlets = {
                (row - row_start, col - col_start)
                for row, col in previous_outlets
                if row_start <= row < row_stop and col_start <= col < col_stop
            }
            previous_primary = self.outlet_cell
            shifted_primary = (
                (int(previous_primary[0]) - row_start, int(previous_primary[1]) - col_start)
                if previous_primary is not None
                and row_start <= int(previous_primary[0]) < row_stop
                and col_start <= int(previous_primary[1]) < col_stop
                else None
            )
            previous_selection = self.selected_cell
            shifted_selection = (
                (int(previous_selection[0]) - row_start, int(previous_selection[1]) - col_start)
                if previous_selection is not None
                and row_start <= int(previous_selection[0]) < row_stop
                and col_start <= int(previous_selection[1]) < col_stop
                else None
            )

            shifted_boxes: list[Tuple[int, int, int, int]] = []
            cropped_width = x_stop - x_start
            cropped_height = y_stop - y_start
            for box_x1, box_y1, box_x2, box_y2 in self.arrow_boxes:
                new_x1 = max(0, int(box_x1) - x_start)
                new_y1 = max(0, int(box_y1) - y_start)
                new_x2 = min(cropped_width, int(box_x2) - x_start)
                new_y2 = min(cropped_height, int(box_y2) - y_start)
                if new_x2 > new_x1 and new_y2 > new_y1:
                    shifted_boxes.append((new_x1, new_y1, new_x2, new_y2))

            current_world_bbox = vars(self).get("_gis_matrix_render_bbox")
            if current_world_bbox is None:
                current_world_bbox = vars(self).get("_gis_active_render_bbox")
            if current_world_bbox is None and isinstance(vars(self).get("_gis_matrix_meta"), dict):
                meta_bbox = self._gis_matrix_meta.get("bbox")
                if isinstance(meta_bbox, (list, tuple)) and len(meta_bbox) == 4:
                    current_world_bbox = tuple(float(value) for value in meta_bbox)
            cropped_world_bbox = (
                self._world_bbox_for_grid_bounds(tuple(current_world_bbox), old_shape, bounds)
                if current_world_bbox is not None and vars(self).get("source_kind") == "gis"
                else None
            )

            previous_trim_bounds = self.trim_bounds
            previous_trim_source_shape = self.trim_source_shape
            previous_trim_span = (
                (
                    int(previous_trim_bounds[1]) - int(previous_trim_bounds[0]),
                    int(previous_trim_bounds[3]) - int(previous_trim_bounds[2]),
                )
                if previous_trim_bounds is not None
                else None
            )
            if (
                previous_trim_source_shape is not None
                and previous_trim_bounds is not None
                and previous_trim_span == old_shape
            ):
                cumulative_bounds = (
                    int(previous_trim_bounds[0]) + row_start,
                    int(previous_trim_bounds[0]) + row_stop,
                    int(previous_trim_bounds[2]) + col_start,
                    int(previous_trim_bounds[2]) + col_stop,
                )
                trim_source_shape = previous_trim_source_shape
            else:
                cumulative_bounds = bounds
                trim_source_shape = old_shape

            self._clip_workspace_masks(bounds)
            for name, value in cropped_arrays.items():
                setattr(self, name, value)
            self.A, self.B = new_shape
            self.rows_var.set(self.A)
            self.trim_source_shape = tuple(int(value) for value in trim_source_shape)
            self.trim_bounds = tuple(int(value) for value in cumulative_bounds)
            self.trimmed_grid_shape = new_shape
            if self.compressed_grid_shape is not None:
                self.compressed_grid_shape = new_shape

            self._roi_mask_pil = None
            self._set_display_image_from_bgr(cropped_image)
            self.roi_mask = cropped_roi_mask
            if cropped_roi_mask is not None:
                self._roi_mask_pil = Image.fromarray(cropped_roi_mask.astype(np.uint8), mode="L")
                self._build_roi_mask_pyramid()
            else:
                self._roi_mask_pyramid = []
            self.roi_polygon_points = [
                (float(x) - x_start, float(y) - y_start) for x, y in self.roi_polygon_points
            ]
            if self._lasso_preview_point is not None:
                self._lasso_preview_point = (
                    float(self._lasso_preview_point[0]) - x_start,
                    float(self._lasso_preview_point[1]) - y_start,
                )
            self.arrow_boxes = shifted_boxes
            self._set_outlet_cells(shifted_outlets, primary=shifted_primary)
            self.outlet_pick_mode = False

            if cropped_world_bbox is not None:
                self._gis_matrix_render_bbox = cropped_world_bbox
            if isinstance(self._gis_matrix_meta, dict):
                self._gis_matrix_meta["rows_A"] = int(self.A)
                self._gis_matrix_meta["cols_B"] = int(self.B)
                self._gis_matrix_meta["nonzero_cells"] = int(np.count_nonzero(self.current_matrix))
                self._gis_matrix_meta["direction_cell_counts"] = {
                    str(code): int(np.count_nonzero(self.current_matrix == code)) for code in range(5)
                }
                self._gis_matrix_meta["trim_source_shape"] = [int(value) for value in self.trim_source_shape]
                self._gis_matrix_meta["trim_bounds"] = [int(value) for value in self.trim_bounds]
                if cropped_world_bbox is not None:
                    self._gis_matrix_meta["bbox"] = [float(value) for value in cropped_world_bbox]
                self._update_gis_cell_size_label(self._gis_matrix_meta)

            self._invalidate_detection_cache()
            self._invalidate_matrix_arrow_cache()
            self._gis_viewport_render_cache.clear()
            self._gis_viewport_render_generation += 1
            self._gis_viewport_pending_keys.clear()
            self._update_flood_region_label()
            if shifted_selection is None:
                shifted_selection = self._default_selected_cell()
            self._set_selected_cell(*shifted_selection)
            self.reset_view()
            self._update_matrix_preview()
            self.canvas.focus_set()
            removed_rows = old_shape[0] - new_shape[0]
            removed_cols = old_shape[1] - new_shape[1]
            removed_cells = old_shape[0] * old_shape[1] - new_shape[0] * new_shape[1]
            self.status_var.set(
                f"외곽 0 여백 제거 완료: {old_shape[0]}x{old_shape[1]} -> "
                f"{new_shape[0]}x{new_shape[1]} | 행 {removed_rows}개, 열 {removed_cols}개, "
                f"외곽 셀 {removed_cells}개 제거 | 내부 0 보존"
            )
        except Exception as exc:
            messagebox.showerror(TXT_TRIM, str(exc), parent=self)

    def _ask_compressed_shape(self) -> Optional[Tuple[int, int]]:
        default_rows = max(2, self.A // 2)
        answer = simpledialog.askstring(
            TXT_COMPRESS,
            f"\uc555\ucd95\ud560 \ud589 \uac1c\uc218(A)\ub97c \uc785\ub825\ud574 \uc8fc\uc138\uc694.\n\ud604\uc7ac: {self.A}x{self.B}\n\uc5f4 \uac1c\uc218(B)\ub294 \ud604\uc7ac \ube44\uc728\uc5d0 \ub9de\ucdb0 \uc790\ub3d9 \uacc4\uc0b0\ub429\ub2c8\ub2e4.\n\uc608: 5",
            initialvalue=str(default_rows),
            parent=self,
        )
        if answer is None:
            return None
        match = re.fullmatch(r"\s*(\d+)\s*", answer)
        if not match:
            messagebox.showwarning(TXT_COMPRESS, "\ud589 \uac1c\uc218\ub294 5 \ucc98\ub7fc \uc22b\uc790 \ud558\ub098\ub85c \uc785\ub825\ud574 \uc8fc\uc138\uc694.")
            return None
        target_a = int(match.group(1))
        target_b = max(2, int(round(target_a * self.B / max(float(self.A), 1.0))))
        target_b = min(target_b, self.B)
        if target_a < 2 or target_b < 2:
            messagebox.showwarning(TXT_COMPRESS, "\uc555\ucd95 \ud589\ub82c\uc740 \ucd5c\uc18c 2x2 \uc774\uc0c1\uc774\uc5b4\uc57c \ud569\ub2c8\ub2e4.")
            return None
        if target_a > self.A or target_b > self.B:
            messagebox.showwarning(TXT_COMPRESS, "\ud589\ub82c \uc555\ucd95\uc740 \ud604\uc7ac \ud06c\uae30\ubcf4\ub2e4 \ud06c\uac8c \uc9c0\uc815\ud560 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4.")
            return None
        if target_a == self.A and target_b == self.B:
            messagebox.showinfo(TXT_COMPRESS, "\ud604\uc7ac \ud06c\uae30\uc640 \uac19\uc740 \ud06c\uae30\uc785\ub2c8\ub2e4.")
            return None
        return target_a, target_b

    @guarded_action
    def compress_matrix(self) -> None:
        if self.current_matrix is None:
            messagebox.showwarning("\ud589\ub82c \uc5c6\uc74c", "\uba3c\uc800 \ud589\ub82c \uacb0\uacfc\ub97c \uc0dd\uc131\ud574 \uc8fc\uc138\uc694.")
            return
        if self.outlet_cell is None:
            messagebox.showwarning("\ucd9c\uad6c \uc5c6\uc74c", "\ud589\ub82c \uc555\ucd95 \uc804\uc5d0 \uba3c\uc800 outlet \uc140\uc744 \uc120\ud0dd\ud574 \uc8fc\uc138\uc694.")
            return
        if len(self._active_outlet_cells()) > 1:
            messagebox.showwarning(TXT_COMPRESS, "행렬 압축은 단일 Outlet만 지원합니다. [권장 토출부]에서 하나의 토출부를 선택해 주세요.")
            return
        target_shape = self._ask_compressed_shape()
        if target_shape is None:
            return
        target_a, target_b = target_shape
        if not messagebox.askyesno(
            TXT_COMPRESS,
            f"\ud604\uc7ac \ud589\ub82c {self.A}x{self.B}\ub97c {target_a}x{target_b}\ub85c \uc555\ucd95\ud558\uc5ec \uad50\uccb4\ud560\uae4c\uc694?\n\uc5f4 \uac1c\uc218(B)\ub294 \ud604\uc7ac \ube44\uc728\uc744 \ub530\ub77c \uc790\ub3d9 \uacc4\uc0b0\ub410\uc2b5\ub2c8\ub2e4.\n\uc6d0\ubcf8 \uc778\uc2dd \ub2e8\uacc4\ub294 \ubc14\ub00c\uc9c0 \uc54a\uace0, \ud604\uc7ac \uacb0\uacfc\ub9cc \ucd95\uc18c\ub429\ub2c8\ub2e4.",
            parent=self,
        ):
            return
        try:
            prev_shape = (self.A, self.B)
            previous_roi = vars(self).get("roi_cell_mask")
            previous_basin = vars(self).get("_basin_cell_mask")
            previous_flood_mask = (
                self._flood_cell_mask.copy()
                if isinstance(vars(self).get("_flood_cell_mask"), np.ndarray)
                and self._flood_cell_mask.shape == self.current_matrix.shape
                else None
            )
            compressed, support_mask, compressed_outlet, meta = compress_direction_matrix(
                self.current_matrix.astype(np.uint8),
                target_a,
                target_b,
                outlet=self.outlet_cell,
            )
            next_roi = resample_mask_centers(previous_roi, (target_a, target_b)) if previous_roi is not None else None
            next_basin = resample_mask_centers(previous_basin, (target_a, target_b)) if previous_basin is not None else None
            for boundary in (next_roi, next_basin if vars(self).get("_basin_is_lasso") else None):
                if boundary is not None and np.any((compressed > 0) & (boundary == 0)):
                    raise ValueError("압축 격자가 유역 경계를 넘거나 관망을 자릅니다. 더 세밀한 격자로 압축해 주세요.")
            self.compressed_grid_shape = (target_a, target_b)
            self.compression_source_shape = prev_shape
            self.A, self.B = target_a, target_b
            self.rows_var.set(target_a)
            self.base_matrix = compressed.astype(np.uint8).copy()
            self.current_matrix = self.base_matrix.copy()
            confidence = np.asarray(meta.get("confidence", np.zeros((target_a, target_b))), dtype=np.float32)
            self.base_confidence_matrix = confidence.copy()
            self.confidence_matrix = confidence.copy()
            self.base_model_applied_mask = np.zeros((target_a, target_b), dtype=np.uint8)
            self.model_applied_mask = self.base_model_applied_mask.copy()
            support = support_mask.astype(np.uint8)
            self.occ_matrix = support.copy()
            self.path_occ_matrix = support.copy()
            self.support_mask = support.copy()
            self.unresolved_mask = np.zeros((target_a, target_b), dtype=np.uint8)
            self.user_edit_mask = np.zeros((target_a, target_b), dtype=np.uint8)
            self._flood_cell_mask = (
                self._resize_binary_mask_any(previous_flood_mask, target_a, target_b)
                if previous_flood_mask is not None
                else None
            )
            self._update_flood_region_label()
            self.manual_edits_since_autogen = False
            self.outlet_cell = compressed_outlet
            self.outlet_cells = {compressed_outlet} if compressed_outlet is not None else set()
            self.arrow_boxes = []
            self.roi_cell_mask = next_roi
            self._basin_cell_mask = next_basin
            self.lasso_mode = False
            self._invalidate_detection_cache()
            self._record_support_baseline()
            row, col = compressed_outlet if compressed_outlet is not None else self._default_selected_cell()
            self._set_selected_cell(row, col)
            dropped = int(meta.get("dropped_cells", 0))
            nonzero = int(np.count_nonzero(self.current_matrix))
            self.status_var.set(
                f"\ud589\ub82c \uc555\ucd95 \uc644\ub8cc: {prev_shape[0]}x{prev_shape[1]} -> {target_a}x{target_b}, \uad00\ub85c \uc140 {nonzero}\uac1c, \uc555\ucd95 \uacfc\uc815\uc5d0\uc11c \uc81c\uc678\ub41c \uc140 {dropped}\uac1c. \uc555\ucd95 \ud589\ub82c\uc5d0\uc11c\ub294 \uc140 \ud559\uc2b5\uc774 \ube44\ud65c\uc131\ud654\ub429\ub2c8\ub2e4."
            )
            self._queue_render(high_quality=True, delay=0)
            self._update_matrix_preview()
            self.canvas.focus_set()
        except Exception as exc:
            messagebox.showerror(TXT_COMPRESS, str(exc))

    @staticmethod
    def _sanitize_output_name(value: str) -> str:
        cleaned = re.sub(r'[<>:"/\\|?*]+', "_", str(value).strip())
        cleaned = re.sub(r"\s+", "_", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned).strip(" ._")
        return cleaned or "NFMAT_matrix"

    @staticmethod
    def _format_output_angle(degrees: float) -> str:
        normalized = float(degrees) % 360.0
        if abs(normalized) < 1e-8 or abs(normalized - 360.0) < 1e-8:
            normalized = 0.0
        if abs(normalized - round(normalized)) < 1e-8:
            return str(int(round(normalized)))
        return f"{normalized:.6f}".rstrip("0").rstrip(".")

    def _output_dataset_name(self) -> str:
        source_name = Path(self.img_path).stem if self.img_path else "NFMAT_matrix"
        boundary_value = vars(self).get("_gis_boundary_value")
        if boundary_value:
            source_name = f"{source_name}_{boundary_value}"
        safe_source_name = self._sanitize_output_name(source_name)
        angle_text = self._format_output_angle(self._rotation_degrees())
        return f"{safe_source_name}_row{int(self.A)}_Angle{angle_text}"

    @guarded_action
    def export_matrix(self) -> None:
        if vars(self).get("source_kind") == "gis_project":
            messagebox.showinfo("GIS Clip 필요", "먼저 행정경계를 클릭하거나 LASSO로 행렬을 생성해 주세요.", parent=self)
            return
        if self.current_matrix is None:
            messagebox.showwarning("\ud589\ub82c \uc5c6\uc74c", "\uba3c\uc800 \uc774\ubbf8\uc9c0\ub97c \ubd88\ub7ec\uc624\uace0 \ud589\ub82c\uc744 \uc0dd\uc131\ud574 \uc8fc\uc138\uc694.")
            return
        if self.outlet_cell is None:
            messagebox.showwarning("\ucd9c\uad6c \uc5c6\uc74c", "PLENA \uc785\ub825 \uc790\ub8cc\ub97c \ub0b4\ubcf4\ub0b4\ub824\uba74 \uba3c\uc800 outlet \uc140\uc744 \uc120\ud0dd\ud574 \uc8fc\uc138\uc694.")
            return
        output_name = self._output_dataset_name()
        path = filedialog.asksaveasfilename(
            title="PLENA \uc785\ub825 \uc790\ub8cc \ub0b4\ubcf4\ub0b4\uae30",
            initialdir=str(self.output_root),
            initialfile=f"{output_name}.txt",
            defaultextension=".txt",
            filetypes=[("PLENA \uc785\ub825 TXT", "*.txt")],
        )
        if not path:
            return
        try:
            path = str(Path(path).with_suffix(".txt"))
            self._write_plena_input_file(path)
            cleanup = (
                f" | 0 처리: {self._plena_sanitization_summary()}"
                if int(vars(self).get("last_plena_sanitized_count", 0)) > 0
                else ""
            )
            self.status_var.set(
                f"PLENA \uc785\ub825 \uc790\ub8cc \ub0b4\ubcf4\ub0b4\uae30 \uc644\ub8cc: {os.path.basename(path)}{cleanup}"
            )
        except Exception as exc:
            messagebox.showerror("\ub0b4\ubcf4\ub0b4\uae30 \uc624\ub958", str(exc))

    def toggle_matrix_arrow_view(self) -> None:
        if self.current_matrix is None:
            messagebox.showwarning("행렬 없음", "먼저 행렬을 생성해 주세요.", parent=self)
            return
        if not self.matrix_arrow_view:
            try:
                self._ensure_matrix_arrow_image()
            except Exception as exc:
                messagebox.showerror("화살표 보기 오류", str(exc), parent=self)
                return
            self.matrix_arrow_view = True
            if hasattr(self, "matrix_arrow_button"):
                self.matrix_arrow_button.configure(text="도면 보기")
            self.lasso_mode = False
            self.outlet_pick_mode = False
            self.status_var.set("행렬 화살표 보기 모드입니다. 편집하려면 [도면 보기]로 전환하세요.")
        else:
            self.matrix_arrow_view = False
            if hasattr(self, "matrix_arrow_button"):
                self.matrix_arrow_button.configure(text=TXT_MATRIX_ARROWS)
            self.status_var.set("도면 보기 모드입니다.")
        self.reset_view()

    def show_matrix_arrows(self) -> None:
        self.toggle_matrix_arrow_view()

    def _save_matrix_arrow_image(self, image_bgr: np.ndarray) -> None:
        path = filedialog.asksaveasfilename(
            title="행렬 화살표 PNG 저장",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            parent=self,
        )
        if not path:
            return
        try:
            cv2.imwrite(path, image_bgr)
            self.status_var.set(f"행렬 화살표 PNG 저장 완료: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("PNG 저장 오류", str(exc), parent=self)

    def _default_plena_input_path(self) -> str:
        output_name = self._output_dataset_name()
        out_dir = self.output_root / output_name
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / f"{output_name}.txt")

    def _normalize_plena_outlet_state(self) -> Tuple[Tuple[int, int], int]:
        if self.outlet_cell is None:
            raise RuntimeError("outlet 셀이 선택되지 않았습니다.")
        row, col = (int(self.outlet_cell[0]), int(self.outlet_cell[1]))
        if self.current_matrix is not None:
            rows, cols = self.current_matrix.shape
            if not (0 <= row < rows and 0 <= col < cols):
                raise RuntimeError("outlet 셀이 현재 행렬 범위를 벗어났습니다.")

        primary = (row, col)
        active = self._active_outlet_cells()
        reduced = len(active - {primary})
        return primary, int(max(0, reduced))

    def _matrix_for_plena(
        self,
        protected_outlets: Optional[set[Tuple[int, int]]] = None,
    ) -> Tuple[np.ndarray, int, int]:
        if self.current_matrix is None:
            raise RuntimeError("\ud589\ub82c\uc774 \uc5c6\uc2b5\ub2c8\ub2e4.")
        matrix = validated_direction_matrix(self.current_matrix)
        removed_lasso = 0
        roi_cell_mask = self._active_edit_mask()
        if roi_cell_mask is not None:
            if not isinstance(roi_cell_mask, np.ndarray) or roi_cell_mask.shape != matrix.shape:
                raise RuntimeError(
                    f"LASSO 구역 크기 {getattr(roi_cell_mask, 'shape', None)}가 현재 행렬 크기 {matrix.shape}와 일치하지 않습니다. "
                    "LASSO를 다시 적용해 주세요."
                )
            outside_roi = roi_cell_mask != 1
            removed_lasso = int(np.count_nonzero((matrix > 0) & outside_roi))
            matrix[outside_roi] = 0

        outlets = protected_outlets if protected_outlets is not None else self._active_outlet_cells()
        reaches_outlet = direction_cells_reaching_outlets(matrix, sorted(outlets))
        unreachable = (matrix > 0) & (reaches_outlet == 0)
        removed_unreachable = int(np.count_nonzero(unreachable))
        matrix[unreachable] = 0
        return matrix, removed_lasso, removed_unreachable

    def _plena_sanitization_summary(self) -> str:
        parts: List[str] = []
        lasso_removed = int(vars(self).get("last_plena_lasso_removed_count", 0))
        unreachable_removed = int(vars(self).get("last_plena_unreachable_count", 0))
        if lasso_removed > 0:
            parts.append(f"LASSO 밖 {lasso_removed}칸")
        if unreachable_removed > 0:
            parts.append(f"Outlet 미도달 {unreachable_removed}칸")
        return ", ".join(parts) if parts else "정리 없음"

    def _write_plena_input_file(self, path: str) -> str:
        (row, col), reduced_outlets = self._normalize_plena_outlet_state()
        matrix, removed_lasso, removed_unreachable = self._matrix_for_plena({(row, col)})
        if int(matrix[row, col]) == 0:
            raise RuntimeError("Outlet은 유역 내부의 유효한 관망 셀이어야 합니다.")
        rows, cols = matrix.shape
        lines = [f"{rows} {cols}", f"{row + 1} {col + 1}"]
        lines.extend(" ".join(str(int(value)) for value in values) for values in matrix)
        atomic_write_text(path, "\n".join(lines) + "\n")
        self.last_plena_lasso_removed_count = int(removed_lasso)
        self.last_plena_unreachable_count = int(removed_unreachable)
        self.last_plena_sanitized_count = int(removed_lasso + removed_unreachable)
        self.last_plena_outlet_reduced_count = int(reduced_outlets)
        self._last_plena_input_matrix = matrix.copy()
        return path

    def _show_text_window(self, title: str, text: str) -> None:
        window = tk.Toplevel(self)
        window.title(title)
        window.geometry("900x700")
        if self._icon_photo is not None:
            try:
                window.iconphoto(True, self._icon_photo)
            except Exception:
                pass
        viewer = scrolledtext.ScrolledText(window, font=("Malgun Gothic", 10))
        viewer.pack(fill=tk.BOTH, expand=True)
        viewer.insert("1.0", text)
        viewer.configure(state=tk.DISABLED)

    def _resolve_plena_output_path(self, raw_path: Optional[str], base_dir: Optional[Path] = None) -> Optional[Path]:
        if not raw_path:
            return None
        cleaned = str(raw_path).strip().strip('"')
        if not cleaned:
            return None
        path = Path(cleaned)
        candidates: List[Path] = []
        if path.is_absolute():
            candidates.append(path)
        else:
            if base_dir is not None:
                candidates.append(base_dir / path)
            candidates.append(Path.cwd() / path)
            candidates.append(path)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0] if candidates else path

    def _copy_pil_image_to_clipboard(self, image: Image.Image) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Image clipboard copy is only implemented on Windows.")
        import ctypes
        import io
        from ctypes import wintypes

        output = io.BytesIO()
        image.convert("RGB").save(output, "BMP")
        dib_data = output.getvalue()[14:]
        if not dib_data:
            raise RuntimeError("Clipboard image data is empty.")

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.OpenClipboard.restype = wintypes.BOOL
        user32.EmptyClipboard.argtypes = []
        user32.EmptyClipboard.restype = wintypes.BOOL
        user32.SetClipboardData.argtypes = [wintypes.UINT, ctypes.c_void_p]
        user32.SetClipboardData.restype = ctypes.c_void_p
        user32.CloseClipboard.argtypes = []
        user32.CloseClipboard.restype = wintypes.BOOL
        kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        kernel32.GlobalAlloc.restype = ctypes.c_void_p
        kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalUnlock.restype = wintypes.BOOL
        kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
        kernel32.GlobalFree.restype = ctypes.c_void_p

        handle = kernel32.GlobalAlloc(0x0002, len(dib_data))
        if not handle:
            raise RuntimeError("GlobalAlloc failed.")
        try:
            locked = kernel32.GlobalLock(handle)
            if not locked:
                raise RuntimeError("GlobalLock failed.")
            ctypes.memmove(locked, dib_data, len(dib_data))
            kernel32.GlobalUnlock(handle)
            if not user32.OpenClipboard(self.winfo_id()):
                raise RuntimeError("OpenClipboard failed.")
            try:
                user32.EmptyClipboard()
                transferred = user32.SetClipboardData(8, handle)
                if not transferred:
                    raise RuntimeError("SetClipboardData failed.")
                handle = None
            finally:
                user32.CloseClipboard()
        finally:
            if handle:
                kernel32.GlobalFree(handle)

    def open_plena_plot_window(
        self,
        width_csv: Optional[str] = None,
        nse_path: Optional[str] = None,
        beta_rows: Optional[List[Dict[str, str]]] = None,
        gamma_result: Optional[GammaResult] = None,
    ) -> None:
        if width_csv is None and nse_path is None and not beta_rows and gamma_result is None:
            selected = filedialog.askopenfilename(
                title="PLENA 결과 파일 열기",
                initialdir=str(self.output_root),
                filetypes=[
                    ("PLENA width/NSE", ("*width_functions.csv", "*_NSE.txt")),
                    ("Width function CSV", "*width_functions.csv"),
                    ("NSE text", "*_NSE.txt"),
                    ("All files", "*.*"),
                ],
                parent=self,
            )
            if not selected:
                return
            lower = selected.lower()
            if lower.endswith("width_functions.csv"):
                width_csv = selected
            elif lower.endswith("_nse.txt"):
                nse_path = selected
            else:
                messagebox.showwarning("PLENA 그림", "width_functions.csv 또는 _NSE.txt 파일을 선택해 주십시오.", parent=self)
                return

        blocks = []
        nse_points = []
        errors: List[str] = []
        width_path = self._resolve_plena_output_path(width_csv)
        if width_path is not None and width_path.exists():
            try:
                blocks = load_width_function_blocks(width_path)
            except Exception as exc:
                errors.append(f"폭 함수 CSV 해석 실패: {exc}")
            inferred = infer_nse_path_from_width_csv(width_path)
            if inferred is not None and nse_path is None:
                nse_path = str(inferred)
        elif width_csv:
            errors.append(f"폭 함수 CSV를 찾지 못했습니다: {width_csv}")

        resolved_nse = self._resolve_plena_output_path(nse_path)
        if resolved_nse is not None and resolved_nse.exists():
            try:
                nse_points = load_nse_points(resolved_nse)
            except Exception as exc:
                errors.append(f"NSE 파일 해석 실패: {exc}")
        elif beta_rows:
            nse_points = nse_points_from_summary_rows(beta_rows)
        elif nse_path:
            errors.append(f"NSE 파일을 찾지 못했습니다: {nse_path}")

        items: List[Tuple[str, str, object]] = []
        if gamma_result is not None:
            items.append(("Gamma (Figure 6)", "gamma", gamma_result))
        for block in blocks:
            items.append((f"Width {block.label} ({block.run_count} runs)", "width", block))
        if nse_points:
            nse_points = sorted(nse_points, key=lambda point: point.k)
            items.append(("NSE summary", "nse", nse_points))

        if not items:
            messagebox.showerror("PLENA 그림", "\n".join(errors) if errors else "그릴 PLENA 결과가 없습니다.", parent=self)
            return

        window = tk.Toplevel(self)
        window.title("PLENA plots")
        window.geometry("1040x760")
        if self._icon_photo is not None:
            try:
                window.iconphoto(True, self._icon_photo)
            except Exception:
                pass

        state: Dict[str, object] = {"zoom": 1.0, "image": None, "photo": None}
        toolbar = ttk.Frame(window, padding=(8, 8, 8, 4))
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, text="Plot").pack(side=tk.LEFT)
        choice_var = tk.StringVar(value=items[0][0])
        combo = ttk.Combobox(toolbar, textvariable=choice_var, values=[item[0] for item in items], state="readonly", width=34)
        combo.pack(side=tk.LEFT, padx=(6, 10))
        zoom_label = ttk.Label(toolbar, text="100%")
        zoom_label.pack(side=tk.LEFT, padx=(0, 10))

        frame = ttk.Frame(window)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        canvas = tk.Canvas(frame, background="#ffffff", highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)

        def current_item() -> Tuple[str, str, object]:
            label = choice_var.get()
            for item in items:
                if item[0] == label:
                    return item
            return items[0]

        def render() -> None:
            _, kind, payload = current_item()
            zoom = float(state["zoom"])
            if kind == "width":
                size = (int(900 * zoom), int(650 * zoom))
                image = render_width_function_plot(payload, size=size)  # type: ignore[arg-type]
            elif kind == "nse":
                size = (int(720 * zoom), int(650 * zoom))
                image = render_nse_plot(payload, size=size)  # type: ignore[arg-type]
            else:
                size = (int(900 * zoom), int(650 * zoom))
                image = render_gamma_plot(payload, size=size)  # type: ignore[arg-type]
            photo = ImageTk.PhotoImage(image)
            state["image"] = image
            state["photo"] = photo
            canvas.delete("all")
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            canvas.configure(scrollregion=(0, 0, image.width, image.height))
            zoom_label.configure(text=f"{int(round(zoom * 100))}%")

        def set_zoom(value: float) -> None:
            state["zoom"] = min(5.0, max(0.4, float(value)))
            render()

        def save_png() -> None:
            image = state.get("image")
            if not isinstance(image, Image.Image):
                render()
                image = state.get("image")
            if not isinstance(image, Image.Image):
                return
            safe_label = re.sub(r"[^0-9A-Za-z_.-]+", "_", choice_var.get()).strip("_") or "plena_plot"
            path = filedialog.asksaveasfilename(
                title="PLENA 그림 PNG 저장",
                initialdir=str(self.output_root),
                initialfile=f"{safe_label}.png",
                defaultextension=".png",
                filetypes=[("PNG image", "*.png")],
                parent=window,
            )
            if path:
                image.save(path)
                self.status_var.set(f"PLENA 그림 저장 완료: {os.path.basename(path)}")

        def save_png_300dpi() -> None:
            _label, kind, payload = current_item()
            if kind == "width":
                export_image = render_width_function_plot(payload, size=(1800, 1300))  # type: ignore[arg-type]
            elif kind == "nse":
                export_image = render_nse_plot(payload, size=(1440, 1300))  # type: ignore[arg-type]
            else:
                export_image = render_gamma_plot(payload, size=(1800, 1300))  # type: ignore[arg-type]
            safe_label = re.sub(r"[^0-9A-Za-z_.-]+", "_", choice_var.get()).strip("_") or "plena_plot"
            path = filedialog.asksaveasfilename(
                title="PLENA 그림 300DPI PNG 저장",
                initialdir=str(self.output_root),
                initialfile=f"{safe_label}_300dpi.png",
                defaultextension=".png",
                filetypes=[("PNG image", "*.png")],
                parent=window,
            )
            if path:
                export_image.save(path, dpi=(300, 300))
                self.status_var.set(f"PLENA 300DPI 그림 저장 완료: {os.path.basename(path)}")

        def copy_png() -> None:
            image = state.get("image")
            if not isinstance(image, Image.Image):
                render()
                image = state.get("image")
            if not isinstance(image, Image.Image):
                return
            try:
                self._copy_pil_image_to_clipboard(image)
                self.status_var.set("PLENA 그림을 클립보드에 복사했습니다.")
            except Exception as exc:
                messagebox.showerror("PLENA 그림 복사 오류", str(exc), parent=window)

        ttk.Button(toolbar, text="-", width=3, command=lambda: set_zoom(float(state["zoom"]) / 1.2)).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="+", width=3, command=lambda: set_zoom(float(state["zoom"]) * 1.2)).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(toolbar, text="100%", command=lambda: set_zoom(1.0)).pack(side=tk.LEFT, padx=(6, 10))
        ttk.Button(toolbar, text="PNG 저장", command=save_png).pack(side=tk.RIGHT)
        ttk.Button(toolbar, text="300DPI 저장", command=save_png_300dpi).pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Button(toolbar, text="복사", command=copy_png).pack(side=tk.RIGHT, padx=(0, 6))
        combo.bind("<<ComboboxSelected>>", lambda _event: render())
        canvas.bind("<MouseWheel>", lambda event: (set_zoom(float(state["zoom"]) * (1.15 if event.delta > 0 else 1 / 1.15)), "break")[1])
        render()

    def _ensure_plena_progress_window(self) -> None:
        if self._plena_progress_window is not None and self._plena_progress_window.winfo_exists():
            return
        window = tk.Toplevel(self)
        window.title("PLENA 진행 상황")
        window.geometry("980x720")
        if self._icon_photo is not None:
            try:
                window.iconphoto(True, self._icon_photo)
            except Exception:
                pass
        toolbar = ttk.Frame(window, padding=(8, 8, 8, 0))
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, text="PLENA 실행 로그").pack(side=tk.LEFT)
        ttk.Button(toolbar, text="중단", command=self._terminate_plena_process).pack(side=tk.RIGHT)
        viewer = scrolledtext.ScrolledText(window, font=("Consolas", 10))
        viewer.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        viewer.configure(state=tk.DISABLED)
        self._plena_progress_window = window
        self._plena_progress_text = viewer

    def _append_plena_progress(self, text: str) -> None:
        self._ensure_plena_progress_window()
        if self._plena_progress_text is None:
            return
        self._plena_progress_text.configure(state=tk.NORMAL)
        self._plena_progress_text.insert(tk.END, text)
        self._plena_progress_text.see(tk.END)
        self._plena_progress_text.configure(state=tk.DISABLED)

    def _terminate_plena_process(self) -> None:
        proc = self._plena_process
        if proc is None or proc.poll() is not None:
            self.status_var.set("PLENA 실행 중인 프로세스가 없습니다.")
            return
        try:
            proc.terminate()
            self.status_var.set("PLENA 중단 요청을 보냈습니다.")
            self.after(5000, lambda proc=proc: self._kill_plena_process_if_alive(proc))
        except Exception as exc:
            messagebox.showerror("PLENA 중단 오류", str(exc))

    def _kill_plena_process_if_alive(self, proc: subprocess.Popen[bytes]) -> None:
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.kill()
            self.status_var.set("PLENA가 응답하지 않아 강제 종료했습니다.")
            self._append_plena_progress("\n[PLENA 강제 종료]\n")
        except Exception as exc:
            messagebox.showerror("PLENA 강제 종료 오류", str(exc))

    def _plena_progress_status(self, line: str, elapsed: float, batch_status: str = "") -> str:
        stripped = line.strip()
        prefix = f"PLENA 실행 중... {elapsed:.0f}초 경과, 시간 제한 없음"
        if batch_status:
            prefix = f"{prefix} | {batch_status}"
        if not stripped:
            return prefix
        beta_match = re.search(r"beta\s*=?\s*([0-9.eE+-]+)", stripped)
        k_match = re.search(r"\bk\s*=?\s*([-+]?\d+)", stripped)
        run_match = re.search(r"\brun\s*=?\s*(\d+)", stripped, flags=re.IGNORECASE)
        pieces: List[str] = []
        if k_match:
            pieces.append(f"k={k_match.group(1)}")
        if beta_match:
            pieces.append(f"beta={beta_match.group(1)}")
        if run_match:
            pieces.append(f"run={run_match.group(1)}")
        if "Batch completed" in stripped:
            pieces.append("배치 완료")
        if pieces:
            return f"{prefix} | {' / '.join(pieces)}"
        return f"{prefix} | {stripped[:120]}"

    def _decode_process_output(self, raw: bytes) -> str:
        if not raw:
            return ""
        for encoding in ("cp949", "euc-kr", "utf-8"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("cp949", errors="replace")

    def _read_text_best_effort(self, path: Path) -> str:
        raw = path.read_bytes()
        return self._decode_process_output(raw)


    def _plena_result_path(self, input_path: str) -> Path:
        path = Path(input_path)
        candidates = [
            path.with_name(f"{path.stem}_result{path.suffix}"),
            path.with_name(f"{path.stem}_결과{path.suffix}"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _parse_plena_result_matrix(self, result_path: Path) -> Optional[np.ndarray]:
        if not result_path.exists():
            return None
        text = self._read_text_best_effort(result_path)
        lines = text.splitlines()
        start = None
        for idx, line in enumerate(lines):
            if "Final Direction Matrix D" in line:
                start = idx + 1
                break
        if start is None:
            return None

        rows: List[List[int]] = []
        for line in lines[start:]:
            stripped = line.strip()
            if not stripped:
                if rows:
                    break
                continue
            if stripped.startswith("//------") or stripped.startswith("["):
                break
            try:
                row = [int(token) for token in stripped.split()]
            except ValueError:
                break
            rows.append(row)
        if not rows:
            return None
        return np.asarray(rows, dtype=np.uint8)

    def _parse_plena_beta_summary(self, text: str) -> List[Dict[str, str]]:
        lines = text.splitlines()
        start = None
        for idx, line in enumerate(lines):
            if "[Summary by beta]" in line or ("beta" in line and "meanNSE(run)" in line):
                start = idx + 1
                break
        if start is None:
            return []

        rows: List[Dict[str, str]] = []
        for line in lines[start:]:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("NSE saved:") or stripped.startswith("Saved:") or "width_functions.csv" in stripped:
                break
            if stripped.startswith("k") or stripped.startswith("beta"):
                continue
            parts = re.split(r"\s+", stripped)
            if len(parts) < 5 or not re.fullmatch(r"[-+]?\d+", parts[0]):
                continue
            row = {
                "k": parts[0],
                "beta": parts[1],
                "ok_total": parts[2],
                "mean_nse_run": parts[3],
                "nse_mean_q": parts[4],
            }
            if len(parts) >= 7:
                row["min_nse"] = parts[4]
                row["max_nse"] = parts[5]
                row["nse_mean_q"] = parts[6]
            rows.append(row)
        return rows

    def _extract_plena_saved_paths(self, text: str) -> Dict[str, str]:
        info: Dict[str, str] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Done:"):
                info["result_path"] = line.split(":", 1)[1].strip()
            elif line.startswith("NSE saved:"):
                info["nse_path"] = line.split(":", 1)[1].strip()
            elif line.startswith("Saved:"):
                payload = line.split(":", 1)[1].strip()
                if "width_functions.csv" in payload:
                    info["width_csv"] = payload
                else:
                    info.setdefault("saved_misc", payload)
            elif "width_functions.csv" in line and "Saved:" not in line:
                info["width_csv"] = line.split(":", 1)[1].strip() if ":" in line else line
            elif "MADE BY" in line:
                info["made_by"] = line
            elif "Press Enter to exit" in line or "Input stream is closed" in line:
                info["exit_hint"] = line
            elif "[Computation time]" in line:
                info["compute_time"] = line
            elif "Batch completed." in line:
                info["batch_time"] = line
        return info

    def _format_plena_output_summary(self, output: str) -> str:
        beta_rows = self._parse_plena_beta_summary(output)
        saved = self._extract_plena_saved_paths(output)

        lines: List[str] = ["PLENA 요약"]
        if "compute_time" in saved:
            lines.append(saved["compute_time"])
        if "batch_time" in saved:
            lines.append(saved["batch_time"])

        if beta_rows:
            lines.append("")
            lines.append("[beta별 mean NSE]")
            lines.append("beta\tmeanNSE(run)\tNSE(mean_q)")
            for row in beta_rows:
                lines.append(f"{row['beta']}\t{row['mean_nse_run']}\t{row['nse_mean_q']}")
            scored_rows: List[Tuple[float, Dict[str, str]]] = []
            for row in beta_rows:
                try:
                    scored_rows.append((float(row["mean_nse_run"]), row))
                except Exception:
                    continue
            if scored_rows:
                best = max(scored_rows, key=lambda item: item[0])[1]
                lines.append("")
                lines.append(f"최고 mean NSE beta: {best['beta']} (meanNSE(run)={best['mean_nse_run']})")

        if saved:
            lines.append("")
            if self.last_plena_sanitized_count > 0:
                lines.append(f"PLENA 입력 전 {self._plena_sanitization_summary()}을 0으로 정리했습니다.")
            if "result_path" in saved:
                lines.append(f"결과 txt: {saved['result_path']}")
            if "nse_path" in saved:
                lines.append(f"NSE 저장: {saved['nse_path']}")
            if "width_csv" in saved:
                lines.append(f"폭 함수 CSV 저장: {saved['width_csv']}")
            if "saved_misc" in saved:
                lines.append(f"추가 저장: {saved['saved_misc']}")
            if "made_by" in saved:
                lines.append(saved["made_by"])
            if "exit_hint" in saved:
                lines.append(saved["exit_hint"])

        if len(lines) == 1:
            return output or "(no output)"
        return "\n".join(lines)

    def _plena_match_summary(self, result_matrix: Optional[np.ndarray]) -> str:
        reference = vars(self).get("_last_plena_input_matrix")
        if not isinstance(reference, np.ndarray):
            reference = self.current_matrix
        if result_matrix is None or reference is None:
            return "PLENA 결과 행렬을 해석하지 못했습니다."
        if result_matrix.shape != reference.shape:
            return f"PLENA 행렬 크기 불일치: PLENA={result_matrix.shape}, 입력={reference.shape}"

        total = int(result_matrix.size)
        exact = int(np.count_nonzero(result_matrix == reference))
        original_pipe_mask = reference > 0
        reconstructed_pipe_mask = result_matrix > 0
        pipe_union_mask = original_pipe_mask | reconstructed_pipe_mask
        pipe_intersection_mask = original_pipe_mask & reconstructed_pipe_mask
        pipe_union_total = int(np.count_nonzero(pipe_union_mask))
        pipe_intersection_total = int(np.count_nonzero(pipe_intersection_mask))
        direction_match = int(np.count_nonzero((result_matrix == reference) & pipe_intersection_mask))

        lines = [
            "PLENA 입력 행렬과 재구성한 행렬의 일치율",
            f"- 전체 셀 일치: {exact}/{total} ({(100.0 * exact / max(total, 1)):.2f}%)",
            f"- 관로 영역 일치: {pipe_intersection_total}/{pipe_union_total} ({(100.0 * pipe_intersection_total / max(pipe_union_total, 1)):.2f}%)",
        ]
        if pipe_intersection_total > 0 and direction_match != pipe_intersection_total:
            lines.append(
                f"- 두 행렬이 모두 관로라고 본 셀에서 방향까지 일치: {direction_match}/{pipe_intersection_total} ({(100.0 * direction_match / max(pipe_intersection_total, 1)):.2f}%)"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_gamma_output_summary(
        result: GammaResult,
        summary_path: Path,
        distribution_path: Path,
    ) -> str:
        return "\n".join(
            [
                "감마 지수 (MATLAB beta2 / GravityCenterRatio)",
                "- 계산 대상: PLENA Final Direction Matrix D",
                "- 식: Gamma = sum(L^2 x count) / [sum(L x count) x max(L)]",
                f"- outlet (1-based): row {result.outlet[0] + 1}, col {result.outlet[1] + 1}",
                f"- 기여 셀: {result.contributing_cells}",
                f"- 최장 유로 제거 횟수: {result.removed_lengths_cell.size}",
                f"- 무게중심 X: {result.gravity_center_x:.10f}",
                f"- 최대 제거 유로 길이: {result.maximum_length}",
                f"- Gamma: {result.gamma:.10f}",
                f"- 상세 결과: {summary_path}",
                f"- Figure 6 분포 CSV: {distribution_path}",
            ]
        )

    def _run_plena_worker(self, plena_exe: str, input_path: str, stdin_data: str, batch_status: str = "") -> None:
        proc: Optional[subprocess.Popen[bytes]] = None
        try:
            proc = subprocess.Popen(
                [plena_exe, input_path],
                cwd=str(Path(input_path).parent),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._plena_process = proc
            if self._plena_cancelled:
                proc.terminate()
            if proc.stdin is not None:
                proc.stdin.write(stdin_data.encode("ascii", errors="ignore"))
                proc.stdin.flush()
                proc.stdin.close()

            started = time.monotonic()
            chunks: List[str] = []
            stdout_queue: queue.Queue[Optional[bytes]] = queue.Queue()

            def _read_stdout() -> None:
                try:
                    if proc is not None and proc.stdout is not None:
                        while True:
                            raw_line = proc.stdout.readline()
                            if not raw_line:
                                break
                            stdout_queue.put(raw_line)
                finally:
                    stdout_queue.put(None)

            threading.Thread(target=_read_stdout, daemon=True).start()
            last_line = ""
            while True:
                try:
                    raw_line = stdout_queue.get(timeout=1.0)
                except queue.Empty:
                    elapsed = time.monotonic() - started
                    self._post_plena_callback(
                        0,
                        lambda line=last_line, elapsed=elapsed, batch_status=batch_status: self.status_var.set(
                            self._plena_progress_status(line, elapsed, batch_status)
                        ),
                    )
                    continue
                if raw_line is None:
                    break
                line = self._decode_process_output(raw_line)
                last_line = line
                chunks.append(line)
                elapsed = time.monotonic() - started
                self._post_plena_callback(0, lambda text=line: self._append_plena_progress(text))
                self._post_plena_callback(
                    0,
                    lambda line=line, elapsed=elapsed, batch_status=batch_status: self.status_var.set(
                        self._plena_progress_status(line, elapsed, batch_status)
                    ),
                )

            proc.wait()
            output = "".join(chunks)
            code = int(proc.returncode or 0)
            result_path = self._plena_result_path(input_path)
            result_matrix = self._parse_plena_result_matrix(result_path) if code == 0 else None
            match_summary = self._plena_match_summary(result_matrix) if code == 0 else ""
            output_summary = self._format_plena_output_summary(output) if code == 0 else (output or "(no output)")
            gamma_result: Optional[GammaResult] = None
            gamma_summary = ""
            if code == 0:
                try:
                    input_matrix, outlet = load_plena_input_matrix(input_path)
                    if result_matrix is None:
                        raise ValueError(f"PLENA 결과 행렬을 찾지 못했습니다: {result_path}")
                    if result_matrix.shape != input_matrix.shape:
                        raise ValueError(
                            f"PLENA 결과 행렬 크기가 입력과 다릅니다: result={result_matrix.shape}, input={input_matrix.shape}"
                        )
                    gamma_result = compute_gamma_index(result_matrix, outlet=outlet)
                    gamma_text_path, gamma_csv_path = save_gamma_outputs(input_path, gamma_result)
                    gamma_summary = self._format_gamma_output_summary(gamma_result, gamma_text_path, gamma_csv_path)
                except Exception as exc:
                    gamma_summary = f"감마 지수 계산 실패\n- {exc}"

            def _finish() -> None:
                if self._close_requested:
                    return
                if self._plena_cancelled:
                    self.status_var.set("PLENA 중단 완료 · 부분 결과는 완료 결과로 사용하지 않습니다.")
                    return
                if code == 0:
                    self.status_var.set(
                        f"PLENA \uc2e4\ud589 \uc644\ub8cc: {os.path.basename(input_path)} | \uacb0\uacfc \ud3f4\ub354: {Path(input_path).parent}"
                    )
                else:
                    self.status_var.set(f"PLENA \uc2e4\ud589 \uc2e4\ud328(return code={code}). \ucd9c\ub825 \ucc3d\uc744 \ud655\uc778\ud574 \uc8fc\uc138\uc694.")
                combined = output_summary + (f"\n\n{match_summary}" if match_summary else "")
                if gamma_summary:
                    combined += f"\n\n{gamma_summary}"
                self._append_plena_progress(f"\n[PLENA \uc644\ub8cc]\n{combined}\n")
                self._show_text_window("PLENA \uc2e4\ud589 \ucd9c\ub825", combined)
                if code == 0:
                    saved = self._extract_plena_saved_paths(output)
                    beta_rows = self._parse_plena_beta_summary(output)
                    base_dir = Path(input_path).parent
                    width_plot_path = self._resolve_plena_output_path(saved.get("width_csv"), base_dir)
                    nse_plot_path = self._resolve_plena_output_path(saved.get("nse_path"), base_dir)
                    if width_plot_path is not None or nse_plot_path is not None or beta_rows or gamma_result is not None:
                        try:
                            self.open_plena_plot_window(
                                str(width_plot_path) if width_plot_path is not None else None,
                                str(nse_plot_path) if nse_plot_path is not None else None,
                                beta_rows,
                                gamma_result,
                            )
                        except Exception as exc:
                            self.status_var.set(f"PLENA 그림 표시 실패: {exc}")

            self._post_plena_callback(0, _finish)
        except Exception as exc:
            msg = str(exc)
            self._post_plena_callback(0, lambda msg=msg: messagebox.showerror("PLENA \uc2e4\ud589 \uc624\ub958", msg))
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
            self._post_plena_callback(0, self._finish_plena_task)

    @guarded_action
    def run_plena(self) -> None:
        if vars(self).get("source_kind") == "gis_project":
            messagebox.showinfo("GIS Clip 필요", "먼저 행정경계를 클릭하거나 LASSO로 행렬을 생성해 주세요.", parent=self)
            return
        if self.current_matrix is None:
            messagebox.showwarning("\ud589\ub82c \uc5c6\uc74c", "\uba3c\uc800 \ud589\ub82c\uc744 \uc0dd\uc131\ud574 \uc8fc\uc138\uc694.")
            return
        if self.outlet_cell is None:
            messagebox.showwarning("\ucd9c\uad6c \uc5c6\uc74c", "\uba3c\uc800 outlet \uc140\uc744 \uc120\ud0dd\ud574 \uc8fc\uc138\uc694.")
            return
        if self._plena_process is not None and self._plena_process.poll() is None:
            self._ensure_plena_progress_window()
            messagebox.showwarning("PLENA \uc2e4\ud589 \uc911", "\uc774\ubbf8 PLENA\uac00 \uc2e4\ud589 \uc911\uc785\ub2c8\ub2e4. \ud544\uc694\ud558\uba74 \uc9c4\ud589 \ucc3d\uc5d0\uc11c \uc911\ub2e8\ud558\uc138\uc694.")
            return

        plena_path = _resource_path("PLENA.exe")
        if not plena_path.exists():
            messagebox.showerror(
                "PLENA 없음",
                "PLENA.exe 파일을 찾지 못했습니다.\n"
                f"검색 위치: {_resource_search_text('PLENA.exe')}",
            )
            return
        plena_exe = str(plena_path)

        beta_exp = simpledialog.askfloat(
            "PLENA beta",
            "beta = 10^a \ud615\ud0dc\uc5d0\uc11c a \uac12\uc744 \uc785\ub825\ud574 \uc8fc\uc138\uc694.",
            initialvalue=6.0,
            parent=self,
        )
        if beta_exp is None:
            return
        if not np.isfinite(beta_exp) or not -300 <= beta_exp <= 300:
            messagebox.showwarning("PLENA 설정", "beta 지수는 -300~300 범위의 유한한 값이어야 합니다.", parent=self)
            return

        do_width = messagebox.askyesno("PLENA \ud3ed \ud568\uc218", "\ud3ed \ud568\uc218(width function)\ub3c4 \ud568\uaed8 \uacc4\uc0b0\ud560\uae4c\uc694?", parent=self)
        do_batch = False
        custom_range = False
        k_start = -4
        k_end = 3
        runs = 100
        max_threads = 0
        if do_width:
            do_batch = messagebox.askyesno(
                "PLENA \ubc30\uce58",
                "\uc5ec\ub7ec beta \ubc30\uce58 \uacc4\uc0b0\ub3c4 \ud568\uaed8 \uc2e4\ud589\ud560\uae4c\uc694?",
                parent=self,
            )
            if do_batch:
                custom_range = messagebox.askyesno(
                    "PLENA \ubc30\uce58 \uc124\uc815",
                    "\uae30\ubcf8 beta \ubc94\uc704(k=-4~3, run=100)\ub97c \uadf8\ub300\ub85c \uc0ac\uc6a9\ud560\uae4c\uc694?",
                    parent=self,
                )
                if not custom_range:
                    k_start = simpledialog.askinteger("k start", "k \uc2dc\uc791\uac12\uc744 \uc785\ub825\ud574 \uc8fc\uc138\uc694.", initialvalue=-4, parent=self)
                    if k_start is None:
                        return
                    k_end = simpledialog.askinteger("k end", "k \uc885\ub8cc\uac12\uc744 \uc785\ub825\ud574 \uc8fc\uc138\uc694.", initialvalue=3, parent=self)
                    if k_end is None:
                        return
                    runs = simpledialog.askinteger("runs", "beta\ub2f9 \ubc18\ubcf5 \ud69f\uc218(run)\ub97c \uc785\ub825\ud574 \uc8fc\uc138\uc694.", initialvalue=100, parent=self)
                    if runs is None:
                        return
                max_threads_val = simpledialog.askinteger(
                    "max_threads",
                    "\uc2a4\ub808\ub4dc \uc0c1\ud55c(max_threads)\uc744 \uc785\ub825\ud574 \uc8fc\uc138\uc694. 0\uc740 \uc790\ub3d9\uc785\ub2c8\ub2e4.",
                    initialvalue=0,
                    parent=self,
                )
                if max_threads_val is None:
                    return
                max_threads = max_threads_val

        if do_batch and (int(k_start) > int(k_end) or not -300 <= int(k_start) <= int(k_end) <= 300 or int(runs) < 1 or int(max_threads) < 0):
            messagebox.showwarning("PLENA 설정", "시작 지수 ≤ 종료 지수, 반복 ≥ 1, 스레드 ≥ 0으로 설정해 주세요.", parent=self)
            return
        try:
            dataset = self._output_dataset_name()
            run_dir = self.output_root / dataset / "runs" / (time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000000:09d}")
            input_path = self._write_plena_input_file(str(run_dir / f"{dataset}.txt"))
            atomic_write_text(run_dir / "run_settings.json", json.dumps({
                "app": "NFMAT6", "beta_exponent": beta_exp, "width_function": bool(do_width),
                "batch": bool(do_batch), "k_start": k_start, "k_end": k_end, "runs": runs,
                "max_threads": max_threads, "shape": list(self.current_matrix.shape),
                "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                "area_fill_cells": int(np.count_nonzero(vars(self).get("_area_fill_mask"))) if self._has_area_fill() else 0,
                "outside_removed": self.last_plena_lasso_removed_count,
                "unreachable_removed": self.last_plena_unreachable_count,
                "selected_outlet": list(self.outlet_cell)
            }, ensure_ascii=False, indent=2))
        except Exception as exc:
            messagebox.showerror("PLENA 입력 저장 오류", str(exc), parent=self)
            return

        stdin_lines = [str(beta_exp), "y" if do_width else "n"]
        if do_width:
            stdin_lines.append("y" if do_batch else "n")
            if do_batch:
                stdin_lines.append("n" if custom_range else "y")
                if not custom_range:
                    stdin_lines.extend([str(k_start), str(k_end), str(max(1, int(runs)))])
                stdin_lines.append(str(max_threads))
        stdin_data = "\n".join(stdin_lines) + "\n"
        beta_label = f"beta=10^{float(beta_exp):g}"
        if do_width and do_batch:
            thread_label = "auto" if int(max_threads) <= 0 else str(int(max_threads))
            batch_status = f"{beta_label}, 배치 k={int(k_start)}..{int(k_end)}, run={max(1, int(runs))}, threads={thread_label}"
        elif do_width:
            batch_status = f"{beta_label}, 폭 함수 계산, 배치 없음"
        else:
            batch_status = f"{beta_label}, 폭 함수 없음, 배치 없음"

        if self.last_plena_sanitized_count > 0:
            self.status_var.set(
                f"PLENA \uc2e4\ud589 \uc911... \uc2dc\uac04 \uc81c\ud55c \uc5c6\uc74c | {batch_status} | "
                f"{self._plena_sanitization_summary()}\uc744 0\uc73c\ub85c \uc81c\uac70\ud55c \uc785\ub825 \ud30c\uc77c: {os.path.basename(input_path)}"
            )
        else:
            self.status_var.set(f"PLENA 실행 중... 시간 제한 없음 | {batch_status} | 입력 파일: {os.path.basename(input_path)}")
        self._ensure_plena_progress_window()
        self._append_plena_progress(
            f"PLENA 시작: {os.path.basename(input_path)}\n"
            f"작업 폴더: {Path(input_path).parent}\n"
            f"배치 상태: {batch_status}\n"
            "시간 제한: 없음\n\n"
        )
        self._plena_starting = True
        self._plena_cancelled = False
        self._task_cancellable = True
        self._task_label = "PLENA · " + batch_status
        self._sync_workspace_task_controls()
        self.after(50, self._poll_plena_callbacks)
        threading.Thread(
            target=self._run_plena_worker,
            args=(plena_exe, input_path, stdin_data, batch_status),
            daemon=True,
        ).start()

    def _event_to_cell(self, event: tk.Event) -> Optional[Tuple[int, int]]:
        if self.img_pil is None or self.current_matrix is None:
            return None
        point = self._image_point_from_event(event)
        if point is None:
            return None
        image_x, image_y = point
        row = int(np.clip(np.floor(image_y / max(self.img_pil.size[1] / float(self.A), 1e-6)), 0, self.A - 1))
        col = int(np.clip(np.floor(image_x / max(self.img_pil.size[0] / float(self.B), 1e-6)), 0, self.B - 1))
        return row, col

    def on_canvas_click(self, event: tk.Event) -> None:
        if not self._workspace_mutation_allowed():
            return
        if self.matrix_arrow_view:
            self.status_var.set("화살표 보기 모드입니다. 편집하려면 [도면 보기]로 전환하세요.")
            return
        if bool(vars(self).get("_gis_render_polling", False)):
            self.status_var.set("GIS 전체 지도를 렌더링하는 중입니다. 완료 후 경계를 선택하세요.")
            return
        if vars(self).get("source_kind") == "gis_project":
            if self.lasso_mode:
                point = self._image_point_from_event(event)
                if point is None:
                    return
                self.roi_polygon_points.append(point)
                self._lasso_preview_point = point
                self.status_var.set(f"LASSO 점 {len(self.roi_polygon_points)}개 추가. 더블클릭으로 확정하세요.")
                self._queue_render(high_quality=False, delay=0)
                return
            world_point = self._world_point_from_event(event)
            matched = self._boundary_at_world_point(world_point) if world_point is not None else None
            if matched is None:
                self.status_var.set("선택된 행정경계가 없습니다. 경계 내부를 클릭하거나 경계 레이어 표시 상태를 확인하세요.")
                return
            try:
                self._clip_boundary_feature_to_matrix(*matched)
            except Exception as exc:
                messagebox.showerror("경계 클릭 변환 오류", str(exc), parent=self)
            return
        if not self._ensure_grid():
            return
        if self.lasso_mode:
            point = self._image_point_from_event(event)
            if point is None:
                return
            self.roi_polygon_points.append(point)
            self._lasso_preview_point = point
            self.status_var.set(f"LASSO \uc810 {len(self.roi_polygon_points)}\uac1c \ucd94\uac00. \ub354\ube14\ud074\ub9ad\uc73c\ub85c \ud655\uc815\ud558\uc138\uc694.")
            self._queue_render(high_quality=False, delay=0)
            return
        cell = self._event_to_cell(event)
        if cell is None:
            return
        if self.outlet_pick_mode:
            if self._has_area_fill():
                fill_mask = self._area_fill_mask
                if fill_mask.shape != self.current_matrix.shape or fill_mask[cell] or not self.current_matrix[cell]:
                    self.status_var.set("면적 배수 셀이나 빈 칸은 Outlet으로 선택할 수 없습니다. 기존 관로 셀을 선택해 주세요.")
                    return
            if not self._cell_is_inside_active_roi(cell):
                self.status_var.set("LASSO 적용 구역 밖의 셀은 Outlet으로 선택할 수 없습니다.")
                self.canvas.focus_set()
                return
            self._set_outlet_cells({cell}, primary=cell)
            self.outlet_pick_mode = False
            self._set_selected_cell(cell[0], cell[1])
            self.status_var.set(f"outlet \uc140\uc744 ({cell[0]}, {cell[1]})\ub85c \uc124\uc815\ud588\uc2b5\ub2c8\ub2e4. \uc774\uc81c [BFS \ubcf4\uc870 \uc5f0\uacb0]\uc744 \uc2e4\ud589\ud560 \uc218 \uc788\uc2b5\ub2c8\ub2e4.")
            self._queue_render(high_quality=True, delay=0)
            self._update_matrix_preview()
            self.canvas.focus_set()
            return
        self._set_selected_cell(cell[0], cell[1])
        self._queue_render(high_quality=True, delay=0)
        self.canvas.focus_set()

    def on_canvas_double_click(self, event: tk.Event) -> None:
        if not self._workspace_mutation_allowed():
            return
        if self.matrix_arrow_view:
            return
        if not self.lasso_mode:
            return
        point = self._image_point_from_event(event)
        if point is not None:
            self.roi_polygon_points.append(point)
        if len(self.roi_polygon_points) < 3:
            self.status_var.set("LASSO\ub97c \ud655\uc815\ud558\ub824\uba74 \ucd5c\uc18c 3\uac1c \uc810\uc774 \ud544\uc694\ud569\ub2c8\ub2e4.")
            return
        self.roi_cell_mask = self._roi_cell_mask_from_polygon()
        self.lasso_mode = False
        self._lasso_preview_point = None
        roi_cells = int(np.count_nonzero(self.roi_cell_mask)) if self.roi_cell_mask is not None else 0
        if vars(self).get("source_kind") == "gis_project":
            self.status_var.set("GIS LASSO 구역을 확정했습니다. [GIS 방향 보정] 또는 [LASSO → 행렬]을 누르면 선택 구역 관로만 계산합니다.")
        else:
            self.status_var.set(f"LASSO \uad6c\uc5ed \ud655\uc815: \uc120\ud0dd \uc140 {roi_cells}\uac1c. \uc774\uc81c \uc790\ub3d9 \uc778\uc2dd/\ubaa8\ub378/BFS\ub97c \ubd80\ubd84 \uc2e4\ud589\ud560 \uc218 \uc788\uc2b5\ub2c8\ub2e4.")
        self._queue_render(high_quality=True, delay=0)

    def on_canvas_motion(self, event: tk.Event) -> None:
        if self.matrix_arrow_view:
            return
        if bool(vars(self).get("_gis_render_polling", False)):
            return
        if vars(self).get("source_kind") == "gis_project" and not self.lasso_mode:
            self._gis_hover_canvas_point = (float(event.x), float(event.y))
            if self._gis_hover_after_id is None:
                self._gis_hover_after_id = self.after(18, self._process_gis_hover)
            return
        if not self.lasso_mode:
            return
        point = self._image_point_from_event(event)
        self._lasso_preview_point = point
        self._queue_render(high_quality=False, delay=20)

    def _process_gis_hover(self) -> None:
        self._gis_hover_after_id = None
        point = self._gis_hover_canvas_point
        world_point = self._world_point_from_canvas_xy(*point) if point is not None else None
        matched = self._boundary_at_world_point(world_point) if world_point is not None else None
        if matched == self._gis_hover_boundary:
            return
        self._gis_hover_boundary = matched
        if matched is not None:
            layer = self._gis_layers.get(matched[0])
            if layer is not None:
                boundary = list(layer["geometry"])[matched[1]]
                _field, value = self._boundary_feature_label(
                    boundary,
                    matched[1],
                    str(layer.get("label_field", "") or "") or None,
                )
                self.status_var.set(f"{value} 클릭 → 해당 구역 관로 행렬 변환")
        self._refresh_gis_boundary_overlay()

    def on_pan_start(self, event: tk.Event) -> None:
        if vars(self).get("_gis_hover_after_id") is not None:
            self.after_cancel(self._gis_hover_after_id)
            self._gis_hover_after_id = None
        self._pan_start = (int(event.x), int(event.y))
        self._pan_last_canvas = self._pan_start
        self._pan_origin = (float(self._display_origin[0]), float(self._display_origin[1]))

    def on_pan_drag(self, event: tk.Event) -> None:
        if self._pan_start is None:
            return
        dx = int(event.x) - self._pan_start[0]
        dy = int(event.y) - self._pan_start[1]
        self._display_origin = (self._pan_origin[0] + dx, self._pan_origin[1] + dy)
        self._display_origin_initialized = True
        if vars(self).get("source_kind") in {"gis_project", "gis"} and self._pan_last_canvas is not None:
            step_x = int(event.x) - self._pan_last_canvas[0]
            step_y = int(event.y) - self._pan_last_canvas[1]
            self.canvas.move("all", step_x, step_y)
            self._pan_last_canvas = (int(event.x), int(event.y))
            return
        self._queue_render(high_quality=False, delay=15)
        self._queue_render(high_quality=True, delay=120)

    def on_pan_end(self, _event: tk.Event) -> None:
        self._pan_start = None
        self._pan_last_canvas = None
        if vars(self).get("source_kind") in {"gis_project", "gis"}:
            self._queue_render(high_quality=True, delay=0)

    @staticmethod
    def _zoom_origin_for_anchor(
        origin: Tuple[float, float],
        old_scale: float,
        new_scale: float,
        anchor: Tuple[float, float],
    ) -> Tuple[float, float]:
        safe_old_scale = max(float(old_scale), 1e-9)
        image_x = (float(anchor[0]) - float(origin[0])) / safe_old_scale
        image_y = (float(anchor[1]) - float(origin[1])) / safe_old_scale
        return (
            float(anchor[0]) - image_x * float(new_scale),
            float(anchor[1]) - image_y * float(new_scale),
        )

    def on_mouse_wheel(self, event: tk.Event) -> None:
        display_img = self._active_display_image()
        if display_img is None:
            return
        old_scale = max(self._scale, 1e-6)
        factor = 1.1 if event.delta > 0 else 1.0 / 1.1
        new_zoom = float(np.clip(self._zoom * factor, ZOOM_MIN, ZOOM_MAX))
        if abs(new_zoom - self._zoom) <= 1e-12:
            return
        self._zoom = new_zoom
        canvas_w = max(100, self.canvas.winfo_width())
        canvas_h = max(100, self.canvas.winfo_height())
        img_w, img_h = display_img.size
        self._fit_scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        new_scale = max(self._fit_scale * self._zoom, 1e-6)
        self._display_origin = self._zoom_origin_for_anchor(
            self._display_origin,
            old_scale,
            new_scale,
            (float(event.x), float(event.y)),
        )
        self._display_origin_initialized = True
        self._scale = new_scale
        self._display_size = (max(1, int(round(img_w * new_scale))), max(1, int(round(img_h * new_scale))))
        if vars(self).get("source_kind") == "gis_project":
            self._queue_render(high_quality=False, delay=0)
            self._queue_render(high_quality=True, delay=140)
        else:
            self._queue_render(high_quality=False, delay=0)
            self._queue_render(high_quality=True, delay=120)

    def _move_selection(self, drow: int, dcol: int) -> None:
        if not self._ensure_grid():
            return
        if self.selected_cell is None:
            self._set_selected_cell(0, 0)
        assert self.selected_cell is not None
        row = self.selected_cell[0] + drow
        col = self.selected_cell[1] + dcol
        self._set_selected_cell(row, col)
        self._queue_render(high_quality=True, delay=0)

    def _set_selected_value(self, value: int) -> None:
        if not self._workspace_mutation_allowed() or self._has_area_fill():
            self.status_var.set("계산 또는 면적 배수 적용 중입니다. 완료/되돌리기 후 편집해 주세요.")
            return
        if not self._ensure_grid():
            return
        if self.selected_cell is None:
            self._set_selected_cell(0, 0)
        assert self.current_matrix is not None
        assert self.selected_cell is not None
        row, col = self.selected_cell
        if not self._cell_is_inside_active_roi((row, col)):
            self.status_var.set("LASSO 적용 구역 밖의 셀은 항상 0으로 유지되므로 편집할 수 없습니다.")
            self.canvas.focus_set()
            return
        self.current_matrix[row, col] = np.uint8(value)
        if self.user_edit_mask is not None:
            self.user_edit_mask[row, col] = 1
        if self.model_applied_mask is not None:
            self.model_applied_mask[row, col] = 0
        self.manual_edits_since_autogen = True
        if self.occ_matrix is not None:
            self.occ_matrix[row, col] = np.uint8(1 if value != 0 else 0)
        if self.path_occ_matrix is not None:
            self.path_occ_matrix[row, col] = np.uint8(1 if value != 0 else 0)
        if self.confidence_matrix is not None:
            self.confidence_matrix[row, col] = np.float32(1.0 if value != 0 else 0.0)
        self._set_selected_cell(row, col)
        self._queue_render(high_quality=True, delay=0)
        self._update_matrix_preview()
        self.canvas.focus_set()

    def on_canvas_key(self, event: tk.Event) -> None:
        if self.matrix_arrow_view:
            self.status_var.set("화살표 보기 모드입니다. 편집하려면 [도면 보기]로 전환하세요.")
            return
        if not self._ensure_grid():
            return
        key = event.keysym
        if vars(self).get("source_kind") == "gis_project" and not self.lasso_mode:
            self.status_var.set("GIS 레이어 확인 화면에서는 셀을 직접 편집하지 않습니다. 먼저 Clip하여 행렬을 생성하세요.")
            return
        if key == "Left":
            self._move_selection(0, -1)
            return
        if key == "Right":
            self._move_selection(0, 1)
            return
        if key == "Up":
            self._move_selection(-1, 0)
            return
        if key == "Down":
            self._move_selection(1, 0)
            return
        if key in {"BackSpace", "Delete"}:
            self._set_selected_value(0)
            return
        if key == "Escape" and self.lasso_mode:
            self._clear_roi()
            self.status_var.set("LASSO \uc120\ud0dd\uc744 \ucde8\uc18c\ud588\uc2b5\ub2c8\ub2e4.")
            self._queue_render(high_quality=True, delay=0)
            return
        if key in {"Return", "KP_Enter"} and self.lasso_mode:
            if len(self.roi_polygon_points) >= 3:
                self.roi_cell_mask = self._roi_cell_mask_from_polygon()
                self.lasso_mode = False
                self._lasso_preview_point = None
                self.status_var.set("LASSO \uad6c\uc5ed\uc744 \ud655\uc815\ud588\uc2b5\ub2c8\ub2e4.")
                self._queue_render(high_quality=True, delay=0)
            return
        if key in {"Return", "KP_Enter", "space", "Tab"}:
            self._move_selection(0, 1)
            return

        key_to_value = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "KP_0": 0,
            "KP_1": 1,
            "KP_2": 2,
            "KP_3": 3,
            "KP_4": 4,
        }
        if key in key_to_value:
            self._set_selected_value(key_to_value[key])

    def _visible_cell_range(self) -> Tuple[range, range]:
        canvas_w = max(100, self.canvas.winfo_width())
        canvas_h = max(100, self.canvas.winfo_height())
        disp_w, disp_h = self._display_size
        origin_x, origin_y = self._display_origin
        cell_h = max(disp_h / float(self.A), 1e-6)
        cell_w = max(disp_w / float(self.B), 1e-6)
        row_start = int(np.clip(np.floor((0.0 - origin_y) / cell_h) - 1, 0, self.A))
        row_end = int(np.clip(np.ceil((canvas_h - origin_y) / cell_h) + 1, 0, self.A))
        col_start = int(np.clip(np.floor((0.0 - origin_x) / cell_w) - 1, 0, self.B))
        col_end = int(np.clip(np.ceil((canvas_w - origin_x) / cell_w) + 1, 0, self.B))
        return range(row_start, row_end), range(col_start, col_end)

    def _draw_lasso_polygon_overlay(self, origin_x: float, origin_y: float) -> None:
        polygon_points = self.roi_polygon_points
        if not polygon_points:
            return
        coords: List[float] = []
        for x, y in polygon_points:
            coords.extend([origin_x + x * self._scale, origin_y + y * self._scale])
        if self.lasso_mode and self._lasso_preview_point is not None:
            coords.extend([
                origin_x + self._lasso_preview_point[0] * self._scale,
                origin_y + self._lasso_preview_point[1] * self._scale,
            ])
            self.canvas.create_line(*coords, fill="#0f7a0f", width=5, joinstyle=tk.ROUND)
            self.canvas.create_line(*coords, fill="#35c759", width=3, joinstyle=tk.ROUND)
        elif len(coords) >= 6:
            self.canvas.create_polygon(*coords, outline="#0f7a0f", fill="", width=5, joinstyle=tk.ROUND)
            self.canvas.create_polygon(*coords, outline="#35c759", fill="", width=3, joinstyle=tk.ROUND)

    def _gis_boundary_overlay_paths(self, layer_id: str, feature_index: int) -> List[List[float]]:
        if self._gis_map_bbox is None or self.img_pil is None:
            return []
        zoom_bucket = int(round(math.log2(max(self._zoom, 0.25)) * 2.0))
        key = (
            layer_id,
            int(feature_index),
            tuple(round(float(value), 8) for value in self._gis_map_bbox),
            tuple(self.img_pil.size),
            zoom_bucket,
        )
        cached = self._gis_boundary_overlay_cache.get(key)
        if cached is not None:
            return cached
        layer = self._gis_layers.get(layer_id)
        if layer is None:
            return []
        boundaries = list(layer["geometry"])
        if not (0 <= feature_index < len(boundaries)):
            return []
        xmin, ymin, xmax, ymax = self._gis_map_bbox
        width = max(float(self.img_pil.size[0] - 1), 1.0)
        height = max(float(self.img_pil.size[1] - 1), 1.0)
        x_span = max(float(xmax) - float(xmin), 1e-12)
        y_span = max(float(ymax) - float(ymin), 1e-12)
        epsilon = max(0.3, 1.4 / max(self._zoom, 0.25))
        paths: List[List[float]] = []
        for ring in boundaries[feature_index].rings:
            if len(ring) < 3:
                continue
            points = np.asarray(ring, dtype=np.float64)
            pixels = np.column_stack(
                (
                    (points[:, 0] - xmin) / x_span * width,
                    (ymax - points[:, 1]) / y_span * height,
                )
            ).astype(np.float32)
            closed = pixels if np.allclose(pixels[0], pixels[-1]) else np.vstack((pixels, pixels[0]))
            simplified = cv2.approxPolyDP(closed.reshape((-1, 1, 2)), epsilon=epsilon, closed=True).reshape((-1, 2))
            paths.append(simplified.reshape(-1).astype(float).tolist())
        self._gis_boundary_overlay_cache[key] = paths
        while len(self._gis_boundary_overlay_cache) > 256:
            self._gis_boundary_overlay_cache.pop(next(iter(self._gis_boundary_overlay_cache)))
        return paths

    @staticmethod
    def _gis_boundary_overlay_widths(style_width: float, zoom: float, *, hover: bool) -> Tuple[float, float]:
        if hover:
            return 4.0, 7.0
        inner = float(np.clip(float(style_width) / math.sqrt(max(float(zoom), 1.0)), 0.75, 1.75))
        return inner, inner + 1.5

    def _draw_gis_boundary_overlay(self, origin_x: float, origin_y: float) -> None:
        if self._gis_map_bbox is None or self.img_pil is None:
            return
        selected = self._gis_hover_boundary or self._gis_last_clicked_boundary
        if selected is None:
            return
        layer = self._gis_layers.get(selected[0])
        if layer is None or not bool(layer.get("visible", True)):
            return
        boundaries = list(layer["geometry"])
        if not (0 <= selected[1] < len(boundaries)):
            return
        _field, label = self._boundary_feature_label(
            boundaries[selected[1]],
            selected[1],
            str(layer.get("label_field", "") or "") or None,
        )
        label_id = self.canvas.create_text(
            14,
            14,
            text=label,
            anchor=tk.NW,
            fill="#101820",
            font=("Malgun Gothic", 11, "bold"),
            tags=("gis_boundary_overlay",),
        )
        label_bbox = self.canvas.bbox(label_id)
        if label_bbox is not None:
            background_id = self.canvas.create_rectangle(
                label_bbox[0] - 7,
                label_bbox[1] - 4,
                label_bbox[2] + 7,
                label_bbox[3] + 4,
                fill="#ffffff",
                outline="#d6dde5",
                tags=("gis_boundary_overlay",),
            )
            self.canvas.tag_lower(background_id, label_id)
        style = self._normalized_gis_layer_style(layer)
        layer_color = self._bgr_to_hex(tuple(style["outline_color"]))
        is_hover = self._gis_hover_boundary is not None
        color = "#ff9500" if is_hover else layer_color
        inner_width, halo_width = self._gis_boundary_overlay_widths(
            float(style["line_width"]),
            float(self._zoom),
            hover=is_hover,
        )
        for image_coords in self._gis_boundary_overlay_paths(selected[0], selected[1]):
            coords_array = np.asarray(image_coords, dtype=np.float64).reshape((-1, 2))
            coords_array[:, 0] = origin_x + coords_array[:, 0] * self._scale
            coords_array[:, 1] = origin_y + coords_array[:, 1] * self._scale
            coords = coords_array.reshape(-1).tolist()
            self.canvas.create_line(
                *coords,
                fill="#ffffff",
                width=halo_width,
                joinstyle=tk.ROUND,
                tags=("gis_boundary_overlay",),
            )
            self.canvas.create_line(
                *coords,
                fill=color,
                width=inner_width,
                joinstyle=tk.ROUND,
                tags=("gis_boundary_overlay",),
            )

    def _refresh_gis_boundary_overlay(self) -> None:
        if not hasattr(self, "canvas"):
            return
        self.canvas.delete("gis_boundary_overlay")
        if vars(self).get("source_kind") == "gis_project":
            self._draw_gis_boundary_overlay(float(self._display_origin[0]), float(self._display_origin[1]))

    def _draw_overlay(self) -> None:
        if self.current_matrix is None or self.occ_matrix is None:
            return

        disp_w, disp_h = self._display_size
        origin_x, origin_y = self._display_origin
        if vars(self).get("source_kind") == "gis_project":
            self._draw_gis_boundary_overlay(origin_x, origin_y)
            self._draw_lasso_polygon_overlay(origin_x, origin_y)
            return
        cell_h = disp_h / float(self.A)
        cell_w = disp_w / float(self.B)
        changed_mask = self.current_matrix != self.base_matrix if self.base_matrix is not None else np.zeros_like(self.current_matrix, dtype=bool)
        unresolved_mask = self.unresolved_mask if self.unresolved_mask is not None and self.unresolved_mask.shape == self.current_matrix.shape else np.zeros_like(self.current_matrix, dtype=np.uint8)
        model_applied_mask = self.model_applied_mask if self.model_applied_mask is not None and self.model_applied_mask.shape == self.current_matrix.shape else np.zeros_like(self.current_matrix, dtype=np.uint8)
        flood_mask_value = vars(self).get("_flood_cell_mask")
        flood_cell_mask = flood_mask_value if isinstance(flood_mask_value, np.ndarray) and flood_mask_value.shape == self.current_matrix.shape else np.zeros_like(self.current_matrix, dtype=np.uint8)
        roi_cell_mask = self._active_edit_mask()
        area_fill = vars(self).get("_area_fill_mask")
        if isinstance(area_fill, np.ndarray) and area_fill.shape == self.current_matrix.shape:
            changed_mask &= area_fill == 0
        show_area = vars(self).get("show_fill_var") is None or self.show_fill_var.get()
        outlet_cells = self._active_outlet_cells()
        visible_rows, visible_cols = self._visible_cell_range()
        show_text = min(cell_h, cell_w) >= 18

        for i in visible_rows:
            for j in visible_cols:
                if roi_cell_mask is not None and int(roi_cell_mask[i, j]) != 1:
                    continue
                x1 = origin_x + j * cell_w
                y1 = origin_y + i * cell_h
                x2 = origin_x + (j + 1) * cell_w
                y2 = origin_y + (i + 1) * cell_h
                value = int(self.current_matrix[i, j])
                is_area_fill = isinstance(area_fill, np.ndarray) and area_fill.shape == self.current_matrix.shape and bool(area_fill[i, j])
                if is_area_fill and not show_area:
                    continue
                if is_area_fill:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#d9eef8", outline="", tags="overlay")
                occupied = int(self.occ_matrix[i, j]) == 1 or value != 0

                outline = "#168aad" if is_area_fill else "#506070"
                if bool(changed_mask[i, j]):
                    outline = "#ffb347"
                if self.selected_cell == (i, j):
                    outline = "#fff07a"
                if (i, j) in outlet_cells:
                    outline = "#56d8ff"

                width = 3 if (self.selected_cell == (i, j) or (i, j) in outlet_cells) else 2 if bool(changed_mask[i, j]) else 1

                if occupied and value != 0:
                    self.canvas.create_rectangle(
                        x1,
                        y1,
                        x2,
                        y2,
                        outline=outline,
                        width=width,
                        fill="#d9eef8" if is_area_fill else DIR_COLORS.get(value, "#495466"),
                        stipple="gray25",
                    )
                elif occupied:
                    self.canvas.create_rectangle(
                        x1,
                        y1,
                        x2,
                        y2,
                        outline="#788695",
                        width=width,
                        fill="#b9c1c9",
                        stipple="gray50",
                    )
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline=outline, width=width)

                if value != 0 and show_text:
                    text_x = (x1 + x2) / 2.0
                    text_y = (y1 + y2) / 2.0
                    text_font = ("Segoe UI", max(8, int(cell_h * 0.38)), "bold")
                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        self.canvas.create_text(
                            text_x + dx,
                            text_y + dy,
                            text=str(value),
                            fill=MATRIX_NUMBER_HALO,
                            font=text_font,
                        )
                    self.canvas.create_text(
                        text_x,
                        text_y,
                        text=str(value),
                        fill=MATRIX_NUMBER_COLOR,
                        font=text_font,
                    )
                if (i, j) in outlet_cells:
                    self.canvas.create_text(
                        x1 + 6,
                        y1 + 6,
                        text="O",
                        fill="#56d8ff",
                        anchor=tk.NW,
                        font=("Segoe UI", max(8, int(cell_h * 0.24)), "bold"),
                    )
                if int(model_applied_mask[i, j]) == 1:
                    inset = max(2, int(min(cell_h, cell_w) * 0.10))
                    self.canvas.create_rectangle(
                        x1 + inset,
                        y1 + inset,
                        x2 - inset,
                        y2 - inset,
                        outline="#00bfa5",
                        width=2,
                    )
                    if show_text and min(cell_h, cell_w) >= 24:
                        self.canvas.create_text(
                            x2 - 4,
                            y1 + 4,
                            text="ML",
                            fill="#00bfa5",
                            anchor=tk.NE,
                            font=("Segoe UI", max(7, int(cell_h * 0.20)), "bold"),
                        )
                if int(unresolved_mask[i, j]) == 1:
                    self.canvas.create_rectangle(
                        x1 + 1,
                        y1 + 1,
                        x2 - 1,
                        y2 - 1,
                        outline="#ff4040",
                        width=3,
                    )
                if int(flood_cell_mask[i, j]) == 1 and value != 0:
                    inset = max(2, int(min(cell_h, cell_w) * 0.08))
                    self.canvas.create_rectangle(
                        x1 + inset,
                        y1 + inset,
                        x2 - inset,
                        y2 - inset,
                        outline="#ffffff",
                        width=5,
                    )
                    self.canvas.create_rectangle(
                        x1 + inset,
                        y1 + inset,
                        x2 - inset,
                        y2 - inset,
                        outline="#ff2d55",
                        width=3,
                    )

        for x1, y1, x2, y2 in self.arrow_boxes:
            if self.roi_mask is not None:
                cx = int(np.clip(round((x1 + x2) / 2.0), 0, self.roi_mask.shape[1] - 1))
                cy = int(np.clip(round((y1 + y2) / 2.0), 0, self.roi_mask.shape[0] - 1))
                if int(self.roi_mask[cy, cx]) == 0:
                    continue
            sx1 = origin_x + x1 * self._scale
            sy1 = origin_y + y1 * self._scale
            sx2 = origin_x + x2 * self._scale
            sy2 = origin_y + y2 * self._scale
            if sx2 < 0 or sy2 < 0 or sx1 > self.canvas.winfo_width() or sy1 > self.canvas.winfo_height():
                continue
            self.canvas.create_rectangle(
                sx1,
                sy1,
                sx2,
                sy2,
                outline="#ffd166",
                width=2,
                dash=(6, 4),
            )

        self._draw_lasso_polygon_overlay(origin_x, origin_y)

    def _render_canvas(self, high_quality: bool = True) -> None:
        if high_quality:
            self._render_hq_after_id = None
        else:
            self._render_after_id = None
        self.canvas.delete("all")
        display_img = self._active_display_image()
        if display_img is None:
            return

        canvas_w = max(100, self.canvas.winfo_width())
        canvas_h = max(100, self.canvas.winfo_height())
        img_w, img_h = display_img.size
        self._fit_scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        self._scale = max(self._fit_scale * self._zoom, 1e-6)
        disp_w = max(1, int(round(img_w * self._scale)))
        disp_h = max(1, int(round(img_h * self._scale)))
        self._display_size = (disp_w, disp_h)
        if not bool(vars(self).get("_display_origin_initialized", False)):
            self._display_origin = ((canvas_w - disp_w) / 2.0, (canvas_h - disp_h) / 2.0)
            self._display_origin_initialized = True

        view_x0 = max(0.0, -self._display_origin[0] / self._scale)
        view_y0 = max(0.0, -self._display_origin[1] / self._scale)
        view_x1 = min(float(img_w), (canvas_w - self._display_origin[0]) / self._scale)
        view_y1 = min(float(img_h), (canvas_h - self._display_origin[1]) / self._scale)

        if view_x1 > view_x0 and view_y1 > view_y0:
            image_dest_w = max(1, int(round((view_x1 - view_x0) * self._scale)))
            image_dest_h = max(1, int(round((view_y1 - view_y0) * self._scale)))
            draw_x = self._display_origin[0] + view_x0 * self._scale
            draw_y = self._display_origin[1] + view_y0 * self._scale
            view_img = (
                self._render_gis_viewport_image((view_x0, view_y0, view_x1, view_y1), (image_dest_w, image_dest_h))
                if high_quality
                else None
            )
            if view_img is None:
                factor, source = self._select_active_pyramid_level(self._scale)
                src_x0 = int(np.clip(np.floor(view_x0 * factor), 0, source.size[0] - 1))
                src_y0 = int(np.clip(np.floor(view_y0 * factor), 0, source.size[1] - 1))
                src_x1 = int(np.clip(np.ceil(view_x1 * factor), src_x0 + 1, source.size[0]))
                src_y1 = int(np.clip(np.ceil(view_y1 * factor), src_y0 + 1, source.size[1]))
                crop = source.crop((src_x0, src_y0, src_x1, src_y1))
                dest_w = max(1, int(round((src_x1 - src_x0) * self._scale / max(factor, 1e-6))))
                dest_h = max(1, int(round((src_y1 - src_y0) * self._scale / max(factor, 1e-6))))
                resample = Image.LANCZOS if high_quality else Image.BILINEAR
                view_img = crop.resize((dest_w, dest_h), resample)
                roi_mask_source = None if self.matrix_arrow_view else self._select_roi_mask_level(self._scale)
                if roi_mask_source is not None and self.roi_mask is not None:
                    mask_crop = roi_mask_source.crop((src_x0, src_y0, src_x1, src_y1))
                    mask_view = mask_crop.resize((dest_w, dest_h), Image.NEAREST)
                    white_bg = Image.new("RGB", (dest_w, dest_h), "white")
                    view_img = Image.composite(view_img.convert("RGB"), white_bg, mask_view)
                draw_x = self._display_origin[0] + (src_x0 / max(factor, 1e-6)) * self._scale
                draw_y = self._display_origin[1] + (src_y0 / max(factor, 1e-6)) * self._scale
            self.tk_img = ImageTk.PhotoImage(view_img)
            self.canvas.create_image(draw_x, draw_y, image=self.tk_img, anchor=tk.NW)
        if not self.matrix_arrow_view:
            self._draw_overlay()

    def display_image(self) -> None:
        self._queue_render(high_quality=True, delay=0)

    def _update_matrix_preview(self) -> None:
        if vars(self).get("source_kind") == "gis_project":
            visible_count = sum(
                1
                for layer in self._gis_layers.values()
                if bool(layer.get("visible", True)) and self._gis_layer_crs_ready(layer)
            )
            flood_layers = [
                self._gis_layers[layer_id]
                for layer_id in sorted(self._active_flood_layer_ids())
                if layer_id in self._gis_layers
            ]
            flood_text = f"{len(flood_layers)}개" if flood_layers else "미지정"
            self.summary_var.set(
                f"GIS 레이어: {len(self._gis_layers)}개\n표시 레이어: {visible_count}개\n"
                f"침수지역 레이어: {flood_text}\n행정경계를 클릭하거나 LASSO로 행렬을 생성하세요."
            )
            matrix_text = ""
        elif self.current_matrix is None:
            self.summary_var.set(TXT_NO_MATRIX)
            matrix_text = ""
        else:
            area = vars(self).get("_area_fill_mask")
            pipe_cells = area == 0 if isinstance(area, np.ndarray) and area.shape == self.current_matrix.shape else np.ones_like(self.current_matrix, dtype=bool)
            changed = 0
            if self.base_matrix is not None and self.base_matrix.shape == self.current_matrix.shape:
                changed = int(np.count_nonzero((self.current_matrix != self.base_matrix) & pipe_cells))
            nonzero = int(np.count_nonzero(self.current_matrix))
            low_conf = 0
            if self.confidence_matrix is not None and self.confidence_matrix.shape == self.current_matrix.shape:
                low_conf = int(np.count_nonzero((self.current_matrix > 0) & (self.confidence_matrix < 0.55) & pipe_cells))
            model_applied = 0
            if self.model_applied_mask is not None and self.model_applied_mask.shape == self.current_matrix.shape:
                model_applied = int(np.count_nonzero(self.model_applied_mask))
            active_outlets = self._active_outlet_cells()
            outlet_text = (
                f"{len(active_outlets)}개 / 대표 {self.outlet_cell}"
                if len(active_outlets) > 1
                else str(self.outlet_cell) if self.outlet_cell is not None else "\ubbf8\uc120\ud0dd"
            )
            roi_text = str(int(np.count_nonzero(self.roi_cell_mask))) if self.roi_cell_mask is not None else "\uc5c6\uc74c"
            compress_text = (
                f"\n\uc555\ucd95 \uc6d0\ubcf8: {self.compression_source_shape[0]}x{self.compression_source_shape[1]}"
                if self.compression_source_shape is not None
                else ""
            )
            trim_text = (
                f"\n외곽 제거 원본: {self.trim_source_shape[0]}x{self.trim_source_shape[1]} "
                f"| 유지 범위: {self.trim_bounds}"
                if self.trim_source_shape is not None and self.trim_bounds is not None
                else ""
            )
            runtime_learning_text = "\ube44\ud65c\uc131\ud654" if not RUNTIME_TRAINING_ENABLED else "\ud65c\uc131"
            flood_cells = int(np.count_nonzero(self._flood_cell_mask)) if isinstance(vars(self).get("_flood_cell_mask"), np.ndarray) else 0
            preview = self.current_matrix[:120, :80]
            matrix_text = "\n".join(" ".join(str(int(v)) for v in row) for row in preview)
            if preview.shape != self.current_matrix.shape:
                matrix_text = f"미리보기 {preview.shape[0]}×{preview.shape[1]} / 전체 {self.A}×{self.B}\n전체 행렬은 TXT 내보내기로 저장됩니다.\n\n" + matrix_text
            self.summary_var.set(
                f"A={self.A}, B={self.B}{compress_text}{trim_text}\n\uac12\uc774 \uc788\ub294 \uc140: {nonzero}\n침수지역 중첩 관로 셀: {flood_cells}\n\uc800\uc2e0\ub8b0 \uc140: {low_conf}\n\uc218\uc815\ub41c \uc140: {changed}\n\ubaa8\ub378 \uac1c\uc785 \uc140: {model_applied}\nOutlet: {outlet_text}\nROI \uc140: {roi_text}\n\ub7f0\ud0c0\uc784 \ub204\uc801 \ud559\uc2b5: {runtime_learning_text}\n\ubaa8\ub378: {self.model_path}\n\uc7a5\uce58 \uc124\uc815: {self.device_choice_var.get()}"
            )

        if self._has_area_fill():
            count = int(np.count_nonzero(self._area_fill_mask))
            self.summary_var.set(f"면적 배수: {count:,}셀 · 기존 관망과 구분\n" + self.summary_var.get())
        digest = hashlib.blake2b(matrix_text.encode("utf-8"), digest_size=8).digest()
        if digest != vars(self).get("_matrix_preview_digest"):
            self.matrix_text.configure(state=tk.NORMAL)
            self.matrix_text.delete("1.0", tk.END)
            self.matrix_text.insert("1.0", matrix_text)
            self.matrix_text.configure(state=tk.DISABLED)
            self._matrix_preview_digest = digest
        self._update_action_states()


if __name__ == "__main__":
    app = NFMATApp()
    app.mainloop()
