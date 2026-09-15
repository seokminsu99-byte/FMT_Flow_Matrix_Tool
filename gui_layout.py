# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Uses Tkinter / Tcl-Tk: https://docs.python.org/3/library/tkinter.html (runtime licenses).
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
"""Compact, resizable NFMAT6 workspace built with the standard Tk toolkit.

This module owns presentation only. Application callbacks, variables and matrix
state remain on the supplied app, so changing the layout does not change GIS or
network-analysis behavior.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext, ttk


class _ScrollPanel(ttk.Frame):
    """A vertically scrollable controls panel with a width-tracking interior."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, style="Workspace.TFrame")
        self.viewport = tk.Canvas(self, bd=0, highlightthickness=0, background="#ffffff")
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.viewport.yview)
        self.viewport.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.viewport.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.content = ttk.Frame(self.viewport, padding=12, style="Workspace.TFrame")
        self.window_id = self.viewport.create_window((0, 0), window=self.content, anchor=tk.NW)
        self.viewport.bind("<Configure>", self._resize)
        self.content.bind("<Configure>", self._content_resize)
        self.viewport.bind("<MouseWheel>", self._wheel)

    def _resize(self, event: tk.Event) -> None:
        self.viewport.itemconfigure(self.window_id, width=max(1, int(event.width)))

    def _content_resize(self, _event: tk.Event) -> None:
        self.viewport.configure(scrollregion=self.viewport.bbox("all"))

    def _wheel(self, event: tk.Event) -> str:
        self.viewport.yview_scroll(-int(event.delta / 120), "units")
        return "break"


def _button(app, parent, text, callback, *, primary=False, attr=None, state=None):
    button = ttk.Button(
        parent, text=text, command=getattr(app, callback),
        style="Primary.TButton" if primary else "Workspace.TButton",
    )
    if state is not None:
        button.configure(state=state)
    if attr:
        setattr(app, attr, button)
    app._workspace_action_buttons.append(button)
    return button


def _section(parent, title):
    group = ttk.LabelFrame(parent, text=title, padding=10, style="Workspace.TLabelframe")
    group.pack(fill=tk.X, pady=(0, 12))
    return group


def _action_rows(app, parent, actions, *, columns=2):
    grid = ttk.Frame(parent, style="Workspace.TFrame")
    grid.pack(fill=tk.X)
    for index, action in enumerate(actions):
        text, callback, *attr = action
        button = _button(app, grid, text, callback, attr=attr[0] if attr else None)
        button.grid(row=index // columns, column=index % columns, sticky="ew", padx=2, pady=3)
    for column in range(columns):
        grid.columnconfigure(column, weight=1)
    return grid


def _wrapped(parent, text=None, variable=None, *, muted=False):
    label = ttk.Label(
        parent, text=text, textvariable=variable, justify=tk.LEFT,
        style="Muted.TLabel" if muted else "Workspace.TLabel", wraplength=260,
    )
    label.pack(fill=tk.X, pady=(0, 8))
    label.bind("<Configure>", lambda event: label.configure(wraplength=max(80, event.width - 4)))
    return label


def _cancel(app):
    callback = getattr(app, "cancel_active_task", None)
    if callback is None:
        callback = getattr(app, "_terminate_plena_process", None)
    if callback is not None:
        callback()


def build_workspace_ui(app, *, device_choices=None, theme_names=None, training_enabled=True):
    """Build the three-pane workspace while preserving the legacy widget API."""
    app._workspace_action_buttons = []
    app._workspace_scroll_panels = []
    app._workspace_sashes_initialized = False
    app._workspace_layout = True
    app._toolbar_groups = []
    app._toolbar_rows = []

    outer = ttk.Frame(app, padding=(14, 10, 14, 10), style="Root.TFrame")
    app.outer_frame = outer
    outer.pack(fill=tk.BOTH, expand=True)

    header = ttk.Frame(outer, style="Root.TFrame")
    header.pack(fill=tk.X, pady=(0, 10))
    ttk.Label(header, text="NFMAT 6", style="Brand.TLabel").pack(side=tk.LEFT)
    ttk.Label(header, text="관로 네트워크 분석", style="Subtitle.TLabel").pack(side=tk.LEFT, padx=(14, 0))
    _button(app, header, "도움말", "show_help").pack(side=tk.RIGHT)
    ttk.Label(header, text="데이터  /  행렬  /  연결  /  분석", style="MutedHeader.TLabel").pack(side=tk.RIGHT, padx=16)

    toolbar = ttk.Frame(outer, padding=(10, 8), style="Workspace.TFrame")
    app.toolbar_frame = toolbar
    toolbar.pack(fill=tk.X, pady=(0, 8))
    app._workspace_toolbar_groups = []
    for index, title in enumerate(("01   데이터", "02   격자 · 인식", "03   영역 · 연결", "04   PLENA 분석")):
        group = ttk.Frame(toolbar, padding=(8, 0), style="Workspace.TFrame")
        group.grid(row=0, column=index, sticky="nsew")
        toolbar.columnconfigure(index, weight=(1, 2, 2, 2)[index])
        ttk.Label(group, text=title, style="Step.TLabel").pack(anchor=tk.W, pady=(0, 5))
        content = ttk.Frame(group, style="Workspace.TFrame")
        content.pack(fill=tk.X)
        app._workspace_toolbar_groups.append((group, content))
        if index:
            ttk.Separator(toolbar, orient=tk.VERTICAL).grid(row=0, column=index - 1, sticky="nse", padx=(0, 0))

    data = app._workspace_toolbar_groups[0][1]
    _action_rows(app, data, [("도면 열기", "load_image"), ("GIS 추가", "add_gis_layers")], columns=1)
    grid = app._workspace_toolbar_groups[1][1]
    ttk.Label(grid, text="행 수", style="Workspace.TLabel").grid(row=0, column=0, sticky=tk.W, padx=2)
    app.rows_spinbox = ttk.Spinbox(grid, from_=2, to=300, textvariable=app.rows_var, width=6)
    app._workspace_action_buttons.append(app.rows_spinbox)
    app.rows_spinbox.grid(row=0, column=1, sticky="ew", padx=2, pady=3)
    _button(app, grid, "자동 인식", "run_detection", attr="detect_button").grid(row=1, column=0, sticky="ew", padx=2, pady=3)
    _button(app, grid, "모델 적용", "apply_model", attr="model_button").grid(row=1, column=1, sticky="ew", padx=2, pady=3)
    _button(app, grid, "빈 행렬", "start_empty_grid", attr="empty_grid_button").grid(row=2, column=0, sticky="ew", padx=2, pady=3)
    _button(app, grid, "수정 초기화", "clear_edits").grid(row=2, column=1, sticky="ew", padx=2, pady=3)
    grid.columnconfigure(0, weight=1)
    grid.columnconfigure(1, weight=1)
    connect = app._workspace_toolbar_groups[2][1]
    connection_grid = _action_rows(app, connect, [
        ("LASSO 선택", "toggle_lasso_mode"), ("영역 적용", "apply_lasso_to_matrix"),
        ("Outlet 선택", "enable_outlet_pick_mode"), ("BFS 보정", "apply_bfs_assist", "bfs_button"),
        ("빈 칸 채우기", "fill_empty_cells", "fill_button"), ("채우기 되돌리기", "undo_fill_empty_cells", "undo_fill_button"),
    ])
    app.fill_button.configure(style="Primary.TButton")
    export = app._workspace_toolbar_groups[3][1]
    _button(app, export, "PLENA 실행", "run_plena", primary=True, attr="plena_button").grid(row=0, column=0, columnspan=2, sticky="ew", padx=2, pady=3)
    _button(app, export, "입력 내보내기", "export_matrix", attr="export_button").grid(row=1, column=0, sticky="ew", padx=2, pady=3)
    _button(app, export, "결과 그래프", "open_plena_plot_window").grid(row=1, column=1, sticky="ew", padx=2, pady=3)
    export.columnconfigure(0, weight=1)
    export.columnconfigure(1, weight=1)

    # Keep pipenet5's everyday editing controls visible without tab hunting.
    quickbar = ttk.Frame(outer, padding=(10, 5), style="Workspace.TFrame")
    quickbar.pack(fill=tk.X, pady=(0, 6))
    ttk.Label(quickbar, text="회전 °", style="Workspace.TLabel").pack(side=tk.LEFT, padx=(0, 5))
    app.rotation_spinbox = ttk.Spinbox(quickbar, from_=-360.0, to=360.0, increment=0.1,
                                      textvariable=app.rotation_var, width=7, format="%.1f")
    app.rotation_spinbox.pack(side=tk.LEFT)
    app._workspace_action_buttons.append(app.rotation_spinbox)
    _button(app, quickbar, "회전 적용", "apply_rotation", attr="rotation_apply_button").pack(side=tk.LEFT, padx=4)
    _button(app, quickbar, "0° 초기화", "reset_rotation").pack(side=tk.LEFT)
    ttk.Separator(quickbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
    _button(app, quickbar, "관로 → 행렬", "convert_selected_pipe_layer_to_matrix").pack(side=tk.LEFT, padx=3)
    _button(app, quickbar, "GIS 방향 보정", "recalculate_gis_directions", attr="gis_direction_button").pack(side=tk.LEFT, padx=3)
    _button(app, quickbar, "권장 토출부", "recommend_outlets").pack(side=tk.LEFT, padx=3)
    _button(app, quickbar, "빈 외곽 제거", "trim_matrix_margins", attr="trim_button", state=tk.DISABLED).pack(side=tk.LEFT, padx=3)
    _button(app, quickbar, "행렬 압축", "compress_matrix", attr="compress_button", state=tk.DISABLED).pack(side=tk.LEFT, padx=3)

    taskbar = ttk.Frame(outer, style="Root.TFrame")
    taskbar.pack(fill=tk.X, pady=(0, 8))
    if not hasattr(app, "task_label_var"):
        app.task_label_var = tk.StringVar(master=app, value="작업 대기")
    app.task_label = ttk.Label(taskbar, textvariable=app.task_label_var, style="MutedHeader.TLabel", width=34)
    app.task_label.pack(side=tk.LEFT)
    app.task_progress = ttk.Progressbar(taskbar, mode="indeterminate", length=120, style="Workspace.Horizontal.TProgressbar")
    app.task_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 12))
    app.task_cancel_button = ttk.Button(taskbar, text="작업 중단", command=lambda: _cancel(app), state=tk.DISABLED, style="Workspace.TButton")
    app.task_cancel_button.pack(side=tk.RIGHT)

    body = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
    app.workspace_panes = body
    body.pack(fill=tk.BOTH, expand=True)
    left = ttk.Frame(body, width=256, style="Workspace.TFrame")
    center = ttk.Frame(body, style="Workspace.TFrame")
    right = ttk.Frame(body, width=312, style="Workspace.TFrame")
    left.pack_propagate(False)
    right.pack_propagate(False)
    app.layer_panel_frame = left
    app.left_frame = center
    app.right_frame = right
    body.add(left, weight=0)
    body.add(center, weight=1)
    body.add(right, weight=0)

    ttk.Label(left, text="데이터 · GIS 레이어", style="PanelHeading.TLabel", padding=(12, 10)).pack(fill=tk.X)
    app.layer_notebook = ttk.Notebook(left, style="Workspace.TNotebook")
    app.layer_notebook.pack(fill=tk.BOTH, expand=True)
    layers = ttk.Frame(app.layer_notebook, padding=8, style="Workspace.TFrame")
    area_scroll = _ScrollPanel(app.layer_notebook)
    app._workspace_scroll_panels.append(area_scroll)
    app.layer_notebook.add(layers, text="레이어")
    app.layer_notebook.add(area_scroll, text="영역 분석")

    tree_frame = ttk.Frame(layers, style="Workspace.TFrame")
    tree_frame.pack(fill=tk.BOTH, expand=True)
    tree_frame.columnconfigure(0, weight=1)
    tree_frame.rowconfigure(0, weight=1)
    app.layer_tree = ttk.Treeview(tree_frame, columns=("visible", "type", "role", "crs"), show="tree headings", selectmode="browse", height=5, style="Workspace.Treeview")
    for name, title, width in (("#0", "레이어", 132), ("visible", "표시", 42), ("type", "종류", 56), ("role", "역할", 52), ("crs", "지도 좌표", 122)):
        app.layer_tree.heading(name, text=title)
        app.layer_tree.column(name, width=width, minwidth=width, stretch=name == "#0", anchor=tk.W if name in {"#0", "crs"} else tk.CENTER)
    app.layer_tree.grid(row=0, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=app.layer_tree.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    xscroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=app.layer_tree.xview)
    xscroll.grid(row=1, column=0, sticky="ew")
    app.layer_tree.configure(xscrollcommand=xscroll.set, yscrollcommand=scrollbar.set)
    app.layer_tree.bind("<Double-Button-1>", app._on_gis_layer_tree_double_click)
    files = ttk.Frame(layers, style="Workspace.TFrame")
    files.pack(fill=tk.X, pady=(6, 0))
    _action_rows(app, files, [("추가", "add_gis_layers"), ("표시 전환", "_toggle_selected_layer_visibility"), ("제거", "remove_selected_gis_layer")], columns=3)
    roles = ttk.Frame(layers, style="Workspace.TFrame")
    roles.pack(fill=tk.X)
    _action_rows(app, roles, [("관로 지정", "set_selected_pipe_layer"), ("경계 지정", "set_selected_boundary_layer"), ("침수 지정", "set_selected_flood_layer"), ("색상 · 투명도", "configure_selected_layer_style")])
    orders = ttk.Frame(layers, style="Workspace.TFrame")
    orders.pack(fill=tk.X)
    ttk.Button(orders, text="위로", command=lambda: app._move_selected_gis_layer(-1), style="Workspace.TButton").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=3)
    ttk.Button(orders, text="아래로", command=lambda: app._move_selected_gis_layer(1), style="Workspace.TButton").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=3)
    coords = ttk.Frame(layers, style="Workspace.TFrame")
    coords.pack(fill=tk.X, pady=(8, 0))
    _wrapped(coords, variable=app.gis_project_crs_var, muted=True)
    _action_rows(app, coords, [("레이어 좌표", "configure_selected_layer_crs"), ("지도 좌표", "configure_project_crs")])

    area = area_scroll.content
    clip = _section(area, "행렬 생성")
    _action_rows(app, clip, [
        ("전체 GIS 지도", "return_to_full_gis_map"), ("관로 표고 필드", "configure_pipe_height_fields"),
        ("선택 관로 → 행렬", "convert_selected_pipe_layer_to_matrix"),
        ("경계 클릭 → 행렬", "show_boundary_click_map"), ("LASSO → 행렬", "clip_lasso_to_matrix"),
        ("경계 이름 설정", "configure_boundary_label_field"),
    ], columns=1)
    flood = _section(area, "침수지역")
    _wrapped(flood, variable=app.flood_region_var)
    _action_rows(app, flood, [("침수 중첩 갱신", "refresh_flood_overlap"), ("침수 전체 해제", "clear_flood_layer")], columns=1)

    canvasbar = ttk.Frame(center, padding=(10, 6), style="Workspace.TFrame")
    canvasbar.pack(fill=tk.X)
    ttk.Label(canvasbar, text="작업 화면", style="PanelHeading.TLabel").pack(side=tk.LEFT)
    _button(app, canvasbar, "화살표 보기", "toggle_matrix_arrow_view", attr="matrix_arrow_button").pack(side=tk.RIGHT)
    _button(app, canvasbar, "화면 맞춤", "reset_view").pack(side=tk.RIGHT, padx=(0, 5))
    app.canvas = tk.Canvas(center, background="#e8eef3", highlightthickness=0, takefocus=1, bd=0, relief=tk.FLAT)
    app.canvas.pack(fill=tk.BOTH, expand=True, padx=1)
    for event, callback in (
        ("<Button-1>", "on_canvas_click"), ("<Double-Button-1>", "on_canvas_double_click"),
        ("<ButtonPress-3>", "on_pan_start"), ("<B3-Motion>", "on_pan_drag"),
        ("<ButtonRelease-3>", "on_pan_end"), ("<Motion>", "on_canvas_motion"),
        ("<MouseWheel>", "on_mouse_wheel"), ("<Key>", "on_canvas_key"),
    ):
        app.canvas.bind(event, getattr(app, callback))
    ttk.Label(center, text="휠: 확대·축소   ·   오른쪽 드래그: 이동   ·   숫자 0–4: 셀 편집", style="CanvasHint.TLabel", padding=(10, 7)).pack(fill=tk.X)

    ttk.Label(right, text="검사 · 결과", style="PanelHeading.TLabel", padding=(12, 10)).pack(fill=tk.X)
    app.sidebar_notebook = ttk.Notebook(right, style="Workspace.TNotebook")
    app.sidebar_notebook.pack(fill=tk.BOTH, expand=True)
    inspection = _ScrollPanel(app.sidebar_notebook)
    matrix = ttk.Frame(app.sidebar_notebook, padding=8, style="Workspace.TFrame")
    tools = _ScrollPanel(app.sidebar_notebook)
    settings = _ScrollPanel(app.sidebar_notebook)
    app._workspace_scroll_panels.extend((inspection, tools, settings))
    for frame, title in ((inspection, "검사"), (matrix, "행렬"), (tools, "도구"), (settings, "설정")):
        app.sidebar_notebook.add(frame, text=title)
    app.inspector_tab = inspection
    app.matrix_tab = matrix
    app.tools_tab = tools
    app.settings_tab = settings
    app.info_top_frame = inspection.content
    selected = _section(inspection.content, "선택 셀")
    _wrapped(selected, variable=app.selected_var)
    buttons = ttk.Frame(selected, style="Workspace.TFrame")
    buttons.pack(fill=tk.X)
    for value, label in ((0, "0 · 삭제"), (1, "1 →"), (2, "2 ↓"), (3, "3 ←"), (4, "4 ↑")):
        button = ttk.Button(buttons, text=label, width=6, style="Workspace.TButton", command=lambda value=value: app._set_selected_value(value))
        button.grid(row=value // 3, column=value % 3, sticky="ew", padx=2, pady=2)
        app._workspace_action_buttons.append(button)
    for column in range(3):
        buttons.columnconfigure(column, weight=1)
    metric = _section(inspection.content, "공간 정보")
    _wrapped(metric, variable=app.gis_cell_size_var)
    _wrapped(metric, variable=app.gis_slope_var)
    summary = _section(inspection.content, "행렬 요약")
    _wrapped(summary, variable=app.summary_var)
    area_routing = _section(inspection.content, "유역 면적 배수")
    _wrapped(area_routing, "빈 셀을 연결된 관망으로 배수시킵니다. 실제 관망과 채운 면적 배수 경로를 구분해서 표시합니다.", muted=True)
    if not hasattr(app, "show_fill_var"):
        app.show_fill_var = tk.BooleanVar(master=app, value=True)
    app.show_fill_checkbutton = ttk.Checkbutton(area_routing, text="면적 배수 경로 표시", variable=app.show_fill_var,
        command=lambda: app._queue_render(high_quality=True, delay=0), style="Workspace.TCheckbutton")
    app.show_fill_checkbutton.pack(anchor=tk.W, pady=(0, 6))
    app._workspace_action_buttons.append(app.show_fill_checkbutton)
    _button(app, area_routing, "연결 검사", "inspect_network").pack(fill=tk.X)
    _wrapped(area_routing, "범례  ━ 관망   ··· 면적 배수", muted=True)
    app.rules_frame = _section(inspection.content, "방향값 규칙")
    _wrapped(app.rules_frame, "0: 관로 없음\n1: 동쪽  →    2: 남쪽  ↓\n3: 서쪽  ←    4: 북쪽  ↑\n\nLASSO 밖과 Outlet 미도달 셀은 내보낼 때 0으로 처리됩니다.", muted=True)
    ttk.Label(matrix, text="방향행렬 · 읽기 전용", style="Muted.TLabel", padding=(0, 0, 0, 8)).pack(fill=tk.X)
    app.matrix_text = scrolledtext.ScrolledText(matrix, width=1, height=1, font=("Consolas", 10), wrap=tk.NONE, relief=tk.FLAT, bd=0, padx=8, pady=8)
    app.matrix_text.pack(fill=tk.BOTH, expand=True)
    matrix_xscroll = ttk.Scrollbar(matrix, orient=tk.HORIZONTAL, command=app.matrix_text.xview)
    matrix_xscroll.pack(fill=tk.X)
    app.matrix_text.configure(state=tk.DISABLED, xscrollcommand=matrix_xscroll.set)

    edit = _section(tools.content, "행렬 편집")
    _action_rows(app, edit, [("빈 행렬", "start_empty_grid"), ("수정 초기화", "clear_edits"), ("권장 토출부", "recommend_outlets")], columns=1)
    _wrapped(edit, "회전·빈 외곽 제거·행렬 압축은 상단 도구줄에서 바로 사용할 수 있습니다.", muted=True)
    learning = _section(tools.content, "학습 · 품질 확인")
    _action_rows(app, learning, [
        ("수정값 저장", "save_corrections", "save_button"), ("추가 학습", "train_model", "train_button"),
        ("Gold 평가", "evaluate_current_matrix_against_gold"), ("Gold 확정", "confirm_current_matrix_as_gold"),
        ("GIS 화살표 학습", "generate_gis_arrow_training"), ("모델 초기화", "reset_model_state"),
    ], columns=1)
    if not training_enabled:
        for child in learning.winfo_children():
            if isinstance(child, ttk.Button):
                child.configure(state=tk.DISABLED)

    app.settings_outer_frame = settings.content
    runtime = _section(settings.content, "모델 · 실행 장치")
    _wrapped(runtime, "분석과 학습에 사용할 장치를 선택합니다.", muted=True)
    app.device_combo = ttk.Combobox(runtime, textvariable=app.device_choice_var, values=list(device_choices or ["자동", "CPU"]), state="readonly", width=16)
    app.device_combo.pack(fill=tk.X, pady=(0, 8))
    app._workspace_action_buttons.append(app.device_combo)
    _button(app, runtime, "설정 저장", "apply_settings").pack(fill=tk.X, pady=(0, 8))
    _wrapped(runtime, variable=app.runtime_var, muted=True)
    theme = _section(settings.content, "화면 테마")
    app.theme_combo = ttk.Combobox(theme, textvariable=app.theme_var, values=list(theme_names or ["NFMAT White"]), state="readonly", width=16)
    app.theme_combo.pack(fill=tk.X, pady=(0, 8))
    app._workspace_action_buttons.append(app.theme_combo)
    _button(app, theme, "테마 적용", "apply_settings").pack(fill=tk.X)
    app.theme_preview_label = tk.Label(theme, text="NFMAT6 workspace", bd=0, highlightthickness=0)
    # Keep the preview API for legacy themes without forcing a 320px sidebar.
    app.theme_preview_label.pack(fill=tk.X, pady=(8, 0))
    app.brand_frame = tk.Frame(settings.content, bd=0, highlightthickness=0, padx=8, pady=10)
    app.brand_frame.pack(fill=tk.X)
    app.brand_image_label = tk.Label(app.brand_frame, bd=0, highlightthickness=0)
    app.brand_icon_label = tk.Label(app.brand_frame, bd=0, highlightthickness=0)
    app.brand_title_label = tk.Label(app.brand_frame, text="NFMAT 6", font=("Malgun Gothic", 11, "bold"))
    app.brand_title_label.pack(anchor=tk.W)
    app.brand_credit_label = tk.Label(app.brand_frame, justify=tk.LEFT, wraplength=240, text="영남대학교 건설시스템공학과 스마트수자원\n개발: 석민수 · 개발 보조: 박창민\nSpecial Thanks to 서용원 교수님", font=("Malgun Gothic", 8))
    app.brand_credit_label.pack(anchor=tk.W, pady=(6, 0))

    footer = ttk.Frame(outer, style="Root.TFrame")
    footer.pack(fill=tk.X, pady=(8, 0))
    app.status_label = ttk.Label(footer, textvariable=app.status_var, style="Status.TLabel", anchor=tk.W, wraplength=1150)
    app.status_label.pack(fill=tk.X)
    app.status_label.bind("<Configure>", lambda event: app.status_label.configure(wraplength=max(160, event.width - 8)))
    app._rows_trace_id = app.rows_var.trace_add("write", app._on_requested_rows_changed)
    app.after_idle(lambda: refresh_workspace_layout(app))


def refresh_workspace_layout(app, window_width=None):
    """Set initial pane widths once; subsequently preserve user sash positions."""
    if not getattr(app, "_workspace_layout", False):
        return
    width = int(window_width if window_width is not None else app.winfo_width())
    groups = app._workspace_toolbar_groups
    # Below the supported desktop width, use two compact toolbar rows.
    columns = 4 if width >= 1050 else 2
    layout = (columns,)
    if getattr(app, "_workspace_toolbar_key", None) != layout:
        app._workspace_toolbar_key = layout
        for index, (group, _content) in enumerate(groups):
            group.grid_configure(row=index // columns, column=index % columns, pady=(0 if index < columns else 8, 0))
        for column in range(4):
            app.toolbar_frame.columnconfigure(column, weight=(1 if column < columns else 0))
    if not app._workspace_sashes_initialized and app.workspace_panes.winfo_width() > 600:
        pane_width = app.workspace_panes.winfo_width()
        left_width = 254 if width < 1450 else 280
        right_width = 300 if width < 1450 else 330
        app.workspace_panes.sashpos(0, left_width)
        app.workspace_panes.sashpos(1, max(left_width + 280, pane_width - right_width))
        app._workspace_sashes_initialized = True


def apply_workspace_style(app):
    """Refine common ttk styles after the application's selected theme is set."""
    style = ttk.Style(app)
    palette = app._theme_palette() if hasattr(app, "_theme_palette") else {}
    is_default = not palette or getattr(app, "theme_var", None) is None or app.theme_var.get() == "NFMAT White"
    bg = "#f2f5f8" if is_default else palette.get("bg", "#f2f5f8")
    surface = "#ffffff" if is_default else palette.get("panel", "#ffffff")
    text = "#263549" if is_default else palette.get("text", "#263549")
    muted = "#6b7a8c" if is_default else palette.get("muted", "#6b7a8c")
    accent = "#2563a6" if is_default else palette.get("accent", "#2563a6")
    border = "#dbe3eb" if is_default else palette.get("border", "#dbe3eb")
    app.configure(background=bg)
    style.configure("Root.TFrame", background=bg)
    style.configure("Workspace.TFrame", background=surface)
    style.configure("TPanedwindow", background=bg, sashwidth=8)
    style.configure("Workspace.TLabel", background=surface, foreground=text, font=("Malgun Gothic", 9))
    style.configure("Muted.TLabel", background=surface, foreground=muted, font=("Malgun Gothic", 9))
    style.configure("Brand.TLabel", background=bg, foreground=text, font=("Segoe UI", 18, "bold"))
    style.configure("Subtitle.TLabel", background=bg, foreground=muted, font=("Malgun Gothic", 10))
    style.configure("MutedHeader.TLabel", background=bg, foreground=muted, font=("Malgun Gothic", 9))
    style.configure("PanelHeading.TLabel", background=surface, foreground=text, font=("Malgun Gothic", 10, "bold"))
    style.configure("Step.TLabel", background=surface, foreground=muted, font=("Malgun Gothic", 9, "bold"))
    style.configure("Workspace.TCheckbutton", background=surface, foreground=text, font=("Malgun Gothic", 9))
    style.configure("CanvasHint.TLabel", background=surface, foreground=muted, font=("Malgun Gothic", 8))
    style.configure("Status.TLabel", background=bg, foreground=text, font=("Malgun Gothic", 9))
    style.configure("Workspace.TButton", background="#f6f8fb" if is_default else surface, foreground=text, bordercolor=border, lightcolor=border, darkcolor=border, padding=(8, 5), borderwidth=1, relief="flat", font=("Malgun Gothic", 9))
    style.map("Workspace.TButton", background=[("pressed", "#dce8f5"), ("active", "#eaf1f8")], foreground=[("disabled", "#a2adba"), ("active", text)])
    style.configure("Primary.TButton", background=accent, foreground="#ffffff", bordercolor=accent, lightcolor=accent, darkcolor=accent, padding=(10, 5), borderwidth=1, relief="flat", font=("Malgun Gothic", 9, "bold"))
    style.map("Primary.TButton", background=[("disabled", "#9cb2cb"), ("pressed", "#194d83"), ("active", "#3077bd")], foreground=[("disabled", "#eef3f8"), ("active", "#ffffff")])
    style.configure("Workspace.TLabelframe", background=surface, bordercolor=border, lightcolor=border, darkcolor=border, borderwidth=1, relief="solid")
    style.configure("Workspace.TLabelframe.Label", background=surface, foreground=muted, font=("Malgun Gothic", 9, "bold"))
    style.configure("Workspace.TNotebook", background=surface, borderwidth=0, tabmargins=(0, 0, 0, 0))
    style.configure("Workspace.TNotebook.Tab", background=bg, foreground=muted, padding=(9, 7), font=("Malgun Gothic", 9))
    style.map("Workspace.TNotebook.Tab", background=[("selected", surface), ("active", "#eaf1f8")], foreground=[("selected", accent), ("active", text)])
    style.configure("Workspace.Treeview", background=surface, fieldbackground=surface, foreground=text, bordercolor=border, rowheight=27, font=("Malgun Gothic", 9))
    style.configure("Workspace.Treeview.Heading", background=bg, foreground=muted, padding=(4, 6), font=("Malgun Gothic", 8, "bold"))
    style.map("Workspace.Treeview", background=[("selected", "#dfebf7")], foreground=[("selected", "#174979")])
    style.configure("Workspace.Horizontal.TProgressbar", troughcolor=border, background=accent, bordercolor=border, lightcolor=accent, darkcolor=accent, thickness=5)
    app.canvas.configure(background="#e8eef3" if is_default else palette.get("canvas", "#e8eef3"))
    app.matrix_text.configure(background=surface, foreground=text, insertbackground=text, selectbackground=accent, selectforeground="#ffffff", relief=tk.FLAT, borderwidth=0, highlightthickness=0)
    for panel in getattr(app, "_workspace_scroll_panels", []):
        panel.viewport.configure(background=surface)
    for name in ("toolbar_frame", "layer_panel_frame", "left_frame", "right_frame", "info_top_frame", "settings_outer_frame"):
        widget = getattr(app, name, None)
        if isinstance(widget, ttk.Frame):
            widget.configure(style="Workspace.TFrame")
