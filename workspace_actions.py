# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Uses NumPy: https://github.com/numpy/numpy (BSD-3-Clause plus bundled notices).
# Uses Tkinter / Tcl-Tk: https://docs.python.org/3/library/tkinter.html (runtime licenses).
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
"""NFMAT6 workspace actions; workers only communicate with Tk through a queue."""
from __future__ import annotations

import functools
import queue
import threading
import time
from pathlib import Path
from tkinter import messagebox

import numpy as np

from workspace_state import basin_mask_from_world_rings, intersect_region_masks, resample_mask_centers


BASE_SUPPORT_NAMES = ("occ_matrix", "path_occ_matrix", "support_mask")


def guarded_action(method):
    @functools.wraps(method)
    def guarded(self, *args, **kwargs):
        if not self._workspace_mutation_allowed():
            return None
        if self._has_area_fill() and method.__name__ not in {
            "export_matrix", "run_plena", "clear_edits", "load_image", "add_gis_layers",
            "enable_outlet_pick_mode", "apply_bfs_assist",
        }:
            self.status_var.set("면적 배수가 채워져 있습니다. [채우기 되돌리기] 후 관망을 수정해 주세요.")
            return None
        return method(self, *args, **kwargs)
    return guarded


class WorkspaceActions:
    def _initialize_workspace_actions(self):
        self._task_busy = False
        self._task_label = ""
        self._task_queue = queue.Queue()
        self._task_generation = 0
        self._task_cancel_event = threading.Event()
        self._task_cancellable = True
        self._close_requested = False
        self._closing = False
        self._plena_starting = False
        self._plena_cancelled = False
        self._plena_ui_queue = queue.Queue()
        self._basin_cell_mask = None
        self._basin_is_lasso = False
        self._basin_ring_groups = None
        self._area_fill_mask = None
        self._area_fill_values = None
        self._base_support_state = {}
        self._matrix_preview_after = None
        self._matrix_preview_digest = None

    def _workspace_mutation_allowed(self):
        state = vars(self)
        if state.get("_task_busy") or state.get("_plena_starting"):
            self.status_var.set(f"{state.get('_task_label') or 'PLENA'} 작업 중입니다. 완료 또는 중단 후 편집해 주세요.")
            return False
        proc = state.get("_plena_process")
        if proc is not None and proc.poll() is None:
            self.status_var.set("PLENA 계산 중에는 입력을 변경할 수 없습니다. 먼저 계산을 완료하거나 중단해 주세요.")
            return False
        for flag in ("_gis_crs_job_polling", "_gis_load_in_progress", "_gis_load_polling", "_gis_boundary_job_polling", "_gis_pipe_job_polling"):
            if state.get(flag):
                self.status_var.set("GIS 작업을 처리하는 중입니다. 완료 후 다음 작업을 시작해 주세요.")
                return False
        return True

    def _has_area_fill(self):
        mask = vars(self).get("_area_fill_mask")
        return isinstance(mask, np.ndarray) and bool(np.any(mask))

    def _sync_workspace_task_controls(self):
        state = vars(self)
        proc = state.get("_plena_process")
        busy = bool(state.get("_task_busy") or state.get("_plena_starting") or (proc is not None and proc.poll() is None))
        for widget in state.get("_workspace_action_buttons", []):
            try:
                if busy:
                    widget.state(["disabled"])
                else:
                    widget.state(["!disabled"])
            except Exception:
                pass
        progress = state.get("task_progress")
        if progress is not None:
            progress.start(16) if busy else progress.stop()
        cancel = state.get("task_cancel_button")
        if cancel is not None:
            cancel.configure(state="normal" if busy and state.get("_task_cancellable", True) else "disabled")
        label = state.get("task_label_var")
        if label is not None:
            label.set(state.get("_task_label", "") if busy else "준비됨 · CPU")
        undo = state.get("undo_fill_button")
        if undo is not None:
            undo.configure(state="normal" if self._has_area_fill() and not busy else "disabled")

    def _start_compute_task(self, label, compute, on_done, *, cancellable=True):
        if not self._workspace_mutation_allowed():
            return False
        if "_task_queue" not in vars(self):
            self._initialize_workspace_actions()
        self._task_busy = True
        self._task_label = str(label)
        self._task_generation += 1
        generation = self._task_generation
        self._task_cancel_event = threading.Event()
        cancellation = self._task_cancel_event
        self._task_cancellable = bool(cancellable)
        self._task_on_done = on_done
        self._task_started = time.perf_counter()
        results = self._task_queue
        self.status_var.set(f"{label} 중… 지도 이동과 확대는 계속 사용할 수 있습니다.")
        self._sync_workspace_task_controls()

        def worker():
            try:
                value = compute(cancellation)
                results.put((generation, value, None))
            except Exception as error:
                results.put((generation, None, error))

        threading.Thread(target=worker, name="NFMAT6-compute", daemon=True).start()
        self.after(60, self._poll_compute_task)
        return True

    def _poll_compute_task(self):
        if vars(self).get("_closing"):
            return
        try:
            generation, value, error = self._task_queue.get_nowait()
        except queue.Empty:
            if self._task_busy:
                self.after(60, self._poll_compute_task)
            return
        if generation != self._task_generation:
            self.after(60, self._poll_compute_task)
            return
        self._task_busy = False
        cancelled = self._task_cancel_event.is_set()
        try:
            if cancelled:
                self.status_var.set(f"{self._task_label} 중단 완료 · 계산 결과를 적용하지 않았습니다.")
            elif error is not None:
                self.status_var.set(f"{self._task_label} 실패 · 기존 작업은 유지됩니다.")
                if not self._close_requested:
                    messagebox.showerror(f"{self._task_label} 오류", str(error), parent=self)
            else:
                self._task_on_done(value)
        except Exception as failure:
            self.status_var.set(f"결과 적용 실패: {failure}")
            if not self._close_requested:
                messagebox.showerror("결과 적용 오류", str(failure), parent=self)
        finally:
            self._task_on_done = None
            self._sync_workspace_task_controls()
            self._update_action_states()
            if self._close_requested:
                self._on_workspace_close()

    def cancel_active_task(self):
        if vars(self).get("_task_busy"):
            if self._task_cancellable:
                self._task_cancel_event.set()
                self.status_var.set("중단 요청됨 · 현재 계산 종료 후 결과를 폐기합니다.")
            return
        self._plena_cancelled = True
        self._terminate_plena_process()

    def _post_plena_callback(self, delay, callback):
        self._plena_ui_queue.put(callback)

    def _poll_plena_callbacks(self):
        if vars(self).get("_closing"):
            return
        for _ in range(150):
            try:
                callback = self._plena_ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                callback()
            except Exception as error:
                self.status_var.set(f"PLENA 결과 표시 오류: {error}")
        if self._plena_starting or not self._plena_ui_queue.empty():
            self.after(50, self._poll_plena_callbacks)

    def _finish_plena_task(self):
        self._plena_process = None
        self._plena_starting = False
        self._sync_workspace_task_controls()
        self._update_action_states()
        if self._close_requested:
            self._on_workspace_close()

    def _on_workspace_close(self):
        self._close_requested = True
        if vars(self).get("_task_busy"):
            if self._task_cancellable:
                self._task_cancel_event.set()
            self.status_var.set("현재 작업을 정리한 뒤 창을 닫습니다. 학습 저장 중이면 완료를 기다립니다.")
            return
        proc = vars(self).get("_plena_process")
        if proc is not None and proc.poll() is None:
            self._plena_cancelled = True
            proc.terminate()
            self.after(100, self._on_workspace_close)
            return
        self._closing = True
        self.destroy()

    def _effective_basin_mask(self):
        matrix = vars(self).get("current_matrix")
        if matrix is None:
            return None
        return intersect_region_masks(matrix.shape, vars(self).get("_basin_cell_mask"), vars(self).get("roi_cell_mask"))

    def _active_edit_mask(self):
        """Only explicit LASSO limits the rectangular editing/export workspace.

        GIS watershed polygons select area-fill cells, not permissible pipes or
        outlet locations. A surveyed/edited pipe may leave that fill domain.
        GIS LASSO clipping remains an explicit restriction after rasterization.
        """
        matrix = vars(self).get("current_matrix")
        if matrix is None:
            return None
        gis_lasso = vars(self).get("_basin_cell_mask") if vars(self).get("_basin_is_lasso") else None
        return intersect_region_masks(matrix.shape, vars(self).get("roi_cell_mask"), gis_lasso)

    def _rebuild_basin_mask(self):
        rings = vars(self).get("_basin_ring_groups")
        bbox = vars(self).get("_gis_active_bbox")
        self._basin_cell_mask = (
            basin_mask_from_world_rings(rings, self.current_matrix.shape, bbox, self._rotation_degrees())
            if rings and bbox is not None and self.current_matrix is not None else None
        )

    def _record_support_baseline(self):
        self._base_support_state = {
            name: value.copy() for name in BASE_SUPPORT_NAMES
            if isinstance((value := vars(self).get(name)), np.ndarray)
        }
        self._area_fill_mask = None
        self._area_fill_values = None

    def _restore_support_baseline(self):
        shape = self.base_matrix.shape
        values = vars(self).get("_base_support_state", {})
        for name in BASE_SUPPORT_NAMES:
            baseline = values.get(name)
            setattr(self, name, baseline.copy() if isinstance(baseline, np.ndarray) and baseline.shape == shape else (self.base_matrix > 0).astype(np.uint8))
        self._area_fill_mask = None
        self._area_fill_values = None

    def _clip_workspace_masks(self, bounds):
        r0, r1, c0, c1 = bounds
        for name, value in vars(self).get("_base_support_state", {}).items():
            self._base_support_state[name] = value[r0:r1, c0:c1].copy()

    def _apply_workspace_region(self):
        mask = self._active_edit_mask()
        if mask is None:
            return
        outside = mask == 0
        for name in ("base_matrix", "current_matrix", "base_confidence_matrix", "confidence_matrix", "base_model_applied_mask", "model_applied_mask", "occ_matrix", "path_occ_matrix", "support_mask", "unresolved_mask", "user_edit_mask", "_flood_cell_mask", "_area_fill_mask", "_area_fill_values"):
            value = vars(self).get(name)
            if isinstance(value, np.ndarray) and value.shape == mask.shape:
                value[outside] = 0
        for value in vars(self).get("_base_support_state", {}).values():
            if value.shape == mask.shape:
                value[outside] = 0

    def fill_empty_cells(self):
        from solution import fill_basin_to_network
        if not self._workspace_mutation_allowed():
            return
        if self.current_matrix is None:
            messagebox.showinfo("빈 칸 채우기", "먼저 관망 행렬을 생성해 주세요.", parent=self)
            return
        if self._has_area_fill():
            self.status_var.set("빈 칸 채우기가 이미 적용되어 있습니다. 다시 계산하려면 먼저 되돌려 주세요.")
            return
        try:
            basin = self._effective_basin_mask()
            if basin is None:
                raise ValueError("유역 경계가 필요합니다. GIS에서 유역을 선택하거나 LASSO로 유역 범위를 확정해 주세요.")
            outlets = sorted(self._active_outlet_cells())
            original = self.current_matrix.copy()
            network_mask = self._active_edit_mask()
            if network_mask is None:
                network_mask = np.ones_like(original, dtype=np.uint8)
            else:
                original[network_mask == 0] = 0
            def finish(result):
                filled, fill_mask, stats = result
                self.current_matrix = filled
                self._area_fill_mask = fill_mask.copy()
                self._area_fill_values = np.where(fill_mask > 0, filled, 0).astype(np.uint8)
                self._invalidate_matrix_arrow_cache()
                self.status_var.set(
                    f"최근접 관로 채우기 완료 · {int(np.count_nonzero(fill_mask)):,}셀 추가 · "
                    f"관로까지 경로가 없는 0셀 {stats.get('unfilled_count', 0):,}개. "
                    "Outlet 연결 여부와 무관하게 기존 관로에 연결했습니다. 이후 BFS 연결 보정으로 Outlet 연결을 검사하세요."
                )
                self._queue_render(high_quality=True, delay=0)
                self._update_matrix_preview()
            self._start_compute_task("유역 빈 칸 채우기", lambda cancellation: fill_basin_to_network(original, basin, outlets, network_mask=network_mask), finish)
        except Exception as exc:
            messagebox.showwarning("빈 칸 채우기", str(exc), parent=self)

    def _on_requested_rows_changed(self, *_args):
        matrix = vars(self).get("current_matrix")
        if vars(self).get("source_kind") != "gis" or matrix is None or vars(self).get("_task_busy"):
            return
        try:
            rows = int(self.rows_var.get())
        except Exception:
            return
        if rows != matrix.shape[0]:
            self.status_var.set(f"행 수 {matrix.shape[0]} → {rows} 변경 대기 · [GIS 방향 보정]을 눌러 원본 관로로 다시 계산하세요. 기존 행렬은 유지됩니다.")

    @guarded_action
    def recalculate_gis_directions(self):
        """Regrid the active vector subset, never raster-detect the preview."""
        from gis_matrix import matrix_from_loaded_polylines

        if vars(self).get("source_kind") == "gis_project":
            points = vars(self).get("roi_polygon_points") or []
            if len(points) >= 3:
                # Like [LASSO -> matrix], this button also confirms a valid
                # draft polygon. Never silently discard a visible selection.
                self.clip_lasso_to_matrix()
            elif points or vars(self).get("lasso_mode"):
                messagebox.showwarning("GIS 방향 보정", "LASSO 점을 최소 3개 선택해 주세요. 선택 영역만 계산하며 전체 관로로 전환하지 않습니다.", parent=self)
            else:
                self.convert_selected_pipe_layer_to_matrix()
            return
        records = vars(self).get("_gis_active_records")
        bbox = vars(self).get("_gis_active_bbox")
        if vars(self).get("source_kind") != "gis" or not records or bbox is None:
            messagebox.showwarning("GIS 방향 보정", "먼저 GIS 관로를 불러와 행렬로 변환해 주세요. 도면 이미지는 [자동 인식]을 사용합니다.", parent=self)
            return
        try:
            rows = int(self.rows_var.get())
            if not 2 <= rows <= 300:
                raise ValueError("행 수는 2~300의 정수로 입력해 주세요.")
            angle = self._rotation_degrees()
            previous_angle = float((vars(self).get("_gis_matrix_meta") or {}).get("rotation_degrees", angle))
            if abs(angle - previous_angle) > 1e-6:
                raise ValueError("회전각도 변경되어 있습니다. 먼저 [회전 적용]을 눌러 주세요.")
        except Exception as exc:
            messagebox.showwarning("GIS 방향 보정", str(exc), parent=self)
            return
        if self._has_rotation_reset_risk() and not messagebox.askyesno(
            "GIS 방향 보정",
            f"원본 GIS 관로와 표고로 {rows}행 방향행렬을 다시 계산할까요?\n"
            "현재 수동 수정과 Outlet 선택은 초기화됩니다. 선택한 유역/LASSO 범위는 유지합니다.",
            parent=self,
        ):
            return
        records = list(records)
        bbox = tuple(bbox)
        source_meta = dict(vars(self).get("_gis_active_source_meta") or {})
        old_roi = vars(self).get("roi_cell_mask")
        old_roi = old_roi.copy() if isinstance(old_roi, np.ndarray) else None
        old_points = list(vars(self).get("roi_polygon_points") or [])
        old_image_size = self.img_pil.size if old_points and self.img_pil is not None else None

        def finish(result):
            self._gis_rows = rows
            self._apply_gis_result_to_state(result)
            if old_points and old_image_size:
                width, height = self.img_pil.size
                self.roi_polygon_points = [(x * width / old_image_size[0], y * height / old_image_size[1]) for x, y in old_points]
                self.roi_cell_mask = self._roi_cell_mask_from_polygon()
            elif old_roi is not None:
                self.roi_cell_mask = resample_mask_centers(old_roi, self.current_matrix.shape)
            self._apply_workspace_region()
            self._record_support_baseline()
            self._invalidate_matrix_arrow_cache()
            self._update_matrix_preview()
            self._queue_render(high_quality=True, delay=0)
            self.canvas.focus_set()
            self.status_var.set(f"GIS 방향 보정 완료 · {self.A}×{self.B} · 원본 관로/표고로 재계산 · Outlet을 다시 선택해 주세요.")

        self._start_compute_task("GIS 방향 보정", lambda cancel: matrix_from_loaded_polylines(
            records, rows=rows, bbox=bbox, source_meta=source_meta, rotation_degrees=angle,
        ), finish)

    def undo_fill_empty_cells(self):
        if not self._workspace_mutation_allowed() or not self._has_area_fill():
            return
        mask = self._area_fill_mask > 0
        count = int(np.count_nonzero(mask))
        self.current_matrix[mask] = 0
        self._area_fill_mask = None
        self._area_fill_values = None
        self._invalidate_matrix_arrow_cache()
        self.status_var.set(f"빈 칸 채우기 되돌림 · 면적 배수 {count:,}셀을 원래 0으로 복원했습니다.")
        self._queue_render(high_quality=True, delay=0)
        self._update_matrix_preview()

    def inspect_network(self):
        from solution import analyze_direction_network
        if self.current_matrix is None or not self._workspace_mutation_allowed():
            return
        matrix = self.current_matrix.copy()
        fill = vars(self).get("_area_fill_mask")
        if isinstance(fill, np.ndarray) and fill.shape == matrix.shape:
            matrix[fill > 0] = 0
        outlets = sorted(self._active_outlet_cells())
        def finish(report):
            text = (
                f"관망 연결 구조 검사\n\n"
                f"기존 관망 셀: {report['active_count']:,}\n"
                f"Outlet 도달: {report['connected_count']:,}\n"
                f"Outlet 미도달: {report['unreachable_count']:,}\n"
                f"상류 시작 셀: {report['source_count']:,}\n"
                f"합류 셀: {report['junction_count']:,}\n"
                f"순환 내부 셀: {report['cycle_cell_count']:,}\n\n"
                "간선·지선의 연결과 합류를 격자 방향으로 검사한 결과입니다.\n"
                "실제 간선/지선 구분은 관경·관로 등급 등 원자료로 확인해야 합니다.\n"
                "빈 칸 채우기로 추가한 면적 배수 셀은 관망 검사에서 제외합니다."
            )
            self._show_text_window("연결 구조 검사", text)
            self.status_var.set(f"연결 검사 · 관망 {report['active_count']:,}셀 / 미도달 {report['unreachable_count']:,}셀 / 합류 {report['junction_count']:,}셀")
        self._start_compute_task("관망 연결 구조 검사", lambda cancellation: analyze_direction_network(matrix, outlets), finish)
