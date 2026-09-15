# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
# FMT / pipenet6 lineage and algorithm references: docs/PROVENANCE.md.
# Full third-party terms: THIRD_PARTY_NOTICES.md and third_party/licenses/.
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
DIR_DELTA = {
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
    4: (-1, 0),
}


@dataclass(frozen=True)
class BasinInput:
    excel_path: Path
    rows: int
    cols: int
    outlet_row: int
    outlet_col: int
    matrix: List[List[int]]
    active_count: int
    connected_count: int
    disconnected_count: int


def col_letters_to_index(col_letters: str) -> int:
    value = 0
    for ch in col_letters:
        value = value * 26 + (ord(ch.upper()) - ord("A") + 1)
    return value


def decode_cell_value(cell: ET.Element, shared_strings: List[str]) -> str | None:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        text = "".join(node.text or "" for node in cell.iterfind(".//a:t", NS))
        return text

    value_node = cell.find("a:v", NS)
    if value_node is None or value_node.text is None:
        return None

    raw = value_node.text.strip()
    if cell_type == "s":
        return shared_strings[int(raw)]
    return raw


def load_shared_strings(book: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in book.namelist():
        return []

    root = ET.fromstring(book.read("xl/sharedStrings.xml"))
    items: List[str] = []
    for item in root.findall("a:si", NS):
        text = "".join(node.text or "" for node in item.iterfind(".//a:t", NS))
        items.append(text)
    return items


def first_sheet_xml_path(book: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(book.read("xl/workbook.xml"))
    rels = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    first_sheet = workbook.find("a:sheets", NS)[0]
    rel_id = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
    return "xl/" + rel_map[rel_id].lstrip("/")


def load_numeric_direction_cells(xlsx_path: Path) -> Dict[Tuple[int, int], int]:
    with zipfile.ZipFile(xlsx_path) as book:
        sheet_xml_path = first_sheet_xml_path(book)
        shared_strings = load_shared_strings(book)
        sheet = ET.fromstring(book.read(sheet_xml_path))

    sheet_data = sheet.find("a:sheetData", NS)
    if sheet_data is None:
        raise ValueError(f"sheetData를 찾을 수 없습니다: {xlsx_path}")

    cells: Dict[Tuple[int, int], int] = {}
    for row in sheet_data.findall("a:row", NS):
        for cell in row.findall("a:c", NS):
            cell_ref = cell.attrib.get("r")
            if not cell_ref:
                continue
            match = re.fullmatch(r"([A-Z]+)(\d+)", cell_ref)
            if not match:
                continue
            raw_value = decode_cell_value(cell, shared_strings)
            if raw_value is None or raw_value == "":
                continue
            try:
                value = int(float(raw_value))
            except ValueError as exc:
                raise ValueError(f"숫자가 아닌 셀 값을 발견했습니다: {xlsx_path} {cell_ref}={raw_value!r}") from exc
            if value == 0:
                continue
            if value not in DIR_DELTA:
                raise ValueError(f"방향 코드는 1~4만 허용됩니다: {xlsx_path} {cell_ref}={value}")
            row_idx = int(match.group(2))
            col_idx = col_letters_to_index(match.group(1))
            cells[(row_idx, col_idx)] = value
    if not cells:
        raise ValueError(f"유효한 방향 셀이 없습니다: {xlsx_path}")
    return cells


def find_unique_outlet(active_cells: Dict[Tuple[int, int], int], source: Path) -> Tuple[int, int]:
    sinks: List[Tuple[int, int]] = []
    positions = set(active_cells)
    for (row, col), direction in active_cells.items():
        dr, dc = DIR_DELTA[direction]
        target = (row + dr, col + dc)
        if target not in positions:
            sinks.append((row, col))

    if len(sinks) != 1:
        raise ValueError(f"유일한 outlet을 찾지 못했습니다: {source} sink_count={len(sinks)}")
    return sinks[0]


def count_cells_reaching_outlet(active_cells: Dict[Tuple[int, int], int], outlet: Tuple[int, int]) -> int:
    positions = set(active_cells)
    upstream: Dict[Tuple[int, int], List[Tuple[int, int]]] = {pos: [] for pos in positions}
    for pos, direction in active_cells.items():
        dr, dc = DIR_DELTA[direction]
        target = (pos[0] + dr, pos[1] + dc)
        if target in upstream:
            upstream[target].append(pos)

    stack = [outlet]
    visited = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(upstream[current])
    return len(visited)


def basin_input_from_excel(xlsx_path: Path) -> BasinInput:
    active_cells = load_numeric_direction_cells(xlsx_path)
    outlet_row, outlet_col = find_unique_outlet(active_cells, xlsx_path)
    connected_count = count_cells_reaching_outlet(active_cells, (outlet_row, outlet_col))
    disconnected_count = len(active_cells) - connected_count

    rows = max(row for row, _ in active_cells)
    cols = max(col for _, col in active_cells)
    matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    for (row, col), value in active_cells.items():
        matrix[row - 1][col - 1] = value

    return BasinInput(
        excel_path=xlsx_path,
        rows=rows,
        cols=cols,
        outlet_row=outlet_row,
        outlet_col=outlet_col,
        matrix=matrix,
        active_count=len(active_cells),
        connected_count=connected_count,
        disconnected_count=disconnected_count,
    )


def discover_source_dir(cli_value: str | None) -> Path:
    if cli_value:
        source_dir = Path(cli_value).expanduser()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"입력 폴더가 없습니다: {source_dir}")
        return source_dir

    # Never scan a researcher's personal folders implicitly in a public tool.
    source_dir = Path.cwd()
    if (source_dir / "PLENA.exe").is_file():
        return source_dir
    raise FileNotFoundError("PLENA.exe와 입력 파일이 있는 폴더를 --source-dir로 지정하십시오.")


def select_excel_files(source_dir: Path, min_id: int, max_id: int) -> List[Path]:
    files: List[Path] = []
    for path in source_dir.glob("*.xlsx"):
        if path.name.startswith("~$"):
            continue
        match = re.match(r"^(\d+)", path.stem)
        if not match:
            continue
        number = int(match.group(1))
        if min_id <= number <= max_id:
            files.append(path)
    files.sort(key=lambda path: (int(re.match(r"^(\d+)", path.stem).group(1)), path.name.lower()))
    return files


def write_plena_input(basin: BasinInput, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / f"{basin.excel_path.stem}.txt"
    with txt_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{basin.rows} {basin.cols}\n")
        handle.write(f"{basin.outlet_row} {basin.outlet_col}\n")
        for row in basin.matrix:
            handle.write(" ".join(str(value) for value in row))
            handle.write("\n")
    return txt_path


def run_plena(
    plena_exe: Path,
    txt_path: Path,
    beta_a: float,
    output_dir: Path,
    width_function: bool,
    width_batch: bool,
    timeout_sec: int,
) -> Tuple[subprocess.CompletedProcess[str], Path]:
    width_answer = "y" if width_function else "n"
    batch_answer = "y" if width_batch else "n"
    if width_function:
        stdin_payload = f"{beta_a}\n{width_answer}\n{batch_answer}\n\n"
    else:
        stdin_payload = f"{beta_a}\n{width_answer}\n\n"
    completed = subprocess.run(
        [str(plena_exe), str(txt_path)],
        input=stdin_payload,
        text=True,
        capture_output=True,
        cwd=str(output_dir),
        timeout=timeout_sec,
        errors="replace",
        check=False,
    )
    result_path = txt_path.with_name(f"{txt_path.stem}_결과{txt_path.suffix}")
    return completed, result_path


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="서울 배수분구 엑셀을 PLENA 입력으로 변환하고 일괄 실행합니다.")
    parser.add_argument("--source-dir", help="PLENA.exe와 엑셀들이 있는 폴더")
    parser.add_argument("--output-root", help="결과 폴더들을 저장할 루트 폴더. 기본값은 source-dir")
    parser.add_argument("--min-id", type=int, default=1, help="처리 시작 번호")
    parser.add_argument("--max-id", type=int, default=30, help="처리 끝 번호")
    parser.add_argument(
        "--beta-a",
        type=float,
        default=0.0,
        help="PLENA에 넣을 beta 지수 a 값. 실제 beta는 10^a 입니다. 기본값은 0",
    )
    parser.add_argument(
        "--width-function",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="폭함수 계산 여부. 기본값은 계산",
    )
    parser.add_argument(
        "--width-batch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="폭함수 이후 여러 beta 배치 비교까지 수행할지 여부. 기본값은 수행 안 함",
    )
    parser.add_argument("--timeout-sec", type=int, default=600, help="파일당 PLENA 실행 제한 시간(초)")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    source_dir = discover_source_dir(args.source_dir)
    plena_exe = source_dir / "PLENA.exe"
    if not plena_exe.is_file():
        raise FileNotFoundError(f"PLENA.exe가 없습니다: {plena_exe}")
    output_root = Path(args.output_root).expanduser() if args.output_root else source_dir

    excel_files = select_excel_files(source_dir, args.min_id, args.max_id)
    if not excel_files:
        raise FileNotFoundError(f"{args.min_id}~{args.max_id} 범위의 엑셀 파일을 찾지 못했습니다: {source_dir}")

    summary_rows: List[dict] = []
    print(f"source_dir={source_dir}")
    print(f"plena_exe={plena_exe}")
    print(f"output_root={output_root}")
    print(f"selected_files={len(excel_files)}")
    print(f"beta=10^{args.beta_a}")
    print(f"width_function={args.width_function}")
    print(f"width_batch={args.width_batch}")

    for index, excel_path in enumerate(excel_files, start=1):
        t0 = time.perf_counter()
        basin_name = excel_path.stem
        output_dir = output_root / basin_name
        print(f"[{index}/{len(excel_files)}] {excel_path.name} 처리 시작")

        record = {
            "excel_name": excel_path.name,
            "output_dir": str(output_dir),
            "beta_a": args.beta_a,
        }

        try:
            basin = basin_input_from_excel(excel_path)
            txt_path = write_plena_input(basin, output_dir)
            completed, result_path = run_plena(
                plena_exe,
                txt_path,
                args.beta_a,
                output_dir,
                args.width_function,
                args.width_batch,
                args.timeout_sec,
            )

            write_text(output_dir / "plena_stdout.txt", completed.stdout)
            write_text(output_dir / "plena_stderr.txt", completed.stderr)

            if completed.returncode != 0:
                raise RuntimeError(f"PLENA 종료 코드가 0이 아닙니다: {completed.returncode}")
            if not result_path.is_file():
                raise FileNotFoundError(f"결과 파일이 생성되지 않았습니다: {result_path}")

            fd_path = output_dir / f"{txt_path.name}_FD.csv"
            ls_path = output_dir / f"{txt_path.name}_LS.csv"
            width_functions_path = output_dir / f"{txt_path.name}_width_functions.csv"
            if args.width_function:
                if not fd_path.is_file() or not ls_path.is_file() or not width_functions_path.is_file():
                    raise FileNotFoundError(
                        "폭함수 결과가 생성되지 않았습니다: "
                        f"FD={fd_path.is_file()} LS={ls_path.is_file()} WF={width_functions_path.is_file()}"
                    )

            meta = {
                "excel_path": str(excel_path),
                "txt_path": str(txt_path),
                "result_path": str(result_path),
                "fd_path": str(fd_path) if args.width_function else "",
                "ls_path": str(ls_path) if args.width_function else "",
                "width_functions_path": str(width_functions_path) if args.width_function else "",
                "rows": basin.rows,
                "cols": basin.cols,
                "outlet_row": basin.outlet_row,
                "outlet_col": basin.outlet_col,
                "active_count": basin.active_count,
                "connected_count": basin.connected_count,
                "disconnected_count": basin.disconnected_count,
                "beta_a": args.beta_a,
                "width_function": args.width_function,
                "width_batch": args.width_batch,
                "returncode": completed.returncode,
                "elapsed_sec": round(time.perf_counter() - t0, 6),
            }
            write_text(output_dir / "run_meta.json", json.dumps(meta, ensure_ascii=False, indent=2))

            if basin.disconnected_count > 0:
                warning_text = (
                    f"outlet로 연결되지 않는 셀이 {basin.disconnected_count}개 있습니다. "
                    "원본 엑셀 구조는 그대로 유지한 채 PLENA를 실행했습니다.\n"
                )
                write_text(output_dir / "run_warning.txt", warning_text)
            else:
                warning_text = ""

            record.update(
                {
                    "status": "ok",
                    "rows": basin.rows,
                    "cols": basin.cols,
                    "outlet_row": basin.outlet_row,
                    "outlet_col": basin.outlet_col,
                    "active_count": basin.active_count,
                    "connected_count": basin.connected_count,
                    "disconnected_count": basin.disconnected_count,
                    "txt_path": str(txt_path),
                    "result_path": str(result_path),
                    "fd_path": str(fd_path) if args.width_function else "",
                    "ls_path": str(ls_path) if args.width_function else "",
                    "width_functions_path": str(width_functions_path) if args.width_function else "",
                    "elapsed_sec": round(time.perf_counter() - t0, 6),
                    "warning": warning_text.strip(),
                }
            )
            print(f"[{index}/{len(excel_files)}] {excel_path.name} 완료 -> {result_path}")
        except Exception as exc:  # noqa: BLE001
            error_text = str(exc)
            output_dir.mkdir(parents=True, exist_ok=True)
            write_text(output_dir / "run_error.txt", error_text + "\n")
            record.update({"status": "error", "error": error_text, "elapsed_sec": round(time.perf_counter() - t0, 6)})
            print(f"[{index}/{len(excel_files)}] {excel_path.name} 실패 -> {error_text}", file=sys.stderr)

        summary_rows.append(record)

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "plena_batch_summary.csv"
    fieldnames = [
        "excel_name",
        "status",
        "beta_a",
        "rows",
        "cols",
        "outlet_row",
        "outlet_col",
        "active_count",
        "connected_count",
        "disconnected_count",
        "elapsed_sec",
        "output_dir",
        "txt_path",
        "result_path",
        "fd_path",
        "ls_path",
        "width_functions_path",
        "warning",
        "error",
    ]
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    ok_count = sum(1 for row in summary_rows if row.get("status") == "ok")
    print(f"완료: 성공 {ok_count} / 전체 {len(summary_rows)}")
    print(f"요약 파일: {summary_path}")
    return 0 if ok_count == len(summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
