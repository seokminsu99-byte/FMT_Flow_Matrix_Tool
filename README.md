# FMT — Flow Matrix Tool

FMT converts GIS pipe geometry or raster drawings into editable four-direction
drainage matrices. It supports LASSO constraints, basin-area filling to nearby
pipes, connectivity correction, and PLENA input export. This is the public
research-source successor of **pipenet6**; the GUI retains Korean labels and
some internal `NFMAT` names for compatibility.

This repository supports preparation of research for *Water*. No manuscript
acceptance, DOI, or field-validation result is implied by this source release.

## Install and launch

Verified locally on Windows x64 with Python **3.14.6** and CPU-only PyTorch.
Use a separate environment; do not install into Conda base. For example:

```powershell
conda create -n fmt python=3.14 pip -y
conda activate fmt
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-lock.txt
python -m pip check
python -c "import torch; assert torch.version.cuda is None; print(torch.__version__)"
python fmt.py
```

`requirements.txt` lists direct dependencies. `requirements-lock.txt` is the
installed dependency-closure snapshot, not a universal cross-platform lock or
a guarantee of future wheel availability. The standard Python/Tk runtime must
include Tkinter. No CUDA, NPU, or driver installation is required.

```powershell
python fmt.py --help
python fmt.py --version
python fmt.py --data-dir ./local-session
```

Without `--data-dir` or `FMT_DATA_DIR`, writable data are stored under
`%LOCALAPPDATA%\FMT_Flow_Matrix_Tool` on Windows, or
`$XDG_DATA_HOME/FMT_Flow_Matrix_Tool` (default `~/.local/share`) elsewhere.
Older NFMAT data are **not automatically imported**. Do not publish a local
session directory. `python gui.py` remains supported.

## Scientific workflow and limits

1. Load GIS geometry with a known coordinate reference system, or a pipe drawing.
2. Define grid rows, inspect inferred directions, and correct them manually.
3. Apply LASSO if a hard spatial restriction is needed. A GIS basin boundary is
   a filling domain, not automatically a hard clipping boundary for real pipes.
4. Fill zero cells within the selected basin toward the nearest eligible
   existing pipe, **regardless of its current Outlet reachability**.
5. Select an existing-pipe Outlet and apply BFS connectivity correction.
6. Inspect the result, then export a PLENA input matrix.

Directions are `0=inactive`, `1=east`, `2=south`, `3=west`, `4=north`.
Nearest-pipe distance means the number of allowed orthogonal grid steps through
the fill domain until first contact with an eligible pipe. It is **not** distance
to the Outlet, continuous Euclidean distance to a vector line, or a hydraulic
travel-time estimate. Ties use deterministic traversal order. Unreachable fill
components remain zero. LASSO-exterior cells and Outlet-unreachable cells are
filtered at export. See [method details](docs/METHODS.md).

Area-fill cells are an allocation assumption, not surveyed conduits. Trunk and
branch directions are preserved by filling; BFS is a separate correction step.
Grid resolution, overlapping pipes and compression can alter topology or path
length. Synthetic tests do not establish real-data direction accuracy or flood
prediction skill. Report those validations separately in any paper.

## PLENA is a separate project

The optional engine is [FMT_plug_PLENA](https://github.com/seokminsu99-byte/FMT_plug_PLENA),
not part of the Python direction-extraction algorithm. This source release does
not redistribute a prebuilt `PLENA.exe`. Build the pinned upstream source using
[these instructions](docs/PLENA.md), then set:

```powershell
$env:FMT_PLENA_EXECUTABLE = 'C:\path\to\PLENA.exe'
python fmt.py
```

Matrix editing and text export do not require the engine. PLENA invocation
requires a compatible executable and its own validation. The batch Excel helper
also remains available: `python run_plena_batch.py --help`.

## Tests and examples

```powershell
python -m unittest discover -s tests -v
python scripts/run_gui_checks.py
python qa_workspace.py --demo
python ml_image_audit.py --image-dir ./images --help
```

The demo constructs a synthetic trunk and branches in memory, without municipal
GIS data. Real-Tk checks require a graphical desktop (or a configured display
server). The isolated runner uses temporary writable profiles, covering fill,
nearby outside-basin pipes, fill → Outlet → BFS → export → undo, row changes,
rotation and GIS LASSO handling. See [validation](docs/VALIDATION.md).

## Repository map

| Files | Responsibility |
| --- | --- |
| `fmt.py`, `runtime_paths.py` | Entry point and isolated writable storage |
| `gui.py`, `gui_layout.py`, `workspace_actions.py`, `workspace_state.py` | UI and workspace lifecycle |
| `solution.py`, `matrix_arrows.py` | Raster recognition, direction grids and connectivity |
| `gis_matrix.py`, `gis_project.py`, `crs_support.py`, `crs_catalog.py` | GIS reading, CRS and rasterization |
| `cell_training.py`, `gold_labels.py`, `ml_image_audit.py`, `synthetic_arrows.py` | Optional ML training and reviewed-label evaluation |
| `gamma_index.py`, `plena_plots.py`, `run_plena_batch.py` | Structural summaries and external PLENA workflow |
| `tests/`, `qa_*.py`, `scripts/` | Regression tests, synthetic demo and release checks |

Trained weights, training datasets, surveyed GIS, private outputs, executables,
and cosmetic background/icon assets are intentionally not bundled. Image
recognition without supplied trained weights uses its morphology-based fallback.
The GUI works without optional artwork. Learn only from reviewed labels.

## Provenance, licenses and citation

Project code: [MIT](LICENSE), copyright 2026 Minsoo Seok. Dependencies retain
their own licenses; this project's MIT license does not relicense them.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) maps components to sources and
shipped license texts. Source comments point to those records and
[algorithm provenance](docs/PROVENANCE.md).

Use [CITATION.cff](CITATION.cff), and record the exact commit, environment, inputs,
grid settings and processing sequence in a publication. Cite PLENA separately
when used. This source snapshot is not a substitute for a paper's data and code
availability statement. See [AI disclosure](docs/AI_USE.md) and
[security limitations](SECURITY.md).

## 한국어 안내

원본 pipenet6와 별도로 정리한 공개용 소스입니다. 빈 칸 채우기의 목표는
Outlet이 아니라 가까운 기존 관로이며, 이후 BFS 보정과 PLENA 내보내기는
별도 단계입니다. 새 사용자 데이터 폴더를 사용하므로 기존 모델·설정을
자동으로 가져오지 않습니다. 실제 논문 결과 재현에는 해당 입력 자료와
작업 조건을 따로 기록해야 합니다.
