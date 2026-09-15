# Source release validation

Date: 2026-09-16. Platform: Windows x64. Python: 3.14.6.
PyTorch runtime: 2.11.0+cpu (`torch.version.cuda is None`).
Package versions and upstream notices: `third_party/dependency_inventory.json`.

## Executed locally

| Check | Result |
| --- | --- |
| Original pipenet6 regression baseline | 211 tests passed |
| Public source tests, including six portability checks | 217 tests passed |
| Dependency consistency | `pip check`: no broken requirements |
| Entry point | `fmt.py --help` and `--version` succeeded |
| Synthetic fill → export → undo | 69 pipe cells, 767 filled cells; no Tk callback errors |
| Near outside-basin pipe | 78 pipe cells, 765 filled cells; nearest branch retained |
| Fill → Outlet selection → BFS → export → undo | Reverse-branch case passed; nearest attachment retained |
| GIS editing | Row changes 10 → 20 → 30 and rotation passed |
| GIS LASSO dispatch | 3 input records, 2 clipped records, zero exterior active cells |

The GUI checks ran against separate temporary FMT profiles without importing
old NFMAT data. Dependency notices were copied unchanged from installed packages.
`scripts/audit_release.py --source <original-directory>` checks source headers,
personal-path removal, notice hashes and AST-equivalent scientific modules.

## Deliberate cleanup scope

Scientific algorithms were retained. Source comments, provenance and dependency
notices were added. Runtime paths and launch configuration were separated;
implicit searches of personal folders and automatic legacy-data import were
removed. Internal module/class names and the Korean UI remain largely unchanged.

Fresh-clone inspection caught a broad `dist*` ignore pattern excluding a NumPy
license, and a legacy cancellation test depending on a locally installed PLENA
executable. Ignore patterns are now root-anchored; the test uses an inert temporary
fixture and a separate missing-engine test verifies the error path. The release
audit now verifies that every inventoried license is actually tracked by Git.

## Not established by these checks

- A fresh environment installation on every OS, or future package availability.
- Real-world surveyed-pipe direction accuracy, hydraulic validity, or a Water
  acceptance decision.
- Preservation of all fine-scale trunk/branch topology after arbitrary
  compression, overlap or coarse rasterization.
- Safe handling of hostile checkpoints or pickled label files (see SECURITY.md).
- Reproduction of the paper's full datasets, experiments or external PLENA batch
  analyses. No municipal GIS, private weights or reviewed-label set is bundled.

GitHub Actions, if enabled, performs its own clean-runner dependency installation
and regression run. Its live result must be checked separately from this local
record; do not treat a pending workflow as a pass.
