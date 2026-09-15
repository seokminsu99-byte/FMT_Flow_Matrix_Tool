# Matrix workflow contract

- Grid cells use four orthogonal directions: east 1, south 2, west 3, north 4;
  0 means no active direction. Matrix row zero is at the top.
- GIS line rasterization first forms an ordered Bresenham cell path and then
  bridges diagonal steps. At overlapping features, direction votes are combined;
  individual segment continuity is not a guarantee of final global connectivity.
- A GIS basin boundary limits new area filling. Existing pipes outside that
  basin can still connect through the allowed workspace. Explicit LASSO is a
  separate hard restriction and excludes exterior seeds and exported cells.
- Every eligible nonzero existing pipe can seed filling, even if it is cyclic
  or currently disconnected from the selected Outlet. Filled area is not used
  as a substitute surveyed-pipe seed on subsequent correction.
- Distance is orthogonal grid-step distance through permitted fill cells to
  first pipe contact. Holes and excluded zero cells cannot be crossed. This is
  not a continuous vector-space nearest-line algorithm. On rectangular cells,
  step counts are not physical-distance weights.
- Filling preserves existing pipe directions. The user then selects an existing
  pipe Outlet and may run BFS assistance. That stage repairs pipe directions,
  then updates area filling; it is distinct from nearest-pipe attachment.
- After correction, Outlet-unreachable pipes and their attached area cells are
  zeroed according to the current workflow. Export rechecks reachability and
  LASSO. Undoing filling removes area-fill cells while retaining corrected pipes.
- A single direction per cell cannot represent every crossing or parallel
  conduit. Coarse resolution and compression can change trunk/branch topology.
  Do not equate a long grid path with a surveyed main-pipe classification.

For a reproducible paper, archive the source commit, environment, input access
conditions, CRS, row/column count, rotation, LASSO/basin masks, manual edits,
Outlet choice, fill/BFS/compression sequence, exported input and reference labels.
Report failures and sensitivity to resolution, not only selected successful cases.
