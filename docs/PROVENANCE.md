# Scientific and software provenance

FMT is a publication-oriented source copy of pipenet6. Source comments identify
the libraries used by each module. Dependency license texts are separately
preserved; citations below describe methods, not permissions to copy code.

## GIS line rasterization

`gis_matrix._bresenham_cells` expresses incremental line rasterization following:

J. E. Bresenham (1965), Algorithm for Computer Control of a Digital Plotter,
*IBM Systems Journal* 4(1), 25–30. https://doi.org/10.1147/sj.41.0025

`_expand_diagonal_steps` is a separate FMT postprocessing policy. It adds an
orthogonal intermediate cell for a diagonal step; source-coordinate x displacement
is prioritized when `abs(dx) >= abs(dy)`, otherwise y. Do not attribute this
entire four-neighbor network reconstruction or global connectivity correction to
Bresenham. A method citation does not establish a copied-code lineage.

## Raster primitives and skeletonization

`solution.py` uses OpenCV thresholding, distance transforms, connected components,
morphological operations and template matching. API references:

- https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html
- https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
- https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html

`solution.py` and `cell_training.py` call `skimage.morphology.skeletonize`.
The default two-dimensional method is Zhang–Suen thinning; the implementation
comes from scikit-image, not an FMT reimplementation of that algorithm.

- T. Y. Zhang and C. Y. Suen (1984), A fast parallel algorithm for thinning
  digital patterns, *Communications of the ACM* 27(3), 236–239.
- https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.skeletonize

FMT combines these primitives with pipe-mask filtering, arrow proposals,
direction voting and skeleton tracing. Those integration policies should not
be attributed to OpenCV as a complete off-the-shelf drainage-network algorithm.

## GIS file format and coordinates

`gis_matrix.py` reads selected shapefile and dBASE structures locally using
Python `struct`; it does not import a third-party shapefile parser. Relevant
format authority: ESRI Shapefile Technical Description, July 1998:
https://www.esri.com/library/whitepapers/pdfs/shapefile.pdf

`crs_support.py` calls pyproj/PROJ for parsing and coordinate operations with
`always_xy=True`: https://pyproj4.github.io/pyproj/stable/api/transformer.html
The CRS recommendations in `crs_catalog.py` are heuristics, not provider metadata.

## Connectivity and structural summaries

Nearest-pipe filling uses multi-source shortest-path traversal on the permitted
four-neighbor grid; BFS and graph traversal are standard techniques. FMT-specific
choices include eligible seed selection, masks, deterministic ties, separation
of pipe cells from filled area cells, and export filtering. Do not claim invention
of breadth-first search. `docs/METHODS.md` states the behavioral contract.

`gamma_index.py` and the plotted summaries are project computations; their
scientific interpretation must be justified in the manuscript. Software tests
are not evidence that a scalar index measures sinuosity, branch importance or
flood risk.

## Optional learned correction

PyTorch provides tensor operations, neural-network layers and training utilities.
Suggested library citation: A. Paszke et al. (2019), PyTorch: An Imperative Style,
High-Performance Deep Learning Library, *NeurIPS 32*:
https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html

No reviewed-label dataset or trained weights are supplied in this release.
Project-specific architectures and training policies remain visible in source.

## Limits of provenance inspection

The release audit examined imports, existing source comments, the prior project
attribution document, installed notices and the upstream PLENA repository. It
cannot prove that no unattributed snippet ever entered historical code. Any
additional copied or adapted source identified by the authors must retain its
original attribution and license before redistribution.
