# Third-party sources and notices

Project license: MIT, Minsoo Seok. The following are dependencies invoked through
their public APIs, not software re-authored by FMT. Their licenses are independent.
The Python repository does not vendor library implementations or wheels.

| Component | Source | License summary | Use |
| --- | --- | --- | --- |
| NumPy | https://github.com/numpy/numpy | Primarily BSD-3-Clause; bundled notices also apply | Arrays and numerical operations |
| OpenCV | https://github.com/opencv/opencv | Apache-2.0 for current 4.x | Thresholding, morphology, components, drawing |
| opencv-python | https://github.com/opencv/opencv-python | Wrapper/wheel notices in the installed distribution | Python bindings and bundled dependencies |
| Pillow | https://github.com/python-pillow/Pillow | MIT-CMU | Image loading, Tk images, plots |
| scikit-image | https://github.com/scikit-image/scikit-image | Primarily BSD-3-Clause; per-file terms apply | Skeletonization |
| PyTorch | https://github.com/pytorch/pytorch | BSD-3-Clause and third-party notices | Optional learned correction models |
| pyproj | https://github.com/pyproj4/pyproj | MIT | CRS definitions and transforms |
| PROJ | https://github.com/OSGeo/PROJ | MIT and bundled notices | Coordinate operations underlying pyproj |
| CPython / Tkinter | https://docs.python.org/3/license.html | PSF and included terms | Runtime and standard GUI bindings |
| Tcl/Tk | https://www.tcl-lang.org/software/tcltk/license.html | Tcl/Tk license | Native Tk runtime |
| PLENA | https://github.com/seokminsu99-byte/FMT_plug_PLENA | MIT; separate authors and provenance | Optional external executable, not bundled |

Exact installed versions, transitive dependencies, available package metadata and
SHA-256 hashes of copied notices are in
[`third_party/dependency_inventory.json`](third_party/dependency_inventory.json).
Full notices are copied unmodified under [`third_party/licenses`](third_party/licenses).
For NumPy, OpenCV and Torch in particular, read the full bundled notices rather
than assuming one short license label covers every underlying component.

The inventory reflects the tested Windows environment. It excludes optional
package extras, the separately installed Python/Tcl runtime, and build tools.
It is not a universal binary-distribution compliance report. If distributing a
frozen executable later, audit and include notices for everything actually
bundled, including runtime DLLs, codecs and numerical libraries. PyInstaller's
own license and bootloader exception must also be checked for that distribution.

Algorithm citations are distinct from licenses. See [PROVENANCE](docs/PROVENANCE.md)
for Bresenham, skeletonization, shapefile format and project-specific integration.
No attribution-only claim overrides upstream redistribution conditions.
