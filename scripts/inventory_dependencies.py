# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
"""Record installed dependency closure and copy its shipped license notices.

This audits the environment that executes it; it does not download code, install
packages, infer missing licenses, or establish a clean-machine installation.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
DIRECT = ("numpy", "opencv-python", "pillow", "pyproj", "scikit-image", "torch")


def main() -> None:
    queue = list(DIRECT)
    seen = set()
    records = []
    while queue:
        name = canonicalize_name(queue.pop())
        if name in seen:
            continue
        seen.add(name)
        dist = metadata.distribution(name)
        dependencies = []
        for value in dist.requires or []:
            req = Requirement(value)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                dependencies.append(str(req))
                queue.append(req.name)
        license_paths = []
        for rel in dist.files or []:
            relpath = Path(str(rel))
            # Copy license texts from distribution metadata, not unrelated names
            # such as haarcascade_license_plate.xml or Python license modules.
            if not any(part.endswith(".dist-info") for part in relpath.parts):
                continue
            if not any(word in relpath.name.upper() for word in ("LICENSE", "COPYING", "NOTICE", "COPYRIGHT")):
                continue
            source = Path(dist.locate_file(rel))
            if not source.is_file():
                continue
            # Preserve the full relative path: setuptools vendors several
            # separate dist-info trees whose LICENSE filenames would collide.
            if relpath.is_absolute() or '..' in relpath.parts:
                raise ValueError(f'Unexpected metadata path: {relpath}')
            tail = relpath
            target = ROOT / "third_party" / "licenses" / name / tail
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            license_paths.append({"path":target.relative_to(ROOT).as_posix(),
                                  "sha256":hashlib.sha256(source.read_bytes()).hexdigest()})
        records.append({"name":name,"version":dist.version,"direct":name in DIRECT,
                        "license_expression":dist.metadata.get("License-Expression"),
                        "license_metadata":dist.metadata.get("License"),
                        "project_urls":dist.metadata.get_all("Project-URL") or [],
                        "requires":sorted(dependencies),"license_files":license_paths})
    records.sort(key=lambda r:r["name"])
    report={"python":platform.python_version(),"platform":sys.platform,
            "scope":"Installed direct dependency closure; no optional extras or binary bundle audit",
            "packages":records}
    target=ROOT/"third_party"/"dependency_inventory.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lines=["# Snapshot of the verified Windows Python environment; see README for CPU Torch installation."]
    lines += [f'{r["name"]}=={r["version"]}' for r in records]
    (ROOT/"requirements-lock.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    missing=[r["name"] for r in records if not r["license_files"]]
    print(json.dumps({"packages":len(records),"notice_files":sum(len(r["license_files"]) for r in records),
                      "missing_license_files":missing}))


if __name__ == "__main__":
    main()
