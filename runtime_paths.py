# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
"""Portable FMT runtime locations, separate from source and older NFMAT data."""
from __future__ import annotations

import os
from pathlib import Path

APP_ID = "FMT_Flow_Matrix_Tool"
VERSION = "0.1.0"


def user_data_root() -> Path:
    """Resolve writable storage without importing data from older installations."""
    override = os.environ.get("FMT_DATA_DIR")
    if override:
        root = Path(override).expanduser().resolve()
    elif os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / APP_ID
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / APP_ID
    root.mkdir(parents=True, exist_ok=True)
    return root
