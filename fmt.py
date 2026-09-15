# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
"""Launch Flow Matrix Tool; keep --help/--version free of heavy dependencies."""
from __future__ import annotations

import argparse
import os
from runtime_paths import VERSION


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="FMT: Flow Matrix Tool (pipenet6 research source)")
    parser.add_argument("--version", action="version", version=f"FMT {VERSION}")
    parser.add_argument("--data-dir", help="Directory for settings, models and outputs (or FMT_DATA_DIR).")
    args = parser.parse_args(argv)
    if args.data_dir:
        os.environ["FMT_DATA_DIR"] = args.data_dir
    from gui import NFMATApp
    app = NFMATApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
