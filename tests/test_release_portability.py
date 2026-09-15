# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minsoo Seok
"""Public-release portability and privacy regression checks."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import runtime_paths
import run_plena_batch
import ml_image_audit


class ReleasePortabilityTests(unittest.TestCase):
    def test_data_override_is_created_outside_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = Path(tmp) / "session"
            with mock.patch.dict(os.environ, {"FMT_DATA_DIR": str(expected)}):
                self.assertEqual(runtime_paths.user_data_root(), expected.resolve())
                self.assertTrue(expected.is_dir())

    def test_batch_explicit_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run_plena_batch.discover_source_dir(tmp), Path(tmp))

    def test_batch_missing_executable_does_not_scan_home(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Path, "cwd", return_value=Path(tmp)):
            with mock.patch.object(Path, "home", side_effect=AssertionError("Must not scan home")):
                with self.assertRaisesRegex(FileNotFoundError, "source-dir"):
                    run_plena_batch.discover_source_dir(None)

    def test_image_audit_requires_explicit_directory(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            ml_image_audit.build_parser().parse_args([])

    def test_optional_plena_override(self):
        import gui
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "PLENA.exe"
            with mock.patch.dict(os.environ, {"FMT_PLENA_EXECUTABLE": str(target)}):
                self.assertEqual(gui._resource_path("PLENA.exe"), target.resolve())


if __name__ == "__main__":
    unittest.main()
