from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.validate_release_version import read_app_version, validate_tag


class ReleaseVersionTests(unittest.TestCase):
    def test_reads_version_without_importing_module(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "version.py"
            path.write_text('__version__ = "1.2.3"\nraise RuntimeError("must not run")\n', encoding="utf-8")
            self.assertEqual(read_app_version(path), "1.2.3")

    def test_tag_matches_with_required_v(self) -> None:
        validate_tag("v1.2.3", "1.2.3")

    def test_tag_without_v_fails(self) -> None:
        with self.assertRaises(ValueError):
            validate_tag("1.2.3", "1.2.3")

    def test_tag_mismatch_fails(self) -> None:
        with self.assertRaises(ValueError):
            validate_tag("v1.2.4", "1.2.3")

    def test_malformed_tag_fails(self) -> None:
        for tag in ("v1.2", "v1.2.3-test", "v01.2.3", "latest"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                validate_tag(tag, "1.2.3")

    def test_invalid_version_source_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "version.py"
            path.write_text('__version__ = "1.2"\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_app_version(path)
