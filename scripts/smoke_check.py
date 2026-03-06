"""Minimal smoke checks for import and project structure sanity."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _assert_exists(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing required path: {path}")


def main() -> int:
    required_paths = [
        REPO_ROOT / "run_qt.py",
        REPO_ROOT / "core",
        REPO_ROOT / "ui_qt",
    ]
    for p in required_paths:
        _assert_exists(p)

    modules = [
        "core.file_ops",
        "core.data_io",
        "core.loader",
        "core.processing",
        "core.plotting",
        "core.export",
        "core.processing_run",
        "ui_qt.main_window",
    ]
    for name in modules:
        importlib.import_module(name)

    print("smoke-check: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
