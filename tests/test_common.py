"""Smoke tests for the shared Qt UI symbol boundary (ui_qt.common)."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QStyledItemDelegate

from ui_qt.common import (
    ExportOptions,
    LoadOptions,
    LoadedState,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    UI_METRICS,
    WrappedFilenameDelegate,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _assert_import_isolated(module_name: str) -> None:
    code = (
        "import sys; "
        f"import {module_name}; "
        "assert 'ui_qt.main_window' not in sys.modules, "
        "'ui_qt.main_window was imported transitively'"
    )
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{module_name} imports ui_qt.main_window transitively:\n{result.stderr}"
        )


class CommonSymbolsTests(unittest.TestCase):
    def test_ui_metrics_contains_shell_density_keys(self) -> None:
        self.assertEqual(UI_METRICS["left_width"], 380)
        self.assertLess(UI_METRICS["sidebar_min_width"], UI_METRICS["left_width"])
        self.assertGreater(UI_METRICS["sidebar_max_width"], UI_METRICS["left_width"])
        for key in (
            "sidebar_min_width",
            "sidebar_max_width",
            "main_margin",
            "group_margin",
            "row_spacing",
            "label_col_width",
            "input_h",
            "spin_w",
            "short_combo_w",
            "deriv_combo_w",
            "tool_h",
            "tool_w",
        ):
            self.assertIn(key, UI_METRICS)

    def test_wheel_ignoring_widget_subclasses_import(self) -> None:
        self.assertTrue(issubclass(QDoubleSpinBox, object))
        self.assertTrue(issubclass(QSpinBox, object))
        self.assertTrue(issubclass(QComboBox, object))

    def test_state_dataclasses_construct(self) -> None:
        loaded = LoadedState(mode="PL", folder="")
        self.assertEqual(loaded.mode, "PL")
        self.assertEqual(loaded.drr_baseline_text, "Self (last frame)")

        options = LoadOptions(
            mode="PL",
            folder="",
            selected_files=[],
            baseline_files=[],
            pl_log_scale=False,
            drr_baseline_text="Self (last frame)",
            drr_baseline_which="last",
            compare_log_scale=False,
        )
        self.assertEqual(options.mode, "PL")
        self.assertEqual(options.mcd_candidate_width_mev, 5.0)

        export = ExportOptions(mode="PL", params=None)
        self.assertTrue(export.mcd_show_signed_mean)
        self.assertFalse(export.cleanup_verified_sources)

    def test_wrapped_filename_delegate_is_styled_item_delegate(self) -> None:
        self.assertTrue(issubclass(WrappedFilenameDelegate, QStyledItemDelegate))

    def test_feature_pages_imports_without_main_window(self) -> None:
        _assert_import_isolated("ui_qt.feature_pages")

    def test_controllers_import_without_main_window(self) -> None:
        for module in (
            "ui_qt.controllers_pl",
            "ui_qt.controllers_drr",
            "ui_qt.controllers_mcd",
            "ui_qt.controllers_compare",
            "ui_qt.controllers_power",
            "ui_qt.controllers_shg",
        ):
            with self.subTest(module=module):
                _assert_import_isolated(module)

    def test_main_window_imports_pages_and_controllers_at_top(self) -> None:
        code = (
            "import sys; "
            "import ui_qt.main_window; "
            "assert 'ui_qt.feature_pages' in sys.modules, 'feature_pages not imported by main_window'; "
            "assert 'ui_qt.controllers_mcd' in sys.modules, 'controllers not imported by main_window'; "
            "assert 'ui_qt.common' in sys.modules, 'common not imported by main_window'; "
            "print('ok')"
        )
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            env=env,
        )
        if result.returncode != 0:
            raise AssertionError(f"main_window import graph broken:\n{result.stderr}")


if __name__ == "__main__":
    unittest.main()
