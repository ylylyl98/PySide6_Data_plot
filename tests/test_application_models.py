"""Shared records and workflow metadata must remain usable without Widgets."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, MISSING, fields
import os
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap
import unittest

from core.application_models import ExportOptions, LoadOptions, LoadedState


ROOT = Path(__file__).resolve().parents[1]


class ApplicationModelTests(unittest.TestCase):
    def _fresh_process(self, body: str) -> None:
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["PYTHONIOENCODING"] = "utf-8"
        script = textwrap.dedent("""
            import importlib.abc
            import sys

            class NoWidgets(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == 'PySide6.QtWidgets' or fullname == 'PySide6.QtQuickWidgets':
                        raise AssertionError('Widgets imported: ' + fullname)
                    if fullname in ('ui_qt.common', 'ui_qt.main_window'):
                        raise AssertionError('Legacy UI imported: ' + fullname)

            sys.meta_path.insert(0, NoWidgets())
        """) + textwrap.dedent(body)
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_records_import_without_loading_qt_or_ui(self):
        self._fresh_process("""
            from core.application_models import LoadedState, LoadOptions, ExportOptions
            state = LoadedState(mode='DRR', folder='')
            assert state.drr_baseline_which == 'last'
            assert ExportOptions(mode='DRR', params=None).drr_second_auto_scale
            assert not any(name.startswith('PySide6') for name in sys.modules)
            assert not any(name.startswith('ui_qt') for name in sys.modules)
        """)

    def test_registry_metadata_and_owner_builder_do_not_load_qt(self):
        self._fresh_process("""
            from ui_qt.feature_registry import FEATURES, FeatureSpec
            assert tuple(item.key for item in FEATURES) == (
                'pl', 'drr', 'cmp', 'power', 'mcd', 'mcd_peak_shift', 'shg', 'slides', 'tools')
            marker = object()
            class Owner:
                def make_page(self):
                    return marker
            assert FeatureSpec('test', 'Test', '', 'make_page').build(Owner()) is marker
            assert not any(name.startswith('PySide6') for name in sys.modules)
        """)

    def test_native_gui_application_can_use_records_and_registry(self):
        self._fresh_process("""
            from PySide6.QtGui import QGuiApplication
            app = QGuiApplication([])
            from core.application_models import LoadedState, ExportOptions
            from ui_qt.feature_registry import FEATURES
            assert LoadedState(mode='PL', folder='').mode == 'PL'
            assert ExportOptions(mode='PL', params=None).mcd_show_signed_mean
            assert len(FEATURES) == 9
            app.processEvents()
            assert 'PySide6.QtWidgets' not in sys.modules
            assert 'PySide6.QtQuickWidgets' not in sys.modules
        """)

    def test_legacy_common_reexports_the_same_classes(self):
        from ui_qt import common

        for record in (LoadedState, LoadOptions, ExportOptions):
            with self.subTest(record=record.__name__):
                self.assertIs(getattr(common, record.__name__), record)
                legacy_reference = f"cui_qt.common\n{record.__name__}\n.".encode("ascii")
                self.assertIs(pickle.loads(legacy_reference), record)

    def test_mutable_default_containers_are_not_shared(self):
        for record, args in (
            (LoadedState, dict(mode="DRR", folder="")),
            (LoadOptions, dict(mode="DRR", folder="", selected_files=[], baseline_files=[],
                               pl_log_scale=False, drr_baseline_text="Self (last frame)",
                               drr_baseline_which="last", compare_log_scale=False)),
        ):
            first, second = record(**args), record(**args)
            for item in fields(record):
                if item.default_factory is not MISSING:
                    with self.subTest(record=record.__name__, field=item.name):
                        self.assertIsNot(getattr(first, item.name), getattr(second, item.name))

    def test_request_records_keep_frozen_and_shallow_semantics(self):
        names = ["measurement.csv"]
        options = LoadOptions("DRR", "", names, [], False, "Self (last frame)", "last", False)
        self.assertIs(options.selected_files, names)
        names.append("second.csv")
        self.assertEqual(options.selected_files, ["measurement.csv", "second.csv"])
        with self.assertRaises(FrozenInstanceError):
            options.mode = "PL"
        export = ExportOptions(mode="DRR", params=None)
        with self.assertRaises(FrozenInstanceError):
            export.mode = "PL"
        loaded = LoadedState(mode="DRR", folder="")
        loaded.mode = "PL"
        self.assertEqual(loaded.mode, "PL")

    def test_export_options_preserve_product_and_analysis_snapshots(self):
        raw, second, kk, kkp, vp = (object() for _ in range(5))
        display = {"regions": 3}
        peak_result = {"centers": [1.8, 2.0]}
        options = ExportOptions(
            mode="DRR", params=None, drr_raw_cube=raw, drr_second_cube=second,
            drr_display_state=display, drr_peak_result=peak_result,
            power_kk_cube=kk, power_kkp_cube=kkp, power_vp_cube=vp,
            drr_second_scale_xlim=(1.7, 2.5), drr_second_split_auto_keys=("left_vmin",),
        )
        for field, value in (("drr_raw_cube", raw), ("drr_second_cube", second),
                             ("drr_display_state", display), ("drr_peak_result", peak_result),
                             ("power_kk_cube", kk), ("power_kkp_cube", kkp), ("power_vp_cube", vp)):
            with self.subTest(field=field):
                self.assertIs(getattr(options, field), value)
        self.assertEqual(options.drr_second_scale_xlim, (1.7, 2.5))
        self.assertEqual(options.drr_second_split_auto_keys, ("left_vmin",))
        self.assertTrue(options.mcd_show_signed_mean)
        self.assertFalse(options.cleanup_verified_sources)


if __name__ == "__main__":
    unittest.main()
