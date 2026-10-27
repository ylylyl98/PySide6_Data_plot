"""Keep-background intent survives failed or superseded native group loads."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickKeepBackgroundTests(unittest.TestCase):
    def test_native_keep_background_group_loads(self):
        with TemporaryDirectory(prefix="quick-keep-background-") as temporary:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen",
                QT_QUICK_BACKEND="software", MPLBACKEND="agg", MPLCONFIGDIR=temporary,
                LOCALAPPDATA=temporary, XDG_CONFIG_HOME=temporary,
                XDG_CACHE_HOME=temporary, PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from threading import Event
    import time
    from unittest.mock import patch

    import numpy as np
    from PySide6.QtCore import QSettings, QUrl
    from PySide6.QtGui import QGuiApplication
    from core.data_io import load_drr_external_cube
    from core.drr_bn_store import DrrBnStore, source_records
    from tests.bn_joint_fixtures import accepted_joint_report
    from ui_qt.quick.application_bridge import QuickApplicationBridge
    from ui_qt.quick.drr_background_workflow import prepare_fixed_application

    application = QGuiApplication([])

    def until(predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            application.processEvents()
            if predicate():
                application.processEvents()
                return
            time.sleep(.002)
        raise AssertionError("Native keep-background test timed out")

    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary = TemporaryDirectory(prefix="quick-keep-background-case-")
            self.addCleanup(self.temporary.cleanup)
            self.root = Path(self.temporary.name)
            settings = QSettings(str(self.root / "isolated.ini"), QSettings.Format.IniFormat)
            self.bridge = QuickApplicationBridge(settings=settings)
            self.addCleanup(self.bridge.close)
            # These regressions concern source loading and numeric background
            # division; rendering has its own native integration coverage.
            plot_patch = patch.object(self.bridge, "_schedule_plot")
            plot_patch.start()
            self.addCleanup(plot_patch.stop)
            self.bridge.selectWorkflow("drr")
            self.original = self.write("sample_TG_rep01.csv")
            self.reference = self.write("reference.csv", offset=40, constant=True)
            self.next_group = self.write("sample_TG_rep02.csv", offset=7)
            self.last_group = self.write("sample_TG_rep03.csv", offset=11)
            self.select(self.original)
            self.idle()
            self.bridge.openDrrBackground()
            background = self.bridge.drrBackgroundBridge
            until(lambda: not background.snapshot["busy"])
            background.setChoice("kind", 6)
            self.bridge.acceptDrrBackgroundFiles(self.bridge.backgroundDialogToken,
                "background", [QUrl.fromLocalFile(str(self.reference))])
            background.setFlag("fixed", True)
            background.invoke("apply")
            until(lambda: not background.snapshot["busy"] and not self.bridge._pending)
            self.assertEqual(background.snapshot["error"], "")
            self.assertTrue(self.bridge._drr_background_recipe.fixed)

        def write(self, name, *, offset=0, constant=False, points=41):
            wave, gate = np.linspace(630., 690., points), np.linspace(-2., 2., 9)
            shape = np.linspace(-1., 1., len(wave))
            values = 130 + offset + 4 * shape[None, :] ** 2 + np.zeros((len(gate), points))
            if not constant:
                values += 15 * gate[:, None] + 3 * gate[:, None] * shape[None, :]
            path = self.root / name
            np.savetxt(path, np.column_stack((np.zeros_like(gate) if constant else gate*.5,
                np.zeros_like(gate) if constant else gate, values)), delimiter=",",
                header="Vbg,Vtg," + ",".join(map(str, wave)), comments="")
            return path

        def select(self, path):
            self.bridge.selectFiles([QUrl.fromLocalFile(str(path))])

        def idle(self):
            until(lambda: not self.bridge._pending and not self.bridge.snapshot["busy"])

        def assert_external_group(self, path):
            self.assertTrue(self.bridge.snapshot["canExport"], self.bridge.snapshot["status"])
            products = self.bridge._drr_products
            self.assertEqual(products.request.files, (str(path.resolve()),))
            self.assertEqual(products.request.background_mode, "external")
            self.assertTrue(products.request.background_recipe.fixed)
            self.assertEqual(tuple(item.measurement_file for item in
                products.request.background_recipe.assignments), (path.name,))
            expected = load_drr_external_cube(str(self.root), (str(path),),
                (str(self.reference),), baseline_which="all", y_axis=products.request.y_axis)
            np.testing.assert_array_equal(products.raw.Z, expected.Z)

        def test_applied_keep_background_follows_a_new_group(self):
            self.select(self.next_group)
            self.idle()
            self.assert_external_group(self.next_group)

        def test_incompatible_group_does_not_erase_keep_background(self):
            incompatible = self.write("incompatible_TG.csv", points=39)
            self.select(incompatible)
            self.idle()
            self.assertIn("Background was not changed", self.bridge.snapshot["status"])
            self.select(self.next_group)
            self.idle()
            self.assert_external_group(self.next_group)

        def test_reload_retries_the_fixed_reference_after_a_failed_group_load(self):
            incompatible = self.write("incompatible_TG.csv", points=39)
            self.select(incompatible)
            self.idle()
            self.assertIn("Background was not changed", self.bridge.snapshot["status"])
            self.write(incompatible.name, points=41)
            self.bridge.invoke("reload")
            self.idle()
            self.assert_external_group(incompatible)

        def test_invalid_fixed_sg_window_reports_error_without_gui_exception(self):
            before = self.bridge._drr_products
            self.bridge.setParameter("sgWindow", 22)
            self.idle()
            self.assertIs(self.bridge._drr_products, before)
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.assertIn("odd", self.bridge.snapshot["status"])

        def test_failed_fixed_sg_revalidation_disables_stale_export(self):
            before = self.bridge._drr_products
            with patch("ui_qt.quick.drr_background_workflow.prepare_fixed_application",
                       side_effect=ValueError("synthetic fixed rebind failure")):
                self.bridge.setParameter("sgWindow", 23)
                self.idle()
            self.assertIs(self.bridge._drr_products, before)
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.assertIn("synthetic fixed rebind failure", self.bridge.snapshot["status"])

        def test_superseding_a_pending_group_keeps_the_reference(self):
            started, release = Event(), Event()

            def delayed_fixed(recipe, context):
                if context.measurements == (self.next_group.name,):
                    started.set()
                    if not release.wait(15):
                        raise AssertionError("Pending group test failed to release worker")
                return prepare_fixed_application(recipe, context)

            with patch("ui_qt.quick.drr_background_workflow.prepare_fixed_application",
                       side_effect=delayed_fixed):
                try:
                    self.select(self.next_group)
                    until(started.is_set)
                    self.select(self.last_group)
                finally:
                    release.set()
                self.idle()
            self.assert_external_group(self.last_group)

        def test_accepted_automatic_bn_review_preserves_the_fixed_reference(self):
            self.reference = self.write("chosen_BNbackground.csv", offset=40, constant=True)
            self.bridge.openDrrBackground()
            background = self.bridge.drrBackgroundBridge
            until(lambda: not background.snapshot["busy"])
            background.setChoice("kind", 0)
            self.bridge.acceptDrrBackgroundFiles(self.bridge.backgroundDialogToken,
                "background", [QUrl.fromLocalFile(str(self.reference))])
            background.setFlag("fixed", True)
            background.setFlag("auto_bn_compare", True)

            def accepted_comparison(*_args, **_kwargs):
                # Numerical acceptance is covered by the engine tests. This
                # existing transport fixture exercises a certified candidate
                # arriving at the owner after the user pins a measured BN.
                request = self.bridge._drr_request()
                context = self.bridge._background_context()
                columns = np.linspace(630., 690., 41)
                energy = np.sort(1240 / columns)
                baseline = 170 + 4 * np.linspace(-1., 1., len(energy)) ** 2
                candidate = baseline + .25
                identity = dict(schema_version=1, root=str(self.root),
                    measurements=source_records(self.root, context.measurements),
                    baseline_files=source_records(self.root, (self.reference.name,)),
                    baseline_which="all", spectral_columns=columns.tolist(),
                    y_axis=request.y_axis, repeat_order=list(context.measurements), ordering={})
                return dict(source_identity=identity,
                    report=accepted_joint_report(request.window, request.polyorder),
                    settings=dict(derivative_window=request.window, derivative_polyorder=request.polyorder),
                    energy=energy, gates=np.linspace(-2., 2., 9),
                    original_background=baseline, candidate_background=candidate,
                    candidate_backgrounds={"original": baseline, "local_shift": candidate},
                    mean_raw=np.full((9, len(energy)), 130.),
                    input_snapshots={"measurement_0": np.full((9, len(energy)), 130.)},
                    eligible=True, status="accepted", selected_candidate="local_shift",
                    summary="Transport fixture", reasons=[], candidates=[])

            def review_finished():
                return (not background.snapshot["busy"] and not self.bridge._pending
                    and not self.bridge.snapshot["busy"] and not self.bridge.snapshot["bnBusy"])

            with patch("core.drr_bn_service.compare_bn_group", side_effect=accepted_comparison) as compare:
                background.invoke("apply")
                until(review_finished)
                self.assertEqual(background.snapshot["error"], "")
                self.assertEqual(compare.call_count, 1)
                self.assert_external_group(self.original)
                self.assertIsNone(DrrBnStore(self.root).find_applied((self.original.name,)))
                self.select(self.next_group)
                until(review_finished)
            self.assert_external_group(self.next_group)
            self.assertIsNone(DrrBnStore(self.root).find_applied((self.next_group.name,)))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Cases)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_isolated())
    unittest.main()
