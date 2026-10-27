"""Load original CSVs through the native picker after real DRR exports."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickBackgroundRestoreIntegrationTests(unittest.TestCase):
    def test_processed_original_csv_background_restore(self):
        with TemporaryDirectory(prefix="quick-background-restore-") as temporary:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen",
                QT_QUICK_BACKEND="software", MPLBACKEND="agg", MPLCONFIGDIR=temporary,
                LOCALAPPDATA=temporary, XDG_CONFIG_HOME=temporary,
                XDG_CACHE_HOME=temporary, PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from concurrent.futures import Future
    from threading import Event
    import time
    from unittest.mock import patch

    import numpy as np
    from PySide6.QtCore import QSettings, QUrl
    from PySide6.QtGui import QGuiApplication
    from core.data_io import load_drr_external_cube, load_drr_self_cube
    from ui_qt.quick.application_bridge import QuickApplicationBridge
    from ui_qt.quick.drr_background_restore import restore_saved_background
    from ui_qt.quick.drr_workflow import export_drr
    from tests.quick_drr_helpers import apply_self_background

    application = QGuiApplication([])

    def until(predicate, timeout=25):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            application.processEvents()
            if predicate():
                application.processEvents()
                return
            time.sleep(.003)
        raise AssertionError("Native saved-background integration timed out")

    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary = TemporaryDirectory(prefix="quick-background-restore-case-")
            self.addCleanup(self.temporary.cleanup)
            self.root = Path(self.temporary.name).resolve()
            self.source = self.write("sample_REF_730nmc_TG_rep01.csv")
            self.other = self.write("another_REF_740nmc_TG_rep01.csv", offset=7)
            self.bn = self.write("chosen_BNbackground.csv", offset=40, constant=True)
            self.custom = self.write("custom_reference.csv", offset=80, constant=True)
            self.bridge_count = 0

        def write(self, name, *, offset=0, constant=False):
            wave, gate = np.linspace(630., 690., 41), np.linspace(-2., 2., 9)
            shape = np.linspace(-1., 1., len(wave))
            values = 130 + offset + 4 * shape[None, :] ** 2 + np.zeros((len(gate), len(wave)))
            if not constant:
                values += 15 * gate[:, None] + 3 * gate[:, None] * shape[None, :]
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(path, np.column_stack((np.zeros_like(gate) if constant else gate*.5,
                np.zeros_like(gate) if constant else gate, values)), delimiter=",",
                header="Vbg,Vtg," + ",".join(map(str, wave)), comments="")
            return path

        def bridge(self, *, select_folder=True):
            self.bridge_count += 1
            settings = QSettings(str(self.root / f"profile-{self.bridge_count}.ini"), QSettings.Format.IniFormat)
            bridge = QuickApplicationBridge(settings=settings)
            self.addCleanup(bridge.close)
            # Keep real loaders, exports, history and picker inspection. Plot
            # scheduling is unrelated to adopting a validated background.
            plot_patch = patch.object(bridge, "_schedule_plot")
            plot_patch.start()
            self.addCleanup(plot_patch.stop)
            bridge.selectWorkflow("drr")
            if select_folder:
                bridge.setSourcePath(str(self.root))
            return bridge

        def idle(self, bridge):
            until(lambda: not bridge._pending and not bridge.snapshot["busy"]
                and not bridge.snapshot["bnBusy"])

        def select(self, bridge, path):
            bridge.selectFiles([QUrl.fromLocalFile(str(path))])
            self.idle(bridge)
            self.assertTrue(bridge.snapshot["canExport"] or bridge.snapshot["backgroundRequired"],
                            bridge.snapshot["status"])

        def apply(self, bridge, reference, *, kind=6, fixed=False):
            bridge.openDrrBackground()
            background = bridge.drrBackgroundBridge
            until(lambda: not background.snapshot["busy"])
            background.setChoice("kind", kind)
            bridge.acceptDrrBackgroundFiles(bridge.backgroundDialogToken,
                "background", [QUrl.fromLocalFile(str(reference))])
            background.setFlag("fixed", fixed)
            background.invoke("apply")
            until(lambda: not background.snapshot["busy"] and not bridge._pending)
            self.assertEqual(background.snapshot["error"], "")
            self.assertTrue(bridge.snapshot["canExport"], bridge.snapshot["status"])

        def save(self, reference, *, kind=6):
            bridge = self.bridge()
            self.select(bridge, self.source)
            self.apply(bridge, reference, kind=kind)
            products = bridge._drr_products
            expected = products.raw.Z.copy()
            outputs = export_drr(products, folder=str(self.root),
                raw_params=bridge.drrDisplayBridge.heatmap_params(products.raw),
                second_params=None, view="raw")
            self.assertTrue(outputs)
            self.assertTrue(all(Path(path).is_file() for path in outputs.values()))
            self.assertTrue(any(Path(path).suffix == ".png" for path in outputs.values()))
            self.assertTrue(any(Path(path).suffix == ".dat" for path in outputs.values()))
            self.assertTrue(list((self.root / "Processed Data" / "DRR").glob("*.metadata.json")))
            bridge.close()
            return expected

        def load_processed_csv(self, bridge):
            # This is the picker acceptance path used by Load selected for the
            # original CSV, whose Processed badge comes from the real export.
            catalog = bridge.sourceCatalogBridge
            catalog.setParameter("onlyNew", False)
            catalog.open("drr", str(self.root), [str(self.source)], False)
            until(lambda: not catalog.snapshot["busy"])
            self.assertEqual(catalog.snapshot["error"], "")
            row = next(row for row in catalog._catalog["files"] if row["id"] == str(self.source))
            self.assertFalse(row["saved"])
            self.assertEqual(row["status"], "processed")
            self.assertTrue(catalog.snapshot["canAccept"])
            catalog.invoke("accept")
            self.idle(bridge)

        def assert_external(self, bridge, reference, expected, *, fixed=False):
            self.assertTrue(bridge.snapshot["canExport"], bridge.snapshot["status"])
            request = bridge._drr_products.request
            self.assertEqual(request.files, (str(self.source),))
            self.assertEqual(request.background_mode, "external")
            self.assertIsNotNone(request.background_recipe)
            self.assertEqual(request.background_recipe.files, (reference.name,))
            self.assertEqual(request.background_recipe.fixed, fixed)
            self.assertEqual(bridge._drr_background_recipe, request.background_recipe)
            self.assertEqual(bridge._drr_processing["backgroundMode"], "external")
            self.assertEqual(bridge._drr_backgrounds, (reference.name,))
            np.testing.assert_array_equal(bridge._drr_products.raw.Z, expected)

        def test_load_csv_restores_exported_bn_recipe_and_numbers(self):
            expected = self.save(self.bn, kind=0)
            bridge = self.bridge()
            self.load_processed_csv(bridge)
            self.assert_external(bridge, self.bn, expected)
            self.assertEqual(bridge._drr_background_recipe.draft.kind, "bn")

        def test_load_csv_restores_exported_custom_recipe_and_numbers(self):
            expected = self.save(self.custom)
            bridge = self.bridge()
            self.load_processed_csv(bridge)
            self.assert_external(bridge, self.custom, expected)
            self.assertEqual(bridge._drr_background_recipe.draft.kind, "custom")

        def test_direct_nested_csv_open_finds_the_experiment_history(self):
            self.source = self.write("Initial Data/REF/nested_REF_750nmc_TG.csv", offset=3)
            expected = self.save(self.bn, kind=0)
            bridge = self.bridge(select_folder=False)
            bridge.openFile(QUrl.fromLocalFile(str(self.source)))
            self.idle(bridge)
            self.assertEqual(bridge._folder, str(self.root))
            self.assert_external(bridge, self.bn, expected)

        def test_explicit_fixed_reference_wins_over_saved_group_history(self):
            self.save(self.bn, kind=0)
            bridge = self.bridge()
            self.select(bridge, self.other)
            self.apply(bridge, self.custom, fixed=True)
            expected = load_drr_external_cube(str(self.root), (str(self.source),),
                (str(self.custom),), baseline_which="all", y_axis=bridge._drr_request().y_axis)
            with patch("ui_qt.quick.drr_background_restore.restore_saved_background",
                       side_effect=AssertionError("Explicit Keep background must take precedence.")):
                self.load_processed_csv(bridge)
            self.assert_external(bridge, self.custom, expected.Z, fixed=True)

        def test_missing_latest_reference_blocks_self_and_older_saved_fallback(self):
            self.save(self.bn, kind=0)
            self.save(self.custom)
            self.custom.unlink()
            bridge = self.bridge()
            self.load_processed_csv(bridge)
            self.assertFalse(bridge.snapshot["canExport"])
            self.assertIsNone(bridge._drr_products)
            self.assertIsNone(bridge._cube)
            self.assertIn("background", bridge.snapshot["status"].casefold())
            self.assertIn("restore", bridge.snapshot["status"].casefold())
            self.assertTrue(bridge.snapshot["backgroundRequired"])
            self.assertTrue(bridge._background_open)
            self.assertIn("restore", bridge.drrBackgroundBridge.snapshot["error"].casefold())
            self.assertFalse(bridge.drrBackgroundBridge.snapshot["busy"])
            self.assertTrue(bridge.drrBackgroundBridge.snapshot["applyEnabled"])

        def test_reload_retries_failed_saved_history_until_reference_is_restored(self):
            self.save(self.bn, kind=0)
            expected = self.save(self.custom)
            self.custom.unlink()
            bridge = self.bridge()
            self.load_processed_csv(bridge)
            self.assertFalse(bridge.snapshot["canExport"])
            bridge.cancelDrrBackground()
            opened = []
            bridge.backgroundDialogRequested.connect(lambda: opened.append(True))
            bridge.invoke("reload")
            self.idle(bridge)
            self.assertFalse(bridge.snapshot["canExport"])
            self.assertIsNone(bridge._drr_products)
            self.assertIn("restore", bridge.snapshot["status"].casefold())
            self.write(self.custom.name, offset=80, constant=True)
            bridge.invoke("reload")
            self.idle(bridge)
            self.assert_external(bridge, self.custom, expected)
            self.assertEqual(opened, [], "Reload must not automatically reopen background settings")

        def test_manual_self_after_failed_history_is_retained_on_reload(self):
            self.save(self.bn, kind=0)
            self.bn.unlink()
            bridge = self.bridge()
            self.load_processed_csv(bridge)
            self.assertFalse(bridge.snapshot["canExport"])
            bridge.openDrrBackground()
            background = bridge.drrBackgroundBridge
            until(lambda: not background.snapshot["busy"])
            background.setChoice("kind", 4)
            background.invoke("apply")
            until(lambda: not background.snapshot["busy"] and not bridge._pending)
            self.assertEqual(background.snapshot["error"], "")
            self.assertTrue(bridge.snapshot["canExport"], bridge.snapshot["status"])
            self.assertEqual(bridge._drr_products.request.background_mode, "self-last")
            with patch("ui_qt.quick.drr_background_restore.restore_saved_background",
                       side_effect=AssertionError("Successful explicit Self must replace pending history.")):
                bridge.invoke("reload")
                self.idle(bridge)
            self.assertTrue(bridge.snapshot["canExport"], bridge.snapshot["status"])
            self.assertEqual(bridge._drr_products.request.background_mode, "self-last")
            expected = load_drr_self_cube(str(self.root), (str(self.source),),
                use_first_frame=False, y_axis=bridge._drr_products.request.y_axis)
            np.testing.assert_array_equal(bridge._drr_products.raw.Z, expected.Z)

        def test_superseded_history_restore_cannot_adopt_a_late_recipe(self):
            self.save(self.bn, kind=0)
            bridge = self.bridge()
            started, release = Event(), Event()

            def delayed(request, **kwargs):
                restored = restore_saved_background(request, **kwargs)
                if request.files == (str(self.source),):
                    started.set()
                    if not release.wait(20):
                        raise AssertionError("History test did not release the old worker")
                return restored

            with patch("ui_qt.quick.drr_background_restore.restore_saved_background", side_effect=delayed):
                try:
                    bridge.selectFiles([QUrl.fromLocalFile(str(self.source))])
                    until(started.is_set)
                    bridge.selectFiles([QUrl.fromLocalFile(str(self.other))])
                    until(lambda: bridge.snapshot["backgroundRequired"] and not bridge.snapshot["busy"])
                finally:
                    release.set()
                self.idle(bridge)
            self.assertIsNone(bridge._drr_products)
            self.assertIsNone(bridge._drr_background_recipe)
            self.assertEqual(bridge._sources, (str(self.other),))
            self.assertEqual(bridge.drrBackgroundBridge.context.measurements, (self.other.name,))
            products = apply_self_background(bridge, until)
            self.assertEqual(products.request.files, (str(self.other),))
            self.assertEqual(products.request.background_mode, "self-last")
            expected = load_drr_self_cube(str(self.root), (str(self.other),),
                use_first_frame=False, y_axis=products.request.y_axis)
            np.testing.assert_array_equal(products.raw.Z, expected.Z)

        def test_new_csv_waits_for_explicit_background_and_opens_once(self):
            from ui_qt.quick import drr_workflow
            bridge = self.bridge()
            opened = []
            bridge.backgroundDialogRequested.connect(lambda: opened.append(True))
            with patch.object(drr_workflow, "load_drr", wraps=drr_workflow.load_drr) as load, \
                    patch.object(drr_workflow, "process_drr", wraps=drr_workflow.process_drr) as process:
                self.select(bridge, self.source)
                self.assertEqual(opened, [True])
                self.assertTrue(bridge._background_open)
                self.assertTrue(bridge.snapshot["backgroundRequired"])
                self.assertFalse(bridge.snapshot["canExport"])
                self.assertFalse(bridge.snapshot["canAnalyzePeaks"])
                self.assertFalse(bridge.snapshot["derivativeAvailable"])
                self.assertIsNone(bridge._cube)
                self.assertIsNone(bridge._drr_products)
                background = bridge.drrBackgroundBridge
                self.assertEqual(background.draft.which, "last")
                background.setChoice("frame", 0)
                background.setChoice("frame", 1)
                load.assert_not_called()
                process.assert_not_called()
                bridge.cancelDrrBackground()
                for action in ("reload", "workflow-roundtrip"):
                    with self.subTest(action=action):
                        if action == "reload":
                            bridge.invoke(action)
                        else:
                            bridge.selectWorkflow("pl")
                            bridge.selectWorkflow("drr")
                        self.idle(bridge)
                        self.assertTrue(bridge.snapshot["backgroundRequired"])
                        self.assertFalse(bridge.snapshot["canExport"])
                        self.assertFalse(bridge._background_open)
                        self.assertEqual(opened, [True])
                        load.assert_not_called()
                        process.assert_not_called()
                products = apply_self_background(bridge, until, frame=0)
                self.assertEqual(load.call_count, 1)
                self.assertEqual(process.call_count, 1)
                self.assertFalse(bridge.snapshot["backgroundRequired"])
                self.assertEqual(products.request.background_mode, "self-first")
                expected = load_drr_self_cube(str(self.root), (str(self.source),),
                    use_first_frame=True, y_axis=products.request.y_axis)
                np.testing.assert_array_equal(products.raw.Z, expected.Z)
                dialog_count = len(opened)
                bridge.invoke("clear-background")
                self.idle(bridge)
                self.assertTrue(bridge.snapshot["backgroundRequired"])
                self.assertIsNone(bridge._drr_products)
                self.assertIsNone(bridge._cube)
                self.assertFalse(bridge.snapshot["canExport"])
                self.assertEqual(len(opened), dialog_count)
                bridge.invoke("reload")
                self.idle(bridge)
                self.assertEqual(load.call_count, 1)
                self.assertEqual(process.call_count, 1)
                products = apply_self_background(bridge, until, frame=1)
                self.assertEqual(load.call_count, 2)
                self.assertEqual(process.call_count, 2)
                self.assertEqual(products.request.background_mode, "self-last")
                expected = load_drr_self_cube(str(self.root), (str(self.source),),
                    use_first_frame=False, y_axis=products.request.y_axis)
                np.testing.assert_array_equal(products.raw.Z, expected.Z)
                # A fast cache hit can finish before add_done_callback is
                # registered. Delivery must still wait for owner bookkeeping.
                def completed(operation):
                    future = Future()
                    try:
                        future.set_result(operation())
                    except Exception as exc:
                        future.set_exception(exc)
                    return future
                with self.subTest(completion="already-completed-future"), \
                        patch.object(bridge._executor, "submit", side_effect=completed):
                    dialog_count = len(opened)
                    self.select(bridge, self.other)
                    self.assertTrue(bridge.snapshot["backgroundRequired"])
                    self.assertTrue(bridge._background_open)
                    self.assertEqual(len(opened), dialog_count + 1)
                    self.assertEqual(load.call_count, 2)
                    products = apply_self_background(bridge, until)
                    self.assertEqual(load.call_count, 3)
                    self.assertEqual(process.call_count, 3)
                    self.assertIs(products.request.background_recipe, bridge._drr_background_recipe)
                    self.assertFalse(bridge.snapshot["backgroundRequired"])
                    self.assertEqual(products.request.files, (str(self.other),))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    print("QUICK_BACKGROUND_RESTORE_INTEGRATION_CASES=" + str(result.testsRun))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_isolated())
    unittest.main()
