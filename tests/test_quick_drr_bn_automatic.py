"""Native BN scheduling with real store/trial loads and isolated QGuiApplication.

Accepted reports are transport fixtures, not claims of numerical acceptance.
The independent BN scientific suites verify the fitting/certificate algorithm.
"""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class NativeAutomaticBnTests(unittest.TestCase):
    def test_automatic_bn_in_widgets_free_process(self):
        with TemporaryDirectory(prefix="native-bn-process-") as temporary:
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software", MPLBACKEND="agg",
                MPLCONFIGDIR=temporary, LOCALAPPDATA=temporary, XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import csv
    from copy import deepcopy
    from dataclasses import replace
    import importlib.abc
    import json
    from threading import Event
    import time
    from unittest.mock import patch

    class RejectWidgets(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {"PySide6.QtWidgets", "PySide6.QtQuickWidgets", "ui_qt.common", "ui_qt.main_window",
                "matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_qt"}:
                raise AssertionError("Native BN imported Widgets: "+fullname)
    sys.meta_path.insert(0, RejectWidgets())
    from PySide6.QtCore import QCoreApplication, QEvent, QObject, QSettings, Qt, QUrl
    from PySide6.QtQuickControls2 import QQuickStyle
    from PySide6.QtGui import QGuiApplication
    import numpy as np
    from core.drr_bn_store import DrrBnStore, source_records
    from core.load_cancellation import LoadCancellation
    from tests.bn_joint_fixtures import accepted_joint_report
    from ui_qt.quick.application_bridge import QuickApplicationBridge
    from ui_qt.quick.drr_background_workflow import BackgroundDraft, prepare_application, commit_application
    from ui_qt.quick.drr_bn_automatic import check_background
    from ui_qt.quick.drr_workflow import DrrRequest, background_context, load_drr

    QQuickStyle.setStyle("Basic")
    app = QGuiApplication([])
    def wait(predicate, timeout=30):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                app.processEvents()
                return
            time.sleep(.003)
        raise AssertionError("Native BN task did not finish")

    class Cases(unittest.TestCase):
        def setUp(self):
            temporary = TemporaryDirectory(prefix="native-bn-case-")
            self.addCleanup(temporary.cleanup)
            self.root = Path(temporary.name)
            self.energy = np.linspace(1.8, 2.04, 121)
            self.columns = np.round(1240/self.energy, 9)
            self.energy = 1240/self.columns
            self.files = tuple(f"sample_TG_rep{i:02d}.csv" for i in range(4))
            self.background = "reference_BNbackground.csv"
            for index, name in enumerate((*self.files, self.background)):
                with (self.root/name).open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["Vbg", "Vtg", *self.columns])
                    for gate in range(9):
                        coordinate = 0 if name == self.background else gate
                        writer.writerow([coordinate, coordinate, *(np.full(len(self.energy), 110.+index*.1+gate))])
            self.store = DrrBnStore(self.root)
            self.settings = QSettings(str(self.root/"native.ini"), QSettings.IniFormat)
            self.request = DrrRequest(tuple(str(self.root/name) for name in self.files),
                backgrounds=(self.background,), background_mode="external", background_frame="all",
                experiment_folder=str(self.root), y_axis="tg", window=21, polyorder=2)

        def payload(self, window=21, *, eligible=True):
            baseline = np.full(len(self.energy), 100.)
            candidate = baseline.copy()
            candidate[40:70] += .25
            identity = dict(schema_version=1, root=str(self.root), measurements=source_records(self.root, self.files),
                baseline_files=source_records(self.root, (self.background,)), baseline_which="all",
                spectral_columns=self.columns.tolist(), y_axis="tg", repeat_order=list(self.files), ordering={})
            return dict(source_identity=identity, report=accepted_joint_report(window, 2),
                settings=dict(derivative_window=window, derivative_polyorder=2), energy=self.energy,
                gates=np.arange(9.), original_background=baseline, candidate_background=candidate,
                candidate_backgrounds={"original": baseline, "local_shift": candidate},
                mean_raw=np.full((9, len(self.energy)), 114.), input_snapshots={"measurement_0":np.full((9,len(self.energy)),114.)},
                eligible=eligible, status="accepted" if eligible else "exploratory", selected_candidate="local_shift",
                summary="Transport fixture", reasons=[], candidates=[])

        def managed(self, *, automatic=False):
            version = self.store.save_result(self.payload())
            payload = self.store.load_result(version)
            prepared = prepare_application(background_context(self.request), BackgroundDraft(kind="bn",
                files=(self.background,), which="all", auto_bn_compare=automatic,
                calibration=dict(payload=payload, original=False)))
            commit_application(prepared, current_context=prepared.context, settings=self.settings)
            return replace(self.request, backgrounds=prepared.recipe.files, background_frame=prepared.recipe.which,
                background_recipe=prepared.recipe), version

        def check(self, request=None, *, automatic=True, reload=False):
            return check_background(request or self.request, automatic=automatic, reload=reload, cancellation=LoadCancellation())

        def bridge(self):
            bridge = QuickApplicationBridge(settings=self.settings)
            self.addCleanup(bridge.close)
            bridge.selectWorkflow("drr")
            self.assertTrue(bridge.drrDisplayBridge.setChoice("yaxis.left", 3))
            bridge.selectFiles([QUrl.fromLocalFile(str(self.root/name)) for name in self.files])
            wait(lambda: not bridge._pending and not bridge.snapshot["busy"])
            self.assertTrue(bridge.snapshot["canExport"], bridge.snapshot["status"])
            return bridge

        def load_managed_bridge(self, *, automatic=False):
            request, version = self.managed(automatic=automatic)
            bridge = self.bridge()
            bridge._drr_background_recipe = request.background_recipe
            bridge._drr_backgrounds = request.backgrounds
            bridge._drr_processing.update(backgroundMode="external", backgroundFrame=request.background_frame)
            bridge._load_drr()
            wait(lambda: not bridge.snapshot["bnBusy"] and not bridge._pending)
            self.assertTrue(bridge.snapshot["canExport"], bridge.snapshot["status"])
            return bridge, version

        def test_real_comparison_rejection_retains_original_without_activation(self):
            outcome = self.check()
            self.assertIsNone(outcome.products)
            self.assertFalse(outcome.payload["eligible"])
            self.assertIsNone(self.store.find_applied(self.files))
            self.assertTrue(self.store.get(outcome.payload["version_id"]).csv_path.is_file())

        def test_accepted_trial_uses_exact_denominator_but_activation_is_separate(self):
            with patch("core.drr_bn_service.compare_bn_group", return_value=self.payload()):
                outcome = self.check()
            self.assertIsNone(self.store.find_applied(self.files))
            self.assertIsNotNone(outcome.prepared)
            from core.loader import load_drr_avg
            expected = load_drr_avg(str(self.root), self.files, bg_mode="external", y_axis="tg",
                external_vector=self.payload()["candidate_background"], external_energy=self.energy)
            np.testing.assert_array_equal(outcome.products.source.Z, expected.Z)
            recipe = commit_application(outcome.prepared, current_context=background_context(self.request), settings=self.settings)
            self.assertEqual(self.store.find_applied(self.files).version_id, outcome.payload["version_id"])
            self.assertEqual(recipe.mode, "External")

        def test_explicit_original_choice_blocks_automatic_candidate(self):
            self.store.deactivate(self.files, (self.background,), "all", restore_original=True)
            with patch("core.drr_bn_service.compare_bn_group", return_value=self.payload()):
                outcome = self.check()
            self.assertIsNone(outcome.prepared)
            self.assertIsNone(outcome.products)
            self.assertTrue(outcome.payload["original_chosen"])
            self.assertEqual(self.store.selection_state(self.files), "original")

        def test_current_managed_certificate_reloads_without_comparing(self):
            request, version = self.managed()
            with patch("core.drr_bn_service.compare_bn_group", side_effect=AssertionError("Unnecessary comparison")):
                outcome = self.check(request, automatic=False, reload=True)
            self.assertIsNotNone(outcome.products)
            self.assertEqual(outcome.payload["version_id"], version.version_id)
            self.assertFalse(outcome.withdraw_activation)

        def test_sg_change_falls_back_without_creating_an_explicit_original_choice(self):
            request, version = self.managed()
            changed = replace(request, window=23)
            with self.assertRaisesRegex(ValueError, "current joint"):
                load_drr(changed)
            outcome = self.check(changed, automatic=False, reload=True)
            self.assertTrue(outcome.withdraw_activation)
            self.assertEqual(self.store.find_applied(self.files).version_id, version.version_id)
            expected = load_drr(replace(self.request, window=23, backgrounds=(str(self.root/self.background),)))
            np.testing.assert_array_equal(outcome.products.raw.Z, expected.raw.Z)
            self.assertIsNone(outcome.prepared.recipe.draft.calibration)

        def test_sg_change_compares_with_new_settings_and_trial_loads_new_certificate(self):
            request, old_version = self.managed(automatic=True)
            with patch("core.drr_bn_service.compare_bn_group", return_value=self.payload(23)) as compare:
                outcome = self.check(replace(request, window=23), automatic=True, reload=True)
            self.assertEqual(compare.call_args.args[4]["derivative_window"], 23)
            self.assertEqual(outcome.products.request.window, 23)
            self.assertFalse(outcome.withdraw_activation)
            self.assertNotEqual(outcome.payload["version_id"], old_version.version_id)
            self.assertEqual(self.store.find_applied(self.files).version_id, old_version.version_id)

        def test_root_runs_comparison_after_load_and_applies_only_after_trial(self):
            bridge = self.bridge()
            self.settings.setValue("drr/bn_auto_compare", True)
            with patch("core.drr_bn_service.compare_bn_group", return_value=self.payload()) as compare:
                bridge.selectBackgroundFiles([QUrl.fromLocalFile(str(self.root/self.background))])
                wait(lambda: not bridge._pending and not bridge.snapshot["busy"] and not bridge.snapshot["bnBusy"])
            self.assertEqual(compare.call_count, 1)
            self.assertTrue(bridge.snapshot["canExport"], bridge.snapshot["status"])
            applied = self.store.find_applied(self.files)
            self.assertIsNotNone(applied)
            self.assertIn(applied.csv_path.name, bridge._drr_backgrounds[0])
            self.assertEqual(bridge._drr_products.request.background_recipe, bridge._drr_background_recipe)
            bridge.openDrrBackground()
            background = bridge.drrBackgroundBridge
            wait(lambda: not background.snapshot["busy"])
            self.assertTrue(background.snapshot["autoCompareSupported"])
            self.assertEqual(background._bn_payload["version_id"], applied.version_id)
            background.invoke("page:bn")
            self.assertTrue(background.snapshot["hasPreview"])

        def test_native_cancel_button_and_reopened_report_use_current_context(self):
            from PySide6.QtGui import QMouseEvent
            from PySide6.QtQml import QQmlApplicationEngine
            from ui_qt.quick.plot_surface import register_plot_surface
            register_plot_surface()
            bridge = self.bridge()
            bridge.selectBackgroundFiles([str(self.root/self.background)])
            wait(lambda: not bridge._pending)
            bridge.setParameter("backgroundFrame", "all")
            wait(lambda: not bridge._pending)
            engine = QQmlApplicationEngine()
            errors = []
            engine.warnings.connect(lambda values: errors.extend(item.toString() for item in values))
            engine.setInitialProperties({"appBridge": bridge})
            engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/"ui_qt/quick/qml/Application.qml")))
            self.assertTrue(engine.rootObjects(), errors)
            window = engine.rootObjects()[0]
            def dispose():
                window.close()
                engine.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                app.processEvents()
            self.addCleanup(dispose)
            def find(item, name):
                if item.objectName() == name:
                    return item
                for child in item.childItems():
                    found = find(child, name)
                    if found is not None:
                        return found
            self.settings.setValue("drr/bn_auto_compare", True)
            started, release = Event(), Event()
            payload = self.payload()
            def delayed(*_args, **_kwargs):
                started.set()
                release.wait(10)
                return deepcopy(payload)
            with patch("core.drr_bn_service.compare_bn_group", side_effect=delayed):
                try:
                    bridge._maybe_check_bn()
                    wait(started.is_set)
                    button = find(window.contentItem(), "drr_cancel_bn")
                    self.assertIsNotNone(button)
                    wait(button.isVisible)
                    point = button.mapToScene(button.boundingRect().center())
                    for kind, buttons in ((QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton),
                                          (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton)):
                        QCoreApplication.sendEvent(window, QMouseEvent(kind, point, point, Qt.MouseButton.LeftButton, buttons, Qt.NoModifier))
                    wait(lambda: not bridge.snapshot["bnBusy"])
                finally:
                    release.set()
                    wait(lambda: not bridge._bn_controller._pending)
            self.assertIsNone(self.store.find_applied(self.files))
            bridge.openDrrBackground()
            background = bridge.drrBackgroundBridge
            wait(lambda: not background.snapshot["busy"])
            dialog = window.findChild(QObject, "drr_background_window")
            flag = find(dialog.contentItem(), "drr_background_flag_auto_bn_compare")
            self.assertIsNotNone(flag)
            self.assertTrue(flag.isVisible())
            self.assertTrue(flag.isEnabled())
            report = self.payload()
            self.assertTrue(background.restore_comparison(report))
            different_frame = deepcopy(report)
            different_frame["source_identity"]["baseline_which"] = "last"
            self.assertFalse(background.restore_comparison(different_frame))
            stale = deepcopy(report)
            stale["settings"]["derivative_window"] = 23
            self.assertFalse(background.restore_comparison(stale))
            self.assertEqual(background._bn_payload["settings"]["derivative_window"], 21)
            self.assertEqual(errors, [])

        def test_invalid_managed_sg_input_reports_error_without_gui_exception(self):
            bridge, _version = self.load_managed_bridge()
            bridge.setParameter("sgWindow", 22)
            wait(lambda: not bridge.snapshot["bnBusy"])
            self.assertFalse(bridge.snapshot["canExport"])
            self.assertIn("odd", bridge.snapshot["status"])

        def test_root_sg_change_with_auto_off_revalidates_and_withdraws_old_activation(self):
            bridge, _version = self.load_managed_bridge()
            before = bridge._cube
            bridge.setParameter("sgWindow", 23)
            self.assertFalse(bridge.snapshot["canExport"])
            self.assertIs(bridge._cube, before)
            wait(lambda: not bridge.snapshot["bnBusy"] and not bridge._pending)
            self.assertTrue(bridge.snapshot["canExport"], bridge.snapshot["status"])
            self.assertEqual(bridge._drr_products.request.window, 23)
            self.assertIsNone(self.store.find_applied(self.files))
            self.assertFalse(self.store.has_original_choice(self.files))
            self.assertEqual(bridge._drr_backgrounds, (self.background,))

        def test_cancel_and_changed_selection_never_publish_a_late_candidate(self):
            bridge = self.bridge()
            bridge.selectBackgroundFiles([str(self.root/self.background)])
            wait(lambda: not bridge._pending)
            self.settings.setValue("drr/bn_auto_compare", True)
            old_cube = bridge._cube
            started, release = Event(), Event()
            payload = self.payload()
            def delayed(*_args, **_kwargs):
                started.set()
                release.wait(10)
                return deepcopy(payload)
            with patch("core.drr_bn_service.compare_bn_group", side_effect=delayed):
                try:
                    bridge._maybe_check_bn()
                    wait(started.is_set)
                    self.assertTrue(bridge.snapshot["bnBusy"])
                    bridge.invoke("cancel-bn")
                    self.assertFalse(bridge.snapshot["bnBusy"])
                finally:
                    release.set()
                    wait(lambda: not bridge._bn_controller._pending)
            self.assertIs(bridge._cube, old_cube)
            self.assertIsNone(self.store.find_applied(self.files))
            started.clear()
            release.clear()
            bridge._bn_checked_scope = None
            with patch("core.drr_bn_service.compare_bn_group", side_effect=delayed):
                try:
                    bridge._maybe_check_bn()
                    wait(started.is_set)
                    bridge.selectWorkflow("pl")
                finally:
                    release.set()
                    wait(lambda: not bridge._bn_controller._pending)
            self.assertIsNone(self.store.find_applied(self.files))
            self.assertEqual(bridge.snapshot["workflow"], "pl")

        def test_changed_source_between_trial_and_owner_commit_is_rejected(self):
            bridge = self.bridge()
            bridge.selectBackgroundFiles([str(self.root/self.background)])
            wait(lambda: not bridge._pending)
            before, scope = bridge._cube, bridge._bn_scope()
            with patch("core.drr_bn_service.compare_bn_group", return_value=self.payload()):
                outcome = self.check(bridge._drr_request())
            with (self.root/self.files[0]).open("a") as stream:
                stream.write("\n")
            bridge._bn_finished(scope, outcome, 0.)
            self.assertIs(bridge._cube, before)
            self.assertIsNone(self.store.find_applied(self.files))
            self.assertIn("not changed", bridge.snapshot["status"])

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    assert "PySide6.QtWidgets" not in sys.modules
    print("NATIVE_BN_CASES="+str(result.testsRun))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_isolated())
    unittest.main()
