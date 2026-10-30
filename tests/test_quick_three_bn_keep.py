"""Three measured BN references retain real QML Keep intent across group loads."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickThreeBnKeepProcessTests(unittest.TestCase):
    def test_three_bn_keep_in_real_qml(self):
        with TemporaryDirectory(prefix="quick-three-bn-process-") as temporary:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen",
                QT_QUICK_BACKEND="software", QT_SCALE_FACTOR="1", MPLBACKEND="agg",
                MPLCONFIGDIR=temporary, LOCALAPPDATA=temporary, XDG_CONFIG_HOME=temporary,
                XDG_CACHE_HOME=temporary, QML_DISK_CACHE_PATH=temporary,
                PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("THREE_BN_KEEP_TESTS=", result.stdout)
        print(result.stdout.strip())


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from time import monotonic, sleep
    from unittest.mock import patch

    import numpy as np
    from PySide6.QtCore import Q_ARG, QEvent, QMetaObject, QObject, QPointF, QSettings, Qt, QUrl
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from PySide6.QtTest import QTest

    from core.data_io import load_drr_external_cube
    from ui_qt.quick.appearance import quick_colors
    from ui_qt.quick.application_bridge import QuickApplicationBridge
    from ui_qt.quick.plot_surface import register_plot_surface

    QQuickStyle.setStyle("Basic")
    application = QGuiApplication([])
    register_plot_surface()

    def visual_items(parent):
        yield parent
        for child in parent.childItems():
            yield from visual_items(child)

    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary = TemporaryDirectory(prefix="quick-three-bn-case-")
            self.addCleanup(self.temporary.cleanup)
            self.root = Path(self.temporary.name).resolve()
            self.raw = self.root / "Initial Data" / "REF"
            self.raw.mkdir(parents=True)
            prefix = "YZ383_p2n1_9T_1.67KREF_"
            self.original = tuple(self.write(f"{prefix}650nmc_0p3sx10_RotIn94deg_TG_rep{index:02}.csv", offset=index)
                                  for index in range(1, 4))
            self.next_group = tuple(self.write(f"{prefix}660nmc_0p3sx10_RotIn94deg_TG_rep{index:02}.csv", offset=7+index)
                                    for index in range(1, 4))
            self.references = tuple(self.write(f"{prefix}650nmc_0p3sx10_RotIn94deg_BNbackground_repeat_{index:02}.csv",
                                               offset=40, constant=True) for index in range(1, 4))
            # Each original is closest to the same measured BN. This permits
            # testing homogeneous matched plans without a synthetic plan.
            for index, path in enumerate(self.references):
                os.utime(path, (1700000000-index*600, 1700000000-index*600))
            for path in (*self.original, *self.next_group):
                os.utime(path, (1700000001, 1700000001))
            self.bridge = QuickApplicationBridge(settings=QSettings(
                str(self.root / "profile.ini"), QSettings.Format.IniFormat))
            self.addCleanup(self.bridge.close)
            plotting = patch.object(self.bridge, "_schedule_plot")
            plotting.start()
            self.addCleanup(plotting.stop)
            self.bridge.selectWorkflow("drr")
            self.bridge.setSourcePath(str(self.root))
            self.warnings, self.wrappers = [], []
            self.engine = QQmlApplicationEngine()
            self.engine.warnings.connect(lambda messages: self.warnings.extend(message.toString() for message in messages))
            self.engine.setInitialProperties(dict(bridge=self.bridge, colors=quick_colors("light")))
            qml = Path(__file__).resolve().parents[1] / "ui_qt" / "quick" / "qml"
            self.engine.loadData(b'''import QtQuick
import QtQuick.Controls.Basic as Basic
Basic.ApplicationWindow {
    id: window
    required property var bridge
    required property var colors
    width: 1200; height: 850; visible: true
    color: colors.surface
    palette.window: colors.surface
    palette.windowText: colors.text
    palette.base: colors.control
    palette.text: colors.text
    palette.button: colors.control
    palette.buttonText: colors.text
    palette.highlight: colors.accent
    palette.highlightedText: colors.selectedText
    palette.placeholderText: colors.muted
    Loader { id: background; anchors.fill: parent }
    SourceCatalog {
        id: catalog
        bridge: window.bridge.sourceCatalogBridge
        colors: window.colors
        parent: Basic.Overlay.overlay
    }
    Connections {
        target: window.bridge
        function onSourceCatalogRequested() { catalog.open() }
        function onBackgroundDialogRequested() {
            background.setSource(Qt.resolvedUrl("DrrBackgroundSettings.qml"),
                {bridge: window.bridge.drrBackgroundBridge, colors: window.colors})
        }
        function onBackgroundDialogClosed() { background.source = "" }
    }
}''', QUrl.fromLocalFile(str(qml / "three-bn-keep-test.qml")))
            self.assertTrue(self.engine.rootObjects(), "\n".join(self.warnings))
            self.window = self.engine.rootObjects()[0]
            self.addCleanup(self.dispose)
            QTest.qWait(30)
            self.load_group(self.original)

        def write(self, name, *, offset=0, constant=False):
            wave, gate = np.linspace(630., 690., 41), np.linspace(-2., 2., 9)
            shape = np.linspace(-1., 1., len(wave))
            values = 130 + offset + 4*shape[None, :]**2 + np.zeros((len(gate), len(wave)))
            if not constant:
                values += 15*gate[:, None] + 3*gate[:, None]*shape[None, :]
            path = self.raw / name
            np.savetxt(path, np.column_stack((np.zeros_like(gate) if constant else gate*.5,
                np.zeros_like(gate) if constant else gate, values)), delimiter=",",
                header="Vbg,Vtg," + ",".join(map(str, wave)), comments="")
            return path.resolve()

        def wait(self, predicate, timeout=25):
            deadline = monotonic() + timeout
            while monotonic() < deadline:
                application.processEvents()
                if predicate():
                    application.processEvents()
                    return
                sleep(.003)
            self.fail("Three-BN QML test timed out: " + self.bridge.snapshot["status"])

        def idle(self):
            self.wait(lambda: not self.bridge._pending and not self.bridge.snapshot["busy"]
                and not self.bridge.snapshot["bnBusy"])

        def dispose(self):
            self.window.setVisible(False)
            self.engine.deleteLater()
            application.sendPostedEvents(None, QEvent.DeferredDelete)
            application.processEvents()
            self.wrappers.clear()
            self.assertFalse(self.warnings, "\n".join(self.warnings))

        def find(self, name):
            item = self.window.findChild(QObject, name)
            if item is None:
                item = next((child for child in visual_items(self.window.contentItem())
                    if child.objectName() == name), None)
            self.assertIsNotNone(item, name)
            self.wrappers.append(item)
            return item

        def click(self, item):
            self.assertTrue(item.isVisible(), item.objectName())
            self.assertTrue(item.isEnabled(), item.objectName())
            point = item.mapToScene(QPointF(item.width()*.5, item.height()*.5)).toPoint()
            QTest.mouseClick(self.window, Qt.LeftButton, Qt.NoModifier, point)
            QTest.qWait(20)

        def expose(self, name, view_name, model, identity):
            view = self.find(view_name)
            index = model.indexOfId(str(identity))
            self.assertGreaterEqual(index, 0, str(identity) + " visible=" + repr(model.rows))
            self.assertTrue(QMetaObject.invokeMethod(view, "positionViewAtIndex", Qt.DirectConnection,
                Q_ARG(int, index), Q_ARG(int, 0)))
            self.assertTrue(QMetaObject.invokeMethod(view, "forceLayout", Qt.DirectConnection))
            QTest.qWait(20)
            return self.find(name)

        def load_group(self, paths):
            # The application action and real Load selected button exercise the
            # measurement catalog's acceptance callback into selectFiles.
            self.bridge.invoke("select-files")
            catalog = self.bridge.sourceCatalogBridge
            self.wait(lambda: not catalog.snapshot["busy"])
            self.assertEqual(catalog.snapshot["error"], "")
            self.assertTrue(catalog.filesModel.rows, repr(catalog.snapshot))
            if catalog.snapshot["selected"]:
                self.click(self.find("source_catalog_clear"))
            # Reopening restores the last successfully loaded group. Choose
            # the next group through its real QML row before selecting files.
            target_paths = {str(path) for path in paths}
            group = next((row for row in catalog.groupsModel.rows
                if target_paths <= set(row["members"])), None)
            self.assertIsNotNone(group, repr(catalog.groupsModel.rows))
            self.click(self.expose("source_catalog_group_" + group["id"],
                "source_catalog_groups", catalog.groupsModel, group["id"]))
            self.assertEqual(catalog.snapshot["selectedGroup"], group["id"])
            for path in paths:
                self.click(self.expose("source_catalog_file_" + path.name,
                    "source_catalog_files", catalog.filesModel, path))
            self.assertEqual(catalog.snapshot["selected"], [str(path) for path in paths])
            self.click(self.find("source_catalog_load"))
            self.idle()
            self.assertTrue(self.bridge.snapshot["canExport"] or self.bridge.snapshot["backgroundRequired"],
                            self.bridge.snapshot["status"])
            if self.bridge.snapshot["backgroundRequired"]:
                self.assertTrue(self.bridge._background_open)
                self.assertIsNone(self.bridge._drr_products)

        def open_background(self, *, matched=False):
            self.bridge.openDrrBackground()
            self.background = self.bridge.drrBackgroundBridge
            self.wait(lambda: not self.background.snapshot["busy"])
            self.click(self.find("drr_background_source_bn"))
            # The current picker uses explicit Common/Matched radio buttons;
            # merely selecting BN is no longer an assignment decision.
            self.click(self.find("drr_background_assignment_" + ("matched" if matched else "common")))
            self.assertEqual(self.background.snapshot["method"], "matched" if matched else "common")
            self.catalog = self.background.inlineFileSelectionBridge
            self.wait(lambda: self.catalog.snapshot["selectionOpen"] and not self.catalog.snapshot["busy"])

        def keep(self):
            control = self.find("drr_background_flag_fixed")
            self.assertFalse(control.property("checked"))
            self.click(control)
            self.assertTrue(control.property("checked"))
            self.assertTrue(self.background.snapshot["flags"]["fixed"])
            self.assertTrue(self.background.draft.fixed)

        def choose_references(self):
            for path in self.references:
                self.click(self.expose("drr_background_available_" + path.name,
                    "drr_background_available_files", self.catalog.filesModel, path))
            self.assertEqual(self.catalog.snapshot["selected"], [str(path) for path in self.references])

        def assert_fixed(self, paths, references):
            recipe = self.bridge._drr_background_recipe
            self.assertIsNotNone(recipe)
            self.assertTrue(recipe.fixed)
            self.assertTrue(recipe.draft.fixed)
            self.assertEqual(recipe.draft.kind, "bn")
            self.assertEqual(recipe.mode, "External")
            self.assertEqual(recipe.files, tuple(path.relative_to(self.root).as_posix() for path in references))
            request = self.bridge._drr_products.request
            self.assertEqual(request.background_mode, "external")
            self.assertTrue(request.background_recipe.fixed)
            self.assertEqual(request.files, tuple(str(path) for path in paths))
            expected = load_drr_external_cube(str(self.root), tuple(str(path) for path in paths),
                tuple(str(path) for path in references), baseline_which="all", y_axis=request.y_axis)
            np.testing.assert_array_equal(self.bridge._drr_products.raw.Z, expected.Z)

        def apply_and_switch(self, references, *, clear_measurements=False):
            self.assertTrue(self.background.snapshot["flags"]["fixed"])
            self.assertTrue(self.background.draft.fixed)
            self.click(self.find("drr_background_apply"))
            self.wait(lambda: not self.background.snapshot["busy"])
            self.idle()
            self.assertEqual(self.background.snapshot["error"], "")
            self.assert_fixed(self.original, references)
            if clear_measurements:
                self.bridge.invoke("clear-files")
                self.assertIsNotNone(self.bridge._drr_background_recipe,
                    "Clearing measurements erased an explicitly kept background")
                self.assertTrue(self.bridge._drr_background_recipe.fixed)
            self.load_group(self.next_group)
            self.assert_fixed(self.next_group, references)
            self.bridge.openDrrBackground()
            self.wait(lambda: not self.background.snapshot["busy"])
            self.assertEqual(self.background.snapshot["kind"], "bn")
            self.assertTrue(self.find("drr_background_flag_fixed").property("checked"))

        def test_common_keep_before_selecting_three_references(self):
            self.open_background()
            self.keep()
            self.choose_references()
            self.apply_and_switch(self.references)

        def test_common_keep_after_selecting_three_references(self):
            self.open_background()
            self.choose_references()
            self.keep()
            self.apply_and_switch(self.references)

        def test_common_keep_survives_clear_reselect_and_catalog_refresh(self):
            self.open_background()
            self.choose_references()
            self.keep()
            self.click(self.find("drr_background_inline_clear"))
            self.assertTrue(self.background.snapshot["flags"]["fixed"])
            self.choose_references()
            self.write(self.references[0].name, offset=41, constant=True)
            self.catalog.refresh(True)
            self.wait(lambda: not self.catalog.snapshot["busy"])
            self.assertTrue(self.background.snapshot["flags"]["fixed"])
            self.apply_and_switch(self.references)

        def test_common_keep_survives_clear_measurements_before_next_group(self):
            self.open_background()
            self.choose_references()
            self.keep()
            self.apply_and_switch(self.references, clear_measurements=True)

        def test_clear_measurements_still_clears_an_unfixed_background(self):
            self.open_background()
            self.choose_references()
            self.click(self.find("drr_background_apply"))
            self.wait(lambda: not self.background.snapshot["busy"])
            self.idle()
            self.assertEqual(self.background.snapshot["error"], "")
            self.assertIsNotNone(self.bridge._drr_background_recipe)
            self.assertFalse(self.bridge._drr_background_recipe.fixed)
            self.bridge.invoke("clear-files")
            self.assertIsNone(self.bridge._drr_background_recipe)
            self.assertEqual(self.bridge._drr_backgrounds, ())
            self.assertEqual(self.bridge._drr_processing["backgroundMode"], "self-last")

        def test_explicit_clear_background_clears_kept_references(self):
            self.open_background()
            self.choose_references()
            self.keep()
            self.click(self.find("drr_background_apply"))
            self.wait(lambda: not self.background.snapshot["busy"])
            self.idle()
            self.assert_fixed(self.original, self.references)
            self.bridge.invoke("clear-background")
            self.idle()
            self.assertIsNone(self.bridge._drr_background_recipe)
            self.assertEqual(self.bridge._drr_backgrounds, ())
            self.assertIsNone(self.bridge._drr_products)
            self.assertTrue(self.bridge.snapshot["backgroundRequired"])
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.assertFalse(self.bridge._background_open)

        def test_matched_disables_keep_until_common_reference_is_chosen(self):
            self.open_background(matched=True)
            self.choose_references()
            self.click(self.find("drr_background_match"))
            self.wait(lambda: not self.background.snapshot["busy"])
            self.assertEqual(self.background.snapshot["error"], "")
            plan = self.background.draft.bn_plan
            self.assertIsNotNone(plan)
            self.assertTrue(plan.ready, repr([(row.measurement_file, row.assignment,
                row.evidence, row.warnings) for row in plan.rows]) + repr(plan.warnings))
            self.assertEqual(len({row.assignment.baseline_files for row in plan.rows}), 1)
            if self.background.snapshot["warnings"]:
                options = self.find("drr_background_settings_options").property("contentItem")
                options.setProperty("contentY", max(0., options.property("contentHeight") - options.height()))
                QTest.qWait(25)
                self.click(self.find("drr_background_flag_acknowledged"))
                self.assertTrue(self.background.snapshot["flags"]["acknowledged"])
            chosen = tuple(self.root / name for name in plan.rows[0].assignment.baseline_files)
            # Matching remains a per-measurement policy even when its current
            # suggestions coincide. Keeping a reference requires an explicit
            # switch to Common and selection of that shared reference.
            self.assertFalse(self.background.snapshot["canFix"])
            self.assertFalse(self.find("drr_background_flag_fixed").isEnabled())
            self.click(self.find("drr_background_assignment_common"))
            self.wait(lambda: not self.catalog.snapshot["busy"])
            if self.catalog.snapshot["selected"]:
                self.click(self.find("drr_background_inline_clear"))
            for path in chosen:
                self.click(self.expose("drr_background_available_" + path.name,
                    "drr_background_available_files", self.catalog.filesModel, path))
            self.assertEqual(self.catalog.snapshot["selected"], [str(path) for path in chosen])
            self.keep()
            self.apply_and_switch(chosen)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    print(f"THREE_BN_KEEP_TESTS={result.testsRun}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_isolated())
    unittest.main()
