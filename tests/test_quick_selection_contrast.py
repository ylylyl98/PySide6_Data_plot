"""Exercise rendered Quick selection states with isolated settings and no data.

These tests inspect live delegates, including the popup that Qt creates, rather
than treating a color token or a source-code expression as proof of contrast.
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")
os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")

from pathlib import Path
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from time import monotonic
import unittest

from PySide6.QtCore import QMetaObject, QObject, Property, QSettings, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtQml import QJSValue, QQmlApplicationEngine, QQmlComponent, QQmlProperty, qmlRegisterType
from PySide6.QtQuick import QQuickItem
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from tests.test_quick_application_shell import _Bridge, _PlotSurface
from ui_qt.quick.appearance import QuickAppearance
from ui_qt.quick.source_catalog import SourceCatalogBridge
from ui_qt.typography import install_application_font


ROOT = Path(__file__).resolve().parents[1]
QML = ROOT / "ui_qt" / "quick" / "qml"
MODES = ("light", "dark", "high-contrast")


def contrast(foreground, background):
    def luminance(value):
        color = QColor(value)
        def linear(channel):
            return channel / 12.92 if channel <= .04045 else ((channel + .055) / 1.055) ** 2.4
        return sum(weight * linear(channel) for weight, channel in
                   zip((.2126, .7152, .0722), (color.redF(), color.greenF(), color.blueF())))
    first, second = sorted((luminance(foreground), luminance(background)))
    return (second + .05) / (first + .05)


def visual_items(parent):
    yield parent
    for child in parent.childItems():
        yield from visual_items(child)


class _SelectionBridge(_Bridge):
    sourceCatalogRequested = Signal()

    def __init__(self, settings):
        super().__init__()
        self.appearance = QuickAppearance(settings, self)
        self.appearance.changed.connect(self.colorsChanged)
        self.catalog = SourceCatalogBridge(settings=settings, parent=self)
        self.update(pngExportSupported=True, pngLayout="wide", pngWidth=2,
                    inspectionVisible=False, parameterGroups=[])

    @Property("QVariantMap", notify=_Bridge.colorsChanged)
    def colors(self):
        return self.appearance.colors

    @Property(QObject, constant=True)
    def sourceCatalogBridge(self):
        return self.catalog

    @Slot(str, "QVariant")
    def setParameter(self, key, value):
        if key == "export.pngLayout":
            self.calls.append(("parameter", key, value))
            self.update(pngLayout=value)
        else:
            super().setParameter(key, value)


HARNESS = b'''import QtQuick
import QtQuick.Controls as Controls
import QtQuick.Controls.Basic as Basic
import QtQuick.Dialogs as Dialogs
import "DialogState.js" as DialogState
import "."
Controls.ApplicationWindow {
    id: window
    required property var bridge
    readonly property var colors: bridge.colors
    property var options: ["PNG 1.0x", "Wide PNG", "Tall PNG"]
    property string optionRole: ""
    ListModel {
        id: mutableOptions
        ListElement { caption: "Channel A"; value: 10 }
        ListElement { caption: "Channel B"; value: 20 }
    }
    function useMutableOptions() { optionRole = "caption"; options = mutableOptions }
    function renameFirst() { mutableOptions.setProperty(0, "caption", "Renamed channel A") }
    function renameSecond() { mutableOptions.setProperty(1, "caption", "Renamed channel B") }
    property url pickerFolder
    property url acceptedFile
    property url acceptedFolder
    property int fileRejections: 0
    property int folderRejections: 0
    function openFilePicker() {
        DialogState.prepareFile(filePicker, pickerFolder)
        filePicker.open()
    }
    function openFolderPicker() {
        DialogState.prepareFolder(folderPicker, pickerFolder)
        folderPicker.open()
    }
    function cancelFileImmediately() { openFilePicker(); filePicker.reject() }
    function cancelFolderImmediately() { openFolderPicker(); folderPicker.reject() }
    AppFileDialog {
        id: filePicker; objectName: "selection_file_picker"
        colors: window.colors; parentWindow: window
        options: Dialogs.FileDialog.DontUseNativeDialog
        fileMode: Dialogs.FileDialog.OpenFile
        nameFilters: ["CSV measurements (*.csv)"]
        onAccepted: window.acceptedFile = selectedFile
        onRejected: window.fileRejections++
    }
    AppFolderDialog {
        id: folderPicker; objectName: "selection_folder_picker"
        colors: window.colors; parentWindow: window
        onAccepted: window.acceptedFolder = selectedFolder
        onRejected: window.folderRejections++
    }
    width: 540; height: 430; visible: true
    color: colors.inset
    palette.window: colors.inset
    palette.windowText: colors.text
    palette.base: colors.control
    palette.alternateBase: colors.surface
    palette.text: colors.text
    palette.button: colors.control
    palette.buttonText: colors.text
    palette.highlight: colors.accent
    palette.highlightedText: colors.selectedText
    palette.mid: colors.border
    palette.dark: colors.border
    palette.light: colors.control
    palette.placeholderText: colors.muted
    AppComboBox {
        objectName: "choice"; x: 30; y: 25; width: 230; height: 36
        colors: window.colors; model: window.options; textRole: window.optionRole
        currentIndex: 1
    }
    AppItemDelegate {
        objectName: "disabled_row"; x: 290; y: 25; width: 220; height: 36
        colors: window.colors; text: "Unavailable result"; enabled: false; highlighted: true
    }
    Basic.TextField {
        objectName: "text_editor"; x: 30; y: 225; width: 230; height: 36
        text: "Selectable source path"; selectByMouse: true
    }
    AppComboBox {
        objectName: "editable_choice"; x: 290; y: 225; width: 220; height: 36
        colors: window.colors; model: ["Editable selection", "Second choice"]
        editable: true; currentIndex: 0
    }
    AppTabButton {
        objectName: "checked_tab"; x: 30; y: 290; width: 230; height: 36
        colors: window.colors; text: "Selected results"; checked: true
    }
    AppTabButton {
        objectName: "normal_tab"; x: 290; y: 290; width: 220; height: 36
        colors: window.colors; text: "Other results"
    }
}
'''


class QuickSelectionContrastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if QApplication.instance() is None:
            QQuickStyle.setStyle("Basic")
        cls.app = QApplication.instance() or QApplication([])
        install_application_font(cls.app)
        qmlRegisterType(_PlotSurface, "Dptk", 1, 0, "PlotSurface")

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.settings = QSettings(str(Path(self.temporary.name) / "selection.ini"), QSettings.IniFormat)
        self.bridge = _SelectionBridge(self.settings)
        self.engines = []
        self.windows = []
        self.adapters = []

    def tearDown(self):
        for window in self.windows:
            for name in ("selection_file_picker", "selection_folder_picker"):
                picker = window.findChild(QObject, name)
                if picker is not None:
                    QMetaObject.invokeMethod(picker, "reject")
            window.close()
        self.bridge.appearance.close()
        self.bridge.catalog.close()
        for engine in self.engines:
            engine.deleteLater()
        self.app.processEvents()
        self.temporary.cleanup()

    def wait(self, predicate, message):
        deadline = monotonic() + 3
        while monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            QTest.qWait(10)
        self.fail(message)

    def load(self, *, shell=False):
        engine = QQmlApplicationEngine()
        self.engines.append(engine)
        warnings = []
        engine.warnings.connect(lambda messages: warnings.extend(message.toString() for message in messages))
        engine.setInitialProperties({"appBridge" if shell else "bridge": self.bridge})
        if shell:
            engine.load(QUrl.fromLocalFile(str(QML / "Application.qml")))
        else:
            engine.loadData(HARNESS, QUrl.fromLocalFile(str(QML / "selection-test.qml")))
        self.assertTrue(engine.rootObjects(), "\n".join(warnings))
        window = engine.rootObjects()[0]
        self.windows.append(window)
        self.wait(window.isVisible, "test window did not become visible")
        QTest.qWait(30)
        return engine, window

    def item(self, window, name):
        result = next((item for item in visual_items(window.contentItem()) if item.objectName() == name), None)
        self.assertIsNotNone(result, name)
        return result

    def popup(self, engine, choice):
        # QQuickPopup* has no Python converter. A typed QtObject property exposes
        # the same object through QObject*, without replacing Qt's real popup.
        adapter = QQmlComponent(engine)
        adapter.setData(b'''import QtQuick
QtObject {
    required property Item control
    property QtObject popupObject: control.popup
    property Item listObject: control.popup.contentItem
}''', QUrl())
        instance = adapter.createWithInitialProperties({"control": choice})
        self.assertIsNotNone(instance, "\n".join(error.toString() for error in adapter.errors()))
        self.adapters.append((adapter, instance))
        return instance.property("popupObject"), instance.property("listObject")

    def click(self, window, item):
        point = item.mapToScene(item.boundingRect().center()).toPoint()
        QTest.mouseClick(window, Qt.LeftButton, Qt.NoModifier, point)
        QTest.qWait(20)

    def open_choice(self, engine, window, choice, *, keyboard=False):
        popup, listing = self.popup(engine, choice)
        if keyboard:
            choice.forceActiveFocus()
            QTest.keyClick(window, Qt.Key_Space)
        else:
            self.click(window, choice)
        self.wait(lambda: popup.property("visible"), "ComboBox popup did not open")
        self.wait(lambda: len(self.rows(listing)) == choice.property("count"), "popup rows were not instantiated")
        return popup, listing

    @staticmethod
    def rows(listing):
        return [item for item in visual_items(listing)
                if item.property("highlighted") is not None and item.property("text") is not None
                and item.property("background") is not None]

    def assert_readable(self, control, *, minimum=4.5, label_text=None):
        content = control.property("contentItem")
        background = control.property("background")
        self.assertTrue(background.isVisible())
        # A selected row can include a remove button alongside its text.
        # Inspect the rendered label rather than treating its layout as text.
        expected_text = control.property("text") if label_text is None else label_text
        label = content if content.property("color") is not None else next((item for item in visual_items(content)
                      if item.isVisible() and item.property("color") is not None
                      and item.property("text") == expected_text), None)
        self.assertIsNotNone(label, control.property("text"))
        foreground = label.property("color")
        surface = background.property("color")
        ratio = contrast(foreground, surface)
        self.assertGreaterEqual(ratio, minimum,
            f"{control.property('text')!r}: {QColor(foreground).name()} on {QColor(surface).name()} = {ratio:.2f}:1")

    def screenshot(self, window, name):
        if os.name == "nt":
            destination = ROOT / "build" / "diagnostics" / "selection-contrast"
            destination.mkdir(parents=True, exist_ok=True)
            self.assertTrue(window.grabWindow().save(str(destination / f"{name}.png")))

    def test_visible_popup_hover_and_keyboard_selection_are_readable_in_every_theme(self):
        engine, window = self.load()
        choice = self.item(window, "choice")
        for mode in MODES:
            with self.subTest(mode=mode):
                self.bridge.appearance.setMode(mode)
                choice.setProperty("currentIndex", 1)
                popup, listing = self.open_choice(engine, window, choice)
                rows = self.rows(listing)
                self.assertEqual([row.property("text") for row in rows], ["PNG 1.0x", "Wide PNG", "Tall PNG"])
                self.assertTrue(rows[1].property("highlighted"))
                for row in rows:
                    self.assert_readable(row)
                QTest.mouseMove(window, rows[0].mapToScene(rows[0].boundingRect().center()).toPoint())
                self.wait(lambda: rows[0].property("hovered") and rows[0].property("highlighted"), "hover did not highlight first row")
                self.assert_readable(rows[0])
                QTest.keyClick(window, Qt.Key_Down)
                self.wait(lambda: rows[1].property("highlighted"), "Down did not move the popup highlight")
                self.assert_readable(rows[1])
                if mode == "high-contrast":
                    background = rows[1].property("background")
                    self.assertGreater(QQmlProperty.read(background, "border.width"), 0)
                    self.assertGreaterEqual(contrast(QQmlProperty.read(background, "border.color"), background.property("color")), 3)
                self.screenshot(window, f"fixed-popup-{mode}")
                QTest.keyClick(window, Qt.Key_Down)
                QTest.keyClick(window, Qt.Key_Return)
                self.wait(lambda: not popup.property("visible"), "Return did not close the popup")
                self.assertEqual(choice.property("currentIndex"), 2)
                self.assertEqual(choice.property("displayText"), "Tall PNG")

    def test_text_role_models_keep_display_and_mouse_activation(self):
        engine, window = self.load()
        window.setProperty("optionRole", "caption")
        window.setProperty("options", [{"caption": "Channel A", "value": 10}, {"caption": "Channel B", "value": 20}])
        choice = self.item(window, "choice")
        choice.setProperty("currentIndex", 0)
        popup, listing = self.open_choice(engine, window, choice, keyboard=True)
        rows = self.rows(listing)
        self.assertEqual([row.property("text") for row in rows], ["Channel A", "Channel B"])
        for row in rows:
            self.assert_readable(row)
        self.click(window, rows[1])
        self.wait(lambda: not popup.property("visible"), "click did not close the popup")
        self.assertEqual(choice.property("currentIndex"), 1)
        self.assertEqual(choice.property("displayText"), "Channel B")

    def test_open_popup_tracks_live_theme_changes_without_losing_navigation(self):
        engine, window = self.load()
        choice = self.item(window, "choice")
        popup, listing = self.open_choice(engine, window, choice, keyboard=True)
        for mode in ("dark", "high-contrast", "light"):
            with self.subTest(mode=mode):
                self.bridge.appearance.setMode(mode)
                self.app.processEvents()
                self.assertTrue(popup.property("visible"))
                highlighted = [row for row in self.rows(listing) if row.property("highlighted")]
                self.assertEqual(len(highlighted), 1)
                for row in self.rows(listing):
                    self.assert_readable(row)
                self.assertEqual(choice.property("currentIndex"), 1)
        QTest.keyClick(window, Qt.Key_Down)
        QTest.keyClick(window, Qt.Key_Return)
        self.assertEqual(choice.property("currentIndex"), 2)

    def test_mutable_text_role_model_updates_open_popup_labels(self):
        engine, window = self.load()
        self.assertTrue(QMetaObject.invokeMethod(window, "useMutableOptions"))
        choice = self.item(window, "choice")
        choice.setProperty("currentIndex", 0)
        _, listing = self.open_choice(engine, window, choice)
        rows = self.rows(listing)
        self.assertTrue(rows[0].property("highlighted"))
        self.assertTrue(QMetaObject.invokeMethod(window, "renameFirst"))
        self.wait(lambda: rows[0].property("text") == "Renamed channel A", "current popup row kept a stale model label")
        self.assertEqual(choice.property("displayText"), "Renamed channel A")
        QTest.keyClick(window, Qt.Key_Down)
        self.assertTrue(QMetaObject.invokeMethod(window, "renameSecond"))
        self.wait(lambda: rows[1].property("text") == "Renamed channel B", "highlighted popup row kept a stale model label")
        self.assertTrue(rows[1].property("highlighted"))
        QTest.keyClick(window, Qt.Key_Return)
        self.assertEqual(choice.property("displayText"), "Renamed channel B")

    def test_disabled_controls_and_tabs_remain_visible_in_all_themes(self):
        engine, window = self.load()
        choice = self.item(window, "choice")
        choice.setProperty("enabled", False)
        popup, _ = self.popup(engine, choice)
        for mode in MODES:
            with self.subTest(mode=mode):
                self.bridge.appearance.setMode(mode)
                self.app.processEvents()
                self.click(window, choice)
                self.assertFalse(popup.property("visible"))
                self.assertEqual(choice.property("currentIndex"), 1)
                self.assert_readable(choice, minimum=2)
                self.assert_readable(self.item(window, "disabled_row"), minimum=2)
                for name in ("checked_tab", "normal_tab"):
                    self.assert_readable(self.item(window, name))

    def test_text_selection_keeps_an_accent_background_with_readable_text(self):
        _, window = self.load()
        for mode in MODES:
            with self.subTest(mode=mode):
                self.bridge.appearance.setMode(mode)
                for name in ("text_editor", "editable_choice"):
                    control = self.item(window, name)
                    editor = control.property("contentItem") if name == "editable_choice" else control
                    editor.forceActiveFocus()
                    QTest.keyClick(window, Qt.Key_A, Qt.ControlModifier)
                    self.wait(lambda: bool(editor.property("selectedText")), f"{name} text was not selected")
                    self.assertEqual(editor.property("selectedText"), editor.property("text"))
                    self.assertGreaterEqual(contrast(editor.property("selectedTextColor"), editor.property("selectionColor")), 4.5)
                self.screenshot(window, f"fixed-text-selection-{mode}")

    def test_real_application_png_toolbar_selection_is_readable_and_commits(self):
        engine, window = self.load(shell=True)
        choice = self.item(window, "app_png_layout")
        background_window = window.findChild(QObject, "drr_background_window")
        self.assertIsNotNone(background_window)
        for mode in MODES:
            with self.subTest(mode=mode):
                self.bridge.appearance.setMode(mode)
                self.app.processEvents()
                for foreground, background in (("text", "base"), ("highlightedText", "highlight")):
                    self.assertGreaterEqual(contrast(QQmlProperty.read(background_window, "palette." + foreground),
                                                    QQmlProperty.read(background_window, "palette." + background)), 4.5)
                popup, listing = self.open_choice(engine, window, choice)
                rows = self.rows(listing)
                self.assertEqual([row.property("text") for row in rows], ["PNG 1.0×", "Wide PNG"])
                for row in rows:
                    self.assert_readable(row)
                self.screenshot(window, f"fixed-application-toolbar-{mode}")
                self.click(window, rows[0])
                self.wait(lambda: not popup.property("visible"), "toolbar selection did not close popup")
                self.assertIn(("parameter", "export.pngLayout", "original"), self.bridge.calls)
                self.assertEqual(self.bridge.snapshot["pngLayout"], "original")
                self.assertEqual(choice.property("currentIndex"), 0)
                self.bridge.update(pngLayout="wide")
        self.bridge.update(busy=True)
        self.app.processEvents()
        self.assertFalse(choice.isEnabled())

    def test_real_source_catalog_group_file_and_chosen_rows_keep_readable_hover(self):
        catalog = self.bridge.catalog
        identifier = str(Path(self.temporary.name) / "synthetic_selection.csv")
        catalog._configure("drr", self.temporary.name, [identifier], False)
        catalog._catalog = dict(sources=(), files=[dict(id=identifier, name="synthetic_selection.csv",
            source="synthetic_selection.csv", kind="measurement", status="new", saved=False,
            background=False, detail="Synthetic source for UI contrast", processedAt="")],
            groups=[dict(id="synthetic_group", name="Synthetic DRR run", session="2026-10-24",
                count=1, processedCount=0, detail="Synthetic acquisition", members=[identifier])])
        catalog._catalog_validated = True
        catalog._selected_group = "synthetic_group"
        catalog._refresh_models()
        _, window = self.load(shell=True)
        self.bridge.sourceCatalogRequested.emit()
        dialog = window.findChild(QObject, "quick_source_catalog")
        self.wait(lambda: dialog is not None and dialog.property("visible"), "source catalog did not open")
        for mode in MODES:
            with self.subTest(mode=mode):
                self.bridge.appearance.setMode(mode)
                self.app.processEvents()
                candidates = [self.item(window, name) for name in (
                    "source_catalog_group_synthetic_group",
                    "source_catalog_file_synthetic_selection.csv",
                    "source_catalog_chosen_synthetic_selection.csv")]
                self.assertTrue(any(item.property("highlighted") for item in candidates))
                for row in candidates:
                    QTest.mouseMove(window, row.mapToScene(row.boundingRect().center()).toPoint())
                    self.app.processEvents()
                    # Group and file cards compose separate title/status labels.
                    self.assert_readable(row, label_text="Synthetic DRR run"
                        if row.objectName().startswith("source_catalog_group_") else "synthetic_selection.csv")
                self.screenshot(window, f"fixed-source-catalog-{mode}")

    @staticmethod
    def palette_snapshot(window):
        return {role: QColor(QQmlProperty.read(window, "palette." + role)).name()
                for role in ("light", "highlight", "highlightedText", "mid")}

    def assert_restored_palette(self, window):
        colors = self.bridge.colors
        expected = dict(light=colors["control"], highlight=colors["accent"],
                        highlightedText=colors["selectedText"], mid=colors["border"])
        self.wait(lambda: self.palette_snapshot(window) == {key: value.lower() for key, value in expected.items()},
                  "picker left a palette override or failed to restore the live theme binding")

    def picker_surface(self, kind):
        name = "fileDialogListView" if kind == "file" else "folderDialogListView"
        def locate():
            for window in self.app.allWindows():
                if not window.isVisible() or not hasattr(window, "contentItem"):
                    continue
                for listing in visual_items(window.contentItem()):
                    if listing.objectName() != name or not listing.isVisible():
                        continue
                    surface = listing
                    while surface.parentItem() is not None and not surface.inherits("QQuickPopupItem"):
                        surface = surface.parentItem()
                    return window, listing, surface
            return None
        self.wait(lambda: locate() is not None, f"no visible {kind} picker ListView was rendered")
        window, listing, surface = locate()
        self.wait(window.isExposed, f"{kind} picker window was not exposed")
        self.assertGreater(listing.width(), 0)
        self.assertGreater(listing.height(), 0)
        return window, listing, surface

    def picker_row(self, kind, name):
        owner, listing, surface = self.picker_surface(kind)
        def locate():
            return next((item for item in visual_items(listing)
                         if item.property("fileName") == name and item.isVisible()
                         and item.property("background") is not None), None)
        try:
            self.wait(lambda: locate() is not None, f"{kind} picker has no visible row for {name}")
        except AssertionError:
            self.screenshot(owner, f"debug-missing-{kind}-row")
            names = [(item.objectName(), item.property("fileName"), item.isVisible())
                     for item in visual_items(listing) if item.objectName()]
            self.fail(f"{kind} picker has no visible row for {name}; rows={names}")
        row = locate()
        self.assertGreater(row.width(), 0)
        self.assertGreater(row.height(), 0)
        self.click(owner, row)
        self.wait(lambda: row.property("highlighted"), f"click did not select {kind} picker row {name}")
        return owner, row, surface

    def assert_picker_row_readable(self, row):
        background = row.property("background")
        self.assertTrue(row.isVisible())
        self.assertTrue(background.isVisible())
        labels = [item for item in visual_items(row.property("contentItem"))
                  if item.isVisible() and isinstance(item.property("text"), str) and item.property("text")
                  and item.property("color") is not None]
        self.assertTrue(labels, "selected picker row has no rendered text labels")
        for label in labels:
            ratio = contrast(label.property("color"), background.property("color"))
            self.assertGreaterEqual(ratio, 4.5,
                f"picker {label.property('text')!r}: {QColor(label.property('color')).name()} on "
                f"{QColor(background.property('color')).name()} = {ratio:.2f}:1")

    def assert_picker_text_selection(self, kind):
        owner, listing, surface = self.picker_surface(kind)
        listing.forceActiveFocus()
        QTest.keyClick(owner, Qt.Key_L, Qt.ControlModifier)
        def editors():
            return [item for item in visual_items(surface) if item.isVisible() and item.isEnabled()
                    and item.property("selectionColor") is not None and item.width() > 0 and item.height() > 0]
        self.wait(lambda: bool(editors()), f"Ctrl+L did not expose the {kind} picker location editor")
        editor = editors()[0]
        editor.forceActiveFocus()
        QTest.keyClick(owner, Qt.Key_A, Qt.ControlModifier)
        self.wait(lambda: bool(editor.property("selectedText")), f"{kind} picker location text was not selected")
        self.assertEqual(editor.property("selectedText"), editor.property("text"))
        self.assertGreaterEqual(contrast(editor.property("selectedTextColor"), editor.property("selectionColor")), 4.5)

    def assert_picker_capture(self, owner, name):
        QTest.qWait(30)
        rendered = owner.grabWindow()
        self.assertFalse(rendered.isNull(), "visible picker produced an empty screenshot")
        colors = {rendered.pixelColor(x, y).rgba()
                  for y in range(0, rendered.height(), 5) for x in range(0, rendered.width(), 5)}
        self.assertGreater(len(colors), 8, "picker screenshot contains only a blank background")
        self.screenshot(owner, name)

    @contextmanager
    def picker_cleanup(self, picker):
        try:
            yield
        finally:
            if picker.property("visible"):
                QMetaObject.invokeMethod(picker, "reject")
                self.app.processEvents()

    def exercise_picker(self, kind):
        _, window = self.load()
        window.setWidth(900)
        window.setHeight(680)
        first, second = Path(self.temporary.name) / "a", Path(self.temporary.name) / "b"
        for directory in (first, second):
            directory.mkdir()
            (directory / (directory.name + "_folder")).mkdir()
            (directory / (directory.name + "_measurement.csv")).write_text("x,y\n1,2\n", encoding="utf-8")
        picker = window.findChild(QObject, f"selection_{kind}_picker")
        self.assertIsNotNone(picker)
        open_method = "openFilePicker" if kind == "file" else "openFolderPicker"
        cancel_method = "cancelFileImmediately" if kind == "file" else "cancelFolderImmediately"
        selected_role = "selectedFile" if kind == "file" else "selectedFolder"
        accepted_role = "acceptedFile" if kind == "file" else "acceptedFolder"
        for index, mode in enumerate(MODES):
            with self.subTest(kind=kind, mode=mode), self.picker_cleanup(picker):
                self.bridge.appearance.setMode(mode)
                self.assert_restored_palette(window)
                original = self.palette_snapshot(window)
                window.setProperty("pickerFolder", QUrl.fromLocalFile(str(first)))
                self.assertTrue(QMetaObject.invokeMethod(window, open_method))
                self.wait(lambda: picker.property("visible"), f"{kind} picker public visible state is false")
                name = "a_measurement.csv" if kind == "file" else "a_folder"
                owner, row, _ = self.picker_row(kind, name)
                active = picker.property("_dialog") if kind == "folder" else picker
                active = active.toQObject() if isinstance(active, QJSValue) else active
                self.assertEqual(Path(active.property("currentFolder").toLocalFile()), first)
                self.wait(lambda: Path(active.property(selected_role).toLocalFile()) == first / name,
                          f"click did not update public {selected_role}")
                self.assert_picker_row_readable(row)
                self.assert_picker_text_selection(kind)
                self.assert_picker_capture(owner, f"fixed-picker-{kind}-{mode}")

                # A theme change while the modal picker is open must update its
                # actual text, preserve the selection and restore the new theme.
                self.bridge.appearance.setMode(MODES[(index + 1) % len(MODES)])
                self.app.processEvents()
                self.assertTrue(picker.property("visible"))
                self.assertTrue(row.property("highlighted"))
                self.assert_picker_row_readable(row)
                self.assert_picker_text_selection(kind)
                self.assertTrue(QMetaObject.invokeMethod(picker, "reject"))
                self.wait(lambda: not picker.property("visible"), f"{kind} picker did not reject")
                self.assert_restored_palette(window)

                self.bridge.appearance.setMode(mode)
                self.assert_restored_palette(window)
                self.assertEqual(self.palette_snapshot(window), original)
                window.setProperty("pickerFolder", QUrl.fromLocalFile(str(second)))
                self.assertTrue(QMetaObject.invokeMethod(window, cancel_method))
                self.wait(lambda: not picker.property("visible"), f"{kind} picker failed immediate cancellation")
                self.assert_restored_palette(window)
                self.assertEqual(self.palette_snapshot(window), original)

                self.assertTrue(QMetaObject.invokeMethod(window, open_method))
                name = "b_measurement.csv" if kind == "file" else "b_folder"
                owner, row, surface = self.picker_row(kind, name)
                active = picker.property("_dialog") if kind == "folder" else picker
                active = active.toQObject() if isinstance(active, QJSValue) else active
                self.assertEqual(Path(active.property("currentFolder").toLocalFile()), second)
                self.assert_picker_row_readable(row)
                buttons = [item for item in visual_items(surface) if item.isVisible() and item.isEnabled()
                           and item.property("text") in ("Open", "&Open") and item.property("background") is not None]
                self.assertEqual(len(buttons), 1, "picker has no unique enabled Open button")
                self.click(owner, buttons[0])
                self.wait(lambda: not picker.property("visible"), f"{kind} picker Open button did not accept")
                self.assertEqual(Path(window.property(accepted_role).toLocalFile()), second / name)
                self.assert_restored_palette(window)

    def test_fallback_file_picker_selection_and_palette_lifecycle(self):
        self.exercise_picker("file")

    def test_fallback_folder_picker_selection_and_palette_lifecycle(self):
        self.exercise_picker("folder")


if __name__ == "__main__":
    unittest.main()
