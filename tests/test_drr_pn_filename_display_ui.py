"""Complete P/N filenames must fit their actual painted table rows."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from PySide6.QtCore import QEventLoop, QTimer, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QStyleOptionViewItem

from core.drr_pn_pairing import DrrPnPair
from tests import test_drr_pn_background_ui as _fixtures
from ui_qt.common import WrappedFilenameDelegate
from ui_qt.theme import install_theme


class PnFilenameDisplayUiTests(unittest.TestCase):
    # Bind fixtures, without importing/inheriting a second discoverable suite.
    setUp = _fixtures.PnBackgroundDialogTests.setUp
    tearDown = _fixtures.PnBackgroundDialogTests.tearDown
    write = _fixtures.PnBackgroundDialogTests.write
    dialog = _fixtures.PnBackgroundDialogTests.dialog
    time_catalog = _fixtures.PnBackgroundDialogTests.time_catalog

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        font = Path("C:/Windows/Fonts/segoeui.ttf")
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))
            cls.app.setFont(QFont("Segoe UI", 9))
        install_theme(cls.app, mode="light")

    def wait(self, dialog, *, timeout=15):
        self.app.processEvents()
        if dialog._worker is None:
            return
        loop = QEventLoop()
        tick, watchdog = QTimer(), QTimer()
        tick.setInterval(10)
        tick.timeout.connect(lambda: loop.quit() if dialog._worker is None else None)
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(loop.quit)
        tick.start()
        watchdog.start(int(timeout * 1000))
        loop.exec()
        tick.stop()
        watchdog.stop()
        self.assertIsNone(dialog._worker, dialog.status.text())

    def long_catalog(self):
        catalog, old_names, samples = self.time_catalog()
        names, substitutions = {}, {}
        for (role, repeat), old_name in old_names.items():
            edge = "EARLY_A" if repeat == "001" else "LATER_Z"
            name = (f"{edge}_test_1K_1T_RotIn17.5deg_730nmc_0.3sx10_"
                    f"gate_sweep_repeatability_reference_full_native_spectra_"
                    f"{role}background_distinct_tail_{repeat}.csv")
            self.assertGreater(len(name), 100)
            (self.root / name).write_bytes((self.root / old_name).read_bytes())
            names[role, repeat] = name
            substitutions[old_name] = name
        return [replace(source, source=substitutions[source.source],
                        filename=substitutions[source.source])
                if source.source in substitutions else source for source in catalog], names, samples

    def test_complete_names_fit_first_tab_display_resize_and_actual_selection(self):
        catalog, names, samples = self.long_catalog()
        dialog = self.dialog(catalog=catalog, measurements=samples, sample_scoped=True)
        dialog.show_other_conditions.setChecked(True)
        painted = []
        original_paint = WrappedFilenameDelegate.paint

        def observe_paint(delegate, painter, option, index):
            table = delegate.parent()
            if table in dialog.role_tables.values() and index.column() == 1:
                opt = QStyleOptionViewItem(option)
                delegate.initStyleOption(opt, index)
                layout = delegate._layout_text(
                    delegate._normalize_text(opt.text), opt.font,
                    option.rect.width() - 2 * delegate.HORIZONTAL_PADDING)
                painted.append(dict(
                    table=table.objectName(), text=opt.text, width=option.rect.width(),
                    actual_height=option.rect.height(),
                    required_height=layout.boundingRect().height() + 2 * delegate.VERTICAL_PADDING,
                    laid_out_characters=sum(layout.lineAt(i).textLength()
                                            for i in range(layout.lineCount())),
                    lines=layout.lineCount()))
            return original_paint(delegate, painter, option, index)

        def check_visible_rows(role):
            table, model = dialog.role_tables[role], dialog.role_models[role]
            self.assertTrue(table.wordWrap())
            self.assertEqual(table.textElideMode(), Qt.ElideNone)
            self.assertIsInstance(table.itemDelegateForColumn(1), WrappedFilenameDelegate)
            for row, entry in enumerate(dialog._role_entries[role]):
                expected = Path(entry.source.source).name
                self.assertEqual(model.index(row, 1).data(Qt.DisplayRole), expected)
            start = len(painted)
            table.viewport().repaint()
            dialog.grab()
            observed = [row for row in painted[start:] if row["table"] == table.objectName()]
            self.assertTrue(observed, f"No actual filename paint for {role}")
            for row in observed:
                self.assertEqual(row["laid_out_characters"], len(row["text"]))
                self.assertGreaterEqual(row["actual_height"], row["required_height"], row)

        with patch.object(WrappedFilenameDelegate, "paint", observe_paint):
            dialog.resize(1180, 850)
            dialog.show()
            self.app.processEvents()
            self.assertIs(dialog.role_tabs.currentWidget(), dialog.role_tables["P"])
            self.assertFalse(dialog.role_tables["N"].isVisibleTo(dialog))
            check_visible_rows("P")
            # N is still hidden when the visible table becomes narrow. Its
            # first paint must use its current width, not a hidden-page cache.
            for width in (850, 1180, 850):
                dialog.resize(width, 850)
                self.app.processEvents()
                for role in ("P", "N", "P"):
                    dialog.role_tabs.setCurrentWidget(dialog.role_tables[role])
                    self.app.processEvents()
                    check_visible_rows(role)
            for role, repeat in (("P", "001"), ("N", "002")):
                table = dialog.role_tables[role]
                dialog.role_tabs.setCurrentWidget(table)
                row = next(i for i, entry in enumerate(dialog._role_entries[role])
                           if entry.source.source == names[role, repeat])
                index = dialog.role_models[role].index(row, 1)
                table.scrollTo(index)
                self.app.processEvents()
                QTest.mouseClick(table.viewport(), Qt.LeftButton, pos=table.visualRect(index).center())
                self.wait(dialog)
                label = dialog.source_labels[role]
                self.assertEqual(label.text().replace("\u200b", ""), names[role, repeat])
                self.assertEqual(label.accessibleName(), names[role, repeat])
                self.assertEqual(label.textFormat(), Qt.PlainText)
                self.assertEqual(label.toolTip(), str((self.root / names[role, repeat]).resolve()))
                self.assertEqual(dialog._p_files if role == "P" else dialog._n_files,
                                 (names[role, repeat],))
                self.app.processEvents()
                check_visible_rows(role)
        self.assertEqual(dialog.pair, DrrPnPair((names["P", "001"],), (names["N", "002"],)))
        self.assertTrue(any(row["lines"] > 1 for row in painted))
        self.assertGreater(len({row["width"] for row in painted}), 1)
        self.assertIsNone(dialog.selected_version)
        self.assertFalse(self.store.directory.exists())


if __name__ == "__main__":
    unittest.main()
