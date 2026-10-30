"""MCD source picker drafts must not change the loaded measurement."""
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PySide6.QtCore import QCoreApplication, QSettings, QUrl
from ui_qt.quick.mcd_bridge import McdBridge
from ui_qt.quick.mcd_peak_bridge import McdPeakBridge


class McdSourceSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "experiment"
        (self.folder / "mcd").mkdir(parents=True)
        self.old = self.folder / "mcd" / "old.csv"
        self.new = self.folder / "mcd" / "a long measurement name.csv"
        self.external = self.root / "external.csv"
        for path in (self.old, self.new, self.external):
            path.write_text("stub\n", encoding="utf-8")

    def bridge(self, cls):
        settings = QSettings(str(self.root / (cls.__name__ + ".ini")), QSettings.Format.IniFormat)
        bridge = cls(settings=settings)
        bridge._folder = str(self.folder.resolve())
        bridge._source = str(self.old.resolve())
        self.addCleanup(bridge.close)
        return bridge

    def wait_catalog(self, bridge):
        deadline = time.monotonic() + 10
        while bridge._catalog_busy and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertFalse(bridge._catalog_busy)
        self.assertFalse(bridge._error, bridge._error)

    def test_draft_cancel_and_accept(self):
        for cls in (McdBridge, McdPeakBridge):
            with self.subTest(bridge=cls.__name__):
                bridge = self.bridge(cls)
                dialogs = []
                bridge.sourceDialogRequested.connect(lambda *args: dialogs.append(args))
                with patch.object(bridge, "_queue_load") as load:
                    bridge.invoke("select-files")
                    self.assertEqual(bridge.viewState["sheet"], "sources")
                    self.assertEqual(bridge.viewState["sourceDraft"], str(self.old.resolve()))
                    self.assertEqual(dialogs, [])
                    bridge.setSourceDraft(str(self.new))
                    self.assertEqual(bridge._source, str(self.old.resolve()))
                    load.assert_not_called()
                    bridge.invoke("close-sheet")
                    self.assertEqual(bridge._source, str(self.old.resolve()))
                    bridge.invoke("select-files")
                    self.assertEqual(bridge._source_draft, str(self.old.resolve()))
                    bridge.setSourceDraft(str(self.new))
                    bridge.invoke("accept-source")
                    self.assertEqual(bridge._source, str(self.new.resolve()))
                    self.assertEqual(bridge._sheet, "")
                    load.assert_called_once_with(immediate=True)
                    self.wait_catalog(bridge)

    def test_external_browse_remains_draft_until_accepted(self):
        for cls in (McdBridge, McdPeakBridge):
            with self.subTest(bridge=cls.__name__):
                bridge = self.bridge(cls)
                dialogs = []
                bridge.sourceDialogRequested.connect(lambda *args: dialogs.append(args))
                with patch.object(bridge, "_queue_load") as load:
                    bridge.invoke("select-files")
                    bridge.invoke("browse-sources")
                    self.assertEqual(dialogs, [("catalog-source", False)])
                    bridge.openFiles("catalog-source", [QUrl.fromLocalFile(str(self.external))])
                    self.wait_catalog(bridge)
                    self.assertEqual(bridge._source_draft, str(self.external.resolve()))
                    self.assertEqual(bridge._source, str(self.old.resolve()))
                    self.assertIn(str(self.external.resolve()), [row["path"] for row in bridge.viewState["catalog"]])
                    load.assert_not_called()
                    bridge.invoke("accept-source")
                    self.assertEqual(bridge._source, str(self.external.resolve()))
                    load.assert_called_once_with(immediate=True)
                    self.wait_catalog(bridge)

    def test_empty_invalid_and_multiple_sources_never_load(self):
        for cls in (McdBridge, McdPeakBridge):
            with self.subTest(bridge=cls.__name__):
                bridge = self.bridge(cls)
                with patch.object(bridge, "_queue_load") as load:
                    bridge.invoke("select-files")
                    bridge.setSourceDraft("")
                    bridge.invoke("accept-source")
                    bridge.openFiles("catalog-source", [])
                    bridge.setSourceDraft(str(self.root / "missing.csv"))
                    bridge.openFiles("catalog-source", [str(self.new), str(self.external)])
                    bridge.openFiles("sources", [str(self.new), str(self.external)])
                    self.assertEqual(bridge._source, str(self.old.resolve()))
                    self.assertEqual(bridge._source_draft, "")
                    load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
