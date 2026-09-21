import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import imread
from PySide6.QtWidgets import QApplication


class AsyncFigureSaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_snapshot_matches_reference_and_survives_live_changes(self):
        from ui_qt.async_figure_save import save_figure_async
        with tempfile.TemporaryDirectory() as folder:
            fig = Figure(figsize=(3, 2), dpi=90)
            ax = fig.subplots()
            ax.plot([0, 1], [2, 3]); ax.set_title('Frozen title')
            reference = Path(folder) / 'reference.png'
            fig.savefig(reference, dpi=120)
            target = Path(folder) / 'target.png'
            job = save_figure_async(fig, target, dpi=120)
            fig.clear()
            deadline = time.monotonic() + 10
            while not job.done and time.monotonic() < deadline:
                self.app.processEvents(); time.sleep(.005)
            self.assertTrue(job.done)
            self.assertIsNone(job.error)
            np.testing.assert_array_equal(imread(reference), imread(target))

    def test_failure_preserves_existing_destination(self):
        from ui_qt.async_figure_save import save_figure_async
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'existing.png'; path.write_bytes(b'original')
            with patch('ui_qt.async_figure_save.render_snapshot', side_effect=RuntimeError('failed')):
                job = save_figure_async(Figure(), path)
                deadline = time.monotonic() + 10
                while not job.done and time.monotonic() < deadline:
                    self.app.processEvents(); time.sleep(.005)
            self.assertTrue(job.error)
            self.assertEqual(path.read_bytes(), b'original')
