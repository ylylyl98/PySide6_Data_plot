import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from PySide6.QtCore import QCoreApplication, QObject
from ui_qt.main_window import MainWindow


class PowerPrevalidatedLoadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def owner(self):
        owner = QObject()
        owner._is_closing = False
        owner._load_in_progress = False
        owner.current_folder = 'experiment'
        owner.power_controller = SimpleNamespace(_power_selected_group_key=lambda: 'pair')
        result = SimpleNamespace(records=(), cube=None, groups={}, group_key='pair')
        owner._power_prevalidated_load = ('experiment', 'pair', result)
        owner._set_stage = Mock()
        owner._on_loaded = Mock()
        owner._on_load_finished = Mock()
        owner.thread_pool = Mock()
        return owner

    def test_validated_load_runs_after_callback_without_second_worker(self):
        owner = self.owner()
        MainWindow._start_load(owner, 'Power Dependent')
        owner._on_loaded.assert_not_called()
        self.assertTrue(owner._load_in_progress)
        self.app.processEvents()
        owner._on_loaded.assert_called_once()
        self.assertEqual(owner._on_loaded.call_args.args[0].power_group_key, 'pair')
        owner._on_load_finished.assert_called_once()
        owner.thread_pool.start.assert_not_called()
        self.assertIsNone(owner._power_prevalidated_load)

    def test_folder_change_before_deferred_handoff_discards_result(self):
        owner = self.owner()
        MainWindow._start_load(owner, 'Power Dependent')
        owner.current_folder = 'other'
        self.app.processEvents()
        owner._on_loaded.assert_not_called()
        owner._on_load_finished.assert_called_once()
