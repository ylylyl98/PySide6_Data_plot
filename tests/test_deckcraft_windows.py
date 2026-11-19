import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from ui_qt.quick.deckcraft_windows import DeckCraftWindow, DeckCraftWindows, _MouseClicks
from ui_qt.quick.powerpoint_mouse import DoubleClickDetector


MODULE = "ui_qt.quick.deckcraft_windows"


class DeckCraftWindowsTests(unittest.TestCase):
    def setUp(self):
        self.desktop = object.__new__(DeckCraftWindows)
        self.desktop._gui = Mock()
        self.desktop._process = Mock()
        self.desktop._process.GetWindowThreadProcessId.return_value = (10, os.getpid())
        self.desktop._discovery_lock = threading.Lock()
        self.desktop._windows = {}
        self.desktop._pending = {}
        self.desktop._children = {}
        self.desktop._recent_executable = None

    def test_multiple_control_windows_match_their_owning_ports(self):
        executable = Path("deckcraft.exe")
        frames = [(11, 111, executable), (22, 222, executable), (33, 333, executable),
                  (44, 444, executable), (45, 444, executable), (55, 555, executable)]
        ports = {111: {7979}, 222: {12345}, 444: {12346}, 555: {12347, 12348}, 999: {12349}}
        with patch(f"{MODULE}._deckcraft_frames", return_value=frames), patch(
            f"{MODULE}._loopback_ports", return_value=ports,
        ):
            first = DeckCraftWindow(11, 111, 7979)
            second = DeckCraftWindow(22, 222, 12345)
            self.assertEqual(self.desktop.connections(), (first, second))
            self.assertEqual(self.desktop._windows, {11: first, 22: second})

    def test_click_must_hit_the_current_window_client_with_the_expected_owner(self):
        window = DeckCraftWindow(11, 111, 7979)
        self.desktop._windows = {11: window}
        gui = self.desktop._gui
        gui.GetForegroundWindow.return_value = 11
        gui.WindowFromPoint.return_value = 12
        gui.GetAncestor.return_value = 11
        gui.ScreenToClient.return_value = (80, 35)
        gui.GetClientRect.return_value = (0, 0, 100, 50)
        self.desktop._process.GetWindowThreadProcessId.return_value = (10, 111)
        self.assertEqual(self.desktop._hit(180, 135), (window, 0.8, 0.7))
        gui.ScreenToClient.return_value = (80, -5)
        self.assertIsNone(self.desktop._hit(180, 95))
        gui.ScreenToClient.return_value = (80, 35)
        gui.GetAncestor.return_value = 22
        self.assertIsNone(self.desktop._hit(180, 135))
        gui.GetAncestor.return_value = 11
        self.desktop._process.GetWindowThreadProcessId.return_value = (10, 222)
        self.assertIsNone(self.desktop._hit(180, 135))

    def test_activation_only_targets_exact_current_window(self):
        window = DeckCraftWindow(11, 111, 7979)
        recycled = DeckCraftWindow(11, 222, 7979)
        gui = self.desktop._gui
        self.desktop.connections = Mock(return_value=(recycled,))
        self.assertFalse(self.desktop.activate(window))
        gui.SetForegroundWindow.assert_not_called()
        self.desktop.connections.return_value = (window,)
        gui.IsIconic.return_value = True
        gui.GetForegroundWindow.side_effect = (22, 22, 11)
        self.assertTrue(self.desktop.activate(window))
        gui.ShowWindow.assert_called_once_with(11, 9)
        gui.SetForegroundWindow.assert_called_once_with(11)

    def test_activation_waits_for_foreground_after_a_win32_error(self):
        window = DeckCraftWindow(11, 111, 7979)
        self.desktop.connections = Mock(return_value=(window,))
        gui = self.desktop._gui
        gui.error = OSError
        gui.IsIconic.return_value = False
        gui.SetForegroundWindow.side_effect = OSError
        gui.GetForegroundWindow.side_effect = (22, 22, 22, 11)
        with patch(f"{MODULE}.sleep") as pause:
            self.assertTrue(self.desktop.activate(window))
        pause.assert_called_once()
        gui.SetForegroundWindow.assert_called_once_with(11)

    def test_activation_reports_when_foreground_stays_elsewhere(self):
        window = DeckCraftWindow(11, 111, 7979)
        self.desktop.connections = Mock(return_value=(window,))
        gui = self.desktop._gui
        gui.IsIconic.return_value = False
        gui.GetForegroundWindow.return_value = 22
        with patch(f"{MODULE}.monotonic", side_effect=(1, 1, 1.4)), patch(f"{MODULE}.sleep") as pause:
            self.assertFalse(self.desktop.activate(window))
        pause.assert_called_once()
        gui.SetForegroundWindow.assert_called_once_with(11)

    def test_activation_cannot_take_focus_back_from_an_unrelated_process(self):
        window = DeckCraftWindow(11, 111, 7979)
        self.desktop.connections = Mock(return_value=(window,))
        gui = self.desktop._gui
        gui.GetForegroundWindow.return_value = 22
        self.desktop._process.GetWindowThreadProcessId.return_value = (12, os.getpid() + 1)
        self.assertFalse(self.desktop.activate(window))
        gui.ShowWindow.assert_not_called()
        gui.SetForegroundWindow.assert_not_called()

    def test_plot_activation_detaches_input_after_verified_transfer(self):
        origin = DeckCraftWindow(11, 111, 7979)
        self.desktop._windows = {11: origin}
        gui = self.desktop._gui
        gui.error = OSError
        gui.GetAncestor.return_value = 22
        gui.GetForegroundWindow.side_effect = (11, 11, 11, 11, 22)
        gui.SetForegroundWindow.side_effect = (OSError, None)
        gui.IsIconic.return_value = True
        target_thread = threading.get_native_id()
        self.desktop._process.GetWindowThreadProcessId.side_effect = ((target_thread, os.getpid()), (33, 111))
        self.assertTrue(self.desktop.activate_plot_window(origin, 22))
        gui.ShowWindow.assert_called_once_with(22, 9)
        self.assertEqual(self.desktop._process.AttachThreadInput.call_args_list,
                         [((target_thread, 33, True), {}), ((target_thread, 33, False), {})])

    def test_plot_activation_rechecks_foreground_before_using_attached_input(self):
        origin = DeckCraftWindow(11, 111, 7979)
        self.desktop._windows = {11: origin}
        gui = self.desktop._gui
        gui.error = OSError
        gui.GetAncestor.return_value = 22
        gui.GetForegroundWindow.side_effect = (11, 11, 11, 44, 44)
        gui.SetForegroundWindow.side_effect = OSError
        gui.IsIconic.return_value = False
        target_thread = threading.get_native_id()
        self.desktop._process.GetWindowThreadProcessId.side_effect = ((target_thread, os.getpid()), (33, 111))
        self.assertFalse(self.desktop.activate_plot_window(origin, 22))
        gui.SetForegroundWindow.assert_called_once_with(22)
        self.desktop._process.AttachThreadInput.assert_called_with(target_thread, 33, False)

    def test_plot_activation_requires_the_origin_to_still_be_foreground(self):
        origin = DeckCraftWindow(11, 111, 7979)
        self.desktop._windows = {11: origin}
        gui = self.desktop._gui
        gui.GetAncestor.return_value = 22
        gui.GetForegroundWindow.return_value = 44
        self.desktop._process.GetWindowThreadProcessId.return_value = (threading.get_native_id(), os.getpid())
        self.assertFalse(self.desktop.activate_plot_window(origin, 22))
        gui.ShowWindow.assert_not_called()
        gui.SetForegroundWindow.assert_not_called()
        self.desktop._process.AttachThreadInput.assert_not_called()

    def test_launch_opens_the_selected_file_once_on_a_dedicated_port(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            executable = directory / "deckcraft.exe"
            path = directory / "a file.pptx"
            executable.touch()
            path.touch()
            process = Mock(pid=52)
            process.poll.return_value = None
            frames = [(11, 52, executable), (12, 53, executable)]
            with patch(f"{MODULE}._deckcraft_frames", side_effect=([], frames, frames)), patch(
                f"{MODULE}._loopback_ports", side_effect=({52: {7777}, 53: {22222}}, {52: {22222}}),
            ), patch(f"{MODULE}.sleep"), patch(
                f"{MODULE}._installation_candidates", return_value=iter((executable,)),
            ), patch(f"{MODULE}._free_port", return_value=22222), patch(
                f"{MODULE}.subprocess.Popen", return_value=process,
            ) as spawn:
                self.assertEqual(self.desktop.launch(path), DeckCraftWindow(11, 52, 22222))
                spawn.assert_called_once_with(
                    [str(executable.resolve()), "--control", "22222", str(path.resolve())], cwd=directory.resolve(),
                )

    def test_launch_reports_early_exit_and_missing_control_connection(self):
        process = Mock(pid=52)
        process.poll.return_value = 3
        with self.assertRaisesRegex(RuntimeError, "exit code 3"):
            self.desktop._wait_for_window(None, process, 22222)
        process.poll.return_value = None
        self.desktop.connections = Mock(return_value=())
        with patch(f"{MODULE}.monotonic", side_effect=(0, 0, 9)), patch(f"{MODULE}.sleep"):
            with self.assertRaisesRegex(RuntimeError, "did not open a controllable window"):
                self.desktop._wait_for_window(None, process, 22222)

    def test_launch_rejects_the_same_uncontrolled_document(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a file.pptx"
            path.touch()
            with patch(f"{MODULE}._deckcraft_frames", return_value=[(11, 111, Path("deckcraft.exe"))]), patch(
                f"{MODULE}._loopback_ports", return_value={},
            ), patch(f"{MODULE}._process_arguments", return_value=["source startup command"]) as commands, patch(
                f"{MODULE}._windows_argv", return_value=("deckcraft.exe", str(path)),
            ), patch(f"{MODULE}.subprocess.Popen") as spawn:
                with self.assertRaisesRegex(ValueError, "already open"):
                    self.desktop.launch(path)
                commands.assert_called_once_with({111})
                spawn.assert_not_called()


class MouseClickTests(unittest.TestCase):
    def test_emits_at_second_down_only_after_a_same_window_first_click(self):
        first = DeckCraftWindow(11, 111, 7979)
        second = DeckCraftWindow(22, 222, 23456)
        hit = Mock(return_value=(first, .1, .2))
        callback = Mock()
        clicks = _MouseClicks(DoubleClickDetector(500, 8, 8), hit, callback)

        clicks.receive(True, 100, 100, 200)
        clicks.receive(False, 120, 100, 200)
        hit.return_value = (second, .3, .4)
        clicks.receive(True, 200, 100, 200)
        clicks.receive(False, 220, 100, 200)
        callback.assert_not_called()
        clicks.receive(True, 300, 101, 200)
        callback.assert_called_once_with(second, .3, .4)
        clicks.receive(False, 320, 101, 200)
        clicks.receive(True, 400, 100, 200)
        callback.assert_called_once()

    def test_first_drag_or_outside_click_cannot_trigger_a_double_click(self):
        window = DeckCraftWindow(11, 111, 7979)
        hit = Mock(return_value=(window, .1, .2))
        callback = Mock()
        clicks = _MouseClicks(DoubleClickDetector(500, 8, 8), hit, callback)
        clicks.receive(True, 100, 100, 200)
        clicks.receive(False, 120, 120, 200)
        clicks.receive(True, 200, 120, 200)
        clicks.receive(False, 220, 120, 200)
        hit.return_value = None
        clicks.receive(True, 300, 120, 200)
        clicks.receive(False, 320, 120, 200)
        hit.return_value = (window, .1, .2)
        clicks.receive(True, 400, 120, 200)
        callback.assert_not_called()

    def test_geometry_seed_is_available_at_first_completed_click_only(self):
        window = DeckCraftWindow(11, 111, 7979)
        hit = Mock(return_value=(window, .1, .2))
        double_click = Mock()
        first_click = Mock()
        clicks = _MouseClicks(DoubleClickDetector(500, 8, 8), hit, double_click, first_click)
        clicks.receive(True, 100, 100, 200)
        clicks.receive(False, 120, 130, 200)
        clicks.receive(True, 200, 100, 200)
        first_click.assert_not_called()
        clicks.receive(False, 220, 100, 200)
        first_click.assert_called_once_with(window, .1, .2)
        double_click.assert_not_called()
        clicks.receive(True, 300, 101, 200)
        clicks.receive(False, 320, 100, 200)
        double_click.assert_called_once_with(window, .1, .2)
        first_click.assert_called_once()
