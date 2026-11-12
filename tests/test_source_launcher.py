"""Exercise real Git histories and Windows shell/instance behavior in isolation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import source_launcher as launcher
from windows_source import INSTALLED_APP_ID, source_app_id

ROOT = Path(__file__).resolve().parents[1]


class SourceUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="DPTK source space 中文 ")
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.remote = base / "remote repository.git"
        self.seed = base / "author 中文"
        self.checkout = base / "source checkout 中文"
        self.call(base, "init", "--bare", str(self.remote))
        self.call(base, "init", "-b", "main", str(self.seed))
        self.configure(self.seed)
        (self.seed / "app.txt").write_text("original", encoding="utf-8")
        (self.seed / ".gitignore").write_text("ignored-local.txt\n", encoding="utf-8")
        self.call(self.seed, "add", "app.txt", ".gitignore")
        self.call(self.seed, "commit", "-m", "Initial")
        self.call(self.seed, "remote", "add", "team", str(self.remote))
        self.call(self.seed, "push", "-u", "team", "main")
        self.call(base, "clone", "--branch", "main", "--origin", "team", str(self.remote), str(self.checkout))
        self.configure(self.checkout)
        self.before = self.call(self.checkout, "rev-parse", "HEAD")

    def call(self, root, *args):
        process = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                 text=True, encoding="utf-8", timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)
        return process.stdout.strip()

    def configure(self, root):
        self.call(root, "config", "user.email", "launcher-tests@example.invalid")
        self.call(root, "config", "user.name", "Launcher Tests")

    def upstream_commit(self, name="app.txt", content="updated"):
        (self.seed / name).write_text(content, encoding="utf-8")
        self.call(self.seed, "add", "-f", name)
        self.call(self.seed, "commit", "-m", "Remote change")
        self.call(self.seed, "push", "team", "main")

    def assert_preserved(self):
        result = launcher.safe_update(self.checkout)
        self.assertTrue(result.skipped, result)
        self.assertEqual(self.call(self.checkout, "rev-parse", "HEAD"), self.before)
        return result

    def test_clean_fast_forward_uses_non_origin_remote_and_unicode_path(self):
        self.upstream_commit()
        result = launcher.safe_update(self.checkout)
        self.assertFalse(result.skipped, result)
        self.assertEqual((self.checkout / "app.txt").read_text(), "updated")
        self.assertEqual(self.call(self.checkout, "status", "--porcelain"), "")
        self.assertEqual(self.call(self.checkout, "rev-parse", "HEAD"), self.call(self.seed, "rev-parse", "HEAD"))

    def test_up_to_date(self):
        self.assertIn("up to date", launcher.safe_update(self.checkout).message)

    def test_unstaged_staged_and_untracked_are_preserved(self):
        self.upstream_commit()
        for mode in ("unstaged", "staged", "untracked"):
            with self.subTest(mode=mode):
                path = self.checkout / ("new 中文.txt" if mode == "untracked" else "app.txt")
                path.write_text("local", encoding="utf-8")
                if mode == "staged":
                    self.call(self.checkout, "add", "app.txt")
                self.assert_preserved()
                self.assertEqual(path.read_text(), "local")
                if mode != "untracked":
                    self.call(self.checkout, "restore", "--staged", "--worktree", "app.txt")

    def test_wrong_branch(self):
        self.call(self.checkout, "checkout", "-b", "experiment")
        self.assertIn("not main", self.assert_preserved().message)
        self.assertEqual(self.call(self.checkout, "branch", "--show-current"), "experiment")

    def test_detached_head(self):
        self.call(self.checkout, "checkout", "--detach")
        self.assert_preserved()

    def test_no_upstream(self):
        self.call(self.checkout, "branch", "--unset-upstream")
        self.assert_preserved()

    def test_wrong_upstream_branch(self):
        self.call(self.checkout, "config", "branch.main.merge", "refs/heads/experiment")
        self.assertIn("track main", self.assert_preserved().message)

    def test_local_commits_and_divergence(self):
        (self.checkout / "local.txt").write_text("local", encoding="utf-8")
        self.call(self.checkout, "add", "local.txt")
        self.call(self.checkout, "commit", "-m", "Local")
        self.before = self.call(self.checkout, "rev-parse", "HEAD")
        self.assertIn("unpushed", self.assert_preserved().message)
        self.upstream_commit()
        self.assertIn("diverged", self.assert_preserved().message)

    def test_failed_fetch_does_not_merge_cached_tracking_ref(self):
        self.upstream_commit()
        self.call(self.checkout, "fetch", "team")
        self.call(self.checkout, "remote", "set-url", "team", str(self.remote / "missing"))
        self.assert_preserved()
        self.assertEqual((self.checkout / "app.txt").read_text(), "original")

    def test_git_operation_in_progress(self):
        self.upstream_commit()
        (self.checkout / ".git/rebase-merge").mkdir()
        self.assertIn("in progress", self.assert_preserved().message)

    def test_ignored_file_collision_is_not_overwritten(self):
        self.upstream_commit("ignored-local.txt", "remote")
        (self.checkout / "ignored-local.txt").write_text("precious local", encoding="utf-8")
        self.assert_preserved()
        self.assertEqual((self.checkout / "ignored-local.txt").read_text(), "precious local")

    def test_change_during_fetch_is_rechecked(self):
        self.upstream_commit()
        real_git = launcher.git

        def concurrent_change(root, *args, **kwargs):
            output = real_git(root, *args, **kwargs)
            if args[0] == "fetch":
                (root / "app.txt").write_text("edited during fetch", encoding="utf-8")
            return output

        with patch.object(launcher, "git", side_effect=concurrent_change):
            self.assert_preserved()
        self.assertEqual((self.checkout / "app.txt").read_text(), "edited during fetch")

    def test_fetch_timeout_is_a_skip(self):
        with patch.object(launcher.subprocess, "run", side_effect=subprocess.TimeoutExpired("git", 45)):
            self.assertIn("timed out", launcher.safe_update(self.checkout).message)


class LaunchFlowTests(unittest.TestCase):
    def test_pre_lock_main_window_is_activated_without_update_or_duplicate(self):
        with patch("windows_source.acquire_source_instance", return_value=True), \
             patch("windows_source.running_source_processes", return_value=[{"ProcessId": 123, "entry": "run_qt.py"}]), \
             patch("windows_source.activate_existing_processes", return_value=True) as activate, \
             patch.object(launcher, "guarded_update") as update, \
             patch.object(launcher.runpy, "run_path") as run, patch.object(sys, "argv", ["launch_data_plot.pyw"]):
            self.assertEqual(launcher.main(), 0)
        activate.assert_called_once()
        update.assert_not_called()
        run.assert_not_called()

    def test_running_instance_returns_before_git_dependencies_or_application(self):
        with patch("windows_source.acquire_source_instance", return_value=False), \
             patch.object(launcher, "safe_update") as update, \
             patch.object(launcher, "ensure_runtime") as runtime, \
             patch.object(launcher.runpy, "run_path") as run:
            self.assertEqual(launcher.main(), 0)
        update.assert_not_called()
        runtime.assert_not_called()
        run.assert_not_called()

    def test_skipped_update_still_launches_local_and_reports_reason(self):
        with patch("windows_source.acquire_source_instance", return_value=True), \
             patch("windows_source.running_source_processes", return_value=[]), \
             patch.object(launcher, "guarded_update", return_value=launcher.UpdateResult("Network failed.", True)), \
             patch.object(launcher, "ensure_runtime"), patch.object(launcher, "notify") as notify, \
             patch.object(launcher.runpy, "run_path") as run, patch.object(sys, "argv", ["launch_data_plot.pyw"]):
            self.assertEqual(launcher.main(), 0)
            self.assertIn("NOT updated", notify.call_args.args[0])
            run.assert_called_once_with(str(ROOT / "run_qt.py"), run_name="__main__")


@unittest.skipUnless(sys.platform == "win32", "Windows shell APIs")
class WindowsSourceTests(unittest.TestCase):
    def test_occupied_code_or_failed_process_inspection_skips_git(self):
        from windows_source import CodeLease

        with TemporaryDirectory(prefix="DPTK update guard 中文 ") as directory:
            root = Path(directory)
            with patch.object(launcher, "ROOT", root), patch.object(launcher, "safe_update") as update:
                with CodeLease(root):
                    self.assertTrue(launcher.guarded_update().skipped)
                with patch("windows_source.running_source_processes", side_effect=RuntimeError("Access denied")):
                    self.assertTrue(launcher.guarded_update().skipped)
                update.assert_not_called()

    def test_shared_code_leases_prevent_update_until_last_reader_exits(self):
        from windows_source import CodeLease

        with TemporaryDirectory(prefix="DPTK readers 中文 ") as directory:
            root = Path(directory)
            first, second = CodeLease(root), CodeLease(root)
            try:
                with CodeLease(root, exclusive=True) as update:
                    self.assertFalse(update.locked)
                first.close()
                with CodeLease(root, exclusive=True) as update:
                    self.assertFalse(update.locked)
            finally:
                first.close()
                second.close()
            with CodeLease(root, exclusive=True) as update:
                self.assertTrue(update.locked)

    def test_pre_lock_analysis_process_is_detected_in_unicode_path(self):
        from windows_source import running_source_processes

        with TemporaryDirectory(prefix="DPTK legacy source 中文 ") as directory:
            root = Path(directory)
            entry = root / "run_drr_analysis.py"
            entry.write_text('import os,sys\nprint(os.getpid(), flush=True)\nsys.stdin.readline()\n', encoding="utf-8")
            process = subprocess.Popen([sys.executable, str(entry)], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, text=True)
            try:
                actual_pid = int(process.stdout.readline())
                matches = running_source_processes(root)
                self.assertIn(actual_pid, {row["ProcessId"] for row in matches})
                self.assertTrue(all(row["entry"] == entry.name for row in matches))
                self.assertEqual(running_source_processes(root / "different folder"), [])
            finally:
                process.communicate("\n", timeout=10)

    def test_batch_shortcut_and_copied_pin_launch_with_spaces_and_chinese(self):
        """Execute the real batch, .lnk and relaunch command in a fixture checkout.

        Only the app body and Desktop location are substituted; Git, Windows
        command-line quoting, pythonw, shortcut COM and the updater are real.
        """
        import pythoncom
        from win32com.propsys import propsys, pscon
        from win32com.shell import shell

        fixture = SourceUpdateTests(methodName="test_up_to_date")
        fixture.setUp()
        try:
            for name in ("Data_Plot_App.bat", "windows_source.py", "launch_data_plot.pyw", "scripts/source_launcher.py", "assets/icons/app_icon.ico"):
                target = fixture.seed / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            # Redirect this fixture's Desktop to itself; never alter the user's
            # Desktop or real pinned shortcuts during automated tests.
            entry = fixture.seed / "launch_data_plot.pyw"
            content = entry.read_text(encoding="utf-8").replace(
                'from scripts.source_launcher import main',
                'from win32com.shell import shell\nshell.SHGetFolderPath = lambda *args: str(PROJECT_ROOT)\nfrom scripts.source_launcher import main')
            entry.write_text(content, encoding="utf-8")
            (fixture.seed / "run_qt.py").write_text(
                'import json, os\nfrom pathlib import Path\n'
                'Path("receipt.json").write_text(json.dumps({"cwd":os.getcwd(), "version":Path("app.txt").read_text()}), encoding="utf-8")\n',
                encoding="utf-8")
            (fixture.seed / "requirements.txt").write_text("", encoding="utf-8")
            with (fixture.seed / ".gitignore").open("a", encoding="utf-8") as stream:
                stream.write(".venv/\n__pycache__/\n*.lnk\nreceipt.json\n")
            fixture.call(fixture.seed, "add", ".")
            fixture.call(fixture.seed, "commit", "-m", "Source launch fixture")
            fixture.call(fixture.seed, "push", "team", "main")
            fixture.call(fixture.checkout, "pull", "--ff-only")
            root = fixture.checkout
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root / ".venv")], check=True, capture_output=True, timeout=30)
            # Reuse installed test dependencies through .pth, without linking or
            # copying a live environment and without network installation.
            paths = [p for p in sys.path if "site-packages" in p.lower()]
            (root / ".venv/Lib/site-packages/fixture.pth").write_text(
                "\n".join(paths) + "\nimport pywin32_bootstrap\n", encoding="utf-8")

            def wait_receipt(expected):
                receipt = root / "receipt.json"
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if receipt.exists():
                        value = json.loads(receipt.read_text(encoding="utf-8"))
                        self.assertTrue(Path(value["cwd"]).samefile(root))
                        self.assertEqual(value["version"], expected)
                        receipt.unlink()
                        return
                    time.sleep(.1)
                self.fail("The Windows source entry did not launch the fixture app")

            fixture.upstream_commit(content="batch updated")
            env = dict(os.environ)
            env.pop("DPTK_DEBUG_CONSOLE", None)
            subprocess.run(["cmd.exe", "/d", "/c", str(root / "Data_Plot_App.bat")],
                           cwd=root.parent, env=env, check=True, capture_output=True, timeout=15)
            wait_receipt("batch updated")
            shortcut = root / "DPTK Source.lnk"
            self.assertTrue(shortcut.exists())
            fixture.upstream_commit(content="shortcut updated")
            os.startfile(shortcut)
            wait_receipt("shortcut updated")
            copied = root / "Pinned Source.lnk"
            shutil.copyfile(shortcut, copied)
            fixture.upstream_commit(content="copied pin updated")
            os.startfile(copied)
            wait_receipt("copied pin updated")
            link = pythoncom.CoCreateInstance(shell.CLSID_ShellLink, None,
                pythoncom.CLSCTX_INPROC_SERVER, shell.IID_IShellLink)
            link.QueryInterface(pythoncom.IID_IPersistFile).Load(str(copied))
            command = link.QueryInterface(propsys.IID_IPropertyStore).GetValue(pscon.PKEY_AppUserModel_RelaunchCommand).GetValue()
            fixture.upstream_commit(content="relaunch updated")
            process = subprocess.Popen(command, cwd=root.parent)
            self.assertEqual(process.wait(timeout=30), 0)
            wait_receipt("relaunch updated")
            # Verify updated Python entry code, not just updated data, is loaded.
            fresh_code = (fixture.seed / "run_qt.py").read_text(encoding="utf-8").replace(
                'Path("app.txt").read_text()', '"fresh application code"')
            fixture.upstream_commit("run_qt.py", fresh_code)
            os.startfile(copied)
            wait_receipt("fresh application code")
        finally:
            fixture.doCleanups()

    def test_shortcut_properties_roundtrip_in_unicode_folder(self):
        import pythoncom
        from win32com.propsys import propsys, pscon
        from win32com.shell import shell
        from windows_source import create_shortcut, relaunch_properties

        with TemporaryDirectory(prefix="DPTK source 中文 ") as directory:
            # CI can supply TEMP through an 8.3 alias such as RUNNER~1. Shell
            # links store the canonical path, even when creation used that alias.
            root = Path(directory).resolve()
            for relative in (".venv/Scripts/pythonw.exe", "assets/icons/app_icon.ico", "launch_data_plot.pyw"):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source = Path(sys.executable).with_name("pythonw.exe") if relative.endswith("pythonw.exe") else ROOT / relative
                shutil.copyfile(source, target)
            path = create_shortcut(root / "DPTK Source.lnk", root)
            link = pythoncom.CoCreateInstance(shell.CLSID_ShellLink, None,
                pythoncom.CLSCTX_INPROC_SERVER, shell.IID_IShellLink)
            link.QueryInterface(pythoncom.IID_IPersistFile).Load(str(path))
            self.assertEqual(link.GetPath(4)[0], str(root / ".venv/Scripts/pythonw.exe"))
            self.assertEqual(link.GetArguments(), subprocess.list2cmdline([str(root / "launch_data_plot.pyw")]))
            self.assertEqual(link.GetWorkingDirectory(), str(root))
            self.assertEqual(link.GetIconLocation(), (str(root / "assets/icons/app_icon.ico"), 0))
            store = link.QueryInterface(propsys.IID_IPropertyStore)
            for key, expected in relaunch_properties(root).items():
                self.assertEqual(store.GetValue(key).GetValue(), expected)
            self.assertNotEqual(store.GetValue(pscon.PKEY_AppUserModel_ID).GetValue(), INSTALLED_APP_ID)
            self.assertNotEqual(source_app_id(root), source_app_id(ROOT))

    def test_lifetime_lock_activation_and_crash_recovery(self):
        from windows_source import SourceInstance

        with TemporaryDirectory(prefix="DPTK mutex 中文 ") as directory:
            root = Path(directory)
            code = "from pathlib import Path; from windows_source import SourceInstance; import sys, os; i=SourceInstance(Path(sys.argv[1])); print(i.owner, os.getpid(), flush=True); sys.stdin.readline()"
            process = subprocess.Popen([sys.executable, "-c", code, str(root)], cwd=ROOT,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                owned, child_pid = process.stdout.readline().split()
                self.assertEqual(owned, "True")
                second = SourceInstance(root)
                self.assertFalse(second.owner)
                self.assertTrue(second.activation_requested())
                second.close()
                # Windows venv python.exe is a redirector. Kill the actual owner,
                # not just that redirector (which would orphan the live owner).
                os.kill(int(child_pid), signal.SIGTERM)
                process.wait(timeout=10)
                recovered = SourceInstance(root)
                self.assertTrue(recovered.owner)
                recovered.close()
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
                process.stdin.close()
                process.stdout.close()

    def test_quick_window_pin_properties_and_activation(self):
        code = '''
import ctypes
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtQuick import QQuickWindow
from win32com.propsys import propsys
import windows_source as source
app = QGuiApplication([])
assert source.acquire_source_instance()
window = QQuickWindow()
window.show()
source.attach_source_window(window)
store = propsys.SHGetPropertyStoreForWindow(int(window.winId()), propsys.IID_IPropertyStore)
for key, value in source.relaunch_properties().items():
    assert store.GetValue(key).GetValue() == value
window.showMinimized()
app.processEvents()
source._instance.kernel.SetEvent(source._instance.event)
window._source_activation_timer.timeout.emit()
app.processEvents()
assert not window.windowState() & Qt.WindowState.WindowMinimized
window.close()
print('window metadata and activation passed')
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
            env=dict(os.environ, QT_QPA_PLATFORM="windows"), capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
