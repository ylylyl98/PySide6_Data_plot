"""Safe Windows source updates. No application modules are imported before Git."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import runpy
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class UpdateResult:
    message: str
    skipped: bool = False


class GitError(RuntimeError):
    pass


def git(root: Path, *args: str, timeout: int = 15) -> str:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never", LC_ALL="C",
               GIT_SSH_COMMAND="ssh -oBatchMode=yes -oConnectTimeout=10")
    try:
        result = subprocess.run(
            ["git", "-c", "core.hooksPath=NUL" if os.name == "nt" else "core.hooksPath=/dev/null",
             "-c", "credential.interactive=false", "-C", str(root), *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitError(f"Git is unavailable or timed out ({type(exc).__name__}).") from exc
    if result.returncode:
        raise GitError(result.stderr.strip() or "Git command failed.")
    return result.stdout.strip()


def unsafe_reason(root: Path) -> str | None:
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root.resolve():
        return "This folder is not the root of its Git checkout."
    if git(root, "symbolic-ref", "--quiet", "--short", "HEAD") != "main":
        return "The current branch is not main."
    for marker in ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge",
                   "rebase-apply", "sequencer", "BISECT_LOG", "index.lock", "HEAD.lock"):
        path = Path(git(root, "rev-parse", "--git-path", marker))
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return "A Git operation is already in progress."
    if git(root, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"):
        return "The checkout has uncommitted changes or untracked files."
    return None


def safe_update(root: Path = ROOT, *, pre_merge_check=None) -> UpdateResult:
    """Only move a clean main forward to the configured remote's fresh main."""
    try:
        reason = unsafe_reason(root)
        if reason:
            return UpdateResult(reason, True)
        remote = git(root, "config", "--get", "branch.main.remote")
        merge = git(root, "config", "--get", "branch.main.merge")
        if remote == "." or remote not in git(root, "remote").splitlines() or merge != "refs/heads/main":
            return UpdateResult("main must track main on a configured remote.", True)
        before = git(root, "rev-parse", "HEAD")
        remote_ref = f"refs/remotes/{remote}/main"
        # Fetch exactly main even when the remote's default fetch refspec differs.
        # A remote-tracking ref can follow a force-push; local history never does.
        git(root, "fetch", "--no-tags", "--no-recurse-submodules", "--", remote,
            f"+refs/heads/main:{remote_ref}", timeout=45)
        target = git(root, "rev-parse", "--verify", remote_ref + "^{commit}")
        ahead, behind = map(int, git(root, "rev-list", "--left-right", "--count", f"HEAD...{target}").split())
        if ahead:
            return UpdateResult("Local main has diverged from remote main." if behind else
                                "Local main has unpushed commits; keeping the local version.", True)
        # A fetch can take time: repeat the checks immediately before any write.
        reason = unsafe_reason(root)
        if reason or git(root, "rev-parse", "HEAD") != before:
            return UpdateResult(reason or "HEAD changed while checking for updates.", True)
        if not behind:
            return UpdateResult("Source is up to date with remote main.")
        if pre_merge_check is not None:
            reason = pre_merge_check()
            if reason:
                return UpdateResult(reason, True)
        # Avoid overwriting even ignored local files; never recurse into submodules.
        git(root, "-c", "submodule.recurse=false", "merge", "--ff-only", "--no-autostash",
            "--no-overwrite-ignore", target, timeout=120)
        return UpdateResult("Source updated successfully from remote main.")
    except GitError as exc:
        return UpdateResult(f"Git update could not be completed: {exc}", True)


def guarded_update() -> UpdateResult:
    from windows_source import CodeLease, running_source_processes

    def check_processes():
        if running_source_processes(ROOT):
            return "Another source application or analysis window is still running."
        return None

    try:
        with CodeLease(ROOT, exclusive=True) as lease:
            if not lease.locked:
                return UpdateResult("Source code is in use by another application or analysis window.", True)
            reason = check_processes()
            if reason:
                return UpdateResult(reason, True)
            return safe_update(ROOT, pre_merge_check=check_processes)
    except Exception:
        return UpdateResult("Running source processes could not be checked safely.", True)


def notify(message: str, *, error: bool = False):
    if sys.stdout is not None:
        print(message, flush=True)
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, message, "DPTK Source", 0x10 if error else 0x40)


def ensure_runtime(root: Path = ROOT):
    """Preserve the existing launcher's missing-dependency repair behavior."""
    python = root / ".venv/Scripts/python.exe"
    probe = "import PySide6, pythoncom, win32com.client, numpy, pandas, scipy, matplotlib, openpyxl, pptx, PIL, imageio_ffmpeg"
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    result = subprocess.run([str(python), "-c", probe], capture_output=True, creationflags=flags)
    if result.returncode:
        notify("Some source dependencies are missing. DPTK will install requirements into this folder's .venv.")
        result = subprocess.run([str(python), "-m", "pip", "install", "-r", str(root / "requirements.txt")],
                                capture_output=True, creationflags=flags)
        if result.returncode:
            raise RuntimeError("Dependency installation failed. Run .venv\\Scripts\\python.exe -m pip install -r requirements.txt in this folder.")


def main() -> int:
    from windows_source import (acquire_source_instance, activate_existing_processes,
                                create_shortcut, running_source_processes)

    # This lock is held by THIS process through runpy and the complete Qt lifetime.
    # A second launch signals activation and returns before touching Git or pip.
    if not acquire_source_instance():
        return 0
    try:
        if "--create-shortcut" in sys.argv:
            ensure_runtime()
            path = create_shortcut()
            notify(f"Created desktop shortcut: {path}\n\nLaunch DPTK Source and pin its running taskbar icon. Remove any old source pin first.")
            return 0
        # Older app versions do not hold the new lock. Preserve and activate
        # their windows too, rather than updating beneath a live old process.
        try:
            existing = running_source_processes(ROOT)
        except Exception:
            existing = []  # guarded_update will fail closed if inspection fails.
        if existing and activate_existing_processes(existing):
            if "--ensure-shortcut" in sys.argv:
                create_shortcut()
            return 0
        result = guarded_update()
        if result.skipped:
            notify("Source was NOT updated.\n\n" + result.message + "\n\nClick OK to launch the local version.")
        ensure_runtime()
        if "--ensure-shortcut" in sys.argv:
            try:
                create_shortcut()
            except Exception:
                notify("The desktop shortcut could not be created. DPTK Source will still start.\n\nCheck Desktop folder access and run Data_Plot_App.bat again.")
        # Load the just-updated entry point without retaining imported app code.
        os.chdir(ROOT)
        sys.argv = [str(ROOT / "run_qt.py"), *[a for a in sys.argv[1:] if a != "--ensure-shortcut"]]
        runpy.run_path(str(ROOT / "run_qt.py"), run_name="__main__")
        return 0
    except Exception:
        log = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "DPTK/logs/source-launcher.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        notify(f"DPTK Source could not start.\n\nSee details: {log}", error=True)
        return 1
