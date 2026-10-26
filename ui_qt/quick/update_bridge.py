"""Native Quick Help/updater state; network and checksums run off the GUI thread."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hmac
from pathlib import Path
import tempfile
from urllib.parse import urlsplit

from PySide6.QtCore import QObject, Property, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices

from app_version import __version__
from core.update_checker import (CheckResult, DownloadResult, check_for_update, download_installer,
                                 expected_installer_name, format_version, sha256_file)


@dataclass(frozen=True)
class UpdateServices:
    check: object = check_for_update
    download: object = download_installer
    checksum: object = sha256_file
    open_url: object = QDesktopServices.openUrl


def _enabled(value):
    return value is None or value is True or str(value).strip().lower() in {"1", "true", "yes", "on"}


def _safe_message(message, fallback="The update check failed."):
    return (str(message or fallback).splitlines() or [fallback])[0][:700]


def _valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


class UpdateBridge(QObject):
    snapshotChanged = Signal()
    statusMessage = Signal(str)
    logMessage = Signal(str)
    _completed = Signal(int, str, object, str)
    _progressed = Signal(int, int)

    def __init__(self, *, settings, services=None, current_version=__version__, download_directory=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.services = services or UpdateServices()
        self.current_version = str(current_version)
        default_downloads = Path.home() / "Downloads"
        self.download_directory = Path(download_directory or (default_downloads if default_downloads.is_dir() else tempfile.gettempdir())).resolve()
        self._automatic = _enabled(settings.value("updates/check_automatically", True))
        self._closed = self._busy = self._dialog = self._indicator = self._about = False
        self._stage, self._message, self._error = "idle", "", ""
        self._pending = self._download = None
        self._progress, self._generation = 0, 0
        self._manual_check = self._launch_consent = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dptk-update")
        self._publish_timer = QTimer(self)
        self._publish_timer.setSingleShot(True)
        self._publish_timer.timeout.connect(self.snapshotChanged)
        self._automatic_timer = QTimer(self)
        self._automatic_timer.setSingleShot(True)
        self._automatic_timer.timeout.connect(lambda: self.check(automatic=True))
        self._completed.connect(self._finish)
        self._progressed.connect(self._set_progress)

    @Property("QVariantMap", notify=snapshotChanged)
    def snapshot(self):
        pending = self._pending
        about = self._about
        stage = "about" if about else self._stage
        latest = format_version(pending.latest_version) if pending and pending.latest_version else ""
        release = pending.release if pending else None
        titles = {"checking": "Check for Updates", "update_available": "Update Available", "up_to_date": "Check for Updates",
            "downloading": "Download Update", "downloaded": "Download Complete", "verifying": "Verify Installer",
            "confirm-install": "Install Update", "launch-verifying": "Verify Installer", "launched": "Installer Started",
            "error": "Update Error", "about": "About DPTK Desktop"}
        return dict(currentVersion=self.current_version, latestVersion=latest, stage=stage,
            title=titles.get(stage, "DPTK Updates"), message="DPTK Desktop — Version " + self.current_version if about else self._message, error="" if about else self._error,
            busy=self._busy and not about, progress=self._progress, automaticChecksEnabled=self._automatic,
            indicatorVisible=self._indicator, indicatorText="Update available: v" + latest if latest else "",
            dialogOpen=self._dialog or about, releaseNotes=release.body[:50000] if release and not about else "",
            releaseUrl=release.html_url if release and not about else "", installerPath=self._download.installer_path if self._download and not about else "",
            canDownload=bool(pending and self._stage == "update_available" and not self._busy and not about),
            canInstall=bool(self._download and self._stage == "downloaded" and not self._busy and not about),
            installConfirmationRequired=self._stage == "confirm-install" and not self._busy and not about,
            canOpenFolder=bool(self._download and self._download.installer_path and not about),
            downloadDirectory=str(self.download_directory))

    def _publish(self):
        if not self._closed and not self._publish_timer.isActive():
            self._publish_timer.start(0)

    def _status(self, message):
        self.statusMessage.emit(message)
        self.logMessage.emit(message)

    def _failure(self, message, *, show=True):
        self._stage = "error"
        self._message = self._error = _safe_message(message)
        self._dialog = bool(show)
        self._launch_consent = False
        if show:
            self._status(self._message)
        else:
            self.logMessage.emit("Automatic update check: " + self._message)
        self._publish()

    @Slot(bool)
    def setAutomatic(self, enabled):
        if self._closed:
            return
        self._automatic = bool(enabled)
        self.settings.setValue("updates/check_automatically", self._automatic)
        if not self._automatic:
            self._automatic_timer.stop()
        self._publish()

    def scheduleAutomaticCheck(self, delay_ms=1500):
        """Explicit startup hook; constructing this bridge never starts network IO."""
        if not self._closed and self._automatic and not self._busy:
            self._automatic_timer.start(max(0, int(delay_ms)))

    def _start(self, kind, work):
        if self._closed or self._busy:
            return False
        self._generation += 1
        token = self._generation
        self._busy, self._progress = True, 0
        future = self._executor.submit(work, lambda value: self._progressed.emit(token, int(value)) if not self._closed else None)
        def done(task):
            try:
                result, error = task.result(), ""
            except Exception as exception:
                result, error = None, _safe_message(exception)
            if not self._closed:
                self._completed.emit(token, kind, result, error)
        future.add_done_callback(done)
        self._publish()
        return True

    def check(self, *, automatic=False):
        if self._closed or automatic and not self._automatic:
            return
        if self._busy:
            if self._stage == "checking" and not automatic:
                self._manual_check = self._dialog = True
                self._publish()
            return
        self._automatic_timer.stop()
        if not automatic:
            self._about = False
        self._manual_check = not automatic
        self._dialog = not automatic
        self._stage, self._message, self._error = "checking", "Checking for updates…", ""
        self._launch_consent = False
        if not automatic:
            self._status(self._message)
        current, check = self.current_version, self.services.check
        self._start("check", lambda progress: check(current))

    @Slot(int, int)
    def _set_progress(self, token, value):
        if not self._closed and self._busy and token == self._generation:
            self._progress = max(0, min(100, value))
            self._publish()

    @Slot(int, str, object, str)
    def _finish(self, token, kind, result, error):
        if self._closed or token != self._generation:
            return
        self._busy = False
        if error:
            if kind == "check":
                self._pending, self._indicator = None, False
            self._failure(error, show=kind != "check" or self._manual_check)
            return
        if kind == "check":
            if not isinstance(result, CheckResult):
                self._pending, self._indicator = None, False
                self._failure("The update server returned an invalid response.", show=self._manual_check)
                return
            if result.status == "error":
                self._pending, self._indicator = None, False
                self._failure(result.message, show=self._manual_check)
                return
            if result.status == "update_available":
                if not result.latest_version or not result.installer_url or not result.sums_url:
                    self._pending, self._indicator = None, False
                    self._failure("This update is missing its installer files.", show=self._manual_check)
                    return
                self._pending, self._indicator = result, True
                self._stage, self._message, self._error = "update_available", _safe_message(result.message, "A new version is available."), ""
            else:
                self._pending, self._indicator = None, False
                self._stage, self._message, self._error = "up_to_date", _safe_message(result.message, "You are running the latest version."), ""
            self._dialog = self._manual_check
        elif kind == "download":
            if (not isinstance(result, DownloadResult) or result.status != "ok" or not result.installer_path
                    or not _valid_digest(result.expected_sha256) or not Path(result.installer_path).is_file()):
                self._download = None
                self._failure(result.message if isinstance(result, DownloadResult) else "Download failed.")
                return
            self._download = result
            self._stage, self._message, self._error = "downloaded", "The update has been downloaded and verified.", ""
            self._progress, self._dialog = 100, True
            self._status(self._message)
        elif kind in {"verify", "launch-verify"}:
            expected = self._download.expected_sha256 if self._download else None
            if not _valid_digest(result) or not expected or not hmac.compare_digest(result.lower(), expected.lower()):
                self._download = None
                self._failure("The installer checksum no longer matches. It will not be launched.")
                return
            if kind == "launch-verify" and self._launch_consent:
                self._launch_consent = False
                try:
                    opened = self.services.open_url(QUrl.fromLocalFile(str(Path(self._download.installer_path).resolve())))
                except Exception:
                    opened = False
                if not opened:
                    self._failure("The installer could not be opened. You can open its folder and try again.")
                    return
                self._stage, self._message, self._error = "launched", "The installer will guide you through the remaining steps.", ""
                self._dialog = self._indicator = False
                self._status(self._message)
            elif kind == "verify":
                if self._dialog:
                    self._stage, self._message = "confirm-install", "Launch the installer now? The installer will guide you through the remaining steps."
                else:
                    self._stage, self._message = "downloaded", "The update has been downloaded and verified."
            else:
                self._stage, self._message = "downloaded", "The update has been downloaded and verified."
        self._publish()

    def _verify(self, *, launch=False):
        if self._closed or self._busy or self._download is None:
            return
        path = Path(self._download.installer_path or "")
        if not path.is_file() or not _valid_digest(self._download.expected_sha256):
            self._download = None
            self._failure("The downloaded installer could not be found or its checksum is unavailable. It will not be launched.")
            return
        self._stage, self._message, self._error = "launch-verifying" if launch else "verifying", "Verifying installer…", ""
        self._status(self._message)
        checksum = self.services.checksum
        self._start("launch-verify" if launch else "verify", lambda progress: checksum(path))

    @Slot(str)
    def invoke(self, action):
        if self._closed:
            return
        if action in {"later", "close-dialog"}:
            if self._about:
                self._about = False
                self._publish()
                return
            self._dialog = self._launch_consent = False
            if self._stage == "confirm-install":
                self._stage, self._message = "downloaded", "The update has been downloaded and verified."
            self._publish()
            return
        if action == "check":
            self.check()
            return
        if action == "show-update":
            if self._pending:
                self._about = False
                self._dialog = True
            self._publish()
            return
        if action == "about":
            self._about = True
            self._publish()
            return
        if self._busy:
            return
        if action == "download" and self._pending and self._stage == "update_available":
            result = self._pending
            self._download = None
            self._stage, self._message, self._error = "downloading", "Downloading update…", ""
            self._dialog = True
            self._status(self._message)
            download, destination = self.services.download, self.download_directory
            self._start("download", lambda progress: download(result.installer_url, result.sums_url,
                expected_installer_name(result.latest_version), destination, progress=progress))
        elif action == "install" and self._stage == "downloaded":
            self._launch_consent = False
            self._verify()
        elif action == "confirm-install" and self._stage == "confirm-install":
            self._launch_consent = True
            # Verify once more after confirmation: a file edited while the
            # prompt was visible must never use the earlier successful hash.
            self._verify(launch=True)
        elif action == "release-notes" and self._pending and self._pending.release:
            url = self._pending.release.html_url
            try:
                parts = urlsplit(url)
            except ValueError:
                return
            if parts.scheme == "https" and parts.netloc == "github.com" and parts.path.startswith("/ylylyl98/PySide6_Data_plot/releases/"):
                self.services.open_url(QUrl(url))
        elif action == "open-folder" and self._download and self._download.installer_path:
            self.services.open_url(QUrl.fromLocalFile(str(Path(self._download.installer_path).parent)))
        self._publish()

    @Slot()
    def close(self):
        if self._closed:
            return
        self._closed = True
        self._automatic_timer.stop()
        self._publish_timer.stop()
        self._launch_consent = False
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.settings.sync()


__all__ = ["UpdateBridge", "UpdateServices"]
