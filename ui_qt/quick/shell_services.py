"""Recent-folder and local-drop behavior shared by the native workbench."""
from pathlib import Path

from PySide6.QtCore import QUrl, Slot


class ShellServicesMixin:
    def _shell_init(self):
        recent = self.settings.value("data/recent_folders", []) or []
        self._recent_folders = [str(value) for value in ([recent] if isinstance(recent, str) else recent)][:8]
        self._remembered_folder = ""

    def _remember_folder(self, folder):
        if not folder or folder == self._remembered_folder:
            return
        path = Path(folder)
        if not path.is_dir():
            return
        self._remembered_folder = str(path)
        self._recent_folders = [str(path), *[value for value in self._recent_folders
                                           if value.casefold() != str(path).casefold()]][:8]
        self.settings.setValue("data/last_folder", str(path))
        self.settings.setValue("data/last_parent_folder", str(path.parent))
        self.settings.setValue("data/recent_folders", self._recent_folders)

    @Slot()
    def restoreSourceFolder(self):
        """Called after the first native frame, only for a persistent session."""
        folder = str(self.settings.value("data/last_folder", "") or "")
        if not self._folder and folder and Path(folder).is_dir():
            self.setSourcePath(folder)

    @Slot("QVariantList", result=bool)
    def dropSources(self, urls):
        if self._closed or self.snapshot["busy"]:
            self._message("Wait for the current operation before dropping sources.")
            return False
        paths = []
        for value in urls:
            url = value if isinstance(value, QUrl) else QUrl(str(value))
            if url.isLocalFile():
                path = Path(url.toLocalFile()).resolve()
                if path.is_dir() or path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".dat"}:
                    paths.append(path)
        paths = list(dict.fromkeys(paths))
        folders = [path for path in paths if path.is_dir()]
        files = [path for path in paths if path.is_file()]
        if len(folders) > 1 or folders and files or len({path.parent for path in files}) > 1:
            self._message("Drop one folder or measurement files from the same folder.")
            return False
        if folders:
            self.setSourcePath(str(folders[0]))
            return True
        if not files:
            self._message("Drop local CSV, XLSX or DAT measurements, or one folder.")
            return False
        if self._workflow in {"tools", "slides"}:
            self.setSourcePath(str(files[0].parent))
            return True
        self.setSourcePath(str(files[0].parent))
        if self._workflow == "pl" and len(files) > 1:
            self._pl_open_catalog()
            self._message("Choose one of the dropped PL measurements in the source catalog.")
        else:
            self.selectFiles([QUrl.fromLocalFile(str(path)) for path in files])
        return True
