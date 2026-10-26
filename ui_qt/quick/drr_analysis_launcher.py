"""Asynchronous, schema-compatible handoff to native DRR Analysis."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from uuid import uuid4

from PySide6.QtCore import QObject, QProcess, QSettings, Signal, Slot
from ui_qt.quick.settings_storage import analysis_session_path,appearance_preference


def snapshot_products(products, path):
    from core.drr_analysis_workspace import create_dataset
    from core.drr_workspace_session import save_session
    request, recipe = products.request, products.request.background_recipe
    provenance = dict(measurement_files=list(request.files), background_files=list(recipe.files if recipe else request.backgrounds),
        background_mode=recipe.mode if recipe else {"external": "External", "self-first": "Self (first frame)",
            "self-last": "Self (last frame)", "fitted": "Fitted (sample)", "imported": "Imported map"}[request.background_mode],
        background_frame=request.background_frame, y_axis=request.y_axis,
        assignments=[item.to_dict() for item in recipe.assignments] if recipe else [])
    if products.background_references:
        from core.drr_fitted_background import reference_matches_files
        references = products.background_references
        if len(references) != 1 or not reference_matches_files(references[0], request.files):
            raise ValueError("Reload DRR to fit one background to the selected raw intensity average before adding it to Analysis.")
        provenance.update(average_method="Raw intensity average, one fitted background, then I/B_fit - 1",
                          fitted_background=[references[0].metadata()])
    name = Path(request.files[0]).stem if request.files else products.source.title
    # The standalone analysis computes its own SG products. Always hand over
    # the committed ΔR/R source, even when the main plot shows a derivative.
    dataset = create_dataset(products.source, name, provenance)
    save_session(path, [dataset], active_key=dataset.key, compressed=False)
    return str(path)


def launch_arguments(folder, session, settings, snapshot=None):
    root = Path(__file__).resolve().parents[2]
    executable = sys.executable
    if getattr(sys, "frozen", False):
        arguments = ["--drr-analysis-quick"]
    else:
        if sys.platform == "win32" and Path(executable).with_name("pythonw.exe").is_file():
            executable = str(Path(executable).with_name("pythonw.exe"))
        arguments = [str(root / "run_drr_analysis_quick.py")]
    arguments.extend([str(folder or root), "--session", str(session), "--settings-file", str(settings)])
    if snapshot:
        arguments.extend(["--snapshot", str(snapshot)])
    return executable, arguments, str(root)


class QuickDrrAnalysisLauncher(QObject):
    snapshotChanged = Signal()
    statusMessage = Signal(str)
    _completed = Signal(object, str)

    def __init__(self, *, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.session_path = analysis_session_path(settings,"drr")
        self.directory = self.session_path.parent
        self.settings_path = self.directory / "quick.ini"
        self.busy = self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dptk-drr-handoff")
        self._completed.connect(self._finish)

    def open(self, *, folder="", products=None):
        if self._closed or self.busy:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        settings = QSettings(str(self.settings_path), QSettings.IniFormat)
        settings.setValue("quick/appearance", appearance_preference(self.settings))
        settings.sync()
        if products is None:
            self._launch(folder, None)
            return
        destination = self.directory / "inbox" / (uuid4().hex + ".npz")
        self.busy = True
        self.snapshotChanged.emit()
        self.statusMessage.emit("Preparing the current DRR source for Analysis…")
        future = self._executor.submit(lambda: (folder, snapshot_products(products, destination)))
        def completed(task):
            try:
                result, error = task.result(), ""
            except Exception as exc:
                result, error = None, str(exc)
            if not self._closed:
                self._completed.emit(result, error)
        future.add_done_callback(completed)

    @Slot(object, str)
    def _finish(self, result, error):
        if self._closed:
            return
        self.busy = False
        if error:
            self.statusMessage.emit(error)
        else:
            self._launch(*result)
        self.snapshotChanged.emit()

    def _launch(self, folder, snapshot):
        executable, arguments, directory = launch_arguments(folder, self.session_path, self.settings_path, snapshot)
        launched, _ = QProcess.startDetached(executable, arguments, directory)
        self.statusMessage.emit(("DRR Analysis opened with the current source snapshot." if snapshot else "DRR Analysis opened.")
                                if launched else "Could not start native DRR Analysis.")

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
