"""Controller for SHG processing view interactions."""

from ui_qt.main_window import *


class _ShgWorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class _ShgWorker(QRunnable):
    """Run the CPU-heavy SHG processing/fit away from the GUI thread."""

    def __init__(self, payload: tuple[object, ...]) -> None:
        super().__init__()
        self.payload = payload
        self.signals = _ShgWorkerSignals()

    def run(self) -> None:
        try:
            data, data_b, settings, fit_settings, background, background_b, compare = self.payload
            result = process_shg_sweep(data, settings, background=background)
            result_b = None
            fit = fit_b = twist = None
            if compare:
                result_b = process_shg_sweep(data_b, settings, background=background_b)
                if fit_settings.enabled:
                    try:
                        twist = fit_shg_twist_comparison(result, result_b, fit_settings)
                        fit, fit_b = twist.reference_fit, twist.sample_fit
                    except ValueError:
                        pass
            elif fit_settings.enabled:
                try:
                    fit = fit_shg_angular_result(result, fit_settings)
                except ValueError:
                    pass
            self.signals.result.emit((result, result_b, fit, fit_b, twist))
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class ShgController:
    def __init__(self, owner):
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name):
        return object.__getattribute__(object.__getattribute__(self, "_owner"), name)

    def __setattr__(self, name, value):
        if name == "_owner":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_owner"), name, value)

    def _on_shg_source_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._shg_update_summary()
        if self.loaded and self.loaded.mode == "SHG Processing" and self.loaded.shg_compare == self._shg_compare_mode():
            self._start_load("SHG Processing")

    def _on_shg_workflow_changed(self) -> None:
        compare = self._shg_compare_mode()
        self.shg_fit_branch_spin.setEnabled(compare)
        self._shg_update_summary()
        self._status("SHG Compare / Twist Angle selected." if compare else "SHG Single File selected.")

    def _on_shg_param_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._shg_update_background_controls()
        if self.loaded and self.loaded.mode == "SHG Processing" and self.loaded.shg_compare == self._shg_compare_mode():
            self._request_shg_reprocess()

    def _on_shg_cosmic_param_changed(self) -> None:
        self._shg_update_cosmic_controls()
        self._on_shg_param_changed()

    def _on_shg_fit_param_changed(self) -> None:
        enabled = bool(self.shg_fit_enable_chk.isChecked())
        for widget in (self.shg_fit_min_spin, self.shg_fit_max_spin, self.shg_fit_weighted_chk, self.shg_fit_include_excluded_chk):
            widget.setEnabled(enabled)
        self.shg_fit_branch_spin.setEnabled(enabled and self._shg_compare_mode())
        self._on_shg_param_changed()

    def _on_shg_spectrum_view_changed(self) -> None:
        if self.loaded and self.loaded.mode == "SHG Processing" and self.loaded.shg_compare == self._shg_compare_mode():
            # View-only changes do not require reprocessing; coalesce the
            # Matplotlib redraw with other controls.
            self._schedule_plot_redraw("SHG Processing")

    def _request_shg_reprocess(self) -> None:
        timer = getattr(self, "_shg_reprocess_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(100)
            timer.timeout.connect(self._start_shg_reprocess)
            self._shg_reprocess_timer = timer
        timer.start()

    def _start_shg_reprocess(self) -> None:
        loaded = self.loaded
        if not loaded or loaded.mode != "SHG Processing" or loaded.shg_data is None:
            return
        try:
            settings = self._shg_settings_from_ui()
            fit_settings = self._shg_fit_settings_from_ui()
        except ValueError as exc:
            self._status(str(exc))
            return
        compare = bool(loaded.shg_compare)
        if compare and loaded.shg_data_b is None:
            self._status("The SHG comparison sample file is not loaded.")
            return
        self._shg_reprocess_generation = getattr(self, "_shg_reprocess_generation", 0) + 1
        generation = self._shg_reprocess_generation
        self._shg_reprocess_key = (generation, settings, fit_settings, compare)
        self._status("Updating SHG processing…")
        worker = _ShgWorker((
            loaded.shg_data,
            loaded.shg_data_b,
            settings,
            fit_settings,
            loaded.shg_background,
            loaded.shg_background_b,
            compare,
        ))
        worker.signals.result.connect(lambda result, g=generation: self._on_shg_reprocessed(g, result))
        worker.signals.error.connect(lambda message, g=generation: self._on_shg_reprocess_error(g, message))
        workers = getattr(self, "_shg_reprocess_workers", None)
        if workers is None:
            workers = []
            self._shg_reprocess_workers = workers
        workers.append(worker)
        worker.signals.finished.connect(lambda w=worker: self._finish_shg_worker(w))
        self.thread_pool.start(worker)

    def _finish_shg_worker(self, worker: _ShgWorker) -> None:
        workers = getattr(self, "_shg_reprocess_workers", [])
        try:
            workers.remove(worker)
        except ValueError:
            pass

    def _on_shg_reprocessed(self, generation: int, payload: tuple[object, ...]) -> None:
        if generation != getattr(self, "_shg_reprocess_generation", 0):
            return
        loaded = self.loaded
        if not loaded or loaded.mode != "SHG Processing":
            return
        loaded.shg_result, loaded.shg_result_b, loaded.shg_fit, loaded.shg_fit_b, loaded.shg_twist = payload
        _generation, settings, fit_settings, _compare = self._shg_reprocess_key
        loaded.shg_settings = settings
        loaded.shg_fit_settings = fit_settings
        self._shg_loaded_processing_key = self._shg_reprocess_key
        self._shg_update_summary()
        self._schedule_plot_redraw("SHG Processing", delay_ms=0)

    def _on_shg_reprocess_error(self, generation: int, message: str) -> None:
        if generation != getattr(self, "_shg_reprocess_generation", 0):
            return
        self._status(f"SHG processing unavailable: {message.splitlines()[0]}")

    def _on_shg_background_method_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._shg_update_background_controls()
        if not self.loaded or self.loaded.mode != "SHG Processing":
            return
        if self.shg_background_method_combo.currentText() == "External + local residual":
            if self._shg_compare_mode():
                if not all(self._shg_compare_background_files()):
                    self._status("Select external SHG background CSVs for both comparison files.")
                    return
            elif not self._shg_background_file():
                self._status("Select an SHG background CSV for external background mode.")
                return
            self._start_load("SHG Processing")
            return
        self._request_shg_reprocess()
