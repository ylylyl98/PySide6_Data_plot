"""Controller for the DRR plot-control workflow."""

from __future__ import annotations

from ui_qt.main_window import *


class _DrrFitSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


def _drr_multi_lorentz_model(x: np.ndarray, *p: float) -> np.ndarray:
    x_arr = np.asarray(x, float)
    out = p[0] + p[1] * x_arr
    for i in range((len(p) - 2) // 3):
        amp, cen, gam = p[2 + 3*i], p[3 + 3*i], max(1e-12, p[4 + 3*i])
        out = out + amp * (gam * gam) / ((x_arr - cen) * (x_arr - cen) + gam * gam)
    return out


class _DrrFitWorker(QRunnable):
    def __init__(self, x: np.ndarray, y: np.ndarray, p0: list[float], lo: list[float], hi: list[float]) -> None:
        super().__init__()
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.p0, self.lo, self.hi = p0, lo, hi
        self.signals = _DrrFitSignals()

    def run(self) -> None:
        try:
            popt, _ = curve_fit(_drr_multi_lorentz_model, self.x, self.y,
                                p0=np.asarray(self.p0, float),
                                bounds=(np.asarray(self.lo, float), np.asarray(self.hi, float)),
                                maxfev=50000)
            self.signals.result.emit(np.asarray(popt, float))
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class DrrController:
    """Own DRR control reactions while sharing the application context."""

    def __init__(self, owner) -> None:
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_owner"), name)

    def __setattr__(self, name, value) -> None:
        if name == "_owner":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_owner"), name, value)

    def _on_drr_derivative_changed(self) -> None:
        self._invalidate_pending_drr_fit("Fit discarded: derivative changed.")
        self._invalidate_export_move_sources()
        derivative_active = self._drr_derivative_value() is not None
        self.drr_sg_window_spin.setVisible(derivative_active)
        self.drr_sg_poly_spin.setVisible(derivative_active)
        self._enforce_drr_sg_constraints(show_status=True)
        if self.loaded and self.loaded.mode == "DRR" and not self._suspend_drr_autoplot:
            self._refresh_automatic_ranges("DRR", refresh_split=True)
            self._schedule_plot_redraw("DRR")

    def _on_drr_plot_param_changed(self) -> None:
        self._invalidate_pending_drr_fit("Fit discarded: plot range or display settings changed.")
        self._invalidate_export_move_sources()
        external_baseline = self.drr_baseline_combo.currentText() == "External"
        self.drr_external_baseline_row.setVisible(external_baseline)
        self.drr_baseline_combine_combo.setVisible(external_baseline)
        self.drr_pin_baseline_chk.setVisible(external_baseline)
        if external_baseline and not self.drr_baseline_files_manual:
            self._invalidate_drr_for_background_selection(
                "Select an external background before processing."
            )
            return
        if self.loaded and self.loaded.mode == "DRR" and not self._suspend_drr_autoplot:
            sender = self.sender()
            if sender in (
                self.drr_spins["xmin"], self.drr_spins["xmax"],
                self.drr_spins["ymin"], self.drr_spins["ymax"],
                self.drr_log_chk, self.drr_center_zero_chk,
            ):
                self._refresh_automatic_ranges(
                    "DRR",
                    refresh_split=True,
                    center_split=sender in (self.drr_spins["xmin"], self.drr_spins["xmax"]),
                )
            self._schedule_plot_redraw("DRR")

    def _on_drr_baseline_mode_changed(self) -> None:
        self._status(f"State: Baseline mode set: {self.drr_baseline_combine_combo.currentText()}.")
        self._update_drr_selection_labels()
        if self.loaded and self.loaded.mode == "DRR" and not self._suspend_drr_autoplot:
            self._schedule_plot_redraw("DRR")

    def _auto_drr_vrange(self) -> None:
        if not self.loaded or self.loaded.mode != "DRR":
            return
        cube = self._drr_cube_for_display()
        x = np.asarray(cube.energy, float).ravel()
        y = np.asarray(cube.gate, float).ravel()
        z = np.asarray(cube.Z, float)
        x0, x1 = sorted((float(self.drr_spins["xmin"].value()), float(self.drr_spins["xmax"].value())))
        y0, y1 = sorted((float(self.drr_spins["ymin"].value()), float(self.drr_spins["ymax"].value())))
        x_mask = (x >= x0) & (x <= x1)
        y_mask = (y >= y0) & (y <= y1)
        if np.any(y_mask) and np.any(x_mask):
            z_roi = z[np.ix_(y_mask, x_mask)]
        else:
            z_roi = z
        finite = z_roi[np.isfinite(z_roi)]
        if finite.size == 0:
            self._status("State: Auto vmin/vmax skipped (no finite values in selected x/y range).")
            return
        if self._mode_log("DRR"):
            pos = z_roi[np.isfinite(z_roi) & (z_roi > 0)]
            if pos.size:
                vmin, vmax = np.nanpercentile(pos, [0.01, 99.99])
                vmin = float(max(vmin, 1e-12))
                vmax = float(max(vmax, vmin * 1.01))
            else:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        else:
            vmin, vmax = np.nanpercentile(finite, [0.01, 99.99])
            vmin, vmax = float(vmin), float(vmax)
        spins = self.drr_spins
        spins["vmin"].setValue(vmin)
        spins["vmax"].setValue(vmax)
        self._status(f"State: Auto vmin/vmax (ROI) = {vmin:.4g}, {vmax:.4g}")
        self._schedule_plot_redraw("DRR")

    def _auto_drr_xrange(self) -> None:
        if not self.loaded or self.loaded.mode != "DRR":
            return
        cube = self._drr_cube_for_display()
        self.drr_spins["xmin"].setValue(float(np.nanmin(cube.energy)))
        self.drr_spins["xmax"].setValue(float(np.nanmax(cube.energy)))
        self._status("State: Auto xmin/xmax set from energy axis.")
        self._schedule_plot_redraw("DRR")

    def _auto_drr_yrange(self) -> None:
        if not self.loaded or self.loaded.mode != "DRR":
            return
        cube = self._drr_cube_for_display()
        self.drr_spins["ymin"].setValue(float(np.nanmin(cube.gate)))
        self.drr_spins["ymax"].setValue(float(np.nanmax(cube.gate)))
        self._status("State: Auto ymin/ymax set from gate axis.")
        self._schedule_plot_redraw("DRR")
    def _update_drr_selection_labels(self) -> None:
        def _brief(names: List[str]) -> str:
            if not names:
                return "none"
            # Zero-width break opportunities preserve the complete first name
            # while allowing underscore-heavy measurement names to wrap.
            first = Path(names[0]).name.replace("_", "_\u200b").replace("-", "-\u200b")
            return first if len(names) == 1 else f"{first} (+{len(names) - 1} more)"

        mode_map = {
            "Last frame from each file, then average": "last",
            "First frame from each file, then average": "first",
            "Average all frames in each file, then average files": "all frames",
        }
        mode_short = mode_map.get(self.drr_baseline_combine_combo.currentText(), "last")
        self.drr_measurement_summary.setText(f"Measurement: {len(self.drr_selected_files)} files ({_brief(self.drr_selected_files)})")
        self.drr_measurement_summary.setToolTip(
            "Selected measurement files:\n" + "\n".join(self.drr_selected_files)
            if self.drr_selected_files
            else "No measurement files selected."
        )
        self.drr_baseline_summary.setText(
            f"Baselines: {len(self.drr_baseline_files_manual)} files (mode: {mode_short})"
        )
        self.drr_baseline_summary.setToolTip(
            "Selected baseline files:\n" + "\n".join(self.drr_baseline_files_manual)
            if self.drr_baseline_files_manual
            else "No external baseline files selected."
        )
        self._repopulate_drr_yaxis()
    def _edit_drr_measurements(self) -> None:
        previous = list(self.drr_selected_files)
        selected = self._owner._open_drr_source_dialog(
            title="Choose DRR Measurement Group",
            selected=self.drr_selected_files,
            baseline_mode=False,
        )
        self._reject_mixed_xlsx_selection(selected)
        self.drr_selected_files = selected
        measurement_changed = selected != previous
        if measurement_changed and not self.drr_pin_baseline_chk.isChecked():
            self.drr_baseline_files_manual = []
            self.drr_baseline_files_found = []
            self._drr_background_guess = None
            if not self._restore_saved_drr_recipe():
                self._guess_drr_background_for_selection()
        self._update_drr_selection_labels()
        if measurement_changed:
            self._clear_loaded_drr_view()
        if (
            measurement_changed
            and self.drr_baseline_combo.currentText() == "External"
            and not self.drr_baseline_files_manual
        ):
            self._invalidate_drr_for_background_selection(
                "Measurement changed. Select an external background."
            )
            return
        if self.drr_selected_files:
            self._start_load("DRR")
    def _clear_drr_measurements(self) -> None:
        self.drr_selected_files = []
        if not self.drr_pin_baseline_chk.isChecked():
            self.drr_baseline_files_manual = []
        self._update_drr_selection_labels()
        self._clear_loaded_drr_view()
        self._set_stage("No DRR measurement")
        self._update_action_states()
    def _edit_drr_baselines_dialog(self) -> None:
        self.drr_baseline_files_manual = self._owner._open_drr_source_dialog(
            title="Choose Historical or External Baseline",
            selected=self.drr_baseline_files_manual,
            baseline_mode=True,
        )
        self._drr_background_guess = None
        if self.drr_baseline_files_manual:
            blocked = self.drr_baseline_combo.blockSignals(True)
            self.drr_baseline_combo.setCurrentText("External")
            self.drr_baseline_combo.blockSignals(blocked)
            self.drr_external_baseline_row.setVisible(True)
            self.drr_baseline_combine_combo.setVisible(True)
            self.drr_pin_baseline_chk.setVisible(True)
        self._update_drr_selection_labels()
        if not self._apply_drr_background_gate_default():
            return
        if self.drr_selected_files:
            self._start_load("DRR")
    def _clear_drr_baselines(self) -> None:
        self.drr_baseline_files_manual = []
        self.drr_baseline_files_found = []
        self._drr_background_guess = None
        self.drr_pin_baseline_chk.setChecked(False)
        self._update_drr_selection_labels()
        if self.drr_baseline_combo.currentText() == "External":
            self._invalidate_drr_for_background_selection(
                "External background cleared. Select a background before processing."
            )
    def _on_drr_pin_baseline_toggled(self, checked: bool) -> None:
        if checked and not self.drr_baseline_files_manual:
            blocked = self.drr_pin_baseline_chk.blockSignals(True)
            self.drr_pin_baseline_chk.setChecked(False)
            self.drr_pin_baseline_chk.blockSignals(blocked)
            self._status("Select an external background before pinning it.")
            return
        self._status("External background pinned." if checked else "External background follows measurement selection.")
    def _open_drr_source_dialog(
        self,
        *,
        title: str,
        selected: List[str],
        baseline_mode: bool,
    ) -> List[str]:
        """Browse recent DRR groups without flattening the complete device history."""
        dlg = QDialog(self._owner)
        dlg.setWindowTitle(title)
        if not self.windowIcon().isNull():
            dlg.setWindowIcon(self.windowIcon())
        dlg.setMinimumSize(920, 560)
        dlg.resize(1120, 680)
        layout = QVBoxLayout(dlg)

        hint = QLabel(
            "Background history includes earlier measurement groups; select any compatible file or group."
            if baseline_mode
            else "The newest unprocessed measurement group is shown first. Search to reach older sessions."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        filter_row = QHBoxLayout()
        filter_edit = QLineEdit()
        filter_edit.setPlaceholderText("Search group, date, or filename...")
        show_all = QCheckBox("Show all history")
        show_incompatible = QCheckBox("Show other wavelengths")
        show_incompatible.setVisible(baseline_mode)
        show_incompatible.setToolTip(
            "Show background candidates whose filename wavelength center does not match the measurement."
        )
        unprocessed_only = QCheckBox("Unprocessed only")
        unprocessed_only.setChecked(not baseline_mode)
        unprocessed_only.setVisible(not baseline_mode)
        include_backgrounds = QCheckBox("Include background candidates")
        include_backgrounds.setVisible(not baseline_mode)
        include_backgrounds.setToolTip(
            "Show files classified as backgrounds so an unusual constant-gate measurement can be restored manually."
        )
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Scan Initial Data and its measurement folders for new files.")
        filter_row.addWidget(QLabel("Find"))
        filter_row.addWidget(filter_edit, 1)
        filter_row.addWidget(unprocessed_only)
        filter_row.addWidget(include_backgrounds)
        filter_row.addWidget(show_incompatible)
        filter_row.addWidget(show_all)
        filter_row.addWidget(refresh_btn)
        layout.addLayout(filter_row)

        panes = QSplitter(Qt.Horizontal)
        group_list = QListWidget()
        file_list = QListWidget()
        selected_list = QListWidget()
        for widget in (group_list, file_list, selected_list):
            widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
            widget.setWordWrap(True)
            widget.setTextElideMode(Qt.ElideNone)
            widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            widget.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
            widget.setUniformItemSizes(False)
            # Adjust relayouts every row whenever the dialog is resized.  The
            # delegate still computes wrapped row heights, while Fixed avoids
            # the repeated full-list relayout that made this dialog feel
            # sluggish during opening and resizing.
            widget.setResizeMode(QListView.Fixed)
            widget.setSpacing(3)
            widget.setItemDelegate(WrappedFilenameDelegate(widget))

        def _panel(label: str, widget: QListWidget) -> QWidget:
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            panel_layout.addWidget(QLabel(label))
            panel_layout.addWidget(widget, 1)
            return panel

        panes.addWidget(_panel("Data history" if baseline_mode else "Measurement sessions", group_list))
        panes.addWidget(_panel("Files in selected session", file_list))
        panes.addWidget(_panel("Chosen files", selected_list))
        panes.setStretchFactor(0, 4)
        panes.setStretchFactor(1, 3)
        panes.setStretchFactor(2, 3)
        panes.setSizes([440, 330, 330])
        layout.addWidget(panes, 1)

        action_row = QHBoxLayout()
        add_group_btn = QPushButton("Add Entire Group")
        add_files_btn = QPushButton("Add Selected Files")
        remove_btn = QPushButton("Remove")
        clear_btn = QPushButton("Clear")
        browse_btn = QPushButton("Browse File Anywhere...")
        browse_btn.setVisible(baseline_mode)
        action_row.addWidget(add_group_btn)
        action_row.addWidget(add_files_btn)
        action_row.addWidget(browse_btn)
        action_row.addStretch(1)
        action_row.addWidget(remove_btn)
        action_row.addWidget(clear_btn)
        layout.addLayout(action_row)

        def _catalog_groups():
            catalog_sources = (
                [
                    source
                    for source in self.drr_available_sources
                    if Path(source.source).suffix.lower() == ".csv"
                ]
                if baseline_mode
                else self.drr_available_sources
            )
            result = group_drr_sources(catalog_sources)
            if baseline_mode:
                result = sorted(
                    result,
                    key=lambda group: (0 if group.is_background else 1, -group.modified_time),
                )
            return result

        groups = _catalog_groups()
        groups_by_key = {group.key: group for group in groups}
        measurement_center = self._drr_selected_wavelength_center()

        def _group_search_text(group) -> str:
            return " ".join(
                [group.title, group.session_date, *(source.filename for source in group.files)]
            ).casefold()

        group_search_text = {group.key: _group_search_text(group) for group in groups}

        def _add_chosen(source: str) -> None:
            existing = {
                str(selected_list.item(index).data(Qt.UserRole) or selected_list.item(index).text())
                for index in range(selected_list.count())
            }
            if source in existing:
                return
            item = QListWidgetItem(Path(source).name)
            item.setData(Qt.UserRole, source)
            item.setToolTip(source)
            selected_list.addItem(item)

        selected_list.setUpdatesEnabled(False)
        try:
            for source in selected:
                _add_chosen(source)
        finally:
            selected_list.setUpdatesEnabled(True)

        def _selected_group():
            item = group_list.currentItem()
            return groups_by_key.get(str(item.data(Qt.UserRole))) if item is not None else None

        def _populate_files() -> None:
            group = _selected_group()
            file_list.setUpdatesEnabled(False)
            try:
                file_list.clear()
                if group is None:
                    return
                for source in group.files:
                    detail = (
                        f"\n{source.classification_reason}"
                        if source.classification != "measurement"
                        else ""
                    )
                    item = QListWidgetItem(f"{source.filename}{detail}")
                    item.setData(Qt.UserRole, source.source)
                    item.setToolTip(f"{source.source}\n{source.classification_reason}")
                    file_list.addItem(item)
            finally:
                file_list.setUpdatesEnabled(True)
                file_list.viewport().update()

        def _refresh_groups() -> None:
            nonlocal group_search_text
            needle = filter_edit.text().strip().casefold()
            all_history = show_all.isChecked() or bool(needle)
            current_key = (
                str(group_list.currentItem().data(Qt.UserRole))
                if group_list.currentItem() is not None
                else None
            )
            visible = []
            for group in groups:
                if not baseline_mode and group.is_background and not include_backgrounds.isChecked():
                    continue
                if (
                    baseline_mode
                    and measurement_center is not None
                    and group.wavelength_centers_nm
                    and not show_incompatible.isChecked()
                    and not all(
                        wavelength_centers_match(measurement_center, center)
                        for center in group.wavelength_centers_nm
                    )
                ):
                    continue
                if unprocessed_only.isVisible() and unprocessed_only.isChecked() and group.processed:
                    continue
                if needle and needle not in group_search_text.get(group.key, ""):
                    continue
                visible.append(group)
            if not all_history:
                visible = visible[:25]
            group_list.setUpdatesEnabled(False)
            signals_blocked = group_list.blockSignals(True)
            try:
                group_list.clear()
                for group in visible:
                    kind = {
                        "background": "background",
                        "likely_background": "likely background",
                        "review": "constant gate · review",
                    }.get(group.classification, "measurement")
                    center_text = (
                        " · " + "/".join(f"{center:g}" for center in group.wavelength_centers_nm) + " nm"
                        if group.wavelength_centers_nm
                        else " · wavelength unknown"
                    )
                    summary = (
                        f"{group.session_date} · {len(group.files)} file"
                        f"{'s' if len(group.files) != 1 else ''} · {kind}{center_text}"
                    )
                    if not baseline_mode:
                        badge = "✓ PROCESSED" if group.processed else "● NEW"
                        summary = f"{badge} · {summary}"
                    item = QListWidgetItem(f"{summary}\n{group.title}")
                    item.setData(Qt.UserRole, group.key)
                    item.setToolTip(
                        f"{group.title}\n{summary}\n\n"
                        + "\n".join(
                            f"{source.source} — {source.classification_reason}"
                            for source in group.files
                        )
                    )
                    if not baseline_mode:
                        item.setForeground(QColor("#237A3B" if group.processed else "#1769AA"))
                        font = item.font(); font.setBold(not group.processed); item.setFont(font)
                    group_list.addItem(item)
                selected_row = next(
                    (
                        index
                        for index in range(group_list.count())
                        if str(group_list.item(index).data(Qt.UserRole)) == current_key
                    ),
                    0,
                )
                if group_list.count():
                    group_list.setCurrentRow(selected_row)
            finally:
                group_list.blockSignals(signals_blocked)
                group_list.setUpdatesEnabled(True)
                group_list.viewport().update()
            _populate_files()

        refresh_generation = 0
        refresh_in_progress = False
        dialog_closed = False
        refresh_workers = []

        def _mark_dialog_closed(_result: int) -> None:
            nonlocal refresh_generation, dialog_closed
            dialog_closed = True
            # Invalidate any queued worker result before the dialog's widgets
            # are destroyed.  The worker itself may finish after exec() exits.
            refresh_generation += 1

        def _scan_catalog(folder: str, *, progress, log):
            # Work on a snapshot so a file-watcher refresh in the main window
            # cannot mutate the cache while this worker is reading files.
            cache = self._owner._drr_source_cache.clone()
            sources = discover_drr_sources(folder, cache=cache)
            return folder, sources, cache

        def _apply_catalog_result(result, generation: int) -> None:
            nonlocal groups, groups_by_key, measurement_center, group_search_text
            nonlocal refresh_in_progress
            folder, sources, cache = result
            if dialog_closed or generation != refresh_generation:
                return
            if str(folder).casefold() != str(self.current_folder).casefold():
                return
            self.drr_available_sources = sources
            self._owner._drr_source_cache = cache
            groups = _catalog_groups()
            groups_by_key = {group.key: group for group in groups}
            group_search_text = {group.key: _group_search_text(group) for group in groups}
            measurement_center = self._drr_selected_wavelength_center()
            _refresh_groups()
            refresh_in_progress = False
            refresh_btn.setEnabled(True)
            self._status(f"DRR catalog refreshed: {len(self.drr_available_sources)} files.")

        def _catalog_scan_error(message: str, generation: int) -> None:
            nonlocal refresh_in_progress
            if dialog_closed or generation != refresh_generation:
                return
            refresh_in_progress = False
            refresh_btn.setEnabled(True)
            self._status(f"DRR catalog refresh failed: {str(message).splitlines()[0]}")

        def _catalog_scan_finished(generation: int, worker) -> None:
            if worker in refresh_workers:
                refresh_workers.remove(worker)
            if not dialog_closed:
                refresh_btn.setText("Refresh")

        def _reload_catalog() -> None:
            nonlocal refresh_generation, refresh_in_progress
            if refresh_in_progress or not self.current_folder:
                return
            refresh_in_progress = True
            refresh_generation += 1
            generation = refresh_generation
            refresh_btn.setEnabled(False)
            refresh_btn.setText("Refreshing...")
            worker = Worker(_scan_catalog, self.current_folder)
            refresh_workers.append(worker)
            worker.signals.result.connect(
                lambda result, generation=generation: _apply_catalog_result(result, generation)
            )
            worker.signals.error.connect(
                lambda message, generation=generation: _catalog_scan_error(message, generation)
            )
            worker.signals.finished.connect(
                lambda generation=generation, worker=worker: _catalog_scan_finished(
                    generation, worker
                )
            )
            self._owner.thread_pool.start(worker)

        def _add_group() -> None:
            group = _selected_group()
            if group is not None:
                for source in group.files:
                    _add_chosen(source.source)

        def _add_files() -> None:
            for item in file_list.selectedItems():
                _add_chosen(str(item.data(Qt.UserRole)))

        def _remove() -> None:
            for item in selected_list.selectedItems():
                selected_list.takeItem(selected_list.row(item))

        def _browse_external() -> None:
            paths, _selected_filter = QFileDialog.getOpenFileNames(
                dlg,
                "Choose External DRR Baseline",
                self.current_folder or self._browse_start_folder(),
                "DRR baseline files (*.csv)",
            )
            for path in paths:
                _add_chosen(str(Path(path).resolve()))

        group_list.currentRowChanged.connect(lambda _row: _populate_files())
        group_list.itemDoubleClicked.connect(lambda _item: _add_group())
        file_list.itemDoubleClicked.connect(lambda _item: _add_files())
        filter_timer = QTimer(dlg)
        filter_timer.setSingleShot(True)
        filter_timer.setInterval(180)
        filter_timer.timeout.connect(_refresh_groups)
        filter_edit.textChanged.connect(lambda _text: filter_timer.start())
        show_all.toggled.connect(lambda _checked: _refresh_groups())
        unprocessed_only.toggled.connect(lambda _checked: _refresh_groups())
        include_backgrounds.toggled.connect(lambda _checked: _refresh_groups())
        show_incompatible.toggled.connect(lambda _checked: _refresh_groups())
        add_group_btn.clicked.connect(_add_group)
        add_files_btn.clicked.connect(_add_files)
        remove_btn.clicked.connect(_remove)
        clear_btn.clicked.connect(selected_list.clear)
        browse_btn.clicked.connect(_browse_external)
        refresh_btn.clicked.connect(_reload_catalog)
        _refresh_groups()
        dlg.finished.connect(_mark_dialog_closed)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        if dlg.exec() != QDialog.Accepted:
            return selected
        return [
            str(selected_list.item(index).data(Qt.UserRole) or selected_list.item(index).text())
            for index in range(selected_list.count())
        ]
    def _set_drr_gate_spin_value(self, gate_value: float) -> None:
        spin = self.drr_spins["gate"]
        old = spin.blockSignals(True)
        try:
            spin.setValue(float(gate_value))
        finally:
            spin.blockSignals(old)
    def _current_drr_spectrum(self, cube: DataCube) -> tuple[float, np.ndarray, np.ndarray]:
        gate_value = float(self.drr_spins["gate"].value())
        gate_used, y = nearest_gate_spectrum(cube, gate_value)
        x = np.asarray(cube.energy, float).ravel()
        return gate_used, x, np.asarray(y, float).ravel()
    def _draw_drr_analysis_overlays(self, cube: DataCube, gate_used: float, x: np.ndarray, y: np.ndarray) -> None:
        if self._drr_spectrum_ax is None or self._drr_heatmap_ax is None:
            return
        if self._drr_heatmap_peak_artist is not None:
            try:
                self._drr_heatmap_peak_artist.remove()
            except Exception:
                pass
            self._drr_heatmap_peak_artist = None
        if self._drr_heatmap_fit_artist is not None:
            try:
                self._drr_heatmap_fit_artist.remove()
            except Exception:
                pass
            self._drr_heatmap_fit_artist = None
        if self._drr_peak_gate is not None and abs(float(gate_used) - float(self._drr_peak_gate)) > 1e-9:
            self._drr_peak_gate = None
            self._drr_peak_indices = None
        if self._drr_fit_gate is not None and abs(float(gate_used) - float(self._drr_fit_gate)) > 1e-9:
            self._drr_fit_gate = None
            self._drr_fit_x = None
            self._drr_fit_y = None
            self._drr_fit_centers = None
            self.drr_fit_status.setText("")

        if (
            self.drr_peak_show_chk.isChecked()
            and self._drr_peak_indices is not None
            and self._drr_peak_gate is not None
            and abs(float(gate_used) - float(self._drr_peak_gate)) <= 1e-9
            and self._drr_peak_indices.size > 0
        ):
            pidx = np.asarray(self._drr_peak_indices, dtype=int)
            self._drr_spectrum_ax.scatter(x[pidx], y[pidx], s=26, marker="o", facecolor="#ffd84d", edgecolor="#222", zorder=30)
            self._drr_heatmap_peak_artist = self._drr_heatmap_ax.scatter(
                x[pidx],
                np.full(pidx.size, float(gate_used)),
                s=28,
                marker="o",
                facecolor="#ffd84d",
                edgecolor="#111",
                zorder=35,
            )

        if (
            self.drr_fit_show_chk.isChecked()
            and self._drr_fit_gate is not None
            and self._drr_fit_x is not None
            and self._drr_fit_y is not None
            and abs(float(gate_used) - float(self._drr_fit_gate)) <= 1e-9
        ):
            self._drr_spectrum_ax.plot(self._drr_fit_x, self._drr_fit_y, color="#f28e2b", linewidth=1.6, zorder=28)
            if self._drr_fit_centers is not None and self._drr_fit_centers.size:
                self._drr_heatmap_fit_artist = self._drr_heatmap_ax.scatter(
                    np.asarray(self._drr_fit_centers, float),
                    np.full(int(self._drr_fit_centers.size), float(gate_used)),
                    s=34,
                    marker="x",
                    color="#f28e2b",
                    linewidths=1.5,
                    zorder=36,
                )
    def _on_drr_find_peaks(self) -> None:
        self.results_dock.show()
        self._update_results_dock_page()
        if self.last_plotted_mode != "DRR" or self._last_plot_cube is None:
            return
        gate_used, x, y = self._current_drr_spectrum(self._last_plot_cube)
        peaks = self._compute_peak_indices(
            x,
            y,
            spins=self.drr_spins,
            prom_spin=self.drr_peak_prom_spin,
            dist_spin=self.drr_peak_dist_spin,
            max_spin=self.drr_peak_max_spin,
            mode=self.drr_peak_mode_combo.currentText(),
        )
        self._drr_peak_gate = float(gate_used)
        self._drr_peak_indices = peaks
        if peaks.size > 0:
            self._set_fit_n_from_found(self.drr_fit_n_spin, int(peaks.size))
        self.drr_fit_status.setText(f"Peaks: {int(peaks.size)}")
        self._update_drr_spectrum_and_gate_line(self._last_plot_cube)
    def _on_drr_fit_lorentz(self) -> None:
        self.results_dock.show()
        self._update_results_dock_page()
        if self.last_plotted_mode != "DRR" or self._last_plot_cube is None:
            return
        gate_used, x, y = self._current_drr_spectrum(self._last_plot_cube)
        mask = self._visible_x_mask(x, self.drr_spins) & np.isfinite(y)
        if np.count_nonzero(mask) < 8:
            self.drr_fit_status.setText("Fit failed: not enough points in x-range.")
            return
        x_sel = x[mask]
        y_sel = y[mask]
        n_peaks = int(self.drr_fit_n_spin.value())
        if self._drr_peak_indices is None or self._drr_peak_gate is None or abs(float(gate_used) - float(self._drr_peak_gate)) > 1e-9:
            peak_idx = self._compute_peak_indices(
                x,
                y,
                spins=self.drr_spins,
                prom_spin=self.drr_peak_prom_spin,
                dist_spin=self.drr_peak_dist_spin,
                max_spin=self.drr_peak_max_spin,
                mode=self.drr_peak_mode_combo.currentText(),
            )
        else:
            peak_idx = np.asarray(self._drr_peak_indices, dtype=int)
        peak_idx = peak_idx[(peak_idx >= 0) & (peak_idx < x.size)]
        if peak_idx.size < n_peaks:
            order = np.argsort(np.abs(y_sel - np.nanmedian(y_sel)))[::-1]
            idx_sel = np.where(mask)[0]
            extra = idx_sel[order[: max(1, n_peaks - peak_idx.size)]]
            peak_idx = np.unique(np.concatenate([peak_idx, extra]))
        if peak_idx.size == 0:
            self.drr_fit_status.setText("Fit failed: no peak candidates.")
            return
        centers0 = np.asarray(np.sort(x[peak_idx])[:n_peaks], float)
        while centers0.size < n_peaks:
            centers0 = np.append(centers0, float(np.nanmean(x_sel)))
        base0 = float(np.nanmedian(y_sel))
        slope0 = 0.0
        y_amp = float(np.nanmax(np.abs(y_sel - base0))) if np.isfinite(np.nanmax(np.abs(y_sel - base0))) else 1.0
        x_rng = max(1e-9, float(np.nanmax(x_sel) - np.nanmin(x_sel)))
        dx = float(np.nanmedian(np.diff(x_sel))) if x_sel.size > 2 else x_rng / 100.0
        g0 = max(abs(dx) * 2.0, x_rng / 80.0, 1e-6)

        p0: list[float] = [base0, slope0]
        lo: list[float] = [float(np.nanmin(y_sel) - 3 * y_amp), -1e9]
        hi: list[float] = [float(np.nanmax(y_sel) + 3 * y_amp), 1e9]
        for c0 in centers0:
            p0.extend([y_amp * 0.7, float(c0), g0])
            lo.extend([-5 * y_amp, float(np.nanmin(x_sel)), max(abs(dx) * 0.25, 1e-8)])
            hi.extend([5 * y_amp, float(np.nanmax(x_sel)), x_rng])
        self._drr_fit_generation = getattr(self, "_drr_fit_generation", 0) + 1
        generation = self._drr_fit_generation
        cube = self._last_plot_cube
        source_key = tuple(self.drr_selected_files)
        worker = _DrrFitWorker(x_sel, y_sel, p0, lo, hi)
        workers = getattr(self, "_drr_fit_workers", None)
        if workers is None:
            workers = []; self._drr_fit_workers = workers
        workers.append(worker)
        worker.signals.result.connect(
            lambda popt, g=generation, c=cube, sk=source_key, gate=gate_used,
            requested=float(self.drr_spins["gate"].value()), peaks=n_peaks, xx=x.copy():
            self._on_drr_fit_finished(g, c, sk, gate, requested, peaks, xx, popt)
        )
        worker.signals.error.connect(lambda message, g=generation: self._on_drr_fit_error(g, message))
        worker.signals.finished.connect(lambda w=worker: self._finish_drr_fit_worker(w))
        self.drr_fit_status.setText("Fitting Lorentz peaks…")
        self.thread_pool.start(worker)

    def _finish_drr_fit_worker(self, worker) -> None:
        try:
            self._drr_fit_workers.remove(worker)
        except (AttributeError, ValueError):
            pass

    def _on_drr_fit_error(self, generation: int, message: str) -> None:
        if generation == getattr(self, "_drr_fit_generation", 0):
            self.drr_fit_status.setText(f"Fit failed: {message}")

    def _on_drr_fit_finished(self, generation: int, cube, source_key, gate_used: float,
                             requested_gate: float, n_peaks: int, x: np.ndarray, popt: np.ndarray) -> None:
        if generation != getattr(self, "_drr_fit_generation", 0):
            return
        if (self.last_plotted_mode != "DRR" or self._last_plot_cube is not cube
                or tuple(self.drr_selected_files) != tuple(source_key)):
            if str(self.drr_fit_status.text()).startswith("Fitting"):
                self.drr_fit_status.setText("Fit discarded: DRR source changed.")
            return
        current_gate, _ = self._current_drr_spectrum(cube)
        if abs(float(current_gate) - float(gate_used)) > 1e-9 or abs(float(self.drr_spins["gate"].value()) - float(requested_gate)) > 1e-9:
            if str(self.drr_fit_status.text()).startswith("Fitting"):
                self.drr_fit_status.setText("Fit discarded: gate changed.")
            return
        y_fit = _drr_multi_lorentz_model(x, *popt)
        centers_fit = np.asarray([popt[3 + 3 * i] for i in range(n_peaks)], float)
        self._drr_fit_gate = float(gate_used)
        self._drr_fit_x = x
        self._drr_fit_y = np.asarray(y_fit, float)
        self._drr_fit_centers = np.asarray(np.sort(centers_fit), float)
        self.drr_fit_status.setText("Fit centers: " + ", ".join(f"{c:.4f}" for c in self._drr_fit_centers[:4]))
        self._update_drr_spectrum_and_gate_line(cube)

    def _invalidate_pending_drr_fit(self, message: str = "") -> None:
        self._drr_fit_generation = getattr(self, "_drr_fit_generation", 0) + 1
        if message and hasattr(self, "drr_fit_status") and str(self.drr_fit_status.text()).startswith("Fitting"):
            self.drr_fit_status.setText(message)
    def _on_drr_clear_fit(self) -> None:
        self._invalidate_pending_drr_fit()
        self._drr_fit_gate = None
        self._drr_fit_x = None
        self._drr_fit_y = None
        self._drr_fit_centers = None
        self.drr_fit_status.setText("")
        if self.last_plotted_mode == "DRR" and self._last_plot_cube is not None:
            self._update_drr_spectrum_and_gate_line(self._last_plot_cube)
    def _on_drr_analysis_view_changed(self) -> None:
        if self.last_plotted_mode == "DRR" and self._last_plot_cube is not None:
            self._update_drr_spectrum_and_gate_line(self._last_plot_cube)
    def _update_drr_analysis_text(self, gate_used: float, x: np.ndarray, y: np.ndarray) -> None:
        lines: list[str] = []
        if (
            self._drr_peak_indices is not None
            and self._drr_peak_gate is not None
            and abs(float(gate_used) - float(self._drr_peak_gate)) <= 1e-9
            and self._drr_peak_indices.size > 0
        ):
            pidx = np.asarray(self._drr_peak_indices, dtype=int)
            pairs = [f"({float(x[i]):.5f}, {float(y[i]):.4g})" for i in pidx]
            lines.append("Found: " + "; ".join(pairs))
        else:
            lines.append("Found: none")
        if (
            self._drr_fit_centers is not None
            and self._drr_fit_centers.size > 0
            and self._drr_fit_gate is not None
            and abs(float(gate_used) - float(self._drr_fit_gate)) <= 1e-9
            and self._drr_fit_x is not None
            and self._drr_fit_y is not None
        ):
            yc = np.interp(np.asarray(self._drr_fit_centers, float), np.asarray(self._drr_fit_x, float), np.asarray(self._drr_fit_y, float))
            fit_pairs = [f"({float(cx):.5f}, {float(cy):.4g})" for cx, cy in zip(np.asarray(self._drr_fit_centers, float), np.asarray(yc, float))]
            lines.append("Fit: " + "; ".join(fit_pairs))
        else:
            lines.append("Fit: none")
        self.drr_analysis_text.setPlainText("\n".join(lines))
    def _remove_nearest_drr_peak(self, x_click: float) -> bool:
        if self._last_plot_cube is None or self._drr_peak_indices is None or self._drr_peak_indices.size == 0:
            return False
        gate_used, _y = nearest_gate_spectrum(self._last_plot_cube, float(self.drr_spins["gate"].value()))
        if self._drr_peak_gate is None or abs(float(gate_used) - float(self._drr_peak_gate)) > 1e-9:
            return False
        x = np.asarray(self._last_plot_cube.energy, float).ravel()
        pidx = np.asarray(self._drr_peak_indices, dtype=int)
        j = int(np.argmin(np.abs(x[pidx] - float(x_click))))
        self._drr_peak_indices = np.delete(pidx, j)
        self.drr_fit_status.setText(f"Peaks: {int(self._drr_peak_indices.size)}")
        self._update_drr_spectrum_and_gate_line(self._last_plot_cube)
        return True
    def _remove_peak_from_drr_heatmap_click(self, x_click: float, y_click: float) -> bool:
        if (
            self._last_plot_cube is None
            or self._drr_peak_indices is None
            or self._drr_peak_indices.size == 0
            or self._drr_peak_gate is None
        ):
            return False
        gate_axis = np.asarray(self._last_plot_cube.gate, float).ravel()
        if gate_axis.size == 0:
            return False
        gmin, gmax = float(np.nanmin(gate_axis)), float(np.nanmax(gate_axis))
        grng = max(1e-12, gmax - gmin)
        dg = float(np.nanmedian(np.abs(np.diff(gate_axis)))) if gate_axis.size > 1 else grng * 0.02
        tol = max(0.02 * grng, 0.75 * max(dg, 1e-9))
        if abs(float(y_click) - float(self._drr_peak_gate)) > tol:
            return False
        return self._remove_nearest_drr_peak(float(x_click))
    def _update_drr_spectrum_and_gate_line(self, cube: DataCube) -> None:
        if self._drr_spectrum_ax is None:
            return
        gate_value = float(self.drr_spins["gate"].value())
        gate_used, y = nearest_gate_spectrum(cube, gate_value)
        x = np.asarray(cube.energy, float).ravel()
        self._drr_spectrum_ax.clear()
        self._drr_spectrum_ax.plot(x, np.asarray(y, float), linewidth=1.3)
        self._drr_spectrum_ax.set_title(f"Spectrum @ {gate_used:.6g} V")
        self._drr_spectrum_ax.set_xlabel("Photon Energy (eV)")
        self._drr_spectrum_ax.set_ylabel(cube.cbar_label)
        self._drr_spectrum_ax.grid(alpha=0.25)
        xlim = self._safe_spectrum_xlim(
            x,
            (float(self.drr_spins["xmin"].value()), float(self.drr_spins["xmax"].value())),
        )
        self._drr_spectrum_ax.set_xlim(xlim)
        self._auto_scale_spectrum_y(self._drr_spectrum_ax, x, y, xlim)
        self._set_drr_gate_spin_value(gate_used)
        self._ensure_gate_line(cube, gate_used)
        self._draw_drr_analysis_overlays(cube, gate_used, x, np.asarray(y, float))
        self._update_drr_analysis_text(gate_used, x, np.asarray(y, float))
        self.canvas.draw_idle()
