"""Agg/Quick inspection and text-only cursor for native Peak Analysis."""
from __future__ import annotations

from core.typography import matplotlib_font_context,try_render_lock


class PeakAnalysisPlotMixin:
    def _plot_init(self):
        self._canvas = self._navigation = None
        self._heat = self._spectrum = self._residual = self._gate_line = None
        self._spectrum_pool = None
        self._trend_axes = ()
        self._background_image = self._background_bounds = None
        self._plot_signature = None
        self._overlay_artists = []
        self._building_plot = False
        self._plot_tool = ""
        self._last_painted_row = None
        self._cursor_pending = None
        self._cursor_state = dict(text="Cursor · X: — · Y: — · Z: —", tooltip="Move over a plot to read mouse coordinates. Hovering does not change the selected spectrum or analysis.")
        self._cursor_last_tip = ""
        self._preview_index = None
        self._preview_signature = None
        self._peak_drag = None
        self._peak_preview = ()
        self._peak_selection_line = None
        self._middle_pan = None
        self._middle_pan_dataset = None
        self._syncing_x = False
        self._home_limits = {}

    def _ensure_plot(self):
        if self._canvas is not None:
            return
        from matplotlib.figure import Figure
        from ui_qt.quick.figure_canvas import QuickNavigationToolbar
        from ui_qt.quick.plot_surface import QuickFigureCanvas
        self._canvas = QuickFigureCanvas(Figure(figsize=(9, 7), dpi=100))
        owner = self
        class PeakNavigationToolbar(QuickNavigationToolbar):
            def home(self, *args):
                owner._plot_home()
        self._navigation = PeakNavigationToolbar(self._canvas)
        self._canvas.mpl_connect("button_press_event", self._plot_clicked)
        self._canvas.mpl_connect("motion_notify_event", self._peak_motion)
        self._canvas.mpl_connect("button_release_event", self._peak_release)
        self._canvas.mpl_connect("key_press_event", self._peak_key)
        self._canvas.mpl_connect("motion_notify_event", self._cursor_motion)
        self._canvas.mpl_connect("axes_leave_event", self._cursor_leave)
        self._canvas.mpl_connect("figure_leave_event", self._cursor_leave)
        self._canvas.mpl_connect("draw_event", self._plot_drawn)
        self._canvas.mpl_connect("resize_event", self._resize_plot)
        self._canvas.mpl_connect("scroll_event", self._plot_scroll)
        self._canvas.mpl_connect("button_press_event", self._middle_press)
        self._canvas.mpl_connect("motion_notify_event", self._middle_motion)
        self._canvas.mpl_connect("button_release_event", self._middle_release)
        self._canvas.mpl_connect("close_event", self._cancel_middle_pan)
        self._canvas.backend.drawFailed.connect(lambda message: self._message("Peak Analysis plot: " + message, error=True))
        self.plotBackendChanged.emit()

    def _schedule_plot(self):
        if not self._closed and self.dataset is not None and not self._plot_timer.isActive():
            self._plot_timer.start(0)

    def _clear_plot_background(self):
        self._background_image = self._background_bounds = None

    def _default_map_share(self):
        if self.dataset and self.dataset.get('analysis_scope') == 'spectrum':
            return 29
        return 45 if self.dataset and self.dataset.get('kind') == 'PL' and self._controls_page == 'branches' else 69

    def _resize_plot(self, event=None):
        self._clear_plot_background()
        if self._heat is not None:
            # Keep room for tick labels and axis titles when the Quick surface
            # shrinks; fixed fractional margins clip labels in compact windows.
            height = max(self._canvas.figure.get_figheight(), 1.)
            width = max(self._canvas.figure.get_figwidth(), 1.)
            grid = self._heat.get_subplotspec().get_gridspec()
            share = self.dataset.get("map_height_percent", self._default_map_share())
            grid.set_height_ratios([share, (100 - share) * .8, (100 - share) * .2]
                if self._residual is not None else [share, 100 - share])
            self._heat.get_subplotspec().get_gridspec().update(
                bottom=max(.09, min(.24, .52 / height)), top=min(.95, 1 - .28 / height),
                left=max(.11, min(.22, .62 / width)), right=min(.9, max(.74, 1 - .85 / width)),
                hspace=max(.22, min(.65, 1.5 / height)))
            # These embedded figures have no pyplot manager for GridSpec.update
            # to reposition; apply the revised subplot boxes directly.
            for axis in self._canvas.figure.axes:
                spec = axis.get_subplotspec()
                if spec is not None:
                    axis.set_position(spec.get_position(self._canvas.figure))

    def _style_figure(self):
        palette = self._colors
        figure = self._canvas.figure
        figure.set_facecolor(palette["inset"])
        for axis in figure.axes:
            axis.set_facecolor(palette["inset"])
            axis.tick_params(colors=palette["muted"], labelsize=8)
            for text in (axis.title, axis.xaxis.label, axis.yaxis.label, axis.xaxis.get_offset_text(), axis.yaxis.get_offset_text()):
                text.set_color(palette["text"])
            for spine in axis.spines.values():
                spine.set_color(palette["border"])
            for text in axis.texts:
                if getattr(text, "_use_theme_text", False):
                    text.set_color(palette["text"])
            legend = axis.get_legend()
            if legend is not None:
                legend.get_frame().set_facecolor(palette["inset"])
                legend.get_frame().set_edgecolor(palette["border"])
                for text in legend.get_texts():
                    text.set_color(palette["text"])

    def _render(self):
        with try_render_lock() as available:
            if available:self._render_locked()
            elif not self._closed:self._plot_timer.start(16)

    def _render_locked(self):
        self._cancel_middle_pan()
        dataset = self.dataset
        if self._closed or dataset is None:
            return
        from core.peak_plotting import build_figure
        try:
            self._ensure_plot()
            self._cancel_peak_drag(redraw=False)
            self._building_plot = True
            self._cursor_clear()
            self._preview_signature = None
            if dataset.get('plot_view') == 'trends' and dataset['kind'] == 'PL':
                from core.peak_plotting import build_fit_trends
                self._clear_plot_background()
                self._heat = self._spectrum = self._residual = self._gate_line = None
                self._spectrum_pool = None
                self._overlay_artists = []
                self._plot_signature = None
                self._trend_axes = tuple(build_fit_trends(dataset, self._canvas.figure))
                self._capture_home_limits()
                self._style_figure()
                self._navigation.update()
                self._canvas.draw_idle()
                self._publish()
                return
            self._trend_axes = ()
            if self._plot_signature == self._static_plot_signature(dataset) and self._spectrum_pool is not None:
                # Detection/filtering/fitting replace peak metadata, not the
                # measured grid. Keep the mesh, colorbars, axes and navigation
                # history while refreshing only the changing overlay/spectrum.
                from core.peak_plotting import draw_overlay
                for artist in self._overlay_artists:
                    artist.remove()
                self._overlay_artists = draw_overlay(self._heat, dataset) if dataset.get("show_overlay", True) else []
                for artist in self._overlay_artists:
                    artist.set_animated(True)
                self._spectrum_pool.update(dataset, dataset["row"], self._colors)
                self._refresh_peak_selection(redraw=False)
                self._gate_line.set_ydata([float(dataset["cube"].gate[dataset["row"]])] * 2)
                self._redraw_dynamic()
                self._last_painted_row = (dataset["key"], dataset["row"])
                self._publish()
                return
            self._clear_plot_background()
            figure = self._canvas.figure
            figure.set_layout_engine(None)
            self._heat, self._spectrum, self._residual, self._gate_line = build_figure(dataset, dataset["row"],
                dataset.get("show_overlay", True), dataset.get("show_residual", False) and dataset["kind"] == "PL"
                and bool((dataset.get('result') or {}).get('fits')), figure)
            # The source is already identified by the window title and file tabs.
            # Keep the embedded map title short so it cannot collide with colorbars.
            self._heat.set_title(dataset.get("product_label") or ("ΔR/R" if dataset["kind"] == "DRR" and dataset["channel"] == "raw" else "Spectral map"), fontsize=10)
            self._heat.set_xlabel("")
            self._resize_plot()
            from ui_qt.quick.peak_analysis_spectrum import RetainedPeakSpectrum
            self._peak_selection_line = None
            self._spectrum_pool = RetainedPeakSpectrum(self._spectrum, self._residual, dataset, dataset["row"], self._colors)
            self._refresh_peak_selection(redraw=False)
            for axis in (self._spectrum, self._residual):
                if axis is not None:
                    axis.set_animated(True)
            self._gate_line.set_animated(True)
            self._overlay_artists = [artist for artist in (*self._heat.lines, *self._heat.collections)
                if artist.get_gid() in {"peak-rejected", "peak-candidates", "peak-track", "peak-uncertain", "peak-detected-reference", "peak-fit-review", "peak-shoulders"}]
            for artist in self._overlay_artists:
                artist.set_animated(True)
            self._heat.callbacks.connect("xlim_changed", self._remember_view)
            self._heat.callbacks.connect("ylim_changed", self._remember_view)
            self._heat.callbacks.connect("xlim_changed", self._sync_x_limits)
            self._spectrum.callbacks.connect("xlim_changed", self._sync_x_limits)
            self._capture_home_limits()
            self._remember_view(self._heat)
            self._plot_signature = self._static_plot_signature(dataset)
            self._style_figure()
            self._navigation.update()
            self._plot_tool = str(self._navigation.mode.name).lower() if self._navigation.mode else ""
            self._canvas.draw_idle()
            self._last_painted_row = (dataset["key"], dataset["row"])
            self._publish()
        except (ValueError, RuntimeError) as error:
            self._message(str(error), error=True)
        finally:
            self._building_plot = False

    def _static_plot_signature(self, dataset):
        view = dataset.get("view") or {}
        return (dataset["key"], id(dataset["cube"]), dataset["name"], dataset.get("product_label"),
            dataset.get("analysis_scope", "map"),
            bool(dataset.get("show_residual", False) and dataset["kind"] == "PL" and (dataset.get('result') or {}).get('fits')),
            repr(view), repr(self._colors))

    def _draw_dynamic(self):
        with matplotlib_font_context(self._canvas.figure):
            for axis in (self._spectrum, self._residual):
                if axis is not None:
                    self._canvas.figure.draw_artist(axis)
            if self._gate_line is not None:
                self._heat.draw_artist(self._gate_line)
            for artist in sorted(self._overlay_artists, key=lambda item: item.get_zorder()):
                self._heat.draw_artist(artist)

    def _plot_drawn(self, event):
        self._cursor_clear()
        if self._canvas is None or self._heat is None or event.canvas.is_saving():
            return
        self._background_image = self._canvas.copy_from_bbox(self._canvas.figure.bbox)
        self._background_bounds = self._canvas.figure.bbox.bounds
        self._draw_dynamic()

    def _select_plot_row(self):
        if self._plot_timer.isActive():
            return
        dataset = self.dataset
        if dataset and dataset.get('plot_view') == 'trends':
            return
        if dataset is None or self._spectrum is None:
            self._schedule_plot()
            return
        if self._last_painted_row == (dataset["key"], dataset["row"]):
            return
        self._cancel_peak_drag(redraw=False)
        self._cursor_clear()
        self._spectrum_pool.update(dataset, dataset["row"], self._colors)
        self._refresh_peak_selection(redraw=False)
        self._gate_line.set_ydata([float(dataset["cube"].gate[dataset["row"]])] * 2)
        self._redraw_dynamic()
        self._last_painted_row = (dataset["key"], dataset["row"])

    def _redraw_dynamic(self):
        if self._background_image is not None and self._background_bounds == self._canvas.figure.bbox.bounds and not self._canvas.backend._timer.isActive():
            with try_render_lock() as available:
                if available:
                    self._canvas.restore_region(self._background_image)
                    self._draw_dynamic()
                    self._canvas.blit(self._canvas.figure.bbox)
                else:self._canvas.draw_idle()
        else:
            self._canvas.draw_idle()

    def _remember_view(self, axis):
        dataset = self.dataset
        if dataset is None or axis is not self._heat:
            return
        dataset["view"] = {**(dataset.get("view") or {}), "xlim": list(axis.get_xlim()),
            "ylim": list(axis.get_ylim()), "y_axis_log": axis.get_yscale() == "log"}
        if not self._building_plot and dataset.get("scope", 0) == 1:
            self._view_timer.start(200)

    def _navigation_axes(self):
        return tuple(axis for axis in (self._heat, self._spectrum, self._residual, *self._trend_axes) if axis is not None)

    def _sync_x_limits(self, axis):
        if self._syncing_x or axis not in (self._heat, self._spectrum):
            return
        other = self._spectrum if axis is self._heat else self._heat
        if other is None:
            return
        self._syncing_x = True
        try:
            other.set_xlim(axis.get_xlim())
        finally:
            self._syncing_x = False

    def _capture_home_limits(self):
        self._cancel_middle_pan()
        self._home_limits = {axis: (axis.get_xlim(), axis.get_ylim()) for axis in self._navigation_axes()}
        if self._heat is not None and self.dataset is not None:
            import numpy as np
            cube = self.dataset['cube']
            x = np.asarray(cube.energy, dtype=float)
            y = np.asarray(cube.gate, dtype=float)
            x = x[np.isfinite(x) & ((x > 0) if self._heat.get_xscale() == 'log' else True)]
            y = y[np.isfinite(y) & ((y > 0) if self._heat.get_yscale() == 'log' else True)]
            if x.size and y.size:
                xlim, ylim = self._home_limits[self._heat]
                full_x = (float(x.min()), float(x.max())) if x.min() != x.max() else xlim
                full_y = (float(y.min()), float(y.max())) if y.min() != y.max() else ylim
                if xlim[0] > xlim[1]: full_x = full_x[::-1]
                if ylim[0] > ylim[1]: full_y = full_y[::-1]
                self._home_limits[self._heat] = (full_x, full_y)
                for axis in (self._spectrum, self._residual):
                    if axis is not None:
                        self._home_limits[axis] = (full_x, self._home_limits[axis][1])

    def _plot_home(self):
        if self._busy or self._peak_drag is not None or self._middle_pan is not None:
            return
        self._navigation.push_current()
        for axis, (xlim, ylim) in self._home_limits.items():
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)
        self._navigation.push_current()
        self._canvas.draw_idle()

    def _plot_scroll(self, event):
        axis = event.inaxes
        if (self._busy or self._peak_drag is not None or self._middle_pan is not None
                or getattr(self._navigation, '_zoom_info', None) is not None
                or getattr(self._navigation, '_pan_info', None) is not None
                or axis not in self._navigation_axes() or not event.step):
            return
        # Work in display space so log and inverted scales zoom around the cursor.
        import numpy as np
        factor = 1.2 ** (-event.step)
        points = axis.transData.inverted().transform(
            [(event.x + (axis.bbox.x0 - event.x) * factor, event.y + (axis.bbox.y0 - event.y) * factor),
             (event.x + (axis.bbox.x1 - event.x) * factor, event.y + (axis.bbox.y1 - event.y) * factor)])
        if not np.isfinite(points).all():
            return
        self._navigation.push_current()
        axis.set_xlim(points[:, 0])
        axis.set_ylim(points[:, 1])
        self._navigation.push_current()
        self._canvas.draw_idle()

    def _middle_press(self, event):
        if (event.button != 2 or self._busy or self._peak_drag is not None
                or self._navigation.mode or event.inaxes not in self._navigation_axes()):
            return
        self._navigation.push_current()
        self._middle_pan = event.inaxes
        self._middle_pan_dataset = self.dataset
        self._middle_pan.start_pan(event.x, event.y, 1)

    def _middle_motion(self, event):
        if self._middle_pan is None:
            return
        if (self._busy or self._peak_drag is not None or self.dataset is not self._middle_pan_dataset
                or self._middle_pan not in self._navigation_axes()):
            self._cancel_middle_pan()
            return
        self._middle_pan.drag_pan(1, event.key, event.x, event.y)
        self._canvas.draw_idle()

    def _middle_release(self, event):
        if event.button != 2 or self._middle_pan is None:
            return
        if self.dataset is not self._middle_pan_dataset or self._busy:
            self._cancel_middle_pan()
            return
        self._cancel_middle_pan()
        self._navigation.push_current()
        self._canvas.draw_idle()

    def _cancel_middle_pan(self, event=None):
        if self._middle_pan is not None:
            self._middle_pan.end_pan()
        self._middle_pan = None
        self._middle_pan_dataset = None

    def _apply_view_range(self):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication
        if self._closed or self._closing or self.dataset is None or self.dataset.get("scope", 0) != 1:
            return
        if self._busy or QGuiApplication.mouseButtons() != Qt.MouseButton.NoButton:
            self._view_timer.start(100)
            return
        self._apply_scope_settings()

    def _plot_clicked(self, event):
        if self.dataset is None or event.button != 1 or event.xdata is None or event.ydata is None or self._navigation.mode:
            return
        if event.inaxes is self._heat:
            import numpy as np
            self.setParameter("row", int(np.argmin(abs(self.dataset["cube"].gate - event.ydata))))
        elif event.inaxes is self._spectrum and self._peak_edit_enabled():
            point = self._peak_at(event)
            if event.dblclick:
                self._cancel_peak_drag()
                if point is None:
                    self._start_seed(float(event.xdata))
            elif self._pick_seed:
                self._start_seed(float(event.xdata))
            elif point is not None:
                self.setParameter("manual.target", point['branch_id'])
                self._refresh_peak_selection()
                self._peak_drag = dict(key=self.dataset['key'], row=self.dataset['row'],
                    branch=point['branch_id'], x=event.x, y=event.y, energy=None)
        elif (event.inaxes is self._spectrum and self._pick_seed and not self._busy
                and not (self.dataset.get('kind') == 'PL' and self._controls_page == 'branches')):
            self._start_seed(float(event.xdata))


    def _peak_edit_enabled(self):
        return (self.dataset is not None and self.dataset.get('kind') == 'PL'
            and self._controls_page == 'branches' and not self._busy
            and self._navigation is not None and not self._navigation.mode
            and self.dataset.get('plot_view') != 'trends'
            and not self.dataset.get('detection_dirty') and not self.dataset.get('filters_pending'))

    def _refresh_peak_selection(self, redraw=True):
        """Synchronize the selected marker after page or target changes."""
        if not self._peak_edit_enabled():
            self._cancel_peak_drag(redraw=False)
        if self._peak_selection_line is not None:
            try:
                self._peak_selection_line.remove()
            except ValueError:
                pass
            self._peak_selection_line = None
        if self._spectrum is not None and self._peak_edit_enabled():
            point = next((p for p in (self.dataset.get('result') or {}).get('points', [])
                if p.get('row_index') == self.dataset['row'] and p.get('branch_id') == self._selected_branch), None)
            if point is not None:
                self._peak_selection_line = self._spectrum.axvline(point['energy'], color='tab:orange',
                    linestyle=':', linewidth=1.6, alpha=.9, zorder=19)
        if redraw and self._canvas is not None:
            self._redraw_dynamic()

    def _peak_at(self, event):
        import numpy as np
        cube = self.dataset['cube']
        row = self.dataset['row']
        enabled = {b['id'] for b in self.dataset.get('branches', []) if b.get('enabled', True)}
        points = [p for p in (self.dataset.get('result') or {}).get('points', [])
                  if p.get('row_index') == row and p.get('branch_id') in enabled]
        nearest, distance = None, 12.0
        for point in points:
            energy = point['energy']
            height = float(np.interp(energy, cube.energy, cube.Z[row]))
            px, py = self._spectrum.transData.transform((energy, height))
            d = float(np.hypot(px - event.x, py - event.y))
            if np.isfinite(d) and d <= distance:
                nearest, distance = point, d
        return nearest

    def _cancel_peak_drag(self, redraw=True):
        self._peak_drag = None
        artists, self._peak_preview = self._peak_preview, ()
        for artist in artists:
            try:
                artist.remove()
            except ValueError:
                pass
        if artists and redraw and self._canvas is not None:
            self._redraw_dynamic()

    def _peak_motion(self, event):
        drag = self._peak_drag
        if drag is None:
            return
        if (not self._peak_edit_enabled() or self.dataset['key'] != drag['key']
                or self.dataset['row'] != drag['row']):
            self._cancel_peak_drag()
            return
        if event.inaxes is not self._spectrum or event.xdata is None:
            return
        if drag['energy'] is None and (event.x-drag['x'])**2 + (event.y-drag['y'])**2 < 25:
            return
        cube = self.dataset['cube']
        energy = max(float(cube.energy.min()), min(float(cube.energy.max()), float(event.xdata)))
        drag['energy'] = energy
        if not self._peak_preview:
            line = self._spectrum.axvline(energy, color='tab:orange', linestyle='--', zorder=20)
            label = self._spectrum.text(energy, .98, '', transform=self._spectrum.get_xaxis_transform(),
                color='tab:orange', ha='left', va='top', zorder=21)
            self._peak_preview = (line, label)
        line, label = self._peak_preview
        line.set_xdata([energy, energy])
        label.set_position((energy, .98))
        label.set_text(f'{energy:.6f} eV')
        self._redraw_dynamic()

    def _peak_release(self, event):
        drag = self._peak_drag
        if drag is None:
            return
        valid = (event.button == 1 and event.inaxes is self._spectrum and
            self._peak_edit_enabled() and self.dataset['key'] == drag['key'] and
            self.dataset['row'] == drag['row'] and self._selected_branch == drag['branch'])
        energy = drag['energy']
        self._cancel_peak_drag()
        if valid and energy is not None:
            from core.pl_manual_peaks import edit_pl_peak
            from ui_qt.quick.peak_analysis_workflow import capture_request
            dataset = capture_request(self.dataset)
            draft = dict(self._manual_peak_draft(self.dataset))
            self._start('seed', lambda cancel, progress: edit_pl_peak(dataset, drag['row'], energy,
                branch_id=drag['branch'], width_mev=draft['width_mev'],
                move_limit_mev=draft['move_limit_mev'], fixed=draft['fixed']), dataset_key=drag['key'])

    def _peak_key(self, event):
        if event.key == 'home':
            self._plot_home()
        elif event.key == 'escape':
            self._cancel_peak_drag()
            if self._pick_seed:
                self._pick_seed = False
                self._publish()
        elif event.key == 'delete' and self._peak_edit_enabled():
            point = next((p for p in (self.dataset.get('result') or {}).get('points', [])
                if p.get('row_index') == self.dataset['row'] and p.get('branch_id') == self._selected_branch), None)
            if point is not None:
                self._cancel_peak_drag()
                self.invoke('manual-exclude')

    def _cursor_clear(self, *, keep_details=False):
        self._cursor_timer.stop()
        self._cursor_pending = None
        if not keep_details:
            self._cursor_last_tip = ""
        self._cursor_state = dict(text="Cursor · X: — · Y: — · Z: —", tooltip=
            "Last cursor reading (pointer outside the plot)\n" + self._cursor_last_tip if self._cursor_last_tip else
            "Move over a plot to read mouse coordinates. Heatmap Z is the input grid value before color clipping and detection smoothing. Hovering does not select a spectrum or rerun analysis.")
        self.cursorChanged.emit()

    def _cursor_leave(self, event=None):
        self._cursor_clear(keep_details=True)

    def _cursor_motion(self, event):
        from math import isfinite
        axes = {axis for axis in (self._heat, self._spectrum, self._residual) if axis is not None}
        if self.dataset is None or event.inaxes not in axes or event.xdata is None or event.ydata is None or not all(map(isfinite, (event.xdata, event.ydata))):
            self._cursor_leave()
            return
        self._cursor_pending = (event.inaxes, float(event.xdata), float(event.ydata))
        if not self._cursor_timer.isActive():
            self._cursor_timer.start(34)

    def _cursor_flush(self):
        pending, self._cursor_pending = self._cursor_pending, None
        if pending is None or self.dataset is None:
            return
        import numpy as np
        from core.plotting import _axis_edges_from_centers
        axis, x, y = pending
        cube = self.dataset["cube"]
        label = cube.gate_label + (" (" + cube.gate_unit + ")" if cube.gate_unit and cube.gate_unit not in cube.gate_label else "")
        def number(value):
            return f"{value:.6g}" if np.isfinite(value) else "—"
        name = "Heatmap" if axis is self._heat else "Residual" if axis is self._residual else "Spectrum"
        text = f"{name} · X: {x:.5f} eV"
        tip = f"{name}\nMouse X — Energy (eV): {number(x)}"
        if axis is self._heat:
            rows = np.argsort(cube.gate)
            if axis.get_yscale() == "log":
                rows = rows[cube.gate[rows] > 0]
            x_edges = _axis_edges_from_centers(cube.energy)
            y_edges = _axis_edges_from_centers(cube.gate[rows], log_scale=axis.get_yscale() == "log")
            cell = None
            if x_edges[0] <= x <= x_edges[-1] and y_edges[0] <= y <= y_edges[-1]:
                column = min(int(np.searchsorted(x_edges, x, side="right")) - 1, len(cube.energy) - 1)
                row = int(rows[min(int(np.searchsorted(y_edges, y, side="right")) - 1, len(rows) - 1)])
                cell = row, column
            z = float(cube.Z[cell]) if cell is not None else np.nan
            text += f" · Y ({label}): {number(y)} · Z (nearest): {number(z)}"
            tip += f"\nMouse Y — {label}: {number(y)}\nZ — {cube.cbar_label}: {number(z)}"
            if cell is not None:
                tip += f"\nNearest cell: Energy = {number(cube.energy[cell[1]])} eV; {label} = {number(cube.gate[cell[0]])}"
            else:
                tip += "\nOutside the heatmap grid."
            tip += "\nInput value before color clipping/normalization and extra detection smoothing; log scan axes use displayed cell boundaries."
        else:
            text += f" · Y ({cube.cbar_label}): {number(y)}"
            tip += f"\nMouse Y — {cube.cbar_label}: {number(y)}\nMouse coordinates, not snapped to a curve."
        if self.dataset.get('show_rejected') and (axis is self._spectrum or axis is self._heat and self.dataset.get('show_overlay', True)):
            from core.peak_filter_preview import REASONS
            from ui_qt.quick.peak_analysis_preview import CandidatePreviewIndex
            signature = (self.dataset['key'], id(cube), id(self.dataset.get('result')), id(self.dataset.get('candidates')))
            if signature != self._preview_signature:
                self._preview_index = CandidatePreviewIndex(self.dataset)
                self._preview_signature = signature
            nearest = self._preview_index.nearest(axis, x, y,
                spectrum_row=self.dataset['row'] if axis is self._spectrum else None)
            if nearest is not None:
                point, reason = nearest
                explanation = REASONS.get(reason, 'Reason unavailable for this saved result; apply filters again.') if reason else 'Retained by applied filters'
                status = 'Excluded' if reason else 'Retained'
                text = (f"{status} {point['polarity']} at {point['energy']:.6g} eV · {explanation} · " + text)
                tip += (f"\n{status} {point['polarity']} at {point['energy']:.6g} eV: {explanation}"
                    f"\nSNR: {point.get('snr')}; width: {point.get('width_mev'):.4g} meV. Last applied filters.")
                if self.dataset.get('filters_pending'):
                    tip += '\nDraft changes are not applied yet.'
        self._cursor_last_tip = tip
        self._cursor_state = dict(text=text, tooltip=tip)
        self.cursorChanged.emit()
