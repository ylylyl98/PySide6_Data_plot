"""Agg/Quick inspection and text-only cursor for native Peak Analysis."""
from __future__ import annotations

from core.typography import matplotlib_font_context,try_render_lock


class PeakAnalysisPlotMixin:
    def _plot_init(self):
        self._canvas = self._navigation = None
        self._heat = self._spectrum = self._residual = self._gate_line = None
        self._spectrum_pool = None
        self._background_image = self._background_bounds = None
        self._plot_signature = None
        self._overlay_artists = []
        self._building_plot = False
        self._plot_tool = ""
        self._cursor_pending = None
        self._cursor_state = dict(text="Cursor · X: — · Y: — · Z: —", tooltip="Move over a plot to read mouse coordinates. Hovering does not change the selected spectrum or analysis.")
        self._cursor_last_tip = ""

    def _ensure_plot(self):
        if self._canvas is not None:
            return
        from matplotlib.figure import Figure
        from matplotlib.backend_bases import NavigationToolbar2
        from ui_qt.quick.plot_surface import QuickFigureCanvas
        self._canvas = QuickFigureCanvas(Figure(figsize=(9, 7), dpi=100))
        self._navigation = NavigationToolbar2(self._canvas)
        self._canvas.mpl_connect("button_press_event", self._plot_clicked)
        self._canvas.mpl_connect("motion_notify_event", self._cursor_motion)
        self._canvas.mpl_connect("axes_leave_event", self._cursor_leave)
        self._canvas.mpl_connect("figure_leave_event", self._cursor_leave)
        self._canvas.mpl_connect("draw_event", self._plot_drawn)
        self._canvas.mpl_connect("resize_event", lambda event: self._clear_plot_background())
        self._canvas.backend.drawFailed.connect(lambda message: self._message("Peak Analysis plot: " + message, error=True))
        self.plotBackendChanged.emit()

    def _schedule_plot(self):
        if not self._closed and self.dataset is not None and not self._plot_timer.isActive():
            self._plot_timer.start(0)

    def _clear_plot_background(self):
        self._background_image = self._background_bounds = None

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
        dataset = self.dataset
        if self._closed or dataset is None:
            return
        from core.peak_plotting import build_figure
        try:
            self._ensure_plot()
            self._building_plot = True
            self._cursor_clear()
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
                self._gate_line.set_ydata([float(dataset["cube"].gate[dataset["row"]])] * 2)
                self._redraw_dynamic()
                self._publish()
                return
            self._clear_plot_background()
            figure = self._canvas.figure
            figure.set_layout_engine(None)
            self._heat, self._spectrum, self._residual, self._gate_line = build_figure(dataset, dataset["row"],
                dataset.get("show_overlay", True), dataset.get("show_residual", False) and dataset["kind"] == "PL", figure)
            from ui_qt.quick.peak_analysis_spectrum import RetainedPeakSpectrum
            self._spectrum_pool = RetainedPeakSpectrum(self._spectrum, self._residual, dataset, dataset["row"], self._colors)
            for axis in (self._spectrum, self._residual):
                if axis is not None:
                    axis.set_animated(True)
            self._gate_line.set_animated(True)
            self._overlay_artists = [artist for artist in (*self._heat.lines, *self._heat.collections)
                if artist.get_gid() in {"peak-rejected", "peak-candidates", "peak-track", "peak-uncertain"}]
            for artist in self._overlay_artists:
                artist.set_animated(True)
            self._heat.callbacks.connect("xlim_changed", self._remember_view)
            self._heat.callbacks.connect("ylim_changed", self._remember_view)
            self._remember_view(self._heat)
            self._plot_signature = self._static_plot_signature(dataset)
            self._style_figure()
            self._navigation.update()
            self._plot_tool = str(self._navigation.mode.name).lower() if self._navigation.mode else ""
            self._canvas.draw_idle()
            self._publish()
        except (ValueError, RuntimeError) as error:
            self._message(str(error), error=True)
        finally:
            self._building_plot = False

    def _static_plot_signature(self, dataset):
        view = dataset.get("view") or {}
        return (dataset["key"], id(dataset["cube"]), dataset["name"], dataset.get("product_label"),
            bool(dataset.get("show_residual", False) and dataset["kind"] == "PL"),
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
        dataset = self.dataset
        if dataset is None or self._spectrum is None:
            self._schedule_plot()
            return
        self._cursor_clear()
        self._spectrum_pool.update(dataset, dataset["row"], self._colors)
        self._gate_line.set_ydata([float(dataset["cube"].gate[dataset["row"]])] * 2)
        self._redraw_dynamic()

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
        elif event.inaxes is self._spectrum and self._pick_seed and not self._busy:
            self._start_seed(float(event.xdata))

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
        self._cursor_last_tip = tip
        self._cursor_state = dict(text=text, tooltip=tip)
        self.cursorChanged.emit()
