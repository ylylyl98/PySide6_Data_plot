"""Agg figures and interaction owned exclusively by the native Power bridge."""
from __future__ import annotations

from core.typography import matplotlib_font_context, try_render_lock


class PowerPlotMixin:
    _WORKSPACE_FIELDS = ("_canvas", "_navigation", "_grid", "_map_grid", "_spectrum", "_residual",
        "_trend_axes", "_maps", "_lines", "_gate_lines", "_heatmap_renders", "_display_coordinates",
        "_caption", "_background_image", "_background_bounds", "_divider_position", "_spectrum_grid",
        "_analysis_artists", "_analysis_trends", "_analysis_residual", "_analysis_spectrum", "_plot_tool")

    def _activate_plot_workspace(self, target):
        if target == self._plot_workspace:
            return False
        if self._canvas is not None:
            width, height = self._canvas.get_width_height()
            self._plot_viewport = (width, height, self._canvas.device_pixel_ratio)
            self._canvas.backend._timer.stop()
        self._plot_workspaces[self._plot_workspace] = {key: getattr(self, key, None) for key in self._WORKSPACE_FIELDS}
        stored = self._plot_workspaces.get(target, {})
        dictionaries = {"_maps", "_lines", "_gate_lines", "_heatmap_renders", "_display_coordinates"}
        sequences = {"_trend_axes", "_analysis_artists", "_analysis_trends"}
        for key in self._WORKSPACE_FIELDS:
            setattr(self, key, stored.get(key, {} if key in dictionaries else () if key in sequences else "" if key == "_plot_tool" else None))
        self._plot_workspace = target
        if self._canvas is not None:
            self.plotBackendChanged.emit()
        return True

    def _main_plot_values(self):
        ignored = {"query", "historyFilter", "showOlder", "includeLegacy"}
        return (tuple((key, value) for key, value in sorted(self._values.items()) if key not in ignored),
                tuple(sorted(self._analysis_values.items())), tuple(sorted(self._analysis_selected_rows.items())),
                tuple(sorted(self._palette.items())), self._analysis_law_error)

    def _remember_main_plot(self):
        if self._plot_workspace == "plot" and self._stage == "plot":
            self._main_plot_context = (self._products, self._analysis_results, self._analysis_law_fits,
                self._main_plot_values(), self._canvas.get_width_height(physical=True))

    def _main_plot_unchanged(self):
        context = self._main_plot_context
        return (context is not None and context[0] is self._products and context[1] is self._analysis_results
            and context[2] is self._analysis_law_fits and context[3] == self._main_plot_values()
            and context[4] == self._canvas.get_width_height(physical=True) and self._canvas.frame_generation > 0)

    def _ensure_plot(self):
        if self._canvas is not None:
            return
        from matplotlib.figure import Figure
        from matplotlib.backend_bases import NavigationToolbar2
        from ui_qt.quick.plot_surface import QuickFigureCanvas
        from core.colormaps import register_colormaps
        register_colormaps()
        width, height, ratio = self._plot_viewport
        self._canvas = QuickFigureCanvas(Figure(figsize=(width / 100, height / 100), dpi=100))
        self._navigation = NavigationToolbar2(self._canvas)
        self._canvas.mpl_connect("draw_event", self._on_draw)
        self._canvas.mpl_connect("draw_event", self._analysis_special_drawn)
        self._canvas.mpl_connect("resize_event", lambda _: self._apply_split())
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.backend.drawFailed.connect(lambda message: self._message(message, error=True))
        if ratio != 1.:
            self._canvas.set_logical_size(width, height, ratio)
        self.plotBackendChanged.emit()

    def _schedule_plot(self):
        if not self._closed:
            self._plot_timer.start(0)

    def _clear_plot(self):
        self._plot_timer.stop()
        for state in self._plot_workspaces.values():
            canvas = state.get("_canvas")
            if canvas is not None and canvas is not self._canvas and not canvas.closed:
                canvas.close()
        self._plot_workspaces.clear()
        self._main_plot_context = None
        self._special_plot_key = None
        self._special_plot_context = None
        self._special_background = self._special_background_bounds = None
        self._refit_seed_lines, self._law_data_lines, self._law_fit_artists = [], [], {}
        self._law_axis = self._law_legend = self._special_spectrum = None
        self._grid = self._map_grid = self._spectrum = self._residual = None
        self._maps, self._lines, self._gate_lines = {}, {}, {}
        self._heatmap_renders = {}
        self._trend_axes = ()
        self._selected_row_index = None
        self._background_image = self._background_bounds = self._divider_position = None
        if self._canvas is not None and not self._canvas.closed:
            self._canvas.figure.clear()
            self._canvas.draw_idle()

    def _render(self):
        if self._closed:
            return
        try:
            if self._stage == "selection":
                return
            switched = self._activate_plot_workspace("special" if self._stage in {"refit", "law"} else "plot")
            self._ensure_plot()
            if self._stage in {"refit", "law"}:
                self._grid = self._map_grid = self._spectrum = self._residual = None
                self._maps, self._lines, self._gate_lines, self._trend_axes = {}, {}, {}, ()
                self._background_image = None
                if self._analysis_render_special(self._canvas.figure):
                    return
            if switched and self._stage == "plot" and self._main_plot_unchanged():
                self._publish()
                return
            if self._stage == "review":
                self._render_review()
            elif self._products is not None and self._products.get("mode", self._values["mode"]) == self._values["mode"]:
                self._build_plot()
        except Exception as exc:
            self._message("Power plot: " + str(exc), error=True)

    def _update_plot_colors(self):
        cubes = self._products["cubes"] if self._products is not None else {}
        if self._plot_timer.isActive() or set(self._heatmap_renders) != set(cubes):
            self._schedule_plot()
            return
        from ui_qt.quick.heatmap_style import update_heatmap_colors
        try:
            for key, cube in cubes.items():
                shown, *_ = self._display_cube(cube)
                update_heatmap_colors(self._heatmap_renders[key], shown, self._params(cube, vp=key == "VP"))
            self._background_image = None
            self._canvas.draw_idle()
        except (ValueError, TypeError):
            self._schedule_plot()

    def _style_figure(self):
        if self._canvas is None:
            return
        palette = self._palette
        figure = self._canvas.figure
        figure.set_facecolor(palette["inset"])
        for axis in figure.axes:
            axis.set_facecolor(palette["inset"])
            axis.tick_params(colors=palette["muted"], labelsize=8)
            for spine in axis.spines.values():
                spine.set_color(palette["border"])
            for text in [axis.title, axis.xaxis.label, axis.yaxis.label, *axis.texts]:
                text.set_color(palette["text"])
            for child in axis.child_axes:
                child.tick_params(colors=palette["muted"], labelsize=8)

    def _params(self, cube, *, vp=False):
        import numpy as np
        from core.plotting import HeatmapParams, SplitColorScale
        value = self._values
        split = SplitColorScale(value["splitX"], value["leftMin"], value["leftMax"], value["rightMin"], value["rightMax"], value["showBoundary"]) if value["split"] and not vp else None
        ylim = (value["ymin"], value["ymax"])
        if value["powerLog"]:
            positive = np.asarray(cube.gate)[np.isfinite(cube.gate) & (np.asarray(cube.gate) > 0)]
            if not positive.size:
                raise ValueError("Log power axis requires a positive measured power; use Linear for this sweep.")
            lower = max(ylim[0], float(min(positive)))
            upper = max(ylim[1], lower * 1.01)
            ylim = (lower, upper)
        return HeatmapParams(title=cube.title, xlabel="Photon Energy (eV)", ylabel="Power (uW)",
            cbar_label="VP" if vp else "PL corr. (a.u.)", vmin=value["vpMin"] if vp else value["vmin"],
            vmax=value["vpMax"] if vp else value["vmax"], xlim=(value["xmin"], value["xmax"]), ylim=ylim,
            cmap="RdBu_r" if vp else value["cmap"], log_scale=value["colorLog"] and not vp,
            y_axis_log=value["powerLog"], center_zero=vp, clip_outliers=value["clip"] and not vp, split_scale=split)

    def _display_cube(self, cube):
        import numpy as np
        from core.loader import DataCube
        true = np.asarray(cube.gate, float)
        indices = np.arange(len(true))
        if self._values["powerLog"]:
            indices = indices[np.isfinite(true) & (true > 0)]
        if not len(indices):
            raise ValueError("Log power axis needs positive powers. Use Linear to inspect all measured data.")
        true = true[indices]
        display = true.copy()
        epsilon = max(float(np.ptp(display)), float(np.max(abs(display))), 1.) * 1e-6
        for index in range(1, len(display)):
            if display[index] <= display[index - 1]:
                display[index] = display[index - 1] + epsilon
        shown = DataCube(cube.energy, display, cube.Z[indices], cube.gate_label, cube.title, cube.cbar_label)
        return shown, true, display, indices

    def _build_plot(self):
        from core.plotting import plot_heatmap, plain_log_ticks
        from ui_qt.inspection_colorbar import add_inspection_colorbar
        from matplotlib.ticker import PercentFormatter
        figure = self._canvas.figure
        figure.set_layout_engine(None)
        figure.clear()
        self._maps, self._lines, self._gate_lines, self._display_coordinates = {}, {}, {}, {}
        self._heatmap_renders = {}
        self._background_image = self._divider_position = None
        self._residual, self._trend_axes = None, ()
        analysis = self._analysis_enabled()
        columns = 2 if analysis else 1
        fraction = self._values["heatmapFraction"]
        self._grid = figure.add_gridspec(2, columns, height_ratios=[fraction, 1 - fraction],
            width_ratios=[2.2, 1.] if analysis else None, left=.08, right=.97, top=.91, bottom=.09, hspace=.25, wspace=.42)
        self._map_grid = self._grid[0, 0].subgridspec(1, len(self._products["cubes"]), wspace=.34)
        if analysis:
            self._spectrum_grid = self._grid[1, 0].subgridspec(2, 1, height_ratios=[3., 1.], hspace=.13)
            self._spectrum = figure.add_subplot(self._spectrum_grid[0])
            self._residual = figure.add_subplot(self._spectrum_grid[1], sharex=self._spectrum)
            self._residual.set_animated(True)
            self._spectrum.tick_params(axis="x", labelbottom=False)
            trend_grid = self._grid[:, 1].subgridspec(2, 1, hspace=.32)
            self._trend_axes = tuple(figure.add_subplot(trend_grid[index]) for index in range(2))
        else:
            self._spectrum_grid = None
            self._spectrum = figure.add_subplot(self._grid[1, 0])
        self._spectrum.set_animated(True)
        for index, (label, cube) in enumerate(self._products["cubes"].items()):
            axis = figure.add_subplot(self._map_grid[0, index])
            shown, true, display, rows = self._display_cube(cube)
            params = self._params(cube, vp=label == "VP")
            render = plot_heatmap(axis, shown, params)
            self._heatmap_renders[label] = render
            add_inspection_colorbar(figure, render, axis, label=params.cbar_label, title=label)
            if len(true) <= 14 and any(abs(a - b) > max(abs(a), 1.) * 1e-10 for a, b in zip(true, display)):
                axis.set_yticks(display)
                axis.set_yticklabels([format(float(value), ".6g") for value in true])
            if self._values["powerLog"]:
                plain_log_ticks(axis.yaxis)
            self._maps[label] = axis
            self._display_coordinates[label] = (true, display, rows)
            self._gate_lines[label] = axis.axhline(display[0], ls="--", lw=1, color="#536b86", animated=True)
            self._lines[label], = self._spectrum.plot(cube.energy, cube.Z[0], label=label, lw=1.15)
        self._spectrum.set_xlabel("" if analysis else "Photon Energy (eV)")
        self._spectrum.set_ylabel("VP (%)" if self._values["mode"] == "vp" else "PL corr. (a.u.)")
        self._spectrum.set_xlim(self._values["xmin"], self._values["xmax"])
        self._spectrum.grid(alpha=.2)
        if self._values["mode"] == "vp":
            self._spectrum.yaxis.set_major_formatter(PercentFormatter(xmax=1.))
            self._spectrum.axhline(0., color="#52647e", lw=.7)
        self._caption = self._spectrum.text(.02, .98, "", transform=self._spectrum.transAxes, va="top", fontsize=8)
        self._spectrum.callbacks.connect("xlim_changed", self._fit_spectrum_y)
        if analysis:
            self._analysis_render_trends(*self._trend_axes)
        self._update_gate(publish=False)
        self._style_figure()
        self._navigation.update()
        self._apply_split()

    def _fit_spectrum_y(self, axis=None):
        from ui_qt.quick.spectrum_limits import fit_spectrum_y
        lines = (*self._lines.values(), *self._analysis_artists)
        return fit_spectrum_y(self._spectrum if axis is None else axis, lines)

    def _update_gate(self, *, publish=True):
        if self._spectrum is None or self._products is None or not self._products["cubes"]:
            return
        import numpy as np
        cubes = self._products["cubes"]
        first = next(iter(cubes.values()))
        wanted = float(self._values["gate"])
        selected = getattr(self, "_selected_row_index", None)
        selected_rows = getattr(self, "_analysis_selected_rows", {})
        first_label = next(iter(cubes))
        selected = selected_rows.get(first_label, selected)
        first_index = selected if selected is not None and 0 <= selected < len(first.gate) else int(np.argmin(abs(first.gate - wanted)))
        used = float(first.gate[first_index]) if len(cubes) == 1 else wanted
        for label, cube in cubes.items():
            index = selected_rows.get(label)
            if index is None or not 0 <= index < len(cube.gate):
                index = first_index if cube is first else int(np.argmin(abs(cube.gate - used)))
            spectrum = np.asarray(cube.Z[index], float)
            self._lines[label].set_data(cube.energy, spectrum)
            self._lines[label].set_label(f"{label}: {float(cube.gate[index]):.6g} uW")
            true, display, rows = self._display_coordinates[label]
            exact = np.flatnonzero(rows == index)
            position = float(display[exact[0] if exact.size else np.argmin(abs(true - used))])
            self._gate_lines[label].set_ydata([position, position])
        self._values["gate"] = used
        self._caption.set_text(f"Spectrum @ {used:.9g} uW")
        self._spectrum.legend(fontsize=7, loc="upper right")
        if self._analysis_enabled():
            self._analysis_render_spectrum(self._spectrum, self._residual)
        self._fit_spectrum_y()
        if publish:
            if (self._background_image is not None and self._background_bounds == self._canvas.figure.bbox.bounds
                    and not self._canvas.backend._timer.isActive()):
                with try_render_lock() as acquired:
                    if acquired:
                        self._canvas.restore_region(self._background_image)
                        self._draw_dynamic()
                        self._canvas.blit(self._canvas.figure.bbox)
                        self._remember_main_plot()
                    else:
                        self._canvas.draw_idle()
            else:
                self._canvas.draw_idle()
            self._publish()

    def _on_draw(self, event):
        if event.canvas is not self._canvas or self._spectrum is None or event.canvas.is_saving():
            return
        self._background_image = self._canvas.copy_from_bbox(self._canvas.figure.bbox)
        self._background_bounds = self._canvas.figure.bbox.bounds
        self._draw_dynamic()
        self._remember_main_plot()
        if self._values["spectrumVisible"] and self._maps:
            renderer = self._canvas.get_renderer()
            bottom = min(axis.get_tightbbox(renderer).y0 for axis in self._maps.values())
            top = self._spectrum.bbox.y1
            position = float(1 - (bottom + top) / (2 * self._canvas.figure.bbox.height))
            if self._divider_position is None or abs(position - self._divider_position) > .0001:
                self._divider_position = position
                self._publish()

    def _draw_dynamic(self):
        with matplotlib_font_context(self._spectrum):
            if self._residual is not None:
                from core.typography import apply_figure_font
                apply_figure_font(self._residual)
            for axis in (self._spectrum, self._residual):
                if axis is not None and axis.get_visible():
                    self._canvas.figure.draw_artist(axis)
            for label, line in self._gate_lines.items():
                self._maps[label].draw_artist(line)

    def _divider_y(self):
        if self._grid is None or self._spectrum is None:
            return .7
        if self._divider_position is not None:
            return self._divider_position
        bottom = min(axis.get_position().y0 for axis in self._maps.values())
        return float(1 - (bottom + self._spectrum.get_position().y1) / 2)

    def _divider_right(self):
        return float(max((axis.get_position().x1 for axis in self._maps.values()), default=.96))

    def _apply_split(self):
        if self._grid is None or self._canvas is None or self._spectrum is None:
            return
        figure = self._canvas.figure
        width, height = figure.bbox.width, figure.bbox.height
        scale = figure.dpi / 100
        self._grid.update(left=min(.22, 65 * scale / max(1, width)), right=1 - min(.1, 28 * scale / max(1, width)),
            top=1 - min(.25, 48 * scale / max(1, height)), bottom=min(.22, 48 * scale / max(1, height)))
        visible = self._values["spectrumVisible"]
        span = (self._grid.top - self._grid.bottom) * height
        gap = min(64 * scale, span * .3) if visible else 0.
        self._grid.update(hspace=2 * gap / max(1, span - gap))
        fraction = self._values["heatmapFraction"] if visible else 1.
        self._grid.set_height_ratios([fraction, max(.001, 1 - fraction)])
        for index, axis in enumerate(self._maps.values()):
            axis.set_position(self._map_grid[0, index].get_position(figure))
            axis.set_xlabel("" if visible else "Photon Energy (eV)")
        if self._spectrum_grid is not None:
            self._spectrum.set_position(self._spectrum_grid[0].get_position(figure))
            self._residual.set_position(self._spectrum_grid[1].get_position(figure))
            self._residual.set_visible(visible)
        else:
            self._spectrum.set_position(self._grid[1, 0].get_position(figure))
        self._spectrum.set_visible(visible)
        self._background_image = self._divider_position = None
        self._canvas.draw_idle()
        self._publish()

    def _on_press(self, event):
        if self._navigation is not None and self._navigation.mode:
            return
        if self._analysis_click(event):
            return
        if event.button == 1 and event.inaxes in self._maps.values() and event.ydata is not None:
            import numpy as np
            label = next(label for label, axis in self._maps.items() if axis is event.inaxes)
            true, display, rows = self._display_coordinates[label]
            index = int(np.argmin(abs(display - event.ydata)))
            self._selected_row_index = int(rows[index]) if label == next(iter(self._maps)) else None
            self._analysis_selected_rows = {label: int(rows[index])}
            self._analysis_selected_channel = label
            self._values["gate"] = float(true[index])
            self._update_gate()

    def _auto_axes(self, axis):
        if self._products is None:
            return
        import numpy as np
        arrays = [np.asarray(cube.energy if axis == "x" else cube.gate) for cube in self._products["cubes"].values()]
        if axis == "y" and self._values["powerLog"]:
            arrays = [values[values > 0] for values in arrays]
        arrays = [values[np.isfinite(values)] for values in arrays if values.size]
        if not arrays:
            return
        low, high = min(float(np.min(array)) for array in arrays), max(float(np.max(array)) for array in arrays)
        if low >= high:
            low, high = low - max(abs(low) * .01, .01), high + max(abs(high) * .01, .01)
        for key, value in ((axis + "min", low), (axis + "max", high)):
            if not self._fixed[key]:
                self._values[key] = value
        if axis == "x" and not self._fixed["splitX"]:
            self._values["splitX"] = (self._values["xmin"] + self._values["xmax"]) / 2

    def _auto_color(self, side=None):
        if self._products is None:
            return
        import numpy as np
        value = self._values
        parts = []
        for cube in self._products["cubes"].values():
            xm = (cube.energy >= value["xmin"]) & (cube.energy <= value["xmax"])
            ym = (cube.gate >= value["ymin"]) & (cube.gate <= value["ymax"])
            if side:
                xm &= cube.energy < value["splitX"] if side == "left" else cube.energy >= value["splitX"]
            region = cube.Z[np.ix_(ym, xm)] if np.any(ym) and np.any(xm) else cube.Z if side is None else np.array([])
            finite = region[np.isfinite(region)]
            if value["colorLog"] and value["mode"] != "vp" and np.any(finite > 0):
                finite = finite[finite > 0]
            if finite.size:
                parts.append(finite)
        if not parts:
            return
        finite = np.concatenate(parts)
        vp = value["mode"] == "vp"
        low, high = (float(np.min(finite)), float(np.max(finite))) if vp else map(float, np.percentile(finite, [.01, 99.99]))
        if high <= low:
            pad = max(abs(low) * .01, .005 if vp else 1e-12)
            low, high = low - pad, high + pad
        if vp:
            low, high = max(-1., low), min(1., high)
        keys = ("vpMin", "vpMax") if vp else (side + "Min", side + "Max") if side else ("vmin", "vmax")
        bounds = [value[key] if self._fixed[key] else bound for key, bound in zip(keys, (low, high))]
        if bounds[0] >= bounds[1]:
            self._message("Auto color range conflicts with a fixed bound.")
            return
        for key, bound in zip(keys, bounds):
            value[key] = bound

    def _review_pairs(self):
        if self._review is None:
            return [], 0, 0
        import numpy as np
        keys = list(self._queue)
        reference = keys.index(self._review_state["reference"])
        other = keys.index(self._review_state.get("comparison")) if self._review_state.get("comparison") in keys else next(index for index in range(len(keys)) if index != reference)
        if other == reference:
            other = next(index for index in range(len(keys)) if index != reference)
        self._review_state["comparison"] = keys[other]
        a, b = self._review["inputs"][reference].cube, self._review["inputs"][other].cube
        pairs = [(i, int(j)) for i, power in enumerate(a.gate) for j in np.flatnonzero(abs(b.gate - power) <= .02 * np.maximum(np.maximum(abs(power), abs(b.gate)), 1e-30))]
        return pairs, reference, other

    def _render_review(self):
        if self._review is None:
            return
        import numpy as np
        from core.plotting import HeatmapParams, plot_heatmap
        from ui_qt.inspection_colorbar import add_inspection_colorbar
        figure = self._canvas.figure
        figure.set_layout_engine(None)
        figure.clear()
        self._grid = self._spectrum = self._residual = None
        self._maps, self._lines, self._gate_lines = {}, {}, {}
        self._background_image = None
        trend, spectrum, heat = figure.subplots(1, 3)
        figure.subplots_adjust(left=.06, right=.98, top=.87, bottom=.15, wspace=.5)
        for index, result in enumerate(self._review["inputs"]):
            trend.plot(result.cube.gate, np.trapezoid(result.cube.Z, result.cube.energy, axis=1), "o-", ms=3, label=f"Sweep {index + 1}")
        trend.set(xlabel="Power (uW)", ylabel="Integrated PL (a.u. eV)", title="Corrected segments")
        trend.legend(fontsize=7)
        pairs, reference, other = self._review_pairs()
        if pairs:
            pair = pairs[max(0, min(len(pairs) - 1, int(self._review_state.get("pair", 0))))]
            for index, row in zip((reference, other), pair):
                result = self._review["inputs"][index]
                spectrum.plot(result.cube.energy, result.cube.Z[row], label=f"Sweep {index + 1}: {result.cube.gate[row]:.6g} uW")
                if self._review_state.get("showOriginal"):
                    original = self._review["originals"][index]
                    original_row = int(np.argmin(abs(original.cube.gate - result.cube.gate[row])))
                    spectrum.plot(original.cube.energy, original.cube.Z[original_row], "--", alpha=.5, label=f"Original {index + 1}")
            spectrum.legend(fontsize=7)
        else:
            spectrum.text(.5, .5, "No measured powers within 2%", transform=spectrum.transAxes, ha="center")
        spectrum.set(xlabel="Photon Energy (eV)", ylabel="PL (a.u.)", title="Overlap preview")
        cube = self._review["preview"].cube
        finite = cube.Z[np.isfinite(cube.Z)]
        low, high = (float(np.min(finite)), float(np.max(finite))) if finite.size else (0., 1.)
        if low >= high:
            high = low + max(abs(low) * .01, 1e-12)
        params = HeatmapParams("", "Photon Energy (eV)", "Power (uW)", "PL corr. (a.u.)", low, high,
            (float(min(cube.energy)), float(max(cube.energy))), (float(min(cube.gate)), float(max(cube.gate))), cmap=self._values["cmap"])
        render = plot_heatmap(heat, cube, params)
        add_inspection_colorbar(figure, render, heat, label=params.cbar_label, title="Combined")
        self._style_figure()
        self._navigation.update()
        self._canvas.draw_idle()
        self._publish()
