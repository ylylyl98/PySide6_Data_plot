"""Agg canvas, batched peak markers and reused linecuts for DRR Quick."""
from __future__ import annotations

from core.typography import matplotlib_font_context,try_render_lock


class DrrAnalysisPlotMixin:
    def _plot_init(self):
        self._canvas = self._navigation = None
        self._maps, self._linecuts, self._selectors, self._derivatives = [], [], [], {}
        self._locked_colors = {}
        self._building_plot = False
        self._background = None
        self._base_background = None
        self._background_bounds = None
        self._p2p_inspection = None
        self._peak_selection = []
        self._peak_overlays = []
        self._peak_signature = None
        self._overlays_dirty = False
        self._spectrum_markers = {}

    def _ensure_plot(self):
        if self._canvas: return
        from matplotlib.figure import Figure
        from matplotlib.backend_bases import NavigationToolbar2
        from ui_qt.quick.plot_surface import QuickFigureCanvas
        self._canvas = QuickFigureCanvas(Figure(figsize=(10, 7), dpi=100))
        self._navigation = NavigationToolbar2(self._canvas)
        self._canvas.mpl_connect("button_press_event", self._clicked)
        self._canvas.mpl_connect("pick_event", self._picked)
        self._canvas.mpl_connect("draw_event", self._drawn)
        self._canvas.mpl_connect("resize_event", lambda event: self._drop_background())
        self._canvas.backend.drawFailed.connect(lambda error: self._message("DRR Analysis plot: "+error, error=True))
        self.plotBackendChanged.emit()

    def _drop_background(self):
        self._background = self._background_bounds = None
        self._base_background = None

    def _clear_plot(self):
        for selector in self._selectors: selector.disconnect_events()
        self._selectors,self._peak_selection,self._spectrum_markers = [],[],{}
        self._p2p_inspection = None
        self._peak_signature = None
        self._peak_overlays = []
        self._drop_background()
        if self._canvas:
            self._maps, self._linecuts = [], []
            self._canvas.figure.clear()
            self._canvas.draw_idle()

    def _product(self, dataset, source):
        if source == "raw": return dataset.cube
        sg = tuple(self.preview_sg.setdefault(dataset.key, (dataset.settings.sg_window, dataset.settings.sg_polyorder)))
        cached = self._derivatives.get(dataset.key)
        if cached and cached[0] == sg: return cached[1]
        kind = "preview:"+dataset.key
        if kind not in self._pending:
            cube, key = dataset.cube, dataset.key
            def run(cancellation, progress):
                from core.processing import apply_sg_derivative_energy
                cancellation.check()
                return apply_sg_derivative_energy(cube, derivative=2, window_length=sg[0], polyorder=sg[1])[0]
            def received(value):
                if self.datasets.get(key) is dataset and tuple(self.preview_sg[key]) == sg:
                    self._derivatives[key] = (sg, value)
                    self._publish(plot=True)
            self._start(kind, run, received)
        return None

    def _render(self):
        with try_render_lock() as available:
            if available:self._render_locked()
            elif not self._closed:self._plot_timer.start(16)

    def _render_locked(self):
        if self._closed or self.dataset is None: return
        try:
            self._ensure_plot()
            self._building_plot = True
            if self._values["tab"] == 0 and self._maps and self._peak_signature == self._static_peak_signature():
                # Filtering/detection changes candidates while the measured
                # map, colorbars, layout and navigation remain valid.
                self._refresh_peak_overlays()
                self._refresh_linecuts()
                return
            self._drop_background()
            for selector in self._selectors: selector.disconnect_events()
            self._selectors, self._maps, self._linecuts = [], [], []
            self._p2p_inspection = None
            self._peak_selection,self._spectrum_markers,self._peak_overlays = [],{},[]
            figure = self._canvas.figure
            figure.clear()
            figure.set_layout_engine(None)
            if self._values["tab"] == 1 and self._ensure_p2p(): self._render_p2p(figure)
            else: self._render_peaks(figure)
            self._style_figure(figure)
            figure.tight_layout()
            self._peak_signature = self._static_peak_signature() if self._values["tab"] == 0 else None
            self._navigation.update()
            self._canvas.draw_idle()
        except (RuntimeError, ValueError, IndexError) as error:
            self._message(str(error), error=True)
        finally:
            self._building_plot = False

    def _style_figure(self, figure):
        palette = self._colors
        figure.set_facecolor(palette.get("canvas",palette["inset"]))
        for axis in figure.axes:
            axis.set_facecolor(palette.get("canvas",palette["inset"]))
            axis.tick_params(colors=palette["muted"], labelsize=8)
            for text in (axis.title, axis.xaxis.label, axis.yaxis.label, axis.xaxis.get_offset_text(), axis.yaxis.get_offset_text()): text.set_color(palette["text"])
            for spine in axis.spines.values(): spine.set_color(palette["border"])
            legend = axis.get_legend()
            if legend:
                legend.get_frame().set_facecolor(palette["inset"])
                for text in legend.get_texts(): text.set_color(palette["text"])

    def _map(self, figure, axis, cube, source, *, bounds=None):
        import numpy as np
        order, energy_order = np.argsort(cube.gate), np.argsort(cube.energy)
        mesh = axis.pcolormesh(cube.energy[energy_order], cube.gate[order], cube.Z[np.ix_(order, energy_order)],
            shading="nearest", cmap="RdBu_r", rasterized=True)
        if bounds:
            axis.set_xlim(*bounds[:2])
            if bounds[2] < bounds[3]: axis.set_ylim(*bounds[2:])
        axis.set_ylabel(cube.gate_label)
        figure.colorbar(mesh, ax=axis, pad=.02)
        self._maps.append((source, cube, axis, mesh))
        self._color_limits(source, cube, axis, mesh)
        axis.callbacks.connect("xlim_changed", self._view_changed)
        axis.callbacks.connect("ylim_changed", self._view_changed)
        return mesh

    def _color_limits(self, source, cube, axis, mesh):
        import numpy as np
        limits = self._locked_colors.get(source) if self._values["colorLock"] else None
        if limits is None:
            low, high = sorted(axis.get_xlim()); bottom, top = sorted(axis.get_ylim())
            values = cube.Z[np.ix_((cube.gate >= bottom)&(cube.gate <= top), (cube.energy >= low)&(cube.energy <= high))]
            finite = values[np.isfinite(values)]
            scale = max(float(np.quantile(abs(finite), .98)), 1e-12) if len(finite) else 1.
            limits = (-scale, scale)
            if self._values["colorLock"]: self._locked_colors[source] = limits
        with mesh.norm.callbacks.blocked(): mesh.set_clim(*limits)
        mesh.changed()

    def _render_peaks(self, figure):
        import numpy as np
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import SpanSelector
        pairs = self._peak_pairs()
        axes = figure.subplots(2, len(pairs), squeeze=False, gridspec_kw={"height_ratios": [3, 1]})
        for column, (dataset, source) in enumerate(pairs):
            axis, spectrum = axes[:, column]
            cube = self._product(dataset, source)
            if cube is None:
                axis.text(.5, .5, "Preparing derivative preview…", ha="center", transform=axis.transAxes); continue
            settings = dataset.settings
            bounds = self._values["range"] if dataset.key == self.active_key or self._values["fixed"] else None
            if bounds is None:
                view = self.views.get(dataset.key)
                bounds = [*view[0], *view[1]] if view else self._full_bounds(dataset)
            self._map(figure, axis, cube, source, bounds=bounds)
            axis._drr_dataset_key, axis._drr_source = dataset.key, source
            spectrum._drr_dataset_key, spectrum._drr_source = dataset.key, source
            sg = self.preview_sg.get(dataset.key, (settings.sg_window, settings.sg_polyorder))
            axis.set_title(dataset.name[:42]+"\n"+("DRR" if source == "raw" else f"d²/dE² · SG {sg[0]}/{sg[1]}"), fontsize=9)
            axis.add_patch(Rectangle((settings.x_min, settings.y_min), settings.x_max-settings.x_min, settings.y_max-settings.y_min,
                fill=False, edgecolor="green", lw=1))
            row = int(np.argmin(abs(cube.gate-self._values["gate"])))
            gate_line = axis.axhline(cube.gate[row], color=".3", ls="--", lw=.7, animated=True)
            line, = spectrum.plot(cube.energy, cube.Z[row], lw=1)
            spectrum.set_animated(True)
            spectrum.set_xlabel("Energy (eV)")
            self._linecuts.append((cube, axis, spectrum, gate_line, line))
            self._set_linecut(cube, axis, spectrum, gate_line, line)
            self._draw_peak_overlay(dataset,source,axis)
            if dataset.key == self.active_key:
                selector = SpanSelector(axis, self._interval_selected, "horizontal", useblit=False, button=1, props=dict(alpha=.2, facecolor="orange"))
                selector.set_active(self._values["resultDrag"])
                self._selectors.append(selector)
        self._update_peak_selection()

    def _peak_pairs(self):
        mode=self._values["display"]
        if mode>=3:return [(self.datasets[key],"raw" if mode==3 else "second") for key in (self.selected_keys or [self.active_key])]
        return [(self.dataset,source) for source in (("raw","second") if mode==0 else ("raw",) if mode==1 else ("second",))]

    def _static_peak_signature(self):
        products=[]
        for dataset,source in self._peak_pairs():
            derivative=self._derivatives.get(dataset.key)
            product_id=id(dataset.cube) if source=="raw" else id(derivative[1]) if derivative else None
            settings=dataset.settings
            bounds=self._values["range"] if dataset.key==self.active_key or self._values["fixed"] else None
            if bounds is None:
                view=self.views.get(dataset.key)
                bounds=[*view[0],*view[1]] if view else self._full_bounds(dataset)
            products.append((dataset.key,dataset.name,source,product_id,self.preview_sg.get(dataset.key),
                settings.x_min,settings.x_max,settings.y_min,settings.y_max,tuple(bounds)))
        # Autosave adds row/toolbar/display metadata to views. Only the bounds
        # actually used by each map can invalidate its measured grid/layout.
        return (tuple(products),self.active_key,self._values["colorLock"],repr(self._colors))

    def _refresh_peak_overlays(self):
        for artist in self._peak_overlays:artist.remove()
        for key,source,axis,marker in self._peak_selection:marker.remove()
        self._peak_overlays,self._peak_selection=[],[]
        for source,cube,axis,mesh in self._maps:
            self._draw_peak_overlay(self.datasets[axis._drr_dataset_key],source,axis)
        self._overlays_dirty=True

    def _draw_peak_overlay(self,dataset,source,axis):
        if not self._values["showPeaks"]:return
        from matplotlib.collections import LineCollection
        result=self._display_result(dataset.key)
        points=(result or {}).get("products",{}).get(source,{}).get("points",[])
        artists=[]
        for polarity,marker in (("peak","o"),("dip","v")):
            for uncertain in (False,True):
                batch=[point for point in points if point["polarity"]==polarity and (point["status"]!="accepted" or point.get("confidence")=="low")==uncertain]
                if not batch:continue
                artist=axis.scatter([point["energy"] for point in batch],[point["y"] for point in batch],s=14,
                    marker="x" if uncertain else marker,color="gray" if uncertain else None,
                    **({} if uncertain else dict(facecolors="none",edgecolors="black")),picker=5)
                artist._drr_peaks=(dataset.key,source,batch);artists.append(artist)
        settings=dataset.settings
        if settings.mode=="seed" and source==settings.source:
            artists.append(axis.scatter([settings.seed_energy],[settings.seed_y],marker="*",s=95,c="#ffe100",edgecolors="black",linewidths=.8,zorder=45))
        tracks={};rows={float(value):index for index,value in enumerate(sorted((result or {}).get("y_values",[])))}
        for point in points:
            if point["status"]=="accepted" and point.get("confidence")!="low" and point.get("track_id",0)>0:tracks.setdefault(point["track_id"],[]).append(point)
        segments=[]
        for tracked in tracks.values():
            tracked.sort(key=lambda point:rows.get(float(point["y"]),0))
            segments.extend([(left["energy"],left["y"]),(right["energy"],right["y"])] for left,right in zip(tracked,tracked[1:])
                if rows.get(float(right["y"]),0)-rows.get(float(left["y"]),0)==1)
        if segments:artists.append(axis.add_collection(LineCollection(segments,colors="black",linewidths=.6,alpha=.65,zorder=39)))
        for artist in artists:artist.set_animated(True)
        self._peak_overlays.extend(artists)
        marker=axis.scatter([],[],s=55,facecolors="none",edgecolors="red",linewidths=1.5,animated=True)
        self._peak_selection.append((dataset.key,source,axis,marker))

    def _set_linecut(self, cube, axis, spectrum, gate_line, line):
        import numpy as np
        row = int(np.argmin(abs(cube.gate-self._values["gate"])))
        value = float(cube.gate[row])
        gate_line.set_ydata([value, value])
        line.set_ydata(cube.Z[row])
        spectrum.set_xlim(axis.get_xlim())
        spectrum.set_title(f"{cube.gate_label} = {value:.6g}", fontsize=9)
        visible = (cube.energy >= min(axis.get_xlim())) & (cube.energy <= max(axis.get_xlim()))
        local = cube.Z[row, visible]
        local = local[np.isfinite(local)]
        if local.size:
            low, high = float(local.min()), float(local.max())
            pad = max((high-low)*.08, abs(high)*.01, 1e-10)
            spectrum.set_ylim(low-pad, high+pad)
        for artist in self._spectrum_markers.pop(spectrum,[]): artist.remove()
        if self._values["showPeaks"]:
            key,source = getattr(spectrum,"_drr_dataset_key",None),getattr(spectrum,"_drr_source",None)
            result = self._display_result(key)
            points = [point for point in (result or {}).get("products",{}).get(source,{}).get("points",[]) if abs(point["y"]-value) <= 1e-9]
            if points:
                artists = []
                for uncertain in (False,True):
                    energies = [point["energy"] for point in points if (point["status"] == "uncertain" or point.get("confidence") == "low") == uncertain]
                    if energies: artists.append(spectrum.vlines(energies,0,1,transform=spectrum.get_xaxis_transform(),colors="black",linewidth=.7,linestyles=":" if uncertain else "--",alpha=.7))
                self._spectrum_markers[spectrum] = artists

    def _update_peak_selection(self):
        import numpy as np
        for key,source,axis,marker in self._peak_selection:
            result = self._display_result(key)
            selected = next((point for point in (result or {}).get("products",{}).get(source,{}).get("points",[]) if key == self.active_key and self._point_id(source,point) == self._selected_point),None)
            marker.set_offsets([[selected["energy"],selected["y"]]] if selected else np.empty((0,2)))

    def _refresh_linecuts(self):
        if self._values["tab"] == 1:
            self._publish(plot=True); return
        if self._canvas is None: return
        self._update_peak_selection()
        for values in self._linecuts: self._set_linecut(*values)
        if self._background is not None and self._background_bounds == self._canvas.figure.bbox.bounds and not self._canvas.backend._timer.isActive():
            self._blit_dynamic()
        else: self._canvas.draw_idle()

    def _blit_dynamic(self):
        with try_render_lock() as available:
            if available:
                if self._overlays_dirty and self._base_background is not None:
                    self._canvas.restore_region(self._base_background)
                    self._draw_peak_overlays()
                    self._background=self._canvas.copy_from_bbox(self._canvas.figure.bbox)
                    self._overlays_dirty=False
                self._canvas.restore_region(self._background);self._draw_dynamic();self._canvas.blit(self._canvas.figure.bbox)
            else:self._canvas.draw_idle()

    def _draw_peak_overlays(self):
        if not self._peak_overlays:return
        with matplotlib_font_context(self._canvas.figure):
            for artist in sorted(self._peak_overlays,key=lambda item:item.get_zorder()):artist.axes.draw_artist(artist)

    def _draw_dynamic(self):
        with matplotlib_font_context(self._canvas.figure):
            for cube, axis, spectrum, gate_line, line in self._linecuts:
                self._canvas.figure.draw_artist(spectrum)
                axis.draw_artist(gate_line)
            for key,source,axis,marker in self._peak_selection: axis.draw_artist(marker)
            if self._p2p_inspection:
                key,cube,record,map_axis,spectrum,map_cursor,metric_cursors = self._p2p_inspection
                self._canvas.figure.draw_artist(spectrum)
                map_axis.draw_artist(map_cursor)
                for cursor in metric_cursors: cursor.axes.draw_artist(cursor)

    def _drawn(self, event):
        if self._canvas and (self._linecuts or self._p2p_inspection) and not event.canvas.is_saving():
            self._base_background = self._canvas.copy_from_bbox(self._canvas.figure.bbox)
            self._draw_peak_overlays()
            self._background = self._canvas.copy_from_bbox(self._canvas.figure.bbox)
            self._background_bounds = self._canvas.figure.bbox.bounds
            self._overlays_dirty=False
            self._draw_dynamic()

    def _view_changed(self, axis):
        if self._building_plot or self._closed: return
        if self._values["tab"] == 0 and getattr(axis, "_drr_dataset_key", None) == self.active_key:
            self._values["range"] = [*sorted(axis.get_xlim()), *sorted(axis.get_ylim())]
            self._save_view()
            for source, cube, current, mesh in self._maps:
                if current is axis: self._color_limits(source, cube, current, mesh)
            self._refresh_linecuts()
            self._publish(changed=True)
        elif self._values["tab"] == 1 and self.p2p and getattr(axis, "_drr_p2p_map", False):
            bounds = [*sorted(axis.get_xlim()), *sorted(axis.get_ylim())]
            self.p2p["display_bounds"] = bounds
            self.p2p["display_ranges"][self.active_key] = bounds
            self._publish(changed=True)

    def _interval_selected(self, low, high):
        if low >= high: return
        bounds = self._effective_filter(self.active_key)
        bounds.update(x_min=float(low), x_max=float(high), limit_x=True)
        self.result_filters[self.active_key] = bounds
        self._values["resultDrag"] = False
        for selector in self._selectors: selector.set_active(False)
        self._queue_filter(); self._publish(changed=True)

    def _clicked(self, event):
        if event.button != 1 or event.xdata is None or event.ydata is None or self._navigation.mode: return
        if self._values["tab"] == 1:
            if getattr(event.inaxes, "_drr_p2p_map", False):
                import numpy as np
                self.setParameter("p2p.row", int(np.argmin(abs(self.dataset.cube.gate-event.ydata)))+1)
            return
        key = getattr(event.inaxes, "_drr_dataset_key", None)
        if key is None: return
        if key != self.active_key: self._activate(key)
        if self._values["seedPick"]:
            from dataclasses import replace
            from core.drr_analysis_workspace import apply_settings
            source = getattr(event.inaxes, "_drr_source", "raw")
            is_map = any(axis is event.inaxes for _, _, axis, _ in self._maps)
            settings = replace(self.dataset.settings, seed_energy=float(event.xdata), seed_y=float(event.ydata) if is_map else self._values["gate"],
                source=source, mode="seed", polarity="peaks" if self.dataset.settings.polarity == "both" else self.dataset.settings.polarity)
            apply_settings(self.dataset, settings)
            self._values["seedPick"] = False
            self._update_table(); self._publish(changed=True, plot=True)
        elif not self._values["resultDrag"] and any(axis is event.inaxes for _, _, axis, _ in self._maps):
            self.setParameter("gate", float(event.ydata))

    def _picked(self, event):
        if not len(event.ind): return
        peaks = getattr(event.artist, "_drr_peaks", None)
        if peaks:
            key, source, points = peaks
            if key != self.active_key: self._activate(key)
            point_id = self._point_id(source, points[int(event.ind[0])])
            index = next((index for index, (src, point) in enumerate(self._points) if self._point_id(src, point) == point_id), None)
            if index is not None: self.selectPeak(index)
            return
        curve = getattr(event.artist, "_drr_p2p", None)
        if curve:
            import numpy as np
            key, order, values = curve
            record = next(record for record in self.p2p["records"] if record["dataset_key"] == key)
            indices = order[np.asarray(event.ind, dtype=int)]
            coordinates = event.artist.axes.transData.transform(np.column_stack((record["result"][indices, 0], values[indices])))
            mouse = event.mouseevent
            chosen = indices[np.argmin(np.sum((coordinates-[mouse.x, mouse.y])**2, axis=1))]
            if key != self.active_key: self._activate(key)
            bounds = record["bounds"]
            source_rows = np.flatnonzero((self.dataset.cube.gate>=bounds[2])&(self.dataset.cube.gate<=bounds[3]))
            self.setParameter("p2p.row", int(source_rows[chosen])+1)

    def _navigate(self, action):
        if self._navigation is None: return
        getattr(self._navigation, action)()
        self._values["plotTool"] = str(self._navigation.mode.name).lower() if self._navigation.mode else ""
        self._publish()

    def _render_p2p(self, figure):
        import numpy as np
        from ui_qt.quick.drr_analysis_p2p import draw_curves
        from matplotlib.widgets import SpanSelector
        from matplotlib.patches import Rectangle
        state, dataset = self.p2p, self.dataset
        records = [record for record in state["records"] if record["dataset_key"] in self.datasets]
        if not records:
            axis = figure.add_subplot()
            axis.text(.5,.5,"Calculate P2P to inspect all datasets",ha="center",transform=axis.transAxes)
            return
        compare = bool(state["view_mode"])
        metrics_count = 2 if compare and state.get("comparison_metric") == 1 else 1
        inspection = not compare or state.get("show_inspection")
        if inspection:
            grid = figure.add_gridspec(max(2, metrics_count), 2, width_ratios=[1.1, 1.])
            map_axis = figure.add_subplot(grid[0, 0])
            spectrum = figure.add_subplot(grid[1, 0])
            metric_axes = [figure.add_subplot(grid[index, 1]) for index in range(metrics_count)]
            if metrics_count == 1: metric_axes[0].set_subplotspec(grid[:, 1])
            record = next((record for record in records if record["dataset_key"] == self.active_key), None)
            cube = dataset.cube
            self._map(figure, map_axis, cube, "raw", bounds=state["display_bounds"])
            map_axis._drr_p2p_map = True
            low, high, bottom, top = state["bounds"]
            if not state["limit_y"]: bottom, top = float(cube.gate.min()), float(cube.gate.max())
            map_axis.add_patch(Rectangle((low,bottom),high-low,top-bottom,fill=False,edgecolor="orange",lw=1.2))
            for value in (low, high): map_axis.axvline(value, color="orange", ls="--", lw=.7)
            selector = SpanSelector(map_axis, self._p2p_interval_selected,"horizontal",useblit=False,button=1,props=dict(alpha=.2,facecolor="orange"))
            self._selectors.append(selector)
            row = max(0,min(len(cube.gate)-1,int(state["inspection_rows"].get(self.active_key,1))-1))
            map_cursor = map_axis.axhline(cube.gate[row],color="gray",ls="--",lw=.8,animated=True)
            spectrum.plot(cube.energy,cube.Z[row],lw=1,label="Raw DRR")
            processed = record.get("processed_Z") if record else None
            if processed is not None: spectrum.plot(cube.energy,processed[row],lw=1.4,label="SG DRR")
            spectrum.axvspan(low,high,alpha=.15,color="orange")
            if record and record["result"] is not None:
                result = record["result"]
                selected = self._p2p_result_row(record,cube,row)
                if len(selected):
                    values = result[selected[0]]
                    spectrum.scatter([values[4],values[5]],[values[2],values[3]],c=["tab:blue","tab:red"],s=30)
                    spectrum.set_title(f"Y = {cube.gate[row]:.6g} · P2P = {values[1]:.5g}",fontsize=9)
            spectrum.set_xlim(*state["display_bounds"][:2]); spectrum.set_xlabel("Energy (eV)")
            spectrum.legend(fontsize=8)
            map_axis.set_title(dataset.name[:42],fontsize=9)
        else: metric_axes = np.atleast_1d(figure.subplots(metrics_count,1,sharex=True))
        draw_curves(metric_axes,records,self.datasets,state,compare=compare)
        if inspection:
            metric_cursors = [axis.axvline(dataset.cube.gate[row],color="gray",ls="--",lw=.8,animated=True) for axis in metric_axes]
            spectrum.set_animated(True)
            self._p2p_inspection = (dataset.key,cube,record,map_axis,spectrum,map_cursor,metric_cursors)

    def _refresh_p2p_inspection(self):
        if not self._p2p_inspection or self._p2p_inspection[0] != self.active_key:
            self._publish(plot=True); return
        import numpy as np
        key,cube,record,map_axis,spectrum,map_cursor,metric_cursors = self._p2p_inspection
        row = max(0,min(len(cube.gate)-1,int(self.p2p["inspection_rows"].get(key,1))-1))
        y = cube.gate[row]
        map_cursor.set_ydata([y,y])
        for cursor in metric_cursors: cursor.set_xdata([y,y])
        spectrum.clear()
        spectrum.plot(cube.energy,cube.Z[row],lw=1,label="Raw DRR")
        if record and record.get("processed_Z") is not None: spectrum.plot(cube.energy,record["processed_Z"][row],lw=1.4,label="SG DRR")
        spectrum.axvspan(*self.p2p["bounds"][:2],alpha=.15,color="orange")
        if record and record["result"] is not None:
            result = record["result"]
            selected = self._p2p_result_row(record,cube,row)
            if len(selected):
                values = result[selected[0]]
                spectrum.scatter([values[4],values[5]],[values[2],values[3]],c=["tab:blue","tab:red"],s=30)
                spectrum.set_title(f"Y = {y:.6g} · P2P = {values[1]:.5g}",fontsize=9)
        spectrum.set_xlim(*self.p2p["display_bounds"][:2]); spectrum.set_xlabel("Energy (eV)")
        spectrum.legend(fontsize=8)
        self._style_figure(self._canvas.figure)
        if self._background is not None and self._background_bounds == self._canvas.figure.bbox.bounds and not self._canvas.backend._timer.isActive():
            self._blit_dynamic()
        else: self._canvas.draw_idle()

    @staticmethod
    def _p2p_result_row(record,cube,row):
        import numpy as np
        bounds = record["bounds"]
        source_rows = np.flatnonzero((cube.gate>=bounds[2])&(cube.gate<=bounds[3]))
        return np.flatnonzero(source_rows == row)

    def _p2p_interval_selected(self, low, high):
        if self.p2p and low < high:
            self.p2p["bounds"][:2] = [float(low),float(high)]
            self.p2p["pending"] = True
            self._p2p_generation += 1
            self._p2p_timer.start()
            self._publish(changed=True,plot=True)

    def _inspection_export_figure(self):
        # Compatibility helper for direct callers. Normal export captures state
        # in the bridge and calls the independent builder on its worker.
        from copy import deepcopy
        from dataclasses import replace
        from ui_qt.quick.drr_analysis_p2p import copy_state
        datasets={key:replace(dataset,result=None,provenance=deepcopy(dataset.provenance)) for key,dataset in self.datasets.items()}
        return inspection_export_figure(datasets,copy_state(self.p2p),self.active_key,deepcopy(self._values),dict(self._locked_colors))


class _InspectionExportPlot(DrrAnalysisPlotMixin):
    """Plain captured renderer: no QObject, live canvas or GUI callbacks."""

    @property
    def dataset(self):
        return self.datasets[self.active_key]

    def _view_changed(self,axis):
        pass


def inspection_export_figure(datasets,state,active_key,values,locked_colors):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    renderer=_InspectionExportPlot()
    renderer._plot_init()
    renderer.datasets,renderer.active_key=datasets,active_key
    renderer.p2p=dict(state,show_inspection=True)
    renderer._values,renderer._locked_colors=values,locked_colors
    figure=Figure(figsize=(state["plot_style"]["width"],state["plot_style"]["height"]),dpi=100)
    FigureCanvasAgg(figure)
    try:
        renderer._render_p2p(figure)
        figure.tight_layout()
        # Independent export figures draw every inspection artist normally.
        for artist in figure.findobj():
            if artist.get_animated(): artist.set_animated(False)
        for axis in figure.axes:
            for signal in list(axis.callbacks.callbacks):
                for callback_id in list(axis.callbacks.callbacks[signal]): axis.callbacks.disconnect(callback_id)
        return figure
    finally:
        for selector in renderer._selectors:selector.disconnect_events()
