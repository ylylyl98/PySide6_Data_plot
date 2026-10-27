"""Bounded spectrum artists matching core.peak_plotting.draw_spectrum.

The core renderer remains the scientific/export reference. This pool changes
artist data only; component models, smoothing and marker interpolation use the
same scientific functions and inputs as that renderer.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from matplotlib import rcParams
from matplotlib.colors import to_rgba

from core.peak_plotting import branch_color
from core.power_multi_peaks import multi_peak_model
from core.power_peaks import peak_model


class RetainedPeakSpectrum:
    def __init__(self, axis, residual_axis, dataset, row, palette):
        self.axis, self.residual_axis = axis, residual_axis
        lines = list(axis.lines)
        self.measured = lines.pop(0)
        smoothing = bool(dataset.get("candidates") and dataset["candidates"]["settings"]["detection_method"] == "sg")
        self.smoothed = lines.pop(0) if smoothing else None
        fit = next((item for item in (dataset.get("result") or {}).get("fits", []) if item["row_index"] == row), None)
        self.total = lines.pop(0) if fit else None
        self.components = lines
        self.annotations = list(axis.texts)
        for artist in list(axis.collections):
            artist.remove()
        self.markers = axis.scatter([], [], s=20, zorder=10)
        self.markers.set_visible(False)
        self.residual = None
        if residual_axis is not None:
            residual_lines = list(residual_axis.lines)
            self.residual = residual_lines[1] if len(residual_lines) > 1 else None
        self._dataset_signature = None
        self._legend_signature = None
        self._branch_by_id, self._branch_order = {}, {}
        self._points_by_row, self._seeds_by_row, self._fits_by_row = {}, {}, {}
        self._cycle = rcParams["axes.prop_cycle"].by_key()["color"]
        self.update(dataset, row, palette)

    def _index_dataset(self, dataset):
        result = dataset.get("result") or {}
        branches = dataset["branches"]
        signature = (id(dataset["cube"]), id(branches), len(branches), id(dataset.get("result")),
                     len(result.get("points", ())), len(result.get("fits", ())))
        if signature == self._dataset_signature:
            return
        self._dataset_signature = signature
        self._branch_by_id = {branch["id"]: branch for branch in branches}
        self._branch_order = {key: index for index, key in enumerate(self._branch_by_id)}
        points, seeds = defaultdict(dict), defaultdict(set)
        for point in result.get("points", ()):
            points[point["row_index"]].setdefault(point["branch_id"], point)
        for identity, branch in self._branch_by_id.items():
            seeds[branch["seed_row"]].add(identity)
        self._points_by_row, self._seeds_by_row = dict(points), dict(seeds)
        # The reference picks the first fit for a row if an old workspace has
        # duplicate records. Preserve that choice while avoiding repeated scans.
        fits = {}
        for fit in result.get("fits", ()):
            fits.setdefault(fit["row_index"], fit)
        self._fits_by_row = fits
        marker_cap = max((len(set(points.get(row, ())) | seeds.get(row, set()))
                          for row in set(points) | set(seeds)), default=0)
        while len(self.annotations) > marker_cap:
            self.annotations.pop().remove()
        component_cap = max((len(fit["branch_ids"]) for fit in fits.values()), default=0)
        while len(self.components) > component_cap:
            self.components.pop().remove()

    @staticmethod
    def _autoscale_y(axis):
        axis.relim(visible_only=True)
        axis.set_autoscaley_on(True)
        if np.isfinite(axis.dataLim.intervaly).any():
            axis.autoscale_view(scalex=False, scaley=True)
        else:
            # Axes.clear() in the reference resets an all-NaN row to the
            # locator's empty-data limits instead of retaining the last row.
            lo, hi = axis.yaxis.get_major_locator().nonsingular(-np.inf, np.inf)
            margin = (hi - lo) * axis.margins()[1]
            axis.set_ylim(lo - margin, hi + margin, auto=True)

    def _legend(self, handles, palette):
        handles = [line for line in handles if line.get_label() and not line.get_label().startswith("_")]
        signature = tuple((id(line), line.get_label(), to_rgba(line.get_color()), line.get_linestyle()) for line in handles)
        if signature != self._legend_signature:
            legend = self.axis.get_legend()
            if legend is None or len(legend.get_lines()) != len(handles):
                legend = self.axis.legend(handles, [line.get_label() for line in handles], fontsize=7, loc="upper right")
            else:
                for proxy, text, line in zip(legend.get_lines(), legend.get_texts(), handles):
                    proxy.set_color(line.get_color())
                    proxy.set_linestyle(line.get_linestyle())
                    proxy.set_linewidth(line.get_linewidth())
                    proxy.set_alpha(line.get_alpha())
                    text.set_text(line.get_label())
            self._legend_signature = signature
            legend.get_frame().set_facecolor(palette["inset"])
            legend.get_frame().set_edgecolor(palette["border"])
            for text in legend.get_texts():
                text.set_color(palette["text"])

    def update(self, dataset, row, palette):
        self._index_dataset(dataset)
        cube = dataset["cube"]
        x, spectrum = cube.energy, cube.Z[row]
        self.measured.set_data(x, spectrum)
        handles = [self.measured]
        pool = dataset.get("candidates")
        smoothing = bool(pool and pool["settings"]["detection_method"] == "sg")
        if smoothing:
            from core.peak_candidates import detection_spectrum
            if self.smoothed is None:
                self.smoothed, = self.axis.plot([], [], lw=1, label="SG detection signal", color=self._cycle[0])
            self.smoothed.set_data(x, detection_spectrum(x, spectrum, pool["settings"]))
            self.smoothed.set_visible(True)
            handles.append(self.smoothed)
        elif self.smoothed is not None:
            self.smoothed.set_visible(False)
        fit = self._fits_by_row.get(row)
        active_components = 0
        if fit:
            mask = (x >= fit["x_min"]) & (x <= fit["x_max"])
            xf, parameters = x[mask], fit["parameters"]
            yf = multi_peak_model(xf, parameters, model=fit["model"], origin=fit["origin"])
            if self.total is None:
                self.total, = self.axis.plot([], [], "--", lw=1.3, label="Total fit")
            self.total.set_color(self._cycle[int(smoothing) % len(self._cycle)])
            self.total.set_data(xf, yf)
            self.total.set_visible(True)
            handles.append(self.total)
            baseline = parameters[0] + parameters[1] * (xf - fit["origin"])
            for index, identity in enumerate(fit["branch_ids"]):
                branch = self._branch_by_id.get(identity)
                if not branch or not branch.get("enabled", True):
                    continue
                component = peak_model(xf, 0, 0, *parameters[2 + 3 * index:5 + 3 * index], model=fit["model"], origin=fit["origin"])
                if active_components == len(self.components):
                    line, = self.axis.plot([], [], lw=1)
                    self.components.append(line)
                line = self.components[active_components]
                line.set_data(xf, baseline + component)
                line.set_color(branch_color(branch))
                line.set_label(branch["name"])
                line.set_visible(True)
                handles.append(line)
                active_components += 1
            if self.residual_axis is not None:
                if self.residual is None:
                    self.residual, = self.residual_axis.plot([], [], lw=.8, color=self._cycle[0])
                self.residual.set_data(xf, spectrum[mask] - yf)
                self.residual.set_visible(True)
        else:
            if self.total is not None:
                self.total.set_visible(False)
            if self.residual is not None:
                self.residual.set_visible(False)
        for line in self.components[active_components:]:
            line.set_visible(False)
        points = self._points_by_row.get(row, {})
        identities = set(points) | self._seeds_by_row.get(row, set())
        identities = sorted((identity for identity in identities if identity in self._branch_by_id), key=self._branch_order.__getitem__)
        positions, colors, labels = [], [], []
        for identity in identities:
            branch = self._branch_by_id[identity]
            if not branch.get("enabled", True):
                continue
            point = points.get(identity)
            energy = point["energy"] if point else branch["seed_energy"] if branch["seed_row"] == row else None
            if energy is None:
                continue
            positions.append((energy, float(np.interp(energy, x, spectrum))))
            colors.append(branch_color(branch))
            labels.append(branch["name"])
        self.markers.set_offsets(np.asarray(positions).reshape((-1, 2)))
        self.markers.set_facecolors(colors)
        self.markers.set_edgecolors("face")
        self.markers.set_visible(bool(positions))
        for index, (position, label) in enumerate(zip(positions, labels)):
            if index == len(self.annotations):
                annotation = self.axis.annotate("", position, xytext=(3, 8), textcoords="offset points", fontsize=8)
                annotation._use_theme_text = True
                self.annotations.append(annotation)
            annotation = self.annotations[index]
            annotation.xy = position
            annotation.set_text(label)
            annotation.set_color(palette["text"])
            annotation.set_visible(True)
        for annotation in self.annotations[len(positions):]:
            annotation.set_visible(False)
        self.axis.title.set_text(f"{cube.gate_label} = {cube.gate[row]:.6g}")
        self._autoscale_y(self.axis)
        if self.residual_axis is not None:
            if fit:
                # Recomputing a blended axhline transform against a previous
                # 1e-15 residual range can introduce roundoff into its exact
                # zero baseline. The reference creates that baseline on fresh
                # (0, 1) axes before plotting; reproduce its exact data bounds.
                self.residual_axis.relim(visible_only=True)
                values = np.asarray(self.residual.get_ydata(), dtype=float)
                values = values[np.isfinite(values)]
                self.residual_axis.dataLim.intervaly = (min(0., float(values.min())) if values.size else 0.,
                                                       max(0., float(values.max())) if values.size else 0.)
                self.residual_axis.set_autoscaley_on(True)
                self.residual_axis.autoscale_view(scalex=False, scaley=True)
            else:
                # The reference's lone axhline(0) lies on the default lower
                # bound and does not request autoscale when no fit is present.
                self.residual_axis.set_ylim(0., 1., auto=True)
        self._legend(handles, palette)


__all__ = ["RetainedPeakSpectrum"]
