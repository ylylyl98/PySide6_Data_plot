"""Repaint isolated heatmap/colorbar rectangles without repainting linecuts."""
import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.transforms import Bbox
from core.typography import matplotlib_font_context


def _bounds(axes, renderer):
    boxes = [axis.get_tightbbox(renderer) for axis in axes if axis.get_visible()]
    return _union(boxes)


def _union(boxes):
    boxes = [box for box in boxes if box is not None]
    return Bbox.union(boxes) if boxes else None


class ColorFrame:
    def __init__(self, canvas, groups):
        self.canvas = canvas
        self.groups = tuple(groups)
        renderer = canvas.get_renderer()
        with matplotlib_font_context(canvas.figure):
            self.axis_bounds = {axis: _bounds((axis,), renderer) for group in self.groups for axis in group}
        self.bounds = {group: _union([self.axis_bounds[axis] for axis in group]) for group in self.groups}
        self.spectra_bounds = None
        self.blank = None

    def redraw(self, helper, changed):
        """The caller has verified unchanged data, gate, layout and view limits."""
        with matplotlib_font_context(self.canvas.figure):
            return self._redraw(helper, changed)

    def _redraw(self, helper, changed):
        canvas = self.canvas
        if (canvas.is_saving() or getattr(canvas, '_draw_pending', False)
                or any(group not in self.bounds for group in changed)
                or canvas.figure.get_facecolor()[3] != 1):
            return False
        # New color limits can change tick counts/offset text. Apply the same
        # display presentation as a full draw before measuring their bounds.
        if hasattr(canvas, 'apply_display_theme'):
            canvas.apply_display_theme()
        if hasattr(canvas, 'apply_ui_font'):
            canvas.apply_ui_font()
        renderer = canvas.get_renderer()
        # Map labels/limits are unchanged. Only colorbar labels can acquire
        # different extents; avoid relaying out every unchanged axis.
        new_hosts = {host: _bounds((host,), renderer) for _, host in changed}
        new_bounds = {group: _union([self.axis_bounds[group[0]], new_hosts[group[1]]]) for group in changed}
        boxes = [box for group in changed for box in (self.bounds[group], new_bounds[group])]
        if not boxes or any(box is None or not np.isfinite(box.extents).all() for box in boxes):
            return False
        box = Bbox.union(boxes)
        # Cover antialiased text edges and use integer pixel boundaries.
        x0, y0, x1, y1 = box.extents
        dirty = Bbox.from_extents(np.floor(x0)-2, np.floor(y0)-2, np.ceil(x1)+2, np.ceil(y1)+2)
        changed_axes = {axis for group in changed for axis in group}
        if self.spectra_bounds is None:
            self.spectra_bounds = {axis: axis.get_tightbbox(renderer) for axis in helper.spectra}
        for axis in canvas.figure.axes:
            if axis in changed_axes or not axis.get_visible():
                continue
            if axis in self.axis_bounds:
                other = self.axis_bounds[axis]
            elif axis in self.spectra_bounds:
                other = self.spectra_bounds[axis]
            else:
                other = axis.get_tightbbox(renderer)
            if other is not None and dirty.overlaps(other):
                return False
        for artist in (*canvas.figure.texts, *canvas.figure.legends):
            if artist.get_visible() and dirty.overlaps(artist.get_window_extent(renderer)):
                return False
        dirty = Bbox.intersection(dirty, canvas.figure.bbox)
        if dirty is None:
            return False
        if self.blank is None:
            # Only the figure surface, with no axes/artists, supplies the
            # eraser. This also clears old colorbar labels outside its axes.
            blank = RendererAgg(int(renderer.width), int(renderer.height), renderer.dpi)
            canvas.figure.patch.draw(blank)
            self.blank = blank
        complete = canvas.copy_from_bbox(canvas.figure.bbox)
        try:
            canvas.restore_region(helper._background)
            canvas.restore_region(self.blank.copy_from_bbox(dirty))
            from ui_qt.drr_viewport import visible_drr_meshes
            with visible_drr_meshes(canvas):
                for axis in canvas.figure.axes:
                    if axis in changed_axes:
                        axis.draw(renderer)
            updated = canvas.copy_from_bbox(dirty)
            background = canvas.copy_from_bbox(canvas.figure.bbox)
            # Keep the already rendered spectra, including automatic ticks
            # and fit annotations. Only this rectangle loses its gate overlay.
            canvas.restore_region(complete)
            canvas.restore_region(updated)
            for artist in helper.overlays:
                if artist.axes in changed_axes and artist.get_visible():
                    artist.axes.draw_artist(artist)
            helper._background = background
            helper._static_signature = helper._signature()
            self.bounds.update(new_bounds)
            self.axis_bounds.update(new_hosts)
            canvas.blit(dirty)
            return True
        except Exception:
            canvas.restore_region(complete)
            helper.invalidate()
            raise
