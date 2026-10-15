"""Redraw DRR linecut axes over a retained heatmap, including new ticks/titles."""
from __future__ import annotations


class DrrGateBlitter:
    def __init__(self, canvas):
        self.canvas = canvas
        self.axes = None
        self.spectra = ()
        self.overlays = ()
        self.artists = ()
        self._animated_before = {}
        self._background = None
        self._static_signature = None
        self._draw_cid = None
        self._full_redraw_prepared = False
        self.heatmaps = ()
        self._color_frame = None

    @staticmethod
    def spectrum_artists(spectra):
        # Keep the Axes themselves non-animated. Matplotlib includes animated
        # children when saving PNG/PDF/SVG, including a frozen async snapshot.
        return tuple(a for axis in spectra for a in axis.get_children()
                     if a is not axis.patch)

    def configure(self, spectra, overlays, heatmaps=()):
        self.disconnect()
        self.spectra = tuple(spectra)
        self.axes = self.spectra[0] if self.spectra else None
        self.overlays = tuple(overlays)
        self.heatmaps = tuple(heatmaps)
        self.artists = tuple(dict.fromkeys((*self.spectrum_artists(self.spectra), *self.overlays)))
        self._animated_before = {a: a.get_animated() for a in self.artists}
        for artist in self.artists:
            artist.set_animated(True)
        self._full_redraw_prepared = False
        self._draw_cid = self.canvas.mpl_connect('draw_event', self._on_canvas_draw)

    def matches(self, spectra, overlays, heatmaps=()):
        return (self.spectra == tuple(spectra) and self.overlays == tuple(overlays)
                and self.heatmaps == tuple(heatmaps)
                and self.artists == tuple(dict.fromkeys((*self.spectrum_artists(spectra), *overlays))))

    def invalidate(self):
        self._background = None
        self._color_frame = None

    def prepare_color_update(self):
        if (self._background is not None and self._color_frame is not None
                and not getattr(self.canvas, '_draw_pending', False)
                and self._signature() == self._static_signature):
            return self._background
        return None

    def draw_colors(self, changed, token):
        if token is None or token is not self._background or self._color_frame is None:
            return False
        return self._color_frame.redraw(self, changed)

    def _owns_axes(self):
        axes = self.canvas.figure.axes
        if (not self.spectra or any(axis not in axes for axis in self.spectra)
                or any(artist.axes not in axes for artist in self.overlays)):
            self.disconnect()
            return False
        return True

    def _signature(self):
        figure = self.canvas.figure
        return (tuple(figure.bbox.bounds), figure.dpi,
                tuple((axis, axis.get_visible(), tuple(axis.bbox.bounds),
                       # Linecut Y limits, titles and tick labels are dynamic.
                       None if axis in self.spectra else
                       (tuple(axis.get_xlim()), tuple(axis.get_ylim()),
                        axis.get_title(), axis.get_xlabel(), axis.get_ylabel(),
                        tuple((a, a.get_visible(), a.get_zorder()) for a in axis.get_children()
                              if a not in self.overlays))) for axis in figure.axes))

    def _draw_dynamic(self):
        renderer = self.canvas.get_renderer()
        # Repaint complete linecut axes: a new gate may change the title,
        # autoscaled Y limits, ticks, scientific offset and fit overlays.
        for axis in self.spectra:
            children = self.spectrum_artists((axis,))
            previous = [(a, a.get_animated()) for a in children]
            try:
                for artist, _ in previous:
                    artist.set_animated(False)
                axis.draw(renderer)
            finally:
                for artist, animated in previous:
                    artist.set_animated(animated)
        for artist in self.overlays:
            if artist.get_visible():
                artist.axes.draw_artist(artist)

    def _on_canvas_draw(self, event):
        if not self._owns_axes():
            return
        if event.canvas.is_saving():
            self.invalidate()
            return
        if self._full_redraw_prepared:
            return
        # One shared background avoids capturing another product's old
        # linecut/ticks. It also covers text outside the axes rectangles.
        self._background = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self._static_signature = self._signature()
        self._draw_dynamic()
        if self.heatmaps:
            from ui_qt.drr_color_redraw import ColorFrame
            self._color_frame = ColorFrame(self.canvas, self.heatmaps)

    def draw(self):
        if not self._owns_axes() or self._background is None:
            return False
        if self._signature() != self._static_signature:
            self.invalidate()
            return False
        self.canvas.restore_region(self._background)
        self._draw_dynamic()
        if self._color_frame is not None:
            self._color_frame.spectra_bounds = None
        self.canvas.blit(self.canvas.figure.bbox)
        return True

    def prepare_full_redraw(self):
        self.invalidate()
        self._full_redraw_prepared = True
        for artist in self.artists:
            artist.set_animated(False)

    def restore_interactive_drawing(self):
        self.invalidate()
        self._full_redraw_prepared = False
        for artist in self.artists:
            artist.set_animated(True)
        self.canvas.draw_idle()

    def disconnect(self):
        if self._draw_cid is not None:
            self.canvas.mpl_disconnect(self._draw_cid)
            self._draw_cid = None
        for artist, animated in self._animated_before.items():
            artist.set_animated(animated)
        self._animated_before = {}
        self.invalidate()
        self._static_signature = None
        self.axes = None
        self.spectra = self.overlays = self.artists = ()
        self.heatmaps = ()
