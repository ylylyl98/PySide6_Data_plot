"""Small, ownership-aware Matplotlib region drawing helper."""
from __future__ import annotations


class AxesRegionBlitter:
    def __init__(self, canvas):
        self.canvas = canvas
        self.axes = None
        self.artists = ()
        self._background = None
        self._animated_before = {}
        self._draw_cid = None
        self._full_redraw_prepared = False
        self._static_signature = None

    def configure(self, axes, artists):
        self.disconnect()
        self.axes = axes
        self.artists = tuple(artists or ())
        self._animated_before = {a: bool(a.get_animated()) for a in self.artists}
        self._draw_cid = self.canvas.mpl_connect("draw_event", self._on_canvas_draw)
        for artist in self.artists:
            artist.set_animated(True)
        self.invalidate()

    def invalidate(self):
        self._background = None

    def prepare_full_redraw(self):
        self.invalidate()
        self._full_redraw_prepared = True
        for artist in self.artists:
            artist.set_animated(False)

    def restore_interactive_drawing(self):
        if self.axes is None:
            return
        # Suppress the draw-event bridge while capturing the clean static
        # background.  Otherwise the bridge would paint the animated artist
        # into the background and the subsequent blit would double it.
        self._full_redraw_prepared = True
        for artist in self.artists:
            artist.set_animated(True)
        self.canvas.draw()
        self._background = self.canvas.copy_from_bbox(self.axes.bbox)
        self._static_signature = self._signature()
        for artist in self.artists:
            artist.set_animated(True)
        self._full_redraw_prepared = False
        self.draw()

    def draw(self) -> bool:
        if self.axes is None or self._background is None:
            return False
        signature = self._signature()
        if self._static_signature != signature:
            for artist in self.artists:
                artist.set_animated(False)
            self._full_redraw_prepared = True
            self.canvas.draw()
            self._static_signature = self._signature()
            for artist in self.artists:
                artist.set_animated(True)
            # Draw once with dynamic artists omitted, then capture that
            # clean background before blitting them interactively.
            self.canvas.draw()
            self._background = self.canvas.copy_from_bbox(self.axes.bbox)
            self._full_redraw_prepared = False
        self._blit_dynamic()
        return True

    def _blit_dynamic(self):
        if self.axes is None or self._background is None:
            return
        self.canvas.restore_region(self._background)
        for artist in self.artists:
            if artist.get_visible():
                self.axes.draw_artist(artist)
        self.canvas.blit(self.axes.bbox)

    def _signature(self):
        if self.axes is None:
            return None
        static_artists = tuple(
            (id(artist), bool(artist.get_visible()), float(artist.get_zorder()))
            for artist in self.axes.get_children()
            if artist not in self.artists
        )
        return (self.axes.get_title(), tuple(self.axes.get_xlim()),
                tuple(self.axes.get_ylim()), tuple(round(v, 3) for v in self.axes.bbox.bounds),
                tuple(t.get_text() for t in self.axes.get_xticklabels()),
                tuple(t.get_text() for t in self.axes.get_yticklabels()), static_artists)

    def _on_canvas_draw(self, _event=None):
        """Refresh static background after an ordinary/full canvas draw."""
        if self.axes is None:
            return
        # Matplotlib emits draw_event while savefig is rendering.  That
        # renderer may include animated artists, so capturing here poisons the
        # interactive background and leaves a ghost on the next blit.
        is_saving = getattr(self.canvas, "is_saving", None)
        if callable(is_saving) and is_saving():
            return
        if self._full_redraw_prepared:
            return
        self._background = self.canvas.copy_from_bbox(self.axes.bbox)
        self._static_signature = self._signature()
        # draw_event is emitted from inside Figure.draw.  Calling canvas.draw()
        # here recursively emits another draw_event and can overflow the stack.
        self._blit_dynamic()

    def disconnect(self):
        if self._draw_cid is not None:
            try:
                self.canvas.mpl_disconnect(self._draw_cid)
            except (AttributeError, RuntimeError):
                pass
            self._draw_cid = None
        for artist, animated in self._animated_before.items():
            try:
                artist.set_animated(animated)
            except (AttributeError, RuntimeError):
                pass
        self._animated_before = {}
        self._background = None
        self._static_signature = None
        self.axes = None
        self.artists = ()
