"""Thread-scoped PNG stages without changing the Agg draw/encode sequence."""
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.image import imsave

_context = ContextVar('png_render_timings', default=None)
_labels = {'build': 'Building PNG', 'title': 'Fitting PNG title',
           'draw': 'Drawing PNG', 'encode_write': 'Encoding/writing PNG'}


@contextmanager
def collect_png_timings(progress=None):
    durations = {}
    token = _context.set((durations, progress))
    try:
        yield durations
    finally:
        _context.reset(token)


@contextmanager
def time_png_stage(name):
    context = _context.get()
    if context is None:
        yield
        return
    durations, progress = context
    if progress is not None and name in _labels:
        progress(_labels[name])
    started = perf_counter()
    try:
        yield
    finally:
        durations[name] = durations.get(name, 0.) + perf_counter() - started


class TimedPngCanvas(FigureCanvasAgg):
    def _print_pil(self, filename_or_obj, fmt, pil_kwargs, metadata=None):
        # Agg's PIL path calls FigureCanvasAgg.draw directly, so overriding
        # canvas.draw would miss the save. Keep its one-draw sequence here.
        with time_png_stage('draw'):
            FigureCanvasAgg.draw(self)
        with time_png_stage('encode_write'):
            imsave(filename_or_obj, self.buffer_rgba(), format=fmt, origin='upper',
                   dpi=self.figure.dpi, metadata=metadata, pil_kwargs=pil_kwargs)
