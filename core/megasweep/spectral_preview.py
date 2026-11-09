"""Bounded caches for interactive slice playback; preview never encodes video."""
from dataclasses import replace
from .spectral_cache import SpectralSliceCache

from .spectral_movie import (_check_cancel, render_frame, resolve_sequence_view,
                            timeline)


def preview_dimensions(options):
    scale = min(1., 960 / max(options.width, options.height))
    return tuple(max(2, round(value * scale / 2) * 2) for value in (options.width, options.height))


class SpectralPreviewSession(SpectralSliceCache):
    def frame(self, value, view, options, cancel=None):
        _check_cancel(cancel)
        width, height = preview_dimensions(options)
        key = (float(value), width, height, tuple(view['xlim']), tuple(view['ylim']),
               tuple(view['clim']) if view['clim'] is not None else None,
               view.get('title_precision', 6), view.get('title_reference'))
        def render():
            image = render_frame(self.get_cut(value), view, replace(options, width=width, height=height))
            _check_cancel(cancel)
            return image
        image = self.cached_frame(key, render)
        _check_cancel(cancel)
        return image

    def prepare(self, values, options, ranges, colors, progress=None, cancel=None):
        options.validate()
        view = resolve_sequence_view(self.get_cut, values, ranges, options.color_mode, colors,
                                     progress, cancel, self.statistics)
        value = view['values'][timeline(len(view['values']), options)[0][0]]
        image = self.frame(value, view, options, cancel)
        return dict(session=self, view=view, options=vars(options), image=image, value=value)
