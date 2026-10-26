"""Compact export-style headers for the interactive PL/DRR inspection maps."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from matplotlib.ticker import FuncFormatter, Locator, NullLocator
from matplotlib.transforms import Bbox


class _Endpoints(Locator):
    """Follow the colorbar view interval when its normalization changes."""

    def __call__(self):
        return self.tick_values(*self.axis.get_view_interval())

    def tick_values(self, vmin, vmax):
        return [vmin, vmax]


def _compact(value, position, *, region="", significant=3):
    text = f"{value:.{significant}g}"
    if value and abs(value) < (.01 if significant == 2 else .001):
        text = f"{value:.{significant - 1}e}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        text = f"{mantissa.rstrip('0').rstrip('.') if '.' in mantissa else mantissa}e{int(exponent)}"
    return f"{region}: {text}" if region and position == 0 else text


def _bar_fraction(host, map_width):
    count = len(host.child_axes)
    base = {1: .29, 2: .46, 3: .66}[count]
    minimum = {1: 125, 2: 185, 3: 225}[count] * host.figure.dpi / 100
    return min(.74, max(base, minimum / max(1, map_width)))


@dataclass
class _MapHeaderLocator:
    """Module-level callable, so toolbar snapshots can pickle the figure."""

    host: object
    previous: object = None

    def __call__(self, axis, renderer):
        position = (self.previous(axis, renderer) if self.previous is not None
                    else axis.get_position(original=True))
        width = position.width * axis.figure.bbox.width
        available = max(12, width * (1 - _bar_fraction(self.host, width)) - 12 * axis.figure.dpi / 100)
        title = axis.title
        full = self.host._dptk_header_title
        title.set_text(full)
        for size in (9, 8.5, 8, 7.5):
            title.set_fontsize(size)
            if title.get_window_extent(renderer).width <= available:
                break
        else:
            # Full source names remain in the source controls. Fit an
            # unbroken identifier here without crossing the colorbar labels.
            text = full
            while len(text) > 3 and title.get_window_extent(renderer).width > available:
                text = text[:-1]
                title.set_text(text.rstrip() + "…")
        return position


@dataclass
class _HeaderLocator:
    heatmap: object

    def __call__(self, host, renderer):
        figure = host.figure
        position = self.heatmap.get_position()
        scale = figure.dpi / 100
        width = position.width * figure.bbox.width
        fraction = _bar_fraction(host, width)
        bar_width = width * fraction
        count = len(host.child_axes)
        each = max(1, (bar_width - 8 * scale * (count - 1)) / count)
        for child in host.child_axes:
            labels = child.get_xticklabels()
            for label in labels:
                label.set_fontsize(8 if count == 1 else 7)
            if labels:
                labels[0].set_horizontalalignment("left")
                labels[-1].set_horizontalalignment("right")
                required = sum(label.get_window_extent(renderer).width for label in labels) + 4 * scale
                if required > each:
                    size = max(5.5, (8 if count == 1 else 7) * each / required)
                    for label in labels:
                        label.set_fontsize(size)
        quantity = host._dptk_header_quantity
        quantity.set_fontsize(8)
        required = quantity.get_window_extent(renderer).width
        if required > bar_width:
            quantity.set_fontsize(max(5.5, 8 * bar_width / required))
        return Bbox.from_bounds(
            position.x1 - position.width * fraction,
            position.y1 + 5 * scale / figure.bbox.height,
            position.width * fraction,
            7 * scale / figure.bbox.height,
        )


def add_inspection_colorbar(figure, render, map_axis, *, label, title=None, host=None):
    """Attach one, two or three horizontal bars without a right-side column.

    The figure-owned host preserves the DRR reuse/dirty-region contract;
    inset children, endpoint labels and quantity text form one header group.
    """
    if host is None:
        host = figure.add_axes([0, 0, .1, .01])
    else:
        for child in tuple(host.child_axes):
            child.remove()
        host.clear()
    if title is None:
        title = getattr(host, "_dptk_header_title", map_axis.get_title())
    host._dptk_inspection_header = True
    host._dptk_header_title = str(title)
    host.set_axis_off()
    render.colorbar_host = host
    count = len(render.images)
    names = {1: ("",), 2: ("L", "R"), 3: ("L", "M", "R")}[count]
    gap = .035 if count > 1 else 0
    each = (1 - gap * (count - 1)) / count
    for index, (image, region) in enumerate(zip(render.images, names)):
        child = host.inset_axes([index * (each + gap), 0, each, 1])
        colorbar = figure.colorbar(image, cax=child, orientation="horizontal", ticklocation="top")
        colorbar.locator = _Endpoints()
        colorbar.formatter = FuncFormatter(partial(_compact, region=region, significant=3 if count == 1 else 2))
        colorbar.update_ticks()
        child.xaxis.set_minor_locator(NullLocator())
        child.xaxis.get_offset_text().set_visible(False)
        child.xaxis.set_ticks_position("top")
        child.tick_params(axis="x", labelsize=8 if count == 1 else 7, pad=1, length=2)
    quantity = host.text(1, 3.45, label, transform=host.transAxes,
                         ha="right", va="bottom", fontsize=8, clip_on=False)
    quantity._use_theme_text = True
    host._dptk_header_quantity = quantity
    host.set_axes_locator(_HeaderLocator(map_axis))
    previous = map_axis.get_axes_locator()
    if isinstance(previous, _MapHeaderLocator):
        previous = previous.previous
    map_axis.set_axes_locator(_MapHeaderLocator(host, previous))
    map_axis.set_title(title, loc="center", x=0, y=1.02, ha="left", va="bottom", pad=0, fontsize=9)
    return host
