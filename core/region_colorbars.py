"""Compact, vertically stacked legends for three-region heatmaps."""
from matplotlib.ticker import FuncFormatter, NullLocator, LinearLocator
from matplotlib.transforms import Bbox


def three_preview_axes(parent):
    """Keep label space in points when the window or display DPI changes."""
    parent.set_axis_off()
    axes = []
    for index in range(3):
        axis = parent.inset_axes([0, 0, 1, .2])
        def locate(_axis, renderer, i=index):
            box = parent.get_position()
            figure = parent.figure
            gap = (20 / 72) / figure.get_figheight()
            height = min(.24 * box.height, max(.02 * box.height, .36 * box.height - gap))
            return Bbox.from_bounds(box.x0, box.y1 - height - i * .36 * box.height,
                                    box.width, height)
        axis.set_axes_locator(locate)
        axes.append(axis)
    return axes


def add_three_colorbars(figure, render, axes, *, label='', fontsize=8):
    x1, x2 = render.boundaries
    titles = (f'L: x ≤ {x1:.5g}', f'M: {x1:.5g} to {x2:.5g}', f'R: x ≥ {x2:.5g}')
    bars = []
    for image, axis, title in zip(render.images, axes, titles):
        cb = figure.colorbar(image, cax=axis, orientation='vertical')
        # Endpoints suffice on short bars; compact labels avoid scientific
        # offset text colliding with the next region's heading.
        cb.locator = LinearLocator(2)
        cb.formatter = FuncFormatter(lambda value, pos: f'{value:.3g}')
        cb.update_ticks()
        cb.ax.yaxis.set_minor_locator(NullLocator())
        cb.ax.yaxis.get_offset_text().set_visible(False)
        cb.ax.tick_params(labelsize=fontsize)
        cb.ax.set_title(title, fontsize=fontsize, loc='left', pad=4)
        bars.append(cb)
    if label:
        bars[1].set_label(label, fontsize=fontsize)
    return bars
