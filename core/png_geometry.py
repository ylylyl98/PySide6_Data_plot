"""Horizontal canvas expansion shared by PNG export paths."""
import math


def widened_figsize(size, width_scale=1.0):
    """Keep height and typography unchanged while widening the figure."""
    scale = float(width_scale)
    if not math.isfinite(scale) or not 1.0 <= scale <= 3.0:
        raise ValueError("PNG width must be between 1 and 3 times the original width.")
    return float(size[0]) * scale, float(size[1])


def compact_horizontal_margins(figure, *, padding_points=10.0):
    """Give wide exports' unused side margins to their axes, without cropping.

    Measure labels (including secondary axes) and colorbars at export DPI.
    Only horizontal axes positions and figure-text anchors change; canvas size,
    vertical layout, font sizes and data limits remain intact.
    """
    axes = [axis for axis in figure.axes if axis.get_visible()]
    if not axes:
        return
    renderer = figure.canvas.get_renderer()
    width = figure.bbox.width
    padding = padding_points * figure.dpi / 72.0
    texts = [text for text in figure.texts
             if text.get_visible() and text.get_transform() == figure.transFigure]
    # Freeze the captured layout so savefig cannot undo the fitted margins.
    figure.set_layout_engine('none')
    for _ in range(4):
        boxes = [axis.get_tightbbox(renderer) for axis in axes]
        boxes += [text.get_window_extent(renderer) for text in texts if text.get_text()]
        boxes = [box for box in boxes if box is not None
                 and all(math.isfinite(value) for value in box.extents)]
        if not boxes:
            return
        left_gap = min(box.x0 for box in boxes)
        right_gap = width - max(box.x1 for box in boxes)
        if max(abs(left_gap - padding), abs(right_gap - padding)) < 0.5:
            break
        positions = [axis.get_position().frozen() for axis in axes]
        left = min(position.x0 for position in positions)
        right = max(position.x1 for position in positions)
        target_left = left + (padding - left_gap) / width
        target_right = right + (right_gap - padding) / width
        if right <= left or target_right <= target_left:
            break
        scale = (target_right - target_left) / (right - left)
        for axis, position in zip(axes, positions):
            axis.set_position((target_left + (position.x0 - left) * scale,
                               position.y0, position.width * scale, position.height))
        for text in texts:
            x, y = text.get_position()
            text.set_position((target_left + (x - left) * scale, y))
