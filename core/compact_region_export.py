"""Fixed-canvas DRR PNG headers, independent of interactive legends."""
import numpy as np
import re
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter, NullLocator


def complete_title(fig, text, left, right, *, top=.95, bottom=.85, obstacles=()):
    """Fit all text by wrapping and reducing font size, never dropping words."""
    renderer = fig.canvas.get_renderer()
    widths = sorted({(right-left) * fig.bbox.width,
                     *(box.x0 - left * fig.bbox.width - 8 for box in obstacles)}, reverse=True)
    widths = [w for w in widths if 0 < w <= (right-left) * fig.bbox.width]
    height = (top-bottom) * fig.bbox.height
    artist = fig.text(left, top, '', ha='left', va='top', fontweight='bold', linespacing=1.0)
    for size in np.arange(16., .5, -.5):
        for width in widths:
            font = FontProperties(size=size, weight='bold')
            lines, current = [], ''
            for word in re.findall(r'\$[^$]*\$|\S+', str(text)):
                trial = f'{current} {word}' if current else word
                if current and renderer.get_text_width_height_descent(trial, font, False)[0] > width:
                    lines.append(current)
                    current = ''
                if word.startswith('$') and word.endswith('$'):
                    current = (current + ' ' + word).strip()
                    continue
                # Long unbroken file identifiers must fit as well.
                for char in (' ' if current else '') + word:
                    trial = current + char
                    if current and renderer.get_text_width_height_descent(trial, font, False)[0] > width:
                        lines.append(current.rstrip())
                        current = char.lstrip()
                    else:
                        current = trial
            if current:
                lines.append(current)
            artist.set_text('\n'.join(lines))
            artist.set_fontsize(size)
            extent = artist.get_window_extent(renderer)
            if (extent.width <= width + .5 and extent.height <= height
                    and all(not extent.overlaps(box.padded(4)) for box in obstacles)):
                return artist
    raise ValueError('Title is too long to fit the fixed export header.')


def compact_colorbars(fig, render, *, left, right, bottom, label, drr_header=False):
    count = len(render.images)
    names = {1: ('',), 2: ('L', 'R'), 3: ('L', 'M', 'R')}[count]
    height = .018 if drr_header and count < 3 else .011
    tick_size = (14 if count == 1 else 11 if count == 2 else 7) if drr_header else 7
    header_boxes = []
    gap = .014
    width = (right-left-(count-1)*gap)/count
    for index, (image, name) in enumerate(zip(render.images, names)):
        axis = fig.add_axes([left+index*(width+gap), bottom, width, height])
        cb = fig.colorbar(image, cax=axis, orientation='horizontal')
        cb.set_ticks([image.norm.vmin, image.norm.vmax])
        def compact_number(value, _):
            text = f'{value:.3g}'
            if value and abs(value) < .001:
                text = f'{value:.2e}'
            if 'e' in text:
                mantissa, exponent = text.split('e')
                text = f'{mantissa.rstrip("0").rstrip(".")}e{int(exponent)}'
            return text
        if drr_header and name:
            def regional_number(value, pos, region=name, lo=image.norm.vmin):
                value_text = compact_number(value, pos)
                return f'{region}: {value_text}' if value == lo else value_text
            cb.formatter = FuncFormatter(regional_number)
        else:
            cb.formatter = FuncFormatter(compact_number)
        cb.update_ticks()
        axis.xaxis.set_minor_locator(NullLocator())
        axis.xaxis.get_offset_text().set_visible(False)
        axis.xaxis.set_ticks_position('top')
        axis.tick_params(axis='x', labelsize=tick_size, pad=1, length=2)
        labels = axis.get_xticklabels()
        labels[0].set_horizontalalignment('left')
        labels[-1].set_horizontalalignment('right')
        renderer = fig.canvas.get_renderer()
        required = sum(t.get_window_extent(renderer).width for t in labels) + 5
        size = min(tick_size, tick_size * width * fig.bbox.width / required)
        for tick in labels:
            tick.set_fontsize(size)
        if drr_header:
            header_boxes.extend(t.get_window_extent(renderer) for t in labels)
        else:
            axis.set_title(name, fontsize=8, pad=2, fontweight='bold')
    if drr_header:
        y = (max(box.y1 for box in header_boxes) + 2) / fig.bbox.height
        fig.text((left+right)/2, y, label, ha='center', va='bottom',
                 fontsize=16 if count == 1 else 9, fontweight='bold', gid='drr-colorbar-quantity')
    else:
        fig.text((left+right)/2, .964, label, ha='center', va='top', fontsize=9)
