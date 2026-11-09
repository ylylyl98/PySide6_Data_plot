"""Stream Megasweep slice plots to silent MP4s with bounded rendering reuse."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import textwrap
import time

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .megasweep_analysis import plot_line_cut_spectrogram, spectral_slice_title, spectral_title_precision
from .spectral_cache import SpectralSliceCache

_render_lock = threading.RLock()
_publish_lock = threading.Lock()


class MovieCancelled(Exception):
    pass


@dataclass(frozen=True)
class MovieOptions:
    hold: float = .25
    speed: float = 1.
    fps: int = 30
    head: float = 0.
    tail: float = 0.
    direction: str = 'forward'
    repeats: int = 1
    width: int = 1400
    height: int = 960
    crf: int = 18
    color_mode: str = 'global'

    def validate(self):
        for name in ('hold', 'speed', 'head', 'tail'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name}: enter a finite, nonnegative value.')
        if self.hold <= 0 or self.speed <= 0:
            raise ValueError('Slice hold and speed must be positive.')
        for name, low, high in (('fps', 1, 240), ('repeats', 1, 1000),
                                ('width', 320, 7680), ('height', 240, 4320), ('crf', 0, 51)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f'{name}: enter an integer from {low} to {high}.')
        if self.width % 2 or self.height % 2:
            raise ValueError('MP4 width and height must be even pixel counts.')
        if self.hold / self.speed < 1 / self.fps - 1e-12:
            raise ValueError('Each slice must last at least one video frame. Increase hold or fps.')
        if self.direction not in ('forward', 'reverse', 'pingpong'):
            raise ValueError('Choose forward, reverse or pingpong playback.')
        if self.color_mode not in ('global', 'manual', 'frame'):
            raise ValueError('Choose global, manual or per-frame color limits.')


def select_values(values, first=None, last=None, stride=1):
    values = sorted(set(float(value) for value in values))
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError('No finite measured fixed coordinates are available.')
    first = values[0] if first is None else float(first)
    last = values[-1] if last is None else float(last)
    if not math.isfinite(first) or not math.isfinite(last) or first > last:
        raise ValueError('First coordinate must be no larger than last coordinate.')
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError('Coordinate stride must be a positive integer.')
    selected = [value for value in values if first <= value <= last][::stride]
    if not selected:
        raise ValueError('The selected range contains no measured coordinates.')
    return selected


def timeline(count, options):
    options.validate()
    if count < 1:
        raise ValueError('Select at least one valid slice.')
    order = list(range(count))
    if options.direction == 'reverse':
        order.reverse()
    elif options.direction == 'pingpong':
        order += order[-2::-1]
    segments, elapsed, written = [], 0., 0
    for _ in range(options.repeats):
        for position, index in enumerate(order):
            elapsed += options.hold / options.speed
            if position == 0:
                elapsed += options.head
            if position == len(order) - 1:
                elapsed += options.tail
            end = math.floor(elapsed * options.fps + .5)
            segments.append((index, end - written))
            written = end
    return segments


def _check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise MovieCancelled('Movie export cancelled.')


def _limits(spec, fallback):
    if not spec or spec.get('auto', True):
        return tuple(float(value) for value in fallback)
    low, high = float(spec['min']), float(spec['max'])
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError('Display/color limits must be finite with Min < Max.')
    return low, high


def _visible_signal(cut, xlim, ylim):
    energy, axis = np.asarray(cut['energy']), np.asarray(cut['axis_values'])
    def visible_cells(centers, limits):
        # Match pcolormesh(shading='auto'): a visible cell can intersect a
        # cropped range even when its center lies outside that range.
        edges = np.concatenate(([centers[0] - (centers[1] - centers[0]) / 2],
                                (centers[:-1] + centers[1:]) / 2,
                                [centers[-1] + (centers[-1] - centers[-2]) / 2]))
        return ((np.maximum(edges[:-1], edges[1:]) > limits[0]) &
                (np.minimum(edges[:-1], edges[1:]) < limits[1]))
    mask_x = visible_cells(energy, xlim)
    mask_y = visible_cells(axis, ylim)
    signal = np.asarray(cut['spectra'])[np.ix_(mask_y, mask_x)]
    return signal[np.isfinite(signal)]


def _signal_limits(low, high, derivative=False):
    if derivative:
        amplitude = max(abs(low), abs(high), 1e-12)
        return -amplitude, amplitude
    if low == high:
        high = low + max(abs(low) * .01, 1e-12)
    return float(low), float(high)


def resolve_sequence_view(cut_factory, values, ranges=None, color_mode='global', colors=None,
                          progress=None, cancel=None, statistics=None):
    """Resolve shared axes and visible color statistics in one extraction pass.

    Automatic bounds include all coordinates in each cut, so their union also
    includes every visible cell from those cuts. Manual bounds are already known.
    Scalar statistics can therefore be cached independently of the selected subset.
    """
    if color_mode not in ('global', 'manual', 'frame'):
        raise ValueError('Unknown color scale mode.')
    ranges, colors = ranges or {}, colors or {}
    manual = {axis: (_limits(ranges[axis], (0, 1)) if axis in ranges and
                     not ranges[axis].get('auto', True) else None) for axis in ('x', 'y')}
    statistics = {} if statistics is None else statistics
    visible_values, skipped = [], []
    xmin, xmax, ymin, ymax = math.inf, -math.inf, math.inf, -math.inf
    low, high = math.inf, -math.inf
    derivative, representative, structural_count = False, None, 0
    for index, value in enumerate(values):
        _check_cancel(cancel)
        key = (float(value), manual['x'], manual['y'])
        stats = statistics.get(key)
        if stats is None:
            cut = cut_factory(value)
            energy, axis = np.asarray(cut['energy']), np.asarray(cut['axis_values'])
            if len(np.unique(axis)) < 2 or len(np.unique(energy)) < 2:
                stats = dict(reason='Fewer than two distinct coordinates.')
            else:
                if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(axis)):
                    raise ValueError('Slice axes must contain finite coordinates.')
                bounds_x = (float(energy.min()), float(energy.max()))
                bounds_y = (float(axis.min()), float(axis.max()))
                finite = _visible_signal(cut, manual['x'] or bounds_x, manual['y'] or bounds_y)
                stats = dict(x=bounds_x, y=bounds_y,
                             low=float(finite.min()) if finite.size else None,
                             high=float(finite.max()) if finite.size else None,
                             derivative=cut.get('processing') == 'second_derivative',
                             title={name: cut[name] for name in ('cut_type', 'c_value_used', 'fixed_axis_label')
                                    if name in cut})
            statistics[key] = stats
        if 'reason' in stats:
            skipped.append(dict(value=float(value), reason=stats['reason']))
        else:
            structural_count += 1
            xmin, xmax = min(xmin, stats['x'][0]), max(xmax, stats['x'][1])
            ymin, ymax = min(ymin, stats['y'][0]), max(ymax, stats['y'][1])
            derivative |= stats['derivative']
            if representative is None:
                representative = stats['title']
            if stats['low'] is None:
                skipped.append(dict(value=float(value), reason='No finite signal inside the display range.'))
            else:
                visible_values.append(float(value))
                low, high = min(low, stats['low']), max(high, stats['high'])
        if progress:
            progress(index + 1, len(values), 'Finding shared coordinate ranges and color limits')
    if not structural_count:
        raise ValueError('No selected slice has two distinct points on both axes.')
    if not visible_values:
        raise ValueError('No finite signal inside the selected display range.')
    xlim, ylim = manual['x'] or (xmin, xmax), manual['y'] or (ymin, ymax)
    clim = (_limits(dict(colors, auto=False), (low, high)) if color_mode == 'manual'
            else _signal_limits(low, high, derivative) if color_mode == 'global' else None)
    precision = spectral_title_precision(visible_values)
    numeric_width = max(len(f'{value:.{precision}g}') for value in visible_values)
    return dict(xlim=xlim, ylim=ylim, clim=clim, color_mode=color_mode,
                values=visible_values, skipped=skipped, title_precision=precision,
                title_reference=_slice_title(representative, '8' * numeric_width, precision))


def _slice_title(cut, value_text=None, precision=6):
    title = spectral_slice_title(cut, precision)
    if value_text is not None:
        title = title.replace(f" = {cut['c_value_used']:.{precision}g}", ' = ' + value_text, 1)
    return '\n'.join(textwrap.fill(line, width=62) for line in title.splitlines())


def render_frame(cut, view, options):
    """Keep the Megasweep heatmap style with stable margins and white frames."""
    derivative = cut.get('processing') == 'second_derivative'
    label = cut.get('signal_label', 'PL Intensity (a.u.)')
    cmap = 'RdBu_r' if derivative or 'RC' in label else 'jet'
    clim = view['clim']
    if clim is None:
        finite = _visible_signal(cut, view['xlim'], view['ylim'])
        clim = _signal_limits(float(finite.min()), float(finite.max()), derivative)
    title = _slice_title(cut, precision=view.get('title_precision', 6))
    with _render_lock:
        figure, axes = plot_line_cut_spectrogram(
            cut, title=title, cmap=cmap, z_label=label, vmin=clim[0], vmax=clim[1],
            ylim=view['ylim'], figsize=(7, 4.8))
        try:
            # Scale typography with pixels using the existing 7-inch slice
            # layout, so smaller video dimensions do not crop fixed-size text.
            scale = min(options.width / 700, options.height / 480)
            figure.set_dpi(100)
            figure.set_size_inches(options.width / 100, options.height / 100)
            figure.set_facecolor('white')
            axes.set_facecolor('white')
            axes.set_xlim(view['xlim'])
            axes.set_ylabel(textwrap.fill(axes.get_ylabel(), width=52))
            figure.axes[1].set_ylabel(textwrap.fill(label, width=52))
            axes.set_position([.14, .14, .60, .72])
            figure.axes[1].set_position([.78, .14, .025, .72])
            axes.set_title(view.get('title_reference', title), pad=6 * scale)
            for axis in figure.axes:
                axis.tick_params(colors='black', pad=3.5 * scale, length=3.5 * scale)
                axis.xaxis.labelpad = axis.yaxis.labelpad = 4 * scale
                texts = [axis.title, axis.xaxis.label, axis.yaxis.label,
                         axis.xaxis.get_offset_text(), axis.yaxis.get_offset_text(),
                         *axis.get_xticklabels(), *axis.get_yticklabels()]
                for text in texts:
                    text.set_color('black')
                    text.set_fontsize(text.get_fontsize() * scale)
            canvas = FigureCanvasAgg(figure)
            canvas.draw()
            renderer = canvas.get_renderer()
            # Long measured column names can need several lines. Fit their
            # actual glyph bounds into fixed margins, without moving the axes.
            for axis in figure.axes:
                for text in (axis.title, axis.xaxis.label, axis.yaxis.label):
                    if not text.get_text():
                        continue
                    for _ in range(40):
                        bounds = text.get_window_extent(renderer)
                        if (bounds.x0 >= 1 and bounds.y0 >= 1 and
                                bounds.x1 <= options.width - 1 and bounds.y1 <= options.height - 1):
                            break
                        text.set_fontsize(text.get_fontsize() * .9)
                    else:
                        raise ValueError('Plot labels cannot fit this output size. Increase movie dimensions.')
            axes.title.set_text(title)
            canvas.draw()
            return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
        finally:
            figure.clear()


def find_ffmpeg():
    local = Path(__file__).resolve().parent / 'ffmpeg.exe'
    if local.is_file():
        return str(local)
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError('MP4 export requires FFmpeg. Install requirements.txt or place ffmpeg.exe beside main.py.') from exc


def _wait_process(process, cancel):
    while process.poll() is None:
        _check_cancel(cancel)
        time.sleep(.05)
    _check_cancel(cancel)


def _publish_pair(partial, destination, info, cancel=None, progress=None):
    """Reserve both names, preserving existing MP4s and orphan sidecars alike."""
    with _publish_lock:
        number = 1
        pattern = re.compile(re.escape(destination.stem) + r'_(\d{3,})(?:\.mp4|\.settings\.json)$')
        for path in destination.parent.iterdir():
            match = pattern.fullmatch(path.name)
            if match:
                number = max(number, int(match[1]) + 1)
        while True:
            _check_cancel(cancel)
            target = destination if number == 1 else destination.with_name(f'{destination.stem}_{number:03d}.mp4')
            sidecar = target.with_suffix('.settings.json')
            if target.exists() or sidecar.exists():
                number += 1
                continue
            try:
                video_file = target.open('xb')
            except FileExistsError:
                number += 1
                continue
            try:
                try:
                    settings_file = sidecar.open('x', encoding='utf-8')
                except FileExistsError:
                    video_file.close()
                    target.unlink()
                    number += 1
                    continue
                with video_file, settings_file, partial.open('rb') as source:
                    size, copied = partial.stat().st_size, 0
                    while True:
                        _check_cancel(cancel)
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        video_file.write(block)
                        copied += len(block)
                        if progress:
                            progress(int(copied * 100 / max(size, 1)), 100, 'Saving validated MP4')
                    _check_cancel(cancel)
                    json.dump(info, settings_file, ensure_ascii=False, indent=2, allow_nan=False)
                    _check_cancel(cancel)
                return target, sidecar
            except BaseException:
                video_file.close()
                target.unlink(missing_ok=True)
                sidecar.unlink(missing_ok=True)
                raise


def export_movie(cut_factory, values, destination, options=None, ranges=None, colors=None,
                 metadata=None, progress=None, cancel=None):
    cache = SpectralSliceCache(cut_factory)
    try:
        return _export_movie(cache, values, destination, options, ranges, colors, metadata, progress, cancel)
    finally:
        cache.close()


def _export_movie(cache, values, destination, options, ranges, colors, metadata, progress, cancel):
    options = options or MovieOptions()
    options.validate()
    _check_cancel(cancel)
    ffmpeg = find_ffmpeg()
    view = resolve_sequence_view(cache.get_cut, values, ranges, options.color_mode, colors,
                                 progress, cancel, cache.statistics)
    segments = timeline(len(view['values']), options)
    reuse_images = len(segments) > len({index for index, _ in segments})
    total = sum(count for _, count in segments)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.movie-', suffix='.mp4', dir=destination.parent)
    os.close(descriptor)
    partial = Path(temporary)
    flags = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    process = None
    try:
        with tempfile.TemporaryFile() as log:
            command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                       '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{options.width}x{options.height}',
                       '-r', str(options.fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                       '-preset', 'medium', '-crf', str(options.crf), '-pix_fmt', 'yuv420p',
                       '-movflags', '+faststart', str(partial)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=log, **flags)
            written = 0
            try:
                for index, count in segments:
                    _check_cancel(cancel)
                    def render():
                        image = render_frame(cache.get_cut(view['values'][index]), view, options)
                        _check_cancel(cancel)
                        return image
                    # A single pass needs no image files. Repeated values reuse
                    # full-resolution frames; preview dimensions never enter here.
                    pixels = cache.cached_frame(index, render) if reuse_images else render()
                    frame = pixels.tobytes()
                    for _ in range(count):
                        _check_cancel(cancel)
                        process.stdin.write(frame)
                        written += 1
                        if progress:
                            progress(written, total, f'Encoding {written}/{total} video frames')
                process.stdin.close()
                _wait_process(process, cancel)
            except BrokenPipeError as exc:
                log.seek(0)
                raise RuntimeError('FFmpeg stopped encoding: ' + log.read().decode('utf-8', errors='replace')[-2000:]) from exc
            if process.returncode:
                log.seek(0)
                raise RuntimeError('FFmpeg encoding failed: ' + log.read().decode('utf-8', errors='replace')[-2000:])
        if progress:
            progress(0, 0, 'Validating MP4 decode')
        with tempfile.TemporaryFile() as log:
            process = subprocess.Popen([ffmpeg, '-hide_banner', '-v', 'error', '-nostdin', '-xerror',
                                        '-i', str(partial), '-map', '0:v:0', '-f', 'null', '-'],
                                       stdout=subprocess.DEVNULL, stderr=log, **flags)
            _wait_process(process, cancel)
            if process.returncode:
                log.seek(0)
                raise RuntimeError('MP4 decode validation failed: ' + log.read().decode('utf-8', errors='replace')[-2000:])
        _check_cancel(cancel)
        info = dict(app='Megasweep Analysis', options=asdict(options), values=view['values'],
                    requested_values=[float(value) for value in values], skipped=view['skipped'],
                    view=view, frame_count=total, duration_s=total / options.fps,
                    segments=segments, metadata=metadata or {}, validation='FFmpeg full decode passed')
        target, sidecar = _publish_pair(partial, destination, info, cancel, progress)
        return dict(path=str(target), sidecar=str(sidecar), info=info)
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
        partial.unlink(missing_ok=True)
