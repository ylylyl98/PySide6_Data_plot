"""Conservative scan-coordinate suggestions; never rewrite measured values."""
from __future__ import annotations

from itertools import combinations
import re


def _normalized(name):
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def _resolve_header(value, columns):
    if not isinstance(value, str) or not value.strip():
        return None
    if value in columns:
        return value
    matches = [name for name in columns if _normalized(name) == _normalized(value)]
    return matches[0] if len(matches) == 1 else None


def _is_record_or_signal(name, unit):
    key = _normalized(name)
    if key in {'passindex', 'fastdirection', 'index', 'row', 'rownumber', 'stepindex',
               'timestamp', 'elapsedtime', 'elapsed', 'i', 'r', 'g'}:
        return True
    if any(token in key for token in ('measured', 'readback', 'monitor', 'timestamp')):
        return True
    if key.startswith(('raw', 'ids', 'ibias', 'current', 'signal', 'response',
                       'lockin', 'resistance', 'conductance', 'rxx', 'rxy', 'gxx', 'gxy', 'didv')):
        return True
    return str(unit).strip().lower() in {'a', 'ma', 'ua', 'µa', 'na', 'pa', 'fa', 'ohm', 'ω', 's/cm'}


def suggest_transport_axes(data):
    """Return kind/x/y/source/reason/candidates, leaving ambiguous axes unset."""
    frame, columns = data['df'], data['columns']
    params = data.get('metadata', {}).get('params', {})
    notes = []
    if not isinstance(params, dict):
        params = {}
        notes.append('Invalid metadata parameters; inspected measured columns instead.')

    def result(kind, x, y, source, reason, candidates=(), channel=None):
        return dict(kind=kind, x=x, y=y, source=source, channel=channel,
                    reason=' '.join(notes + [reason]), candidates=list(candidates))

    fast =_resolve_header(params.get('axis_fast'), columns)
    slow = _resolve_header(params.get('axis_slow'), columns)
    if fast and slow and fast != slow and not frame[[fast, slow]].dropna().empty:
        x, y = slow, fast
        orientation = 'slow scan on X, fast scan on Y'
        for key in ('plot_x_resolved', 'plot_x_axis'):
            preferred = _resolve_header(params.get(key), columns)
            if preferred in (fast, slow):
                x, y = preferred, slow if preferred == fast else fast
                orientation = f'{key} selects X'
                break
        return result('map', x, y, 'metadata',
                      f'Metadata: axis_slow={slow}, axis_fast={fast}; {orientation}. '
                      f'Suggested X={x}, Y={y}. A constant slow axis may be a partial scan.')
    if params.get('axis_fast') or params.get('axis_slow'):
        notes.append('Metadata does not identify two distinct available coordinate columns; inspected measured data instead.')

    # Megasweep acquisition names its two sweep axes explicitly
    # (<name>_axis_a_set / <name>_axis_b_set); use them before guessing.
    axis_a = [name for name in columns if 'axisaset' in _normalized(name)]
    axis_b = [name for name in columns if 'axisbset' in _normalized(name)]
    if len(axis_a) == 1 and len(axis_b) == 1 and not frame[[axis_a[0], axis_b[0]]].dropna().empty:
        # Same orientation as the optical views (e.g. Doping on X, Vbias on Y).
        from .megasweep_analysis import find_ibias_column, guess_sweep_axis_columns
        x, y = guess_sweep_axis_columns([axis_a[0], axis_b[0]])
        signal = find_ibias_column([name for name in columns if name not in (x, y)])
        return result('map', x, y, 'acquisition',
                      f'Acquisition sweep axes: {axis_a[0]}, {axis_b[0]}. Suggested X={x}, Y={y}.',
                      [(x, y)], channel=signal)

    candidates = [name for name in columns
                  if not _is_record_or_signal(name, data.get('units', {}).get(name))
                  and frame[name].nunique(dropna=True) > 1]
    if not candidates:
        return result('map', None, None, 'ambiguous',
                      'No varying scan coordinate could be identified. Choose columns manually.')
    if len(candidates) == 1:
        return result('curve', candidates[0], None, 'data',
                      f'Only {candidates[0]} varies among the non-signal scan candidates. '
                      'Suggested 1D curve: X is this column; Signal is the vertical axis.')
    if len(candidates) > 24:
        return result('map', None, None, 'ambiguous',
                      'Many varying numeric columns are present. Choose scan coordinates manually.')

    grids = []
    for first, second in combinations(candidates, 2):
        pairs = frame[[first, second]].dropna().drop_duplicates()
        nx, ny = pairs[first].nunique(), pairs[second].nunique()
        size = nx * ny
        # Linked gates and continuous signal noise have no independent grid cells.
        if nx < 2 or ny < 2 or len(pairs) <= max(nx, ny) or size > 2_000_000:
            continue
        if len(pairs) / size < .1:
            continue
        changes = []
        for name in (first, second):
            values = frame[name].dropna()
            changes.append(int(values.ne(values.shift()).sum()) - 1)
        x, y = (first, second) if changes[0] <= changes[1] else (second, first)
        grids.append((x, y))
    if len(grids) == 1:
        x, y = grids[0]
        return result('map', x, y, 'data',
                      f'Measured data form one plausible scan grid: X={x}, Y={y}. '
                      'The less frequently changing coordinate is placed on X; header order breaks ties.', grids)
    if grids:
        options = '; '.join(f'X={x}, Y={y}' for x, y in grids[:6])
        return result('map', None, None, 'ambiguous',
                      f'Multiple plausible grids: {options}. Choose axes manually; linked gate representations may be equivalent.', grids)
    return result('map', None, None, 'ambiguous',
                  'Several columns vary but no independent two-dimensional grid is clear. '
                  'Choose coordinates, or select 1D curve and its scan column manually.')
