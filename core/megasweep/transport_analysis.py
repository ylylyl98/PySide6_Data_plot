"""Scalar transport CSV loading and analysis, independent of optical spectra."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .transport_axes import suggest_transport_axes


def load_transport_csv(path):
    path = Path(path)
    # Snapshot the file once: acquisition may still be appending rows.
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError('CSV is empty.')
    headers = [value.strip() for value in rows[0]]
    if not all(headers) or len(set(headers)) != len(headers):
        raise ValueError('CSV headers must be nonempty and unique.')
    rows = [row for row in rows[1:] if any(value.strip() for value in row)]
    units = {}
    known_units = {'v', 'a', 'mv', 'ma', 'ua', 'µa', 'na', 'pa', 't', 'k',
                   's', 'hz', 'ohm', 'ω', 's/cm', '#', 'deg', 'rad', 'a.u.', ''}
    if rows and all(value.strip().lower() in known_units for value in rows[0]):
        units = dict(zip(headers, rows.pop(0)))
    warnings = []
    if rows and len(rows[-1]) != len(headers):
        rows.pop()
        warnings.append('Incomplete trailing row ignored; reload after acquisition writes it.')
    if any(len(row) != len(headers) for row in rows):
        raise ValueError('CSV contains a malformed row before its last row.')
    frame = pd.DataFrame(rows, columns=headers)
    columns = []
    for name in headers:
        if name == 'FastDirection':
            frame[name] = frame[name].str.strip().str.lower()
            continue
        numeric = pd.to_numeric(frame[name], errors='coerce').replace([np.inf, -np.inf], np.nan)
        frame[name] = numeric
        if name != 'PassIndex' and numeric.notna().any():
            columns.append(name)
    if not columns:
        raise ValueError('No numeric measurement columns found.')
    metadata = {}
    metadata_path = path.with_name(path.stem + '_metadata.json')
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
            if not isinstance(metadata, dict):
                raise ValueError('Expected an object')
        except (OSError, ValueError) as exc:
            warnings.append(f'Metadata could not be read: {exc}')
            metadata = {}
    data = {'df': frame, 'columns': columns, 'units': units, 'metadata': metadata,
            'warnings': warnings, 'source_path': str(path.resolve())}
    data['axis_suggestion'] = suggest_transport_axes(data)
    return data


def _grid_axis(data, name, observed):
    """Use planned bounds only when metadata identifies this exact scan axis."""
    params = data['metadata'].get('params', {})
    if not isinstance(params, dict) or name not in (params.get('axis_fast'), params.get('axis_slow')):
        return observed
    prefix = name.lower().replace('-', '').replace(' ', '')
    try:
        start, stop, step = (float(params[prefix + '_' + suffix]) for suffix in ('start', 'stop', 'step'))
        if not all(np.isfinite([start, stop, step])) or step == 0:
            return observed
        count = int(round((stop - start) / step)) + 1
        if not 1 <= count <= 100000:
            return observed
        planned = np.sort(start + np.arange(count) * step)
        indices = np.clip(np.searchsorted(planned, observed), 0, len(planned) - 1)
        previous = np.maximum(indices - 1, 0)
        indices = np.where(abs(planned[previous] - observed) < abs(planned[indices] - observed), previous, indices)
        if len(np.unique(indices)) != len(observed):
            return observed
        if not np.allclose(planned[indices], observed, rtol=1e-9, atol=abs(step) * 1e-6):
            return observed
        # Preserve the exact measured coordinates for cuts and exports.
        planned[indices] = observed
        return planned
    except (KeyError, TypeError, ValueError, OverflowError):
        return observed


def _filtered_frame(data, direction, pass_index):
    frame = data['df']
    if direction is not None:
        if 'FastDirection' not in frame:
            raise ValueError('This file has no FastDirection column.')
        frame = frame.loc[frame['FastDirection'] == direction]
    if pass_index is not None:
        if 'PassIndex' not in frame:
            raise ValueError('This file has no PassIndex column.')
        frame = frame.loc[frame['PassIndex'] == pass_index]
    return frame


def transport_curve(data, x_name, channel, *, direction=None, pass_index=None):
    if x_name == channel or any(name not in data['columns'] for name in (x_name, channel)):
        raise ValueError('Choose different numeric scan and signal columns from this CSV.')
    frame = _filtered_frame(data, direction, pass_index)
    frame = frame.loc[frame[x_name].notna()].copy()
    if frame.empty or not frame[channel].notna().any():
        raise ValueError('No finite measurements for this selection.')
    return {'kind': 'curve', 'x_name': x_name, 'y_name': None, 'channel': channel,
            'samples': frame, 'units': data['units'], 'direction': direction, 'pass_index': pass_index}


def transport_map(data, x_name, y_name, channel, *, direction=None, pass_index=None):
    if x_name == y_name:
        raise ValueError('Choose two different coordinate columns.')
    if channel in (x_name, y_name):
        raise ValueError('Choose a signal column different from both coordinates.')
    if any(name not in data['columns'] for name in (x_name, y_name, channel)):
        raise ValueError('Choose numeric X, Y and signal columns from this CSV.')
    frame = _filtered_frame(data, direction, pass_index)
    frame = frame.loc[frame[[x_name, y_name]].notna().all(axis=1)].copy()
    if frame.empty or not frame[channel].notna().any():
        raise ValueError('No finite measurements for this selection.')
    if frame.duplicated([x_name, y_name]).any():
        raise ValueError('Duplicate coordinates: choose the scan coordinate columns or select a pass/direction before plotting.')
    x = _grid_axis(data, x_name, np.sort(frame[x_name].unique()))
    y = _grid_axis(data, y_name, np.sort(frame[y_name].unique()))
    if len(x) * len(y) > 2_000_000:
        raise ValueError('Selected columns create more than 2 million cells; choose the scan coordinates or filter a pass.')
    z = np.full((len(x), len(y)), np.nan)
    z[np.searchsorted(x, frame[x_name]), np.searchsorted(y, frame[y_name])] = frame[channel]
    xx, yy = np.meshgrid(x, y, indexing='ij')
    return {'kind': 'map', 'X2D': xx, 'Y2D': yy, 'Z2D': z, 'x_name': x_name, 'y_name': y_name,
            'channel': channel, 'samples': frame, 'units': data['units'],
            'direction': direction, 'pass_index': pass_index}


def transport_cut(result, fixed_axis, value):
    if fixed_axis not in (result['x_name'], result['y_name']):
        raise ValueError('Fixed axis must be one of the map coordinates.')
    axis = result['y_name'] if fixed_axis == result['x_name'] else result['x_name']
    # Exact choices come from measured values; never snap onto another trace silently.
    rows = result['samples'].loc[result['samples'][fixed_axis] == value].sort_values(axis)
    if rows.empty:
        raise ValueError('No measured points at this fixed coordinate.')
    return {'axis': axis, 'coordinates': rows[axis].to_numpy(),
            'values': rows[result['channel']].to_numpy(), 'samples': rows,
            'fixed_axis': fixed_axis, 'fixed_value': value, 'channel': result['channel']}
