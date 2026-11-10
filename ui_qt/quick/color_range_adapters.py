"""Adapters connecting Compare and MCD color limits to the shared slider."""
from __future__ import annotations

import math


class CompareColorAdapter:
    def __init__(self, bridge):
        self.bridge = bridge
        self._cached_context = None
        self._cached_values = {}

    def render_seconds(self):
        canvas = self.bridge._canvas
        return float(getattr(canvas, "last_render_ms", 0.) or 0.) / 1000.

    def context(self):
        b = self.bridge
        v = b._values
        return (b._closed, b._busy, v['view'], v['split'], v['log'],
                *(v[k] for k in ('xmin', 'xmax', 'ymin', 'ymax', 'splitX')),
                tuple((key, id(cube), id(cube.Z)) for key, cube in b._derived.items()),
                tuple(sorted(b._assignments.items())))

    def _keys(self, group):
        b = self.bridge
        if b._values['view'] == 'vp':
            return ('vpMin', 'vpMax') if group == 'vp' else None
        if b._values['split']:
            return (group + 'Min', group + 'Max') if group in ('left', 'right') else None
        return ('vmin', 'vmax') if group == 'intensity' else None

    def describe(self):
        b = self.bridge
        if not b._derived or b._busy or b._closed:
            return dict(groups=[])
        import numpy as np
        vp = b._values['view'] == 'vp'
        ids = ['vp'] if vp else ['left', 'right'] if b._values['split'] else ['intensity']
        groups = []
        token = self.context()
        if token != self._cached_context:
            self._cached_context, self._cached_values = token, {}
        for group in ids:
            keys = self._keys(group)
            if group not in self._cached_values:
                parts = []
                for cube in b._derived.values():
                    values = np.asarray(cube.Z)
                    if group in ('left', 'right'):
                        energy = np.asarray(cube.energy)
                        mask = energy < b._values['splitX'] if group == 'left' else energy >= b._values['splitX']
                        values = values[:, mask]
                    parts.append(values[np.isfinite(values)])
                parts = [part for part in parts if part.size]
                values = np.concatenate(parts) if parts else np.array([])
                if not vp and b._values['log']:
                    values = values[values > 0]
                self._cached_values[group] = values
            values = self._cached_values[group]
            groups.append(dict(id=group, title={'vp': 'VP', 'left': 'Left intensity', 'right': 'Right intensity', 'intensity': 'Intensity'}[group],
                low=b._values[keys[0]], high=b._values[keys[1]], values=values,
                fixedLow=b._fixed[keys[0]], fixedHigh=b._fixed[keys[1]], positive=not vp and b._values['log'], enabled=bool(values.size), canRecommend=False))
        return dict(groups=groups)

    def apply_pair(self, group, low, high):
        b = self.bridge
        keys = self._keys(group)
        if not keys or not b._derived or b._busy or b._closed:
            return False
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            return False
        if b._values['view'] == 'vp' and not (-1 <= low < high <= 1):
            return False
        if b._values['view'] != 'vp' and b._values['log'] and low <= 0:
            return False
        # Fix constrains dragging/Auto, while explicit numeric edits stay legal.
        b._values.update(zip(keys, (low, high)))
        if b._heatmap_renders:
            b._update_plot_colors()
        else:
            b._schedule_plot()
        b._publish()
        return True

    def set_fixed(self, group, endpoint, fixed):
        keys = self._keys(group)
        if keys:
            self.bridge._fixed[keys[0 if endpoint == 'low' else 1]] = bool(fixed)
            self.bridge._publish()

    def auto(self, group):
        b = self.bridge
        if not self._keys(group) or not b._derived or b._busy:
            return
        b._auto_color(group if group in ('left', 'right') else None)
        if b._heatmap_renders:
            b._update_plot_colors()
        else:
            b._schedule_plot()
        b._publish()


class McdColorAdapter:
    def __init__(self, bridge):
        self.bridge = bridge

    def render_seconds(self):
        canvas = self.bridge._canvas
        return float(getattr(canvas, "last_render_ms", 0.) or 0.) / 1000.

    def context(self):
        b = self.bridge
        return (b._closed, b._current(), id(b._products), b._values['map'], b._values['panel'], b._compact,
                *(b._values[k] for k in ('xmin', 'xmax', 'ymin', 'ymax')))

    def describe(self):
        b = self.bridge
        if b._closed or not b._current() or (b._compact and b._values['panel'] != 'map'):
            return dict(groups=[])
        cube = b._products.result.cube(b._values['map'])
        v = b._values
        return dict(groups=[dict(id='map', title='MCD', low=v['vmin'], high=v['vmax'], values=cube.Z,
            fixedLow=v['fixColors'], fixedHigh=v['fixColors'], symmetric=v['centerZero'], enabled=True, canRecommend=False)])

    def apply_pair(self, group, low, high):
        b = self.bridge
        if group != 'map' or b._closed or b._exporting or not b._current() or (b._compact and b._values['panel'] != 'map'):
            return False
        if not (math.isfinite(low) and math.isfinite(high) and -1e12 <= low < high <= 1e12):
            return False
        b._values.update(vmin=low, vmax=high)
        if not b._recolor_map():
            b._queue_plot()
        b._publish()
        return True

    def set_fixed(self, group, endpoint, fixed):
        if group == 'map':
            self.bridge.setParameter('fixColors', bool(fixed))

    def auto(self, group):
        b = self.bridge
        if group != 'map' or not b._current() or b._values['fixColors']:
            return
        from core.processing import compute_auto_limits
        bounds = compute_auto_limits(b._products.result.cube(b._values['map']))
        self.apply_pair(group, bounds.vmin, bounds.vmax)
