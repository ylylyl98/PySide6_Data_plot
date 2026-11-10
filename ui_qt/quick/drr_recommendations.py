"""Explicit display suggestions; never alter processing or source measurements."""
from dataclasses import replace


class DrrRecommendationsMixin:
    def _init_recommendations(self):
        self._recommend_strength = 1
        self._recommend_status = {}

    def _recommendation_snapshot(self, state):
        loaded = not self._closed and self._raw is not None
        state['choices']['recommendStrength'] = dict(
            index=self._recommend_strength, enabled=loaded, visible=True,
            options=[dict(text=name, enabled=True) for name in
                     ('Gentle', 'Standard', 'Strong', 'Maximum')],
            accessibleName='Weak feature contrast',
            tooltip='Choose contrast, then click Weak features. Stronger contrast saturates more signal.')
        state['actions']['recommend.x'] = dict(
            text='Trim empty X edges', enabled=loaded,
            tooltip='Suggest an energy range from the full data. Keeps all scan rows and fixed limits; Auto X restores the full extent.')
        for product in ('raw', 'second'):
            enabled = loaded and (product == 'raw' or self._second is not None)
            for side in (None, 'left', 'middle', 'right'):
                suffix = product + ('.' + side if side else '')
                for mode, label in (('overall', 'Overall'), ('weak', 'Weak features')):
                    state['actions']['recommend.' + mode + '.' + suffix] = dict(
                        text=label, enabled=enabled,
                        tooltip=('Apply suggested color limits to this region. Retains Fix and the selected colormap.'
                                 + (' Uses background contrast and turns Center Zero off.' if mode == 'weak' else '')))
        for key, message in self._recommend_status.items():
            state['texts']['recommend.' + key] = dict(text=message, visible=bool(message))

    def _recommend_colors(self, mode, product, side=None):
        import numpy as np
        from core.display_recommendations import recommend_color_bounds
        from core.plotting import split_region_slices

        cube = self._raw if product == 'raw' else self._second
        if cube is None:
            return False
        xlim = tuple(self._memory.viewport.get('xlim', (self._numbers['axis.xmin'], self._numbers['axis.xmax'])))
        ylim = tuple(self._memory.viewport.get('ylim', (self._numbers['axis.ymin'], self._numbers['axis.ymax'])))
        count = self._choices['regionCount'] + 1
        if side is not None:
            sides = ('left', 'right') if count == 2 else ('left', 'middle', 'right')
            if count == 1 or side not in sides:
                return False
            slices, _ = split_region_slices(cube.energy, self._split_scale(product),
                                           (self._numbers['axis.xmin'], self._numbers['axis.xmax']))
            section = slices[sides.index(side)]
            cube = replace(cube, energy=cube.energy[section], Z=cube.Z[:, section])
        elif count > 1:
            return False  # A split view uses explicit per-region commands.
        prefix = product + '.' + (side + '_' if side else '')
        keys = (prefix + 'vmin', prefix + 'vmax')
        unlocked = [key for key in keys if not self._flags.get('fix.' + key, False)]
        if not unlocked:
            raise ValueError('Both color limits are fixed. Release a Fix to apply a recommendation.')
        suggestion = recommend_color_bounds(cube, xlim=xlim, ylim=ylim, mode=mode,
            strength=(.5, 1., 2., 4.)[self._recommend_strength], log_scale=self._flags['log'])
        if suggestion is None:
            raise ValueError('No usable values in this visible region for a color recommendation.')
        low, high = suggestion.low, suggestion.high
        if mode == 'overall' and self._flags['zero'] and not self._flags['log']:
            low, high = -max(abs(low), abs(high)), max(abs(low), abs(high))
        values = dict(self._numbers)
        for key, value in zip(keys, (low, high)):
            if key in unlocked:
                values[key] = self._quantize(key, value)
        low, high = (values[key] for key in keys)
        # Tiny/constant signals still need a representable, ordered range.
        if high <= low and len(unlocked) == 2:
            quantum = 10 ** -self._decimals(keys[0])
            low, high = low - quantum, high + quantum
            if self._flags['log']:
                low = max(low, quantum)
                high = max(high, low + quantum)
            values.update(zip(keys, (low, high)))
        if high <= low or (self._flags['log'] and low <= 0):
            raise ValueError('The recommendation conflicts with a fixed color limit. Adjust or release its Fix.')
        for key in unlocked:
            self._numbers[key] = values[key]
            self._memory.edited('numbers', key)
        if product == 'second' and side is None:
            self._second_auto_scale = False
        if mode == 'weak' and not self._flags['log'] and self._flags['zero']:
            self._flags['zero'] = False
            self._memory.edited('flags', 'zero')
        x, y = np.asarray(cube.energy), np.asarray(cube.gate)
        visible = cube.Z[np.ix_((y >= min(ylim)) & (y <= max(ylim)), (x >= min(xlim)) & (x <= max(xlim)))]
        finite = visible[np.isfinite(visible)]
        if self._flags['log']:
            finite = finite[finite > 0]
        fraction = np.mean((finite < low) | (finite > high)) if finite.size else 0.
        message = f'Applied {low:.5g} to {high:.5g}; {fraction:.1%} saturated.'
        if len(unlocked) < 2:
            message += ' Fixed endpoint retained.'
        if mode == 'weak':
            message += (' Log scale uses positive percentiles.' if self._flags['log'] else ' Center Zero off; colormap retained.')
        self._recommend_status[product + ('.' + side if side else '')] = message
        return True

    def _recommend_x(self):
        import numpy as np
        from core.display_recommendations import recommend_x_range

        cubes = tuple(cube for cube in (self._raw, self._second) if cube is not None)
        limits = recommend_x_range(cubes)
        if limits is None:
            self._recommend_status['x'] = 'No reliable empty edges found. X range unchanged; Auto X shows the full extent.'
            return True
        lo, hi = limits
        # Keep existing region geometry usable; trimming must not silently move
        # a boundary (including a fixed boundary) through a spectral feature.
        x = np.sort(np.asarray(self._raw.energy))
        count = self._choices['regionCount'] + 1
        for key in ('boundary.x0', 'boundary.x1')[:count - 1]:
            b = self._numbers[key]
            index = int(np.searchsorted(x, b))
            margin = 2 * 10 ** -self._decimals('axis.xmin')
            lo = min(lo, float(x[max(0, index - 2)]) - margin)
            hi = max(hi, float(x[min(len(x) - 1, index + 1)]) + margin)
        values = dict(self._numbers)
        unlocked = [key for key in ('axis.xmin', 'axis.xmax') if not self._flags['fix.' + key]]
        if not unlocked:
            raise ValueError('Both X limits are fixed. Release a Fix to apply the suggested range.')
        for key, value in (('axis.xmin', lo), ('axis.xmax', hi)):
            if key in unlocked:
                values[key] = self._quantize(key, value)
        if values['axis.xmax'] <= values['axis.xmin']:
            raise ValueError('The suggested X range conflicts with a fixed endpoint.')
        self._split_scale('raw', values=values, validate=True)
        for key in unlocked:
            self._numbers[key] = values[key]
            self._memory.edited('numbers', key)
        self._recommend_status.clear()
        self._recommend_status['x'] = (f"Applied X {values['axis.xmin']:.6g} to {values['axis.xmax']:.6g} eV. "
            'Y, color limits and region boundaries retained. Auto X restores the full extent (except Fix).')
        return True

    def _invoke_recommendation(self, key):
        try:
            if key == 'recommend.x':
                result = self._recommend_x()
            else:
                parts = key.split('.')
                if len(parts) not in (3, 4) or parts[1] not in ('overall', 'weak') or parts[2] not in ('raw', 'second'):
                    return False
                result = self._recommend_colors(parts[1], parts[2], parts[3] if len(parts) == 4 else None)
        except (ValueError, IndexError) as error:
            self._set_error(key, str(error))
            return False
        if result:
            self._set_error(key, '')
            self.actionRequested.emit(key)
            self._changed(key, True)
        return result
