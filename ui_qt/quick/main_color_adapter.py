"""DRR and PL adapter for the shared color range controls."""
from __future__ import annotations
import math

class MainColorAdapter:
    def __init__(self, bridge):
        self.bridge = bridge
        self._cut_token = None
        self._cuts = None

    def render_seconds(self):
        return getattr(self.bridge._canvas, 'last_render_ms', 0.) / 1000.

    def _state(self):
        b = self.bridge
        return b._drr_display if b._workflow == 'drr' else b._pl_display if b._workflow == 'pl' else None

    def _product(self):
        b = self.bridge; state = self._state()
        if state is None: return ''
        if b._workflow == 'drr':
            # The Display tabs select the edited product even in a single-map
            # view. Keep colormap, exact inputs and sliders on that same bank.
            return ('raw', 'second')[state._choices['colorProduct']]
        return state.values['scale']

    def context(self):
        b = self.bridge; s = self._state()
        if s is None or b._cube is None or getattr(b, '_closed', False): return (b._workflow, None, getattr(b, '_sequence', 0))
        product = self._product()
        busy = bool(getattr(b, '_busy', False))
        if b._workflow == 'drr':
            cube = s._raw if product == 'raw' else s._second
            return (b._workflow, getattr(b, '_sequence', 0), id(cube), product, s._choices['regionCount'], s._numbers['boundary.x0'], s._numbers['boundary.x1'], s._flags['log'], s._flags['zero'], busy)
        return (b._workflow, getattr(b, '_sequence', 0), id(b._cube), product, s.geometry['count'], s.geometry['x0'], s.geometry['x1'], busy)

    def describe(self):
        b = self.bridge; s = self._state()
        if s is None or b._cube is None or getattr(b, '_closed', False): return {}
        import numpy as np
        product = self._product(); drr = b._workflow == 'drr'
        cube = (s._raw if product == 'raw' else s._second) if drr else b._cube
        if cube is None: return {}
        count = s._choices['regionCount'] + 1 if drr else s.geometry['count']
        sides = [''] if count == 1 else ['left', 'right'] if count == 2 else ['left', 'middle', 'right']
        bounds = ([s._numbers['boundary.x0'], s._numbers['boundary.x1']] if drr else [s.geometry['x0'], s.geometry['x1']])[:count-1]
        energy = np.asarray(cube.energy)
        token = self.context()
        if token != self._cut_token:
            from core.plotting import resolve_split_boundary
            self._cuts = [0, *(resolve_split_boundary(energy, boundary)[0] for boundary in bounds), len(energy)]
            self._cut_token = token
        cuts = self._cuts
        groups = []
        for i, side in enumerate(sides):
            prefix = product + '.' + (side + '_' if side else '')
            lowkey, highkey = prefix + 'vmin', prefix + 'vmax'
            if drr:
                low, high = s._numbers[lowkey], s._numbers[highkey]
                fixedlo, fixedhi = s._flags.get('fix.' + lowkey, False), s._flags.get('fix.' + highkey, False)
                positive = s._flags['log']
                status = s._recommend_status.get(product + ('.' + side if side else ''), '')
            else:
                bank, lo = s._location(lowkey); _, hi = s._location(highkey)
                low, high = bank[lo], bank[hi]
                fixedlo, fixedhi = bank['fixed'][lo], bank['fixed'][hi]
                positive = product == 'log'; status = ''
            section = slice(cuts[i], cuts[i+1])
            groups.append(dict(id=side or 'all', title=(side.title() + ' region' if side else 'Color limits'),
                low=low, high=high, values=lambda section=section: np.asarray(cube.Z)[:, section], positive=positive,
                symmetric=drr and s._flags['zero'] and not positive,
                fixedLow=fixedlo, fixedHigh=fixedhi, enabled=not getattr(b, '_busy', False), canRecommend=drr, status=status))
        pair = b._drr_processing['drrView'] == 'pair' if drr else s.values['sideBySide']
        products = ([dict(id='raw', label='ΔR/R'), dict(id='second', label='Derivative')] if drr else
                    [dict(id='linear', label='Linear'), dict(id='log', label='Log')]) if pair else []
        return dict(groups=groups, products=products, productId=product,
            strengthOptions=['Gentle', 'Standard', 'Strong', 'Maximum'] if drr else [],
            strengthIndex=s._recommend_strength if drr else 0)

    def _keys(self, ident):
        prefix = self._product() + '.' + (ident + '_' if ident != 'all' else '')
        return prefix + 'vmin', prefix + 'vmax'

    def apply_pair(self, ident, low, high):
        s = self._state(); lo, hi = self._keys(ident)
        if not s or getattr(self.bridge, '_busy', False) or getattr(self.bridge, '_closed', False): return False
        if max(abs(low), abs(high)) > 1e12: raise ValueError('Color limits must be between -1e12 and 1e12.')
        if self.bridge._workflow == 'drr':
            if low >= high or s._flags['log'] and low <= 0: raise ValueError('Color limits must increase; logarithmic limits must be positive.')
            changed = s._numbers[lo] != low or s._numbers[hi] != high
            s._numbers.update({lo: low, hi: high})
            if changed:
                s._memory.edited('numbers', lo); s._memory.edited('numbers', hi)
                if lo == 'second.vmin': s._second_auto_scale = False
                s._changed(lo, low)
        else:
            if low >= high or lo.startswith('log.') and low <= 0: raise ValueError('Color limits must increase; logarithmic limits must be positive.')
            bank, lf = s._location(lo); _, hf = s._location(hi)
            changed = bank[lf] != low or bank[hf] != high
            bank[lf], bank[hf] = low, high
            if changed:
                if bank['fixed'][lf]: s._manual[lo] = low
                if bank['fixed'][hf]: s._manual[hi] = high
                s._changed(lo, low)
        return True

    def set_fixed(self, ident, endpoint, fixed):
        if getattr(self.bridge, '_busy', False) or getattr(self.bridge, '_closed', False): return False
        keys = self._keys(ident); return self._state().setFlag('fix.' + keys[0 if endpoint == 'low' else 1], fixed)
    def auto(self, ident):
        if getattr(self.bridge, '_busy', False) or getattr(self.bridge, '_closed', False): return False
        return self._state().invoke('auto.' + self._product() + ('.' + ident if ident != 'all' else ''))
    def recommend(self, ident, mode):
        if getattr(self.bridge, '_busy', False) or getattr(self.bridge, '_closed', False): return False
        if self.bridge._workflow == 'drr' and mode in ('overall','weak'):
            return self._state().invoke('recommend.' + mode + '.' + self._product() + ('.' + ident if ident != 'all' else ''))
        return False
    def set_strength(self, index):
        if getattr(self.bridge, '_busy', False) or getattr(self.bridge, '_closed', False): return False
        if self.bridge._workflow == 'drr': return self._state().setChoice('recommendStrength', index)
        return False
    def set_product(self, product):
        if getattr(self.bridge, '_busy', False) or getattr(self.bridge, '_closed', False): return False
        s = self._state()
        if self.bridge._workflow == 'drr' and product in ('raw','second'): return s.setChoice('colorProduct', ('raw','second').index(product))
        if self.bridge._workflow == 'pl' and product in ('linear','log'): return s.setChoice('colorProduct', ('linear','log').index(product))
        return False
