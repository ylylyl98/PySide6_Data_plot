"""Isolate temporary PowerPoint display edits from measurement preferences."""
from copy import deepcopy


class _SettingsOverlay:
    """Read through to QSettings, retaining writes/removals only in memory."""
    _REMOVED = object()

    def __init__(self, original):
        self.original = original
        self.values = {}

    def value(self, key, defaultValue=None, type=None):
        if key not in self.values:
            parent = key
            while '/' in parent:
                parent = parent.rsplit('/', 1)[0]
                if self.values.get(parent) is self._REMOVED:
                    return defaultValue
            return self.original.value(key, defaultValue, type=type) if type is not None else self.original.value(key, defaultValue)
        value = self.values[key]
        if value is self._REMOVED:
            value = defaultValue
        if type is not None and value is not None:
            if type is bool and isinstance(value, str):
                return value.lower() in ('1', 'true', 'yes', 'on')
            return type(value)
        return value

    def setValue(self, key, value):
        self.values[key] = deepcopy(value)

    def remove(self, key):
        self.values[key] = self._REMOVED
        # QSettings.remove also removes children of a group.
        for name in tuple(self.values):
            if name.startswith(key.rstrip('/') + '/'):
                self.values[name] = self._REMOVED

    def sync(self):
        pass


class DisplaySession:
    """Snapshot a display object once; discard all PPT-only edits on restore.

    Construct before replacing its measurement cube. Never swap the QObject:
    QML retains the same display instance. `restore` is safe to call twice.
    """
    _DRR_FIELDS = ('_numbers', '_flags', '_choices', '_fix_sources', '_coordinates',
                   '_second_auto_scale', '_has_geometry', '_figure_mode',
                   '_memory_status', '_memory_available', '_errors', '_error')
    _DRR_MEMORY_FIELDS = ('context', 'borrowed', 'protected', 'viewport', 'pending',
                          'pending_edits', '_borrowed_second_auto', 'preserve_manual_geometry_once')
    _PL_FIELDS = ('values', 'geometry', 'axis_fixed', 'banks', 'axis_context',
                  'coordinates', 'axis_signature', '_revision', '_manual',
                  '_manual_context', '_manual_signature', '_preferred_cmap',
                  '_errors', '_error', '_status')

    def __init__(self, display):
        self.display = display
        self.original_settings = display.settings
        self.kind = 'drr' if hasattr(display, '_memory') and hasattr(display, '_raw') else 'pl'
        self._restored = False
        fields = self._DRR_FIELDS if self.kind == 'drr' else self._PL_FIELDS
        self.state = {name: deepcopy(getattr(display, name)) for name in fields}
        self.cubes = ((display._raw, display._second) if self.kind == 'drr' else (display._cube,))
        if self.kind == 'drr':
            memory = display._memory
            self.memory = {name: deepcopy(getattr(memory, name)) for name in self._DRR_MEMORY_FIELDS}
        else:
            self.memory = None
        display.settings = _SettingsOverlay(self.original_settings)

    def restore(self, *, preserve_next_drr=True):
        if self._restored:
            return
        display = self.display
        self._restored = True
        display._debounce.stop()
        display._editing.clear()
        display._pending.clear()
        display.settings = self.original_settings
        for name, value in self.state.items():
            setattr(display, name, deepcopy(value))
        if self.kind == 'drr':
            display._raw, display._second = self.cubes
            for name, value in self.memory.items():
                setattr(display._memory, name, deepcopy(value))
            if preserve_next_drr:
                display._memory.preserve_manual_geometry_once = True
        else:
            display._cube = self.cubes[0]
            display._availability_cache = None
        display.refresh()
