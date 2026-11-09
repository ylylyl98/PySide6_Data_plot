"""Qt Quick Megasweep session and asynchronous analysis bridge."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Event
from time import perf_counter
from PySide6.QtCore import QObject, Property, QStandardPaths, QTimer, QUrl, Signal, Slot
from ui_qt.quick.megasweep_state import new_state
from ui_qt.quick.megasweep_plot import MegasweepPlotMixin

_BACKGROUND_ROLES = ('custom', 'p', 'n', 'bn')
_BACKGROUND_OPTIONS = (('custom', 'Custom'), ('p', 'P'), ('n', 'N'), ('bn', 'BN'))

_DISPLAY_KEYS = ('cmap','color_auto','vmin','vmax','x_auto','xmin','xmax','y_auto','ymin','ymax','export_dpi')

class MegasweepBridge(MegasweepPlotMixin, QObject):
    snapshotChanged = Signal()
    plotBackendChanged = Signal()
    statusMessage = Signal(str)
    logMessage = Signal(str)
    sourceDialogRequested = Signal(str, bool)
    operationCompleted = Signal(str, float)
    _completed = Signal(int, str, object, str, float)

    def __init__(self, *, settings, parent=None, session_directory=None):
        super().__init__(parent)
        self.settings = settings
        from ui_qt.quick.settings_storage import settings_storage_dir
        self._session_directory = Path(session_directory) if session_directory else settings_storage_dir(settings) / 'megasweep-sessions'
        self._legacy_directory = None if session_directory else Path(QStandardPaths.writableLocation(QStandardPaths.GenericDataLocation)) / 'MegasweepAnalysis' / 'sessions'
        self._dataset_cache = {}
        self._states = {mode: new_state(mode) for mode in ('PL', 'Reflection', 'Transport')}
        self._restore_locations()
        self._mode = 'PL'
        self._folder = self._states['PL']['picker_folder']
        self._status = 'Select a Megasweep CSV file.'
        self._error = ''
        self._busy = self._closed = False
        self._operation = ''
        self._result = None
        self._result_sequence = -1
        self._sequence = 0
        self._cancel = None
        self._result_paths = []
        self._reflection_preview = None
        self._reflection_visible = False
        self._movie_preview = ''
        self._movie_error = ''
        self._movie_export = False
        self._preview_session = None
        self._preview_frame = 0
        self._preview_count = 0
        self._preview_duration = 0.0
        self._pending_preview_frame = None
        self._plot_tool = ''
        self._palette = {'inset':'#ffffff', 'text':'#172b4d'}
        self._canvas = self._navigation = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='dptk-megasweep')
        self._pending = set()
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.timeout.connect(self._load)
        self._completed.connect(self._finish)
        from ui_qt.quick.background_source_catalog import MegasweepBackgroundSourceCatalogBridge
        self._background = MegasweepBackgroundSourceCatalogBridge(settings=settings, parent=self, inline=True)
        self._background.selectionEdited.connect(self._background_edited)
        self._background.dialogRequested.connect(self.sourceDialogRequested)
        self._background_mode = None
        self._background_error = ''

    @staticmethod
    def _existing_path(value, *, directory):
        if not value: return ''
        try:
            path = Path(str(value)).expanduser().resolve()
            return str(path) if (path.is_dir() if directory else path.is_file()) else ''
        except (OSError, ValueError, TypeError): return ''

    def _restore_locations(self):
        for mode, state in self._states.items():
            prefix = 'quick/megasweep/locations/' + mode + '/'
            for key in ('picker_folder', 'experiment_root', 'background_folder'):
                state[key] = self._existing_path(self.settings.value(prefix + key, ''), directory=True)
            for role in _BACKGROUND_ROLES:
                state['background_folders'][role] = self._existing_path(self.settings.value(prefix + 'background_folders/' + role, ''), directory=True)
            if not state['background_folders']['custom']:
                state['background_folders']['custom'] = state['background_folder']
            state['recent_source'] = self._existing_path(self.settings.value(prefix + 'recent_source', ''), directory=False)
            if not state['picker_folder']:
                state['picker_folder'] = (state['experiment_root'] or
                    (str(Path(state['recent_source']).parent) if state['recent_source'] else ''))

    def _save_locations(self):
        prefix = 'quick/megasweep/locations/' + self._mode + '/'
        for key in ('picker_folder', 'experiment_root', 'recent_source', 'background_folder'):
            self.settings.setValue(prefix + key, self._state[key])
        for role in _BACKGROUND_ROLES:
            self.settings.setValue(prefix + 'background_folders/' + role, self._state['background_folders'][role])

    def _picker_folder(self):
        state = self._state
        return (self._existing_path(state['picker_folder'], directory=True)
                or self._existing_path(state['experiment_root'], directory=True)
                or (str(Path(recent).parent) if (recent := self._existing_path(state['recent_source'], directory=False)) else ''))

    def _background_folder(self):
        state = self._state
        return (self._existing_path(state['background_folders'][state['values']['background_role']], directory=True)
                or self._existing_path(state['experiment_root'], directory=True)
                or self._picker_folder())

    @property
    def _state(self): return self._states[self._mode]
    def _current(self): return self._result is not None and self._result_sequence == self._sequence
    @Property(QObject, notify=plotBackendChanged)
    def plotBackend(self): return self._canvas.backend if self._canvas else None
    @Property(QObject, constant=True)
    def backgroundSelectionBridge(self): return self._background
    @Property('QVariantMap', notify=snapshotChanged)
    def snapshot(self):
        state = self._state; result = self._result if self._current() else None
        summary = ''
        if result:
            summary = str(result.get('title', '')) + ' · ' + Path(state['source']).name
            dataset = next((data for key, data in self._dataset_cache.items() if key[0] == state['source'] and key[1] == self._mode), None)
            if dataset and dataset.get('x_name') and dataset.get('y_name'):
                summary += ' · X: ' + str(dataset['x_name']) + ' · Y: ' + str(dataset['y_name'])
                validation = dataset.get('axis_validation', {})
                if validation.get('message'): summary += ' · ' + str(validation['message'])
            diagnostics = result.get('diagnostics', {})
            for key, label in (('grid_kind','Grid'), ('missing_count','Missing cells'), ('background_scale_factor','Background scale'), ('background_scale_frame','Reference frame')):
                if key in diagnostics and isinstance(diagnostics[key], (str, int, float)):
                    summary += ' · ' + label + ': ' + str(diagnostics[key])
        return dict(workflow='megasweep', title='Megasweep', mode=self._mode, values=deepcopy(state['values']),
            parameterGroups=self._groups(), status=self._status, error=self._error, busy=self._busy,
            progress=-1 if self._busy else 0, canExport=bool(result) and not self._busy and not self._error and not self._reflection_visible, reflectionPreviewVisible=self._reflection_visible,
            sourcePath=self._picker_folder(), sourceFile=state['source'], selectedSources=[state['source']] if state['source'] else [],
            pickerFolder=self._picker_folder(),
            experimentRoot=self._existing_path(state['experiment_root'], directory=True),
            pickerSelected=[selected] if (selected := self._existing_path(state['source'] or state['recent_source'], directory=False)) else [],
            backgroundFolder=self._background_folder(),
            backgroundFiles=list(state['backgrounds']), backgroundSelectionError=self._background_error, backgroundRole=state['values']['background_role'],
            backgroundGroups=deepcopy(state['background_groups']),
            backgroundRoleOptions=[dict(value=role, label=label) for role, label in _BACKGROUND_OPTIONS], summary=summary, plotTool=self._plot_tool,
            movieOptions=self._movie_options(), moviePreviewUrl=self._movie_preview, moviePreviewStatus=self._status, movieError=self._movie_error, moviePreviewFrameIndex=self._preview_frame, moviePreviewFrameCount=self._preview_count, movieDuration=self._preview_duration, moviePreviewBusy=self._busy and self._operation == 'preview',
            resultPaths=list(self._result_paths), inspectionVisible=bool(result), workspaceMessage=self._error or self._status)

    def _movie_options(self):
        values = self._state['values']; ranges = values.get('movie_ranges', {}); colors = values.get('movie_colors', {})
        result = dict(hold=.25, speed=1., fps=30, head=0., tail=0., direction='forward', repeats=1, width=1400, height=960, crf=18, color_mode='global', first=None, last=None, stride=1, vmin=None, vmax=None, x_min=None, x_max=None, y_min=None, y_max=None)
        result.update(values.get('movie_options', {}))
        for key in ('first', 'last', 'stride'): result[key] = values.get('movie_' + key, result[key])
        for axis in ('x', 'y'):
            for bound in ('min', 'max'): result[axis + '_' + bound] = ranges.get(axis, {}).get(bound)
        result['vmin'] = colors.get('min'); result['vmax'] = colors.get('max')
        return result

    def _message(self, text, *, error=False):
        self._status = str(text); self._error = self._status if error else ''
        self.statusMessage.emit(self._status); self.logMessage.emit(self._status); self.snapshotChanged.emit()

    def _save(self):
        state = self._state
        if not state['source']: return
        from core.megasweep_store import save_session
        try: save_session(self._session_directory, state['source'], self._mode, state['values'], state['backgrounds'], state.get('source_root', ''), state.get('output_directory', ''), background_groups=state['background_groups'])
        except (OSError, ValueError, TypeError) as exc: self.logMessage.emit('Could not save Megasweep session: ' + str(exc))

    def _restore(self):
        state = self._state
        from core.megasweep_store import load_session
        try:
            data = load_session(self._session_directory, state['source'], self._mode)
            if not data and self._legacy_directory:
                data = load_session(self._legacy_directory, state['source'], self._mode)
            for notice in data.get('migration_notices', []): self.logMessage.emit('Megasweep session migration: ' + str(notice))
            for key, value in data.get('values', {}).items():
                if key in state['values'] or key in ('movie_first', 'movie_last', 'movie_stride'): state['values'][key] = value
            state['background_groups'] = {role: list(data.get('background_groups', {}).get(role, [])) for role in _BACKGROUND_ROLES}
            state['backgrounds'] = list(state['background_groups'][state['values']['background_role']])
            if data.get('source_root'):
                saved_root = Path(data['source_root']).expanduser().resolve()
                if Path(state['source']).resolve().is_relative_to(saved_root): state['source_root'] = str(saved_root)
            state['output_directory'] = str(data.get('output_directory', ''))
        except (OSError, ValueError, TypeError) as exc: self.logMessage.emit('Could not restore Megasweep session: ' + str(exc))

    def _invalidate(self):
        self._timer.stop(); self._sequence += 1
        self._reflection_visible = False; self._reflection_preview = None
        preview = self._preview_session; self._preview_session = None
        self._pending_preview_frame = None
        self._preview_frame = self._preview_count = 0; self._preview_duration = 0.0
        if self._movie_preview:
            try: Path(QUrl(self._movie_preview).toLocalFile()).unlink(missing_ok=True)
            except OSError: pass
            self._movie_preview = ''
        if preview is not None: self._executor.submit(preview.close)
        if self._cancel: self._cancel.set()
        self._busy = False; self._operation = ''; self._result_paths = []

    def _cached_dataset(self, path, mode):
        return next((data for key, data in self._dataset_cache.items() if key[0] == path and key[1] == mode), None)

    def _effective_values(self):
        values = deepcopy(self._state['values'])
        values['background_groups'] = deepcopy(self._state['background_groups'])
        values.update(color_min=None if values['color_auto'] else values['vmin'], color_max=None if values['color_auto'] else values['vmax'], x_limits=None if values['x_auto'] else (values['xmin'], values['xmax']), y_limits=None if values['y_auto'] else (values['ymin'], values['ymax']))
        return values

    def _queue(self, immediate=False, force=False):
        self._invalidate(); self._save()
        if self._state['source']:
            self._status = 'Waiting to update Megasweep…'; self._error = ''
            if self._state['values'].get('auto_update', True) or force: self._timer.start(0 if immediate else 180)
            else: self._status = 'Parameters changed. Select Update to process.'
        self.snapshotChanged.emit()

    def _rebind_background(self):
        if self._background_mode == self._mode and self._background.snapshot['selectionOpen'] and self._background_folder():
            self._background.open_inline_selection(self._background_folder(), self._state['values']['background_role'], self._state['backgrounds'])

    @Slot()
    def invalidateBackgroundSelection(self):
        self._background_error = ''
        self._background_mode = None
        self._background._stop_selection()
        self._background.snapshotChanged.emit()

    @Slot(str)
    def setExperimentFolder(self, value):
        if not value or self._closed or (self._busy and self._operation == 'export'): return
        url = QUrl(str(value)); folder = Path(url.toLocalFile() if url.isLocalFile() else str(value)).expanduser().resolve()
        if not folder.is_dir(): self._message('Experiment folder does not exist.', error=True); return
        changed = str(folder) != self._folder or str(folder) != self._state['experiment_root']
        self._folder = self._state['picker_folder'] = self._state['experiment_root'] = str(folder)
        self._save_locations()
        if changed: self._rebind_background()
        self.snapshotChanged.emit()

    @Slot('QVariantList')
    def selectFiles(self, values):
        if self._closed or not values or (self._busy and self._operation == 'export'): return
        raw = values[0]; url = raw if isinstance(raw, QUrl) else QUrl(str(raw)); raw = str(raw)
        path = Path(url.toLocalFile() if url.isLocalFile() else raw).expanduser()
        if not path.is_absolute() and self._folder: path = Path(self._folder) / path
        path = path.resolve()
        if not path.is_file(): self._message('Source file does not exist: ' + str(path), error=True); return
        experiment = self._existing_path(self._state['experiment_root'], directory=True)
        external = not experiment or not path.is_relative_to(Path(experiment))
        if external:
            self._state['experiment_root'] = str(path.parent)
        if str(path) != self._state['source']:
            self._background_error = ''
            self._save(); self._state['source'] = str(path)
            root = Path(self._state['experiment_root']).resolve()
            self._state['source_root'] = str(root if path.is_relative_to(root) else path.parent)
            self._state['values'] = new_state(self._mode)['values']; self._state['backgrounds'] = []
            self._state['background_groups'] = {role: [] for role in _BACKGROUND_ROLES}
            self._restore()
            self._rebind_background()
        if external: self._state['source_root'] = str(path.parent)
        self._folder = self._state['picker_folder'] = str(path.parent)
        self._state['recent_source'] = str(path)
        self._save_locations()
        self._queue(immediate=True)

    @Slot(str, 'QVariantList')
    def openFiles(self, purpose, values):
        if self._busy and self._operation == 'export': return
        if purpose.startswith('inline-catalog:'):
            try: token = int(purpose.split(':', 1)[1])
            except ValueError: return
            if self._background_mode == self._mode: self._background.accept_external(token, values)
        elif purpose in ('sources', 'select-files', 'source'): self.selectFiles(values)
        elif purpose == 'background': self._background_accepted('drr', values)

    @Slot(str, 'QVariant')
    def setParameter(self, key, value):
        if self._closed: return
        if self._busy and self._operation == 'export':
            self.statusMessage.emit('Wait for export to finish or cancel it first.'); return
        if key == 'mode':
            mode = 'Reflection' if str(value) == 'DRR' else str(value)
            if mode not in self._states or mode == self._mode: return
            previous_root = self._existing_path(self._state['experiment_root'], directory=True)
            self._save(); self._invalidate(); self.invalidateBackgroundSelection(); self._mode = mode
            if not self._state['experiment_root']:
                self._state['experiment_root'] = previous_root
            if not self._state['picker_folder']:
                self._state['picker_folder'] = self._existing_path(self._folder, directory=True)
            self._folder = self._picker_folder()
            self._save_locations()
            if self._canvas:
                self._canvas.figure.clear(); self._canvas.draw_idle()
            self._queue(immediate=True); return
        values = self._state['values']
        if key == 'background_role':
            role = str(value).lower()
            if role not in _BACKGROUND_ROLES or role == values['background_role']: return
            self._background_error = ''
            values['background_role'] = role
            self._state['backgrounds'] = list(self._state['background_groups'][role])
            self._rebind_background()
            self._queue()
            return
        if key == 'movie.previewFrame':
            try: self._pending_preview_frame = max(0, int(value))
            except (ValueError, TypeError): return
            if not self._busy: self.invoke('movie-preview')
            return
        if key.startswith('scale_window_'):
            try:
                _, _, index, bound = key.split('_'); index, bound = int(index), int(bound)
                windows = [list(pair) for pair in values['background_scale_windows']]
                windows[index][bound] = float(value)
                values['background_scale_windows'] = windows
                self._queue()
            except (ValueError, IndexError, TypeError): self._message('Invalid background scaling window.', error=True)
            return
        if key.startswith('movie.'):
            self._movie_error = ''
            field = key[6:]
            defaults = dict(hold=.25, speed=1., fps=30, head=0., tail=0., direction='forward', repeats=1, width=1400, height=960, crf=18, color_mode='global')
            try:
                if field in defaults:
                    if isinstance(defaults[field], int) and (value is None or float(value) != int(float(value))): raise ValueError('Integer required')
                    values['movie_options'][field] = type(defaults[field])(value)
                elif field in ('first', 'last', 'stride'):
                    if field == 'stride' and value is not None and float(value) != int(float(value)): raise ValueError('Integer required')
                    values['movie_' + field] = (int(value) if field == 'stride' else float(value)) if value is not None else (1 if field == 'stride' else None)
                elif field in ('vmin', 'vmax'):
                    bound = 'min' if field == 'vmin' else 'max'
                    if value is None: values['movie_colors'].pop(bound, None)
                    else: values['movie_colors'][bound] = float(value)
                elif field in ('x_min', 'x_max', 'y_min', 'y_max'):
                    axis, bound = field.split('_'); spec = values['movie_ranges'].setdefault(axis, {})
                    if value is None: spec.pop(bound, None)
                    else: spec[bound] = float(value)
                    spec['auto'] = not ('min' in spec and 'max' in spec)
                else: return
            except (TypeError, ValueError) as exc:
                self._movie_error = 'Invalid movie option: ' + str(exc); self.snapshotChanged.emit(); return
            current = self._current()
            self._invalidate()
            if current: self._result_sequence = self._sequence
            self._save(); self.snapshotChanged.emit(); return
        if key not in values: return
        old = values[key]
        try:
            if key in ('background_scale_frame','pass_index'): value = None if value in ('', None) else int(value)
            elif key == 'direction': value = None if value in ('', None) else str(value)
            elif isinstance(old, bool): value = bool(value)
            elif isinstance(old, int): value = int(value)
            elif isinstance(old, float): value = float(value)
            elif isinstance(old, str): value = str(value)
            if isinstance(value, float):
                from math import isfinite
                if not isfinite(value): raise ValueError('Value must be finite')
            if key in ('sg_window', 'derivative_window') and (value < 3 or value % 2 != 1): raise ValueError('Window must be odd and at least 3')
            if key in ('sg_poly', 'background_scale_frame') and value is not None and value < 0: raise ValueError('Value cannot be negative')
            if key == 'min_points' and value < 1: raise ValueError('Minimum points must be positive')
            if key == 'epsilon' and value < 0: raise ValueError('Tolerance cannot be negative')
            if value == old: return
            if key in ('view', 'transformed'):
                display_key = values['view'] + ('/transformed' if values['transformed'] else '/original')
                values['_display_states'][display_key] = {name: values[name] for name in _DISPLAY_KEYS}
            values[key] = value
            if self._mode == 'Transport' and key == 'cut_type':
                data = next((item for cache_key, item in self._dataset_cache.items() if cache_key[0] == self._state['source'] and cache_key[1] == self._mode), None)
                if data is not None and value in data['columns']:
                    measured = data['df'][value].dropna()
                    if len(measured): values['cut_value'] = float(measured.iloc[0])
            if key in ('view', 'transformed'):
                display_key = values['view'] + ('/transformed' if values['transformed'] else '/original')
                saved = values['_display_states'].get(display_key, {})
                defaults = new_state(self._mode)['values']
                for name in _DISPLAY_KEYS: values[name] = saved.get(name, defaults[name])
            if key in _DISPLAY_KEYS:
                display_key = values['view'] + ('/transformed' if values['transformed'] else '/original')
                values['_display_states'][display_key] = {name: values[name] for name in _DISPLAY_KEYS}
                self._save()
                if self._reflection_visible: self._render_reflection_preview()
                else: self._render()
                self.snapshotChanged.emit()
            elif key == 'auto_update':
                self._save()
                if value and not self._current(): self._queue()
                else: self.snapshotChanged.emit()
            else: self._queue()
        except (ValueError, TypeError) as exc: self._message('Invalid parameter: ' + str(exc), error=True)

    @Slot('QVariantList')
    def _background_edited(self, paths):
        if self._background_mode == self._mode:
            self._background_accepted('drr', paths)

    @Slot(str, 'QVariantList')
    def _background_accepted(self, workflow, paths):
        if self._closed or (self._busy and self._operation == 'export'): return
        normalized = [str(Path(p.toLocalFile() if isinstance(p, QUrl) and p.isLocalFile() else str(p)).expanduser().resolve()) for p in paths]
        role = self._state['values']['background_role']
        try:
            from core.megasweep.background_roles import validate_background_role
            validate_background_role(role, normalized)
        except ValueError as exc:
            self._background.rebind_inline_selection(self._state['backgrounds'])
            self._background_error = str(exc)
            self.statusMessage.emit(self._background_error)
            self.logMessage.emit(self._background_error)
            self.snapshotChanged.emit()
            return
        self._background_error = ''
        self._state['backgrounds'] = normalized
        self._state['background_groups'][role] = list(normalized)
        if normalized:
            path = Path(normalized[-1])
            if path.parent.is_dir():
                self._state['background_folders'][role] = str(path.parent)
                if role == 'custom': self._state['background_folder'] = str(path.parent)
        self._save_locations(); self._queue(immediate=True)

    def _start(self, kind, fn):
        current = self._current()
        self._sequence += 1; sequence = self._sequence
        if current and kind in ('export', 'preview', 'slice-step', 'reflection-preview'): self._result_sequence = sequence
        self._cancel = Event(); cancel = self._cancel
        self._busy = True; self._operation = kind
        if kind == 'export': self._background.set_edit_enabled(False)
        start = perf_counter(); future = self._executor.submit(fn, cancel)
        self._pending.add(future)
        def done(task):
            self._pending.discard(task)
            try: payload, error = task.result(), ''
            except Exception as exc:
                payload, error = None, str(exc) or type(exc).__name__
                completed = getattr(exc, 'completed_paths', ())
                if completed: error += ' Completed files remain: ' + ', '.join(map(str, completed))
            if kind == 'preview' and payload and (self._closed or sequence != self._sequence):
                session, _ = payload
                session.close()
            if not self._closed: self._completed.emit(sequence, kind, payload, error, (perf_counter()-start)*1000)
        future.add_done_callback(done); self.snapshotChanged.emit()

    def _load(self):
        state = self._state; path = state['source']; mode = self._mode
        if not path or self._closed: return
        values = self._effective_values(); backgrounds = tuple(state['backgrounds'])
        self._status = 'Processing Megasweep…'
        def work(cancel):
            from core.megasweep.workflow import load_dataset, analyze
            cache_key = (path, mode, values.get('x_column'), values.get('y_column'))
            data = self._dataset_cache.get(cache_key)
            if data is None: data = load_dataset(path, mode, values.get('x_column') or None, values.get('y_column') or None)
            if cancel.is_set(): raise InterruptedError('Cancelled')
            if mode == 'Transport':
                suggestion = data.get('axis_suggestion', {})
                values['x_column'] = values['x_column'] or suggestion.get('x') or ''
                values['y_column'] = values['y_column'] or suggestion.get('y') or ''
                if suggestion.get('kind') == 'curve' and not values['y_column']: values['view'] = 'transport_curve'
                candidates = [col for col in data.get('columns', []) if col not in (values['x_column'], values['y_column'])]
                values['channel'] = values['channel'] or (candidates[0] if candidates else '')
                if values['cut_type'] not in data.get('columns', []):
                    values['cut_type'] = values['x_column'] or values['y_column']
                    if values['cut_type']:
                        measured = data['df'][values['cut_type']].dropna()
                        if len(measured): values['cut_value'] = float(measured.iloc[0])
            try:
                result = analyze(data, values, backgrounds, cancelled=cancel.is_set); error = ''
            except Exception as exc:
                result = None; error = str(exc) or type(exc).__name__
            return (cache_key, data, values, result, error)
        self._start('load', work)

    @Slot(int, str, object, str, float)
    def _finish(self, sequence, kind, payload, error, elapsed):
        if self._closed or sequence != self._sequence: return
        self._busy = False; self._operation = ''
        if kind == 'export': self._background.set_edit_enabled(True)
        if error:
            if kind == 'preview' or (kind == 'export' and self._movie_export):
                self._movie_error = error; self._status = error
                self.statusMessage.emit(error); self.logMessage.emit(error); self.snapshotChanged.emit()
            else: self._message(error, error=True)
            return
        if kind == 'preview' or (kind == 'export' and self._movie_export): self._movie_error = ''
        if kind == 'slice-step':
            if payload is not None: self.setParameter('cut_value', payload)
            else: self._message('No measured slice coordinates are available.')
        elif kind == 'reflection-preview':
            self._reflection_preview = payload; self._reflection_visible = True
            self._message('Reference spectra preview ready.')
            self._render_reflection_preview()
        elif kind == 'preview':
            self._preview_session, payload = payload
            import tempfile, os
            fd, preview_path = tempfile.mkstemp(prefix='dptk-megasweep-', suffix='.png')
            with os.fdopen(fd, 'wb') as stream: stream.write(payload['png'])
            if self._movie_preview:
                try: Path(QUrl(self._movie_preview).toLocalFile()).unlink(missing_ok=True)
                except OSError: pass
            self._movie_preview = QUrl.fromLocalFile(preview_path).toString()
            self._preview_frame = payload.get('frame_index', 0); self._preview_count = payload.get('frame_count', len(payload.get('values', []))); self._preview_duration = payload.get('duration', 0.)
            self._message('Movie preview ready.')
            if self._pending_preview_frame is not None and self._pending_preview_frame != self._preview_frame: self.invoke('movie-preview')
        elif kind == 'load':
            cache_key, data, used_values, result, analysis_error = payload
            if self._mode == 'Transport':
                for key in ('x_column', 'y_column', 'channel', 'view', 'cut_type', 'cut_value'): self._state['values'][key] = used_values[key]
            self._dataset_cache = {key: value for key, value in self._dataset_cache.items() if key[1] != cache_key[1]}
            self._dataset_cache[cache_key] = data
            if analysis_error:
                self._message(analysis_error, error=True); return
            self._result = result; self._result_sequence = sequence
            self._message('Megasweep analysis complete.'); self._render()
        else:
            self._result_sequence = sequence
            self._result_paths = [str(p) for p in (payload.values() if isinstance(payload, dict) else payload if isinstance(payload, (list, tuple)) else [payload])]
            self._message('Megasweep export complete.')
        self.operationCompleted.emit(kind, elapsed)

    @Slot(str)
    def invoke(self, action):
        if self._closed: return
        if self._busy and self._operation == 'export' and action != 'cancel':
            self.statusMessage.emit('Wait for export to finish or cancel it first.'); return
        if action == 'return-to-map':
            self._reflection_visible = False; self._reflection_preview = None
            self._render(); self.snapshotChanged.emit()
        elif action == 'select-files': self.sourceDialogRequested.emit('sources', False)
        elif action == 'choose-background':
            self._background_error = ''
            if self._background_folder():
                self._background_mode = self._mode
                self._background.open_inline_selection(self._background_folder(), self._state['values']['background_role'], self._state['backgrounds'])
                self.snapshotChanged.emit()
            else: self._message('Select an experiment folder first.', error=True)
        elif action == 'clear-background':
            self._background_error = ''
            self._state['backgrounds'] = []
            self._state['background_groups'][self._state['values']['background_role']] = []
            self._rebind_background(); self._queue()
        elif action in ('update', 'reload'):
            if action == 'reload': self._dataset_cache.clear()
            self._queue(immediate=True, force=True)
        elif action in ('cancel', 'movie-cancel-preview'):
            if self._busy and self._operation == 'export':
                if self._cancel: self._cancel.set()
                self._message('Cancelling export; completed files will be retained.')
                return
            preserve = action == 'movie-cancel-preview' and self._current()
            self._invalidate()
            if preserve: self._result_sequence = self._sequence
            self._message('Megasweep operation cancelled.')
        elif action in ('slice-previous', 'slice-next'):
            state = self._state; path = state['source']; mode = self._mode; values = deepcopy(state['values']); cached = self._cached_dataset(path, mode)
            if not path: return
            def step(cancel):
                from core.megasweep.workflow import load_dataset, slice_values
                data = cached if cached is not None else load_dataset(path, mode, values.get('x_column') or None, values.get('y_column') or None)
                points = slice_values(data, values)
                if cancel.is_set() or len(points) == 0: return None
                index = min(range(len(points)), key=lambda i: abs(points[i] - float(values['cut_value'])))
                index = min(len(points)-1, max(0, index + (1 if action == 'slice-next' else -1)))
                return float(points[index])
            self._start('slice-step', step)
        elif action == 'reflection-preview':
            state = self._state; path = state['source']; mode = self._mode; values = self._effective_values(); backgrounds = tuple(state['backgrounds'])
            if mode != 'Reflection' or not path: return
            cached = self._cached_dataset(path, mode)
            def reflection_preview(cancel):
                from core.megasweep.workflow import load_dataset, reflection_preview
                data = cached if cached is not None else load_dataset(path, mode, values.get('x_column') or None, values.get('y_column') or None)
                return reflection_preview(data, values, backgrounds, cancelled=cancel.is_set)
            self._start('reflection-preview', reflection_preview)
        elif action == 'movie-preview':
            if not self._current():
                self._movie_error = 'Update the current analysis before previewing.'; self.snapshotChanged.emit(); return
            state = self._state; path = state['source']; mode = self._mode; values = self._effective_values(); backgrounds = tuple(state['backgrounds'])
            frame_index = self._pending_preview_frame or 0; self._pending_preview_frame = None
            existing = self._preview_session; cached = self._cached_dataset(path, mode)
            def preview(cancel):
                from core.megasweep.workflow import load_dataset, MoviePreviewSession
                session = existing
                if session is None:
                    data = cached if cached is not None else load_dataset(path, mode, values.get('x_column') or None, values.get('y_column') or None)
                    session = MoviePreviewSession(data, values, backgrounds, cancelled=cancel.is_set)
                try: return session, session.frame(frame_index, cancelled=cancel.is_set)
                except Exception:
                    if session is not existing: session.close()
                    raise
            self._start('preview', preview)
        elif action in ('home', 'back', 'forward', 'pan', 'zoom'): self.plotAction(action)
        elif action in ('export', 'save', 'export-batch', 'export-video'):
            if not self.snapshot['canExport']:
                self.statusMessage.emit('Return to the current processed map before exporting.'); return
            state = self._state; path = state['source']; mode = self._mode; values = self._effective_values(); backgrounds = tuple(state['backgrounds']); result = self._result
            cached = self._cached_dataset(path, mode)
            if action in ('export', 'save') and self._canvas and self._canvas.figure.axes:
                axis = self._canvas.figure.axes[0]
                xmin, xmax = map(float, axis.get_xlim())
                ymin, ymax = map(float, axis.get_ylim())
                values.update(x_auto=False, xmin=xmin, xmax=xmax,
                              y_auto=False, ymin=ymin, ymax=ymax)
            directory = (Path(state['output_directory']) / Path(path).stem if state.get('output_directory') else Path(state.get('source_root') or Path(path).parent) / 'Processed Data' / 'Megasweep' / Path(path).stem)
            def work(cancel):
                from core.megasweep.workflow import export_result, export_batch, export_video, load_dataset
                if action in ('export', 'save'): return export_result(result, directory, parameters=values)
                data = cached if cached is not None else load_dataset(path, mode, values.get('x_column') or None, values.get('y_column') or None)
                if action == 'export-batch': return export_batch(data, values, backgrounds, directory, cancelled=cancel.is_set)
                return export_video(data, values, backgrounds, directory, cancelled=cancel.is_set)
            self._movie_export = action == 'export-video'
            self._start('export', work); self._message('Exporting Megasweep…')

    @Slot('QVariantMap')
    def setPalette(self, colors):
        self._palette.update(dict(colors))
        if self._reflection_visible: self._render_reflection_preview()
        else: self._render()

    @Slot(str)
    def plotAction(self, action):
        super().plotAction(action)

    @Slot()
    def close(self):
        if self._closed: return
        self._save(); self._closed = True; self._invalidate()
        self._background.close()
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._canvas: self._canvas.close()
        if self._movie_preview:
            try: Path(QUrl(self._movie_preview).toLocalFile()).unlink(missing_ok=True)
            except OSError: pass

    def _groups(self):
        values = self._state['values']; enabled = not (self._busy and self._operation == 'export')
        view = values['view']; mode = self._mode
        choices = {'view': [('intensity','Intensity'),('peak','Peak energy'),('slice','Spectral slice'),('ibias','Bias current'),('resistance','Resistance')] if mode == 'PL' else [('rc_p2p','Reflection peak-to-peak'),('rc_energy','Reflection at energy'),('rc_peak','Reflection peak'),('slice','Spectral slice'),('ibias','Bias current'),('resistance','Resistance')] if mode == 'Reflection' else [('transport_map','Transport map'),('transport_curve','Transport curve'),('slice','Transport cut')],
            'convention':[('TG+rBG','TG + r·BG'),('rTG+BG','r·TG + BG')], 'cut_type':[('doping','Doping'),('efield','Electric field'),('x','Original X'),('y','Original Y')],
            'derivative':[('spectrum','Spectrum'),('second_derivative','Second derivative')],
            'background_average':[('all_frames','All frames equally'),('file_equal','Files equally'),('first_frame','First frame'),('last_frame','Last frame')],
            'rc_feature':[('auto','Automatic'),('peak','Peak'),('dip','Dip')], 'cmap':[(v,v) for v in ('turbo','viridis','plasma','inferno','magma','coolwarm','RdBu_r','gray')],
            'cut_coordinates':[('explicit','Explicit cut type'),('follow_map','Follow map axes'),('original','Original axes'),('transformed','Transformed axes')], 'cut_axis':[('first','First axis'),('second','Second axis')], 'batch_axes':[('selected','Selected axis'),('both','Both axes')]}
        cache = next((data for key,data in self._dataset_cache.items() if key[0] == self._state['source'] and key[1] == mode), None)
        if cache:
            columns = list(cache.get('columns', cache['df'].columns))
            for key in ('x_column','y_column','channel'):
                choices[key] = [('', 'Automatic' if key != 'channel' else 'Select signal')] + [(str(c),str(c)) for c in columns]
            if mode == 'Transport': choices['cut_type'] = [(str(c),str(c)) for c in columns]
        def field(key):
            value=values[key]; item=dict(key=key,label=key.replace('_',' ').title(),value=value,enabled=enabled)
            if key in ('background_scale_frame','pass_index'): item['type']='text'; item['value']='' if value is None else str(value)
            elif key in choices: item.update(type='choice', options=[dict(value=k,label=v) for k,v in choices[key]])
            elif isinstance(value,bool): item['type']='flag'
            elif isinstance(value,(float,int)): item.update(type='number',min=-1000000000,max=1000000000,step=1 if isinstance(value,int) else .01,decimals=0 if isinstance(value,int) else 6)
            else: item['type']='text'; item['value']='' if value is None else value
            return item
        data = ['view','auto_update','x_column','y_column']
        if mode == 'Transport': data += ['channel','direction','pass_index']
        processing=[]
        if mode != 'Transport':
            if view in ('intensity','rc_p2p','rc_peak'): processing += ['energy_min','energy_max']
            if view == 'rc_energy': processing += ['energy_target']
            if view == 'intensity': processing += ['auto_baseline','baseline']
            if view in ('peak','rc_peak'): processing += ['sg_window','sg_poly']
            if view == 'rc_peak': processing += ['rc_feature']
            if mode == 'Reflection' and view not in ('ibias','resistance'):
                processing += ['background_average','auto_background_scale','background_scale','background_scale_x','background_scale_y','background_scale_frame']
        coordinates = [] if mode == 'Transport' else ['ratio','tg_is_y','convention','transformed']
        if view == 'slice': coordinates += (['cut_coordinates','cut_axis','batch_axes'] if mode != 'Transport' else []) + ['cut_type','cut_value'] + ([] if mode == 'Transport' else ['epsilon','min_points','exact_coordinates','derivative','derivative_window'])
        groups=[dict(key='data', title='Data', fields=[field(k) for k in data])]
        if processing:
            fields=[field(k) for k in processing]
            if mode == 'Reflection' and values['auto_background_scale']:
                for index, pair in enumerate(values['background_scale_windows']):
                    for bound, value in enumerate(pair): fields.append(dict(key=f'scale_window_{index}_{bound}',label=f'Scale window {index+1} '+('minimum' if not bound else 'maximum')+' (eV)',type='number',value=value,min=0,max=100,step=.001,decimals=6,enabled=enabled))
            groups.append(dict(key='processing',title='Processing',fields=fields))
        if coordinates: groups.append(dict(key='coordinates',title='Coordinates and slices',fields=[field(k) for k in coordinates]))
        display=['cmap','color_auto'] if view != 'transport_curve' else []
        if display and not values['color_auto']: display += ['vmin','vmax']
        display += ['export_dpi', 'x_auto'] + ([] if values['x_auto'] else ['xmin','xmax']) + ['y_auto'] + ([] if values['y_auto'] else ['ymin','ymax'])
        groups.append(dict(key='display',title='Display and export',fields=[field(k) for k in display]))
        return groups
