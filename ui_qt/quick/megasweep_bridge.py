"""Qt Quick Megasweep session and asynchronous analysis bridge."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Event
from time import perf_counter
from PySide6.QtCore import QObject, Property, QStandardPaths, QTimer, QUrl, Signal, Slot
from ui_qt.quick.megasweep_state import new_state, restore_export_resolution
from ui_qt.quick.megasweep_plot import MegasweepPlotMixin

_BACKGROUND_ROLES = ('custom', 'p', 'n', 'bn')
_BACKGROUND_OPTIONS = (('custom', 'Custom'), ('p', 'P'), ('n', 'N'), ('bn', 'BN'))
_MAP_COLORMAPS = ('RdBu_r', 'RdBu')

_DISPLAY_KEYS = ('cmap','color_auto','vmin','vmax','x_auto','x_reverse','xmin','xmax','y_auto','y_reverse','ymin','ymax','export_dpi')

def _resolve_reflection_energies(data, values):
    if not (values['auto_energy_window'] or values['auto_energy_target']):
        return
    import numpy as np
    energy = data['energy']
    measured = energy[np.isfinite(energy)]
    if not measured.size:
        raise ValueError('No finite energies in the reflection scan.')
    lo, hi = float(measured.min()), float(measured.max())
    if values['auto_energy_window']:
        values.update(energy_min=lo, energy_max=hi)
    if values['auto_energy_target']:
        values['energy_target'] = float(measured[np.argmin(np.abs(measured - (lo + hi) / 2))])

class MegasweepBridge(MegasweepPlotMixin, QObject):
    snapshotChanged = Signal()
    plotBackendChanged = Signal()
    plotLimitsChanged = Signal()
    statusMessage = Signal(str)
    logMessage = Signal(str)
    sourceDialogRequested = Signal(str, bool)
    operationCompleted = Signal(str, float)
    savedFigureCloseRequested = Signal()
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
        self._saved_figure = None
        self._measurement_workspace = None
        self._reflection_preview = None
        self._reflection_visible = False
        self._reopen_reference = False
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
        if self._saved_figure is not None: return
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
            if result.get('kind') == 'slice':
                summary += f" · {len(result['data']['axis_values'])} measured points in cut"
            diagnostics = result.get('diagnostics', {})
            for key, label in (('grid_kind','Grid'), ('missing_count','Missing cells'), ('background_scale_factor','Background scale'), ('background_scale_frame','Reference frame')):
                if key in diagnostics and isinstance(diagnostics[key], (str, int, float)):
                    summary += ' · ' + label + ': ' + str(diagnostics[key])
        groups, reference_groups = self._groups()
        return dict(workflow='megasweep', title='Megasweep', mode=self._mode, values=deepcopy(state['values']),
            parameterGroups=groups, referenceParameterGroups=reference_groups, status=self._status, error=self._error, busy=self._busy,
            savedFigureActive=self._saved_figure is not None,
            progress=-1 if self._busy else 0, canExport=bool(result) and not self._busy and not self._error and not self._reflection_visible, reflectionPreviewVisible=self._reflection_visible,
            sourcePath=self._picker_folder(), sourceFile=state['source'], selectedSources=[state['source']] if state['source'] else [],
            pickerFolder=self._picker_folder(),
            experimentRoot=self._existing_path(state['experiment_root'], directory=True),
            pickerSelected=[selected] if (selected := self._existing_path(state['source'] or state['recent_source'], directory=False)) else [],
            backgroundFolder=self._background_folder(),
            referenceFrame=(self._reflection_preview or {}).get('frame', 0),
            referencePoint=self._reference_point(),
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
        if self._saved_figure is not None: return
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
            saved = data.get('values', {})
            if self._mode == 'Reflection' and 'auto_energy_window' not in saved:
                # Older sessions stored the fixed defaults even before a first analysis.
                state['values']['auto_energy_window'] = all(
                    saved.get(key, state['values'][key]) == state['values'][key]
                    for key in ('energy_min', 'energy_max')
                )
            if self._mode == 'Reflection' and 'auto_energy_target' not in saved:
                default_target = state['values']['energy_target']
                state['values']['auto_energy_target'] = saved.get('energy_target', default_target) == default_target
            if self._mode == 'Reflection' and 'auto_scale_windows' not in saved and 'background_scale_windows' in saved:
                from core.megasweep import has_legacy_scale_windows
                state['values']['auto_scale_windows'] = has_legacy_scale_windows(saved['background_scale_windows'])
            for key, value in saved.items():
                if key in state['values'] or key in ('movie_first', 'movie_last', 'movie_stride'): state['values'][key] = value
            restore_export_resolution(state['values'])
            for display in (state['values'], *state['values']['_display_states'].values()):
                if 'cmap' in display and display['cmap'] not in _MAP_COLORMAPS:
                    display['cmap'] = _MAP_COLORMAPS[0]
            state['background_groups'] = {role: list(data.get('background_groups', {}).get(role, [])) for role in _BACKGROUND_ROLES}
            state['backgrounds'] = list(state['background_groups'][state['values']['background_role']])
            if data.get('source_root'):
                saved_root = Path(data['source_root']).expanduser().resolve()
                if Path(state['source']).resolve().is_relative_to(saved_root): state['source_root'] = str(saved_root)
            state['output_directory'] = str(data.get('output_directory', ''))
        except (OSError, ValueError, TypeError) as exc: self.logMessage.emit('Could not restore Megasweep session: ' + str(exc))

    def _invalidate(self):
        self._timer.stop(); self._sequence += 1
        self._reopen_reference = False
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
        if mode != 'Transport':
            values = self._states[mode]['values']
            return self._dataset_cache.get((path, mode, values['x_column'], values['y_column']))
        return next((data for key, data in self._dataset_cache.items() if key[0] == path and key[1] == mode), None)

    def _effective_values(self):
        values = deepcopy(self._state['values'])
        values['background_groups'] = deepcopy(self._state['background_groups'])
        values.update(color_min=None if values['color_auto'] else values['vmin'], color_max=None if values['color_auto'] else values['vmax'], x_limits=None if values['x_auto'] else (values['xmin'], values['xmax']), y_limits=None if values['y_auto'] else (values['ymin'], values['ymax']))
        return values

    def _export_values(self):
        values = self._effective_values()
        if self._canvas and self._canvas.figure.axes:
            axis = self._canvas.figure.axes[0]
            for name, bounds in (('x', axis.get_xlim()), ('y', axis.get_ylim())):
                first, last = map(float, bounds)
                values.update({name + '_auto': False, name + '_reverse': first > last,
                               name + 'min': min(first, last), name + 'max': max(first, last)})
        return values

    def open_saved_figure(self, record):
        """Open the original scan with the exported recipe in an isolated workspace."""
        from core.pptx_figure_edit import experiment_root
        if self._busy and self._operation == 'export':
            raise ValueError('Wait for the Megasweep export to finish.')
        recipe = record.snapshot
        mode = recipe['mode']
        root = experiment_root(record.png_path)

        def source_file(value):
            path = Path(value)
            if not path.is_absolute():
                path = root / path
            else:
                # A copied experiment may still contain paths from its previous machine.
                for parent in path.parents:
                    if parent.name.casefold() == root.name.casefold():
                        local = root / path.relative_to(parent)
                        if local.is_file():
                            path = local
                        break
            if not path.is_file():
                raise FileNotFoundError(f'Original Megasweep CSV could not be found: {path}')
            return str(path.resolve())

        source = source_file(recipe['source_path'])
        backgrounds = [source_file(path) for path in recipe.get('background_paths', [])]
        if mode not in self._states:
            raise ValueError(f'Unknown Megasweep mode: {mode}')
        state = new_state(mode)
        values = state['values']
        params = recipe['parameters']
        values.update({k: deepcopy(params[k]) for k in values if k in params})
        restore_export_resolution(values)
        if mode == 'Reflection':
            # Early exports recorded the actual energies without their later Auto flags.
            for flag in ('auto_energy_window', 'auto_energy_target', 'auto_scale_windows'):
                if flag not in params: values[flag] = False
        role = values['background_role']
        if role not in _BACKGROUND_ROLES:
            raise ValueError(f'Unknown Megasweep background role: {role}')
        groups = params.get('background_groups') or {}
        state['background_groups'].update({key: list(groups.get(key, [])) for key in _BACKGROUND_ROLES})
        state['background_groups'][role] = backgrounds
        state.update(source=source, recent_source=source, source_root=str(root),
                     experiment_root=str(root), picker_folder=str(Path(source).parent), backgrounds=backgrounds)
        if backgrounds:
            state['background_folders'][role] = str(Path(backgrounds[0]).parent)

        if self._saved_figure is None:
            self._save()
            axis = self._canvas.figure.axes[0] if self._canvas and self._canvas.figure.axes else None
            self._measurement_workspace = dict(
                states=self._states, mode=self._mode, folder=self._folder, cache=self._dataset_cache,
                result=self._result if self._current() else None, paths=self._result_paths,
                preview=self._reflection_preview if self._reflection_visible else None,
                limits=(axis.get_xlim(), axis.get_ylim()) if axis and not self._reflection_visible else None,
                status=self._status, error=self._error, pending=self._timer.isActive() or self._busy)
        self._invalidate()
        self.invalidateBackgroundSelection()
        self._saved_figure = record
        self._states = {key: new_state(key) for key in self._states}
        self._states[mode] = state
        self._mode, self._folder = mode, state['picker_folder']
        self._dataset_cache = {}
        self._result = None
        if self._canvas:
            self._canvas.figure.clear(); self._canvas.draw_idle()
        self._queue(immediate=True, force=True)

    def saved_figure_export_request(self):
        """Capture the displayed result on the UI thread for the slide export worker."""
        from core.megasweep.workflow import export_result
        if self._saved_figure is None or not self.snapshot['canExport']:
            raise ValueError('Wait for the edited map or cut to finish processing before saving.')
        directory = self._saved_figure.png_path.parent
        result, values = self._result, self._export_values()
        return lambda: export_result(result, directory, parameters=values)

    def close_saved_figure(self):
        if self._saved_figure is None: return
        self._invalidate()
        self.invalidateBackgroundSelection()
        previous = self._measurement_workspace
        self._states, self._mode, self._folder = previous['states'], previous['mode'], previous['folder']
        self._measurement_workspace = self._saved_figure = None
        self._dataset_cache = previous['cache']
        self._result, self._result_paths = previous['result'], previous['paths']
        self._result_sequence = self._sequence if self._result is not None else -1
        self._status, self._error = previous['status'], previous['error']
        if previous['pending']:
            self._queue(immediate=True, force=True)
        elif previous['preview'] is not None:
            self._reflection_visible = True
            self._reflection_preview = previous['preview']
            self._render_reflection_preview()
        elif self._current():
            self._render()
            if previous['limits'] and self._canvas and self._canvas.figure.axes:
                xlim, ylim = previous['limits']
                self._canvas.figure.axes[0].set(xlim=xlim, ylim=ylim)
                self._canvas.draw_idle()
        elif self._canvas:
            self._canvas.figure.clear(); self._canvas.draw_idle()
        self.snapshotChanged.emit()

    def _queue(self, immediate=False, force=False, keep_reference=False):
        reopen = keep_reference and (self._reflection_visible or self._reopen_reference)
        self._invalidate(); self._save()
        self._reopen_reference = reopen
        if self._state['source']:
            self._status = 'Waiting to update Megasweep…'; self._error = ''
            if self._state['values'].get('auto_update', True) or force: self._timer.start(0 if immediate else 180)
            else: self._status = 'Parameters changed. Select Update to process.'
        self.snapshotChanged.emit()

    def _rebind_background(self):
        if self._background_mode == self._mode and self._background.snapshot['selectionOpen'] and self._background_folder():
            self._background.open_inline_selection(self._background_folder(), self._state['values']['background_role'], self._state['backgrounds'])

    def _update_normalization(self):
        if self._reflection_visible or self._operation in ('reflection-preview', 'background-preview'):
            self._invalidate()
            self._save()
            self.invoke('reflection-preview')
        else:
            self._queue()

    @Slot()
    def invalidateBackgroundSelection(self):
        self._background_error = ''
        self._background_mode = None
        self._background._stop_selection()
        self._background.snapshotChanged.emit()

    @Slot(str)
    def setExperimentFolder(self, value):
        if self._saved_figure is not None: return
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
        if self._saved_figure is not None:
            self.statusMessage.emit('Leave figure editing before opening another scan.')
            return
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
            self._state['preview_frame'] = None
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
            if self._saved_figure is not None: return
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
                values['auto_scale_windows'] = False
                self._update_normalization()
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
        if key == 'cmap' and value not in _MAP_COLORMAPS: return
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
            if value == old and key not in ('x_reverse', 'y_reverse'): return
            if key in ('view', 'transformed'):
                display_key = values['view'] + ('/transformed' if values['transformed'] else '/original')
                values['_display_states'][display_key] = {name: values[name] for name in _DISPLAY_KEYS}
            if self._mode == 'Reflection' and key == 'view' and old != 'slice':
                values['last_map_view'] = old
                if value == 'slice' and values['cut_coordinates'] == 'explicit':
                    data = self._cached_dataset(self._state['source'], self._mode)
                    if data and data['axis_space'] == 'generic' and data.get('top_gate_axis') is None and values['cut_type'] in ('doping', 'efield'):
                        import numpy as np
                        xs = data['x_data']
                        finite = xs[np.isfinite(xs)]
                        if finite.size:
                            values.update(cut_type='x', cut_value=float(finite[np.argmin(np.abs(finite - values['cut_value']))]))
            if key == 'auto_scale_windows' and not value:
                from core.megasweep import has_legacy_scale_windows
                info = self._reference_scale_info()
                if info.get('windows') and has_legacy_scale_windows(values['background_scale_windows']):
                    values['background_scale_windows'] = list(info['windows'])
            values[key] = value
            if self._mode == 'Reflection' and key in ('energy_min', 'energy_max'):
                values['auto_energy_window'] = False
            if self._mode == 'Reflection' and key == 'energy_target':
                values['auto_energy_target'] = False
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
                color_artist = None
                if (key in ('color_auto', 'cmap') and self._current() and not self._reflection_visible
                        and self._canvas and self._canvas.figure.axes and self._canvas.figure.axes[0].collections):
                    color_artist = self._canvas.figure.axes[0].collections[0]
                    if key == 'color_auto' and not value:
                        values['vmin'], values['vmax'] = map(float, color_artist.get_clim())
                display_key = values['view'] + ('/transformed' if values['transformed'] else '/original')
                values['_display_states'][display_key] = {name: values[name] for name in _DISPLAY_KEYS}
                self._save()
                if self._reflection_visible: self._render_reflection_preview()
                elif color_artist is not None:
                    if key == 'cmap':
                        color_artist.set_cmap(value)
                    else:
                        color_artist.set_clim(*self._map_color_limits())
                    self._canvas.draw_idle()
                    self.plotLimitsChanged.emit()
                elif key in ('x_reverse', 'y_reverse') and self._current() and self._canvas and self._canvas.figure.axes:
                    ax = self._canvas.figure.axes[0]
                    axis = ax.xaxis if key == 'x_reverse' else ax.yaxis
                    axis.set_inverted(value)
                    self._canvas.draw_idle()
                else: self._render()
                self.snapshotChanged.emit()
            elif key in ('auto_background_scale', 'auto_scale_windows', 'background_average',
                         'background_scale', 'background_scale_frame', 'background_scale_x', 'background_scale_y'):
                self._update_normalization()
            elif key == 'auto_update':
                self._save()
                if value and not self._current(): self._queue()
                else: self.snapshotChanged.emit()
            else: self._queue(keep_reference=self._mode == 'Reflection' and values['view'] in ('rc_p2p', 'rc_peak', 'rc_energy'))
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
        self._save_locations()
        if normalized and self._mode == 'Reflection' and self._state['source']:
            self._invalidate()
            self._save()
            self.invoke('background-preview')
        else:
            self._queue(immediate=True)

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
            if mode == 'Reflection':
                _resolve_reflection_energies(data, values)
            if mode == 'Transport':
                suggestion = data.get('axis_suggestion', {})
                values['x_column'] = values['x_column'] or suggestion.get('x') or ''
                values['y_column'] = values['y_column'] or suggestion.get('y') or ''
                if suggestion.get('kind') == 'curve' and not values['y_column']: values['view'] = 'transport_curve'
                candidates = [col for col in data.get('columns', []) if col not in (values['x_column'], values['y_column'])]
                suggested_channel = suggestion.get('channel')
                if suggested_channel not in candidates: suggested_channel = ''
                values['channel'] = values['channel'] or suggested_channel or (candidates[0] if candidates else '')
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
        elif kind in ('reflection-preview', 'background-preview'):
            if kind == 'background-preview':
                values = self._state['values']
                key = (self._state['source'], self._mode, values['x_column'], values['y_column'])
                self._dataset_cache = {k: v for k, v in self._dataset_cache.items() if k[1] != self._mode}
                self._dataset_cache[key] = payload['dataset']
                for name in ('energy_min', 'energy_max', 'energy_target'):
                    values[name] = payload['parameters'][name]
            self._reflection_preview = payload; self._reflection_visible = True
            selected_frame = self._state.get('preview_frame')
            if selected_frame is not None and selected_frame < payload.get('frame_count', 0):
                self._select_reference_frame(selected_frame)
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
            if self._mode == 'Reflection':
                for key in ('energy_min', 'energy_max', 'energy_target'):
                    self._state['values'][key] = used_values[key]
            if self._mode == 'Transport':
                for key in ('x_column', 'y_column', 'channel', 'view', 'cut_type', 'cut_value'): self._state['values'][key] = used_values[key]
            self._dataset_cache = {key: value for key, value in self._dataset_cache.items() if key[1] != cache_key[1]}
            self._dataset_cache[cache_key] = data
            if analysis_error:
                self._message(analysis_error, error=True); return
            self._result = result; self._result_sequence = sequence
            if self._reopen_reference:
                self._reopen_reference = False
                self.invoke('reflection-preview')
            else:
                self._message('Megasweep analysis complete.'); self._render()
        else:
            self._result_sequence = sequence
            self._result_paths = [str(p) for p in (payload.values() if isinstance(payload, dict) else payload if isinstance(payload, (list, tuple)) else [payload])]
            self._message('Megasweep export complete.')
        self.operationCompleted.emit(kind, elapsed)

    def _reference_point(self):
        from core.megasweep.megasweep_analysis import display_axis_name

        preview = self._reflection_preview
        if not preview or 'points' not in preview:
            return {}
        x, y = preview['x'], preview['y']
        frames = preview['points'].get((x, y), [])
        return dict(xName=display_axis_name(preview['dataset']['x_name']),
                    yName=display_axis_name(preview['dataset']['y_name']),
                    x=x, y=y,
                    repeats=len(frames), repeat=frames.index(preview['frame']) + 1 if frames else 0)

    @Slot(str, float)
    def setReferenceCoordinate(self, axis, value):
        from math import isfinite
        preview = self._reflection_preview
        if not self._reflection_visible or not preview or not preview['points']:
            return
        if axis not in ('x', 'y') or not isfinite(value):
            return
        index = 0 if axis == 'x' else 1
        other = preview['y' if axis == 'x' else 'x']
        point = min(preview['points'], key=lambda p: (abs(p[index] - value), abs(p[1 - index] - other)))
        if point != (preview['x'], preview['y']):
            self.setReferenceFrame(preview['points'][point][0])

    @Slot(float, float, int)
    def setReferencePoint(self, x, y, repeat):
        if not self._reflection_visible:
            return
        frames = self._reflection_preview['points'].get((x, y), [])
        if 1 <= repeat <= len(frames):
            self.setReferenceFrame(frames[repeat - 1])

    def _select_reference_frame(self, frame):
        from core.megasweep.megasweep_analysis import compute_rc_spectra
        preview = self._reflection_preview
        data = preview['dataset']
        raw = data['Intensity'][frame]
        preview.update(frame=frame, raw=raw, x=float(data['x_data'][frame]), y=float(data['y_data'][frame]),
                       rc=compute_rc_spectra(raw, preview['background'], preview['scale']))

    @Slot(int)
    def setReferenceFrame(self, frame):
        if not self._reflection_visible or not 0 <= frame < self._reflection_preview.get('frame_count', 0):
            return
        self._state['preview_frame'] = frame
        self._select_reference_frame(frame)
        self._render_reflection_preview()
        self.snapshotChanged.emit()

    @Slot(str)
    def invoke(self, action):
        if self._closed: return
        if action == 'close-slide-figure':
            if self._saved_figure is not None: self.savedFigureCloseRequested.emit()
            return
        if self._busy and self._operation == 'export' and action != 'cancel':
            self.statusMessage.emit('Wait for export to finish or cancel it first.'); return
        if action in ('return-to-map', 'show-spectral-cut'):
            self._reopen_reference = False
            self._reflection_visible = False; self._reflection_preview = None
            values = self._state['values']
            if action == 'show-spectral-cut' and values['view'] != 'slice':
                self.setParameter('view', 'slice')
                return
            if action == 'return-to-map' and self._mode == 'Reflection' and values['view'] == 'slice':
                self.setParameter('view', values['last_map_view'])
                return
            if self._current():
                self._render(); self.snapshotChanged.emit()
            else:
                if self._canvas:
                    self._canvas.figure.clear(); self._canvas.draw_idle()
                self._queue(immediate=True, force=True)
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
        elif action in ('reflection-preview', 'background-preview'):
            state = self._state; path = state['source']; mode = self._mode; values = self._effective_values(); backgrounds = tuple(state['backgrounds'])
            if mode != 'Reflection' or not path: return
            cached = self._cached_dataset(path, mode)
            def reflection_preview(cancel):
                from core.megasweep.workflow import load_dataset, reflection_preview
                data = cached if cached is not None else load_dataset(path, mode, values.get('x_column') or None, values.get('y_column') or None)
                _resolve_reflection_energies(data, values)
                preview = reflection_preview(data, values, backgrounds, cancelled=cancel.is_set)
                if values['view'] == 'slice':
                    values['view'] = values['last_map_view']
                from math import isfinite
                points = {}
                for frame, (x, y) in enumerate(zip(data['x_data'], data['y_data'])):
                    if isfinite(x) and isfinite(y):
                        points.setdefault((float(x), float(y)), []).append(frame)
                preview.update(dataset=data, frame_count=len(data['Intensity']), parameters=values,
                               points=points)
                return preview
            self._start(action, reflection_preview)
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
            if action in ('export', 'save'):
                values = self._export_values()
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

    @Slot(result='QVariantList')
    def mapLimits(self):
        if not self._current() or self._reflection_visible or not self._canvas or not self._canvas.figure.axes:
            return []
        ax = self._canvas.figure.axes[0]
        values = self._state['values']
        bounds = [('x_auto', 'X', ax.get_xlabel(), ax.get_xlim()), ('y_auto', 'Y', ax.get_ylabel(), ax.get_ylim())]
        if self._result.get('kind') in ('map', 'slice') and ax.collections:
            bounds.append(('color_auto', 'Color', self._result['signal_label'], ax.collections[0].get_clim()))
        rows = [dict(key=key, axis=axis, label=label, automatic=values[key], reversed=bool(limits[0] > limits[1]),
                     low=float(min(limits)), high=float(max(limits)))
                for key, axis, label, limits in bounds]
        if rows[-1]['key'] == 'color_auto':
            rows[-1].update(cmap=values['cmap'], colormaps=list(_MAP_COLORMAPS))
        return rows

    @Slot('QVariantList', result=str)
    def applyMapLimits(self, rows):
        from math import isfinite
        if not self._current() or self._reflection_visible or self._busy:
            return 'Update the analysis map before setting limits.'
        values = self._state['values']
        keys = [('x_auto', 'xmin', 'xmax'), ('y_auto', 'ymin', 'ymax')]
        if self._result.get('kind') in ('map', 'slice'):
            keys.append(('color_auto', 'vmin', 'vmax'))
        if len(rows) == 1:
            keys = [entry for entry in keys if entry[0] == rows[0].get('key')]
        changes = {}
        try:
            for row, (key, low, high) in zip(rows, keys, strict=True):
                changes[key] = bool(row['automatic'])
                if not changes[key]:
                    left, right = float(row['low']), float(row['high'])
                    if not (isfinite(left) and isfinite(right) and left < right):
                        raise ValueError()
                    changes.update({low: left, high: right})
        except (ValueError, TypeError, KeyError):
            return 'Enter two finite numbers with Min below Max.'
        values.update(changes)
        display_key = values['view'] + ('/transformed' if values['transformed'] else '/original')
        values['_display_states'][display_key] = {key: values[key] for key in _DISPLAY_KEYS}
        self._save()
        ax = self._canvas.figure.axes[0]
        for key, low, high in keys:
            if key == 'color_auto':
                ax.collections[0].set_clim(*self._map_color_limits())
            else:
                name = key[0]
                axis = ax.xaxis if name == 'x' else ax.yaxis
                if values[key]:
                    ax.autoscale(enable=True, axis=name)
                else:
                    bounds = (values[low], values[high])
                    setter = ax.set_xlim if name == 'x' else ax.set_ylim
                    if values[name + '_reverse']:
                        bounds = bounds[::-1]
                    setter(*bounds)
                axis.set_inverted(values[name + '_reverse'])
        self._canvas.draw_idle()
        self.plotLimitsChanged.emit()
        self.snapshotChanged.emit()
        return ''

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

    def _reference_scale_info(self):
        if self._reflection_visible and self._reflection_preview:
            return self._reflection_preview.get('scale_info', {})
        if self._current():
            return self._result.get('diagnostics', {}).get('background_scale_info', {})
        return {}

    def _groups(self):
        values = self._state['values']; enabled = not (self._busy and self._operation == 'export')
        view = values['view']; mode = self._mode
        choices = {'view': [('intensity','Intensity'),('peak','Peak energy'),('slice','Spectral slice'),('ibias','Bias current'),('resistance','Resistance')] if mode == 'PL' else [('rc_p2p','RC peak-to-peak'),('rc_energy','RC at fixed energy'),('rc_peak','RC peak position'),('slice','Spectral slice'),('ibias','Bias current'),('resistance','Resistance')] if mode == 'Reflection' else [('transport_map','Transport map'),('transport_curve','Transport curve'),('slice','Transport cut')],
            'convention':[('TG+rBG', 'D = TG + r × BG\nE = TG − r × BG'),
                          ('rTG+BG', 'D = r × TG + BG\nE = r × TG − BG')], 'cut_type':[('doping','Doping'),('efield','Electric field'),('x','Original X'),('y','Original Y')],
            'derivative':[('spectrum','Spectrum'),('second_derivative','Second derivative')],
            'background_average':[('all_frames','All frames equally'),('file_equal','Files equally'),('first_frame','First frame'),('last_frame','Last frame')],
            'rc_feature':[('auto','Automatic'),('peak','Peak'),('dip','Dip')],
            'cut_coordinates':[('explicit','Explicit cut type'),('follow_map','Follow map axes'),('original','Original axes'),('transformed','Transformed axes')], 'cut_axis':[('first','First axis'),('second','Second axis')], 'batch_axes':[('selected','Selected axis'),('both','Both axes')]}
        cache = next((data for key,data in self._dataset_cache.items() if key[0] == self._state['source'] and key[1] == mode), None)
        current_data = self._dataset_cache.get((self._state['source'], mode, values['x_column'], values['y_column']))
        axes = {}
        from core.megasweep.megasweep_analysis import display_axis_name
        for axis in ('x', 'y'):
            name = values[axis + '_column'] or (current_data or {}).get(axis + '_name', '')
            axes[axis] = f'{axis.upper()} ({display_axis_name(name)})' if name else axis.upper()
        labels = dict(energy_target='Target energy (eV)', x_column='X column', y_column='Y column',
                      transformed='Use doping / E-field coordinates', tg_is_y='Top gate (TG)', auto_scale_windows='Auto-select scale windows',
                      convention='Doping (D) and E-field (E)', ratio='Lever-arm ratio (r)',
                      background_scale_x='Reference ' + axes['x'], background_scale_y='Reference ' + axes['y'],
                      background_scale_frame='Reference frame (blank uses X/Y)')
        choices['tg_is_y'] = [(True, axes['y']), (False, axes['x'])]
        if mode != 'Transport' and view == 'slice':
            x, y = axes['x'], axes['y']
            directions = [('x', f'Along {y} · hold {x}'), ('y', f'Along {x} · hold {y}'),
                          ('doping', 'Along E-field · hold doping'), ('efield', 'Along doping · hold E-field')]
            choices['cut_type'] = directions
            transformed = values['cut_coordinates'] == 'transformed' or (
                values['cut_coordinates'] == 'follow_map' and values['transformed'])
            axis_directions = directions[2:] if transformed else directions[:2]
            choices['cut_axis'] = [(key, option[1]) for key, option in zip(('first', 'second'), axis_directions)]
            labels.update(cut_type='Cut direction', cut_axis='Cut direction', cut_value='Fixed coordinate value')
        if cache:
            columns = cache['columns'] if mode == 'Transport' else cache['axis_columns']
            for key in ('x_column', 'y_column', 'channel'):
                label = 'Select signal' if key == 'channel' else 'Detect from CSV'
                choices[key] = [('', label)] + [(str(c), str(c)) for c in columns]
            if mode == 'Transport': choices['cut_type'] = [(str(c),str(c)) for c in columns]
        def field(key):
            value=values[key]; item=dict(key=key,label=labels.get(key, key.replace('_',' ').title()),value=value,enabled=enabled)
            if key in ('x_column', 'y_column') and mode != 'Transport':
                if not value and current_data:
                    item['value'] = current_data['x_name' if key == 'x_column' else 'y_name']
                item['description'] = 'Choose a scan coordinate. A _set column is the requested setpoint and _meas is the instrument readback.'
            if key in ('background_scale_frame','pass_index'): item['type']='text'; item['value']='' if value is None else str(value)
            elif key in choices: item.update(type='choice', options=[dict(value=k,label=v) for k,v in choices[key]])
            elif isinstance(value,bool): item['type']='flag'
            elif isinstance(value,(float,int)): item.update(type='number',min=-1000000000,max=1000000000,step=1 if isinstance(value,int) else .01,decimals=0 if isinstance(value,int) else 6)
            else: item['type']='text'; item['value']='' if value is None else value
            if item['type'] == 'number':
                item['inputWidth'] = 128
            if key in ('energy_min', 'energy_max'):
                item.update(pair='energy_window', pairLabel='Energy (eV)')
            return item
        data = ['x_column', 'y_column', 'view', 'auto_update']
        if mode == 'Transport': data += ['channel','direction','pass_index']
        processing=[]
        if mode != 'Transport':
            if mode == 'Reflection' and view in ('rc_p2p', 'rc_peak'):
                processing.append('auto_energy_window')
            if view in ('intensity','rc_p2p','rc_peak'): processing += ['energy_min','energy_max']
            if view == 'rc_energy': processing += ['auto_energy_target', 'energy_target']
            if view == 'intensity': processing += ['auto_baseline','baseline']
            if view == 'peak': processing += ['sg_window','sg_poly']
        groups = [dict(key='data', title='Scan axes and signal' if mode == 'Transport' else 'Scan axes',
                       fields=[field(k) for k in data])]
        if mode != 'Transport':
            top_gate = current_data.get('top_gate_axis') if current_data else None
            cut_coordinates = values['cut_coordinates']
            if view != 'slice' or cut_coordinates == 'follow_map':
                uses_gate_transform = values['transformed']
            elif cut_coordinates == 'explicit':
                uses_gate_transform = values['cut_type'] in ('doping', 'efield')
            else:
                uses_gate_transform = cut_coordinates == 'transformed'
            gate_fields = []
            if view != 'slice' or cut_coordinates == 'follow_map':
                if top_gate is not None or values['transformed']:
                    gate_fields.append(field('transformed'))
            if uses_gate_transform and top_gate is not None:
                if top_gate == 'manual':
                    gate_fields.append(field('tg_is_y'))
                gate_fields.extend([field('convention'), field('ratio')])
            if gate_fields:
                groups.append(dict(key='coordinates', title='Gate coordinates', fields=gate_fields))
        if view == 'slice':
            if mode == 'Transport':
                cut = ['cut_type', 'cut_value']
            else:
                cut_key = 'cut_type' if values['cut_coordinates'] == 'explicit' else 'cut_axis'
                cut = ['cut_coordinates', cut_key, 'cut_value', 'epsilon', 'min_points', 'exact_coordinates',
                       'derivative', 'derivative_window']
            groups.append(dict(key='slice', title='Transport cut' if mode == 'Transport' else 'Spectral cut',
                               fields=[field(k) for k in cut]))
        if processing:
            title = {'rc_p2p': 'Energy window', 'rc_peak': 'Energy window', 'rc_energy': 'Target energy'}.get(view, 'Processing')
            groups.append(dict(key='processing', title=title, fields=[field(k) for k in processing]))
        if view == 'rc_peak':
            groups.append(dict(key='peak_detection', title='Peak detection',
                               fields=[field(k) for k in ('rc_feature', 'sg_window', 'sg_poly')]))
        reference_groups = []
        if mode == 'Reflection':
            normalization = [field('background_average'), field('auto_background_scale')]
            if values['auto_background_scale']:
                normalization.append(field('auto_scale_windows'))
                if values['auto_scale_windows']:
                    info = self._reference_scale_info()
                    if info.get('automatic_windows'):
                        ranges = ', '.join(f'{lo:.6g}–{hi:.6g}' for lo, hi in info['windows'])
                        normalization.append(dict(key='selected_scale_windows', type='summary',
                                                  value=f'Selected windows: {ranges} eV'))
            else:
                normalization.append(field('background_scale'))
            reference_groups.append(dict(key='background_normalization', title='Background normalization', fields=normalization))
            fields = [field('background_scale_frame')]
            if values['background_scale_frame'] is None:
                fields.extend([field('background_scale_x'), field('background_scale_y')])
            if values['auto_background_scale'] and not values['auto_scale_windows']:
                for index, pair in enumerate(values['background_scale_windows']):
                    for bound, value in enumerate(pair): fields.append(dict(key=f'scale_window_{index}_{bound}',pair=f'scale_window_{index}',label=f'Scale window {index+1} '+('minimum' if not bound else 'maximum')+' (eV)',type='number',value=value,min=0,max=100,step=.001,decimals=6,enabled=enabled))
            reference_groups.append(dict(key='reference_settings', title='Advanced reference settings', advanced=True, fields=fields))
        if view == 'slice' and mode != 'Transport':
            groups.append(dict(key='display', title='Batch export', fields=[field('batch_axes')]))
        return groups, reference_groups
