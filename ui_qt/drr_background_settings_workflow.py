"""One background entry point; editors prepare drafts and Apply commits them.

The legacy processing widgets remain the numerical adapter. Source intent is
separate from that adapter so changing groups cannot silently change BN to P/N.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, Qt, Slot, QTimer
from PySide6.QtWidgets import QDialog, QFileDialog

from core.drr_sources import DrrBackgroundResolution, DrrMeasurementAssignment, portable_source_name
from core.load_cancellation import LoadCancellation, check_load_cancelled
from ui_qt.common import Worker


FRAME_LABELS = {'first': 'First frame from each file, then average',
                'last': 'Last frame from each file, then average',
                'all': 'Average all frames in each file, then average files'}
KIND_LABELS = {'bn': 'BN', 'p': 'P only', 'n': 'N only', 'pn': 'P + N composite',
               'self': 'Self reference', 'fitted': 'Fitted background', 'custom': 'Custom background'}


def _context(owner):
    from ui_qt.drr_pn_workflow import _context as recipe_context
    return (recipe_context(owner), str(owner._selected_y_axis_spec('drr')),
            owner.drr_sg_window_spin.value(), owner.drr_sg_poly_spin.value())


def _root_key(owner):
    return str(Path(owner.current_folder or '.').resolve()).casefold()


def _silent(widget, value):
    blocked = widget.blockSignals(True)
    try:
        if isinstance(value, bool):
            widget.setChecked(value)
        else:
            widget.setCurrentText(value)
    finally:
        widget.blockSignals(blocked)


def _cache(owner):
    from core.drr_bn_matching import BnMatchCache
    cache = getattr(owner, '_drr_confirmed_bn_cache', None)
    if cache is None:
        try:
            cache = BnMatchCache.from_dict(json.loads(owner.settings.value('drr/confirmed_bn_matches_v1', '{}')))
        except (TypeError, ValueError, KeyError):
            cache = BnMatchCache()
        owner._drr_confirmed_bn_cache = cache
    return cache


def source_intent(owner):
    """Explicit source type is scoped to this experiment, not the whole app."""
    if getattr(owner, '_drr_background_intent_root', None) == _root_key(owner):
        return getattr(owner, '_drr_background_intent', None)
    return None


def current_draft(owner):
    from ui_qt.drr_background_settings import BackgroundDraft
    from ui_qt.drr_region_backgrounds import frame_method
    from core.drr_measured_bn import is_measured_bn_name
    from core.drr_pn_pairing import detect_pn_role
    mode = owner.drr_baseline_combo.currentText()
    files = tuple(owner.drr_baseline_files_manual)
    intent = source_intent(owner)
    kind = ('self' if mode.startswith('Self') else 'fitted' if mode == 'Fitted (sample)' else
            intent.kind if intent is not None else 'custom')
    if mode == 'External' and intent is None and files:
        if all(is_measured_bn_name(name) or '.drr_bn_calibration' in Path(name).parts for name in files):
            kind = 'bn'
        elif all(detect_pn_role(name) == 'P' for name in files):
            kind = 'p'
        elif all(detect_pn_role(name) == 'N' for name in files):
            kind = 'n'
        elif all('pn backgrounds' in str(name).replace('\\', '/').casefold() for name in files):
            kind = 'pn'
    plan = intent.bn_plan if intent is not None else None
    if plan is not None and tuple(plan.selected) != tuple(owner.drr_selected_files):
        plan = None
    which = ('first' if 'first' in mode else 'last') if kind == 'self' else frame_method(owner)
    return BackgroundDraft(kind=kind, files=files, which=which,
        regions=tuple(owner._drr_background_regions),
        version=intent.version if intent is not None and kind == 'pn' and files else None,
        allow_stale=intent.allow_stale if intent is not None else False,
        fixed=owner.drr_pin_baseline_chk.isChecked(),
        auto_bn_compare=owner.drr_bn_auto_chk.isChecked(),
        fitted_result=dict(getattr(owner, '_drr_fitted_settings', {})),
        calibration=intent.calibration if intent is not None and kind == 'bn' and files else None,
        bn_plan=plan if kind == 'bn' else None,
        bn_candidates=intent.bn_candidates if intent is not None else None,
        bn_warnings_acknowledged=bool(intent is not None and plan is intent.bn_plan
                                      and intent.bn_warnings_acknowledged))


def _select_files(owner, dialog, kind, current, title):
    owner._drr_source_dialog_accepted = None
    role = {'p': 'P', 'n': 'N', 'bn': 'BN', 'bn_candidates': 'BN'}.get(kind)
    role_options = {'background_role': role} if role else {}
    result = owner.drr_controller._open_drr_source_dialog(
        title=title, selected=list(current), baseline_mode=True, independent_background=True,
        **role_options)
    if owner._drr_source_dialog_accepted is False:
        return None
    # A manual choice can supply an unlabelled source, but an explicitly
    # contradictory source name must never be presented as P, N or measured BN.
    from core.drr_measured_bn import is_measured_bn_name
    from core.drr_pn_selection import pn_source_matches_role
    for name in result:
        if kind in ('p', 'n') and not pn_source_matches_role(name, kind.upper()):
            raise ValueError(f'{Path(name).name} is not a {kind.upper()} reference. Choose the matching source type.')
        if kind in ('bn', 'bn_candidates') and not is_measured_bn_name(name):
            raise ValueError(f'{Path(name).name} is not named BNback / BNbackground. Use Advanced for a custom reference.')
    return tuple(result)


def _browse_background(owner, dialog, role, single):
    """Native external import; source-body validation belongs to Apply's worker."""
    title = f'Choose one {role} background' if single else f'Choose {role} background files'
    if single:
        name, _ = QFileDialog.getOpenFileName(dialog, title, str(owner.current_folder), 'DRR CSV files (*.csv)')
        files = (name,) if name else ()
    else:
        files, _ = QFileDialog.getOpenFileNames(dialog, title, str(owner.current_folder), 'DRR CSV files (*.csv)')
    return tuple(portable_source_name(owner.current_folder, name) for name in files)


class _InlineCatalogBridge(QObject):
    """Subscribe only while settings are open; reuse the owner's async catalog."""
    def __init__(self, owner, dialog):
        super().__init__(dialog)
        self.owner, self.dialog = owner, dialog
        self.folder = str(owner.current_folder)
        self._connections = []
        for name, slot in (('drr_catalog_preview_ready', self.preview),
                           ('drr_catalog_refresh_finished', self.finished)):
            signal = getattr(owner, name, None)
            if signal is not None:
                signal.connect(slot)
                self._connections.append((signal, slot))

    def _same_folder(self, folder):
        return (str(folder).casefold() == self.folder.casefold()
                and str(self.owner.current_folder).casefold() == self.folder.casefold())

    def refresh(self, *_, automatic=False):
        if getattr(self.dialog, '_closed', False):
            return
        if not self._same_folder(self.owner.current_folder):
            self.dialog.set_catalog_busy(False, 'Experiment changed. Reopen Background settings.')
            return
        if automatic:
            from ui_qt.drr_picker_cache import freshness
            if (getattr(self.owner, '_drr_refresh_running', False)
                    or not freshness(self.owner).due(self.folder, getattr(self.owner, '_drr_include_all_sources', False))):
                return
        self.dialog.set_catalog_busy(True)
        selected = tuple(dict.fromkeys((*self.dialog.measurements, *self.dialog.draft.files,
                                        *(self.dialog.draft.bn_candidates or ()))))
        self.owner._queue_drr_catalog_refresh(auto=automatic, force=not automatic,
            old_source_files={source.source for source in self.owner.drr_available_sources},
            selected_sources=selected, catalog_only=True)

    @Slot(str)
    def preview(self, folder):
        if self._same_folder(folder):
            self.dialog.set_catalog(tuple(self.owner.drr_available_sources))

    @Slot(str, bool)
    def finished(self, folder, success):
        if self._same_folder(folder):
            if success:
                self.dialog.set_catalog(tuple(self.owner.drr_available_sources))
            self.dialog.set_catalog_busy(False, '' if success else 'Refresh failed; current choices are retained.')

    def disconnect(self):
        for signal, slot in self._connections:
            signal.disconnect(slot)
        self._connections.clear()


def _composite(owner, parent, draft, *, editing=False):
    from ui_qt.drr_background_workspace import DrrBackgroundWorkspace
    dialog = DrrBackgroundWorkspace(parent, folder=owner.current_folder,
        catalog=tuple(owner.drr_available_sources), measurements=tuple(owner.drr_selected_files),
        initial_files=draft.files, initial_which=draft.which, initial_regions=draft.regions,
        select_files=lambda files, title: _select_files(owner, parent, 'custom', files, title) or files,
        pool=owner.thread_pool, initial_page='edit' if editing else 'current', kept_background=draft.fixed)
    dialog.setWindowTitle('DRR · Edit background draft' if editing else 'DRR · P + N composite')
    dialog.tabs.setTabVisible(0 if editing else 1, False)
    dialog.edit_selected_btn.hide()
    dialog.apply_btn.setText('Use in settings')
    dialog.note.setText('Saving creates a reusable version. Use in settings returns a draft; Apply in Background settings changes the main view.')
    try:
        if dialog.exec() != QDialog.Accepted or dialog.selection is None:
            return None
        selected = dialog.selection
        return replace(draft, kind='custom' if editing else 'pn', files=selected.files,
            which=selected.which, regions=selected.regions, version=selected.version,
            allow_stale=selected.allow_stale, bn_plan=None, calibration=None)
    finally:
        dialog.deleteLater()


def _custom(owner, parent, draft):
    files = _select_files(owner, parent, 'custom', draft.files, 'Choose custom background files')
    return (replace(draft, kind='custom', files=files, bn_plan=None, calibration=None,
                    version=None, regions=(), fixed=False) if files else None)


def _fit(owner, parent, draft):
    from ui_qt.drr_background_dialog import DrrBackgroundDialog
    files = tuple(owner.drr_selected_files)
    if not files or any(Path(name).suffix.lower() != '.csv' for name in files):
        raise ValueError('Select DRR CSV measurements before configuring a fitted background.')
    dialog = DrrBackgroundDialog(str(owner.current_folder), files,
        y_axis=owner._selected_y_axis_spec('drr'), settings=draft.fitted_result or {}, references=(),
        pool=owner.thread_pool, initial_gate=owner.drr_spins['gate'].value(), settings_store=None,
        comparison_window=owner.drr_sg_window_spin.value(), comparison_poly=owner.drr_sg_poly_spin.value(),
        parent=parent, draft_mode=True)
    result = []
    def selected(recipe):
        result.append(recipe)
        dialog.accept()
    dialog.applied.connect(selected)
    # This editor updates its button during preview; retain its processing
    # behavior and explicitly describe the outer Apply boundary.
    dialog.setWindowTitle('DRR · Fit background draft — return preview to settings')
    dialog.info.setText(dialog.info.text() + ' Use in settings returns this preview. Apply there changes the main view.')
    try:
        dialog.exec()
        return replace(draft, kind='fitted', fitted_result=result[-1], fixed=False) if result else None
    finally:
        dialog.stop()
        dialog.deleteLater()


def _calibration(owner, parent, draft):
    from ui_qt.drr_bn_calibration_dialog import DrrBnCalibrationDialog
    from ui_qt.drr_bn_workflow import _compare_and_save, _comparison_settings, _cached_payload
    files, which = draft.files, draft.which
    if draft.bn_plan is not None:
        recipes = {(row.assignment.baseline_files, row.assignment.baseline_which)
                   for row in draft.bn_plan.rows if row.assignment is not None}
        if not draft.bn_plan.ready or len(recipes) != 1:
            raise ValueError('BN calibration requires one common measured BN. Per-measurement matching remains available without calibration.')
        files, which = next(iter(recipes))
    if not files:
        raise ValueError('Choose a measured BN before configuring calibration.')
    folder, selected, axis = str(owner.current_folder), tuple(owner.drr_selected_files), owner._selected_y_axis_spec('drr')
    if not selected:
        raise ValueError('Select a measurement group before comparing BN calibration.')
    payload = (draft.calibration or {}).get('payload')
    if payload is None and files == tuple(owner.drr_baseline_files_manual):
        payload = _cached_payload(owner)
    def compare(settings, progress, log, cancellation):
        return _compare_and_save(folder, selected, files, which, settings, axis,
                                 progress=progress, log=log, cancellation=cancellation)
    dialog = DrrBnCalibrationDialog(parent,
        context_label=f'{len(selected)} measurements · BN draft · changes apply from Background settings',
        settings={**_comparison_settings(owner, payload), 'auto_apply': False},
        run_comparison=compare, initial_payload=payload,
        current_mode='calibrated' if any('.drr_bn_calibration' in Path(name).parts for name in files) else 'original')
    dialog.auto_apply.setChecked(False)
    dialog.auto_apply.hide()
    dialog.use_btn.setText('Use validated result in settings')
    dialog.use_btn.setToolTip('Returns the algorithm-selected validated candidate to Background settings.')
    dialog.original_btn.setText('Use original BN in settings')
    result = []
    def choose(value, original=False):
        result.append(replace(draft, kind='bn', files=tuple(files), which=which,
            calibration={'payload': value, 'original': original}, bn_plan=None, fixed=False))
        dialog.accept()
    dialog.applyRequested.connect(lambda value: choose(value))
    dialog.originalRequested.connect(lambda: choose(getattr(dialog, '_payload', None) or payload, True))
    try:
        dialog.exec()
        return result[-1] if result else None
    finally:
        dialog.deleteLater()


def edit_settings(owner, *, initial_kind=None):
    from ui_qt.drr_background_settings import DrrBackgroundSettings
    if not owner.current_folder:
        owner._show_error('Open an experiment folder before choosing a background.')
        return
    previous = getattr(owner, '_drr_settings_dialog', None)
    if previous is not None:
        previous.raise_()
        previous.activateWindow()
        return
    was_automatic = getattr(owner, '_drr_external_auto_selected', False)
    owner.drr_controller._cancel_auto_external()
    owner._drr_external_auto_selected = was_automatic
    context = _context(owner)
    applied = current_draft(owner)
    draft = applied
    if initial_kind is not None and draft.kind != initial_kind:
        draft = replace(draft, kind=initial_kind, files=(), regions=(), version=None,
                        calibration=None, bn_plan=None, fixed=False, which='all')
    dialog = DrrBackgroundSettings(owner, current=applied, initial_draft=draft,
        measurements=tuple(owner.drr_selected_files),
        folder=str(owner.current_folder), pool=owner.thread_pool, confirmed_bn_cache=_cache(owner),
        catalog=tuple(owner.drr_available_sources) if hasattr(owner, 'drr_available_sources') else None,
        refresh_catalog=(lambda current: current._catalog_bridge.refresh()) if hasattr(owner, '_queue_drr_catalog_refresh') else None,
        browse_files=lambda *args: _browse_background(owner, *args),
        select_files=lambda *args: _select_files(owner, *args),
        configure_composite=lambda parent, value: _composite(owner, parent, value),
        import_custom=lambda parent, value: _custom(owner, parent, value),
        edit_composite=lambda parent, value: _composite(owner, parent, value, editing=True),
        configure_fit=lambda parent, value: _fit(owner, parent, value),
        configure_calibration=lambda parent, value: _calibration(owner, parent, value))
    pending = getattr(owner, '_drr_pending_bn_plan', None)
    if pending is not None and pending.selected == tuple(owner.drr_selected_files) and draft.kind == 'bn':
        dialog.set_bn_plan(pending)
    owner._drr_settings_dialog = dialog
    bridge = None
    if hasattr(owner, 'drr_available_sources') and hasattr(owner, '_queue_drr_catalog_refresh'):
        bridge = _InlineCatalogBridge(owner, dialog)
        dialog._catalog_bridge = bridge
        QTimer.singleShot(150, dialog, lambda: bridge.refresh(automatic=True))
    try:
        if dialog.exec() != QDialog.Accepted or dialog.selection is None:
            return
        if _context(owner) != context or getattr(owner, '_is_closing', False):
            owner._status('Selection changed. Reopen Background settings for the current group.')
            return
        apply_settings(owner, dialog.selection)
    finally:
        if bridge is not None:
            bridge.disconnect()
        owner._drr_settings_dialog = None
        dialog.deleteLater()
        owner.drr_edit_baselines_btn.setFocus()


def _validate(folder, selected, draft, axis, derivative_settings, *, progress, log):
    """All expensive file verification runs off the UI thread before mutation."""
    from core.data_io import _validate_drr_external_spectral_grids
    from core.drr_sources import assess_background_gate_files
    check_load_cancelled()
    if draft.bn_plan is not None and draft.calibration is not None:
        raise ValueError('Choose the BN mapping or the calibrated reference, then review the draft again.')
    if draft.kind == 'bn' and draft.bn_plan is not None:
        from core.drr_bn_matching import BnMatchCache
        plan = draft.bn_plan
        if (not plan.ready or tuple(plan.selected) != tuple(selected)
                or Path(plan.root).resolve() != Path(folder).resolve()):
            raise ValueError('BN inputs changed or matching is incomplete. Match BN again before applying.')
        if plan.warnings and not draft.bn_warnings_acknowledged:
            raise ValueError('Review and acknowledge the BN matching warnings before applying.')
        return draft, BnMatchCache().prepare_confirm(plan, warnings_acknowledged=draft.bn_warnings_acknowledged)
    elif draft.calibration is not None:
        payload = draft.calibration.get('payload')
        if not payload:
            raise ValueError('Compare BN before choosing a calibrated or original result.')
        source = payload['source_identity']
        from ui_qt.drr_bn_workflow import _paths, _verify_version, _payload_derivative_settings
        if (_paths(folder, selected) != _paths(folder, [x['path'] for x in source['measurements']])
                or Path(source['root']).resolve() != Path(folder).resolve()
                or str(source['y_axis']) != str(axis)):
            raise ValueError('BN comparison belongs to a different group or axis.')
        if not draft.calibration['original']:
            if _payload_derivative_settings(payload) != derivative_settings:
                raise ValueError('SG settings changed. Compare BN again.')
            version = _verify_version(folder, payload['version_id'], derivative_settings, progress=progress, log=log)
            return draft, version
    elif draft.version is not None:
        from core.drr_pn_store import DrrPnStore
        store = DrrPnStore(folder)
        prepared = store.prepare_use(draft.version.version_id, measurements=selected, allow_stale=draft.allow_stale)
        return replace(draft, files=(portable_source_name(folder, prepared.version.csv_path),)), prepared
    elif draft.kind in ('bn', 'p', 'n', 'custom', 'pn'):
        if not draft.files:
            raise ValueError('Choose background files before applying.')
        sources = tuple(dict.fromkeys((*draft.files, *(name for region in draft.regions for name in region.baseline_files))))
        _validate_drr_external_spectral_grids(folder, selected, sources)
        assessment = assess_background_gate_files(folder, draft.files)
        if assessment.all_constant and len(assessment.profiles) > 1 and not assessment.same_constant_values:
            raise ValueError('These background files have different constant gate values. Use P + N composite to combine P and N.')
    elif draft.kind == 'fitted' and draft.fitted_result:
        from core.drr_fitted_background import validate_recipe
        validate_recipe(draft.fitted_result, selected)
    check_load_cancelled()
    return draft, None


class _SettingsApplyRequest(QObject):
    def __init__(self, owner, draft):
        super().__init__(owner)
        self.owner, self.context = owner, _context(owner)
        self.cancellation = LoadCancellation()
        self.worker = Worker(_validate, str(owner.current_folder), tuple(owner.drr_selected_files), draft,
            self.context[1], self.context[2:], cancellation=self.cancellation)
        self.worker.signals.result.connect(self.complete, Qt.QueuedConnection)
        self.worker.signals.error.connect(self.failed, Qt.QueuedConnection)
        self.worker.signals.finished.connect(self.finished, Qt.QueuedConnection)

    def current(self):
        return (not self.cancellation.cancelled and not getattr(self.owner, '_is_closing', False)
                and getattr(self.owner, '_drr_settings_apply_request', None) is self
                and _context(self.owner) == self.context)

    @Slot(object)
    def complete(self, value):
        if self.current():
            self.owner._drr_settings_apply_request = None
            try:
                _commit(self.owner, *value)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self.owner._show_error('Background could not be applied: ' + str(exc))

    @Slot(str)
    def failed(self, message):
        if self.current():
            self.owner._show_error('Background was not changed.\n' + message.split('\n\n', 1)[0])

    @Slot()
    def finished(self):
        if getattr(self.owner, '_drr_settings_apply_request', None) is self:
            self.owner._drr_settings_apply_request = None
        self.owner.drr_controller._update_drr_baseline_controls()
        self.deleteLater()


def cancel_settings_apply(owner):
    request = getattr(owner, '_drr_settings_apply_request', None)
    if request is not None:
        request.cancellation.cancel()
    owner._drr_settings_apply_request = None


def apply_settings(owner, draft):
    owner.drr_controller._cancel_auto_external()
    request = _SettingsApplyRequest(owner, draft)
    owner._drr_settings_apply_request = request
    owner._status('Checking background inputs before applying…')
    owner.drr_controller._update_drr_baseline_controls()
    owner.thread_pool.start(request.worker)


def _commit(owner, draft, version=None, *, start_load=True, persist=True):
    from ui_qt.drr_bn_workflow import forget_bn_choice
    if not persist and (draft.kind != 'bn' or draft.calibration is not None or draft.version is not None):
        raise ValueError('A group-only background switch requires a measured BN.')
    selected = tuple(owner.drr_selected_files)
    resolution = None
    # Freshness may reject a preview. Do that before forgetting any applied
    # calibration or changing the owning controller's recipe.
    if draft.kind == 'bn' and draft.bn_plan is not None:
        if version is None or version.plan != draft.bn_plan:
            raise ValueError('BN mapping needs verification before application.')
        _cache(owner).commit_confirm(version)
    if draft.calibration is not None:
        from core.drr_bn_store import DrrBnStore
        payload = draft.calibration['payload']
        source = payload['source_identity']
        store = DrrBnStore(owner.current_folder)
        if draft.calibration['original']:
            files = tuple(x['path'] for x in source['baseline_files'])
            store.deactivate(selected, files, source['baseline_which'], restore_original=True)
            draft = replace(draft, files=files, which=source['baseline_which'], fixed=False)
            payload = {**payload, 'original_chosen': True}
            draft = replace(draft, calibration={'payload': payload, 'original': True})
        else:
            store.activate_verified(version, measurements=selected)
            draft = replace(draft, files=(portable_source_name(owner.current_folder, version.csv_path),), which='last', fixed=False)
        from ui_qt.drr_bn_workflow import _remember
        _remember(owner, payload)
    else:
        if draft.version is not None:
            from core.drr_pn_store import DrrPnStore
            DrrPnStore(owner.current_folder).commit_prepared_use(version)
        if persist:
            forget_bn_choice(owner)
    if draft.kind == 'bn' and draft.bn_plan is not None:
        plan = draft.bn_plan
        cache = _cache(owner)
        if persist:
            owner.settings.setValue('drr/confirmed_bn_matches_v1', json.dumps(cache.to_dict()))
        resolution = plan.resolution
        files = tuple(dict.fromkeys(name for item in resolution.assignments for name in item.baseline_files))
        common = len({(item.baseline_files, item.baseline_which) for item in resolution.assignments}) == 1
        draft = replace(draft, files=files, which='all', fixed=draft.fixed and common,
                        bn_plan=None if draft.fixed and common else plan)
    mode = ('Self (first frame)' if draft.which == 'first' else 'Self (last frame)') if draft.kind == 'self' else (
        'Fitted (sample)' if draft.kind == 'fitted' else 'External')
    _silent(owner.drr_baseline_combo, mode)
    _silent(owner.drr_baseline_combine_combo, FRAME_LABELS[draft.which])
    _silent(owner.drr_pin_baseline_chk, draft.fixed and mode == 'External')
    _silent(owner.drr_bn_auto_chk, draft.auto_bn_compare)
    if persist:
        owner.settings.setValue('drr/bn_auto_compare', draft.auto_bn_compare)
    owner._drr_background_regions = draft.regions if mode == 'External' else ()
    owner.drr_baseline_files_manual = list(draft.files) if mode == 'External' else []
    owner.drr_baseline_files_found = []
    owner._drr_assignments = tuple(resolution.assignments) if resolution is not None else ()
    owner._drr_assignments_automatic = resolution is not None
    owner._drr_baseline_user_selected = True
    owner._drr_external_auto_selected = resolution is not None
    owner._drr_background_guess = None
    owner._drr_pending_bn_plan = None
    if draft.kind == 'fitted':
        owner._drr_fitted_settings = draft.fitted_result or {}
        key = 'drr/fitted-background/' + hashlib.sha256(json.dumps(
            [str(owner.current_folder), selected, owner._selected_y_axis_spec('drr')]).encode()).hexdigest()
        owner.settings.setValue(key, json.dumps(owner._drr_fitted_settings))
    owner._drr_background_intent, owner._drr_background_intent_root = draft, _root_key(owner)
    owner._drr_background_intent_selection = selected
    owner._drr_bn_confirmation = version if draft.bn_plan is not None else None
    owner.drr_controller._update_drr_selection_labels()
    owner._status(f'{KIND_LABELS[draft.kind]} applied. All {len(selected)} selected measurements remain in the average.')
    if selected and start_load:
        owner._start_load('DRR', drr_resolution=resolution)


def _follow_bn(folder, selected, draft, cache, cancelled, *, progress, log):
    from core.drr_bn_matching import build_bn_match_plan
    check_load_cancelled()
    confirmed = cache.find(folder, selected, candidate_files=draft.bn_candidates)
    if confirmed is not None:
        return confirmed, True, cache.prepare_confirm(confirmed, warnings_acknowledged=True)
    return build_bn_match_plan(folder, selected, candidate_files=draft.bn_candidates, cancelled=cancelled), False, None


class _FollowBnRequest(QObject):
    def __init__(self, owner, draft):
        super().__init__(owner)
        self.owner, self.draft = owner, draft
        self.folder, self.selected = str(owner.current_folder), tuple(owner.drr_selected_files)
        self.cancelled = Event()
        self.worker = Worker(_follow_bn, self.folder, self.selected, draft, _cache(owner), self.cancelled)
        self.worker.signals.result.connect(self.complete, Qt.QueuedConnection)
        self.worker.signals.error.connect(self.failed, Qt.QueuedConnection)
        self.worker.signals.finished.connect(self.finished, Qt.QueuedConnection)

    def current(self):
        return (not self.cancelled.is_set() and not getattr(self.owner, '_is_closing', False)
                and getattr(self.owner, '_drr_external_request', None) is self
                and str(self.owner.current_folder) == self.folder
                and tuple(self.owner.drr_selected_files) == self.selected
                and self.owner.drr_baseline_combo.currentText() == 'External')

    @Slot(object)
    def complete(self, result):
        if not self.current():
            return
        plan, confirmed, prepared = result
        self.owner._drr_external_request = None
        if confirmed:
            try:
                if (tuple(self.owner._drr_assignments) == tuple(plan.resolution.assignments)
                        and getattr(self.owner, '_drr_background_intent_selection', None) == self.selected):
                    self.owner._drr_bn_confirmation = prepared
                else:
                    _commit(self.owner, replace(self.draft, bn_plan=plan, bn_warnings_acknowledged=True), prepared)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self.owner._drr_pending_bn_plan = None
                self.owner._status('BN inputs changed before reuse. Open Background settings and match again. ' + str(exc))
                self.owner.drr_controller._update_drr_selection_labels()
        else:
            self.owner._drr_pending_bn_plan = plan
            self.owner.drr_controller._update_drr_selection_labels()
            self.owner._status('BN suggestions are available. Review the batches and any flagged measurements in Background settings, then Apply.')

    @Slot(str)
    def failed(self, message):
        if self.current():
            self.owner._status('BN matching could not finish: ' + message.split('\n\n', 1)[0])

    @Slot()
    def finished(self):
        if getattr(self.owner, '_drr_external_request', None) is self:
            self.owner._drr_external_request = None
        self.owner.drr_controller._update_drr_baseline_controls()
        self.deleteLater()


def _bn_confirmation_ready(owner, draft):
    """File freshness alone cannot authorize a mapping for another selection."""
    from core.drr_bn_matching import confirmation_is_current
    prepared = getattr(owner, '_drr_bn_confirmation', None)
    plan = draft.bn_plan
    selected = tuple(owner.drr_selected_files)
    if (plan is None or getattr(prepared, 'plan', None) != plan
            or tuple(plan.selected) != selected
            or getattr(owner, '_drr_background_intent_selection', None) != selected
            or str(Path(plan.root).resolve()).casefold() != _root_key(owner)):
        return False
    return confirmation_is_current(prepared)


def follow_source_type(owner, *, selection_changed=False, restore_saved=True):
    """Handle an explicit type without invoking the legacy cross-type resolver."""
    draft = source_intent(owner)
    if draft is None or owner.drr_baseline_combo.currentText() != 'External':
        return False
    if owner.drr_pin_baseline_chk.isChecked() and draft.bn_plan is None:
        return True
    if restore_saved:
        selection_changed = (selection_changed and tuple(owner.drr_selected_files)
                             != getattr(owner, '_drr_background_intent_selection', None))
    if selection_changed and restore_saved:
        from ui_qt.drr_saved_background import begin_saved_group_restore
        if begin_saved_group_restore(owner, draft):
            return True
    if not selection_changed and draft.bn_plan is None:
        return True
    if not selection_changed and draft.kind == 'bn' and draft.bn_plan is not None:
        if _bn_confirmation_ready(owner, draft):
            return True
    # Preserve the source type, but do not reuse a common background for a new
    # group unless the user explicitly chose Fixed current background.
    owner._drr_assignments = ()
    owner._drr_assignments_automatic = False
    owner.drr_baseline_files_manual = []
    owner.drr_baseline_files_found = []
    owner._drr_background_regions = ()
    owner._drr_baseline_user_selected = True
    owner._drr_external_auto_selected = False
    owner._invalidate_drr_for_background_selection(
        f'{KIND_LABELS[draft.kind]}: open Background settings to review this group. The source type has been retained.')
    owner.drr_controller._update_drr_selection_labels()
    if draft.kind == 'bn' and owner.drr_selected_files:
        request = getattr(owner, '_drr_external_request', None)
        if not isinstance(request, _FollowBnRequest) or request.selected != tuple(owner.drr_selected_files):
            owner.drr_controller._cancel_auto_external()
            request = _FollowBnRequest(owner, draft)
            owner._drr_external_request = request
            owner.thread_pool.start(request.worker)
    return True


def background_ready_for_load(owner):
    """A fast signature guard prevents loading a changed automatic BN recipe."""
    draft = source_intent(owner)
    if draft is None or draft.bn_plan is None or owner.drr_baseline_combo.currentText() != 'External':
        return True
    if _bn_confirmation_ready(owner, draft):
        return True
    follow_source_type(owner, selection_changed=True)
    return False
