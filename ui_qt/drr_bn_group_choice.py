"""Current-group BN overrides, loaded and transformed before publication."""
from dataclasses import dataclass, replace
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Slot

from core.drr_bn_matching import BnMatchCache, confirmation_is_current
from core.drr_measured_bn import is_measured_bn_name
from core.drr_sources import DrrMeasurementAssignment
from core.load_cancellation import LoadCancellation, check_load_cancelled
from ui_qt.common import LoadOptions, Worker
from ui_qt.drr_bn_prepared_cache import GroupBnPreparedCache, frozen_key


AUTO = 'auto'


@dataclass(frozen=True)
class GroupBnState:
    root: str
    selected: tuple[str, ...]
    automatic_plan: object
    automatic_draft: object
    candidates: tuple[str, ...]
    choice: str = AUTO
    applied_intent: object = None


@dataclass(frozen=True)
class PreparedGroupBnChoice:
    choice: str
    draft: object
    confirmation: object
    loaded: object
    derivatives: dict


def _scope(owner):
    return str(Path(owner.current_folder or '.').resolve()).casefold(), tuple(owner.drr_selected_files)


def _intent(owner):
    from ui_qt.drr_background_settings_workflow import source_intent
    return source_intent(owner)


def _transform_specs(owner):
    window = int(owner.drr_sg_window_spin.value())
    poly = int(owner.drr_sg_poly_spin.value())
    specs = [(None, window, poly), (1, window, poly), (2, window, poly)]
    kind = getattr(owner, 'drr_second_kind_combo', None)
    if kind is not None and kind.currentData() == 'mixed':
        specs.append((('mixed', int(owner.drr_sg_y_window_spin.value())), window, poly))
    return tuple(specs)


def _processing_controls(owner):
    flags = tuple((name, getattr(owner, name).isChecked()) for name in
                  ('drr_sg_auto_chk', 'drr_sg_y_auto_chk') if hasattr(owner, name))
    kind = getattr(owner, 'drr_second_kind_combo', None)
    y_window = getattr(owner, 'drr_sg_y_window_spin', None)
    return flags, kind.currentData() if kind is not None else None, y_window.value() if y_window is not None else None


def _context(owner):
    from ui_qt.drr_background_settings_workflow import _context as recipe_context
    tabs = getattr(owner, 'tabs', None)
    active = tabs.tabText(tabs.currentIndex()) if tabs is not None else 'DRR'
    return (_scope(owner), recipe_context(owner), id(getattr(owner, 'loaded', None)),
            active, _processing_controls(owner), _transform_specs(owner))


def _eligible_state(owner):
    intent, scope = _intent(owner), _scope(owner)
    loaded = getattr(owner, 'loaded', None)
    if (intent is None or intent.kind != 'bn' or owner.drr_baseline_combo.currentText() != 'External'
            or loaded is None or loaded.mode != 'DRR'
            or tuple(loaded.selected_files) != scope[1]
            or str(Path(loaded.folder).resolve()).casefold() != scope[0]):
        return None
    existing = getattr(owner, '_drr_bn_group_state', None)
    if existing is not None and (existing.root, existing.selected) == scope and intent is existing.applied_intent:
        expected = (existing.automatic_plan.resolution.assignments if existing.choice == AUTO else
                    tuple(DrrMeasurementAssignment(name, 'External', (existing.choice,), 'all',
                        selection_reason='explicit user baseline selection')
                        for name in existing.selected))
        if tuple(getattr(loaded, 'drr_assignments', ())) == tuple(expected):
            return existing
    plan = getattr(intent, 'bn_plan', None)
    if (plan is None or not plan.ready or tuple(plan.selected) != scope[1]
            or str(Path(plan.root).resolve()).casefold() != scope[0]
            or tuple(getattr(loaded, 'drr_assignments', ())) != tuple(plan.resolution.assignments)):
        return None
    candidates = tuple(dict.fromkeys(name for assignment in plan.resolution.assignments
                                     for name in assignment.baseline_files))
    shared = set(candidates)
    for batch in plan.batches:
        shared.intersection_update(batch.candidate_files)
    candidates = tuple(name for name in candidates if name in shared and is_measured_bn_name(name))
    if not candidates:
        return None
    return GroupBnState(scope[0], scope[1], plan, intent, candidates, applied_intent=intent)


def refresh_group_choices(owner):
    """Read only applied in-memory metadata; never discover or inspect files."""
    if getattr(owner, '_drr_bn_group_publishing', False):
        return
    row, combo = getattr(owner, 'drr_bn_group_row', None), getattr(owner, 'drr_bn_group_combo', None)
    if row is None or combo is None:
        return
    state = _eligible_state(owner)
    previous = getattr(owner, '_drr_bn_group_state', None)
    if state is None:
        if previous is not None:
            cancel_group_choice(owner, clear_scope=True)
        row.setVisible(False)
        return
    if previous is not None and (previous.root, previous.selected) != (state.root, state.selected):
        cancel_group_choice(owner, clear_scope=True)
    owner._drr_bn_group_state = state
    cache = getattr(owner, '_drr_bn_prepared_cache', None)
    if cache is not None:
        cache.activate(_cache_scope(owner, state))
    blocked = combo.blockSignals(True)
    try:
        combo.clear()
        mapped = tuple(dict.fromkeys(name for assignment in state.automatic_plan.resolution.assignments
                                     for name in assignment.baseline_files))
        counts = [sum(name in assignment.baseline_files for assignment in state.automatic_plan.resolution.assignments)
                  for name in mapped]
        combo.addItem('Auto ('+' + '.join(map(str, counts))+')', AUTO)
        combo.setItemData(0, 'Restore the unchanged automatic BN mapping for this selected group.', Qt.ToolTipRole)
        labels = {}
        for name in state.candidates:
            stem = Path(name).stem
            suffix = stem.rsplit('_', 1)[-1]
            short = 'BN'+suffix if suffix.isdigit() else 'BN (unsuffixed)'
            labels[name] = 'All use '+short
        for name in state.candidates:
            label = labels[name]
            if sum(value == label for value in labels.values()) > 1:
                # Same numbered reference from different acquisitions must
                # remain distinguishable without relying on hover text.
                label = 'All use '+Path(name).name
                if sum(Path(other).name == Path(name).name for other in state.candidates) > 1:
                    label = 'All use '+name
            combo.addItem(label, name)
            combo.setItemData(combo.count()-1, Path(name).name+'\n'+name+'\nUse all frames for every selected measurement.', Qt.ToolTipRole)
        request = getattr(owner, '_drr_bn_group_request', None)
        pending = request is not None and request.current()
        index = combo.findData(request.choice if pending else state.choice)
        combo.setCurrentIndex(max(0, index))
        combo.setToolTip(('Calculating this choice; the current plot is unchanged until calculation succeeds.\n' if pending else '')+
                         'Current group only. Changing the measurement group returns to automatic BN matching.\n'
                         'All selected measurements remain equally weighted. Axes, color limits and SG settings stay unchanged for comparison.')
        combo.view().setTextElideMode(Qt.ElideNone)
        text_width = max((combo.fontMetrics().horizontalAdvance(combo.itemText(i)) for i in range(combo.count())), default=0)
        combo.view().setMinimumWidth(max(combo.width(), text_width+32))
        combo.setEnabled(not getattr(owner, '_load_in_progress', False))
        row.setVisible(True)
    finally:
        combo.blockSignals(blocked)


def cancel_group_choice(owner, *, clear_scope=False):
    request = getattr(owner, '_drr_bn_group_request', None)
    if request is not None:
        request.cancellation.cancel()
    owner._drr_bn_group_request = None
    if clear_scope:
        owner._drr_bn_group_state = None
        cache = getattr(owner, '_drr_bn_prepared_cache', None)
        if cache is not None:
            cache.clear()


def _options(owner, state, choice):
    plan = state.automatic_plan
    if choice == AUTO:
        assignments = plan.resolution.assignments
        automatic = True
    else:
        if choice not in state.candidates:
            raise ValueError('Choose a BN already validated for every measurement in this group.')
        assignments = tuple(DrrMeasurementAssignment(name, 'External', (choice,), 'all',
            selection_reason='explicit user baseline selection')
            for name in state.selected)
        automatic = False
    baselines = list(dict.fromkeys(name for assignment in assignments for name in assignment.baseline_files))
    return LoadOptions(mode='DRR', folder=plan.root, selected_files=list(state.selected), baseline_files=baselines,
        pl_log_scale=False, drr_baseline_text='Automatic' if automatic else 'External',
        drr_baseline_which='all', compare_log_scale=False, y_axis_spec=owner._selected_y_axis_spec('drr'),
        drr_restore_files=False, drr_restore_centers=False, drr_resolve_background=False,
        drr_assignments=tuple(assignments), drr_background_selection={
            'selection_method': 'automatic_per_measurement' if automatic else 'explicit_per_measurement',
            'bn_group_choice': choice, 'bn_group_override_scope': 'current_ordered_measurement_selection',
            'drr_background_assignments': [assignment.to_dict() for assignment in assignments]})


def _cache_scope(owner, state):
    return (state.root, state.selected, state.automatic_plan.signature,
            frozen_key(_options(owner, state, AUTO)), _transform_specs(owner), _processing_controls(owner),
            state.automatic_draft.auto_bn_compare, state.automatic_draft.bn_candidates,
            state.automatic_draft.bn_warnings_acknowledged)


def _validate_loaded(loaded, state, options):
    if (loaded.mode != 'DRR' or tuple(loaded.selected_files) != state.selected or loaded.cube is None
            or str(Path(loaded.folder).resolve()).casefold() != state.root
            or tuple(loaded.drr_assignments) != tuple(options.drr_assignments)):
        raise ValueError('The prepared DRR result does not preserve the selected group and BN assignments.')


def prepare_group_choice(state, choice, options, transform_specs, load, *, progress, log,
                         prepared_cache=None, cache_token=None, cache_key=None):
    """Verify, load and calculate both DRR products before touching the owner."""
    from ui_qt.drr_display_compute import compute
    check_load_cancelled()
    if choice != AUTO and choice not in state.candidates:
        raise ValueError('The requested BN is not available for this automatic mapping.')
    if prepared_cache is not None:
        cached = prepared_cache.get(cache_token, cache_key)
        if cached is not None:
            _validate_loaded(cached.loaded, state, options)
            check_load_cancelled()
            if log is not None:
                log.emit('Reusing validated group BN data and derivatives.')
            return cached
    confirmation = BnMatchCache().prepare_confirm(state.automatic_plan,
        warnings_acknowledged=state.automatic_draft.bn_warnings_acknowledged)
    loaded = load(options, progress=progress, log=log)
    check_load_cancelled()
    _validate_loaded(loaded, state, options)
    derivatives = {}
    for derivative, window, poly in transform_specs:
        check_load_cancelled()
        key = (id(loaded.cube), derivative, window, poly)
        _, returned_key, cube, used = compute(loaded.cube, key, progress=progress, log=log)
        derivatives[returned_key] = (cube, int(used))
    check_load_cancelled()
    if not confirmation_is_current(confirmation):
        raise ValueError('BN inputs changed while calculating the new DRR view. The previous background is retained.')
    draft = replace(state.automatic_draft, bn_plan=state.automatic_plan if choice == AUTO else None,
        files=tuple(options.baseline_files), which='all', fixed=False, regions=(), calibration=None, version=None)
    result = PreparedGroupBnChoice(choice, draft, confirmation, loaded, derivatives)
    if prepared_cache is not None:
        prepared_cache.put(cache_token, cache_key, result)
    check_load_cancelled()
    return result


class _GroupChoiceRequest(QObject):
    def __init__(self, owner, state, choice):
        super().__init__(owner if isinstance(owner, QObject) else None)
        self.owner, self.state, self.choice = owner, state, choice
        self.context = _context(owner)
        self.cancellation = LoadCancellation()
        cache = getattr(owner, '_drr_bn_prepared_cache', None)
        if cache is None:
            cache = owner._drr_bn_prepared_cache = GroupBnPreparedCache()
        token = cache.activate(_cache_scope(owner, state))
        options = _options(owner, state, choice)
        self.worker = Worker(prepare_group_choice, state, choice, options,
            self.context[-1], owner._load_task_data, cancellation=self.cancellation,
            prepared_cache=cache, cache_token=token, cache_key=(choice, frozen_key(options), self.context[-1]))
        self.worker.signals.result.connect(self.complete, Qt.QueuedConnection)
        self.worker.signals.error.connect(self.failed, Qt.QueuedConnection)
        self.worker.signals.finished.connect(self.finished, Qt.QueuedConnection)

    def current(self):
        return (not self.cancellation.cancelled and not getattr(self.owner, '_is_closing', False)
                and getattr(self.owner, '_drr_bn_group_request', None) is self
                and self.context[3] == 'DRR'
                and _context(self.owner) == self.context)

    @Slot(object)
    def complete(self, prepared):
        if not self.current():
            return
        owner = self.owner
        if not confirmation_is_current(prepared.confirmation):
            cache = getattr(owner, '_drr_bn_prepared_cache', None)
            if cache is not None:
                cache.clear()
            self.failed('BN inputs changed before the new view could be applied.')
            return
        from ui_qt.drr_background_settings_workflow import _commit
        attributes = (
            'loaded', '_drr_assignments', '_drr_assignments_automatic', '_drr_background_regions',
            'drr_baseline_files_manual', 'drr_baseline_files_found', '_drr_baseline_user_selected',
            '_drr_external_auto_selected', '_drr_background_guess', '_drr_pending_bn_plan',
            '_drr_background_intent', '_drr_background_intent_root', '_drr_background_intent_selection',
            '_drr_bn_confirmation', '_drr_bn_group_state')
        absent = object()
        snapshot = {name: getattr(owner, name, absent) for name in attributes}
        widgets = ((owner.drr_baseline_combo, owner.drr_baseline_combo.currentText()),
                   (owner.drr_baseline_combine_combo, owner.drr_baseline_combine_combo.currentText()),
                   (owner.drr_pin_baseline_chk, owner.drr_pin_baseline_chk.isChecked()),
                   (owner.drr_bn_auto_chk, owner.drr_bn_auto_chk.isChecked()))
        old_derivatives = dict(getattr(owner, '_drr_derivative_cache', {}))
        display_snapshot = None
        if hasattr(owner, 'drr_spins'):
            from ui_qt.drr_display_memory import prepare_background_switch
            display_snapshot = prepare_background_switch(owner)
        owner._drr_bn_group_publishing = True
        try:
            _commit(owner, prepared.draft, prepared.confirmation, start_load=False, persist=False)
            owner._drr_bn_group_state = replace(self.state, choice=prepared.choice, applied_intent=_intent(owner))
            owner._drr_bn_group_request = None
            owner._on_loaded(prepared.loaded, preserve_drr_display=True,
                             drr_derivatives=prepared.derivatives, drr_display_snapshot=display_snapshot)
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
            # Publication is synchronous. Never roll a previous group's state
            # over a newly selected group if a callback changes the context.
            if _scope(owner) == (self.state.root, self.state.selected):
                from ui_qt.drr_background_settings_workflow import _silent
                for name, value in snapshot.items():
                    if value is absent:
                        if hasattr(owner, name):
                            delattr(owner, name)
                    else:
                        setattr(owner, name, value)
                for widget, value in widgets:
                    _silent(widget, value)
                try:
                    owner._on_loaded(snapshot['loaded'], preserve_drr_display=True,
                                     drr_derivatives=old_derivatives, drr_display_snapshot=display_snapshot)
                except (ValueError, OSError, KeyError, TypeError, RuntimeError):
                    owner.loaded = snapshot['loaded']
                owner._status('Group BN was not changed. '+str(exc))
            owner._drr_bn_group_request = None
        finally:
            owner._drr_bn_group_publishing = False
            refresh_group_choices(owner)

    @Slot(str)
    def failed(self, message):
        if self.current():
            self.owner._drr_bn_group_request = None
            self.owner._status('Group BN was not changed. '+message.split('\n\n', 1)[0])
            refresh_group_choices(self.owner)

    @Slot()
    def finished(self):
        if getattr(self.owner, '_drr_bn_group_request', None) is self:
            self.owner._drr_bn_group_request = None
            refresh_group_choices(self.owner)
        self.deleteLater()


def request_group_choice(owner, choice=None):
    """Latest request wins; a failed worker leaves the applied recipe intact."""
    refresh_group_choices(owner)
    state = getattr(owner, '_drr_bn_group_state', None)
    choice = AUTO if choice in (None, AUTO) else str(choice)
    if state is None or getattr(owner, '_load_in_progress', False):
        refresh_group_choices(owner)
        return
    if _context(owner)[3] != 'DRR':
        return
    cancel_group_choice(owner)
    if choice == state.choice:
        refresh_group_choices(owner)
        return
    try:
        request = _GroupChoiceRequest(owner, state, choice)
    except (ValueError, TypeError) as exc:
        owner._status('Group BN was not changed. '+str(exc))
        refresh_group_choices(owner)
        return
    owner._drr_bn_group_request = request
    refresh_group_choices(owner)
    owner._status('Calculating the selected group BN; the current DRR view remains available…')
    owner.thread_pool.start(request.worker)
