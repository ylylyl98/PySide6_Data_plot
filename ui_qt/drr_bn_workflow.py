"""Per-group BN comparison, with guarded background work and explicit rollback."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

from PySide6.QtCore import QObject, Qt, Slot

from core.drr_bn_store import DrrBnStore, joint_validation_passed, optimization_policy_current
from core.drr_sources import DrrMeasurementAssignment, portable_source_name, resolve_source_path
from core.load_cancellation import LoadCancellation, check_load_cancelled
from ui_qt.common import Worker
from ui_qt.drr_pn_workflow import _context as _pn_context
from ui_qt.drr_region_backgrounds import frame_method


def _context(owner):
    # Joint validation is specific to the gate axis and production D2E settings.
    return (*_pn_context(owner), str(owner._selected_y_axis_spec('drr')), _derivative_settings(owner))


def _derivative_settings(owner):
    return (int(owner.drr_sg_window_spin.value()), int(owner.drr_sg_poly_spin.value()))


def _payload_derivative_settings(payload):
    report = payload.get('report', {}).get('derivative_settings', {})
    settings = payload.get('settings', {})
    return (report.get('window_length', settings.get('derivative_window')),
            report.get('polyorder', settings.get('derivative_polyorder')))


def _comparison_settings(owner, payload=None):
    settings = deepcopy(payload.get('settings', {}) if payload else {})
    settings['derivative_window'], settings['derivative_polyorder'] = _derivative_settings(owner)
    return settings


def _comparison_summary(payload, default):
    from ui_qt.drr_bn_calibration_dialog import _readable_text
    return _readable_text(payload.get('summary') or default)


def managed_background(owner):
    return any('.drr_bn_calibration' in Path(name).parts
               for name in getattr(owner, 'drr_baseline_files_manual', ()))


def _group_key(owner):
    return (str(owner.current_folder), tuple(sorted(owner.drr_selected_files)), _derivative_settings(owner))


def _recipe(owner):
    files = tuple(owner.drr_baseline_files_manual)
    which = frame_method(owner)
    assignments = tuple(getattr(owner, '_drr_assignments', ()))
    if assignments:
        recipes = {(x.baseline_mode, x.baseline_files, x.baseline_which, x.regions) for x in assignments}
        if len(recipes) != 1 or assignments[0].baseline_mode != 'External':
            raise ValueError('Choose one common BN reference for this group before comparing.')
        files, which, regions = assignments[0].baseline_files, assignments[0].baseline_which, assignments[0].regions
        if regions:
            raise ValueError('BN calibration currently requires one reference without X-range overrides.')
    if getattr(owner, '_drr_background_regions', ()):
        raise ValueError('BN calibration currently requires one reference without X-range overrides.')
    if not files or not owner.drr_selected_files:
        raise ValueError('Select a measurement group and its BN background first.')
    if owner.drr_baseline_combo.currentText() != 'External':
        raise ValueError('Choose External background mode before comparing BN references.')
    if any(Path(x).suffix.lower() != '.csv' for x in (*files, *owner.drr_selected_files)):
        raise ValueError('BN calibration requires raw measurement and BN CSV files.')
    bn_name = re.compile(r'(?<![a-z0-9])bnback(?:ground)?(?=$|[^a-z])', re.I)
    if any('.drr_bn_calibration' not in Path(name).parts and not bn_name.search(Path(name).stem)
           for name in files):
        raise ValueError('Choose measured BNback/BNbackground references before comparing BN calibration.')
    return tuple(files), which


def _paths(folder, names):
    return tuple(sorted(str(resolve_source_path(folder, x)).casefold() for x in names))


def _matches(owner, payload, *, check_axis=True, check_derivative=True):
    """Cheap recipe check only; worker/store subsequently verify actual content."""
    if not isinstance(payload, dict):
        return False
    source = payload.get('source_identity', {})
    try:
        folder = owner.current_folder
        if Path(source.get('root', '')).resolve() != Path(folder).resolve():
            return False
        if check_axis and source.get('y_axis') != str(owner._selected_y_axis_spec('drr')):
            return False
        if check_derivative and _payload_derivative_settings(payload) != _derivative_settings(owner):
            return False
        if _paths(folder, owner.drr_selected_files) != _paths(folder, [v['path'] for v in source['measurements']]):
            return False
        files, which = _recipe(owner)
        if _paths(folder, files) == _paths(folder, [v['path'] for v in source['baseline_files']]):
            return which == source['baseline_which']
        if (len(files) == 1 and payload.get('csv_path')
                and _paths(folder, files) == _paths(folder, [payload['csv_path']])):
            return True
        # A request against managed v1 may produce v2 from v1's verified raw
        # reference. This binding is attached after persistence by that request,
        # and accepts only the exact input recipe, never any managed CSV.
        compared = payload.get('comparison_input_backgrounds')
        return bool(compared and _paths(folder, files) == _paths(folder, compared)
                    and which == payload.get('comparison_input_which'))
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _remember(owner, payload):
    cache = getattr(owner, '_drr_bn_payloads', {})
    cache[_group_key(owner)] = payload
    while len(cache) > 12:
        del cache[next(iter(cache))]
    owner._drr_bn_payloads = cache


def _cached_payload(owner):
    payload = getattr(owner, '_drr_bn_payloads', {}).get(_group_key(owner))
    return payload if _matches(owner, payload) else None


def managed_joint_validation_current(owner):
    """The reference actually in use, rather than a replacement preview, passed."""
    managed = managed_background(owner)
    loaded = getattr(owner, 'loaded', None)
    loaded_files = tuple(getattr(loaded, 'baseline_files', ()))
    loaded_managed = (getattr(loaded, 'mode', None) == 'DRR'
                      and any('.drr_bn_calibration' in Path(name).parts for name in loaded_files))
    if not managed:
        # Switching the recipe starts an asynchronous reload. Until it arrives,
        # the previous processed cube must retain its derivative guard.
        if loaded_managed:
            return False
        return True
    payload = _cached_payload(owner)
    if not payload or not payload.get('eligible') or not joint_validation_passed(payload.get('report', {})):
        return False
    try:
        files, _which = _recipe(owner)
        if getattr(loaded, 'mode', None) == 'DRR' and hasattr(loaded, 'baseline_files'):
            loaded_folder = getattr(loaded, 'folder', owner.current_folder)
            if (_paths(loaded_folder, loaded_files) != _paths(owner.current_folder, files)
                    or Path(loaded_folder).resolve() != Path(owner.current_folder).resolve()
                    or tuple(loaded.selected_files) != tuple(owner.drr_selected_files)
                    or (hasattr(loaded, 'y_axis_spec')
                        and str(loaded.y_axis_spec) != str(owner._selected_y_axis_spec('drr')))):
                return False
        return (len(files) == 1 and payload.get('csv_path')
                and _paths(owner.current_folder, files) == _paths(owner.current_folder, (payload['csv_path'],)))
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _fallback_unverified_managed(owner, payload, dialog=None):
    """Withdraw a rejected managed choice without recording an explicit original choice."""
    if not managed_background(owner) or not _matches(owner, payload, check_axis=False, check_derivative=False):
        return False
    source = payload['source_identity']
    files = tuple(item['path'] for item in source['baseline_files'])
    try:
        DrrBnStore(owner.current_folder).clear_activation(tuple(owner.drr_selected_files))
        _select_recipe(owner, files, source['baseline_which'], calibrated=False)
        _remember(owner, payload)
        message = 'Original BN restored: the processed reference did not pass DRR + D2E at the current SG settings.'
        if dialog is not None:
            dialog.set_application_state('original', message)
            dialog._bn_workflow_context = _context(owner)
        owner._status(message)
        return True
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        message = 'Could not restore original BN after joint validation failed: ' + str(exc)
        if dialog is not None:
            dialog.set_error(message)
        owner._status(message)
        return False


def derivative_settings_changed(owner):
    """Invalidate joint confirmation after both explicit and blocked SG updates."""
    settings = _derivative_settings(owner)
    if settings == getattr(owner, '_drr_bn_derivative_settings', None):
        return
    owner._drr_bn_derivative_settings = settings
    cancel_bn_requests(owner)
    owner._drr_bn_auto_checked_load = None
    dialog = getattr(owner, '_drr_bn_dialog', None)
    if dialog is not None:
        dialog.set_derivative_settings(*settings)
        if getattr(dialog, '_bn_workflow_context', ())[:-1] == _context(owner)[:-1]:
            dialog._bn_workflow_context = _context(owner)
    update_bn_controls(owner)
    maybe_auto_compare(owner)


def cancel_bn_requests(owner):
    if getattr(owner, '_drr_bn_certificate_request', None) is not None:
        owner._drr_bn_certificate_checked_context = None
    for name in ('_drr_bn_compare_request', '_drr_bn_apply_request', '_drr_bn_open_request',
                 '_drr_bn_certificate_request'):
        request = getattr(owner, name, None)
        if request is not None:
            request.cancellation.cancel()
            dialog = getattr(request, 'dialog', None)
            if dialog is not None:
                try:
                    dialog.set_error('BN application cancelled. The current background is unchanged.')
                except RuntimeError:
                    pass  # A destroyed dialog cannot receive queued UI updates.
        setattr(owner, name, None)
    dialog = getattr(owner, '_drr_bn_dialog', None)
    if dialog is not None:
        dialog.cancel_comparison()
        if getattr(owner, '_is_closing', False):
            dialog.close()


def forget_bn_choice(owner):
    """A later explicit recipe choice takes precedence over a saved BN choice."""
    if not owner.current_folder or not owner.drr_selected_files:
        return
    if not (Path(owner.current_folder) / '.drr_bn_calibration').is_dir():
        return
    store = DrrBnStore(owner.current_folder)
    # This is a small manifest-only mutation; no spectra are read on the GUI thread.
    try:
        store.deactivate(tuple(owner.drr_selected_files),
                         baseline_files=tuple(owner.drr_baseline_files_manual), which=frame_method(owner))
    except (OSError, ValueError) as exc:
        owner._status('Saved BN choice could not be cleared: ' + str(exc))
    getattr(owner, '_drr_bn_payloads', {}).pop(_group_key(owner), None)


def update_bn_controls(owner):
    button = getattr(owner, 'drr_bn_calibration_btn', None)
    if button is None:
        return
    usable = bool(owner.current_folder and owner.drr_selected_files
                  and owner.drr_baseline_combo.currentText() == 'External')
    try:
        if usable:
            _recipe(owner)
    except ValueError:
        usable = False
    pending = any(getattr(owner, name, None) is not None for name in
                  ('_drr_external_request', '_drr_bn_apply_request', '_drr_bn_open_request',
                   '_drr_bn_certificate_request'))
    button.setEnabled(usable and not pending)
    owner.drr_bn_auto_chk.setEnabled(usable and not pending)
    scoped = managed_background(owner)
    if scoped:
        blocked = owner.drr_pin_baseline_chk.blockSignals(True)
        owner.drr_pin_baseline_chk.setChecked(False)
        owner.drr_pin_baseline_chk.blockSignals(blocked)
        owner.drr_pin_baseline_chk.setEnabled(False)
        owner.drr_baseline_combine_combo.setEnabled(False)
    payload = _cached_payload(owner) if usable else None
    if scoped:
        text = ('BN: DRR + D2E verified for this exact group and SG settings'
                if managed_joint_validation_current(owner) else
                'BN: processed for this exact group · DRR + D2E require comparison at current SG settings')
    elif payload:
        text = 'BN: original · ' + _comparison_summary(payload, 'Comparison available')
    else:
        text = 'Compare the selected BN for this group.'
    owner.drr_bn_status.setText(text)
    owner.drr_bn_status.setToolTip(text)


def _compare_and_save(folder, measurements, files, which, settings, y_axis,
                      *, progress, log, cancellation=None):
    from core.drr_bn_service import compare_bn_group
    settings = deepcopy(settings or {})
    settings.pop('auto_apply', None)
    if not set(settings) - {'derivative_window', 'derivative_polyorder'}:
        store = DrrBnStore(folder)
        previous = (store.version_for_csv(files[0])
                    if len(files) == 1 and '.drr_bn_calibration' in Path(files[0]).parts else
                    store.find_current(measurements, files, which))
        if previous is not None and previous.fresh:
            settings = {**previous.settings, **settings}
    emit_progress = getattr(progress, 'emit', progress)
    emit_log = getattr(log, 'emit', log)
    payload = compare_bn_group(folder, measurements, files, which, settings,
                              y_axis=y_axis, progress=emit_progress, log=emit_log,
                              cancellation=cancellation)
    check_load_cancelled()
    if cancellation is not None and hasattr(cancellation, 'is_set') and cancellation.is_set():
        from core.load_cancellation import LoadCancelled
        raise LoadCancelled()
    store = DrrBnStore(folder)
    version = store.get(payload['version_id']) if payload.get('version_id') else store.save_result(payload)
    payload['version_id'] = version.version_id
    payload['csv_path'] = str(version.csv_path)
    source = payload['source_identity']
    payload['original_chosen'] = store.has_original_choice(
        measurements, [v['path'] for v in source['baseline_files']], source['baseline_which'])
    payload['comparison_input_backgrounds'] = tuple(files)
    payload['comparison_input_which'] = which
    return payload


def _verify_version(folder, version_id, derivative_settings, *, progress, log):
    check_load_cancelled()
    version = DrrBnStore(folder).get(version_id)
    if version is None or not version.fresh or not version.eligible:
        raise ValueError('This BN candidate is stale or has not passed validation. Keep the original BN and compare again.')
    if (version.settings.get('derivative_window'), version.settings.get('derivative_polyorder')) != derivative_settings:
        raise ValueError('This BN candidate used different SG settings. Compare DRR and D2E again before applying it.')
    check_load_cancelled()
    return version


def _select_recipe(owner, files, which, *, calibrated):
    owner.drr_controller._cancel_auto_external()
    owner._drr_background_regions = ()
    owner._drr_assignments = tuple(DrrMeasurementAssignment(
        name, 'External', tuple(files), which,
        selection_reason='Validated group BN calibration' if calibrated else 'Original BN restored')
        for name in owner.drr_selected_files)
    owner._drr_assignments_automatic = False
    owner._drr_baseline_user_selected = True
    owner._drr_external_auto_selected = False
    owner._drr_background_guess = None
    owner.drr_baseline_files_manual = list(files)
    owner.drr_baseline_files_found = []
    label = {'first': 'First frame from each file, then average',
             'last': 'Last frame from each file, then average',
             'all': 'Average all frames in each file, then average files'}[which]
    for widget, value in ((owner.drr_baseline_combo, 'External'), (owner.drr_baseline_combine_combo, label)):
        previous = widget.blockSignals(True)
        widget.setCurrentText(value)
        widget.blockSignals(previous)
    previous = owner.drr_pin_baseline_chk.blockSignals(True)
    owner.drr_pin_baseline_chk.setChecked(False)
    owner.drr_pin_baseline_chk.blockSignals(previous)
    # Automatic comparison and original-BN restoration use this adapter too.
    # Keep the source-first settings in step with the recipe actually applied;
    # an earlier matching plan must not reassert its raw BN after calibration.
    from ui_qt.drr_background_settings import BackgroundDraft
    from ui_qt.drr_background_settings_workflow import source_intent, _root_key
    intent = source_intent(owner)
    payload = _cached_payload(owner)
    calibration = ({'payload': payload, 'original': not calibrated}
                   if payload is not None and (calibrated or payload.get('original_chosen')) else None)
    owner._drr_background_intent = BackgroundDraft(
        kind='bn', files=tuple(files), which=which, calibration=calibration,
        auto_bn_compare=owner.drr_bn_auto_chk.isChecked(),
        bn_candidates=intent.bn_candidates if intent is not None else None,
        fitted_result=dict(getattr(owner, '_drr_fitted_settings', {})))
    owner._drr_background_intent_root = _root_key(owner)
    owner._drr_background_intent_selection = tuple(owner.drr_selected_files)
    owner._drr_bn_confirmation = None
    owner._drr_pending_bn_plan = None
    owner.drr_controller._update_drr_selection_labels()
    owner._start_load('DRR')


class _ApplyRequest(QObject):
    def __init__(self, owner, payload, dialog=None):
        super().__init__(owner if isinstance(owner, QObject) else None)
        self.owner, self.payload, self.dialog = owner, payload, dialog
        self.context = _context(owner)
        self.cancellation = LoadCancellation()
        self.worker = Worker(_verify_version, self.context[0], payload['version_id'], self.context[-1],
                             cancellation=self.cancellation)
        self.worker.signals.result.connect(self.complete, Qt.QueuedConnection)
        self.worker.signals.error.connect(self.failed, Qt.QueuedConnection)
        self.worker.signals.finished.connect(self.finished, Qt.QueuedConnection)

    def current(self):
        return (not self.cancellation.cancelled and not getattr(self.owner, '_is_closing', False)
                and getattr(self.owner, '_drr_bn_apply_request', None) is self
                and _context(self.owner) == self.context)

    @Slot(object)
    def complete(self, version):
        if not self.current():
            return
        owner = self.owner
        try:
            # The current-context check and small manifest commit occur in one GUI turn.
            DrrBnStore(self.context[0]).activate_verified(version, measurements=self.context[1])
            name = portable_source_name(self.context[0], version.csv_path)
            _remember(owner, self.payload)
            owner._drr_bn_apply_request = None
            _select_recipe(owner, (name,), 'last', calibrated=True)
            if self.dialog is not None:
                self.dialog.set_application_state('calibrated', 'Validated BN applied to this group. Original BN remains available.')
                self.dialog._bn_workflow_context = _context(owner)
            owner._status('Validated BN applied to this group; the original BN is preserved.')
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            # Selecting the recipe detaches this request before starting the
            # ordinary loader; its errors still need to release the dialog.
            message = 'Could not finish applying/loading the BN candidate: ' + str(exc)
            if self.dialog is not None:
                self.dialog.set_error(message)
            owner._status(message)

    @Slot(str)
    def failed(self, message):
        if self.current():
            message = message.split('\n\n', 1)[0]
            if self.dialog is not None:
                self.dialog.set_error(message)
            self.owner._status('BN was not applied: ' + message)

    @Slot()
    def finished(self):
        if getattr(self.owner, '_drr_bn_apply_request', None) is self:
            if not self.current() and self.dialog is not None:
                self.dialog.set_error('Selection changed. The BN candidate was not applied.')
            self.owner._drr_bn_apply_request = None
        update_bn_controls(self.owner)
        self.deleteLater()


def apply_bn_payload(owner, payload, dialog=None):
    if (not payload.get('eligible') or not joint_validation_passed(payload.get('report', {}))
            or not _matches(owner, payload) or getattr(owner, '_is_closing', False)):
        if dialog is not None:
            dialog.set_error('The comparison is no longer eligible for this selection. Compare the current inputs again.')
        return
    cancel_bn_requests(owner)
    request = _ApplyRequest(owner, payload, dialog)
    owner._drr_bn_apply_request = request
    if dialog is not None:
        dialog.set_application_busy(True)
    owner._status('Verifying the selected BN candidate before application…')
    update_bn_controls(owner)
    owner.thread_pool.start(request.worker)


def restore_original(owner, payload=None, dialog=None):
    payload = payload or _cached_payload(owner)
    if payload is None or not _matches(owner, payload, check_axis=False, check_derivative=False):
        if dialog is not None:
            dialog.set_error('Compare or reload this group first so its original BN can be restored.')
        return
    cancel_bn_requests(owner)
    source = payload['source_identity']
    files = tuple(item['path'] for item in source['baseline_files'])
    try:
        DrrBnStore(owner.current_folder).deactivate(
            tuple(owner.drr_selected_files), files, source['baseline_which'], restore_original=True)
    except (OSError, ValueError) as exc:
        if dialog is not None:
            dialog.set_error('Could not restore the original BN: ' + str(exc))
        owner._status('Original BN was not restored: ' + str(exc))
        return
    payload['original_chosen'] = True
    _select_recipe(owner, files, source['baseline_which'], calibrated=False)
    _remember(owner, payload)
    if dialog is not None:
        dialog.set_application_state('original', 'Original BN restored for this group.')
        dialog._bn_workflow_context = _context(owner)
    owner._status('Original BN restored. Automatic checking will respect this choice.')


class _CompareRequest(QObject):
    def __init__(self, owner, files, which):
        super().__init__(owner if isinstance(owner, QObject) else None)
        self.owner, self.context = owner, _context(owner)
        self.cancellation = LoadCancellation()
        previous = _cached_payload(owner)
        settings = _comparison_settings(owner, previous)
        self.worker = Worker(_compare_and_save, self.context[0], self.context[1], files, which, settings,
                             owner._selected_y_axis_spec('drr'), cancellation=self.cancellation)
        self.worker.signals.result.connect(self.complete, Qt.QueuedConnection)
        self.worker.signals.error.connect(self.failed, Qt.QueuedConnection)
        self.worker.signals.finished.connect(self.finished, Qt.QueuedConnection)

    def current(self):
        return (not self.cancellation.cancelled and not getattr(self.owner, '_is_closing', False)
                and getattr(self.owner, '_drr_bn_compare_request', None) is self
                and _context(self.owner) == self.context)

    @Slot(object)
    def complete(self, payload):
        if not self.current():
            return
        _remember(self.owner, payload)
        update_bn_controls(self.owner)
        self.owner._status('BN comparison: ' + _comparison_summary(payload, 'Original BN retained.'))
        if not payload.get('eligible') or not joint_validation_passed(payload.get('report', {})):
            if managed_background(self.owner):
                self.owner._drr_bn_compare_request = None
                _fallback_unverified_managed(self.owner, payload)
            return
        if managed_background(self.owner) and managed_joint_validation_current(self.owner):
            redraw = getattr(self.owner, '_schedule_plot_redraw', None)
            if callable(redraw):
                redraw('DRR')
            return
        if payload.get('eligible') and not payload.get('original_chosen'):
            apply_bn_payload(self.owner, payload)

    @Slot(str)
    def failed(self, message):
        if self.current():
            self.owner._status('BN comparison unavailable; current background retained. ' + message.split('\n\n', 1)[0])

    @Slot()
    def finished(self):
        if getattr(self.owner, '_drr_bn_compare_request', None) is self:
            self.owner._drr_bn_compare_request = None
        update_bn_controls(self.owner)
        self.deleteLater()


def maybe_auto_compare(owner):
    checkbox = getattr(owner, 'drr_bn_auto_chk', None)
    if (checkbox is None or not checkbox.isChecked() or getattr(owner, '_is_closing', False)
            or getattr(owner, '_drr_bn_dialog', None) is not None
            or getattr(owner, '_drr_settings_dialog', None) is not None
            or getattr(owner, '_drr_bn_apply_request', None) is not None
            or getattr(owner, '_drr_bn_certificate_request', None) is not None
            or getattr(owner, '_drr_bn_compare_request', None) is not None):
        return
    if managed_background(owner) and managed_joint_validation_current(owner):
        return
    loaded = getattr(owner, 'loaded', None)
    if (loaded is None or loaded.mode != 'DRR'
            or str(getattr(loaded, 'folder', owner.current_folder)) != str(owner.current_folder)
            or tuple(loaded.selected_files) != tuple(owner.drr_selected_files)):
        return
    try:
        files, which = _recipe(owner)
    except ValueError:
        return
    key = (id(loaded), _context(owner))
    if key == getattr(owner, '_drr_bn_auto_checked_load', None):
        return
    owner._drr_bn_auto_checked_load = key
    request = _CompareRequest(owner, files, which)
    owner._drr_bn_compare_request = request
    owner._status('Comparing the selected BN for this group…')
    owner.thread_pool.start(request.worker)


def auto_toggled(owner, enabled):
    if not enabled:
        cancel_bn_requests(owner)
    else:
        owner._drr_bn_auto_checked_load = None
        maybe_auto_compare(owner)


def edit_bn_calibration(owner):
    try:
        files, which = _recipe(owner)
    except ValueError as exc:
        owner._show_error(str(exc))
        return
    owner.drr_controller._cancel_auto_external()
    context = _context(owner)
    y_axis = owner._selected_y_axis_spec('drr')
    payload = _cached_payload(owner)
    if payload is not None:
        _show_dialog(owner, context, files, which, y_axis, payload)
        return
    request = _OpenRequest(owner, context, files, which, y_axis)
    owner._drr_bn_open_request = request
    update_bn_controls(owner)
    owner.thread_pool.start(request.worker)


def _load_saved(folder, measurements, files, which, y_axis, derivative_settings=None, *, progress, log):
    store = DrrBnStore(folder)
    if any('.drr_bn_calibration' in Path(name).parts for name in files):
        if len(files) != 1:
            raise ValueError('Choose one calibrated BN version.')
        version = store.version_for_csv(files[0])
        if _paths(folder, measurements) != _paths(folder, [v['path'] for v in version.source_identity['measurements']]):
            raise ValueError('This calibrated BN belongs to another measurement group.')
    else:
        version = store.find_current(measurements, files, which)
    if version is None:
        return None
    payload = store.load_result(version)
    payload['csv_path'] = str(version.csv_path)
    different_axis = version.source_identity.get('y_axis') != str(y_axis)
    different_derivative = (derivative_settings is not None
                            and _payload_derivative_settings(payload) != derivative_settings)
    different_policy = not optimization_policy_current(payload.get('report', {}))
    if not version.fresh or different_axis or different_derivative or different_policy:
        payload['eligible'] = False
        payload['summary'] = ('Saved comparison uses an older optimization policy. '
                              'Compare again with D2E as the target and DRR as a constraint, including the current weak-feature checks.'
                              if different_policy else
                              'Saved comparison used a different gate axis. Compare the current selection again.'
                              if different_axis else
                              'Saved comparison used different SG settings. Compare DRR and D2E at the current settings again.'
                              if different_derivative else 'Saved comparison is stale. Compare the changed inputs again.')
        for candidate in payload.get('candidates', ()):
            candidate['eligible'] = False
    check_load_cancelled()
    return payload


def ensure_managed_bn_validation(owner):
    """Restore a saved current certificate asynchronously, even with Auto off."""
    if not managed_background(owner) or managed_joint_validation_current(owner) or getattr(owner, '_is_closing', False):
        return
    if any(getattr(owner, name, None) is not None for name in
           ('_drr_bn_certificate_request', '_drr_bn_compare_request', '_drr_bn_apply_request', '_drr_bn_open_request')):
        return
    context = _context(owner)
    if context == getattr(owner, '_drr_bn_certificate_checked_context', None):
        return
    try:
        files, which = _recipe(owner)
    except ValueError:
        return
    loaded = getattr(owner, 'loaded', None)
    if (getattr(owner, '_load_in_progress', False) and getattr(loaded, 'mode', None) == 'DRR'
            and hasattr(loaded, 'baseline_files')
            and _paths(getattr(loaded, 'folder', owner.current_folder), loaded.baseline_files)
                != _paths(owner.current_folder, files)):
        return  # Wait for the requested recipe's cube before certifying its use.
    owner._drr_bn_certificate_checked_context = context
    request = _CertificateRequest(owner, context, files, which)
    owner._drr_bn_certificate_request = request
    owner._status('Checking the saved processed BN for the current DRR + D2E settings…')
    update_bn_controls(owner)
    owner.thread_pool.start(request.worker)


class _CertificateRequest(QObject):
    def __init__(self, owner, context, files, which):
        super().__init__(owner if isinstance(owner, QObject) else None)
        self.owner, self.context = owner, context
        self.cancellation = LoadCancellation()
        self.worker = Worker(_load_saved, context[0], context[1], files, which,
                             owner._selected_y_axis_spec('drr'), context[-1], cancellation=self.cancellation)
        self.worker.signals.result.connect(self.complete, Qt.QueuedConnection)
        self.worker.signals.error.connect(self.failed, Qt.QueuedConnection)
        self.worker.signals.finished.connect(self.finished, Qt.QueuedConnection)

    def current(self):
        return (not self.cancellation.cancelled and not getattr(self.owner, '_is_closing', False)
                and getattr(self.owner, '_drr_bn_certificate_request', None) is self
                and _context(self.owner) == self.context)

    @Slot(object)
    def complete(self, payload):
        if not self.current():
            return
        owner = self.owner
        owner._drr_bn_certificate_request = None
        if payload is None:
            owner._status('This processed BN has no current joint report. Choose the original BN or compare again before D2E.')
            return
        _remember(owner, payload)
        if managed_joint_validation_current(owner):
            owner._status('Saved DRR + D2E validation verified for the current processed BN and SG settings.')
            redraw = getattr(owner, '_schedule_plot_redraw', None)
            if callable(redraw):
                redraw('DRR')
        else:
            _fallback_unverified_managed(owner, payload)

    @Slot(str)
    def failed(self, message):
        if self.current():
            self.owner._status('Saved BN validation could not be verified; D2E remains unavailable. '
                               + message.split('\n\n', 1)[0])

    @Slot()
    def finished(self):
        if getattr(self.owner, '_drr_bn_certificate_request', None) is self:
            self.owner._drr_bn_certificate_request = None
        update_bn_controls(self.owner)
        self.deleteLater()


class _OpenRequest(QObject):
    def __init__(self, owner, context, files, which, y_axis):
        super().__init__(owner if isinstance(owner, QObject) else None)
        self.owner, self.context, self.files, self.which, self.y_axis = owner, context, files, which, y_axis
        self.cancellation = LoadCancellation()
        self.worker = Worker(_load_saved, context[0], context[1], files, which, y_axis, context[-1],
                             cancellation=self.cancellation)
        self.worker.signals.result.connect(self.complete, Qt.QueuedConnection)
        self.worker.signals.error.connect(self.failed, Qt.QueuedConnection)
        self.worker.signals.finished.connect(self.finished, Qt.QueuedConnection)

    def current(self):
        return (not self.cancellation.cancelled and not getattr(self.owner, '_is_closing', False)
                and getattr(self.owner, '_drr_bn_open_request', None) is self
                and _context(self.owner) == self.context)

    @Slot(object)
    def complete(self, payload):
        if self.current():
            self.owner._drr_bn_open_request = None
            _show_dialog(self.owner, self.context, self.files, self.which, self.y_axis, payload)

    @Slot(str)
    def failed(self, message):
        if self.current():
            self.owner._show_error('Could not read the saved BN comparison: ' + message.split('\n\n', 1)[0])

    @Slot()
    def finished(self):
        if getattr(self.owner, '_drr_bn_open_request', None) is self:
            self.owner._drr_bn_open_request = None
        update_bn_controls(self.owner)
        self.deleteLater()


def _show_dialog(owner, context, files, which, y_axis, payload):
    from ui_qt.drr_bn_calibration_dialog import DrrBnCalibrationDialog
    if payload:
        _remember(owner, payload)

    def compare(settings, progress, log, cancellation):
        return _compare_and_save(context[0], context[1], files, which, settings, y_axis,
                                 progress=progress, log=log, cancellation=cancellation)

    dialog = DrrBnCalibrationDialog(owner,
        context_label=f'{len(context[1])} measurement file(s) · BN {", ".join(Path(x).name for x in files)}',
        settings=_comparison_settings(owner, payload), run_comparison=compare,
        initial_payload=payload, current_mode='calibrated' if managed_background(owner) else 'original')
    dialog._bn_workflow_context = context
    owner._drr_bn_dialog = dialog

    def apply(result):
        if _context(owner) != dialog._bn_workflow_context:
            dialog.set_error('Selection changed. Reopen BN comparison for the current group.')
            return
        _remember(owner, result)
        apply_bn_payload(owner, result, dialog)

    def original():
        if _context(owner) != dialog._bn_workflow_context:
            dialog.set_error('Selection changed. Reopen BN comparison for the current group.')
            return
        restore_original(owner, getattr(dialog, '_payload', None) or payload, dialog)

    def compared(result):
        if _context(owner) != dialog._bn_workflow_context:
            return
        _remember(owner, result)
        if not result.get('eligible') or not joint_validation_passed(result.get('report', {})):
            _fallback_unverified_managed(owner, result, dialog)
        update_bn_controls(owner)

    dialog.applyRequested.connect(apply)
    dialog.originalRequested.connect(original)
    dialog.comparisonCompleted.connect(compared)
    try:
        dialog.exec()
    finally:
        cancel_bn_requests(owner)
        owner._drr_bn_dialog = None
        final_payload = getattr(dialog, '_payload', None)
        if final_payload and _matches(owner, final_payload):
            _remember(owner, final_payload)
        dialog.deleteLater()
        update_bn_controls(owner)
        owner.drr_bn_calibration_btn.setFocus()
