"""Restore an exact group's exported recipe before following a source type."""
from pathlib import Path
import re

from PySide6.QtCore import QObject, Qt, Slot

from core.load_cancellation import LoadCancellation
from ui_qt.common import Worker


class _Cancellation(LoadCancellation):
    # External requests expose an Event-shaped cancellation handle to the
    # controller; Worker also needs cooperative load cancellation for reads.
    def set(self):
        self.cancel()

    def is_set(self):
        return self.cancelled


def _resolve_saved(folder, catalog, selected, *, progress, log):
    from core.drr_saved_background import resolve_saved_group_background
    return resolve_saved_group_background(folder, catalog, selected,
        progress=getattr(progress, 'emit', progress), log=getattr(log, 'emit', log))


def _kind(assignments):
    """Describe the restored sources without opening files on the GUI thread."""
    from core.drr_measured_bn import is_measured_bn_name
    from core.drr_pn_pairing import detect_pn_role
    if all(item.baseline_mode.startswith('Self') for item in assignments):
        return 'self'
    kinds = set()
    for item in assignments:
        if item.baseline_mode != 'External':
            kinds.add('self')
        for name in item.baseline_source_files:
            parts = str(name).replace('\\', '/').casefold().split('/')
            if '.drr_bn_calibration' in parts or is_measured_bn_name(name):
                kinds.add('bn')
            elif ('pn backgrounds' in parts or re.search(
                    r'(?<![a-z0-9])pn[ _-]*back(?:ground)?(?=$|[^a-z])', parts[-1])):
                kinds.add('pn')
            else:
                role = detect_pn_role(Path(name).name)
                kinds.add(role.lower() if role in ('P', 'N') else 'custom')
    return next(iter(kinds)) if len(kinds) == 1 else 'custom'


def adopt_saved_background(owner, resolved):
    """Adopt a verified exported recipe without starting another load or writing stores."""
    from dataclasses import replace
    from ui_qt.drr_background_settings import BackgroundDraft
    from ui_qt.drr_background_settings_workflow import _root_key
    assignments = tuple(resolved.assignments)
    # A saved recipe is already concrete. An old group's reviewed BN plan or
    # managed-version draft must not veto it or relabel its actual sources.
    draft = BackgroundDraft(kind=_kind(assignments),
        which=assignments[0].baseline_which, auto_bn_compare=owner.drr_bn_auto_chk.isChecked())
    owner._drr_background_intent = draft
    owner._drr_background_intent_root = _root_key(owner)
    owner._drr_background_intent_selection = tuple(owner.drr_selected_files)
    owner._drr_bn_confirmation = None
    owner._drr_pending_bn_plan = None
    owner._drr_baseline_user_selected = True
    owner._drr_external_auto_selected = True
    owner.drr_controller._restore_saved_drr_recipe(resolved)
    if resolved.numerical_path == 'common':
        owner._drr_assignments_automatic = False
    owner._drr_background_intent = replace(draft,
        files=tuple(owner.drr_baseline_files_manual), regions=tuple(owner._drr_background_regions))
    owner.drr_controller._update_drr_selection_labels()
    owner._status('Restored the background used by this group\'s latest saved DRR export.')


def _restore(owner, resolved):
    adopt_saved_background(owner, resolved)
    owner._start_load('DRR', drr_resolution=resolved)


class SavedBackgroundRequest(QObject):
    def __init__(self, owner, draft, key):
        super().__init__(owner)
        from ui_qt.drr_background_settings_workflow import _context
        self.owner, self.draft, self.key = owner, draft, key
        self.context = _context(owner)
        self.auto_bn_compare = owner.drr_bn_auto_chk.isChecked()
        self.cancelled = _Cancellation()
        self.worker = Worker(_resolve_saved, str(owner.current_folder),
            tuple(getattr(owner, 'drr_available_sources', ())), tuple(owner.drr_selected_files),
            cancellation=self.cancelled)
        self.worker.signals.result.connect(self.complete, Qt.QueuedConnection)
        self.worker.signals.error.connect(self.failed, Qt.QueuedConnection)
        self.worker.signals.finished.connect(self.finished, Qt.QueuedConnection)

    def current(self):
        from ui_qt.drr_background_settings_workflow import _context, source_intent
        return (not self.cancelled.is_set() and not getattr(self.owner, '_is_closing', False)
                and getattr(self.owner, '_drr_external_request', None) is self
                and source_intent(self.owner) is self.draft
                and _context(self.owner) == self.context
                and self.owner.drr_bn_auto_chk.isChecked() == self.auto_bn_compare)

    @Slot(object)
    def complete(self, resolved):
        if not self.current():
            return
        self.owner._drr_external_request = None
        self.owner._drr_saved_background_lookup = self.key
        self.owner._drr_saved_background_blocked = resolved is not None and not resolved.resolved
        if resolved is None:
            from ui_qt.drr_background_settings_workflow import follow_source_type
            follow_source_type(self.owner, selection_changed=True, restore_saved=False)
        elif resolved.resolved:
            _restore(self.owner, resolved)
        else:
            self.owner._invalidate_drr_for_background_selection(resolved.reason)
            self.owner.drr_controller._update_drr_selection_labels()

    @Slot(str)
    def failed(self, message):
        if self.current():
            self.owner._drr_external_request = None
            self.owner._drr_saved_background_lookup = self.key
            self.owner._drr_saved_background_blocked = True
            self.owner._invalidate_drr_for_background_selection(
                'Saved DRR background could not be restored: ' + message.split('\n\n', 1)[0])
            self.owner.drr_controller._update_drr_selection_labels()

    @Slot()
    def finished(self):
        if getattr(self.owner, '_drr_external_request', None) is self:
            self.owner._drr_external_request = None
        self.owner.drr_controller._update_drr_baseline_controls()
        self.deleteLater()


def begin_saved_group_restore(owner, draft):
    """Check once on a changed group; same-group explicit edits remain applied."""
    selected = tuple(owner.drr_selected_files)
    if not selected or selected == getattr(owner, '_drr_background_intent_selection', selected):
        return False
    key = (str(owner.current_folder), selected, id(draft))
    if getattr(owner, '_drr_saved_background_lookup', None) == key:
        return bool(getattr(owner, '_drr_saved_background_blocked', False))
    request = getattr(owner, '_drr_external_request', None)
    if isinstance(request, SavedBackgroundRequest) and request.key == key and request.current():
        return True
    owner.drr_controller._cancel_auto_external()
    owner._drr_assignments = ()
    owner._drr_assignments_automatic = False
    owner.drr_baseline_files_manual = []
    owner.drr_baseline_files_found = []
    owner._drr_background_regions = ()
    owner._drr_baseline_user_selected = True
    owner._drr_external_auto_selected = False
    owner._drr_pending_bn_plan = None
    owner._invalidate_drr_for_background_selection('Restoring this group\'s saved DRR background…')
    owner.drr_controller._update_drr_selection_labels()
    request = SavedBackgroundRequest(owner, draft, key)
    owner._drr_external_request = request
    owner.drr_controller._update_drr_baseline_controls()
    owner.thread_pool.start(request.worker)
    return True
