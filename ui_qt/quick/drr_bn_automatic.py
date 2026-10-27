"""Native BN background checks and guarded asynchronous delivery.

Comparison may publish an immutable report, but activation is a separate GUI
commit after the owner verifies its exact request and the trial products.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

from PySide6.QtCore import QObject, Signal, Slot


@dataclass(frozen=True)
class BnOutcome:
    message: str
    payload: dict | None = None
    prepared: object = None
    products: object = None
    withdraw_activation: bool = False


def common_reference(request):
    """Cheap structural check; content and managed-source checks run in worker."""
    recipe = request.background_recipe
    if request.background_mode != "external":
        return None
    if recipe is not None:
        choices = {(item.baseline_mode, item.baseline_files, item.baseline_which, item.regions)
                   for item in recipe.assignments}
        if len(choices) != 1:
            return None
        mode, files, which, regions = next(iter(choices))
        if mode != "External" or regions:
            return None
    else:
        files, which = request.backgrounds, request.background_frame
    if not files or any(Path(name).suffix.lower() != ".csv" for name in (*request.files, *files)):
        return None
    return tuple(files), which


def managed_request(request):
    recipe = request.background_recipe
    calibration = recipe.draft.calibration if recipe is not None and recipe.draft is not None else None
    return bool(calibration and not calibration["original"] or
                any(".drr_bn_calibration" in Path(name).parts for name in request.backgrounds))


def can_compare(request):
    reference = common_reference(request)
    if reference is None:
        return False
    from core.drr_measured_bn import is_measured_bn_name
    return managed_request(request) or all(is_measured_bn_name(name) for name in reference[0])


def check_background(request, *, automatic, reload, cancellation):
    """Use the existing joint certificate, or compare/fall back without commit."""
    from core.drr_bn_store import DrrBnStore, is_bn_calibration_source, joint_validation_passed
    from core.drr_sources import portable_source_name, resolve_source_path
    from core.load_cancellation import load_scope
    from ui_qt.quick.drr_background_workflow import BackgroundDraft, prepare_application
    from ui_qt.quick.drr_workflow import background_context, load_drr

    with load_scope(cancellation):
        context = background_context(request)
        reference = common_reference(request)
        if reference is None:
            raise ValueError("BN comparison needs one common reference without regional overrides.")
        files, which = reference
        files = tuple(portable_source_name(context.folder, resolve_source_path(context.folder, name)) for name in files)
        store = DrrBnStore(context.folder)
        managed = any(is_bn_calibration_source(context.folder, name) for name in files)
        version = None
        previous = None
        if managed:
            if len(files) != 1:
                raise ValueError("Choose one managed BN background for this group.")
            version = store.version_for_csv(files[0])
            source = version.source_identity
            if {item["path"] for item in source["measurements"]} != set(context.measurements):
                raise ValueError("This processed BN belongs to another measurement group.")
            original_files = tuple(item["path"] for item in source["baseline_files"])
            original_which = source["baseline_which"]
            previous = store.load_result(version)
            current = (version.fresh and version.eligible and version.validation_status == "accepted"
                and source.get("y_axis") == context.y_axis
                and (version.settings.get("derivative_window"), version.settings.get("derivative_polyorder"))
                    == (context.derivative_window, context.derivative_polyorder)
                and joint_validation_passed(version.report, version.settings))
            if current:
                draft = BackgroundDraft(kind="bn", files=files, which=which, auto_bn_compare=automatic,
                    calibration=dict(payload=previous, original=False))
                prepared = prepare_application(context, draft)
                captured = replace(request, background_recipe=prepared.recipe, backgrounds=prepared.recipe.files,
                    background_frame=prepared.recipe.which)
                products = load_drr(captured, prepared_background=prepared) if reload else None
                return BnOutcome("Saved BN validation matches the current group and SG settings.", previous,
                    prepared if reload else None, products)
        else:
            from core.drr_measured_bn import is_measured_bn_name
            if not all(is_measured_bn_name(name) for name in files):
                raise ValueError("Choose measured BNback / BNbackground sources before comparing.")
            original_files, original_which = files, which
            version = store.find_current(context.measurements, files, which)
            if version is not None and version.fresh:
                previous = store.load_result(version)

        payload = None
        if automatic:
            from core.drr_bn_service import compare_bn_group
            # Reuse only stored scientific settings. SG values always come from
            # the current request, including a changed derivative window.
            config = dict(previous.get("settings", {}) if previous else {})
            config.pop("auto_apply", None)
            config.update(derivative_window=context.derivative_window,
                          derivative_polyorder=context.derivative_polyorder)
            payload = compare_bn_group(context.folder, context.measurements, files, which, config,
                y_axis=context.y_axis, cancellation=cancellation)
            cancellation.check()
            saved = store.get(payload["version_id"]) if payload.get("version_id") else store.save_result(payload)
            payload = {**payload, "version_id": saved.version_id, "csv_path": str(saved.csv_path)}
            original_chosen = store.has_original_choice(context.measurements, original_files, original_which)
            payload["original_chosen"] = original_chosen
            fixed = request.background_recipe is not None and request.background_recipe.fixed
            if payload.get("eligible") and joint_validation_passed(payload.get("report", {}), payload.get("settings")) and not original_chosen and not fixed:
                draft = BackgroundDraft(kind="bn", files=files, which=which, auto_bn_compare=True,
                    calibration=dict(payload=payload, original=False))
                prepared = prepare_application(context, draft)
                captured = replace(request, background_recipe=prepared.recipe, backgrounds=prepared.recipe.files,
                    background_frame=prepared.recipe.which)
                products = load_drr(captured, prepared_background=prepared)
                cancellation.check()
                return BnOutcome("Validated BN candidate applied for this group.", payload, prepared, products)
            if fixed:
                return BnOutcome("BN comparison finished; Keep background retained your selected reference. Review BN comparison to apply a candidate explicitly.", payload)
            if not managed:
                return BnOutcome("BN comparison finished; the original reference was retained. " +
                    ("Your explicit original-BN choice remains active." if original_chosen else str(payload.get("summary", ""))), payload)

        if managed:
            # Automatic fallback must not masquerade as the user's explicit
            # 'keep original' decision, which suppresses later auto application.
            draft = BackgroundDraft(kind="bn", files=original_files, which=original_which, auto_bn_compare=automatic)
            prepared = prepare_application(context, draft)
            captured = replace(request, background_recipe=prepared.recipe, backgrounds=prepared.recipe.files,
                background_frame=prepared.recipe.which)
            products = load_drr(captured, prepared_background=prepared)
            cancellation.check()
            return BnOutcome("Original BN restored; the processed reference lacks current joint DR/R + D2E validation.",
                payload or previous, prepared, products, withdraw_activation=True)
        return BnOutcome("Automatic BN comparison is off.")


class AutomaticBnController(QObject):
    """Uses the owner's executor; stale/cancelled results never reach its commit."""
    stateChanged = Signal()
    resultReady = Signal(object, object, float)
    failed = Signal(object, str)
    _finished = Signal(int, object, str, float)

    def __init__(self, executor, parent=None):
        super().__init__(parent)
        self._executor = executor
        self._epoch = 0
        self._token = None
        self._scope = None
        self._running = False
        self._closed = False
        self._pending = {}
        self._finished.connect(self._accept)

    @property
    def running(self):
        return self._running

    def start(self, request, scope, *, automatic, reload=False):
        from core.load_cancellation import LoadCancellation
        self.cancel()
        self._scope = scope
        token = self._token = LoadCancellation()
        epoch = self._epoch
        self._running = True
        self.stateChanged.emit()
        start = perf_counter()
        future = self._executor.submit(check_background, request, automatic=automatic,
            reload=reload, cancellation=token)
        self._pending[epoch] = future
        def done(task):
            try:
                result, error = task.result(), ""
            except Exception as exc:
                result, error = None, str(exc) or type(exc).__name__
            if not self._closed:
                self._finished.emit(epoch, result, error, perf_counter()-start)
        future.add_done_callback(done)

    @Slot(int, object, str, float)
    def _accept(self, epoch, result, error, elapsed):
        self._pending.pop(epoch, None)
        if self._closed or epoch != self._epoch:
            return
        self._running = False
        self.stateChanged.emit()
        if error:
            self.failed.emit(self._scope, error)
        else:
            self.resultReady.emit(self._scope, result, elapsed)

    def cancel(self):
        self._epoch += 1
        if self._token is not None:
            self._token.cancel()
        self._running = False
        self.stateChanged.emit()

    def close(self):
        self._closed = True
        self.cancel()
        for future in self._pending.values():
            future.cancel()
