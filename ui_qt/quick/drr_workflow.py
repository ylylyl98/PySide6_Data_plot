"""Scientific DRR operations for the native Quick frontend.

Requests capture plain values on the GUI thread. Loading and differentiation
run in the application's cancellable worker pool; no widget owns this state.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class DrrRequest:
    files: tuple[str, ...]
    backgrounds: tuple[str, ...] = ()
    background_mode: str = "self-last"
    background_frame: str = "last"
    y_axis: str = "auto"
    raw_derivative: int | None = None
    second_derivative: object = 2
    window: int = 21
    polyorder: int = 2
    restore_files: bool = True
    restore_centers: bool = True
    experiment_folder: str = ""
    background_recipe: object = None


@dataclass(frozen=True)
class DrrProducts:
    request: DrrRequest
    source: object
    raw: object
    second: object | None
    raw_window: int
    second_window: int
    second_error: str = ""
    input_signatures: tuple = ()
    history_candidates: object = None
    background_references: tuple = ()
    raw_snapshot: object = None


def background_context(request):
    from ui_qt.quick.drr_background_workflow import BackgroundContext
    return BackgroundContext(request.experiment_folder or str(Path(request.files[0]).parent), request.files,
        request.y_axis, request.window, request.polyorder)


def _input_signatures(request):
    if request.background_recipe is not None:
        from ui_qt.quick.drr_background_workflow import background_input_signatures
        return background_input_signatures(background_context(request), request.background_recipe)
    import hashlib
    from core.load_cancellation import check_load_cancelled
    signatures = []
    backgrounds = request.backgrounds if request.background_mode == "external" else ()
    inputs = list(dict.fromkeys((*request.files, *backgrounds)))
    for name in request.files:
        if Path(name).suffix.lower() == ".dat":
            from core.loader import _dat_sidecar_candidates
            # Track both presence and content: adding/removing a sidecar also
            # changes the scientific label and boundary interpretation.
            for sidecar in _dat_sidecar_candidates(Path(name)):
                if sidecar.is_file():
                    inputs.append(str(sidecar))
                else:
                    signatures.append((str(sidecar.resolve()), None))
    for name in dict.fromkeys(inputs):
        path = Path(name)
        if not path.is_file():
            raise ValueError("A measurement/background changed or disappeared. Reload before exporting.")
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                check_load_cancelled()
                digest.update(block)
        after = path.stat()
        signature = lambda stat: (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if signature(before) != signature(after):
            raise ValueError("A measurement/background changed during loading. Reload before exporting.")
        signatures.append((str(path.resolve()), *signature(after), digest.hexdigest()))
    return tuple(signatures)


def _validate_managed_settings(request):
    """A managed reference cannot bypass its SG certificate via the file picker."""
    if request.background_mode != "external":
        return
    from core.drr_bn_store import DrrBnStore, is_bn_calibration_source
    folder = request.experiment_folder or str(Path(request.files[0]).parent)
    recipe = request.background_recipe
    names = tuple(dict.fromkeys((*request.backgrounds,
        *(name for assignment in recipe.assignments for name in assignment.baseline_source_files)))) if recipe else request.backgrounds
    for name in names:
        if not is_bn_calibration_source(folder, name):
            continue
        version = DrrBnStore(folder).version_for_csv(name)
        if (not version.fresh or not version.eligible or version.validation_status != "accepted"
                or version.source_identity.get("y_axis") != request.y_axis
                or (version.settings.get("derivative_window"), version.settings.get("derivative_polyorder"))
                    != (request.window, request.polyorder)):
            raise ValueError("The managed BN reference needs current joint DR/R + D2E validation at these SG settings.")


def process_drr(source, request: DrrRequest, *, input_signatures=(), background_references=(), raw_snapshot=None) -> DrrProducts:
    from core.processing import apply_sg_derivative_energy
    raw, raw_window = apply_sg_derivative_energy(
        source, derivative=request.raw_derivative,
        window_length=request.window, polyorder=request.polyorder)
    # A valid raw map remains usable when it has too few samples for a
    # derivative. The unavailable derivative is reported and cannot export.
    second, second_window, error = None, request.window, ""
    try:
        second, second_window = apply_sg_derivative_energy(
            source, derivative=request.second_derivative,
            window_length=request.window, polyorder=request.polyorder)
    except ValueError as exc:
        error = str(exc)
    return DrrProducts(request, source, raw, second, raw_window, second_window, error, input_signatures,
        background_references=tuple(background_references), raw_snapshot=raw_snapshot)


def load_drr(request: DrrRequest, *, prepared_background=None) -> DrrProducts:
    from core.data_io import load_drr_external_cube, load_drr_map_cube, load_drr_self_cube
    from core.loader import load_dat
    if not request.files:
        raise ValueError("Select DRR measurement files first.")
    paths = tuple(Path(name) for name in request.files)
    if any(not path.is_file() for path in paths):
        raise ValueError("A selected DRR measurement no longer exists.")
    suffixes = {path.suffix.lower() for path in paths}
    if suffixes & {".dat", ".xlsx"}:
        request = replace(request, background_mode="imported", backgrounds=(), background_recipe=None)
    signatures = _input_signatures(request)
    verified_signatures = None
    references, raw_snapshot = (), None
    if suffixes & {".dat", ".xlsx"}:
        if len(paths) != 1:
            raise ValueError("Open one precomputed DAT/XLSX map at a time; do not mix it with CSV measurements.")
        path = paths[0]
        source = (load_dat(path) if path.suffix.lower() == ".dat" else
                  load_drr_map_cube(str(path.parent), path.name, y_axis=request.y_axis))
    elif suffixes == {".csv"}:
        folder = str(paths[0].parent)
        _validate_managed_settings(request)
        if request.background_recipe is not None:
            from ui_qt.quick.drr_background_workflow import load_with_recipe
            loaded = load_with_recipe(background_context(request), request.background_recipe,
                prepared=prepared_background, before_signatures=signatures)
            source, references, raw_snapshot = loaded.cube, loaded.references, loaded.raw_snapshot
            # The recipe loader already verified the complete input snapshot
            # after reading. Reuse its proof instead of rereading every file.
            verified_signatures = loaded.input_signatures
        elif request.background_mode == "external":
            if not request.backgrounds:
                raise ValueError("Select external background files before loading this DRR measurement.")
            source = load_drr_external_cube(folder, request.files, request.backgrounds,
                baseline_which=request.background_frame, y_axis=request.y_axis)
        else:
            source = load_drr_self_cube(folder, request.files,
                use_first_frame=request.background_mode == "self-first", y_axis=request.y_axis)
    else:
        raise ValueError("DRR supports CSV measurements or one precomputed DAT/XLSX map.")
    if (verified_signatures if verified_signatures is not None else _input_signatures(request)) != signatures:
        raise ValueError("A measurement/background changed during loading. Reload before exporting.")
    from core.drr_display_settings import find_candidates
    candidates = find_candidates(request.experiment_folder or str(paths[0].parent), request.files,
        restore_files=request.restore_files, restore_centers=request.restore_centers)
    return replace(process_drr(source, request, input_signatures=signatures,
        background_references=references, raw_snapshot=raw_snapshot), history_candidates=candidates)


def export_drr(products: DrrProducts, *, folder, raw_params, second_params, view, display_state=None):
    from core.export import build_drr_export_base, export_drr_png_and_dat, _validate_drr_pair_product
    request = products.request
    if not products.input_signatures or _input_signatures(request) != products.input_signatures:
        raise ValueError("A measurement/background changed since loading. Reload before exporting.")
    mode = {"self-first": "DR/R Self First", "self-last": "DR/R Self", "external": "DR/R External", "fitted": "DR/R Fitted", "imported": "DR/R Imported"}[request.background_mode]
    inputs = tuple(("measurement", name) for name in request.files) + tuple(
        ("background", name) for name in request.backgrounds if request.background_mode == "external")
    outputs = {}
    if view not in {"raw", "second", "pair"}:
        raise ValueError("Choose raw, second, or pair for DRR export.")
    # A derivative-only display still saves its underlying DRR product.
    selections = ("raw",) if view == "raw" else ("raw", "second")
    if "second" in selections and (products.second is None or second_params is None):
        raise ValueError(products.second_error or "The derivative is not available.")
    for name in selections:
        _validate_drr_pair_product(products.raw if name == "raw" else products.second,
            raw_params if name == "raw" else second_params, label=name)
    if view == "second":
        raw_params = replace(raw_params, xlim=second_params.xlim, ylim=second_params.ylim)
    background_metadata = {}
    if request.background_recipe is not None:
        from core.drr_sources import portable_source_name, resolve_source_path
        recipe = request.background_recipe
        context = background_context(request)
        background_sources = tuple(dict.fromkeys((*recipe.files,
            *(name for assignment in recipe.assignments for name in assignment.baseline_source_files))))
        texture_reference = (recipe.fitted_settings or {}).get("texture_reference_source")
        if texture_reference and texture_reference not in background_sources:
            background_sources += (texture_reference,)
        inputs = tuple(("measurement", portable_source_name(folder, name)) for name in request.files) + tuple(
            ("background", portable_source_name(folder, resolve_source_path(context.folder, name))) for name in background_sources)
        numerical_path = getattr(products.source, "drr_numerical_path", "fitted_raw_average" if products.background_references else "common")
        background_metadata.update(drr_background_assignments=[item.to_dict() for item in recipe.assignments],
            selection_method="explicit_per_measurement", numerical_path=numerical_path,
            average_method=("per-file dR/R, align to first measurement grid without extrapolation, then nanmean"
                            if numerical_path == "heterogeneous" else "per-file dR/R, then nanmean"))
        if products.background_references:
            from core.drr_fitted_background import export_references, reference_matches_files
            if len(products.background_references) != 1 or not reference_matches_files(products.background_references[0], context.measurements):
                raise ValueError("Reload DRR to fit one background to the selected raw intensity average before saving.")
            references = export_references(folder, products.background_references)
            background_metadata.update(selection_method="fitted_sample", numerical_path="fitted_raw_average",
                average_method="Align raw intensities on the common measured overlap without extrapolation, equal-weight file average, fit one background, then I/B_fit - 1",
                fitted_background=recipe.fitted_settings,
                fitted_background_diagnostics=[item.metadata() for item in products.background_references],
                fitted_background_files=references)
            outputs.update({f"background_{index}": Path(folder) / name for index, name in enumerate(references)})
    for name in selections:
        cube = products.raw if name == "raw" else products.second
        if cube is None:
            raise ValueError(products.second_error or "The derivative is not available.")
        derivative = request.raw_derivative if name == "raw" else request.second_derivative
        used_window = products.raw_window if name == "raw" else products.second_window
        base = build_drr_export_base(request.files[0], len(request.files), mode,
            derivative if isinstance(derivative, int) else None, used_window,
            request.polyorder, "More correct (regrid)")
        if isinstance(derivative, tuple):
            from core.drr_mixed_derivative import effective_window
            used_y = effective_window(derivative[1], len(cube.gate), request.polyorder)
            base += f"_dXdY_W{used_window}_Y{used_y}_O{request.polyorder}"
        processing = dict(background_mode=request.background_mode,
            mode=mode, baseline_selection={"self-first": "Self (first frame)",
                "self-last": "Self (last frame)", "external": "External", "fitted": "Fitted (sample)", "imported": "Imported map"}[request.background_mode],
            baseline_which=request.background_frame, average_count=len(request.files),
            y_axis=request.y_axis, derivative_order=2 if isinstance(derivative, tuple) else derivative,
            sg_window=used_window, sg_polyorder=request.polyorder,
            savgol_window=used_window, savgol_polyorder=request.polyorder,
            derivative_grid="More correct (regrid)")
        processing.update(background_metadata)
        if isinstance(derivative, tuple):
            processing.update(derivative_axes="XY", derivative_order_x=1, derivative_order_y=1,
                savgol_window_y=int(used_y), savgol_window_y_requested=int(derivative[1]),
                derivative_grid="Local polynomial on actual X/Y coordinates",
                y_derivative_coordinate=cube.gate_label)
        result = export_drr_png_and_dat(folder, cube=cube,
            params=raw_params if name == "raw" else second_params,
            export_base=base, processed_name="Processed Data/DRR",
            metadata_input_files=inputs,
            metadata_processing=processing,
            include_original_png=True,
            metadata_extra={"ui_state": display_state, "display_save_pending": True} if display_state is not None else None)
        outputs.update({f"{name}_{key}": value for key, value in result.items()})
    if display_state is not None:
        from core.drr_display_settings import save_snapshot, complete_metadata_snapshots
        save_snapshot(folder, display_state)
        complete_metadata_snapshots(path for key, path in outputs.items()
            if key in {"raw_dat", "second_dat"})
    return outputs
