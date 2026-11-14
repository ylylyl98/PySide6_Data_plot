"""PowerPoint figure handoff for the native PL/DRR shell."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter


def restore_pl_figure(display, record, params):
    """Restore saved figure values, then apply the user's explicit manual locks."""
    from core.pl_display_settings import validate_snapshot
    snapshot = validate_snapshot(record.snapshot) or display.capture_state()
    # These are figure values, not new persistent user locks. Keep Auto available
    # and let an explicit Fix gesture adopt a value as a local preference.
    snapshot["axis_fixed"] = {key: False for key in snapshot["axis_fixed"]}
    snapshot["geometry"]["fixed"] = {key: False for key in snapshot["geometry"]["fixed"]}
    for colors in snapshot["banks"].values():
        colors["fixed"] = {key: False for key in colors["fixed"]}
    scale = "log" if params.log_scale else "linear"
    snapshot["values"].update(xmin=params.xlim[0], xmax=params.xlim[1],
                              ymin=params.ylim[0], ymax=params.ylim[1], cmap=params.cmap,
                              scale=scale, sideBySide=False, clip=params.clip_outliers)
    bank = snapshot["banks"][scale]
    bank.update(vmin=params.vmin, vmax=params.vmax)
    split = params.split_scale
    geometry = snapshot["geometry"]
    geometry["count"] = (3 if split.split_x2 is not None else 2) if split else 1
    if split:
        geometry.update(x0=split.split_x, showBoundary=split.show_boundary)
        if split.split_x2 is not None:
            geometry["x1"] = split.split_x2
        for side in ("left", "right", "middle"):
            low, high = getattr(split, side + "_vmin"), getattr(split, side + "_vmax")
            if low is not None and high is not None:
                bank[side + "Min"], bank[side + "Max"] = low, high
    if not display.restore_state(snapshot, overlay_manual=True):
        raise ValueError("The saved PL figure display is incomplete or invalid.")


def restore_drr_figure(display, params, *, product="raw"):
    """An already computed derivative is shown in the imported-map channel."""
    manual = display._memory._manual()
    previous_y = getattr(display, "_ppt_previous_y_identity", None)
    cube = display._raw
    current_y = ((str(cube.gate_label).strip().casefold(),
                  str(getattr(cube, "gate_unit", "")).strip().casefold(),
                  str(getattr(cube, "y_axis_semantic", "")).strip().casefold())
                 if cube is not None else None)
    y_compatible = previous_y is not None and previous_y == current_y
    y_skipped = not y_compatible and any(("numbers", "axis.y" + end) in manual for end in ("min", "max"))
    old_numbers = dict(display._numbers)
    numbers = dict(old_numbers)
    flags = {key: False for key in display._flags if key.startswith("fix.")}
    sources = {}
    numbers.update(dict(zip(("axis.xmin", "axis.xmax", "axis.ymin", "axis.ymax"),
                            (*params.xlim, *params.ylim))))
    numbers.update({"raw.vmin": params.vmin, "raw.vmax": params.vmax})
    count = 0
    show_boundary = True
    split = params.split_scale
    if split:
        count = 2 if split.split_x2 is not None else 1
        numbers["boundary.x0"] = split.split_x
        if split.split_x2 is not None:
            numbers["boundary.x1"] = split.split_x2
        show_boundary = split.show_boundary
        for side in ("left", "right", "middle"):
            for bound in ("vmin", "vmax"):
                value = getattr(split, side + "_" + bound)
                if value is not None:
                    numbers["raw." + side + "_" + bound] = value
    # Explicit manual Fix values override the PNG; borrowed history does not.
    color_product = "raw" if getattr(display, "_ppt_previous_figure_mode", False) else product
    for (category, key), value in manual.items():
        if category != "numbers" or key.startswith("axis.y") and not y_compatible:
            continue
        # The derivative's whole-map range can be manually adjusted without
        # Fix controls; it is not an explicit lock on a different PNG.
        if key in {"second.vmin", "second.vmax"}:
            continue
        target = key
        if key.startswith(("raw.", "second.")):
            if not key.startswith(color_product + "."):
                continue
            target = "raw." + key.split(".", 1)[1]
        if target not in numbers:
            continue
        numbers[target] = value
        flag = "fix." + target
        if manual.get(("flags", "fix." + key), False) and flag in flags:
            flags[flag] = True
            sources[flag] = "M"
    prefixes = ["axis.x", "axis.y", "raw.v"]
    if count:
        prefixes += ["raw.left_v", "raw.right_v"]
    if count == 2:
        prefixes.append("raw.middle_v")
    for prefix in prefixes:
        lo, hi = prefix + "min", prefix + "max"
        if numbers[lo] >= numbers[hi]:
            fixed_lo, fixed_hi = flags.get("fix." + lo, False), flags.get("fix." + hi, False)
            if fixed_lo == fixed_hi:
                raise ValueError("Fixed PowerPoint display limits conflict with the saved figure; adjust or release Fix values.")
            peer = hi if fixed_lo else lo
            numbers[peer] = old_numbers[peer]
            if numbers[lo] >= numbers[hi]:
                delta = max(abs(numbers[lo if fixed_lo else hi]) * 1e-6, 10 ** -display._decimals(peer))
                numbers[peer] = numbers[lo] + delta if fixed_lo else numbers[hi] - delta
        if params.log_scale and prefix.startswith("raw.") and numbers[lo] <= 0:
            raise ValueError("Fixed color limits must be positive for this logarithmic figure.")
    if count and cube is not None:
        from core.plotting import SplitColorScale, split_region_slices
        region = SplitColorScale(numbers["boundary.x0"], numbers["raw.left_vmin"], numbers["raw.left_vmax"],
            numbers["raw.right_vmin"], numbers["raw.right_vmax"], show_boundary,
            numbers["boundary.x1"] if count == 2 else None,
            numbers["raw.middle_vmin"] if count == 2 else None,
            numbers["raw.middle_vmax"] if count == 2 else None)
        split_region_slices(cube.energy, region, (numbers["axis.xmin"], numbers["axis.xmax"]))
    display._memory.reset_borrowed()
    display._memory.viewport = {}
    display._pending.clear()
    display._figure_mode = True
    display._numbers.update(numbers)
    display._flags.update(flags)
    display._fix_sources = sources
    display._flags.update(log=params.log_scale, zero=params.center_zero, clip=params.clip_outliers, **{"boundary.show": show_boundary})
    display._choices.update(colorProduct=0, regionCount=count, **{"yaxis.right": 0})
    if params.cmap not in {name for name, _ in display._cmaps}:
        display._cmaps = (*display._cmaps, (params.cmap, params.cmap))
    display._choices["raw.cmap"] = next(index for index, (name, _) in enumerate(display._cmaps) if name == params.cmap)
    message = "PowerPoint figure · saved DAT · display edits only"
    if y_skipped:
        message += " · Y Fix not applied: coordinate identity differs or is unknown"
    display.set_memory_status(message, available=False)
    display.refresh()


class PowerPointApplicationMixin:
    def _ppt_init(self):
        self._ppt_applying = False
        self._ppt_display_session = None
        self._ppt_measurement_state = None
        self._ppt_service = None
        self._ppt_root = ""
        self._ppt_record = None
        self._ppt_params = None
        self._ppt_save_record = None
        self._ppt_update_timings = {}

    def _ppt_enter_display(self):
        if self._ppt_display_session is not None:
            return
        from ui_qt.quick.powerpoint_display_session import DisplaySession
        # Keep the normal measurement workflow intact while the saved DAT is
        # displayed. Copies are shallow for numerical products/cubes.
        references = ("_cube", "_source", "_sources", "_folder", "_export_valid",
                      "_drr_products", "_drr_background_recipe", "_drr_backgrounds",
                      "_drr_background_required", "_pl_active_source")
        mappings = ("_values", "_scales", "_pl_values", "_drr_processing")
        self._ppt_measurement_state = {
            **{key: getattr(self, key) for key in references},
            **{key: deepcopy(getattr(self, key)) for key in mappings},
        }
        display = self._pl_display if self._workflow == "pl" else self._drr_display
        self._ppt_display_session = DisplaySession(display)

    def _ppt_leave_display(self, *, preserve_next_drr=True):
        session = self._ppt_display_session
        if session is None:
            return
        self._ppt_display_session = None
        state, self._ppt_measurement_state = self._ppt_measurement_state, None
        self._ppt_record = self._ppt_params = None
        for key, value in state.items():
            setattr(self, key, value)
        session.restore(preserve_next_drr=preserve_next_drr)
        if self._workflow == "pl":
            self._pl_sync_scales()
        if not self._closed:
            self._schedule_plot()

    def _ppt_start(self):
        import sys
        if sys.platform != "win32" or self._ppt_service is not None:
            return
        from ui_qt.quick.powerpoint_service import PowerPointService
        self._ppt_service = PowerPointService(parent=self)
        self._ppt_service.figureRequested.connect(self._ppt_open)
        self._ppt_service.openingStarted.connect(self._ppt_opening)
        self._ppt_service.statusMessage.connect(self._message)
        self._ppt_service.figureUpdated.connect(self._ppt_updated)
        self._ppt_service.requestBinding.connect(self._ppt_binding_needed)
        self._ppt_sync_root()
        self._ppt_service.start()

    def _ppt_sync_root(self):
        if self._ppt_service is not None:
            root = self._folder
            for parent in (Path(root), *Path(root).parents) if root else ():
                if parent.name.casefold() == "processed data":
                    root = str(parent.parent)
                    break
            if root != self._ppt_root:
                self._ppt_root = root
                self._ppt_service.set_root(root)

    def _ppt_binding_needed(self, context):
        if self._busy:
            self._message("PowerPoint could not identify this PNG. Finish the current operation, then double-click it again.")
            return
        self._message("Choose the original exported PNG to associate this PowerPoint picture.")
        self.pptBindingRequested.emit()
        self.pptActivateRequested.emit()

    def _ppt_opening(self):
        if not self._closed and not self._busy:
            self.pptActivateRequested.emit()

    def _ppt_open(self, record):
        if self._closed:
            return
        if self._busy or self._background_open:
            self._message("Finish the current operation, then double-click the PowerPoint picture again.")
            return
        from core.pptx_figure_edit import experiment_root, load_figure
        try:
            experiment_root(record.png_path)
        except (ValueError, TypeError, KeyError) as exc:
            self._message(str(exc))
            return
        # Validate display metadata and load imports/DAT in the worker. Keep the
        # current figure intact until the requested figure is ready to display.
        self._submit("ppt-load", lambda: load_figure(record))
        self.pptActivateRequested.emit()

    def _ppt_finish(self, kind, result):
        if kind != "ppt-load":
            return self._ppt_finish_apply(kind, result)
        self._ppt_applying = True
        try:
            self._sync_source_catalog()
            return self._ppt_finish_apply(kind, result)
        finally:
            self._ppt_applying = False
            self._sync_source_catalog()

    def _ppt_finish_apply(self, kind, result):
        if kind == "ppt-load":
            record, cube, params = result
            from core.pptx_figure_edit import experiment_root
            self.selectWorkflow(record.workflow)
            self._ppt_enter_display()
            self._ppt_record = None
            self._folder = str(experiment_root(record.png_path))
            self._source, self._sources = record.dat_path, (str(record.dat_path),)
            self._values["sideBySide"] = False
            if record.workflow == "drr":
                from ui_qt.quick.drr_workflow import DrrProducts, DrrRequest
                display = self._drr_display
                previous = display._raw
                display._ppt_previous_y_identity = (
                    (str(previous.gate_label).strip().casefold(),
                     str(getattr(previous, "gate_unit", "")).strip().casefold(),
                     str(getattr(previous, "y_axis_semantic", "")).strip().casefold())
                    if previous is not None else None)
                display._ppt_previous_figure_mode = display._figure_mode
                request = DrrRequest(files=self._sources, background_mode="imported", experiment_folder=self._folder)
                self._drr_products = DrrProducts(request, cube, cube, None, 0, 0,
                    "PowerPoint edits use the saved numerical map; load the measurement to reprocess.")
                self._drr_background_recipe = None
                self._drr_backgrounds = ()
                self._drr_background_required = False
                self._drr_processing["drrView"] = "raw"
            self.set_cube(cube)
            if record.workflow == "pl":
                restore_pl_figure(self._pl_display, record, params)
                self._pl_sync_scales()
                self._pl_remember()
            else:
                restore_drr_figure(self._drr_display, params, product=record.role.removesuffix("_wide"))
            self._ppt_record, self._ppt_params = record, params
            self._pl_values.update(pngLayout="wide" if params.png_width_scale > 1 else "original",
                                   pngWidth=params.png_width_scale)
            self._export_valid = True
            self._schedule_plot()
            self._message(f"PowerPoint: {record.png_path.name} · edit display, then Save PNG + DAT. Numerical processing is preserved.")
        elif kind == "ppt-export":
            original, new_record, started = result
            self._result = f"Saved {new_record.png_path}\n{new_record.dat_path}"
            self._message("PNG + DAT saved. Updating the associated PowerPoint picture…")
            if self._ppt_service is not None:
                self._ppt_save_record = new_record
                self._ppt_update_timings[new_record.record_id] = (original.record_id, started)
                # Bound stale entries when PowerPoint reports an uncorrelated failure.
                while len(self._ppt_update_timings) > 32:
                    self._ppt_update_timings.pop(next(iter(self._ppt_update_timings)))
                try:
                    self._ppt_service.update_figure(original.record_id, new_record.png_path, new_record)
                except Exception:
                    self._ppt_update_timings.pop(new_record.record_id, None)
                    self._ppt_save_record = None
                    raise
            else:
                self._message("PNG + DAT saved; PowerPoint is not connected.")

    def _ppt_updated(self, old_id, record):
        if self._ppt_record is not None and self._ppt_record.record_id == old_id:
            self._ppt_record = record
        timing = self._ppt_update_timings.get(record.record_id) if record is not None else None
        if timing is not None and timing[0] == old_id:
            if self._ppt_save_record is not None and self._ppt_save_record.record_id == record.record_id:
                self._ppt_save_record = None
            self._ppt_update_timings.pop(record.record_id, None)
            self._message(f"PowerPoint save and update completed in {perf_counter() - timing[1]:.3f} s.")

    def _ppt_export(self):
        started = perf_counter()
        from core.pptx_figure_edit import save_figure_edit
        record, cube = self._ppt_record, self._cube
        if record.workflow == "pl":
            params = self._params(self._pl_display.values["scale"])
            snapshot = self._pl_display.capture_state()
        else:
            params = self._drr_display.heatmap_params(cube)
            snapshot = self._drr_display.capture_state(self._sources)
        # Preserve axis labels and the original caption; use the edited scale/width.
        params = replace(params, title=self._ppt_params.title, xlabel=self._ppt_params.xlabel,
                         ylabel=self._ppt_params.ylabel, cbar_label=self._ppt_params.cbar_label,
                         png_width_scale=self._png_export_width())
        role = ("log" if params.log_scale else "linear") if record.workflow == "pl" else record.role.removesuffix("_wide")
        edited_record = replace(record, role=role + ("_wide" if params.png_width_scale > 1 else ""))
        self._submit("ppt-export", lambda: (record, save_figure_edit(edited_record, cube, params, snapshot), started))

    def _ppt_guard_parameter(self, key):
        if self._ppt_record is None:
            return False
        blocked = key in {"backgroundMode", "backgroundFrame", "rawDerivative", "secondDerivative", "sgWindow", "sgPolyorder", "sgYWindow"}
        blocked = blocked or key in {"pl.yAxis", "pl.a", "pl.b", "pl.c", "pl.datChoice", "pl.datLabel", "pl.datUnit", "pl.processing", "pl.window", "pl.polynomial", "bnReference"}
        if blocked:
            self._message("PowerPoint editing preserves the exported DAT. Load a measurement normally to change numerical processing.")
        return blocked

    def _ppt_close(self):
        self._ppt_leave_display(preserve_next_drr=False)
        self._ppt_update_timings.clear()
        if self._ppt_service is not None:
            self._ppt_service.close()
