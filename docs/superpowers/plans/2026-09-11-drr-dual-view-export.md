# DRR dual view and paired export implementation plan

> For agentic workers: use executing-plans to implement the checked stages below. User explicitly requested Luna High implementation followed by Astra review; implementation is already authorized. Do not commit existing mixed work.

**Goal:** Present ΔR/R and its energy second derivative using compact plot tabs, optionally side by side, and save both as two independent PNGs and two DATs on every normal DRR save.

**Architecture:** Keep one loaded baseline-corrected DRR cube and use the existing SG energy derivative implementation and bounded cache. Plot view is presentation state, independent of export selection. Reuse the existing owned export worker and per-result DRR writer/fingerprint logic for a two-result package.

**Tech Stack:** PySide6/Qt Widgets, Matplotlib, NumPy/SciPy, unittest, project `.venv/Scripts/python.exe`.

**Spec:** Approved conversation: default compact ΔR/R / Second derivative tabs above the plotting region; optional Side by side; shared source/gate/axis ranges and zoom, separate color scales; default Save both · PNG + DAT; two PNG and two DAT, no combined PNG. Existing metadata sidecars remain necessary and are not extra plots/data products.

## Constraints

- Preserve all existing uncommitted changes and other workflows, especially recent source-status, async lifecycle and DAT/PNG performance fixes.
- Keep derivative definition/sign/grid identical to `core.processing.apply_sg_derivative_energy(..., derivative=2)`. Do not apply a second derivative to an already differentiated cube.
- No new dependencies, user experimental output, process restart, or full Qt suite. All verification outputs/settings use temporary directories.
- UI switching/gate updates must reuse loaded/cached data without file reload or repeated derivative calculation; invalidation includes loaded cube identity, SG window and polynomial order.
- User explicitly reaffirmed X/Y ranges must stay identical: tab switching retains both limits; zoom/pan either paired map synchronizes the other; editing shared range controls overrides any old cached view limits; both exported PNGs use the same frozen xlim/ylim. Only color ranges are independent.
- User specified the second-derivative default colormap: existing `core.colormaps.RDBU_R_P0P60_ID` (`RdBu_r p=0.60`), for both display and export, independent of the raw map colormap. The 0.60 is the existing colormap parameter, not numeric ±0.6 color limits.
- Preserve current source grouping/background resolution, first-derivative capability via advanced controls, peak/fit tools, and error behavior. Default paired save always means raw ΔR/R plus second derivative, irrespective of advanced or visible view.
- Reject or clearly report an unsupported second derivative (insufficient samples/invalid SG settings) without calling a partial pair successful.

## Stage 1 — Separate scientific products from visible view

Files: `ui_qt/controllers_drr.py`, `ui_qt/common.py`; numerical behavior remains in `core/processing.py`. Tests: new `tests/test_drr_dual_view.py`.

Consumes: `LoadedState.cube`, `apply_sg_derivative_energy`, current bounded derivative cache.
Produces: explicit transform selection usable without changing widgets; display and export can request derivative None or 2 independently.

- [x] Add failing numerical/cache regression using a quadratic energy spectrum: base values remain unchanged, derivative matches existing transform, repeated requests hit cache, SG/source changes invalidate.
- [x] Extend `_drr_cube_with_metadata` with an explicit derivative override while retaining current no-argument callers. The override must distinguish omitted from explicit None; use a module sentinel, not None as both meanings.

```python
_CURRENT_DRR_DERIVATIVE = object()
# In the controller method:
# derivative = self._drr_derivative_value() if override is _CURRENT_DRR_DERIVATIVE else override
# key = (id(self.loaded.cube), derivative, actual_window, polynomial_order)
```

- [x] Return the actual SG window/order used for each product, keep cache bounded, and verify targeted regression before proceeding.

## Stage 2 — Plot tabs and optional side-by-side layout

Files: `ui_qt/feature_pages.py` DRR controls, `ui_qt/main_window.py` plot header/draw/DRR action states, `ui_qt/controllers_drr.py` plot interaction. Tests: `tests/test_drr_dual_view.py`.

Consumes: Stage 1 explicit raw/second-derivative products and existing Qt plot-header patterns.
Produces: visible view state (`raw`, `second`, `both`), independent raw/second color settings, synchronized map and spectrum/gate views.

- [x] Add failing real-widget tests for default raw tab, second-derivative switching, Side by side, retained source/axis ranges and independent color limits.
- [x] Add compact plot-header tabs only visible in DRR, with a Side by side toggle. Keep shared gate and x/y limits; each panel uses its own colorbar and color settings. Both mode shows each heatmap with its corresponding linecut; single mode keeps the full plot width.
- [x] Keep existing first derivative available in advanced controls without confusing the default two-product export. SG controls remain in the existing expandable Parameters area with an explicit Advanced view label, preserving their familiar position; ensure selected derivative labeling is truthful.
- [x] Route gate updates to all visible spectra/cursors without rebuilding/copying the scientific cube. Link map zoom/range changes with recursion guards and preserve them across tab switches.
- [x] Verify switching and gate changes cause no source load/metadata scan and no redundant transform after warming the cache. Preserve fitting/selection hooks for the active panel and invalidate stale fits when the product changes.

## Stage 3 — Always export the pair

Execution split: `luna_drr_dual_implementation` owns all Qt integration and view tests; `luna_drr_pair_exporter` owns the paired core helper and `tests/test_drr_dual_export.py`. The helper consumes already prepared full-resolution raw/second cubes and parameters; the Qt export worker computes a missing second cube off the GUI thread.

```python
export_drr_pair_pngs_and_dat(
    folder, *, raw_cube, second_cube, raw_params, second_params,
    raw_export_base, second_export_base, metadata_input_files=(),
    metadata_processing_raw=None, metadata_processing_second=None,
    metadata_extra=None, processed_name="Processed Data/DRR",
)
# ExportPathResult keys: raw_png, raw_dat, second_png, second_dat.
# save_status is reused only if both products reused successfully.
```

Files: `ui_qt/common.py::ExportOptions`, `ui_qt/main_window.py::_start_export/_export_task/_update_action_states`, optional small helper in `core/export.py`. Tests: new `tests/test_drr_dual_export.py`.

Consumes: loaded base cube, frozen raw/second parameters and SG settings; existing `export_drr_png_and_dat` preserves atomic DAT and fingerprint reuse.
Produces: two PNGs plus two DATs with unmistakable raw/second-derivative filename suffixes and correct per-product metadata, never a combined PNG.

- [x] Add failing temporary-output test asserting exactly two `.png` and two `.dat`, correct raw/second numeric data and per-product derivative metadata. Run saves from each visible mode and assert identical product selection.
- [x] Capture both independent color settings and source/SG settings into `ExportOptions` on GUI thread; actual transform/export work goes through the existing owned background export path. Do not read live widgets in a worker.
- [x] Export raw derivative=None and second derivative=2 with distinct stems (e.g. `_DRR` / `_DRR_d2E`) using the existing per-file writer, provenance and reuse policy. Avoid duplicating background-resolution scans per product; prepare common provenance once.
- [x] Update the DRR save label/tooltip to `Save both · PNG + DAT`; reset to the existing wording in all other tabs. Completion/reuse is successful only when both pairs complete. Preserve missing-output repair and changed-parameter invalidation.
- [x] Verify repeated save reuses unchanged artifacts, missing one output restores it, SG changes affect only derivative analysis as appropriate, changing visible tabs alone does not rewrite scientific products, and old single-product history does not cause the second product to be skipped.

## Stage 4 — Astra review and verification

- [x] Luna High freezes implementation and reports exact scoped test commands/results, after running new tests with `.venv/Scripts/python.exe -m unittest -q tests.test_drr_dual_view tests.test_drr_dual_export` (isolate GUI processes if needed).
- [x] Astra reviews scientific transformation, default product selection, independent scales/shared interaction, actual event-loop save lifecycle and repeat/missing-output semantics; fix all actionable regressions through Luna.
- [x] Run targeted existing DRR/export regressions, actual queued worker tests with temporary QSettings, `py_compile` on touched files and ordinary `git diff --check`. No full Qt suite or unsupported speed claim.
- [x] Record verification and any limitations in `docs/superpowers/reports/2026-09-11-drr-dual-view-export-review.md`; deliver concise user-facing changes and the four resulting filenames.
