# DRR dual view and paired export review

## Core helper review

Scope: `core/export.py::export_drr_pair_pngs_and_dat` and its four core tests. No production code changed during review; Qt view/export integration remains pending until its implementation is frozen.

Result: no actionable core-helper findings in the reviewed scope.

Independent verification with `.venv/Scripts/python.exe`:

- Four targeted core tests passed: two PNG/two DAT products, full-resolution numeric contents, derivative metadata None/2, repeat reuse, missing-DAT repair, upfront malformed-product rejection, and independent derivative-settings invalidation.
- Additional temporary-output reproductions passed: existing raw-only history reuses raw and creates second; injected second-writer failure propagates without returning aggregate success; missing second parameters cause no writer call or output directory creation.

The helper can leave the completed first product when the second writer raises. It does not report the partial pair as successful; a subsequent invocation uses the existing per-product repair/reuse behavior. No full suite, UI review, latency measurement, or general performance claim is included in this interim result.

## UI and owned-save verification checkpoint

Independent `.venv/Scripts/python.exe -m unittest -q` runs passed seven reviewer interaction regressions and five `tests.test_drr_dual_save_workflow` tests. These close the initial layout/product-switch zoom loss, manual-range override, Clear exception, raw split-scale leakage, d2 Auto reset, advanced dE rendering/label, product-specific fit invalidation, and same-session missing-output repair findings. The owned-save tests cover raw/second/side-by-side selection, two PNG/two DAT output, frozen SG settings, shared manual export ranges, and the d2 `dptk_rdbu_r_p0p60` colormap.

The final functional finding at that checkpoint was direct map pan/zoom followed immediately by Save capturing the previous viewport. A real `_start_export` reproduction showed visible X=(-0.7, 0.9), Y=(-0.4, 0.6), while both queued product parameter snapshots retained X=(-2, 2), Y=(-1, 1). Added `test_save_captures_current_pan_zoom_without_a_view_switch` to the reviewer regression module.

## Final functional decision

**Review: Approved.** All actionable functional findings raised in this review are closed. The coordinating reviewer also verified the revised side-by-side spacing after the final functional review.

The last fix captures the current shared viewport at the DRR save entry point. Independently ran `.venv/Scripts/python.exe -m unittest -q tests.test_drr_dual_view_regressions.DrrDualViewRegressionTests.test_save_captures_current_pan_zoom_without_a_view_switch`: one test passed. Together with the preceding independently verified seven interaction regressions and five owned-save workflow tests, this covers the eight reviewer reproductions without redundantly rerunning unchanged tests. The final nested layout preserves shared X/Y axes and each product's matching linecut; no new functional issue was found in the final patch. `git diff --check` passed.

Verification used synthetic data, temporary outputs and temporary INI settings. The owned-save workflow exercises the real request/worker/completion path but substitutes PNG rendering with a lightweight file writer in several tests; full PNG appearance is covered separately by the coordinator's synthetic-window screenshots. No full Qt suite, real experimental data, runtime restart, or latency benchmark was used. Review edits are limited to the dedicated regression test file and this report.

## Final coordinator verification

The final actual MainWindow synthetic render completed without errors. Both heatmaps retained X=(1.5, 1.8) and Y=(-2, 2); the revised nested layout separates the central colorbar tick labels from the second map and aligns each map with its linecut. Visual artifact: `artifacts/drr-dual-review/final-side-by-side-figure.png`. This combined image is only a QA artifact; application Save both creates separate PNGs, never a combined export. Offscreen Qt widget font rendering was unsuitable for a full-shell appearance assessment, so the final visual check is limited to the Matplotlib plot layout.

Final `.venv/Scripts/python.exe -m py_compile` on the five touched production files and four new test modules passed. Ordinary `git diff --check` passed (existing LF/CRLF notices only). No remaining review blocker.
