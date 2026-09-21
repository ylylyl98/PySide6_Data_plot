# DRR three-region color scales

Implemented the approved optional three-region mode. In DRR, expand Manual plot ranges, select Color scale → 3 regions, and set Boundary 1 and Boundary 2. Each region has independent min/max, Auto, and Fix controls. Raw and second-derivative products share boundaries but keep separate color limits. Show boundary only controls the dashed lines.

Three colorbars stack vertically in a dedicated right-side area. Preview bar heights reserve label space in points and update on resizing; the dual-product view reserves additional right margin. Three-region PNG exports are 9.5 × 6.2 inches rather than 8 × 6.2 inches, preserving the original heatmap's physical size. One- and two-region colorbar layouts remain unchanged. Region configuration is remembered as a JSON value in application settings and is included in export metadata JSON. A color-only change does not rewrite DAT data.

Implementation extends SplitColorScale with optional middle limits and a second boundary. Drawing and artist reuse share validated region masks. Two boundaries snapping to the same data-cell edge are rejected. Region Auto honors fixed limits and fixed boundaries. Closed-loop colorbar updates retain compact endpoint ticks when normalization type changes.

## Verification

- 65-case targeted suite passed (`artifacts/test_drr_three_region_regressions.py`), covering legacy split controls, masks, layout, artist reuse, paired export, quick export, and save workflows.
- After final adjustments, all 3 dedicated three-region UI tests and the additional normalization/reuse test passed.
- Python compilation and scoped diff whitespace checks passed.
- Screenshot verification at 1400×1000 and 1180×760, both single and dual products, confirmed no overlapping or cropped colorbar labels. A resize regression also checks a 640×480 Matplotlib canvas.
- Raw and second-derivative exported PNGs were rendered and inspected; colorbar bounding boxes are separate and inside the canvas.
- Export regression confirms switching single→triple preserves DAT bytes and modification time while updating PNG and JSON.

Code review found and prompted fixes for stale derivative boundary propagation and cramped bars on small canvases. Three old UI regressions assumed synchronous range/derivative updates; their fixtures now exercise the existing deferred redraw and asynchronous completion paths, and avoid unintended source loading during setup.

Artifacts: `artifacts/drr-three-regions/preview-small-controls.png`, `preview-both.png`, `raw.png`, `second.png`. Verification script: `artifacts/verify_drr_three_regions.py`. Tests isolate application settings and file-inspection caches from user data. No commit was made; unrelated workspace changes were retained.
