# Source picker stability implementation plan

**Goal:** Check DRR baseline and other source selectors for unchanged catalog resets and first-paint filename clipping.

**Architecture:** Compare newly populated row data and flags before replacing visible QListWidget rows. Reuse this boundary in SourcePickerDialog and the custom Power/SHG/MCD lists. Keep existing source filtering, validation and selection fallback behavior when data changes.

**Tech stack:** PySide6 QListWidget, Qt model roles, unittest; project .venv Python.

**Scope:** User requested baseline and other tabs checked for the DRR picker flicker. Preserve all pre-existing work. No theme redesign or changes to scientific processing.

## Steps

- [x] Reproduce redundant resets with shared picker, Power group, SHG and MCD tests; verify DRR baseline uses the existing guard.
- [x] Add SourcePickerDialog.replace_rows_if_changed(widget, populate) and use it in repopulate and custom source lists. Compare full model role data and item flags, including status changes.
- [x] Verify changed catalogs still update and selected identity/scroll remains stable for unchanged catalogs.
- [x] Run related workflow tests and first-paint layout probes with themes/scaling, then review the bounded changes.

Validation: `.venv/Scripts/python.exe -m unittest tests.test_source_picker_dialog tests.test_drr_picker_stability tests.test_source_picker_stability -v`, plus existing Power/PL/MCD/Compare/SHG workflow tests.
