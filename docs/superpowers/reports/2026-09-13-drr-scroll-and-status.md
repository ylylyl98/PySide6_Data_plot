# DRR status labels and scroll retention

## Causes

- Session status text followed all filenames, making it easy to lose below wrapped names. Chosen-file updates explicitly replaced their original status summary with a basename, leaving only color to distinguish processing state.
- The previous no-op guard handled identical catalogs only. A new source or changed history still cleared/rebuilt session and file models, then selected a row. That selection scrolled the view back toward the selected file, often the first one. Newly arriving groups could also push the selected session outside the latest-25 cutoff.

## Changes

- Session rows start with UNPROCESSED, PARTIAL or PROCESSED and counts. Chosen files retain an explicit state, including Missing/Incompatible/STATUS UNKNOWN when applicable.
- Synchronize session/file rows by UserRole identity: update data, move existing items, add/remove only changed identities. Retain current and multiple selections.
- On catalog changes, capture the top visible identity and its pixel offset, lay out updated rows, then restore that visual position. Preserve the current, all selected, and top-visible sessions when new groups move them outside the recent cutoff.
- Explicit filter changes still use the normal filter behavior. Background refresh no longer resets models or selects the first row when the prior selection disappears.

## Verification

- Before changes, the insertion regression moved the visible anchor from -2px to 467px; status-first and chosen-status regressions failed.
- 65 related DRR/catalog/baseline/selection tests passed after changes.
- Three Windows native/Fusion/light tests passed: changed catalog while scrollbar is held, selected item beyond recent cutoff, and status transitions.
- Actual cached-data picker screenshot inspected: UNPROCESSED and PROCESSED appear clearly at the start of each visible session.
- Isolated QSettings added to the stability tests to avoid changing the user's saved preferences.
- Independent review found no blocking defects; its additional multi-selection cutoff edge case was reproduced with two arriving sessions and fixed.
