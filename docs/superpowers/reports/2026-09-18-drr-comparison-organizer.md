# DRR magnetic comparison organizer

Implemented a comparison catalog above the existing dataset list. Search by sample, spot and gate condition, preview members, then open the group. Magnetic field and actual Y coverage do not split groups. Temperature, excitation wavelength, rotation and other retained condition tokens do. `RotIn`/`Rot`, Unicode minus and numeric spellings are normalized; ambiguous or incomplete names remain separate review entries.

Membership JSON is separate from numeric analysis snapshots. Includes, exclusions and Restore automatic survive rescans. Removing a member does not remove source files or its cached analysis. Repeated fields/processed versions remain separate products. New DAT content gets a new numerical identity and cannot reuse old results.

Switching groups saves the outgoing group and restores the incoming common P2P range, display range, peak state and matching P2P records. Earlier completed P2P ranges are archived before recalculation and on group save; Restore saved P2P range can reopen a matching membership/data-version snapshot. Save As does not redirect the membership store during the session. Disk caches use JSON and pickle-free NPZ. This is graceful-save persistence, not crash recovery.

Catalog scans, group loads, exclusion saves and history loading run in background jobs. The group member list shows field, averaging count, actual Y coverage and analysis status. Comparison includes all current members; selecting a member changes inspection. Different Y grids are retained without automatic interpolation or averaging.

Verification:

- 49 unittest cases passed across comparison core/UI, analysis window, P2P, workspace sessions, processed groups, group picker and range amplitude.
- The switch/exclusion test covers restored P2P arrays and display bounds, range history, Save As store stability and source-data invalidation.
- Read-only scan of 321 real saved products produced 261 catalog entries/groups, including 74 incomplete/ambiguous entries requiring review.
- The real `YZ365 / p5n2 / TG-1.087BG=18 / 1.67 K / 720 nm / Rot49` group contains six products at 0, 0, 1, 2, 3 and 4 T. Both 0 T products remain distinct (avg 1 and avg 4).
- Loaded all six real DAT products and calculated the common 1.83–1.87 eV window. Row counts: 231, 157, 157, 157, 157, 93. Actual Y ranges span approximately −5…20, −2…15 and −2…8 V.
- Offscreen visual inspection: `artifacts/drr-comparison/organizer.png`.

Current scope: source filenames provide grouping metadata because the inspected products have no structured condition fields. Manual additions can include otherwise unmatched products. Fixed-Y extraction, interpolation and automatic averaging remain outside this iteration.

## Follow-up: preserve the active analysis page during group loading

Group replacement now keeps the page the user is viewing, bulk-populates members without activating each file, and activates just the final inspection dataset. Hidden peak previews no longer render or calculate derivative products. Switching back to Peak positions generates its preview on demand. Saved P2P state is retained lazily when the user remains on the peak page and remains available to workspace persistence.

Regression coverage verifies four distinct datasets load into P2P while a saved Peak positions tab cannot override the current tab, only one activation occurs, no peak product is computed, and switching back creates the peak plot. The three comparison UI tests pass; the 34 existing analysis-window/P2P tests also passed in the regression run.
