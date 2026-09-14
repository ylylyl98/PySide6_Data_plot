# Unified MCD final verification

Implementation: Luna High. Independent review: Astra High, with final combined functional approval in `2026-09-11-astra-unified-mcd-review.md`.

## Final evidence

- Root: 34 retention/export-integration/unified UI tests passed (19.550 s).
- Root: 64 legacy MCD/valley/export/DRR/workflow regressions passed (32.327 s).
- Core selected-feature analysis: 35 tests passed and Astra independently approved numerical fixes.
- Exporter: 17 tests passed, including absolute source identity and Processed discovery integration.
- Changed-file whitespace checks passed; only existing CRLF conversion warnings.
- Real YZ365 source copied to a temporary directory; normal Open File -> automatic loading -> feature selection -> export completed. No experimental source writes or live-app restart.
- Load: 2.104 s. Background candidate catalog completed at 13.307 s, 155 candidates. Selected spectral peak 1.635985382 eV produced a valid measured track.
- Export: 2.535 s, one workbook (MCD, Slopes, Energy, Shift), four independent PNGs, map DAT, trace CSV, settings JSON; source discovered as Processed.
- Native Qt B display updates: 566.2 ms after full redraw, then 17.3/14.5 ms warm. This small sample is not a percentile benchmark.

## Verified review fixes

Coordinate ordering, selected-track references and bounds, explicit association versus valley-pair distinction, signed metrics, physical-field map clicks, energy/shift units, checked-only immutable retained snapshots, retained slope preservation, source-history identity, map controls, and complete valid-slope/difference readout passed their focused review cases.

## Remaining limits

- Compact slope plot text does not list every invalid-fit reason, sample count, or actual fitting range; exported diagnostics retain them.
- Splitting requires supported channel pairing and shared measured fields; unsupported pairs report unavailable rather than inventing values.
- Full source candidate detection remains a background operation taking roughly 11 seconds after loading in this data set. Warm field interaction is much faster; full redraw still costs about half a second.
- The running application must be restarted to use the changed Python code. No restart was performed.
