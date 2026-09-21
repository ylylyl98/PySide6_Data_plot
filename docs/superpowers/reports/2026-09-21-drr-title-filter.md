# DRR display-title simplification

All DRR PNG exports (one/two/three regions, raw and derivatives) now remove only recognized bookkeeping from the displayed title, then fit the remaining full text in the existing header. Canvas dimensions, heatmap positions, colorbar layouts, file names, source titles, and DAT values are unchanged.

Recognized removal covers duration/count exposure tokens, explicit background-file keys and known background methods, date/time-shaped timestamps, numeric session identifiers, standalone END markers, and known file extensions. Unknown conditions, REF, repeat IDs, and gate relationships remain. Unknown explicit assignments are protected even when their values are END, session003, or date-shaped. Math expressions are preserved as indivisible wrapping tokens. Unrecognized formats are retained conservatively.

The original remains in metadata `plot.title`; `display_title` and `title_policy_version` record the visible title and filtering policy. DRR-only render version 4 causes old PNGs to refresh on export without recalculating DAT. PL/MCD title paths remain unchanged.

Verification: 41 tests passed across title filtering, region rendering, dual export, metadata, and export history caching. Python compilation and scoped whitespace checks passed. Review findings concerning unknown key values and self_first/self_last methods were corrected. Example exports for every region count were generated at 1200×930; the two-region PNG was visually inspected with a retained TG-1.087BG=0 constraint.

Preview script: `artifacts/preview_drr_simplified_title.py`; examples: `artifacts/drr-simplified-title/reference-1.png`, `reference-2.png`, `after-3.png`. No filenames or data were renamed or edited.
