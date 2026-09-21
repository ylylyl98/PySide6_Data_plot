# Compact three-region export preview

Only three-region PNG export now uses the original 8 × 6.2 inch canvas (1200 × 930 px at 150 DPI) and original heatmap axes rectangle. Three short horizontal colorbars occupy the top-right header, with L/M/R labels and two endpoint ticks each. Complete titles wrap and reduce font size to fit instead of truncating trailing words. Interactive preview colorbars are unchanged.

Three-region export render fingerprints advance to version 3 so previously cached PNGs regenerate without rewriting DAT. Single/two-region fingerprints and rendering remain unchanged. Before/after SHA256 checks of the two reference PNGs were identical:

- 1 region: 3933A6A635DC2C5910243C28E30ED415CB11393DCEB7ACBCFC89E89E5D289508
- 2 regions: 63CC6C9BB73AEF0D33497CD2AD6AD37AA5877A42FD5C4A0AA709438FA8A74216

18 tests passed across three-region rendering, legacy split rendering, and dual export. Checks cover fixed geometry, complete title text, separated header bounding boxes, scientific endpoint labels, JSON metadata, DAT preservation, and paired exports. Compilation and scoped whitespace checks passed. Reviewer identified scientific endpoint overlap; endpoint labels now align within their individual bars and use compact numeric formatting with measured font sizing.

Preview data are synthetic and identical across modes. The demonstration title originally used manually inserted vertical bars; these are not added by production code. `after-3-no-separators.png` removes only those example separators, retaining every field. No automatic title-field removal has been implemented. The user asked for recommendations on field importance; acquisition details and redundant sweep descriptions can be moved to metadata only after deciding what is relevant to their figure.

Artifacts: `artifacts/drr-compact-export/before-3.png`, `after-3-no-separators.png`, `comparison.png`, and `reference-1.png`/`reference-2.png`. Script: `artifacts/preview_compact_drr_export.py`.
