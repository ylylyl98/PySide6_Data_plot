# P2P comparison presentation and export

Compare defaults to a full-height curve view with spectrum inspection collapsed. The inspection checkbox restores the spectrum and cursor without recalculating P2P. All comparison curves use equal linewidth and opacity. Field-only legends distinguish duplicate fields with numbered suffixes and available averaging counts.

Save PNG / PNG as in Compare build a separate one-axis Figure from the calculated records. No interactive cursor, raw spectrum, selection emphasis or analysis overlays enter this figure. Export inspection PNG explicitly saves the diagnostic view and restores the previous inspection visibility afterward. Single-file PNG keeps its diagnostic content.

The collapsed Plot style / Export panel controls width/height in inches, DPI, axis/tick/title/legend font sizes, linewidth, marker size/visibility, legend location/columns, title and Energy/SG subtitle. Defaults: 9×5 inches, 300 DPI. Presentation edits do not run P2P calculations. Settings are carried across groups through the owner and stored in workspace/P2P snapshots.

Verification: 42 related tests passed. Additional persistence assertions passed for restored width and legend font size. Export tests verify exactly one comparison axis, field-only labels, equal curve styling, collapsed/expanded inspection, settings restoration and a 2250×1500 image for 7.5×5 inches at 300 DPI. Visual inspection of the user's exported 1.8350–1.8600 eV / SG21points / degree-2 CSV confirmed eight curves (0–7 T) with no diagnostic panel. Example: artifacts/drr-comparison/p2p-clean-comparison.png.

## Magnetic-field colors and optional offset

Screen/export curves now share a truncated plasma mapping based on numeric magnetic field, with fixed editable bounds (default 0–7 T). Removing records cannot renormalize the colors. Repeated fields share a color and receive different markers/linestyles. Legends sort by field. Values outside the fixed bounds produce a visible instruction to adjust the color limits.

Offset is disabled by default. When enabled, each successive curve in field order is displaced by the configured step; the ordinate and PNG filename identify the displacement. CSV and source result arrays remain unchanged. Curve picking includes the displayed displacement when resolving the selected point. Tests cover color stability after removal, repeated-field markers, sorted export order and unchanged result arrays. Actual-data examples: artifacts/drr-comparison/p2p-plasma-comparison.png and p2p-plasma-offset.png.
