# MCD display review — 2026-09-12

## Astra review

Bounded review completed against `artifacts/mcd-display-baseline/mcd_unified_page.py` and `main_window.py`, not Git HEAD. Reviewed only the display change, its new style tests, and relevant existing drawing/center-update checks. No production code was edited by the reviewer.

**Verdict:** no remaining blocking code findings after Luna's final fixes. Root owns final native-window visual acceptance, particularly slope-label legibility at 1280×720.

Findings resolved during review:

- Updated visible legend text with each channel's measured B; animated the entire legend so old B labels do not remain in the blit background. Independent draw spying reproduced the initial child-text defect and verified its resolution.
- Kept physical channel colors and corrected mapped/unknown channel labels, production `raw pos`/`raw neg` identities, and splitting's `E_K−E_K′` label.
- Changed quick metric/Overlay actions to redraw only cached feature data, preserving other axes and B ranges. A changed quantity gets an appropriate y range; Overlay retains the current primary y range.
- Used the actual cached splitting energy fields with eV axes, and displayed the specific splitting-unavailable reason.
- Removed the new fallback's incorrect identification of sweep direction from B sign.
- Kept fit slope/intercept and original endpoints unchanged, distinguished clipped half-span extrapolation, added region/branch slope labels, removed hidden-branch fit/extension/annotation artists, and clipped slope text to its axes.
- Preserved existing snapshot fields and recorded Overlay alongside the metric.

Independent verification:

- Initial frozen change: 11/11 tests passed in 11.4 s (`tests.test_mcd_display_styles`, `tests.test_mcd_render_followup`, and the existing center-refresh fast-path test).
- Final legend/clipping revision: 7/7 style tests passed in 6.0 s.
- Additional focused assertions passed for production physical-channel normalization and reversed mapping, feature identity, cached button actions without full render, retained axes/B limits, appropriate energy limits, unchanged numerical fits and clipping limits, and removal of all hidden fit artists.
- Final draw assertions passed: the animated legend text is absent from static capture, visible legend B text matches its source line, hidden spectra remain hidden after a full draw, and slope text has an axes clip.

Limits: no broad/core/export-renderer redesign review or full suite was performed. Final native screenshots, annotation overlap/legibility, and real-data timing are root's separate checks. Text clipping prevents gutter drawing but is not collision avoidance; dense fits and narrow windows still require visual inspection. Existing numerical support metadata and automatic association paths were not changed by this display diff.

## Final slope-label follow-up

Root's native inspection found that clipping alone cut the high-negative prefix and high-positive standard error. Luna subsequently gave high-negative labels left alignment, high-positive labels right alignment, and low-field labels centered alignment, with region-specific inset anchors and the existing clip as a safety boundary. Review confirms this changes annotation placement only; numerical fit segments are unaffected.

Independent rerun of `tests.test_mcd_display_styles`: **8/8 passed in 6.6 s**, including the six-annotation horizontal-bounding-box regression. The clipping-only concern is closed at code-review level. Final real-data native visual acceptance remains root's check; the bounding-box test does not establish general collision avoidance or vertical containment for arbitrary data/ranges.

## Root final native acceptance
Real YZ365 data exercised after the final alignment fix. Native 1280x720 and 1600x1000 images inspected: channel and branch styles, measured markers, six complete slope annotations, fit extensions, and metric buttons are visible. Small-window labels remain dense but no longer lose region/branch or uncertainty at the axes boundary. Automatic MCD association and Spectrum tracking remain functional. Earlier consolidated 30 tests passed; Astra independently passed final 8 display tests. Full-size native preview: artifacts/unified-mcd-baseline/native-1600x1000.png. Existing independent saved-PNG renderer was outside this screen-display update; numerical export formats were preserved. No live app restart or experimental source write performed.
