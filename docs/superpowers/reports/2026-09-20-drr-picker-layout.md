# DRR picker layout optimization

Preserved all-history visibility, complete session filenames, wrapping and
content-dependent row heights. Hidden population no longer forces an extra
layout; showing the list still forces final-width layout before painting.
Replaced the 512-entry clear-all size cache with a bounded 8192-entry LRU and
memoized filename source classification within each dialog.

DRR session heights persist as `drr-picker-row-heights.json` under Qt's local
CacheLocation. Keys include text, available width, font, font metrics, DPI,
device pixel ratio, style class, Qt version and padding. Invalid JSON rebuilds;
optional write failures are ignored. Writes use atomic replacement on close.
Only hashes and dimensions are stored, not filenames themselves.

The same 621-record catalog used in diagnosis measured 0.455 s on initial open
and 0.200 s on cached reopen, versus the earlier 0.567 s baseline under cProfile.
Explicit QListView layout calls fell from four to two. Offscreen measurements
are diagnostic, not a guarantee for native window latency.

Validation: 39 tests passed covering layout cache, picker stability, source
dialog, startup and metadata caching. Reviewer independently passed another
8 layout-cache/filename-delegate tests and found no actionable regression.

`artifacts/verify_drr_picker_layout.py` captures the actual Qt widgets using
720 synthetic files in 240 groups, with three long filenames per group.
Screenshots at 1120, 920 and 1300 logical pixels, at scale factors 1 and 1.5,
were generated under `artifacts/drr-picker-layout-check`. Explicit Segoe UI
registration makes the offscreen font renderer readable. Both narrow screenshots
were visually inspected: all three filenames in the complete first row end
in `.csv`. Each scale recorded 32 painted row checks and zero insufficient row
heights, including reopen from JSON. Partial rows at the scroll viewport edge
are normal scrolling; their full allocated heights were checked separately.
