# Automatic External background selection

Selecting External without a chosen background now starts background matching
on a Qt worker and then loads DRR automatically. Existing manual selections and
pinned backgrounds remain authoritative. Changing an unpinned measurement
selection rematches it. Failure keeps External selected and opens the manual
background picker with a reason in the status bar.

The resolver first reuses a valid saved External assignment for each measurement.
Otherwise it uses the source picker's deterministic recommendation ranking after
exact spectral-grid compatibility checks, then loads all compatible files in the
top recommendation's existing session/condition/gate group. Selected measurements are excluded as
background candidates. Confirmed constant-gate recommendations average all frames;
other recommendations use the last frame. Saved frame methods are preserved.
Recommendation reasons travel with load assignments and appear in the background
tooltip; the common background filename also appears in the summary.

Matching uses a catalog snapshot and a single saved-history scan. Missing catalog
entries are inspected in the worker. Cancellation and current-selection checks
discard stale results after mode, folder, measurement or manual-background changes,
and on shutdown. Frame-method and pin controls are disabled while matching.
Catalog refresh preserves accepted recommendations, including saved backgrounds
outside the filtered catalog, and restarts matching when new selected repeats arrive.

Validation covers ranking, saved recipe precedence, grid rejection, source exclusion,
thread affinity, manual/pinned overrides, stale completion, failed matching, catalog
refresh and history-cache reuse. The core/new-UI/cache suite passed 82 tests;
the background workflow/picker suites passed 18, and the background UI suite
passed 15 (115 total). Combined GUI execution stalled in repeated application
theme initialization, so the GUI suites were verified in separate processes.

A synthetic cached catalog with one measurement and 200 candidate files took
149.7, 154.0, 147.7, 174.5 and 153.8 ms to resolve (median 153.8 ms). This measures
matching only on this machine; it excludes discovery and DRR data loading and is
not a latency guarantee for the user's datasets.

The group-loading follow-up fixes the original singleton `baseline_files` assignment.
Regression coverage includes two same-condition repeats, exclusion of incompatible
spectral grids and different constant gate values, and every selected group member
reaching the UI load options. Saved/manual file selections remain authoritative.
The focused automatic-selection, candidate, source and background-numerics suites
passed 82 tests after this follow-up.
