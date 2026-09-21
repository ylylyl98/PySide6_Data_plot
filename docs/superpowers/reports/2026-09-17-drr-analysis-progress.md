# Analysis progress and staged preview

Connected worker progress/log signals to a visible progress bar, current-file
message, elapsed timer, cancel action and persistent completion/error/cancelled
status. Numerical analysis reports row/product completion; seed tracking reports
visited rows and finishes at 100% when its connected traversal completes.
Duplicate percent updates are suppressed before emitting Qt signals.
Loading reports completed files, with cancellation between files. Export shows
an indeterminate busy indicator; cancellation is disabled while writing exports.

Parameter editing no longer unconditionally rebuilds the colorplot:
- numeric keyboard input commits on Return/focus departure;
- detection parameters invalidate results and debounce marker refresh only;
- X/Y bounds debounce the existing plot redraw by 300 ms;
- SG parameters are staged until Update preview, or a successful analysis applies
  its actual settings. Plot titles retain the SG settings used by the displayed
  data, so a staged setting cannot mislabel the preview;
- Y inspection updates existing spectrum and horizontal lines without recreating
  heatmap axes. Derivative cache remains keyed to actual preview settings.

Busy analysis actions are disabled, cancellation resets before a new task, and
dataset instance/revision checks still reject stale results. Batch parameter
application redraws comparison plots to remove obsolete target markers.

Validation: 60 relevant core/UI/export/integration tests passed, followed by a
focused window/progress rerun after the compare-overlay fix. Added regression
coverage for progress intermediates/completion, no detector-triggered map rebuild,
staged SG, stable axes on Y changes, cancel/restart, and comparison invalidation.
Native light-theme screenshot inspected. Independent review found no remaining
blockers after the compare-overlay correction. No executable packaging or restart.
