# Find first, filter afterward

The workspace now starts with Find peaks. Full data is the default calculation
scope, with explicit Current display and Custom calculation alternatives.
Calculation bounds, source, polarity, SG and seed tools are under Advanced.
The result-filter section is disabled until results exist and opens after a
successful detection. Redetection starts with all candidates retained; no
metric exclusion or branch linking is silently applied.

Detection already runs on the background queue with row progress and cancellation.
Neutral candidate display now shares the immutable pool instead of deep-copying
and relinking it; the controller also avoids an extra full-pool copy on hydration.
Only 500 table rows are built per page. For pools over 20,000 retained points,
overlays initially turn off with a visible notice and can be explicitly enabled
in the filtering section. No candidate is discarded by this display policy.

Read-only full-data measurement on the YZ365 1T 720nm BG=18 saved group:
157 x 1340 samples, 162,167 raw + derivative candidates, 2.751 seconds in
numeric detection and 3.883 seconds from Find peaks to a ready Qt result view.
Timing is specific to this dataset and machine. Candidate detection still
does not identify scientifically validated peaks; the user filters and checks
the spectra afterward. Seed mode remains a separate targeted workflow.

Validated the full-range default, explicit display-range selection, zero-copy
neutral view, existing filtering/range behavior, process restart, and exports.
