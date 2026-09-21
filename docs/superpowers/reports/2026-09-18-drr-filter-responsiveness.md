# Separate range semantics and responsive result filtering

Display ranges now control plotting only. Detection ranges define the candidate
pool and invalidate candidates when changed. An explicit copy button applies
the displayed range to detection. The result filter selects existing candidates
without changing either range, and displays the candidate pool bounds. Old
workspace display/detection links are no longer automatically restored.

Filtering runs through the background job queue after a short debounce. Late
results are discarded when the active dataset or requested filter changes.
The visible table is paginated at 500 rows; all candidates remain available,
pick navigation can move to the relevant page, and exports include all retained
points. Overlay tracks are grouped once instead of rescanning all points for
each track. Numeric Y comparisons avoid per-point NumPy calls.

New data defaults to no prominence/width/spacing/count exclusion. Branch linking
is opt-in; a numerical track-step value is not evidence of a scientifically
validated setting. Explicit saved filters remain saved preferences.

Measured on 33,723 local YZ365 candidates: the toggle handler returned in
0.000129 seconds; complete background filtering/table/overlay work took 1.836
seconds before the final scalar Y comparison optimization. This is not a
claim that all datasets filter instantly. The regression suite covers candidate
preservation, asynchronous filtering, ranges, exports and workspace restoration.
