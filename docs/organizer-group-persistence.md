# Organizer group persistence

## Fixed group count

In the Conditions panel, enable **Fixed group count** and set N (1–100).
The setting belongs to this comparison series. Assign group and Edit groups
then offer Group 1–N, including empty groups. Existing assignments outside
this set become individually unassigned rather than merged. New records
join only an unambiguous existing group; otherwise they remain unassigned.
Refresh and reopen retain the cap. Disabling the cap allows new group names
again but does not automatically restore previously removed assignments.

Energy groups are saved independently for each comparison variable, fixed
conditions and integration-window width. Refresh preserves existing names
and assignments. New records join an unambiguous matching energy group or
receive an appended number. Hiding records does not renumber groups.

Grouping tolerance affects new assignments. Use **Reset groups** to explicitly
regroup the current series. Existing exclusions survive new series members;
new records start included. Pending selections are flushed on Refresh/close.

Each window saves changes relative to its loaded snapshot. An older window's
refresh or close cannot overwrite another window's newer manual assignments.
Independent include/exclude edits are merged by record ID. Saving uses a file
lock and atomic replacement; failures appear in the status bar. If two windows
explicitly change the same assignment, the later save wins.

The Organizer selection file uses schema 3. Legacy global assignments are
snapshotted independently into each context on first use. Their original
comparison cannot be recovered from schema 2; reset an individual context if
its inherited grouping is inappropriate. Subsequent edits remain independent.

Tests cover adding lower-energy records, independent comparison variables,
separate fixed conditions, nearby distinct partitions, scan/reopen persistence
and preservation of exclusions.

## Export folders

The selected export root contains `Temperature_dependence` for temperature
comparisons and `Efield_dependence` for electric-field comparisons. This also
applies when multiple series of the same comparison type are selected. CW
fit folders are saved under `Temperature_dependence`. Selecting the type folder
itself does not create a duplicate nested folder. Existing exports are not moved.
