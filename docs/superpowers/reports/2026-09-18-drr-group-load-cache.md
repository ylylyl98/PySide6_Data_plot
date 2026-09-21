# DRR group load cache

Loading a saved group previously read its complete NPZ, parsed every DAT again, then read every individual NPZ. Switching groups also compressed outgoing group and per-file snapshots. Batch P2P construction performed a redundant base-class calculation before restoring records.

Changes:
- Record metadata/DAT size, modification time and creation/change time in display state, outside scientific dataset identity.
- Reuse saved group datasets and P2P records only when both source signatures match. Missing signatures or changed sources fall back to the existing loader. Older caches migrate on the next save.
- Opening the unchanged current group preserves its widgets and analysis without a job.
- Internal group/per-file caches use uncompressed atomic NPZ; explicit workspace exports and history snapshots retain compression.
- Skip the base P2P initial calculation when constructing the batch page.
- Report saving and cache restoration stages in the status log.

Read-only measurements on an existing eight-member user group, writes confined to a temporary directory: group-cache load 0.173 s; redundant DAT load 0.531 s. Separate save comparison: compressed 0.700 s / 12.83 MB; uncompressed 0.041 s / 13.69 MB. These are individual operations, not end-to-end GUI load timings.

Regression coverage includes unchanged-group no-op, saved-group reuse without batch recomputation, changed DAT invalidation, membership changes, ranges/results restoration, session round-trip and P2P metrics. Source freshness uses filesystem metadata rather than a full content hash on every load; external tools that preserve all three stat values are outside this fast-path check.
