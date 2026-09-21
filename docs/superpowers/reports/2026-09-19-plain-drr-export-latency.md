# Plain DRR export latency

The user confirmed plain DRR export with no peak analysis: the UI remains
responsive but files take a long time to appear. The peak-workbook optimization
does not address this path.

Profiled the current export worker using a saved YZ365 157 x 1340 raw matrix,
its real measurement/background source paths and 638 existing metadata records.
All new PNG/DAT/metadata files were written into a temporary directory, with
copies of the existing metadata; the real measurement folder was read-only.

| Stage | Before (s) | After (s) |
| --- | ---: | ---: |
| Full worker | 5.934 | 2.381 |
| First history lookup | 4.549 | 1.047 |
| Second history lookup | 0.185 | 0.046 |
| Second derivative | 0.172 | 0.167 |
| Both PNGs | 0.821 | 0.902 |
| Both DATs | 0.177 | 0.192 |

Cold metadata reads dominated this fixture. A separate read-only lookup in the
already warm real history directory took about 0.14 seconds before caching.
The cold-copy timing therefore does not establish every live export's cause
or guarantee a 60% improvement for all datasets.

Changes:

- Prefer the expected metadata filename, still requiring operation and analysis
  fingerprint equality before reusing outputs.
- Cache unchanged JSON records and legacy fingerprints by absolute path, file
  size, mtime_ns and ctime_ns, bounded to 4096 entries.
- Read large history sets with at most four threads; consume results in filename
  order to preserve deterministic fallback selection.
- Keep stat/glob checks so edits, deletions and new records are observed.
- Report queue delay, derivative calculation, source checks, history search,
  DAT writing, PNG rendering, metadata writing and worker total in the log.
  The status bar follows worker stage messages through queued Qt signals.

No output precision, resolution or products were reduced. 27 tests passed across
history caching, export metadata, paired outputs, actual background save and PNG
rendering. Review also covered malformed unrelated metadata and thread safety.
The measurement script is artifacts/profile_plain_drr_export.py. GUI preparation,
live thread-pool contention and post-save catalog refresh remain outside this
worker benchmark; new stage timings can distinguish these during live use.

## Follow-up: persistent history and avoiding repeated work

- A versioned `.drr-export-index.json` stores compact file signatures and
  analysis fingerprints. Startup still checks metadata filenames and stat
  signatures, but only parses new/changed records and matching results.
  Corrupt indices rebuild; optional cache write failures do not block export.
  Index replacement is atomic. Legacy fingerprint fallback is preserved.
- Save reuses the display's second derivative only for the same loaded cube,
  requested SG window and polynomial order. A miss still computes in the worker.
- Successful DRR saves refresh saved roles/background links from the existing
  raw catalog in a worker, without raw discovery or selection rematching.
  Folder/generation/list guards reject stale publications. Existing scans finish
  before this refresh; actual source cleanup still requests a full scan.

On 640 real history records copied to a temporary folder, first index creation
took 1.3079 s. Clearing the metadata memory cache then looking up an absent
fingerprint took 0.0555 s; another lookup took 0.0401 s. These are history-only
timings, not whole-export speedups. The latest cold full-worker run was 2.988 s
(0.168 s derivative, 1.482 s history, 1.129 s PNG, 0.179 s DAT), using a different
latest saved matrix than the earlier fixture, so it is not a controlled before/
after comparison. No real output directory was modified by these benchmarks.

Verification: 34 tests passed across export history, metadata, paired exports,
GUI save workflow and PNG latency; another 17 catalog/picker tests passed.
New cases cover persistent reuse after clearing RAM, corrupt/read-only index,
changed SG parameters, actual processed-state publication, changed folders,
and the cleanup exception identified during code review.
