# DRR first-frame row layout follow-up

The user confirmed clipping occurs in the left Measurement sessions / Data history list, followed by increased row spacing after a few seconds. The earlier unchanged-catalog reset guard did not fix this case.

## Evidence

- Inspected the user's running DPTK app and opened its DRR picker. The first selected item occupied enough space, while later three-line items overlapped the next row. This reproduced the visible clipping in the live app.
- Prior probes did not consistently reproduce it. Some opened before catalog arrival, and screenshot/visualItemRect calls can themselves force pending layout. These are insufficient alone to rule out startup clipping.
- The left list used WrappedFilenameDelegate but inherited Fixed resize mode and explicitly disabled wordWrap with the compact summary lists.

## Change

Use DrrSessionList for the left list, enable WordWrap and Adjust resizing, and synchronously finish item layout after show and width resize events. This removes reliance on a later catalog rebuild to correct row heights. Measurement and baseline dialogs share this list. Compact middle/right summaries are unchanged.

## Verification

- 19 DRR startup, cached preview, stability and filename delegate tests passed.
- New regression opens a preloaded catalog in the natural modal event loop; it observes painting before any grab/visualItemRect probe and checks sufficient, stable row heights across unchanged completion.
- Real cached catalog trace over 0–5 seconds after the fix: first-row paint rect height 60, required text height 60, stable across observations.
- Independent review and an 80-row/six-width probe found no layout recursion, ownership regression or undersized rows.
- The initial clipping remains intermittent in isolated probes, so live observation in the existing process is the pre-fix evidence. That process must restart to load the new list class.
