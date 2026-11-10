"""Step-focused PL controls, using the existing native parameter components."""


def pl_parameter_groups(bridge, legacy, find_legacy):
    from core.peak_analysis_state import analysis_state
    d = bridge.dataset
    state = analysis_state(d)
    page = bridge._controls_page
    editable = not bridge._busy
    fields = {f['key']: f for group in legacy for f in group['fields']}
    find_fields = {f['key']: f for group in find_legacy for f in group['fields']}
    def summary(key, text):
        return dict(key=key, type='summary', value=text)
    def action(key, text, enabled=True):
        return dict(key=key, type='action', label=text, enabled=editable and enabled)
    def choice(key, text, value, options):
        return dict(key=key, type='choice', label=text, value=value, enabled=editable,
                    options=[dict(value=v, label=t) for v, t in options])
    def group(key, title, content, **extra):
        return dict(key=key, title=title, fields=content, **extra)
    if page == 'find':
        basic = ['analysisScope', 'target-note', 'scope', 'x_min', 'x_max', 'y_min', 'y_max']
        content = [fields[k] for k in basic if k in fields]
        for f in content:
            if f['key'] in ('x_min', 'x_max', 'y_min', 'y_max'):
                f['visible'] = d.get('scope', 0) == 2
        content.append(summary('find-explanation', 'Find peaks first. Check and adjust the results in step 2.'))
        if d.get('manual_peak_edits'):
            content.append(summary('edits-preserved', 'Your added, moved and excluded peaks are preserved when finding again.'))
        keys = ['detection_method', 'smoothing_window', 'pl_weak_features', 'candidate_floor', 'method-note']
        return [group('find', 'Data and range', content),
                group('advanced', 'Detection options', [fields[k] for k in keys], advanced=True)]
    if page == 'branches':
        # Only the frequent actions stay open; detailed inputs are optional.
        main = [summary('review-help', 'Drag a peak to move it. Double-click to add; Delete to remove. Current spectrum only.'),
                fields['manual.target'], fields['manual-pick'], fields['manual-exclude']]
        fields['manual-pick']['label'] = 'Cancel adding' if bridge._pick_seed else 'Add peak'
        fields['manual-exclude']['label'] = 'Remove selected peak'
        if state['dirty']:
            for f in main:
                if f.get('type') != 'summary': f['enabled'] = False
        out = [group('review-guide', 'Check peaks', main)]
        filter_fields = [find_fields[k] for k in ('filterPreset', 'auto-filter', 'min_snr', 'prominence', 'min_width_mev',
            'min_distance_mev', 'max_width_mev', 'max_per_row', 'neighbor_support', 'neighbor_tolerance_mev', 'showRejected')]
        filter_fields += [action('filter', 'Apply filters', bool(d.get('candidates')) and not state['detection_dirty']),
                          action('reset-filters', 'Clear quality filters')]
        out.append(group('review-filters', 'Adjust filters', filter_fields, advanced=True))
        exact = [fields[k] for k in ('manual.energy', 'manual.width_mev', 'manual.move_limit_mev', 'manual.fixed',
                    'manual-move', 'manual-add', 'manual-restore', 'manual-constraint-note')]
        if state['dirty']:
            for f in exact: f['enabled'] = False
        out.append(group('manual-constraints', 'Exact position / constraints', exact, advanced=True))
        out.append(group('whole-branches', 'Whole-map branch selection', [
            dict(key='showWholeBranches', type='flag', label='Show all branches', value=d.get('show_whole_branches', False), enabled=editable),
            summary('whole-branch-help', 'Unchecking a branch excludes it across all spectra. Use Exclude selected peak here for one spectrum.')], advanced=True))
        out[-1]['fields'].extend(fields[k] for k in ('max_shift_mev', 'max_gap') if k in fields)
        return out
    if page == 'fit':
        return [group('fit-setup', 'Fit the checked peaks', [
            summary('fit-scope', 'Fits your checked peaks. No new peak search.'),
            summary('fit-count', f"{state['accepted_count']} accepted positions; " + ('current spectrum.' if d.get('analysis_scope') == 'spectrum' else 'all spectra in the selected range.')),
            choice('model', 'Peak shape', d['settings']['model'], [(v, v) for v in ('Lorentzian', 'Gaussian', 'Asymmetric Lorentzian')]),
            summary('fit-initial', 'Start with Lorentzian. Check the residual after fitting.'),
            dict(key='showResidual', type='flag', label='Show residual below spectrum', value=d.get('show_residual', False), enabled=editable)])]
    fits = (d.get('result') or {}).get('fits', [])
    current = next((f for f in fits if f['row_index'] == d['row']), None)
    fit_status = ((current.get('fit_quality') or {}).get('status') or current.get('status')) if current else 'Not fitted'
    return [group('result-summary', 'Inspect the results', [
        summary('result-counts', f"{state['accepted_count']} checked positions\n{state['fitted_count']} valid fitted centers in this selection"),
        summary('current-fit', 'Current spectrum: ' + fit_status),
        choice('overlayPosition', 'Positions on map', d.get('overlay_position', 'detected'), [('detected', 'Detected / edited'), ('fitted', 'Valid fitted centers')]),
        dict(key='showResidual', type='flag', label='Show residual', value=d.get('show_residual', False), enabled=editable),
        action('results', 'Open result table', bool(d.get('result')))]),
        group('save-session', 'Continue later', [action('save-copy', 'Save editable session…')]),
        group('save-results', 'Export / display options', [
        choice('analysisPlot', 'Plot', d.get('plot_view', 'spectra'), [('spectra', 'Map and spectrum'), ('trends', 'Peak centers and widths')]),
        choice('exportCurveScope', 'Spectra included in export', d.get('export_curve_scope', 'current'), [('current', 'Current spectrum'), ('all', 'All analyzed spectra')]),
        summary('export-help', 'Export includes all peak rows. This option controls only the spectra.'),
        action('apply-overlay', 'Show positions in main app', state['can_export']),
        summary('session-help', 'The results package contains XLSX, CSV, settings and PNG figures.')], advanced=True)]
