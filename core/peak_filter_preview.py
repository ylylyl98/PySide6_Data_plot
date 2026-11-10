"""Read-only explanations of the last applied candidate filters."""
REASONS = {
    'snr': 'SNR below threshold or unavailable',
    'prominence': 'Prominence below threshold',
    'min_width': 'Peak narrower than minimum width',
    'max_width': 'Peak wider than maximum width',
    'neighbor': 'Insufficient neighboring-spectrum support',
    'trajectory': 'No trajectory or local support; SNR below protection threshold',
    'spacing': 'Too close to a more prominent candidate',
    'limit': 'Maximum features per spectrum reached',
}
REASON_LABELS = dict(snr='SNR', prominence='Prominence', min_width='Min width',
                    max_width='Max width', neighbor='Neighbor support', trajectory='Isolated cluster', spacing='Spacing', limit='Count limit')


def quality_rejection(point, settings):
    """First failing quality test; coordinates and polarity are checked separately."""
    minimum = settings.get('min_snr', 0)
    if minimum and (point.get('snr') is None or not point['snr'] >= minimum):
        return 'snr'
    if not point['prominence_fraction'] >= settings['prominence']:
        return 'prominence'
    if not point['width_mev'] >= settings['min_width_mev']:
        return 'min_width'
    if settings.get('max_width_mev', 0) and not point['width_mev'] <= settings['max_width_mev']:
        return 'max_width'
    return None


def in_filter_range(point, settings):
    return (settings['x_min'] <= point['energy'] <= settings['x_max']
            and settings['y_min'] <= point['y'] <= settings['y_max']
            and (settings['polarity'] == 'both' or point.get('polarity', '') + 's' == settings['polarity']))


def excluded_candidates(dataset):
    """Preview only applied range/type, never draft settings or hidden branches."""
    result = dataset.get('result') or {}
    pool = dataset.get('candidates') or {}
    if not result:
        return []
    reasons = result.get('filter_preview', {}).get('reasons')
    if reasons is not None:
        return [p for p in pool.get('points', []) if p['candidate_id'] in reasons]
    # Old saved workspaces have no explanations; limit their legacy preview.
    kept = {p.get('candidate_id') for p in result.get('points', [])}
    settings = result.get('settings')
    return [p for p in pool.get('points', []) if p['candidate_id'] not in kept
            and (not settings or in_filter_range(p, settings))]


def preview_summary(result):
    if result and 'filter_preview' not in result:
        return 'Reasons and counts are unavailable for this saved result. Apply filters again to generate explanations.'
    counts = (result or {}).get('filter_preview', {}).get('counts', {})
    if not counts:
        return 'No excluded candidates in the applied range/type.'
    return (f'Excluded: {sum(counts.values()):,} in applied range/type. First failing condition: ' +
            ' · '.join(f'{REASON_LABELS.get(key, key)}: {count:,}' for key, count in counts.items()) +
            '\nFaded × markers are excluded. Hover near a marker for its reason.')


def scatter_summary(result):
    """Describe actual applied cleanup, including conservative fallback."""
    status = (result or {}).get('scatter_filter')
    if not status:
        return ''
    if not status.get('requested'):
        return 'Isolated cluster removal: off in the last applied filters.'
    if status.get('applied'):
        count = result.get('filter_counts', {}).get('trajectory_removed', 0)
        return (f"Isolated cluster removal: {count:,} points excluded. "
            f"{status.get('local_protected_count', 0):,} candidates outside long trajectories have local support. "
            f"Candidates with SNR ≥ {status.get('protected_snr', 8):g}, or unavailable SNR, are protected at this stage. "
            'Earlier quality filters still apply. Disable Remove isolated clusters and apply to restore these points.')
    reason = status.get('reason')
    detail = 'single-spectrum analysis' if reason == 'single_spectrum' else (
        'unsupported product' if reason == 'unsupported_product' else 'insufficient trajectory evidence')
    return f'Isolated cluster removal skipped: {detail}. No additional points removed.'
