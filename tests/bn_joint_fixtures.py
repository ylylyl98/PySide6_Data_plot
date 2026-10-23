"""Transport-only evidence fixture; numerical acceptance has separate tests."""
from copy import deepcopy


def accepted_joint_report(window=21, polyorder=2, method='local_shift'):
    family, strength = method, 1.
    # The withdrawn 12.5% trial remains constructible for legacy-rejection tests.
    for suffix, fraction in (('_s12p5', .125), ('_s25', .25), ('_s50', .5), ('_s75', .75)):
        if method.endswith(suffix):
            family, strength = method[:-len(suffix)], fraction
    final = dict(repeat_indices=[2, 3], original_rms=[1., 1.], candidate_rms=[.5, .5],
                 mean_original_rms=1., mean_candidate_rms=.5, mean_improvement_fraction=.5)
    selection = dict(final, repeat_indices=[1], original_rms=[1.], candidate_rms=[.5])
    checks = dict(passed=True, derivative_checked=True, drr_benefit_passed=True,
                  drr_noninferiority_passed=True, d2_benefit_passed=True)
    training = dict(policy='d2_shrinkage_v1', metric='BN-projected production D2 residual',
                    d2_training_evaluated=True, fixed_bn_basis=True,
                    basis_source='BN-only shift/gain tangent responses', basis_rank=2,
                    context='full training-repeat spectra; score band is not an independent energy holdout',
                    repeat_indices=[0], gate_indices=[0, 1], channel_indices=list(range(max(101, window))),
                    score_channel_indices=list(range(window)),
                    bandwidth_check=dict(passed=True, scoring_energy_span=(window-1)*.001,
                                         sg_footprint_energy=(window-1)*.001,
                                         score_channel_count=window, total_score_channel_count=window,
                                         required_channel_count=window),
                    used_window_length=window)
    training['fit_bands'] = [dict(region=[1.9, 1.9+(window-1)*.001],
                                 channel_indices=list(range(window)), channel_count=window,
                                 energy_span=(window-1)*.001)]
    protection = dict(passed=True, derivative_checked=True, drr=dict(passed=True, checked=True),
                      d2=dict(passed=True, checked=True))
    errors = dict(max_drr_error_fraction=.01, max_d2_error_fraction=.01)
    positions = [1.89+i*.006 for i in range(17)]
    probes = [dict(probe_index=i, probe_source='coarse', center_energy=positions[i//4],
                   sigma_energy=(1.5, 3.)[(i % 4)//2]*.001,
                   sigma_native_spacings=(1.5, 3.)[(i % 4)//2],
                   sign=(-1 if i % 2 == 0 else 1), drr_error_fraction=.01, d2_error_fraction=.01)
              for i in range(68)]
    used = {(p['center_energy'], p['sigma_energy'], p['sign']) for p in probes}
    refined = []
    for seed in (0, 1, 2):
        for offset in (-1., -.5, 0., .5, 1.):
            center = min(max(positions[seed]+offset*.001, 1.88), 2.)
            for scale in (1., 1.25, 1.5, 2., 2.5, 3.):
                for sign in (-1, 1):
                    key = center, scale*.001, sign
                    if key not in used:
                        refined.append(dict(probe_index=len(refined), probe_source='refinement',
                            center_energy=center, sigma_energy=scale*.001, sigma_native_spacings=scale,
                            sign=sign, seed_center_index=seed, center_offset_native_spacings=offset,
                            drr_error_fraction=.01, d2_error_fraction=.01))
                        used.add(key)
    chosen = [probes[0], refined[0], refined[1]]
    references = [dict(source=p['probe_source'], probe_index=p['probe_index']) for p in chosen]
    refits = [dict(probe, passed=True, optimizer_success=True, family=family,
                   correction_strength=strength) for probe in chosen]
    audit = dict(schema_version=3, policy='training_feature_influence_shrinkage_v2', passed=True,
                 evaluated=True, family=family, correction_strength=strength,
                 fit_jacobian_background='unshrunk fitted BN', forcing_background='unshrunk fitted BN',
                 output_response_background='fixed-alpha shrunk BN', output_response_strength=strength,
                 probe_mode='common_to_all_training_repeats_and_gates',
                 training_only=True, selected_method=method, train_repeat_indices=[0],
                 fit_gate_indices=[0, 1], checked_gate_indices=[0, 1], amplitude=2e-4,
                 signs=[-1, 1], sigma_native_spacings=[1.5, 3], position_count=17,
                 positions_energy=positions, native_median_spacing=.001, max_relative_error=.05,
                 screening=dict(errors, passed=True, linear_approximation=True, probes=probes),
                 refinement=dict(errors, schema_version=1, policy='native_feature_grid_v1',
                    evaluated=True, passed=True, max_seed_centers=3, seed_center_indices=[0, 1, 2],
                    seed_centers_energy=positions[:3], center_offsets_native_spacings=[-1., -.5, 0., .5, 1.],
                    sigma_native_spacings=[1., 1.25, 1.5, 2., 2.5, 3.], signs=[-1, 1],
                    center_bounds_energy=[1.88, 2.], max_probe_count=168,
                    probe_count=len(refined), expected_probe_count=len(refined), probes=refined),
                 actual_refits=dict(errors, passed=True, count=3, probes=refits,
                                    chosen_probe_references=deepcopy(references)))
    selector = dict(errors, schema_version=2, policy='pre_final_selection_sensitivity_v2',
                    passed=True, selection_aware=True, pre_final=True, final_repeats_read=False,
                    training_only=False, train_repeat_indices=[0], selection_repeat_indices=[1],
                    baseline_selected_method=method, baseline_family=family,
                    baseline_correction_strength=strength, baseline_qualified_candidates=[method],
                    amplitude=2e-4, max_relative_error=.05, count=3, chosen_probe_references=deepcopy(references),
                    probes=[dict(probe, passed=True, baseline_selected_method=method,
                                 perturbed_selected_method=method, perturbed_family=family,
                                 perturbed_correction_strength=strength,
                                 original_fallback_jump_included=False) for probe in chosen])
    parameters = dict(family=family, correction_strength=strength, shift_pixel=.1, fine_gain=1.,
                      training_objective=training,
                      unshrunk_parameters=dict(shift_pixel=.1, fine_gain=1.,
                                               optimizer_success=True, near_parameter_bound=False),
                      unshrunk_background=dict(sha256='a'*64, point_count=max(101, window),
                                               log_correction_rms=.001, max_absolute_log_correction=.002),
                      shrinkage=dict(post_fit=True, alpha_is_fitted=False,
                                     training_objective_background='unshrunk fitted BN'))
    return dict(schema_version=4, optimization_policy='d2_shrinkage_v1',
                eligible=True, status='accepted', selection_frozen=True,
                selected_method=method, applied_method=method,
                derivative_settings=dict(window_length=window, used_window_length=window, polyorder=polyorder,
                                         processing='production SG energy derivative'),
                split=dict(train=[0], selection=[1], final=[2, 3], independent_final_confirmation_available=True),
                joint_validation=dict(schema_version=3, policy='d2_shrinkage_v1',
                                      d2_training_evaluated=True, fixed_bn_basis=True,
                                      influence_audit=audit,
                                      selection_sensitivity_audit=selector,
                                      passed=True, derivative_checked=True,
                                      selection=deepcopy(checks), final=deepcopy(checks),
                                      protection={role:deepcopy(protection) for role in ('training','selection','final')}),
                final_confirmation=deepcopy(final), final_confirmation_d2=deepcopy(final),
                candidates={'original':dict(selection=dict(selection, mean_improvement_fraction=0.),
                                             selection_d2=dict(selection, mean_improvement_fraction=0.)),
                            method:dict(parameters=parameters, training_influence_audit=deepcopy(audit),
                                        selection=deepcopy(selection), selection_d2=deepcopy(selection),
                                        selection_joint=dict(checks, reasons=[]))})
