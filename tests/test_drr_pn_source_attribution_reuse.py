"""Exact attribution parity when stage-invariant source bands are reused."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_pn_reconstruction import source_v20 as v20, source_v24 as v24


def inputs(*, with_events=True):
    e = np.linspace(1.45, 2.15, 351)
    parameters = np.array([[1.60, .004, .04, -.06],
                           [1.80, .005, -.03, -.08],
                           [2.00, .003, .02, -.04]])
    features, _ = v20.features_function(e, parameters.ravel(), np.ones((3, len(e))))
    common = 5 + .02*(e-1.8) + .01*(e-1.8)**2
    logs = np.stack((common+features[0]+.5*features[2],
                     common+features[1]-.5*features[2])) if with_events else np.tile(common, (2, 1))
    pn = np.exp(logs)
    n = len(e)
    weights = np.full((2, n), .5)
    background = np.exp(np.log(pn).mean(0))
    baseline = dict(energy=e.copy(), background=background,
        v11_binary_background=background.copy(), raw_geometric_mean=np.sqrt(pn[0]*pn[1]),
        source_feature_log=np.zeros((2, n)), source_usable_masks=np.ones((2, n), bool),
        residual_intervention_native_log=np.log(pn[0]/pn[1]),
        residual_intervention_delta_log=np.zeros(n), residual_source_weights=weights,
        residual_source_exposure_log_squared=np.zeros((2, n)), differential_model_log=np.zeros(n),
        component_feature_log=np.empty((0, n)), component_source=np.empty(0, int),
        component_applied=np.empty(0, bool), component_tapers=np.empty((0, n)),
        new_source_feature_log=np.zeros((2, n)), new_component_feature_log=np.empty((0, n)),
        new_component_fit_mask=np.empty((0, n), bool), new_component_nuisance_log=np.empty((0, n)),
        new_component_source_coefficients=np.empty((0, 2, 2)),
        new_component_band_features=np.empty((0, 2, n)),
        new_component_source_targets=np.empty((0, 2, 2, n)),
        new_component_source=np.empty(0, int), new_component_applied=np.empty(0, bool))
    return e, pn, baseline, {"components": [], "residual_pass_components": []}, features


class SourceAttributionReuseTests(unittest.TestCase):
    def assert_exact(self, actual, expected, path="root"):
        if isinstance(expected, np.ndarray):
            self.assertEqual(actual.dtype, expected.dtype, path)
            np.testing.assert_array_equal(actual, expected, err_msg=path)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected), path)
            for key in expected:
                self.assert_exact(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected), path)
            for index, value in enumerate(expected):
                self.assert_exact(actual[index], value, f"{path}[{index}]")
        else:
            self.assertEqual(actual, expected, path)

    def test_standalone_compatibility_and_prepared_targets_preserve_decisions(self):
        e, pn, baseline, _, features = inputs()
        logs = np.log(pn)
        targets = v20._prepare_source_targets(e, logs)
        before = [value.copy() for value in (e, logs, *targets)]
        for value in (e, logs, *targets):
            value.setflags(write=False)
        labels = []
        for center, feature in zip((1.60, 1.80, 2.00), (features[0], -features[1], features[2])):
            mask = abs(e-center) <= .025
            old = v20._attribute_event(e, logs, feature, mask, baseline["source_usable_masks"])
            new = v20._attribute_event(e, logs, feature, mask, baseline["source_usable_masks"],
                                       prepared_targets=targets)
            self.assert_exact(new, old)
            labels.append(new[0])
        self.assertEqual(labels, [0, 1, -1])
        for value, original in zip((e, logs, *targets), before):
            np.testing.assert_array_equal(value, original)

    def _assert_stage_parity(self, module):
        e, pn, baseline, diagnostics, _ = inputs()
        originals = copy.deepcopy((e, pn, baseline, diagnostics))
        # None selects the original per-event Q calls, without depending on
        # external artifact fixtures or changing the actual fit/detection code.
        with patch.object(v20, "_prepare_source_targets", return_value=None), \
             patch.object(v20, "Q", wraps=v20.Q) as old_q:
            old = module.fit_residual_stage(e, pn, baseline=baseline,
                                           baseline_diagnostics=diagnostics)
        with patch.object(v20, "_prepare_source_targets", wraps=v20._prepare_source_targets) as prepare, \
             patch.object(v20, "Q", wraps=v20.Q) as new_q:
            new = module.fit_residual_stage(e, pn, baseline=baseline,
                                           baseline_diagnostics=diagnostics)
        count = len(new["records"])
        self.assertGreaterEqual(count, 2)
        self.assert_exact(new, old)
        prepare.assert_called_once()
        source_widths = lambda mocked: [call.args[2] for call in mocked.call_args_list
                                       if np.shape(call.args[1]) == (2, len(e))]
        self.assertEqual(source_widths(old_q), [.004, .025, .035]*count)
        self.assertEqual(source_widths(new_q), [.004, .025, .035])
        self.assertEqual(old_q.call_count-new_q.call_count, 3*(count-1))
        for family in module.FAMILIES:
            self.assert_exact(module.render(new, family=family), module.render(old, family=family))
        self.assert_exact((e, pn, baseline, diagnostics), originals)

    def test_v20_actual_multi_event_fits_outputs_and_diagnostics_are_exact(self):
        self._assert_stage_parity(v20)

    def test_v24_actual_multi_event_fits_dedup_and_outputs_are_exact(self):
        self._assert_stage_parity(v24)

    def test_empty_stages_do_not_prepare_source_targets_or_fit(self):
        e, pn, baseline, diagnostics, _ = inputs(with_events=False)
        for module in (v20, v24):
            with self.subTest(module=module.__name__), \
                 patch.object(v20, "_prepare_source_targets") as prepare, \
                 patch.object(v20, "_fit_event") as fit, \
                 patch.object(v20, "_attribute_event") as attribute, \
                 patch.object(v20, "Q", wraps=v20.Q) as q:
                result = module.fit_residual_stage(e, pn, baseline=baseline,
                                                  baseline_diagnostics=diagnostics)
            self.assertEqual(result["records"], [])
            prepare.assert_not_called()
            fit.assert_not_called()
            attribute.assert_not_called()
            self.assertEqual(q.call_count, 1)  # Required residual detection only.


if __name__ == "__main__":
    unittest.main()
