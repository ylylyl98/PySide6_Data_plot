import unittest
import tempfile
import json
from dataclasses import replace
from unittest.mock import patch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from core.export import _build_streamlit_style_heatmap_fig, export_drr_png_and_dat, EXPORT_FIGSIZE, EXPORT_AXES_RECT
from tests.test_drr_three_regions import example
import numpy as np


class DrrExportTitleTests(unittest.TestCase):
    def test_available_header_space_preserves_original_font(self):
        cube, params = example()
        params.title = 'YZ365 p5n2 0T 31KREF 690nmc TG-1.087BG=20 (DR/R) (d2E)'
        params.cbar_label = 'd2(DR/R)/dE2'
        fig = _build_streamlit_style_heatmap_fig(cube, params, drr=True)
        FigureCanvasAgg(fig).draw()
        title = fig.texts[0]
        self.assertEqual(title.get_fontsize(), 16)
        renderer = fig.canvas.get_renderer()
        bounds = title.get_window_extent(renderer)
        self.assertGreater(bounds.y0, fig.axes[0].bbox.y1)
        self.assertFalse(bounds.overlaps(fig.texts[1].get_window_extent(renderer)))

    def test_removes_only_recognized_fields(self):
        from core.drr_export_title import display_title
        title = 'YZ365_p5n2_5T_1.67KREF_760nm_0p08sx10_RotIn100deg_TG-1.087BG=0_session003_repeat02_END.csv'
        result = display_title(title)
        self.assertEqual(result, 'YZ365 p5n2 5T 1.67KREF 760nm RotIn100deg TG-1.087BG=0 repeat02')
        self.assertEqual(display_title('sample | 80 ms x 10 | 2026-09-21 12:30:45 | External background | custom ABC'),
                         'sample custom ABC')

    def test_preserves_unknown_conditions_and_math(self):
        from core.drr_export_title import display_title
        title = 'FRIEND | sessionStudy | background study | repeat03 | REF | TG-rBG=0 | $TG-r_{BG}BG=0$ | new-mode=7 | custom=END | custom=session003 | phase=2026-09-21'
        result = display_title(title)
        for value in ('FRIEND', 'sessionStudy', 'background study', 'repeat03', 'REF', 'TG-rBG=0', '$TG-r_{BG}BG=0$', 'new-mode=7', 'custom=END', 'custom=session003', 'phase=2026-09-21'):
            self.assertIn(value, result)

    def test_explicit_background_file_and_processing_suffix(self):
        from core.drr_export_title import display_title
        self.assertEqual(display_title('sample | background file="C:/Data/my background.csv" | TG-rBG=0 (DR/R external, avg 3)'),
                         'sample TG-rBG=0 (DR/R)')
        self.assertEqual(display_title('sample | background mode=self_last | 2026-09-21_12-30-45 | unknown=2'),
                         'sample unknown=2')

    def test_terminal_generated_product_chain_keeps_only_current_product(self):
        from core.drr_export_title import display_title
        cases = (
            ('YZ383 pwn1 0T 1.67KREF 730nmc RotIn49deg TG+1BG=0 PIXIS (DR/R) (d2(DR/R)/dE2)',
             'YZ383 pwn1 0T 1.67KREF 730nmc RotIn49deg TG+1BG=0 PIXIS (d2(DR/R)/dE2)'),
            ('sample_PIXIS (DR/R external, avg 3) (d2(DR/R)/dE2)', 'sample PIXIS (d2(DR/R)/dE2)'),
            ('sample (DR/R) (d(DR/R)/dE)', 'sample (d(DR/R)/dE)'),
            ('sample (DR/R) (d(DR/R)/dE) (d2(DR/R)/dE2)', 'sample (d2(DR/R)/dE2)'),
            ('sample (DR/R) (d2E)', 'sample (d2E)'),
            ('sample (DR/R) (dXdY)', 'sample (dXdY)'),
            ('sample (DR/R) (d2(DR/R)/(dE dGate (V)))', 'sample (d2(DR/R)/(dE dGate (V)))'),
            ('sample (DR/R) (Advanced first derivative, dE)', 'sample (Advanced first derivative, dE)'),
            ('sample (DR/R) (DR/R)', 'sample (DR/R)'),
            ('sample (DR/R)', 'sample (DR/R)'),
            ('sample (DR/R external, avg 3)', 'sample (DR/R)'),
        )
        for original, expected in cases:
            with self.subTest(original=original):
                self.assertEqual(display_title(original), expected)

    def test_product_cleanup_preserves_unknown_conditions_and_math(self):
        from core.drr_export_title import display_title
        original = 'sample note=(DR/R) custom=d2E $x (DR/R) (d2E)$ (DR/R) (d2E)'
        self.assertEqual(display_title(original), 'sample note=(DR/R) custom=d2E $x (DR/R) (d2E)$ (d2E)')
        for original in ('sample (DR/R) (unknown product)', 'sample (DR/R) (d2E) unknown-condition',
                         'sample (DR/R) custom=7 (d2E)', 'sample(DR/R) (d2E)'):
            with self.subTest(original=original):
                self.assertEqual(display_title(original), original)

    def test_derivative_processing_title_remains_full_provenance(self):
        from core.drr_export_title import display_title
        from core.processing import apply_sg_derivative_energy
        cube, _ = example()
        original = 'sample_PIXIS (DR/R external, avg 3)'
        cube.title = original
        second, _ = apply_sg_derivative_energy(cube, derivative=2, window_length=5, polyorder=2)
        self.assertEqual(cube.title, original)
        self.assertEqual(second.title, original + ' (d2(DR/R)/dE2)')
        self.assertEqual(second.cbar_label, 'd2(DR/R)/dE2')
        self.assertEqual(display_title(second.title), 'sample PIXIS (d2(DR/R)/dE2)')

    def test_regular_and_wide_render_only_current_product_marker(self):
        cube, params = example()
        original = 'sample_PIXIS (DR/R external, avg 3) (d2(DR/R)/dE2)'
        for width in (1., 2.):
            with self.subTest(width=width):
                variant = replace(params, title=original, cbar_label='d2(DR/R)/dE2', png_width_scale=width)
                fig = _build_streamlit_style_heatmap_fig(cube, variant, drr=True)
                FigureCanvasAgg(fig).draw()
                self.assertEqual(' '.join(fig.texts[0].get_text().split()), 'sample PIXIS (d2(DR/R)/dE2)')
                quantity = next(text for text in fig.texts if text.get_gid() == 'drr-colorbar-quantity')
                self.assertEqual(quantity.get_text(), 'd2(DR/R)/dE2')
                self.assertEqual(variant.title, original)
                fig.clear()

    def test_old_png_title_policy_refreshes_figures_without_rewriting_dat(self):
        from core import export
        from core.drr_export_title import TITLE_POLICY_VERSION, display_title
        cube, params = example()
        original = 'sample PIXIS (DR/R) (d2(DR/R)/dE2)'
        expected = 'sample PIXIS (d2(DR/R)/dE2)'
        for width in (1., 2.):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as folder:
                variant = replace(params, title=original, cbar_label='d2(DR/R)/dE2', png_width_scale=width)
                kwargs = dict(folder=folder, cube=cube, params=variant, export_base='original_source_name',
                              metadata_processing={'derivative_order': 2}, include_original_png=width > 1.)
                def save_png(path, cube, params, **options):
                    path.write_text(export.display_title(params.title), encoding='utf-8')
                with patch.object(export, '_save_heatmap_png', side_effect=save_png) as render:
                    with patch.object(export, 'display_title', side_effect=lambda title: title):
                        first = export.export_drr_png_and_dat(**kwargs)
                    metadata = first['dat'].with_suffix('.metadata.json')
                    payload = json.loads(metadata.read_text(encoding='utf-8'))
                    def old_fingerprint(plot):
                        return export._fingerprint_json({'render_version': 7,
                            **({'wide_header_version': 2} if plot['png_width_scale'] > 1. else {}), 'params': plot})
                    payload['plot_fingerprint'] = old_fingerprint(payload['plot'])
                    payload['display_title'], payload['title_policy_version'] = original, 1
                    for record in payload['figures'].values():
                        record['plot_fingerprint'] = old_fingerprint(record['plot'])
                    metadata.write_text(json.dumps(payload), encoding='utf-8')
                    dat_state = first['dat'].read_bytes(), first['dat'].stat().st_mtime_ns
                    render.reset_mock()
                    with patch.object(export, '_save_drr_dat_atomic', side_effect=AssertionError('Title change rewrote DAT')):
                        updated = export.export_drr_png_and_dat(**kwargs)
                    self.assertEqual(render.call_count, 2 if width > 1. else 1)
                    self.assertEqual(updated['dat'], first['dat'])
                    self.assertEqual((updated['dat'].read_bytes(), updated['dat'].stat().st_mtime_ns), dat_state)
                    self.assertEqual(updated['dat'].stem, 'original_source_name')
                    self.assertEqual(len(list(updated['dat'].parent.glob('*.dat'))), 1)
                    for key in ('png', 'png_wide') if width > 1. else ('png',):
                        self.assertEqual(first[key].read_text(encoding='utf-8'), original)
                        self.assertEqual(updated[key].read_text(encoding='utf-8'), expected)
                    current = json.loads(metadata.read_text(encoding='utf-8'))
                    self.assertEqual(current['plot']['title'], original)
                    self.assertEqual(current['display_title'], display_title(original))
                    self.assertEqual(current['title_policy_version'], TITLE_POLICY_VERSION)
                    render.reset_mock()
                    reused = export.export_drr_png_and_dat(**kwargs)
                    self.assertEqual(reused, updated)
                    self.assertEqual(reused.save_status, 'reused')
                    render.assert_not_called()

    def test_all_regions_fit_simplified_title_without_mutating_params(self):
        cube, params = example()
        original = 'YZ365_p5n2_5T_1.67KREF_760nm_0p08sx10_RotIn100deg_TG-1.087BG=0_session003_repeat02_END.csv'
        for split in (None, replace(params.split_scale, split_x2=None), params.split_scale):
            p = replace(params, title=original, split_scale=split)
            fig = _build_streamlit_style_heatmap_fig(cube, p, drr=True)
            FigureCanvasAgg(fig).draw()
            text = fig.texts[0]
            self.assertNotIn('session003', text.get_text())
            self.assertIn('TG-1.087BG=0', text.get_text())
            self.assertEqual(p.title, original)
            np.testing.assert_allclose(fig.get_size_inches(), EXPORT_FIGSIZE)
            np.testing.assert_allclose(fig.axes[0].get_position().bounds, EXPORT_AXES_RECT)
            renderer = fig.canvas.get_renderer()
            bounds = text.get_window_extent(renderer)
            self.assertGreater(bounds.y0, fig.axes[0].bbox.y1)
            self.assertTrue(all(not bounds.overlaps(a.get_tightbbox(renderer)) for a in fig.axes[1:]))

    def test_metadata_and_export_names_preserve_original(self):
        cube, params = example()
        params.title = 'sample_80msx10_session003_END.csv'
        with tempfile.TemporaryDirectory() as folder:
            paths = export_drr_png_and_dat(folder, cube=cube, params=params, export_base='sample_session003_END')
            self.assertEqual(paths['png'].stem, 'sample_session003_END')
            data = json.loads(paths['dat'].with_suffix('.metadata.json').read_text())
            self.assertEqual(data['plot']['title'], params.title)
            self.assertEqual(data['display_title'], 'sample')

    def test_math_gate_formula_is_not_broken_across_lines(self):
        cube, params = example()
        params.title = r'sample $TG - r_{BG}BG = 0$ 80msx10 repeat03'
        fig = _build_streamlit_style_heatmap_fig(cube, replace(params, split_scale=None), drr=True)
        FigureCanvasAgg(fig).draw()
        self.assertIn(r'$TG - r_{BG}BG = 0$', fig.texts[0].get_text())
