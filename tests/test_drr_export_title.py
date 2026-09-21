import unittest
import tempfile
import json
from dataclasses import replace
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
