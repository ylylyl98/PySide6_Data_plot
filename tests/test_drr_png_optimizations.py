"""Full-resolution PNG pixels, bounded title work and export diagnostics."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import re
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from PIL import Image

from core import export, plotting
from core.compact_region_export import complete_title
from core.loader import DataCube, load_dat
from core.plotting import HeatmapParams, SplitColorScale


def legacy_title(fig, text, left, right, *, top=.95, bottom=.85, obstacles=()):
    """Frozen pre-optimization layout reference, including character fallback."""
    renderer = fig.canvas.get_renderer()
    widths = sorted({(right-left)*fig.bbox.width,
        *(box.x0-left*fig.bbox.width-8 for box in obstacles)}, reverse=True)
    widths = [w for w in widths if 0 < w <= (right-left)*fig.bbox.width]
    height = (top-bottom)*fig.bbox.height
    artist = fig.text(left, top, '', ha='left', va='top', fontweight='bold', linespacing=1.)
    for size in np.arange(16., .5, -.5):
        for width in widths:
            font = FontProperties(size=size, weight='bold')
            lines, current = [], ''
            for word in re.findall(r'\$[^$]*\$|\S+', str(text)):
                trial = f'{current} {word}' if current else word
                if current and renderer.get_text_width_height_descent(trial,font,False)[0] > width:
                    lines.append(current)
                    current = ''
                if word.startswith('$') and word.endswith('$'):
                    current = (current+' '+word).strip()
                    continue
                for char in (' ' if current else '')+word:
                    trial = current+char
                    if current and renderer.get_text_width_height_descent(trial,font,False)[0] > width:
                        lines.append(current.rstrip())
                        current = char.lstrip()
                    else:
                        current = trial
            if current:
                lines.append(current)
            artist.set_text('\n'.join(lines))
            artist.set_fontsize(size)
            extent = artist.get_window_extent(renderer)
            if (extent.width <= width+.5 and extent.height <= height
                and all(not extent.overlaps(box.padded(4)) for box in obstacles)):
                return artist
    raise ValueError('Title is too long to fit the fixed export header.')


class PngOptimizationTests(unittest.TestCase):
    def test_failed_nested_png_timing_restores_the_previous_scope(self):
        from core.png_render_timing import collect_png_timings, time_png_stage
        with collect_png_timings() as outer:
            with time_png_stage('build'):
                with self.assertRaisesRegex(ValueError,'encoder failed'):
                    with collect_png_timings() as failed:
                        with time_png_stage('encode_write'):
                            raise ValueError('encoder failed')
            with time_png_stage('draw'):
                pass
        self.assertEqual(set(outer),{'build','draw'})
        self.assertEqual(set(failed),{'encode_write'})
        snapshot = outer.copy()
        with time_png_stage('outside'):
            pass
        self.assertEqual(outer,snapshot)

    def test_overlapping_png_workers_keep_their_own_timing_and_progress(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        from core.png_render_timing import collect_png_timings, time_png_stage
        first_entered, second_entered, first_stage, second_finished = (Event() for _ in range(4))
        def first():
            messages = []
            with collect_png_timings(messages.append) as durations:
                first_entered.set()
                self.assertTrue(second_entered.wait(5))
                with time_png_stage('draw'):
                    pass
                first_stage.set()
                self.assertTrue(second_finished.wait(5))
            return durations,messages
        def second():
            self.assertTrue(first_entered.wait(5))
            messages = []
            with collect_png_timings(messages.append) as durations:
                second_entered.set()
                self.assertTrue(first_stage.wait(5))
                with time_png_stage('title'):
                    pass
            second_finished.set()
            return durations,messages
        with ThreadPoolExecutor(max_workers=2) as pool:
            a,b = pool.submit(first),pool.submit(second)
            first_times,first_messages = a.result(timeout=10)
            second_times,second_messages = b.result(timeout=10)
        self.assertEqual(set(first_times),{'draw'})
        self.assertEqual(set(second_times),{'title'})
        self.assertEqual(first_messages,['Drawing PNG'])
        self.assertEqual(second_messages,['Fitting PNG title'])

    def test_title_measures_fitting_words_without_character_prefix_work(self):
        fig = Figure(figsize=(8.,6.2),dpi=150)
        FigureCanvasAgg(fig)
        renderer = fig.canvas.get_renderer()
        calls = []
        original = renderer.get_text_width_height_descent
        def measured(*args, **kwargs):
            calls.append(args[0])
            return original(*args, **kwargs)
        text = 'YZ365 p5n2 0T 20KREF 720nmc TG-1.087BG=24 DR/R repeat03'
        with patch.object(renderer,'get_text_width_height_descent',measured):
            title = complete_title(fig,text,.16,.92,top=.95,bottom=.75)
        self.assertEqual(title.get_text().replace('\n',' '),text)
        self.assertLess(len(calls),30, 'Fitting words should not trigger per-character font measurements')

    def test_title_pixels_match_legacy_layout_with_math_and_long_identifiers(self):
        for text in ('Device $TG-r_{BG}BG=0$ 720nmc repeat03',
                     'Identifier'+'AWVdTp5n2'*9,
                     'YZ365 720nmc TG-1.087BG=24 '*5):
            with self.subTest(text=text):
                pixels, artists = [], []
                for fitter in (legacy_title,complete_title):
                    fig = Figure(figsize=(8.,6.2),dpi=100)
                    canvas = FigureCanvasAgg(fig)
                    obstacle = fig.text(.75,.94,'L: -0.0002',va='bottom').get_window_extent(canvas.get_renderer())
                    artist = fitter(fig,text,.16,.92,obstacles=(obstacle,))
                    artists.append((artist.get_text(),artist.get_fontsize()))
                    canvas.draw()
                    pixels.append(np.asarray(canvas.buffer_rgba()).copy())
                self.assertEqual(artists[0],artists[1])
                np.testing.assert_array_equal(*pixels)

    def test_export_keeps_original_edges_but_skips_offscreen_cells(self):
        cube = DataCube(np.arange(41.),np.arange(31.),np.arange(1271.).reshape(31,41),
                        'Gate','Visible cells','DR/R')
        params = HeatmapParams('Visible cells','Energy','Gate','DR/R',0.,1271.,
                               (10.2,19.8),(12.2,17.8))
        fig = export._build_streamlit_style_heatmap_fig(cube,params,drr=True)
        try:
            mesh = fig.axes[0].collections[0]
            self.assertEqual(mesh.get_array().shape,(9,13))
            np.testing.assert_array_equal(mesh.get_array(),cube.Z[11:20,9:22])
            coordinates = mesh.get_coordinates()
            np.testing.assert_array_equal(coordinates[0,:,0],np.arange(8.5,22.5))
            np.testing.assert_array_equal(coordinates[:,0,1],np.arange(10.5,20.5))
        finally:
            export.plt.close(fig)
        interactive = Figure()
        render = plotting.plot_heatmap(interactive.subplots(),cube,params)
        self.assertEqual(render.primary.get_array().size,1271)

    def test_visible_export_pixels_match_full_mesh_for_nonuniform_nan_and_log_axes(self):
        self.assertTrue(hasattr(plotting,'visible_mesh_data'), 'Visible mesh clipping is not implemented')
        energy = np.cumsum(np.resize([.6,1.4,2.,.8],61))
        gate = np.cumsum(np.resize([.4,.9,1.7],51))
        z = np.sin(energy[None,:]) + np.cos(gate[:,None]) + 2.5
        z[8:12,24] = np.nan
        z[20,30] = 100.
        cube = DataCube(energy,gate,z,'Gate','Pixels','DR/R')
        base = HeatmapParams('Pixels','Energy','Gate','DR/R',.1,5.,(20.2,45.7),(2.7,18.4))
        split = SplitColorScale(29.5,.1,3.,.2,4.)
        triple = replace(split,split_x2=37.,middle_vmin=.3,middle_vmax=5.)
        variants = [base,replace(base,split_scale=split,clip_outliers=True),
                    replace(base,split_scale=triple),replace(base,split_scale=triple,log_scale=True),
                    replace(base,y_axis_log=True,ylim=(18.4,2.7),split_scale=split),
                    replace(base,xlim=(45.7,20.2),ylim=(18.4,2.7),center_zero=True,vmin=-1.),
                    replace(base,xlim=(80.,90.)),replace(base,ylim=(80.,90.))]
        def full_mesh(x,y,values,*_args,**_kwargs):
            return x,y,values
        for params in variants:
            with self.subTest(params=params):
                actual = export._build_streamlit_style_heatmap_fig(cube,params,drr=True)
                try:
                    actual.canvas.draw()
                    pixels = np.asarray(actual.canvas.buffer_rgba()).copy()
                    with ExitStack() as stack:
                        stack.enter_context(patch.object(plotting,'visible_mesh_data',full_mesh))
                        stack.enter_context(patch.object(export,'visible_mesh_data',full_mesh))
                        reference = export._build_streamlit_style_heatmap_fig(cube,params,drr=True)
                    try:
                        reference.canvas.draw()
                        np.testing.assert_array_equal(pixels,np.asarray(reference.canvas.buffer_rgba()))
                    finally:
                        export.plt.close(reference)
                finally:
                    export.plt.close(actual)

    def test_singular_or_invalid_log_limits_preserve_matplotlib_auto_expansion(self):
        cube = DataCube(1000.+np.arange(61.),1000.+np.arange(51.),
                        np.arange(3111.).reshape(51,61),'Gate','Limits','DR/R')
        base = HeatmapParams('Limits','Energy','Gate','DR/R',0.,3111.,
                             (1020.,1040.),(1010.,1030.))
        variants = [replace(base,xlim=(1030.,1030.)),replace(base,ylim=(1020.,1020.)),
                    replace(base,xlim=(1030.,np.nextafter(1030.,np.inf))),
                    replace(base,y_axis_log=True,ylim=(-2.,-1.))]
        def full_mesh(x,y,values,*_args,**_kwargs):
            return x,y,values
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            for params in variants:
                with self.subTest(params=params):
                    actual = export._build_streamlit_style_heatmap_fig(cube,params,drr=True)
                    try:
                        actual.canvas.draw()
                        pixels = np.asarray(actual.canvas.buffer_rgba()).copy()
                        with patch.object(export,'visible_mesh_data',full_mesh):
                            reference = export._build_streamlit_style_heatmap_fig(cube,params,drr=True)
                        try:
                            reference.canvas.draw()
                            np.testing.assert_array_equal(pixels,np.asarray(reference.canvas.buffer_rgba()))
                            np.testing.assert_allclose(actual.axes[0].get_ylim(),reference.axes[0].get_ylim())
                        finally:
                            export.plt.close(reference)
                    finally:
                        export.plt.close(actual)

    def test_cropping_cannot_hide_a_transposed_source_matrix(self):
        cube = DataCube(np.arange(8.),np.arange(6.),np.arange(48.).reshape(8,6),
                        'Gate','Malformed','DR/R')
        params = HeatmapParams('Malformed','Energy','Gate','DR/R',0.,48.,(1.,3.),(1.,3.))
        previous = set(export.plt.get_fignums())
        try:
            with self.assertRaises((ValueError,TypeError)):
                export._build_streamlit_style_heatmap_fig(cube,params,drr=True)
        finally:
            for number in set(export.plt.get_fignums())-previous:
                export.plt.close(number)

    def test_paired_export_reports_png_stages_without_cropping_dat_or_rerendering(self):
        cube = DataCube(np.arange(41.),np.arange(31.),np.arange(1271.).reshape(31,41),
                        'Gate','Diagnostics','DR/R')
        params = HeatmapParams('Diagnostics','Energy','Gate','DR/R',0.,1271.,(10.2,19.8),(12.2,17.8))
        with tempfile.TemporaryDirectory() as folder:
            logs = []
            options = dict(raw_cube=cube,second_cube=cube,raw_params=params,second_params=params,
                           raw_export_base='raw',second_export_base='second',export_progress=logs.append)
            saved = export.export_drr_pair_pngs_and_dat(folder,**options)
            details = [line for line in logs if 'PNG stages:' in line]
            self.assertEqual(len(details),2,'Each newly rendered product needs its own PNG stage diagnostics')
            for product in ('raw','second'):
                recovered = load_dat(saved[product+'_dat'])
                np.testing.assert_array_equal(recovered.Z,cube.Z)
                with Image.open(saved[product+'_png']) as png:
                    self.assertEqual(png.size,(1200,930))
            for detail in details:
                values = dict((key,float(value)) for key,value in re.findall(r'(\w+)=(\d+\.\d+) s',detail))
                self.assertEqual(set(values),{'build','title','draw','encode_write','total'})
                self.assertTrue(all(value >= 0 for value in values.values()))
                self.assertGreaterEqual(values['build']+.001,values['title'])
                self.assertGreaterEqual(values['total']+.001,values['draw'])
            logs.clear()
            repeated = export.export_drr_pair_pngs_and_dat(folder,**options)
            self.assertEqual(repeated.save_status,'reused')
            self.assertFalse(any('PNG stages:' in line for line in logs))


if __name__ == '__main__':
    unittest.main()
