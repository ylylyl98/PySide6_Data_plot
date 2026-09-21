import unittest
import numpy as np
from matplotlib.figure import Figure
from core.cw_method_comparison import compare_cw_methods
from ui_qt.cw_comparison_plots import draw_method_curves
from ui_qt.theta_comparison_dialog import draw_comparison

class ComparisonPlotTests(unittest.TestCase):
    def test_theta_intervals_are_in_both_plot_legends(self):
        pts=[dict(branch='B increasing',temperature_k=t,slope=.1,slope_se=.01) for t in [2,3,4]]
        rows=[dict(branch='B increasing',method=k,theta_k=-2.,amplitude=.5,background=0.,status='estimated',
                   ci95_low_k=lo,ci95_high_k=hi) for k,lo,hi in [('inverse',-4.,1.),('slope',None,None),('weighted',None,.3)]]
        fig=Figure();draw_method_curves(fig,pts,rows,{'inverse','slope','weighted'})
        for ax in fig.axes:
            labels={line.get_gid():line.get_label() for line in ax.lines}
            self.assertIn('95% CI [-4, 1] K',labels['cw-method-inverse'])
            self.assertIn('95% CI unavailable',labels['cw-method-slope'])
            self.assertIn('95% profile CI [unbounded, 0.3] K',labels['cw-method-weighted'])

    def test_noisy_inverse_point_keeps_errorbar_and_centered_warning(self):
        pts=[dict(branch='B increasing',temperature_k=3.2,slope=.002,slope_se=.005)]
        fig=Figure();draw_method_curves(fig,pts,[],set())
        ax=fig.axes[1]
        bars=[c for c in ax.containers if hasattr(c,'has_yerr') and c.has_yerr]
        self.assertEqual(len(bars),1)
        segment=bars[0].lines[2][0].get_segments()[0]
        np.testing.assert_allclose(segment,[[3.2,-750],[3.2,1750]])
        warning=next(line for line in ax.lines if line.get_gid()=='inverse-uncertainty-warning')
        np.testing.assert_allclose(warning.get_xdata(),[3.2])
        np.testing.assert_allclose(warning.get_ydata(),[500])

    def test_fixed_background_does_not_depend_on_visible_methods(self):
        pts=[dict(branch='B increasing',temperature_k=t,slope=.3,slope_se=None) for t in [2,3,4]]
        fig=Figure();draw_method_curves(fig,pts,[],set(),background=.1)
        np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(),5.)
    def test_pole_not_drawn_in_slope_space(self):
        points=[dict(branch='B increasing',temperature_k=t,slope=s,slope_se=.01) for t,s in [(2,.1),(3,.2),(4,.03)]]
        rows=[dict(branch='B increasing',method='inverse',label='Inverse',theta_k=2.5,amplitude=.1,background=0,status='diagnostic only')]
        fig=Figure();draw_method_curves(fig,points,rows,{'inverse'})
        self.assertFalse(any(line.get_gid()=='cw-method-inverse' for line in fig.axes[0].lines))
        self.assertTrue(any(line.get_gid()=='cw-method-inverse' for line in fig.axes[1].lines))

    def test_methods_provide_plot_parameters(self):
        t=np.array([2,3,4,6,9,15.])
        for r in compare_cw_methods(t,2/(t+4),np.full(6,.01)):
            self.assertAlmostEqual(r['amplitude'],2,places=3)

    def test_batch_three_axes_share_limits(self):
        rows=[]
        for method,value in [('inverse',-2),('slope',-4),('weighted',-8)]:
            rows.append(dict(method=method,series_id='s',group='Group 1',doping=6.3,efield=2,branch='B increasing',halfwidth_t=.2,is_primary=True,n=6,theta_k=value,ci95_low_k=value-1,ci95_high_k=value+1,status='ok',fixed_conditions={},diagnostics=''))
        fig=Figure();draw_comparison(fig,dict(rows=[rows[0]],method_rows=rows),method='all')
        self.assertEqual(len(fig.axes),3)
        self.assertEqual(len({ax.get_ylim() for ax in fig.axes}),1)
