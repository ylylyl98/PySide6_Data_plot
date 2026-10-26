"""Screen rasterization preserves measured cells, masks and export artists."""
from io import BytesIO
import unittest
from unittest.mock import patch
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from ui_qt.quick.raster_heatmaps import screen_heatmaps


class ScreenHeatmapTests(unittest.TestCase):
    def make(self, *, reverse=False, masked=False):
        figure=Figure(figsize=(8,6),dpi=100);canvas=FigureCanvasAgg(figure)
        axis=figure.add_axes([.1,.1,.8,.8])
        x=np.linspace(0,1,501)**1.2;y=np.linspace(0,1,301)**1.1
        values=np.repeat(np.arange(15),20)[:,None]+np.repeat(np.arange(25),20)[None,:]
        values=np.ma.array(values,mask=False)
        if masked:values.mask[100:180,200:300]=True
        if reverse:x=x[::-1];y=y[::-1];values=values[::-1,::-1]
        mesh=axis.pcolormesh(x,y,values,cmap="viridis",vmin=0,vmax=38)
        return canvas,axis,mesh,x,y,values

    def test_irregular_descending_and_masked_cells_have_original_colors(self):
        for reverse in (False,True):
            canvas,axis,mesh,x,y,values=self.make(reverse=reverse,masked=True)
            original=mesh.get_coordinates().copy();data=mesh.get_array().copy()
            with screen_heatmaps(canvas):
                self.assertIn("draw",mesh.__dict__)
                canvas.draw()
            self.assertNotIn("draw",mesh.__dict__)
            np.testing.assert_array_equal(mesh.get_coordinates(),original)
            np.testing.assert_array_equal(mesh.get_array(),data)
            pixels=np.asarray(canvas.buffer_rgba())
            for row,col in ((10,10),(50,130),(130,250),(230,410),(270,470)):
                cx=(x[col]+x[col+1])/2;cy=(y[row]+y[row+1])/2
                px,py=axis.transData.transform((cx,cy));px=int(px);py=pixels.shape[0]-1-int(py)
                expected=np.asarray([255,255,255,255] if np.ma.is_masked(values[row,col])
                                    else mesh.to_rgba(values[row,col],bytes=True))
                np.testing.assert_allclose(pixels[py,px],expected,atol=1)

    def test_export_bypasses_screen_path_and_restores_on_exception(self):
        canvas,axis,mesh,*_=self.make()
        original=mesh.draw
        with self.assertRaisesRegex(RuntimeError,"injected"):
            with screen_heatmaps(canvas):
                raise RuntimeError("injected")
        self.assertEqual(mesh.draw,original)
        with patch.object(canvas,"is_saving",return_value=True):
            with screen_heatmaps(canvas):self.assertEqual(mesh.draw,original)
        before=BytesIO();canvas.figure.savefig(before,format="png")
        with screen_heatmaps(canvas):canvas.draw()
        after=BytesIO();canvas.figure.savefig(after,format="png")
        self.assertEqual(before.getvalue(),after.getvalue())
        nested=BytesIO()
        with screen_heatmaps(canvas):canvas.figure.savefig(nested,format="png")
        self.assertEqual(before.getvalue(),nested.getvalue())

    def test_log_axes_edges_and_custom_alpha_keep_standard_renderer(self):
        canvas,axis,mesh,*_=self.make()
        axis.set_yscale("log");axis.set_ylim(.05,1)
        with screen_heatmaps(canvas):self.assertNotIn("draw",mesh.__dict__)
        axis.set_yscale("linear")
        mesh.set_edgecolor("red")
        with screen_heatmaps(canvas):self.assertNotIn("draw",mesh.__dict__)
        mesh.set_edgecolor("none");mesh.set_alpha(np.ones(mesh.get_array().shape)*.5)
        with screen_heatmaps(canvas):self.assertNotIn("draw",mesh.__dict__)

    def test_zoom_uses_same_source_cells_without_mutating_coordinates(self):
        canvas,axis,mesh,x,y,values=self.make()
        axis.set_xlim(.28,.52);axis.set_ylim(.31,.68)
        original=mesh.get_array().copy()
        with screen_heatmaps(canvas):canvas.draw()
        np.testing.assert_array_equal(mesh.get_array(),original)
        pixels=np.asarray(canvas.buffer_rgba())
        row,col=170,220
        px,py=axis.transData.transform(((x[col]+x[col+1])/2,(y[row]+y[row+1])/2))
        np.testing.assert_allclose(pixels[pixels.shape[0]-1-int(py),int(px)],mesh.to_rgba(values[row,col],bytes=True),atol=1)

    def test_outside_data_is_transparent_over_axes_background(self):
        canvas,axis,mesh,*_=self.make(reverse=True,masked=True)
        axis.set_facecolor("#E5CCB3");axis.set_xlim(-.5,1.5);axis.set_ylim(-.5,1.5)
        with screen_heatmaps(canvas):canvas.draw()
        pixels=np.asarray(canvas.buffer_rgba())
        px,py=axis.transData.transform((-.25,.5))
        np.testing.assert_array_equal(pixels[pixels.shape[0]-1-int(py),int(px)], [229,204,179,255])


if __name__=="__main__":unittest.main()
