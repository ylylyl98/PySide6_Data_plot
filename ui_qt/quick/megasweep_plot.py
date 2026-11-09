"""Agg-only plot presentation for Megasweep Quick."""
class MegasweepPlotMixin:
    def _render(self):
        if self._closed or not self._current(): return
        try:
            from matplotlib.figure import Figure
            from matplotlib.backend_bases import NavigationToolbar2
            from ui_qt.quick.plot_surface import QuickFigureCanvas
            if self._canvas is None:
                self._canvas = QuickFigureCanvas(Figure(figsize=(9, 6), dpi=100, layout='constrained'))
                self._navigation = NavigationToolbar2(self._canvas)
                self.plotBackendChanged.emit()
            fig = self._canvas.figure
            fig.clear()
            fig.set_facecolor(self._palette.get('inset', '#ffffff'))
            ax = fig.add_subplot(111)
            result = self._result
            values = self._state['values']
            colors = dict(cmap=values['cmap'])
            if not values['color_auto']: colors.update(vmin=values['vmin'], vmax=values['vmax'])
            if result.get('kind') in ('curve', 'line'):
                ax.plot(result['x'], result.get('values', result.get('y')))
            else:
                import numpy as np
                if np.asarray(result['z']).ndim == 1:
                    image = ax.scatter(result['x'], result['y'], c=result['z'], **colors)
                else:
                    image = ax.pcolormesh(result['x'], result['y'], result['z'], shading='auto', **colors)
                fig.colorbar(image, ax=ax, label=result.get('signal_label', ''))
            ax.set(xlabel=result.get('x_label', ''), ylabel=result.get('y_label', ''), title=result.get('title', ''))
            if not values['x_auto']: ax.set_xlim(values['xmin'], values['xmax'])
            if not values['y_auto']: ax.set_ylim(values['ymin'], values['ymax'])
            self._style_axes(fig)
            self._canvas.draw_idle()
        except Exception as exc:
            self._message('Plot: ' + str(exc), error=True)

    def plotAction(self, action):
        if action in ('home', 'back', 'forward', 'pan', 'zoom') and self._navigation:
            getattr(self._navigation, action)()
            self._plot_tool = str(self._navigation.mode.name).lower() if self._navigation.mode else ''
            self.snapshotChanged.emit()

    def _render_reflection_preview(self):
        if self._closed or not self._reflection_preview: return
        try:
            if self._canvas is None:
                from matplotlib.figure import Figure
                from matplotlib.backend_bases import NavigationToolbar2
                from ui_qt.quick.plot_surface import QuickFigureCanvas
                self._canvas = QuickFigureCanvas(Figure(figsize=(9, 6), dpi=100, layout='constrained'))
                self._navigation = NavigationToolbar2(self._canvas)
                self.plotBackendChanged.emit()
            fig = self._canvas.figure; fig.clear(); fig.set_facecolor(self._palette.get('inset', '#ffffff'))
            top, bottom = fig.subplots(2, 1, sharex=True)
            data = self._reflection_preview; x = data['energy']
            top.plot(x, data['raw'], label='Measured')
            top.plot(x, data['background'], label='Reference')
            top.plot(x, data['scaled_background'], label='Scaled reference')
            top.legend(); top.set_ylabel('Intensity')
            bottom.plot(x, data['rc']); bottom.set(xlabel='Energy (eV)', ylabel='Reflection contrast')
            fig.suptitle('Reference preview · frame ' + str(data['frame']))
            self._style_axes(fig)
            self._canvas.draw_idle()
        except Exception as exc: self._message('Preview: ' + str(exc), error=True)

    def _style_axes(self, figure):
        text = self._palette.get('text', '#172b4d'); background = self._palette.get('inset', '#ffffff')
        for ax in figure.axes:
            ax.set_facecolor(background)
            ax.tick_params(colors=text)
            ax.xaxis.label.set_color(text); ax.yaxis.label.set_color(text); ax.title.set_color(text)
            for spine in ax.spines.values(): spine.set_color(text)
            legend = ax.get_legend()
            if legend:
                for label in legend.get_texts(): label.set_color(text)
                legend.get_frame().set_facecolor(background)
        if figure._suptitle: figure._suptitle.set_color(text)
