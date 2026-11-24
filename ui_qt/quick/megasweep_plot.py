"""Agg-only plot presentation for Megasweep Quick."""
class MegasweepPlotMixin:
    def _map_color_limits(self):
        values = self._state['values']
        if values['color_auto']:
            from core.megasweep.megasweep_analysis import resolve_color_limits
            return resolve_color_limits(self._result['z'], spectral_slice=self._result['kind'] == 'slice')
        return values['vmin'], values['vmax']

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
            if result.get('kind') in ('curve', 'line'):
                ax.plot(result['x'], result.get('values', result.get('y')))
            else:
                import numpy as np
                z = np.asarray(result['z'])
                low, high = self._map_color_limits()
                colors = dict(cmap=values['cmap'], vmin=low, vmax=high)
                if z.ndim == 1:
                    image = ax.scatter(result['x'], result['y'], c=z, **colors)
                else:
                    image = ax.pcolormesh(result['x'], result['y'], z, shading='auto', **colors)
                fig.colorbar(image, ax=ax, label=result.get('signal_label', ''))
            ax.set(xlabel=result.get('x_label', ''), ylabel=result.get('y_label', ''), title=result.get('title', ''))
            if not values['x_auto']: ax.set_xlim(values['xmin'], values['xmax'])
            if not values['y_auto']: ax.set_ylim(values['ymin'], values['ymax'])
            ax.xaxis.set_inverted(values['x_reverse'])
            ax.yaxis.set_inverted(values['y_reverse'])
            self._style_axes(fig)
            self._canvas.draw_idle()
            ax.callbacks.connect('xlim_changed', self._notify_map_axis_limits)
            ax.callbacks.connect('ylim_changed', self._notify_map_axis_limits)
            self.plotLimitsChanged.emit()
        except Exception as exc:
            self._message('Plot: ' + str(exc), error=True)

    def _notify_map_axis_limits(self, axis):
        if self._closed or self._reflection_visible or not self._current() or self._canvas is None:
            return
        axes = self._canvas.figure.axes
        if axes and axes[0] is axis:
            self.plotLimitsChanged.emit()

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
            if data.get('scale', 1.) != 1.:
                top.plot(x, data['scaled_background'], label=f"Scaled reference (×{data['scale']:.5g})")
            top.legend(); top.set_ylabel('Intensity')
            bottom.plot(x, data['rc']); bottom.set(xlabel='Energy (eV)', ylabel='Reflection contrast')
            title = 'Sample spectrum'
            if 'dataset' in data:
                from core.megasweep.megasweep_analysis import display_axis_name
                from core.megasweep.workflow import rc_map_labels
                if data['parameters']['view'] in ('rc_p2p', 'rc_energy', 'rc_peak'):
                    metric, _ = rc_map_labels(data['parameters'])
                    title = f'{metric} · {title}'
                scan = data['dataset']
                x_name, y_name = (display_axis_name(scan[f'{axis}_name']) for axis in ('x', 'y'))
                title += f" · {x_name} = {data['x']:g}, {y_name} = {data['y']:g}"
                self._mark_rc_measurement(bottom, data)
            fig.suptitle(title)
            self._style_axes(fig)
            self._canvas.draw_idle()
        except Exception as exc: self._message('Preview: ' + str(exc), error=True)

    @staticmethod
    def _mark_rc_measurement(ax, preview):
        import numpy as np
        from core.megasweep.megasweep_analysis import compute_rc_at_energy_map, compute_rc_peak_position
        params = preview['parameters']
        view = params['view']
        energy = preview['energy']
        rc = preview['rc']
        if view == 'rc_energy':
            target = params['energy_target']
            ax.axvline(target, linestyle='--', color='gray')
            if energy.min() <= target <= energy.max() and np.isfinite(energy).sum() >= 2:
                value = compute_rc_at_energy_map(rc[None, :], energy, target)[0]
                ax.plot(target, value, 'o', label=f'{target:.5g} eV · RC {value:.5g}')
                ax.legend()
            else:
                ax.set_title('Target energy outside usable spectrum')
            return
        if view not in ('rc_p2p', 'rc_peak'):
            return
        lo, hi = sorted((params['energy_min'], params['energy_max']))
        ax.axvspan(lo, hi, alpha=0.10)
        indices = np.flatnonzero((energy >= lo) & (energy <= hi) & np.isfinite(rc))
        if not len(indices):
            ax.set_title('No valid RC in the selected window')
        elif view == 'rc_p2p':
            low = indices[np.argmin(rc[indices])]
            high = indices[np.argmax(rc[indices])]
            ax.plot(energy[low], rc[low], 'v', label='Minimum')
            ax.plot(energy[high], rc[high], '^', label='Maximum')
            ax.set_title(f'Peak-to-peak RC = {rc[high] - rc[low]:.5g}')
            ax.legend()
        else:
            position, value = compute_rc_peak_position(energy, rc, lo, hi, params['sg_window'],
                                                       params['sg_poly'], params['rc_feature'])
            if np.isfinite(position):
                ax.plot(position, value, 'o', label=f'Feature at {position:.5g} eV')
                ax.legend()
            else:
                ax.set_title('No feature found in the selected window')

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
