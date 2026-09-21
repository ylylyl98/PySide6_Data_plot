"""Plot matched CW methods without drawing across a model pole."""
import numpy as np

METHODS={'inverse':('Inverse / equal','--'),'slope':('Slope / equal','-.'),'weighted':('Slope / SE weighted','-')}

def draw_method_curves(figure, points, rows, visible_methods, background=0.):
    figure.clear()
    branches=list(dict.fromkeys(p['branch'] for p in points))
    if not branches:return
    for i,branch in enumerate(branches):
        left=figure.add_subplot(len(branches),2,2*i+1)
        right=figure.add_subplot(len(branches),2,2*i+2)
        block=sorted([p for p in points if p['branch']==branch],key=lambda p:p['temperature_k'])
        t=np.array([p['temperature_k'] for p in block]);s=np.array([p['slope'] for p in block]);e=np.array([p.get('slope_se') for p in block],float)
        left.plot(t,s,'o',color='black',label='Data')
        valid=np.isfinite(e)&(e>=0)
        if valid.any():left.errorbar(t[valid],s[valid],yerr=e[valid],fmt='none',color='black',capsize=3)
        fits=[r for r in rows if r['branch']==branch and r['method'] in visible_methods]
        base=float(background)
        y=s-base;mask=np.abs(y)>1e-15
        right.plot(t[mask],1/y[mask],'o',color='black',label='Data (inverse diagnostic)')
        # Show the delta-method error even for low-SNR slopes, but distinguish
        # it from a reliable reciprocal interval (which may be unbounded).
        propagated=np.full_like(y,np.nan)
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
            propagated[mask&valid]=e[mask&valid]/y[mask&valid]**2
        error_mask=mask&valid&np.isfinite(propagated)
        if error_mask.any():
            right.errorbar(t[error_mask],1/y[error_mask],yerr=propagated[error_mask],
                           fmt='none',color='black',capsize=3,label='First-order propagated SE')
        uncertain=mask&valid&(np.abs(y)<=1.96*e)
        if uncertain.any():
            from matplotlib import patheffects
            warning,=right.plot(t[uncertain],1/y[uncertain],'x',color='crimson',markersize=8,
                               markeredgewidth=1.5,label='×: near zero; inverse SE unreliable',zorder=6)
            warning.set_gid('inverse-uncertainty-warning')
            warning.set_path_effects([patheffects.withStroke(linewidth=3,foreground='white'),patheffects.Normal()])
            right.text(.02,.03,'×: first-order error only; not a bounded confidence interval',
                       transform=right.transAxes,fontsize=7,va='bottom')
        if (~mask).any():
            right.text(.02,.09,f'{int((~mask).sum())} zero/near-zero slope(s): reciprocal undefined',
                       transform=right.transAxes,fontsize=7,va='bottom')
        notes=[]
        for r in fits:
            name,style=METHODS[r['method']];theta=r['theta_k'];a=r.get('amplitude')
            if theta is None or a is None or abs(a)<1e-15:
                notes.append(name+': unavailable');continue
            label=f'{name}: theta={theta:.3g} K'+(' [diagnostic]' if r['status']!='estimated' else '')
            low,high=r.get('ci95_low_k'),r.get('ci95_high_k')
            interval_name='95% profile CI' if r['method']=='weighted' else '95% CI'
            if low is None and high is None:
                interval=f'{interval_name} unavailable'
            else:
                low_text='unbounded' if low is None else f'{low:.3g}'
                high_text='unbounded' if high is None else f'{high:.3g}'
                interval=f'{interval_name} [{low_text}, {high_text}] K'
            label+='\n'+interval
            grid=np.linspace(t.min(),t.max(),200)
            color={'inverse':'C0','slope':'C1','weighted':'C2'}[r['method']]
            line,=right.plot(grid,(grid-theta)/a,style,color=color,label=label);line.set_gid('cw-method-'+r['method'])
            if theta>=t.min():
                notes.append(name+': pole not below fit T; shown only at right');continue
            line,=left.plot(grid,r['background']+a/(grid-theta),style,color=color,label=label);line.set_gid('cw-method-'+r['method'])
        if notes:left.text(.02,.03,'\n'.join(notes),transform=left.transAxes,fontsize=7,va='bottom')
        for ax,title,ylabel in [(left,'Slope fits','s (MCD/T)'),(right,'Inverse diagnostic','1 / (s - background)')]:
            ax.set(title=branch+' - '+title,xlabel='Temperature (K)',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=7)
    figure.suptitle('Same data and background; line style = method. Left: slope SE; right: first-order propagated SE (approximate).',fontsize=9)
