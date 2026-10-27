"""Captured PL loading, normalization and manual spectrum calculations.

The native application keeps its existing PL state owner. These operations are
Qt-free and can run in its worker pool without a legacy controller.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class PlRequest:
    path: str
    y_axis: str = "auto"
    dat_choice: str = "Y"
    dat_label: str = ""
    dat_unit: str = ""
    processing: str = "none"
    window: int = 21
    polynomial: int = 2


def load_pl_source(request):
    from core.data_io import load_pl_cube
    from core.loader import load_dat, resolve_dat_y_axis
    from core.processing import apply_sg_derivative_energy
    path = Path(request.path)
    signatures=input_signatures(path)
    cube = load_dat(path) if path.suffix.casefold() == ".dat" else load_pl_cube(str(path.parent),path.name,y_axis=request.y_axis)
    if path.suffix.casefold() == ".dat" and request.dat_choice != "original":
        label,unit,semantic = resolve_dat_y_axis(request.dat_choice,custom_label=request.dat_label,custom_unit=request.dat_unit)
        cube = replace(cube,gate_label=label,gate_unit=unit,y_axis_semantic=semantic)
    if request.processing != "none":
        cube,_ = apply_sg_derivative_energy(cube,derivative={"smooth":0,"first":1,"second":2}[request.processing],
            window_length=request.window,polyorder=request.polynomial)
    for array in (cube.energy,cube.gate,cube.Z): array.setflags(write=False)
    verify_signatures(signatures)
    cube._quick_pl_signatures=signatures
    cube._quick_pl_request=request
    return cube


def input_signatures(path):
    from hashlib import sha256
    from core.load_cancellation import check_load_cancelled
    path=Path(path).resolve();paths=[path]
    if path.suffix.casefold()==".dat":
        from core.loader import _dat_sidecar_candidates
        paths.extend(_dat_sidecar_candidates(path))
    result=[]
    for source in paths:
        check_load_cancelled()
        if not source.exists():
            result.append((str(source.resolve()),None));continue
        before=source.stat();digest=sha256()
        with source.open("rb") as stream:
            while chunk:=stream.read(1024*1024):check_load_cancelled();digest.update(chunk)
        after=source.stat()
        signature=lambda stat:(stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if signature(before)!=signature(after):raise ValueError("PL source changed during reading; load it again.")
        result.append((str(source.resolve()),*signature(after),digest.hexdigest()))
    return tuple(result)


def verify_signatures(signatures):
    if not signatures:return
    current=input_signatures(signatures[0][0])
    if current!=signatures:raise ValueError("PL measurement or metadata changed after loading; reload before export.")


def export_pl(path,cube,params,context):
    from core.export import export_pl_pngs_and_dat
    from core.data_io import discover_pl_processing_status
    path=Path(path).resolve();folder=Path(context["folder"])
    signatures=getattr(cube,"_quick_pl_signatures",())
    verify_signatures(signatures)
    try:relative=path.relative_to(folder).as_posix()
    except ValueError:relative=str(path)
    before=discover_pl_processing_status(folder,[relative])
    context["wasProcessed"]=relative in before or path.suffix.casefold()==".dat"
    request=context.get("request")
    extra=dict(native_pl_axis=dict(y_axis=getattr(request,"y_axis","auto"),dat_choice=getattr(request,"dat_choice","original"),
        dat_label=getattr(request,"dat_label",""),dat_unit=getattr(request,"dat_unit","")))
    paths=export_pl_pngs_and_dat(str(folder),path.name,cube_linear=cube,cube_log=cube,
        params_linear=params["linear"],params_log=params["log"],include_pair=False,
        processed_name=str(Path("Processed Data")/"PL"),
        metadata_input_files=(("measurement",relative),),metadata_extra=extra)
    verify_signatures(signatures)
    return paths


def y_axis_spec(choice, a=1., b=-1., c=0.):
    choices = {"Auto / Default":"auto","TG+BG":"linear:1,1,0","TG-BG":"linear:1,-1,0","TG":"tg","BG":"bg","Bias":"bias","Doping (V)":"doping","Efield (V)":"efield"}
    if choice in choices:return choices[choice]
    if choice == "Advanced…":return f"linear:{float(a):.17g},{float(b):.17g},{float(c):.17g}"
    return "auto"


def refresh_bank(cube, bounds, bank, *, force=False):
    """Refresh every unlocked bound independently, including split regions."""
    from core.processing import compute_auto_limits
    from core.plotting import resolve_split_boundary
    values = dict(bank)
    values["fixed"] = dict(bank.get("fixed",{}))
    roi = tuple(bounds)
    if not force and tuple(bank.get("roi",())) == roi:return values
    fixed = values["fixed"]
    def update(names,xlim):
        if xlim[0] >= xlim[1]:return
        try: limits = compute_auto_limits(cube,log_scale=values["log"],xlim=tuple(xlim),ylim=tuple(bounds[2:]))
        except ValueError:return
        candidate = dict(values)
        for key,field in zip(names,("vmin","vmax")):
            if force or not fixed.get(key,False):candidate[key] = getattr(limits,field)
        if candidate[names[0]] < candidate[names[1]]:values.update({key:candidate[key] for key in names})
    update(("vmin","vmax"),bounds[:2])
    if values.get("split"):
        low,high = bounds[:2]
        boundary = values["splitX"]
        if not low < boundary < high and not fixed.get("splitX",False):boundary = (low+high)/2
        try: _,boundary = resolve_split_boundary(cube.energy,boundary)
        except ValueError: pass
        values["splitX"] = float(boundary)
        update(("leftMin","leftMax"),(low,min(boundary,high)))
        update(("rightMin","rightMax"),(max(low,boundary),high))
    values["roi"] = list(roi)
    return values


def heatmap_params(base,bank):
    from core.plotting import SplitColorScale
    split = None
    if bank.get("split"):
        split = SplitColorScale(bank["splitX"],bank["leftMin"],bank["leftMax"],bank["rightMin"],bank["rightMax"],bank.get("showBoundary",True))
        low,high = sorted(base.xlim)
        if high <= split.split_x:return replace(base,vmin=split.left_vmin,vmax=split.left_vmax,split_scale=None)
        if low >= split.split_x:return replace(base,vmin=split.right_vmin,vmax=split.right_vmax,split_scale=None)
    return replace(base,vmin=bank["vmin"],vmax=bank["vmax"],log_scale=bank["log"],split_scale=split)


def find_spectrum_peaks(x,y,bounds,settings):
    import numpy as np
    from scipy.signal import find_peaks
    x,y = np.asarray(x,float),np.asarray(y,float)
    mask = (x>=min(bounds))&(x<=max(bounds))&np.isfinite(x)&np.isfinite(y)
    indices = np.flatnonzero(mask)
    if len(indices)<3:return np.array([],dtype=int)
    span = float(np.ptp(y[indices]));prominence=max(0.,settings["prominence"]*span)
    # Preserve finite segments, so a NaN gap cannot manufacture a detection.
    cuts = np.flatnonzero(np.diff(indices)>1)+1
    output,strength = [],[]
    for segment in np.split(indices,cuts):
        target = -y[segment] if settings["mode"] == "dips" else y[segment]
        found,props = find_peaks(target,prominence=prominence if prominence>0 else None,distance=max(1,settings["distance"]))
        output.extend(segment[found]);strength.extend(props.get("prominences",np.zeros(len(found))))
    if len(output)>settings["maximum"]:
        order = np.argsort(strength,kind="stable")[::-1][:settings["maximum"]]
        output = np.asarray(output)[order]
    return np.sort(np.asarray(output,dtype=int))


def multi_lorentz(x,*parameters):
    import numpy as np
    x = np.asarray(x,float)
    result = parameters[0]+parameters[1]*x
    for index in range((len(parameters)-2)//3):
        amplitude,center,width = parameters[2+3*index:5+3*index]
        width = max(1e-12,width)
        result = result+amplitude*width**2/((x-center)**2+width**2)
    return result


def fit_spectrum(x,y,bounds,settings,indices=(),*,cancellation=None):
    import numpy as np
    from scipy.optimize import curve_fit
    x,y = np.asarray(x,float),np.asarray(y,float)
    mask=(x>=min(bounds))&(x<=max(bounds))&np.isfinite(x)&np.isfinite(y)
    if np.count_nonzero(mask)<8:raise ValueError("Fit needs at least eight finite samples in the displayed Energy range.")
    selected_x,selected_y=x[mask],y[mask]
    count=int(settings["fitCount"])
    if not 1<=count<=8:raise ValueError("Fit supports one to eight Lorentz peaks.")
    indices=np.asarray(indices,dtype=int)
    indices=indices[(indices>=0)&(indices<len(x))]
    if not len(indices):indices=find_spectrum_peaks(x,y,bounds,settings)
    if len(indices)<count:
        order=np.argsort(abs(selected_y-np.median(selected_y)))[::-1]
        indices=np.unique(np.r_[indices,np.flatnonzero(mask)[order[:max(1,count-len(indices))]]])
    if not len(indices):raise ValueError("No peak candidates to fit.")
    centers=np.sort(x[indices])[:count]
    while len(centers)<count:centers=np.r_[centers,np.mean(selected_x)]
    base=float(np.median(selected_y));amplitude=max(float(np.max(abs(selected_y-base))),1e-12)
    span=max(float(np.ptp(selected_x)),1e-9);step=abs(float(np.median(np.diff(selected_x))))
    width=max(step*2,span/80,1e-6)
    initial=[base,0.];lower=[float(selected_y.min()-3*amplitude),-1e9];upper=[float(selected_y.max()+3*amplitude),1e9]
    for center in centers:
        initial.extend([amplitude*.7*(-1 if settings["mode"]=="dips" else 1),float(center),width])
        lower.extend([-5*amplitude,float(selected_x.min()),max(step*.25,1e-8)])
        upper.extend([5*amplitude,float(selected_x.max()),span])
    def model(x,*parameters):
        if cancellation is not None:cancellation.check()
        return multi_lorentz(x,*parameters)
    parameters,covariance=curve_fit(model,selected_x,selected_y,p0=initial,bounds=(lower,upper),maxfev=50000)
    return dict(x=x.copy(),y=multi_lorentz(x,*parameters),centers=np.sort(parameters[3::3]),parameters=parameters,covariance=covariance)
