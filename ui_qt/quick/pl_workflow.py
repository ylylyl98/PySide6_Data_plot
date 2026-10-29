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
    display = context.get("display_state")
    if display is not None:
        from core.pl_display_settings import source_context, validate_snapshot
        identity = source_context(folder, path, cube, request)
        display = validate_snapshot({**display, **identity})
        if display is None:
            raise ValueError("The PL display settings are invalid; review the ranges before saving.")
        extra.update(pl_display=display, pl_display_save_pending=True)
    paths=export_pl_pngs_and_dat(str(folder),path.name,cube_linear=cube,cube_log=cube,
        params_linear=params["linear"],params_log=params["log"],include_pair=False,include_original_png=True,
        processed_name=str(Path("Processed Data")/"PL"),
        metadata_input_files=(("measurement",relative),),metadata_extra=extra)
    verify_signatures(signatures)
    if display is not None:
        from core.pl_display_settings import publish_snapshot
        publish_snapshot(folder, path, display, paths["dat"])
    return paths


def y_axis_spec(choice, a=1., b=-1., c=0.):
    choices = {"Auto / Default":"auto","TG+BG":"linear:1,1,0","TG-BG":"linear:1,-1,0","TG":"tg","BG":"bg","Bias":"bias","Doping (V)":"doping","Efield (V)":"efield"}
    if choice in choices:return choices[choice]
    if choice == "Advanced…":return f"linear:{float(a):.17g},{float(b):.17g},{float(c):.17g}"
    return "auto"


def refresh_bank(cube, bounds, bank, *, force=False, geometry=None, regions=None):
    """Refresh ROI statistics without changing explicit locks or geometry.

    ``geometry`` is shared by the two native PL color banks. The legacy
    split/splitX representation remains accepted for older callers.
    ``force`` bypasses the ROI cache; it never releases a fixed endpoint.
    """
    import numpy as np
    from types import SimpleNamespace
    from core.color_limits import ColorRequest, calculate_bounds
    values = dict(bank)
    values["fixed"] = dict(bank.get("fixed",{}))
    roi = tuple(bounds)
    count = int(geometry["count"]) if geometry else (2 if bank.get("split") else 1)
    cuts = tuple(geometry[key] for key in ("x0", "x1")[:count-1]) if geometry else ((bank["splitX"],) if count > 1 else ())
    signature = (*roi, count, *cuts)
    if not force and tuple(bank.get("roi",())) == signature:return values
    fixed = values["fixed"]
    sides = ("left", "right") if count == 2 else ("left", "middle", "right")
    selected = (None, *sides) if regions is None else tuple(regions)
    for side in selected:
        if side is not None and (count == 1 or side not in sides):
            continue
        names = ("vmin", "vmax") if side is None else (side+"Min", side+"Max")
        selected_cube = cube
        if side is not None:
            # Canonical boundaries may lie outside this file or its viewport.
            # Slice by physical cell boundaries rather than snapping an
            # off-data boundary to an arbitrary interior column.
            index = sides.index(side)
            energy = np.asarray(cube.energy)
            lower = cuts[index-1] if index else -np.inf
            upper = cuts[index] if index < len(cuts) else np.inf
            mask = (energy >= lower) & (energy < upper)
            selected_cube = SimpleNamespace(energy=energy[mask], gate=cube.gate, Z=np.asarray(cube.Z)[:,mask])
        request = ColorRequest(tuple(bounds[:2]), tuple(bounds[2:]), (), None,
                               bool(values["log"]), False, "robust", False)
        try: limits = calculate_bounds((selected_cube,), request)
        except (ValueError, IndexError):continue
        if limits is None:continue
        candidate = dict(values)
        for key,value in zip(names,limits):
            if not fixed.get(key,False):candidate[key] = value
        if candidate[names[0]] >= candidate[names[1]]:
            # A single fixed endpoint must not prevent its unfixed peer from
            # following Auto, nor may Auto move that fixed endpoint.
            low, high = names
            pad = max(abs(candidate[low])*.01, abs(candidate[high])*.01, 1e-12)
            if not fixed.get(high,False):candidate[high] = candidate[low]+pad
            elif not fixed.get(low,False):
                candidate[low] = candidate[high]/1.01 if values["log"] else candidate[high]-pad
        if candidate[names[0]] < candidate[names[1]] and (not values["log"] or candidate[names[0]] > 0):
            values.update({key:candidate[key] for key in names})
    values["roi"] = list(signature)
    return values


def heatmap_params(base,bank,*,geometry=None):
    """Project canonical regions onto the viewport, retaining their colors."""
    from core.plotting import SplitColorScale
    count = int(geometry["count"]) if geometry else (2 if bank.get("split") else 1)
    if count == 1:
        return replace(base,vmin=bank["vmin"],vmax=bank["vmax"],log_scale=bank["log"],split_scale=None)
    cuts = tuple(geometry[key] for key in ("x0", "x1")[:count-1]) if geometry else (bank["splitX"],)
    sides = ("left", "right") if count == 2 else ("left", "middle", "right")
    low, high = sorted(base.xlim)
    if geometry and "_dataX" in geometry:
        low, high = max(low,geometry["_dataX"][0]), min(high,geometry["_dataX"][1])
    active = tuple(sorted({cut for cut in cuts if low < cut < high}))
    indices = (sum(cut <= low for cut in cuts), *(sum(cut <= boundary for cut in cuts) for boundary in active))
    pairs = tuple((bank[sides[index]+"Min"],bank[sides[index]+"Max"]) for index in indices)
    if not active:
        return replace(base,vmin=pairs[0][0],vmax=pairs[0][1],log_scale=bank["log"],split_scale=None)
    split = SplitColorScale(active[0],*pairs[0],*pairs[-1],
                            geometry.get("showBoundary",True) if geometry else bank.get("showBoundary",True),
                            active[1] if len(active)==2 else None,
                            pairs[1][0] if len(active)==2 else None,pairs[1][1] if len(active)==2 else None)
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
