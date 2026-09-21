"""Non-destructive range selection over already computed peak candidates."""
from copy import deepcopy


def keep_point(point, bounds):
    if bounds is None:return True
    if not bounds.get('show_low_confidence',True) and point.get('confidence')=='low':return False
    if bounds.get('limit_x',True) and not bounds['x_min'] <= point['energy'] <= bounds['x_max']:return False
    if bounds.get('limit_y') and not bounds['y_min'] <= point['y'] <= bounds['y_max']:return False
    if bounds.get('min_prominence',0)>0 and point.get('auto_prominence_fraction',point.get('prominence_fraction',0))<bounds['min_prominence']:return False
    if bounds.get('min_width_mev',0)>0 and point.get('auto_width_mev',point.get('width_mev',0))<bounds['min_width_mev']:return False
    if bounds.get('max_width_mev',0)>0 and point.get('auto_width_mev',point.get('width_mev',float('inf')))>bounds['max_width_mev']:return False
    return True


def filtered_result(result, bounds, *, for_display=False):
    """Build an independent export, or a lightweight UI view of retained points.

    Display metadata is read-only/shared; point dictionaries are copied before
    branch linking. Hidden candidates remain in the original result, not the UI
    cache. Export retains the full audit trail through the default path.
    """
    if for_display and result is not None:
        result={**result,'products':{source:dict(product) for source,product in result['products'].items()}}
        result.pop('filtered_out_points',None)
    else:result=deepcopy(result)
    if result is None or bounds is None:return result
    result['result_filter']=dict(bounds)
    if not for_display:result['filtered_out_points']=[]
    for source,product in result['products'].items():
        original=product['points'];retained=[];retained_rows={}
        rows={}
        for point in original:
            if keep_point(point,bounds):rows.setdefault(point['row_index'],[]).append(dict(point) if for_display else point)
        for row,points in sorted(rows.items()):
            chosen=[]
            for point in sorted(points,key=lambda p:(-p.get('auto_prominence',p['prominence']),p['energy'])):
                if bounds.get('min_distance_mev',0)>0 and any(p['polarity']==point['polarity'] and abs(p['energy']-point['energy'])<bounds['min_distance_mev']/1000 for p in chosen):continue
                chosen.append(point)
                if bounds.get('max_peaks',0)>0 and len(chosen)>=bounds['max_peaks']:break
            retained.extend(chosen)
            if chosen:retained_rows[row]=chosen
        if not for_display:
            identities={id(p) for p in retained}
            result['filtered_out_points'].extend({'source':source,**point} for point in original if id(point) not in identities)
        if result.get('candidate_pool') and bounds.get('link_tracks',True):
            from core.drr_peak_analysis import _link
            previous=[];previous_row=None;next_id=1
            for row,current in sorted(retained_rows.items()):
                current=sorted(current,key=lambda p:p['energy'])
                if previous_row is None or row!=previous_row+1:previous=[]
                next_id=_link(previous,current,next_id,bounds.get('max_shift_mev',3)/1000)
                previous=current;previous_row=row
        product['points']=retained
    return result
