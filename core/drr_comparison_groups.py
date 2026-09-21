"""Conservative magnetic-field comparison catalog and persistent membership."""
import hashlib
import json
import os
import re
from pathlib import Path
from uuid import uuid4


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True).encode()).hexdigest()


def product_id(path):
    # Membership is tied to the saved product; numerical cache uses dataset.key.
    return str(Path(path).resolve()).casefold()


def parse_condition(filename):
    name=str(filename).replace('\\','/').rsplit('/',1)[-1].replace('−','-').replace('–','-')
    name=re.sub(r'\.(csv|dat)$','',name,flags=re.I)
    num=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'
    patterns={'spot':r'(?i)(?:^|_)(p\d+n\d+)(?=_|$)',
              'field':rf'(?:^|_)({num})T(?=_|$)',
              'temperature':rf'(?:^|_)({num})K(?:REF)?(?=_|$)',
              'wavelength':rf'(?:^|_)({num})nm(c)?(?=_|$)',
              'rotation':rf'(?:^|_)Rot(?:In)?({num})deg(?=_|$)',
              'gate':rf'(?:^|_)(TG(?:{num})?BG|TG|BG)=({num})(?=_|$)'}
    found={k:re.search(p,name,re.I) for k,p in patterns.items()}
    if not all(found.values()):return None
    # Multiple gate/field tokens are not a single unambiguous condition.
    if any(len(list(re.finditer(p,name,re.I)))!=1 for p in patterns.values()):return None
    sample=name.split('_')[0].upper()
    if not re.fullmatch(r'[A-Za-z]+\d+',sample):return None
    relation=found['gate'][1].upper()
    relation=re.sub(num,lambda m:('+' if m[0].startswith('+') else '')+format(float(m[0]),'.12g'),relation)
    condition={'sample':sample,'spot':found['spot'][1].lower(),
               'gate':relation+'='+format(float(found['gate'][2]),'.12g'),
               'temperature':float(found['temperature'][1]),
               'wavelength':float(found['wavelength'][1]),
               'optical_suffix':(found['wavelength'][2] or '').lower(),
               'rotation':float(found['rotation'][1])}
    # Retain other optical/power/polarization tokens in the key. Exposure and
    # repetition are acquisition details, not comparison conditions.
    remainder=name[len(name.split('_')[0]):]
    for pattern in patterns.values():remainder=re.sub(pattern,'',remainder,flags=re.I)
    extras=[t.casefold() for t in remainder.split('_') if t and not re.fullmatch(r'(?:\d+(?:p\d+|\.\d+)?sx\d+|rep\d+|avg\d+|\d{3})',t,re.I)]
    condition['extra']=sorted(extras)
    return {**condition,'field':float(found['field'][1]),'group_key':digest(condition)}


def build_catalog(paths):
    groups={};entries={}
    for path in paths:
        path=Path(path)
        try:
            meta=json.loads(path.read_text(encoding='utf-8-sig'))
            sources=meta.get('sources',meta.get('inputs',[]))
            names=[s.get('filename') or s.get('source_path') or s.get('name') or s.get('path')
                   for s in sources if isinstance(s,dict) and s.get('role')=='measurement']
            conditions=[parse_condition(n) for n in names if n]
            valid=bool(conditions) and all(c and c==conditions[0] for c in conditions)
            condition=conditions[0] if valid else None
            pid=product_id(path)
            entry={'id':pid,'path':str(path),'name':path.name.removesuffix('.metadata.json'),
                   'condition':condition,'processing':meta.get('processing',{})}
            entries[pid]=entry
            key=condition['group_key'] if condition else 'pending-'+digest(pid)
            label=(f"{condition['sample']} · {condition['spot']} · {condition['gate']}\n"
                   f"{condition['temperature']:g} K · {condition['wavelength']:g} nm · Rot {condition['rotation']:g}°"
                   + (' · '+' / '.join(condition['extra']) if condition['extra'] else '')) if condition else 'Needs review · '+entry['name']
            group=groups.setdefault(key,{'key':key,'label':label,'members':[],'pending':not valid})
            group['members'].append(pid)
        except (OSError,ValueError,TypeError):continue
    for g in groups.values():
        g['members'].sort(key=lambda p:((entries[p]['condition'] or {}).get('field',float('inf')),entries[p]['name']))
    return groups,entries


class ComparisonStore:
    def __init__(self,root):
        self.root=Path(root);self.path=self.root/'membership.json'
        if self.path.exists():
            self.data=json.loads(self.path.read_text(encoding='utf-8'))
            if self.data.get('schema')!=1:raise ValueError('Unsupported comparison catalog version')
        else:self.data={'schema':1,'groups':{}}

    def save(self):
        self.root.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_suffix('.'+uuid4().hex+'.tmp')
        try:
            with temp.open('w',encoding='utf-8') as stream:
                json.dump(self.data,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
            os.replace(temp,self.path)
        finally:
            if temp.exists():temp.unlink()

    def members(self,key,automatic):
        overrides=self.data['groups'].get(key,{})
        return list(dict.fromkeys([p for p in automatic if overrides.get(p,True)]+
                                 [p for p,include in overrides.items() if include]))

    def set_member(self,key,pid,include):
        self.data['groups'].setdefault(key,{})[pid]=bool(include);self.save()

    def reset(self,key):
        self.data['groups'].pop(key,None);self.save()

    def workspace(self,key):return self.root/'groups'/(digest(key)+'.npz')
    def dataset(self,key):return self.root/'datasets'/(key+'.npz')
