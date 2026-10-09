"""Post-hoc endpoint coverage; not physics contact counts or smaller-mat reruns."""
import json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def required_side(c,b):
    # Full enclosing sphere is conservative for the 5cm cube at any orientation.
    w,x,y,z=b[3:7]
    yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    dx,dy=c[0]-b[0],c[1]-b[1];co,si=math.cos(yaw),math.sin(yaw)
    return 2*(max(abs(co*dx+si*dy),abs(-si*dx+co*dy))+.025*math.sqrt(3))

run=ROOT/'agentic-original-h100-d10/run'
ends={}
for line in (run/'coding/sandbox/logs/episodes.jsonl').open():
    r=json.loads(line)['res']
    if any(isinstance(e,dict) and e.get('ev')=='rest' for e in r['log']):
        ends[r['acct']['steps']]=None
for line in (run/'practice.jsonl').open():
    r=json.loads(line);v=r.get('response',{}).get('result',{})
    if not isinstance(v,dict) or v.get('practice_steps') not in ends or 'observation' not in v:continue
    o={s['name']:s['features'] for s in v['observation']['objects']}
    ends[v['practice_steps']]=required_side([o['cube_0'][k] for k in 'xyz'],[o['bin_0'][k] for k in ['x','y','z','qw','qx','qy','qz']])
agentic=[s for s in ends.values() if s is not None]
structured=[];name=None;last=None

def finish():
    if name=='MoveToTossLocationAndToss' and last is not None:
        c,b=last['cube_0'],last['bin_0']
        if c[2]<.08:structured.append(required_side(c,b))
for line in (ROOT/'expectimax-audit/tossing3d_state_log.jsonl').open():
    r=json.loads(line)
    if r['kind']=='skill':finish();name=r['name'];last=None
    elif r['kind']=='tick':last=r['state']
finish()
report={'note':'Observed endpoint coverage including conservative cube footprint; excludes flight/bounces and changed behavior. Bin may move in these old runs. Not smaller-mat success predictions.'}
for name,values in [('agentic_original',agentic),('expectimax',structured)]:
    report[name]={'endpoints':len(values),'max_required_side_m':max(values),'covered':{str(s):sum(v<=s for v in values) for s in [1,1.5,2,2.5,3,4]}}
(ROOT/'mat-size-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
