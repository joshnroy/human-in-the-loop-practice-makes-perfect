"""Read-only 10-minute monitor; never feeds evaluations to the coding agent."""
import argparse,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
def read(p,default=None):
 try:return json.loads(p.read_text())
 except (OSError,json.JSONDecodeError):return default if default is not None else {}
def atomic(p,v):
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(p)
def check(location):
 manifest=read(ROOT/'planned-runs.json');receipts=read(ROOT/f'{location}-launch-receipt.json',{'runs':[]})
 launched={r['name']:r for r in receipts['runs']};rows=[]
 for run in manifest['runs']:
  if run['location']!=location or run['name'] not in launched:continue
  receipt=launched[run['name']];directory=Path(run['result_directory']);status=read(directory/'status.json')
  if location=='della':
   proc=subprocess.run(['sacct','-X','-n','-P','-j',str(receipt['job_id']),'--format=JobIDRaw,State'],text=True,capture_output=True,timeout=30)
   state=next((l.split('|')[1].split()[0].rstrip('+') for l in proc.stdout.splitlines() if l.startswith(str(receipt['job_id'])+'|')),'UNKNOWN')
   terminal=state in TERMINAL
  else:
   proc=subprocess.run(['systemctl','--user','show',run['unit'],'--property=ActiveState','--value'],text=True,capture_output=True,timeout=20)
   state=proc.stdout.strip();terminal=state in {'inactive','failed'}
  evals=[];errors=[]
  for p in sorted((directory/'evaluations').glob('*/results.json')):
   e=read(p)
   if e.get('complete'):
    evals.append(dict(index=p.parent.name,solved=e.get('num_solved'),total=e.get('num_total')))
    errors.extend(t.get('error') for t in e.get('tasks',[]) if t.get('error') and t.get('error') != 'missing_controller')
  warnings=[]
  if terminal and status.get('phase')!='complete':warnings.append('process ended before clean completion')
  if errors:warnings.append(f'{len(errors)} evaluation task errors')
  paths=[directory/'status.json',directory/'practice.jsonl',directory/'step_events.jsonl',directory/'coding/stream.jsonl']
  age=time.time()-max((p.stat().st_mtime for p in paths if p.exists()),default=time.time())
  if not terminal and age>1800 and state not in {'PENDING','CONFIGURING'}:warnings.append('no activity for 30 minutes')
  rows.append(dict(name=run['name'],state=state,terminal=terminal,status=status,evaluations=evals,warnings=warnings))
 report=dict(checked_unix=time.time(),interval_seconds=600,runs=rows)
 previous=read(ROOT/f'{location}-monitor-latest.json')
 signature=lambda report:[(r['name'],r['terminal'],r['warnings'],r['evaluations']) for r in report.get('runs',[])]
 if signature(previous)!=signature(report):
  with (ROOT/f'{location}-monitor-events.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
  # Notify the workstation user only for completion or a actionable failure.
  changed=[r for r in rows if r['terminal'] or r['warnings']]
  if location=='workstation' and changed:
   subprocess.run(['notify-send','Fragile Tossing3D', '; '.join(r['name']+': '+(', '.join(r['warnings']) or 'complete') for r in changed)],capture_output=True,check=False)
 atomic(ROOT/f'{location}-monitor-latest.json',report)
 with (ROOT/f'{location}-monitor-history.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
 print(json.dumps(dict(checked_unix=report['checked_unix'],runs=[dict(name=r['name'],state=r['state'],steps=r['status'].get('practice_steps'),evals=len(r['evaluations']),warnings=r['warnings']) for r in rows])),flush=True)
 expected=sum(r['location']==location for r in manifest['runs'])
 return len(rows)==expected and all(r['terminal'] for r in rows)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--location',choices=['della','workstation'],required=True);parser.add_argument('--once',action='store_true');args=parser.parse_args()
 while True:
  try:done=check(args.location)
  except Exception as e:
   print(json.dumps(dict(error=str(e))),flush=True);done=False
  if args.once or done:break
  time.sleep(600)
