"""Submit the two $20 agentic pilots once after the structured jobs are submitted."""
import json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'planned-runs.json').read_text());p=root/'workstation-launch-receipt.json'
receipt=json.loads(p.read_text()) if p.exists() else {'runs':[]}
for run in manifest['runs']:
 if run['location']!='workstation' or any(r['name']==run['name'] for r in receipt['runs']):continue
 d=Path(run['directory']);assert not Path(run['result_directory']).exists(),'Refuse overwrite'
 command=json.loads((d/'launch-command.json').read_text())
 result=subprocess.run(command,text=True,capture_output=True,check=True)
 receipt['runs'].append({**run,'launched_unix':time.time(),'service_response':result.stderr.strip()})
 p.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(name=run['name'],unit=run['unit'])),flush=True)
unit='hitl-fragile-monitor-10m-20261008'
subprocess.run(['systemd-run','--user',f'--unit={unit}','/usr/bin/python3',str(root/'monitor.py'),'--location','workstation'],check=True)
print(json.dumps(dict(monitor_unit=unit,interval_seconds=600)),flush=True)
