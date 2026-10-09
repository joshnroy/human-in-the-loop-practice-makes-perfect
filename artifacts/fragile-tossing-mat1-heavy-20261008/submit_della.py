"""Submit each validated structured pilot once and start an independent monitor."""
import json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
source=root/'source'
if not source.exists():
 subprocess.run(['tar','-xzf',str(root/'source.tar.gz'),'-C',str(root)],check=True)
assets=source/'reference/kindergarden/src/kinder/envs/dynamic3d/models/assets/mimiclabs_scenes'
old=Path('/scratch/gpfs/TSILVER/jr2860/experiments/normalized-expectimax-seed0-20261007/grid/source/reference/kindergarden/src/kinder/envs/dynamic3d/models/assets/mimiclabs_scenes')
for name in ['meshes','textures']:
 destination=old/name
 assert destination.is_dir(),destination
 p=assets/name
 if p.is_symlink():p.unlink()
 if not p.exists():p.symlink_to(destination.resolve(),target_is_directory=True)
manifest=json.loads((root/'planned-runs.json').read_text())
receipt=root/'della-launch-receipt.json'
records=json.loads(receipt.read_text()) if receipt.exists() else {'runs':[]}
for run in manifest['runs']:
 if any(r['name']==run['name'] for r in records['runs']):continue
 d=Path(run['directory']);(d/'logs').mkdir(exist_ok=True)
 if not (d/'source').exists():(d/'source').symlink_to(source,target_is_directory=True)
 assert not Path(run['result_directory']).exists(),'Refuse overwrite'
 result=subprocess.run(['sbatch','--parsable','--job-name=fragile-mat1-heavy-'+run['method']+'-h100-d10',str(d/'run.sbatch'),run['arm']],capture_output=True,text=True,check=True)
 job=int(result.stdout.strip().split(';')[0]);records['runs'].append({**run,'job_id':job,'launched_unix':time.time()})
 receipt.write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(dict(name=run['name'],job_id=job)),flush=True)
session='fragile-mat1-heavy-monitor-10m-20261008'
if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode:
 subprocess.run(['tmux','new-session','-d','-s',session,'python3',str(root/'monitor.py'),'--location','della'],check=True)
print(json.dumps(dict(monitor_session=session,interval_seconds=600)),flush=True)
