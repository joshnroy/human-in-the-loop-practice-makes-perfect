"""Prepare four immutable run configurations; never submits a job."""
import json, shutil, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
REMOTE=Path('/scratch/gpfs/TSILVER/jr2860/experiments/fragile-tossing-mat1-heavy-seed0-20261008')
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
cost=json.loads((ROOT/'artifacts/human-cost-high-sweep-seed0-20261008/agentic-c100/costs.json').read_text())
cost['damage_contact']={'value':10}
manifest=dict(environment='FragileTossing3D',mat_size=1,bin_mass_kg=100,damage_cost=10,human_cost=100,seed=0,
 practice_steps=85000,measurement_interval=1700,evaluation_tasks=10,evaluation_steps=500,
 stop_after_perfect_evaluations=3,monitor_interval_seconds=600,runs=[])
for method in ['ees','expectimax']:
 name=f'{method}-h100-d10'; run=HERE/'della-bundle'/name; remote=REMOTE/name
 arm='ees' if method=='ees' else 'pddl'
 template=ROOT/'artifacts/human-cost-high-sweep-seed0-20261008/della-bundle'/f'{method}-c100'
 args=json.loads((template/f'{arm}-arguments.json').read_text())
 args[args.index('--practice-cost-config')+1]=str(remote/'costs.json')
 args+=['--fragile-object','--mat-size','1']
 dump(run/f'{arm}-arguments.json',args);dump(run/'costs.json',cost)
 shutil.copy2(template/'launch_seed.py',run/'launch_seed.py')
 batch=(template/'run.sbatch').read_text().replace(str(Path('/scratch/gpfs/TSILVER/jr2860/experiments/human-cost-high-sweep-seed0-20261008')/f'{method}-c100'),str(remote))
 (run/'run.sbatch').write_text(batch)
 manifest['runs'].append(dict(name=name,method=method,location='della',arm=arm,directory=str(remote),
   result_directory=str(remote/'results'/('ees' if method=='ees' else 'pomdp')/'0')))
for variant in ['original','new-wording']:
 name=f'agentic-{variant}-h100-d10';run=HERE/name;run.mkdir(exist_ok=True)
 dump(run/'costs.json',cost)
 settings=json.loads((ROOT/'artifacts/serialization-relaunch-20261008/continuation-agentic-c01/sandbox.json').read_text())
 settings.update(robocode_checkout=str(HERE/'source/robocode'),artifact_dir=str(run/'execution'),resource_pool_dir=str(run/'resource-pool'))
 dump(run/'sandbox.json',settings)
 unit=f'hitl-fragile-mat1-heavy-{variant}-h100-d10-seed0-20261008'
 cmd=['systemd-run','--user',f'--unit={unit}','--slice=hitlagentichost.slice','-p','MemoryMax=15G','-p','CPUQuota=600%','-p','OOMPolicy=continue',f'--working-directory={HERE}/source',
  '--setenv=TMPDIR=/tmp/hitl-agentic-scratch',f'--setenv=GIT_CEILING_DIRECTORIES={HERE}',
  str(HERE/'source/scripts/with_step_env.sh'),'python',str(HERE/'with_claude_token.py'),'/home/josh/.bashrc',str(HERE/'launch_runner.py'),
  '--sandbox-settings',str(run/'sandbox.json'),'--output',str(run/'run'),'--seed','0','--model-budget','20',
  '--practice-step-budget','85000','--measurement-interval-steps','1700','--human-skill-steps','1',
  '--evaluation-control-steps','500','--num-test-tasks','10','--practice-cost-config',str(run/'costs.json'),
  '--stop-after-perfect-evaluations','3','--fragile-object','--mat-size','1','--prompt-variant',variant]
 dump(run/'launch-command.json',cmd)
 manifest['runs'].append(dict(name=name,method='agentic',prompt_variant=variant,location='workstation',
  directory=str(run),result_directory=str(run/'run'),unit=unit,model='claude-opus-5-5',effort='high',model_budget=20))
for file in ['with_claude_token.py','launch_runner.py']:
 shutil.copy2(ROOT/'artifacts/serialization-relaunch-20261008'/file,HERE/file)
dump(HERE/'planned-runs.json',manifest)
dump(HERE/'della-bundle/planned-runs.json',{**manifest,'runs':[r for r in manifest['runs'] if r['location']=='della']})
print('Prepared exactly four runs.')
