import json,subprocess,sys,shlex
from pathlib import Path
root=Path(__file__).resolve().parents[2];here=Path(__file__).resolve().parent
for name,arm in [('ees','ees'),('expectimax','pddl')]:
 args=json.loads((here/'della-bundle'/f'{name}-h100-d10'/f'{arm}-arguments.json').read_text())
 method=args[args.index('--method')+1]
 del args[args.index('--method'):args.index('--method')+2]
 del args[args.index('--env'):args.index('--env')+2]
 for flag,value in {'--practice-step-budget':'40','--measurement-interval-steps':'20','--evaluation-control-steps':'50','--stop-after-perfect-evaluations':'0','--practice-cost-config':str(here/'della-bundle'/f'{name}-h100-d10/costs.json')}.items():
  args[args.index(flag)+1]=value
 if '--num-test-tasks' in args:args[args.index('--num-test-tasks')+1]='1'
 else:args+=['--num-test-tasks','1']
 command=[sys.executable,'-m','scripts.run_sweep','--env','tossing3d','--methods',method,'--num-seeds','1','--max-workers','1','--results-root',str(here/'smoke'/name),'--shared-args',shlex.join(args)]
 print('SMOKE',name,flush=True)
 subprocess.run(command,cwd=root/'.codex-worktrees/fragile-tossing-integration',check=True)
