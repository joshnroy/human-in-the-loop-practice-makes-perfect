import ast,json,sys
from pathlib import Path
if len(sys.argv)>2 and sys.argv[2]=='old-controller':
    sys.path.insert(0,str(Path.cwd()/'artifacts/human-cost-high-sweep-seed0-20261008/source/reference/kinder-baselines/kinder-models/src'))
from hitl_pmp.step_protocol import StepEvaluation
root=Path(__file__).resolve().parent
raw=json.loads((root.parent/'smoke/expectimax/pomdp/0/config_snapshot.json').read_text())['args']
args={}
for k,v in raw.items():
 try:args[k]=ast.literal_eval(v)
 except (ValueError,SyntaxError):args[k]=v
args.update(num_test_tasks=1,evaluation_control_steps=500,practice_cost_config=root.parent/'della-bundle/expectimax-h100-d10/costs.json',fragile_object=sys.argv[1]=='fragile')
r=StepEvaluation.run(snapshot=str(root/'old-trained.pickle'),configuration=args,output=str(root/('old-policy-'+sys.argv[1]+('-old-controller' if len(sys.argv)>2 else ''))))
print(json.dumps(r),flush=True)
