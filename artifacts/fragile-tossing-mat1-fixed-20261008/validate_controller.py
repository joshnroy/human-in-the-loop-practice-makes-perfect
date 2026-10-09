"""Prelaunch guard: verify the controller range and reproduce a held-out success."""
import ast,json,sys
from pathlib import Path
import numpy as np
from kinder_models.dynamic3d.tossing.parameterized_skills import MoveToTossLocationAndTossController as Controller
from kinder_models.dynamic3d.tossing.toss_swing import toss_profile_limits
from hitl_pmp.step_protocol import StepEvaluation
root=Path(__file__).resolve().parent
assert getattr(Controller,'MAX_SIMULATION_EFFORT',1)==3, 'Wrong controller dependency: long-range effort extension missing'
assert np.isclose(toss_profile_limits(np.deg2rad(300),max_effort=Controller.MAX_SIMULATION_EFFORT)[0],np.deg2rad(300)), 'Requested long-range speed is capped'
if '--range-only' in sys.argv:
 print('Long-range controller range verified');sys.exit(0)
raw=json.loads((root.parent/'fragile-tossing-mat1-heavy-20261008/smoke/expectimax/pomdp/0/config_snapshot.json').read_text())['args']
args={}
for k,v in raw.items():
 try:args[k]=ast.literal_eval(v)
 except (ValueError,SyntaxError):args[k]=v
args.update(num_test_tasks=1,evaluation_control_steps=500,practice_cost_config=root/'della-bundle/expectimax-h100-d10/costs.json',fragile_object=True,mat_size=1)
result=StepEvaluation.run(snapshot=str(root.parent/'fragile-tossing-mat1-heavy-20261008/failure-audit/old-trained.pickle'),configuration=args,output=str(root/'known-success-check'))
assert result['num_solved']==1, result
(root/'controller-validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
