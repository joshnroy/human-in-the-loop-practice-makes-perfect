import json
from pathlib import Path
from hitl_pmp.agentic_runtime.sandbox import SandboxSettings
from hitl_pmp.full_agentic.runner import FullAgenticRunner
root=Path(__file__).resolve().parent
settings=SandboxSettings.model_validate_json((root.parent/'serialization-relaunch-20261008/continuation-agentic-c01/sandbox.json').read_text())
FullAgenticRunner.validate_runtime(settings=settings)
result=FullAgenticRunner.evaluate(submission=root/'eval-smoke-controller',settings=settings,output=root/'agentic-eval-smoke',count=1,budget=2,seed=0,damage_cost=10,mat_size=4)
assert result[0]['steps']==2,result
assert result[0]['error'] is None,result
assert 'damage_cost' in result[0],result
print(json.dumps(result))
