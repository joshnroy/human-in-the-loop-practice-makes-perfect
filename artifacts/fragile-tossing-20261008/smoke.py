from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.agentic_bridge import Tossing3DAgenticBridge
import numpy as np
from PIL import Image

e=Tossing3DEnvironment(damage_cost=10)
try:
 e.hard_reset()
 b=e.backend();o=b._object_centric();print('ENV',type(o).__name__,o.damage_info(),flush=True)
 sim=o._robot_env.sim;model,data=sim.model.mj_model,sim.data.mj_data
 print('MATS',[(model.geom(i).name,int(model.geom_contype[i])) for i in range(model.ngeom) if 'mat_visual' in model.geom(i).name],flush=True)
 bridge=Tossing3DAgenticBridge(env=e,observation_mode='object_state')
 bridge.step(action=np.zeros(18))
 print('STEP',bridge.damage_events(),bridge.observe()['fragile_object'],flush=True)
 Image.fromarray(o.render()).save('artifacts/fragile-tossing-20261008/scene.png')
finally:e.close()
