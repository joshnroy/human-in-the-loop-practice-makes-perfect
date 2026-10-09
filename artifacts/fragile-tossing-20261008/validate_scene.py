import json
import numpy as np
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.agentic_bridge import Tossing3DAgenticBridge
from PIL import Image

results={}
a=Tossing3DEnvironment();b=Tossing3DEnvironment(damage_cost=10)
try:
 a.hard_reset();b.hard_reset()
 ao=a.backend()._object_centric();bo=b.backend()._object_centric()
 ad=ao._robot_env.sim.data.mj_data;bd=bo._robot_env.sim.data.mj_data
 assert np.array_equal(ad.qpos,bd.qpos)
 for _ in range(5):
  ao.step(np.zeros(18));bo.step(np.zeros(18))
  assert np.array_equal(ad.qpos,bd.qpos)
 results['unchanged_physics']='identical qpos after reset and five robot steps'
 model=bo._robot_env.sim.model.mj_model
 joint=model.body('cube_0').jntadr[0];address=model.jnt_qposadr[joint]
 # Drop outside the protective square, on accessible room floor.
 bd.qpos[address:address+3]=[-1,0,1]
 bd.qvel[:]=0
 impacts=[]
 for _ in range(25):
  _,_,_,_,info=bo.step(np.zeros(18));impacts.extend(info['damage_events'])
 assert impacts and all(e['cost']==10 for e in impacts)
 results['outside_impacts']=impacts
 # Move cube over mat (but not over the bin).
 binxy=bd.xpos[model.body('bin_0').id,:2].copy()
 bd.qpos[address:address+3]=[binxy[0]-.5,binxy[1],1]
 bd.qvel[:]=0
 for _ in range(25):
  assert bo.step(np.zeros(18))[4]['damage_cost']==0
 results['inside_drop']='no damage'
 # Human reset follows bin and exempts placement.
 for destination in ('robot_side','opposite_side'):
  assert b.reset_movables(destination=destination)
  b.backend()._env.step(np.zeros(18))
  assert not b.backend().drain_damage_events()
  bo.render()
  mat=model.body('protective_mat_bin_0').id
  assert np.allclose(bd.xpos[mat,:2],bd.xpos[model.body('bin_0').id,:2])
  Image.fromarray(bo.render()).save(f'artifacts/fragile-tossing-20261008/{destination}.png')
 results['human_resets']='both sides follow bin and exempt placement'
 print(json.dumps(results,indent=2))
finally:a.close();b.close()
