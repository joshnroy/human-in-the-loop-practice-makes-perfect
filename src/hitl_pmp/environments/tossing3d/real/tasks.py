"""Real Tossing3D tasks: scene seeds, as in the sim, drawn without touching the room.

The sim's `Tossing3DTasks.build_task` rebuilds the scene to obtain a task's initial
state. On the real robot that is a person setting up a scene, so drawing the test set
up front would ask for every scene before the run starts and again when each episode
begins. A real task's initial state is therefore *planned*: it carries the scene seed
(which is all `set_state` reads, and what every evaluation records) and no observed
features. The person sets the scene up once, when `reset_to_task` asks for it.
"""

import math

from hitl_pmp.core.problem.tasks.types import Goal, Task
from hitl_pmp.environments.tossing3d.predicates import IN_BIN
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks
from hitl_pmp.environments.tossing3d.types import Tossing3DObservation

from .environment import RealTossing3DEnvironment


class RealTossing3DTasks(Tossing3DTasks):
    """`Tossing3DTasks` whose tasks are planned scenes rather than observed ones."""

    env: RealTossing3DEnvironment

    def build_task(self, *, scene_seed: int) -> Task:
        env = self.env
        unknown = {
            obj.name: dict.fromkeys(obj.type.feature_names, math.nan)
            for obj in (env.cube, env.barrier)
        }
        observation = Tossing3DObservation(
            features={
                "robot": dict.fromkeys(env.robot_type.feature_names, math.nan),
                **unknown,
                env.bin.name: {"x": math.nan, "y": math.nan, "z": math.nan},
            },
            goal_region=(math.nan,) * 6,
            solved=False,
        )
        initial_state = env.build_state(observation=observation, seed=scene_seed, steps_taken=0)
        return Task(
            initial_state=initial_state,
            goal=Goal(atoms=frozenset({IN_BIN(state=initial_state, objects=(env.cube, env.bin))})),
        )
