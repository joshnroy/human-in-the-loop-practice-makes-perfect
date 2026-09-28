"""kinder-models' own sweep3D skills (OpenDrawer, PickWiper, Sweep) -- the practice attempt.

Two planning-model fixes are applied to those skills, and only to them, as an override
installed here rather than an edit upstream:

* **Counter-top objects are not base obstacles.** kinder-models' base planner projects
  every movable to the floor, so the wiper and cubes resting on the countertop become
  floor obstacles; a robot that starts beside the island (its init region reaches
  x = 1.2) then starts *in collision* and every OpenDrawer/PickWiper plan fails
  (3/12 seeds). Movables above the chassis (z > 0.15) are dropped from the base check.
* **The island slab is skipped in these skills' arm checks.** Their grasp transforms put
  the fingertips at or below the countertop the wiper lies on, so a faithful slab makes
  every PickWiper goal "collide". Those skills execute a straight joint-space
  interpolation rather than the plan, so the check changes nothing they execute; the
  reset's own planning keeps the slab.
"""

from typing import Any, ClassVar

import numpy as np

from .session import SweepDrawerSession
from .types import ResetStep, SweepDrawerScene


class StockSweepSkills:
    """Runs kinder-models' sweep3D controllers inside a session, with the fixes above."""

    SKILLS: ClassVar[tuple[str, ...]] = ("open_drawer", "pick_wiper", "sweep")
    _installed: ClassVar[bool] = False

    @staticmethod
    def install_planning_fixes() -> None:
        if StockSweepSkills._installed:
            return
        from kinder_models.dynamic3d import utils as kmu
        from kinder_models.dynamic3d.sweep3D import parameterized_skills as sweep

        original_bodies = kmu.PyBulletSim.get_collision_bodies
        original_base = kmu.run_base_motion_planning
        movables = {SweepDrawerScene.WIPER, *SweepDrawerScene.CUBES}

        class SweepPyBulletSim(kmu.PyBulletSim):  # type: ignore[misc, name-defined]
            def get_collision_bodies(self, held_object: int | None = None) -> set[int]:  # noqa: PLR0917
                bodies: set[int] = original_bodies(self, held_object)
                bodies.discard(self._static_colliders.get(SweepDrawerScene.ISLAND_SLAB))
                return bodies

        def base_planner(state: Any, target_base_pose: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: PLR0917
            skip = list(kwargs.pop("disable_collision_objects", None) or [])
            skip += [
                o.name
                for o in state
                if o.name in movables and not 0.05 <= state.get(o, "z") <= 0.15
            ]
            return original_base(  # type: ignore[misc]
                state, target_base_pose, *args, disable_collision_objects=skip, **kwargs
            )

        sweep.PyBulletSim = SweepPyBulletSim  # type: ignore[misc]
        sweep.run_base_motion_planning = base_planner
        StockSweepSkills._installed = True

    @staticmethod
    def run(
        *,
        session: SweepDrawerSession,
        skill: str,
        label: str,
        phase: str,
        limit: int = 600,
        params: np.ndarray | None = None,
    ) -> ResetStep:
        """Ground and run one stock skill to termination (or failure)."""
        StockSweepSkills.install_planning_fixes()
        from kinder_models.dynamic3d.sweep3D import parameterized_skills as sweep

        names = [
            SweepDrawerScene.ROBOT,
            SweepDrawerScene.WIPER,
            SweepDrawerScene.DRAWER,
            *SweepDrawerScene.CUBES,
        ]
        state = session.state
        lifted = sweep.create_lifted_controllers(
            session.env.action_space, None, pybullet_sim=sweep.PyBulletSim(state)
        )
        controller = lifted[skill].ground(tuple(state.get_object_from_name(n) for n in names))
        if params is None:
            params = controller.sample_parameters(state, np.random.default_rng(0))
        else:
            params = np.asarray(params, dtype=float)
            if params.shape != (2,) or not np.isfinite(params).all():
                raise ValueError("Stock Sweep parameters must be two finite values")
        session.begin(name=label, kind=f"kinder-models {skill}", phase=phase)
        error, finished = "", False
        try:
            controller.reset(session.state, params)
            for _ in range(limit):
                session.step(action=controller.step(), clip=False)
                controller.observe(session.state)
                if controller.terminated():
                    finished = True
                    break
        except BaseException as e:  # TrajectorySamplingFailure is not an Exception subclass
            if isinstance(e, KeyboardInterrupt):
                raise
            error = f"{type(e).__name__}: {e}"
        return session.end(success=finished and not error, note=error)

    @staticmethod
    def attempt(*, session: SweepDrawerSession, phase: str, settle: int = 15) -> list[ResetStep]:
        """OpenDrawer -> PickWiper -> Sweep, then let the cubes come to rest."""
        records = [
            StockSweepSkills.run(session=session, skill=k, label=f"{k}", phase=phase)
            for k in StockSweepSkills.SKILLS
        ]
        session.begin(name="settle", kind="wait", phase=phase)
        grip = 1.0 if session.gripper() > 0.2 else 0.0
        for _ in range(settle):
            a = np.zeros(11)
            a[-1] = grip
            session.step(action=a)
        records.append(session.end(success=True))
        return records
