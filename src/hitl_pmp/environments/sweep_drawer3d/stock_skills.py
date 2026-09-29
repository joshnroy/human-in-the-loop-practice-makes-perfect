"""Execute the actual pinned stock controllers without planning overrides."""

from typing import ClassVar

import numpy as np

from .session import SweepDrawerSession
from .types import ResetStep, SweepDrawerScene


class StockSweepSkills:
    """Runs the pinned native controllers inside a session."""

    SKILLS: ClassVar[tuple[str, ...]] = ("open_drawer", "pick_wiper", "sweep")

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
        if not finished and not error:
            error = f"Controller did not terminate within {limit} ticks"
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
