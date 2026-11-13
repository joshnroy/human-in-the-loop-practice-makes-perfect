"""A restored kitchen engine is the engine the paused process had, mid-motion.

`restore_engine` sets MuJoCo's integration state and runs `mj_forward`, so every derived
array (body and site poses, bias forces, contacts) is recomputed at the restored state.
That is only the state the paused process had if that process, too, had run a forward
pass after its last `mj_step`: after a step MuJoCo leaves them describing the step
before. Here it had. Every skill and human reset ends by evaluating the abstract
state, and KINDER's holding classifier (`get_gripper_object_contacts`) sets the scene
state, which runs `mj_forward`. This test pins that, at two boundaries where the scene
is moving: after a pick (cube moving in the hand) and after a toss (arm still moving).
Then it runs the next skill in both processes and compares every control step.
"""

import importlib.util
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
]

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "scripts/kitchen_recycling/experiment-config.json"


class Engine:
    """What the live MuJoCo engine holds, read without changing it."""

    # Arrays a restore leaves different, none of them read before a step rewrites
    # it: the implicit integrator's factorisation (written by `mj_step`, never by
    # `mj_forward`), and the sparse and island constraint-Jacobian bookkeeping. Every
    # solver output and every later integration state is compared and identical.
    SCRATCH: ClassVar[frozenset[str]] = frozenset({
        "qH",
        "qHDiagInv",
        "efc_AR_rowadr",
        "efc_AR_rownnz",
        "efc_J_colind",
        "efc_J_rowadr",
        "efc_J_rownnz",
        "efc_J_rowsuper",
        "iefc_J",
        "iefc_J_colind",
        "iefc_J_rowadr",
        "iefc_J_rownnz",
        "iefc_J_rowsuper",
    })
    CONTACT_FIELDS: ClassVar[tuple[str, ...]] = (
        "pos",
        "frame",
        "dist",
        "geom",
        "efc_address",
        "dim",
        "mu",
        "solref",
        "solimp",
        "includemargin",
    )

    @staticmethod
    def sim(*, env: Any) -> tuple[Any, Any]:
        live = env.kinder_backend()._object_centric()._robot_env.sim  # noqa: SLF001
        return live.model.mj_model, live.data.mj_data

    @staticmethod
    def arrays(*, env: Any) -> dict[str, np.ndarray]:
        """Every numeric MjData array, and the active contacts field by field."""
        _, data = Engine.sim(env=env)
        found = {}
        for name in dir(data):
            value = None if name.startswith("_") else getattr(data, name, None)
            if isinstance(value, np.ndarray) and value.dtype.kind in "fiu":
                found[name] = value.copy()
        for field in Engine.CONTACT_FIELDS:
            found[f"contact.{field}"] = np.array(getattr(data.contact, field))[: data.ncon]
        found["ncon"] = np.array([data.ncon])
        found["time"] = np.array([data.time])
        return found

    @staticmethod
    def differing(*, left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> list[str]:
        return sorted(
            name
            for name in left.keys() | right.keys()
            if name not in left
            or name not in right
            or left[name].shape != right[name].shape
            or not np.array_equal(left[name], right[name])
        )

    @staticmethod
    def integration(*, env: Any) -> np.ndarray:
        import mujoco

        model, data = Engine.sim(env=env)
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, state, spec)
        return state

    @staticmethod
    def control_steps(*, env: Any, skill: Any) -> list[np.ndarray]:
        """The integration state after every control step of one skill."""
        steps: list[np.ndarray] = []
        backend = env.kinder_backend()
        backend.set_control_step_observers(
            before=None, after=lambda: steps.append(Engine.integration(env=env))
        )
        try:
            skill(env=env)
        finally:
            backend.set_control_step_observers(before=None, after=None)
        return steps

    @staticmethod
    def robot_and_cube_speed(*, env: Any) -> tuple[float, float]:
        state = env.kinder_backend().snapshot()
        robot = state.get_object_from_name("robot")
        cube = state.get_object_from_name("cube_0")
        velocities = [f for f in state.type_features[robot.type] if f.startswith("vel_")]
        return (
            max(abs(state.get(robot, f)) for f in velocities),
            float(np.linalg.norm([state.get(cube, f"v{axis}") for axis in "xyz"])),
        )


class Skills:
    @staticmethod
    def pick(*, env: Any) -> None:
        env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0.0]))

    @staticmethod
    def toss(*, env: Any) -> None:
        from hitl_pmp.environments.tossing3d.toss import Tossing3DToss

        env.take_action(
            action=Tossing3DToss.compute_action(
                params=np.array((1.45, 382.65, 438.2)), state=env.get_current_state()
            )
        )

    @staticmethod
    def open_gripper(*, env: Any) -> None:
        env.take_action(action=np.array([env.open_gripper_id, 0, 0, 0, 0.0]))


def kitchen() -> Any:
    from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig
    from hitl_pmp.environments.kitchen_recycling3d.environment import (
        KitchenRecycling3DEnvironment,
    )

    return KitchenRecycling3DEnvironment(
        kitchen_config=KitchenConfig.model_validate_json(CONFIG.read_text()), scene_bg=False
    )


def test_a_restored_engine_matches_the_paused_one_array_for_array_and_step_for_step():
    paused, restored = kitchen(), kitchen()
    try:
        paused.reset_to_seed(seed=0)
        restored.reset_to_seed(seed=1)  # A different scene, overwritten by the restore.
        Skills.pick(env=paused)
        for done, following, moving in (
            ("pick", Skills.toss, "cube"),
            ("toss", Skills.open_gripper, "robot"),
        ):
            robot, cube = Engine.robot_and_cube_speed(env=paused)
            assert (cube if moving == "cube" else robot) > 1e-3, (done, robot, cube)
            checkpoint = paused.simulation_checkpoint()
            expected = Engine.arrays(env=paused)
            restored.restore_simulation_checkpoint(checkpoint=checkpoint)
            differing = Engine.differing(left=expected, right=Engine.arrays(env=restored))
            assert set(differing) <= Engine.SCRATCH, (done, differing)
            ahead = Engine.control_steps(env=paused, skill=following)
            behind = Engine.control_steps(env=restored, skill=following)
            assert ahead, done
            assert len(ahead) == len(behind), done
            first = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(ahead, behind, strict=True))
                    if not np.array_equal(a, b)
                ),
                None,
            )
            assert first is None, (done, first)
    finally:
        paused.close()
        restored.close()
