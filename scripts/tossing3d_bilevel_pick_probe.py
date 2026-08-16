"""Run Tossing3D bilevel planning for one seed and record how refinement went.

Post-run analysis reads the JSON this writes; nothing here plots. One process per
seed, because a KINDER rollout holds a PyBullet client and a MuJoCo model alive and
the refiner grounds a fresh controller per sampling attempt.

Outcome is one of:
  * ``scored``             -- refinement found a plan, execution left the cube in the
                              goal region (``sim._check_goals()``).
  * ``planned_not_scored`` -- refinement found a plan, execution did not score.
  * ``plan_not_found``     -- refinement returned no plan (``AgentFailure``).

Every trajectory-sampling attempt is recorded: which skill, whether it reached the
target abstract state, how long it took, and the continuous parameters it drew.
``TrajectorySamplingFailure`` is deliberately caught as ``BaseException`` -- it does
not subclass ``Exception``, so ``except Exception`` counts zero rejections.
"""

import argparse
import json
import os
import time
import traceback
from typing import Any

import numpy as np


def main() -> None:
    """Run one seed and write its record to --output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--samples-per-step", type=int, default=5)
    parser.add_argument(
        "--pick-mode",
        choices=("sampled", "zero-param", "rng-shift-control"),
        default="sampled",
        help=(
            "sampled: pick_cube draws (distance, rot) as the controller ships. "
            "zero-param: the old hardcoded STANDOFF, drawing nothing. "
            "rng-shift-control: the old hardcoded STANDOFF, but consuming the same "
            "two draws, so the toss's own random stream shifts exactly as it does "
            "under `sampled` and the two differ only in where the robot stands."
        ),
    )
    parser.add_argument("--max-abstract-plans", type=int, default=1)
    parser.add_argument("--planning-timeout", type=float, default=1800.0)
    parser.add_argument("--max-skill-horizon", type=int, default=400)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    record: dict[str, Any] = {
        "seed": args.seed,
        "samples_per_step": args.samples_per_step,
        "max_abstract_plans": args.max_abstract_plans,
        "planning_timeout": args.planning_timeout,
        "pick_mode": args.pick_mode,
    }
    start = time.perf_counter()
    try:
        record.update(_run(args))
        record["outcome_error"] = None
    except BaseException:  # noqa: BLE001 - a crash is a result, not a reason to stop
        record["outcome"] = "crashed"
        record["outcome_error"] = traceback.format_exc()
    record["total_seconds"] = time.perf_counter() - start

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    """Plan and execute one episode, returning everything measured about it."""
    # Imported here so the module still imports where KINDER is absent.
    import kinder  # noqa: PLC0415
    import kinder.envs.dynamic3d.envs  # noqa: F401, PLC0415  (forces the EGL path)
    from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (  # noqa: PLC0415
        ParameterizedControllerTrajectorySampler,
    )
    from kinder_bilevel_planning.agent import (  # noqa: PLC0415
        AgentFailure,
        BilevelPlanningAgent,
    )
    from kinder_bilevel_planning.env_models import (  # noqa: PLC0415
        create_bilevel_planning_models,
    )
    from kinder_models.dynamic3d.tossing import parameterized_skills  # noqa: PLC0415

    attempts: list[dict[str, Any]] = []
    drawn: dict[str, Any] = {}

    original_sample = parameterized_skills.PickCubeController.sample_parameters

    def recording_sample(self, x, rng):  # type: ignore[no-untyped-def]
        if args.pick_mode == "sampled":
            params = original_sample(self, x, rng)
        else:
            if args.pick_mode == "rng-shift-control":
                # Burn exactly what the shipped sampler burns, then discard it.
                original_sample(self, x, rng)
            params = np.array(parameterized_skills.PickCubeController.STANDOFF)
        drawn["pick_cube"] = np.asarray(params, dtype=float).tolist()
        return params

    parameterized_skills.PickCubeController.sample_parameters = (  # type: ignore[method-assign]
        recording_sample
    )

    original_call = ParameterizedControllerTrajectorySampler.__call__

    def recording_call(self, x, s, a, ns, bpg, rng):  # type: ignore[no-untyped-def]
        name = getattr(a, "name", None) or str(a)
        drawn.pop("pick_cube", None)
        began = time.perf_counter()
        try:
            result = original_call(self, x, s, a, ns, bpg, rng)
        except BaseException as exc:  # TrajectorySamplingFailure is not an Exception
            attempts.append(
                {
                    "skill": name,
                    "reached_target_abstract_state": False,
                    "raised": type(exc).__name__,
                    "seconds": time.perf_counter() - began,
                    "pick_params": drawn.get("pick_cube"),
                }
            )
            raise
        attempts.append(
            {
                "skill": name,
                "reached_target_abstract_state": True,
                "raised": None,
                "seconds": time.perf_counter() - began,
                "pick_params": drawn.get("pick_cube"),
            }
        )
        return result

    ParameterizedControllerTrajectorySampler.__call__ = recording_call  # type: ignore[method-assign]

    kinder.register_all_environments()
    # register_all_environments() forces osmesa when DISPLAY is unset, and under osmesa
    # every Dynamic3D env is skipped in silence. Put both back before make().
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    env = kinder.make("kinder/Tossing3D-o1-v0")
    obs, info = env.reset(seed=args.seed)

    env_models = create_bilevel_planning_models(
        "tidybot3d_tossing3D",
        env.observation_space,
        env.action_space,
        num_objects=1,
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=args.seed,
        max_abstract_plans=args.max_abstract_plans,
        samples_per_step=args.samples_per_step,
        planning_timeout=args.planning_timeout,
        max_skill_horizon=args.max_skill_horizon,
    )

    out: dict[str, Any] = {}
    plan_began = time.perf_counter()
    try:
        agent.reset(obs, info)
        out["planning_seconds"] = time.perf_counter() - plan_began
    except AgentFailure:
        out["planning_seconds"] = time.perf_counter() - plan_began
        out["outcome"] = "plan_not_found"
        out["attempts"] = attempts
        env.close()
        return out

    execution_began = time.perf_counter()
    for _ in range(4000):
        action = agent.step()
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, reward, terminated or truncated, info)
        if terminated or truncated or len(agent._current_plan) == 0:  # noqa: SLF001
            break
    out["execution_seconds"] = time.perf_counter() - execution_began

    sim = env.unwrapped._object_centric_env  # noqa: SLF001
    scored = bool(sim._check_goals())  # noqa: SLF001
    out["outcome"] = "scored" if scored else "planned_not_scored"
    out["attempts"] = attempts
    env.close()
    return out


if __name__ == "__main__":
    main()
