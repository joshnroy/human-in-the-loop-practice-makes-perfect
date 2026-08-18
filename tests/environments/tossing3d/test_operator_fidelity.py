"""Tossing3D's operator-dynamics fidelity check: the same invariant, a feasible walk.

`tests/environments/test_operator_dynamics_fidelity.py` enforces, across every domain it
can drive, that **a symbolic operator model must never permit more than the raw dynamics
actually allow** -- any ground skill applicable in a believed symbolic state must, when
executed, change the real state. Two shipped bugs have had exactly that shape.

**This domain is not in that file's `_DOMAINS`, and this file is why it does not need to
be.** Two reasons it cannot join, both structural rather than stylistic:

1. That walk needs KINDER. CI never installs the optional extra, so registering this
   domain there would turn a green cross-domain test into a collection error on CI.
2. Its budget is 8 random walks x 40 steps plus oracle walks, executing *every* applicable
   candidate up to 30 parameter draws each. Here one skill execution is a real MuJoCo
   rollout of several hundred ticks (~1-15 s), so that budget is thousands of rollouts --
   hours to days, per run.

So the invariant is enforced here instead, on the states this domain can actually reach:
the oracle's own trajectory, which is the whole plan shape (`Pick`, `MoveToThrowPose`,
`Toss`) plus the unrecoverable state after the throw. At each of those states every
applicable ground skill is executed speculatively and the world is rewound
(`Tossing3DEnvironment.snapshot`/`restore`, backed by KINDER's own `set_state`), exactly
as the cross-domain walk does with `set_state`.

The narrowing that remains, stated rather than glossed: this walks one trajectory rather
than searching, so it cannot find a *reachable-but-unvisited* symbolic state whose model
is wrong. On a domain with three skills, one plan shape and no branching, there is very
little for such a search to find -- but "very little" is not "nothing", and that is the
honest limit of what this file proves.
"""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_oracle_policy import SkillOraclePolicy
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks
from hitl_pmp.planning.grounding import SkillGrounder

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="KINDER is an optional extra (`kindergarden` + `kinder_models`); CI never installs it",
)

CANONICAL_SEED = 125

# Draws per candidate. Far below the cross-domain file's 30, and affordable for the same
# reason it can walk one trajectory: `Pick` is the only skill here whose parameters can
# miss, `MoveToThrowPose` moves the base for every draw in its range, and `Toss` has no
# parameters at all. A skill counts as silently ignored only if *every* draw is a no-op.
_PARAM_DRAWS = 3


def _describe(*, ground_skill: GroundSkill) -> str:
    return f"{ground_skill.skill.name}({', '.join(o.name for o in ground_skill.objects)})"


def _changed(*, before, after) -> bool:
    return any(not np.array_equal(before.data[obj], after.data[obj]) for obj in before.data)


def test_no_applicable_ground_skill_is_silently_ignored_along_the_oracles_trajectory() -> None:
    """THE property. See this module's docstring for the two shipped bugs it guards, and
    for what this walk covers versus the cross-domain one."""
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    rng = np.random.default_rng(0)
    violations: list[str] = []
    exercised: set[str] = set()

    try:
        goal = Tossing3DTasks(env=env, seed=0).build_task(scene_seed=CANONICAL_SEED).goal
        state = env.reset_to_seed(seed=CANONICAL_SEED)

        # Four states: initial, post-Pick, post-MoveToThrowPose, post-Toss. The last is
        # the unrecoverable one, and is the most valuable of the four -- it is where an
        # over-permissive model would offer a retrieval that cannot happen.
        for _ in range(4):
            atoms = SkillGrounder.abstract_state(
                state=state, objects=provider.objects(), predicates=provider.predicates()
            )
            candidates = SkillGrounder.applicable_ground_skills(
                skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
            )
            snapshot = env.snapshot()
            for ground_skill in candidates:
                if ground_skill.add_effects <= atoms:
                    continue  # symbolically vacuous here: a no-op is the correct outcome
                exercised.add(ground_skill.skill.name)
                draws = _PARAM_DRAWS if ground_skill.skill.param_dim > 0 else 1
                changed = False
                for _ in range(draws):
                    before = env.restore(snapshot=snapshot)
                    after = env.take_action(
                        action=provider.compute_action(
                            ground_skill=ground_skill,
                            params=provider.sample_params(ground_skill=ground_skill, rng=rng),
                            state=before,
                        )
                    )
                    changed = changed or _changed(before=before, after=after)
                if not changed:
                    violations.append(
                        f"{_describe(ground_skill=ground_skill)} is applicable (its "
                        f"preconditions hold) but executing it left the environment state "
                        f"completely unchanged -- the symbolic model claims an "
                        f"applicability the dynamics deny"
                    )
            state = env.restore(snapshot=snapshot)
            state = env.take_action(
                action=SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=goal).action
            )

        assert not violations, "\n".join(f"  - {v}" for v in sorted(set(violations)))
        # Coverage floor, so the property above cannot pass vacuously: the oracle's own
        # trajectory reaches a state where each of the three skills is applicable.
        assert exercised == {skill.name for skill in provider.skills()}, (
            f"the walk never reached a state where "
            f"{sorted({s.name for s in provider.skills()} - exercised)} was applicable, so "
            f"this file proves nothing about those skills"
        )
    finally:
        env.close()


# How far two runs of the same skill from the same snapshot may end up apart. NOT a
# tuned fudge: a KINDER `ObjectCentricState` is a **float32** vector (it is the
# observation, and `ObjectCentricBoxSpace` is float32), while MuJoCo integrates in
# float64. So a snapshot/restore round-trip is faithful to float32 and no further, and
# 200 substeps per env step amplify that seed of ~1.2e-7 relative error. Measured
# residual on the rollout below: ~1e-7 on x, ~1.6e-5 on z, ~2.7e-4 on the quaternion.
# 1e-3 sits an order of magnitude above the worst of those and four orders below the
# centimetre-scale changes the walk above actually looks for.
_RESTORE_TOLERANCE = 1e-3


def test_a_restore_really_rewinds_the_simulator_and_not_just_the_state_object() -> None:
    """The walk above is only worth anything if `restore` genuinely puts MuJoCo back --
    otherwise every speculative execution would run from wherever the previous one left
    off, and the invariant would be checked against the wrong states.

    Two halves. First, the restored observation must match the snapshot it came from,
    which is the direct claim. Second, re-running the *same* skill from the same snapshot
    must reach the *same* place: KINDER's controllers are deterministic given a state, so
    a partial rewind -- poses restored, velocities not -- would diverge visibly rather
    than by the float32 seed documented above. The snapshot is taken mid-episode on
    purpose: at reset every velocity is zero, which a pose-only restore would also get
    right, so a from-reset check would prove nothing about velocities."""
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    try:
        goal = Tossing3DTasks(env=env, seed=0).build_task(scene_seed=CANONICAL_SEED).goal
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        state = env.take_action(
            action=SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=goal).action
        )
        snapshot = env.snapshot()
        # Mid-episode really is mid-episode: the arm is up, holding the cube.
        assert state.get(obj=env.cube, feature_name="z") > 0.1

        ground_skill = GroundSkill(
            skill=provider.skills()[1],
            objects=(env.robot, env.cube, env.bin),
        )
        action = provider.compute_action(
            ground_skill=ground_skill, params=np.array([1.35]), state=state
        )
        first = env.take_action(action=action)

        restored = env.restore(snapshot=snapshot)
        for obj in state.data:
            assert np.allclose(restored.data[obj], state.data[obj], atol=1e-6), (
                f"{obj.name} is not back where the snapshot left it"
            )

        second = env.take_action(action=action)
        assert _changed(before=restored, after=second), "the skill should have moved the base"
        for obj in first.data:
            assert np.allclose(first.data[obj], second.data[obj], atol=_RESTORE_TOLERANCE), (
                f"{obj.name} differs by more than the float32 round-trip explains between "
                f"two runs of the same skill from the same snapshot, so restore did not "
                f"fully rewind the simulator"
            )
    finally:
        env.close()
