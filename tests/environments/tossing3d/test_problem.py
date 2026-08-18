"""Offline tests for `Tossing3DProblem` and `Tossing3DSkillProvider`.

`run_task_episode`'s *simulator* behaviour lives in `test_kinder_fidelity.py`; its frame
bookkeeping is exercised here against a canned backend, because CI never installs KINDER
and that is precisely where a rendering regression would otherwise go unseen.
"""

import numpy as np
import pytest
from pydantic import PrivateAttr

from hitl_pmp.core.method.types import GroundSkill, LabeledAction
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Task
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
from hitl_pmp.environments.tossing3d.predicates import (
    HAND_EMPTY,
    HOLDING,
    IN_BIN,
    ON_GROUND,
    REACHABLE,
)
from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem
from hitl_pmp.environments.tossing3d.renderer import Tossing3DRenderer
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks
from hitl_pmp.planning.grounding import SkillGrounder

from .observations import BIN_X, HOLDING_ATOMS, INITIAL_ATOMS, MISSED_TOSS_ATOMS, state


def _problem() -> Tossing3DProblem:
    env = Tossing3DEnvironment()
    return Tossing3DProblem(env=env, tasks=Tossing3DTasks(env=env, seed=0))


def test_the_horizon_is_the_shortest_solve_plus_two() -> None:
    """Two skills is the shortest solve and there is no shorter route:
    `MoveToTossLocationAndToss` requires `Holding`, and only `PickCube` grants it. The
    `+ 2` buys one recovery from a failed grasp -- and no more, because after a toss there
    is nothing to recover from anyway.

    It was `3 + 2 = 5` under the three-skill decomposition. Leaving it at 5 would have
    given this domain a horizon one step more generous than the convention it claims, and
    more generous than the one every published Tossing3D baseline ran under.
    """
    assert _problem().max_episode_steps() == 4


def test_the_problem_has_no_human_oracle_yet() -> None:
    """Stated as a test rather than left implicit, because this is the domain in the repo
    that most wants one: a tossed cube genuinely needs someone to walk over and pick it
    up, and `Metrics.num_human_interventions()` reporting (0.0, 0) is a gap in what is
    representable, not evidence that no intervention was needed."""
    assert _problem().human is None


def test_the_provider_exposes_every_skill_predicate_type_and_object() -> None:
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment())
    assert provider.skills() == (
        Tossing3DSkills.PICK_CUBE,
        Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
    )
    assert set(provider.predicates()) == {
        IN_BIN,
        HAND_EMPTY,
        HOLDING,
        ON_GROUND,
        REACHABLE,
    }
    assert {obj.type for obj in provider.objects()} == set(provider.types())


def test_every_predicate_a_skill_references_is_one_the_provider_publishes() -> None:
    """`SkillGrounder` abstracts a state using `provider.predicates()` and checks
    applicability against `Skill.preconditions`. A precondition over a predicate the
    provider does not publish is never true, so the skill silently becomes unusable."""
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment())
    published = set(provider.predicates())
    for skill in provider.skills():
        for atom in (*skill.preconditions, *skill.add_effects, *skill.delete_effects):
            assert atom.predicate in published, f"{skill.name} references {atom.predicate.name}"


def test_the_symbolic_layer_grounds_the_oracles_own_plan_shape() -> None:
    """Walked with the real grounder rather than by hand: from the initial abstract state
    only `PickCube` is applicable, and holding the cube unlocks the composed toss and
    nothing else.

    **There is no third rung any more.** Where the robot stands used to decide between
    `MoveToThrowPose` and `Toss`; the composed controller drives itself to the standoff,
    so the base pose is not part of any precondition and the same skill set is applicable
    whether the robot is next to the bin or across the room. That is asserted below rather
    than left implicit, because it is exactly the state the retired predicate named."""
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment())

    def applicable(*, atoms=INITIAL_ATOMS, **kwargs) -> set[str]:
        current = state(abstract_atoms=atoms, **kwargs)
        atoms = SkillGrounder.abstract_state(
            state=current, objects=provider.objects(), predicates=provider.predicates()
        )
        return {
            ground.skill.name
            for ground in SkillGrounder.applicable_ground_skills(
                skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
            )
        }

    assert applicable() == {"PickCube"}
    assert applicable(atoms=HOLDING_ATOMS, cube_z=0.4) == {"MoveToTossLocationAndToss"}
    # Standing at the old throw standoff changes nothing: the composed skill drives there
    # itself, so no precondition reads the base pose -- and no *atom* reports it either,
    # since upstream deleted `RobotAtThrowPose` with the pose it named.
    assert applicable(atoms=HOLDING_ATOMS, cube_z=0.4, base_x=BIN_X - 1.35) == {
        "MoveToTossLocationAndToss"
    }


def test_nothing_is_applicable_once_the_cube_is_past_the_barrier() -> None:
    """The irreversibility, read off the symbolic layer: after a toss the cube is beyond
    the barrier, `Reachable` is false, `Pick` is inapplicable, and no skill remains. A
    planner asked to recover from here correctly finds no plan."""
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment())
    landed = state(
        cube_x=2.6,
        cube_z=0.025,
        gripper=0.0,
        base_x=BIN_X - 1.35,
        abstract_atoms=MISSED_TOSS_ATOMS,
    )
    atoms = SkillGrounder.abstract_state(
        state=landed, objects=provider.objects(), predicates=provider.predicates()
    )
    assert not SkillGrounder.applicable_ground_skills(
        skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
    )


def test_the_provider_delegates_sampling_and_encoding_to_the_skills_container() -> None:
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    ground_skill = GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier),
    )
    params = provider.sample_params(ground_skill=ground_skill, rng=np.random.default_rng(0))
    assert params.shape == (4,)
    action = provider.compute_action(ground_skill=ground_skill, params=params, state=state())
    assert action[0] == Tossing3DEnvironment.move_to_toss_location_and_toss_id
    assert list(action[1:]) == pytest.approx(list(params))


class _CannedBackend(KinderBackend):
    """A `KinderBackend` whose simulator is replaced by canned frames.

    Enough to exercise `run_task_episode`'s drain offline. The first drain is the
    clearing one `run_task_episode` does straight after the reset, and finds nothing;
    every later drain stands in for one skill's worth of physics ticks.
    """

    frames_per_skill: int = 3
    _drains: int = PrivateAttr(default=0)

    def drain_substep_frames(self) -> list[np.ndarray]:
        self._drains += 1
        if not self.record_substeps or self._drains == 1:
            return []
        return [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(self.frames_per_skill)]

    def render(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)


class _CannedEnvironment(Tossing3DEnvironment):
    """A `Tossing3DEnvironment` with no MuJoCo behind it: skills are no-ops."""

    def backend(self) -> KinderBackend:
        if self._backend is None:
            self._backend = _CannedBackend()
        return self._backend

    def set_state(self, *, state: State) -> None:
        self.current_state = state

    def take_action(self, *, action: np.ndarray) -> State:
        return self.get_current_state()


def _unreachable_goal(*, env: Tossing3DEnvironment) -> Goal:
    """`InBin(cube, bin)`, which the initial scene does not satisfy -- so the episode
    runs its whole horizon instead of returning on the first check."""
    return Goal(atoms=frozenset({GroundAtom(predicate=IN_BIN, objects=(env.cube, env.bin))}))


def _canned_episode(*, renderer):
    env = _CannedEnvironment()
    problem = Tossing3DProblem(env=env, tasks=Tossing3DTasks(env=env, seed=0))
    task = Task(initial_state=state(abstract_atoms=INITIAL_ATOMS), goal=_unreachable_goal(env=env))

    def policy(observed) -> LabeledAction:  # noqa: PLR0917  (core.Policy is positional)
        return LabeledAction(action=np.zeros(5), label="PickCube(robot, cube_0, cuboid_barrier)")

    return problem.run_task_episode(task=task, policy=policy, renderer=renderer)


def test_a_rendered_episode_is_physics_rate_rather_than_one_frame_per_skill() -> None:
    """The defect this exists to prevent: `episode.mp4` used to be one frame per
    decision, and a decision here is a whole controller execution, so the CLI's demo clip
    was four frames of a domain whose entire point is a throw. Every sub-step frame the
    backend collected has to reach the returned list."""
    _, frames, _ = _canned_episode(renderer=Tossing3DRenderer)

    # One captioned initial frame, then one skill's worth of canned sub-step frames per
    # step of the horizon. Derived from the horizon rather than pinned as a literal: the
    # claim here is about frames per skill, and `max_episode_steps` is pinned above.
    horizon = _problem().max_episode_steps()
    assert len(frames) == 1 + horizon * _CannedBackend().frames_per_skill


def test_an_unrendered_episode_records_nothing_and_turns_recording_back_off() -> None:
    """Recording is opt-in per episode: a training run passes no renderer and must not
    pay for hundreds of MuJoCo renders per skill, nor be left recording afterwards."""
    env = _CannedEnvironment()
    problem = Tossing3DProblem(env=env, tasks=Tossing3DTasks(env=env, seed=0))
    task = Task(initial_state=state(abstract_atoms=INITIAL_ATOMS), goal=_unreachable_goal(env=env))

    def policy(observed) -> LabeledAction:  # noqa: PLR0917
        return LabeledAction(action=np.zeros(5), label="PickCube(robot, cube_0, cuboid_barrier)")

    solved, frames, _ = problem.run_task_episode(task=task, policy=policy, renderer=None)

    assert frames == []
    assert env.backend().record_substeps is False


def test_every_frame_of_a_rendered_episode_is_the_same_size() -> None:
    """ffmpeg needs one frame size for the whole clip. The boundary frame comes from
    `render_frame` and the sub-step frames from the drain, so this is the join where a
    mismatch would appear."""
    _, frames, _ = _canned_episode(renderer=Tossing3DRenderer)

    assert len({frame.shape for frame in frames}) == 1
