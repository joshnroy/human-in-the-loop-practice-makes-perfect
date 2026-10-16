"""The human reset to a freshly sampled start state, on both 3D domains, offline.

`reset_to_seed` is each domain's single door into the simulator, so recording it
stands in for the scene: what a reset draws, from which stream, and that the human's
`set_state` installs exactly that draw (robot included, `steps_taken` back to 0).
"""

import numpy as np
import pytest
from pydantic import Field

from hitl_pmp.core.problem.human.types import SAMPLED_START_RESET_DESTINATION
from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O
from hitl_pmp.environments.sweepoffshelf3d.problem import SweepOffShelf3DProblem
from hitl_pmp.environments.sweepoffshelf3d.step_protocol import SweepOffShelf3DStepProtocol
from hitl_pmp.environments.sweepoffshelf3d.tasks import SweepOffShelf3DTasks
from hitl_pmp.environments.sweepoffshelf3d.types import SweepOffShelf3DState
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem
from hitl_pmp.environments.tossing3d.step_protocol import Tossing3DStepProtocol
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks
from hitl_pmp.humans.oracle import UnconditionalHumanOracle

from .tossing3d.observations import state as tossing_state


class _RecordingTossing3D(Tossing3DEnvironment):
    seeds_built: list[int] = Field(default_factory=list)

    def reset_to_seed(self, *, seed: int):  # type: ignore[override]
        self.seeds_built.append(seed)
        built = tossing_state(env=self, seed=seed, steps_taken=0)
        self.current_state = built
        return built


class _RecordingShelf(SweepOffShelf3DEnvironment):
    seeds_built: list[int] = Field(default_factory=list)

    def reset_to_seed(self, *, seed: int):  # type: ignore[override]
        self.seeds_built.append(seed)
        built = SweepOffShelf3DState(
            data={O.scene: np.array([float(seed), 0.0])}, abstract_atoms=frozenset()
        )
        self.current_state = built
        return built


def _tossing_problem(*, seed: int = 0):
    env = _RecordingTossing3D(scene_bg=False)
    tasks = Tossing3DTasks(env=env, seed=seed)
    return Tossing3DProblem(env=env, tasks=tasks, human=UnconditionalHumanOracle), env, tasks


def _shelf_problem(*, seed: int = 0):
    env = _RecordingShelf()
    tasks = SweepOffShelf3DTasks(env=env, seed=seed)
    return SweepOffShelf3DProblem(env=env, tasks=tasks, human=UnconditionalHumanOracle), env, tasks


PROBLEMS = {"tossing3d": _tossing_problem, "sweepoffshelf3d": _shelf_problem}


def _scene_seed(*, env, state) -> int:
    scene = env.scene if isinstance(env, Tossing3DEnvironment) else O.scene
    return int(round(state.get(obj=scene, feature_name="seed")))


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_the_reset_installs_a_freshly_sampled_start_state(*, name: str) -> None:
    problem, env, _ = PROBLEMS[name]()
    env.reset_to_seed(seed=125)  # the scene practice was standing in
    env.seeds_built.clear()

    problem.execute_movables_reset(destination=SAMPLED_START_RESET_DESTINATION)

    drawn = env.seeds_built[0]
    assert drawn != 125
    # Drawn, then installed by the human through `set_state` (a scene rebuild).
    assert env.seeds_built == [drawn, drawn]
    current = env.get_current_state()
    assert _scene_seed(env=env, state=current) == drawn
    assert (
        int(
            round(
                current.get(
                    obj=env.scene if isinstance(env, Tossing3DEnvironment) else O.scene,
                    feature_name="steps_taken",
                )
            )
        )
        == 0
    )


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_the_start_states_come_from_their_own_stream(*, name: str) -> None:
    """The train and test streams are untouched, so a run that resets this way sees
    exactly the train and test scenes plain EES sees at the same seed."""
    problem, _, tasks = PROBLEMS[name]()
    train = tasks._train_rng.bit_generator.state  # noqa: SLF001
    test = tasks._test_rng.bit_generator.state  # noqa: SLF001

    problem.execute_movables_reset(destination=SAMPLED_START_RESET_DESTINATION)

    assert tasks._train_rng.bit_generator.state == train  # noqa: SLF001
    assert tasks._test_rng.bit_generator.state == test  # noqa: SLF001


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_the_start_state_stream_is_fixed_by_the_run_seed(*, name: str) -> None:
    first, env_a, _ = PROBLEMS[name](seed=3)
    second, env_b, _ = PROBLEMS[name](seed=3)
    other, env_c, _ = PROBLEMS[name](seed=4)
    for problem in (first, second, other):
        for _ in range(3):
            problem.execute_movables_reset(destination=SAMPLED_START_RESET_DESTINATION)
    assert env_a.seeds_built == env_b.seeds_built
    assert env_a.seeds_built != env_c.seeds_built
    # Three different start scenes, not one scene three times.
    assert len(set(env_a.seeds_built)) == 3


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_the_start_state_stream_differs_from_the_train_and_test_streams(*, name: str) -> None:
    problem, env, tasks = PROBLEMS[name](seed=0)
    problem.execute_movables_reset(destination=SAMPLED_START_RESET_DESTINATION)
    start = env.seeds_built[0]
    assert start not in {
        _scene_seed(env=env, state=tasks.sample_train_task().initial_state),
        _scene_seed(env=env, state=tasks.sample_test_task().initial_state),
    }


@pytest.mark.parametrize(
    "protocol", [Tossing3DStepProtocol, SweepOffShelf3DStepProtocol], ids=lambda p: p.__name__
)
def test_the_step_protocol_charges_the_start_reset_under_its_own_name(*, protocol) -> None:
    from hitl_pmp.step_protocol import StepPracticeRunner

    names = StepPracticeRunner.human_skill_names(environment=protocol)
    assert names[SAMPLED_START_RESET_DESTINATION] == "reset_to_sampled_start"
    # The domain's own destinations keep their names and their order.
    assert list(names)[:-1] == list(protocol.human_skill_names())
