"""One value iteration prices deployment for every domain, and moving the three domains
onto it changed no value in any bit.

The oracles below are the three implementations it replaced, kept verbatim. Every
comparison is `==` on float64, never a tolerance, except the one that compares two
different summation orders of the same expectation.
"""

import json

import numpy as np
import pytest

from hitl_pmp.methods.belief_space.deployment_value_iteration import DeploymentValueIteration
from hitl_pmp.methods.belief_space.practice_domain import (
    SHELF_DEPLOYMENT_SKILLS,
    SWEEP_OFF_SHELF_DOMAIN,
    TOSSING3D_DOMAIN,
    SweepOffShelfDeployment,
)
from hitl_pmp.methods.belief_space.rack_serving_domain import (
    RACK_NUM_FOOD,
    RACK_ROBOT_SKILLS,
    RACK_SERVING_DOMAIN,
    RackServingDeployment,
)
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    TOSS_SKILL,
)
from hitl_pmp.methods.belief_space.tossing3d_deployment_model import (
    DeploymentPolicyExpectation,
    Tossing3DDeployment,
)
from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis
from hitl_pmp.methods.belief_space.types.theta import Tossing3DTheta

from .record_deployment_value_fixture import (
    FIXTURE,
    TABLE_HORIZONS,
    TOSSING_HORIZONS,
    RecordedDeploymentValues,
)


class ReplacedImplementations:
    """What each domain computed before the shared solver existed."""

    @staticmethod
    def tossing_batch(*, toss, pick, opened, horizon):
        ready_values = np.zeros_like(toss)
        holding_values = np.zeros_like(toss)
        closed_gripper_values = np.zeros_like(toss)
        for _ in range(horizon):
            previous_ready = ready_values
            previous_holding = holding_values
            previous_closed_gripper = closed_gripper_values
            holding_values = toss
            ready_values = pick * previous_holding + (1.0 - pick) * previous_closed_gripper
            closed_gripper_values = (
                opened * previous_ready + (1.0 - opened) * previous_closed_gripper
            )
        return ready_values

    @staticmethod
    def tossing_scalar(*, toss, pick, opened, horizon):
        ready_value = holding_value = closed_gripper_value = 0.0
        for _ in range(horizon):
            previous_ready = ready_value
            previous_holding = holding_value
            previous_closed_gripper = closed_gripper_value
            holding_value = toss
            ready_value = pick * previous_holding + (1.0 - pick) * previous_closed_gripper
            closed_gripper_value = (
                opened * previous_ready + (1.0 - opened) * previous_closed_gripper
            )
        return ready_value

    @staticmethod
    def shelf_moves():
        moves = {}
        for shelf in range(5 + 1):
            for hand in ("empty", "wiper", "cube"):
                for home in (True, False):
                    options = []
                    if hand == "empty" and home:
                        options.append(("PickWiper", (shelf, "wiper", False)))
                    if hand == "wiper" and shelf:
                        options.append(("SweepShelf", (0, "wiper", False)))
                    if hand == "wiper":
                        options.append(("ReturnWiper", (shelf, "empty", True)))
                    if hand == "empty" and shelf:
                        options.append(("PickCubeFromShelf", (shelf - 1, "cube", home)))
                    if hand == "cube":
                        options.append(("PlaceCubeInBox", (shelf, "empty", home)))
                    moves[shelf, hand, home] = tuple(options)
        return moves

    @staticmethod
    def rack_moves():
        moves = {}
        for on_rack in range(5 + 1):
            for hand in ("empty", "scraper", "food"):
                if hand == "food" and on_rack == 5:
                    continue
                for home in (True, False):
                    if hand == "scraper" and home:
                        continue
                    options = []
                    if hand == "empty" and home:
                        options.append(("PickScraper", (on_rack, "scraper", False)))
                    if hand == "scraper":
                        options.append(("ReturnScraper", (on_rack, "empty", True)))
                        if on_rack:
                            options.append(("GatherAndSweep", (0, "scraper", False)))
                    if hand == "empty":
                        if on_rack:
                            options.append(("PickFood", (on_rack - 1, "food", home)))
                        if on_rack < 5:
                            options.append(("PickFood", (on_rack, "food", home)))
                    if hand == "food":
                        options.append(("PlaceFoodOnTray", (on_rack, "empty", home)))
                        options.append(("ReturnFoodToRack", (on_rack + 1, "empty", home)))
                    moves[on_rack, hand, home] = tuple(options)
        return moves

    @staticmethod
    def table_values(*, moves, held, competences, horizon):
        """The loop the shelf and the rack each carried a copy of."""
        size = len(next(iter(competences.values())))
        solved = {
            state: np.full(size, float(state[0] == 0 and state[1] != held)) for state in moves
        }
        value = dict(solved)
        for _ in range(horizon):
            previous = value
            value = {}
            for state, options in moves.items():
                best = solved[state].copy()
                if not best.any():
                    for skill, successor in options:
                        p = competences[skill]
                        best = np.maximum(
                            best, p * previous[successor] + (1.0 - p) * previous[state]
                        )
                value[state] = best
        return value[5, "empty", True]


def _tossing_competences(*, count):
    """A dense grid with both endpoints, then draws that crowd the endpoints too."""
    grid = np.linspace(0.0, 1.0, 41)
    pick, toss, opened = (axis.ravel() for axis in np.meshgrid(grid, grid, grid, indexing="ij"))
    rng = np.random.default_rng(20261111)
    draws = np.concatenate([
        rng.uniform(size=(count, 3)),
        rng.beta(0.2, 0.2, size=(count, 3)),
        rng.beta(1.0, 3.0, size=(count, 3)),
        np.nextafter(rng.integers(0, 2, size=(count, 3)).astype(float), 0.5),
    ])
    return (
        np.concatenate([pick, draws[:, 0]]),
        np.concatenate([toss, draws[:, 1]]),
        np.concatenate([opened, draws[:, 2]]),
    )


def _table_competences(*, skills, count):
    rng = np.random.default_rng(20261111)
    draws = np.concatenate([
        rng.uniform(size=(count, len(skills))),
        rng.beta(1.0, 3.0, size=(count, len(skills))),
        rng.beta(0.2, 0.2, size=(count, len(skills))),
        rng.integers(0, 2, size=(count, len(skills))).astype(float),
    ])
    return {name: np.ascontiguousarray(draws[:, index]) for index, name in enumerate(skills)}


def test_the_solver_backs_up_an_explicit_failure_successor_and_takes_the_best_option():
    transitions = {
        "start": (("safe", "goal", "start"), ("risky", "goal", "dead")),
        "detour": (("recover", "start", "detour"),),
    }
    terminal_values = {"goal": 1.0, "dead": 0.0}
    competences = {
        "safe": np.array([0.25, 0.0]),
        "risky": np.array([0.5, 0.75]),
        "recover": np.array([1.0, 1.0]),
    }

    def solve(*, horizon, start="start"):
        return DeploymentValueIteration.values(
            transitions=transitions,
            terminal_values=terminal_values,
            competences=competences,
            horizon=horizon,
            start=start,
        )

    assert solve(horizon=0).tolist() == [0.0, 0.0]
    assert solve(horizon=0, start="goal").tolist() == [1.0, 1.0]
    assert solve(horizon=1).tolist() == [0.5, 0.75]
    # A failed safe attempt keeps its remaining steps; a failed risky one loses them.
    assert solve(horizon=2).tolist() == [0.25 + 0.75 * 0.5, 0.75]
    assert solve(horizon=2, start="detour").tolist() == [0.5, 0.75]
    assert solve(horizon=7, start="dead").tolist() == [0.0, 0.0]


def test_the_result_is_the_callers_own_array():
    values = DeploymentValueIteration.values(
        transitions={"start": ()},
        terminal_values={"goal": 1.0},
        competences={"unused": np.zeros(3)},
        horizon=2,
        start="goal",
    )
    values[:] = 5.0
    again = DeploymentValueIteration.values(
        transitions={"start": ()},
        terminal_values={"goal": 1.0},
        competences={"unused": np.zeros(3)},
        horizon=2,
        start="goal",
    )
    assert again.tolist() == [1.0, 1.0, 1.0]


def test_tossings_table_is_the_three_state_model_with_absorbing_success_and_failure():
    transitions = Tossing3DDeployment.transitions()
    assert transitions == {
        "ready": ((PICK_SKILL, "holding", "closed_gripper"),),
        "holding": ((TOSS_SKILL, "success", "failure"),),
        "closed_gripper": ((OPEN_GRIPPER_SKILL, "ready", "closed_gripper"),),
    }
    assert Tossing3DDeployment.TERMINAL_VALUES == {"success": 1.0, "failure": 0.0}


@pytest.mark.parametrize("horizon", [*range(13), 25, 60])
def test_tossing_batch_values_equal_the_replaced_recursion_in_every_bit(*, horizon: int) -> None:
    pick, toss, opened = _tossing_competences(count=25_000)
    expected = ReplacedImplementations.tossing_batch(
        toss=toss, pick=pick, opened=opened, horizon=horizon
    )
    actual = Tossing3DDeployment.values(
        competences={PICK_SKILL: pick, TOSS_SKILL: toss, OPEN_GRIPPER_SKILL: opened},
        horizon=horizon,
    )
    assert actual.dtype == expected.dtype == np.float64
    assert actual.shape == expected.shape == (41**3 + 100_000,)
    assert int(np.count_nonzero(actual != expected)) == 0


def test_tossing_values_keep_the_shape_of_a_cartesian_competence_grid():
    grid = np.linspace(0.0, 1.0, 7)
    pick, toss, opened = np.meshgrid(grid, grid, grid, indexing="ij")
    actual = Tossing3DDeployment.values(
        competences={PICK_SKILL: pick, TOSS_SKILL: toss, OPEN_GRIPPER_SKILL: opened}, horizon=4
    )
    expected = ReplacedImplementations.tossing_batch(toss=toss, pick=pick, opened=opened, horizon=4)
    assert actual.shape == (7, 7, 7)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("horizon", [0, 1, 2, 3, 4, 5, 8, 12])
def test_tossing_theta_values_equal_the_replaced_scalar_recursion_in_every_bit(
    *, horizon: int
) -> None:
    domain = TOSSING3D_DOMAIN.model_copy(update={"deployment_horizon": horizon})
    pick, toss, opened = _tossing_competences(count=250)
    rows = [*range(0, 41**3, 97), *range(41**3, len(pick))]
    differing = 0
    for row in rows:
        theta = Tossing3DTheta(
            pick=SkillHypothesis(competence=pick[row], learning_rate=0.0),
            toss=SkillHypothesis(competence=toss[row], learning_rate=0.0),
            open_gripper=SkillHypothesis(competence=opened[row], learning_rate=0.0),
        )
        actual = domain.evaluate_theta(theta=theta)
        assert type(actual) is float
        differing += actual != ReplacedImplementations.tossing_scalar(
            toss=float(toss[row]),
            pick=float(pick[row]),
            opened=float(opened[row]),
            horizon=horizon,
        )
    assert differing == 0


@pytest.mark.parametrize("horizon", [*range(15), 20, 30])
def test_shelf_values_equal_the_replaced_loop_in_every_bit(*, horizon: int) -> None:
    competences = _table_competences(skills=SHELF_DEPLOYMENT_SKILLS, count=1024)
    expected = ReplacedImplementations.table_values(
        moves=ReplacedImplementations.shelf_moves(),
        held="cube",
        competences=competences,
        horizon=horizon,
    )
    actual = SweepOffShelfDeployment.values(competences=competences, horizon=horizon)
    assert actual.shape == expected.shape == (4096,)
    assert int(np.count_nonzero(actual != expected)) == 0


@pytest.mark.parametrize("horizon", [*range(15), 20, 30])
def test_rack_values_equal_the_replaced_loop_in_every_bit(*, horizon: int) -> None:
    competences = _table_competences(skills=RACK_ROBOT_SKILLS, count=1024)
    expected = ReplacedImplementations.table_values(
        moves=ReplacedImplementations.rack_moves(),
        held="food",
        competences=competences,
        horizon=horizon,
    )
    actual = RackServingDeployment.values(competences=competences, horizon=horizon)
    assert actual.shape == expected.shape == (4096,)
    assert int(np.count_nonzero(actual != expected)) == 0


def test_the_shelf_and_the_rack_still_leave_a_failed_skills_state_unchanged():
    assert RackServingDeployment.transitions() == ReplacedImplementations.rack_moves()
    assert SweepOffShelfDeployment._transitions() == ReplacedImplementations.shelf_moves()
    for moves, held, start in (
        (ReplacedImplementations.shelf_moves(), "cube", (5, "empty", True)),
        (ReplacedImplementations.rack_moves(), "food", (RACK_NUM_FOOD, "empty", True)),
    ):
        transitions, terminal_values = DeploymentValueIteration.failure_keeps_state(
            moves=moves, solved=lambda state, held=held: state[0] == 0 and state[1] != held
        )
        assert start in transitions
        assert set(transitions) | set(terminal_values) == set(moves)
        assert not set(transitions) & set(terminal_values)
        assert set(terminal_values.values()) == {1.0}
        for state, options in transitions.items():
            assert options == tuple((skill, successor, state) for skill, successor in moves[state])


def test_every_domain_value_recorded_before_the_change_is_reproduced_in_every_bit():
    recorded = json.loads(FIXTURE.read_text())
    actual = RecordedDeploymentValues.compute()
    assert actual.keys() == recorded.keys()
    differing = sorted(key for key in recorded if actual[key] != recorded[key])
    assert not differing
    j_keys = [key for key in recorded if key.startswith("J/")]
    assert len(j_keys) == 20 * len(TOSSING_HORIZONS) * 2
    assert len({recorded[key] for key in j_keys}) > 200
    assert sum("/J/" in key for key in recorded) == 8
    assert sum("/expected/" in key for key in recorded) == 8 * len(TABLE_HORIZONS)


@pytest.mark.parametrize("horizon", range(13))
def test_the_closed_form_is_the_belief_expectation_of_the_shared_solvers_value(
    *, horizon: int
) -> None:
    """J stays an exact integral, not a sample mean. The two sides sum the same terms
    in different orders, so they agree to rounding rather than in every bit."""
    for name, state in RecordedDeploymentValues.tossing_states().items():
        supports, weights = [], []
        for skill in (PICK_SKILL, TOSS_SKILL, OPEN_GRIPPER_SKILL):
            belief = state.skill_beliefs[skill]
            support, indexes = np.unique(belief.competence_values(), return_inverse=True)
            supports.append(support)
            weights.append(np.bincount(indexes, weights=belief.arrays()[1]))
        pick, toss, opened = np.meshgrid(*supports, indexing="ij")
        joint = weights[0][:, None, None] * weights[1][None, :, None] * weights[2][None, None, :]
        integrated = float(
            np.sum(
                joint
                * Tossing3DDeployment.values(
                    competences={PICK_SKILL: pick, TOSS_SKILL: toss, OPEN_GRIPPER_SKILL: opened},
                    horizon=horizon,
                )
            )
        )
        closed_form = DeploymentPolicyExpectation.evaluate(
            pick=state.skill_beliefs[PICK_SKILL],
            toss=state.skill_beliefs[TOSS_SKILL],
            opened=state.skill_beliefs[OPEN_GRIPPER_SKILL],
            horizon=horizon,
        )
        assert closed_form == pytest.approx(integrated, abs=1e-13), name


def test_table_domains_expected_values_are_the_fixed_seed_means_they_always_were():
    for domain, moves, held, skills in (
        (
            SWEEP_OFF_SHELF_DOMAIN,
            ReplacedImplementations.shelf_moves(),
            "cube",
            SHELF_DEPLOYMENT_SKILLS,
        ),
        (RACK_SERVING_DOMAIN, ReplacedImplementations.rack_moves(), "food", RACK_ROBOT_SKILLS),
    ):
        beliefs = RecordedDeploymentValues.table_states(domain=domain)[
            "coarse-trained"
        ].skill_beliefs
        rng = np.random.default_rng(0)
        competences = {name: beliefs[name].sample(rng=rng, count=4096)[:, 0] for name in skills}
        expected = float(
            ReplacedImplementations.table_values(
                moves=moves, held=held, competences=competences, horizon=domain.deployment_horizon
            ).mean()
        )
        assert domain.expected_policy_value(beliefs=beliefs) == expected
