import pytest
from pydantic import BaseModel, ConfigDict

from hitl_pmp.methods.belief_space.expectimax import solve_expectimax
from hitl_pmp.methods.belief_space.types import ChanceOutcome


class _State(BaseModel):
    model_config = ConfigDict(frozen=True)

    competence: int
    cost: int = 0


def _stop_value(*, state: _State) -> float:
    return float(state.competence - state.cost)


def test_horizon_zero_stops() -> None:
    result = solve_expectimax(
        state=_State(competence=1),
        horizon=0,
        stop_value=_stop_value,
        actions=lambda *, state: ("practice",),
        outcomes=lambda *, state, action: (
            ChanceOutcome(probability=1.0, next_state=_State(competence=10)),
        ),
    )
    assert result.value == 1.0
    assert result.action is None


def test_looks_past_an_initially_unhelpful_setup_action() -> None:
    def actions(*, state: _State) -> tuple[str, ...]:
        return ("setup",) if state.cost == 0 else ("practice",)

    def outcomes(*, state: _State, action: str) -> tuple[ChanceOutcome[_State], ...]:
        if action == "setup":
            return (
                ChanceOutcome(
                    probability=1.0,
                    next_state=_State(competence=state.competence, cost=1),
                ),
            )
        return (
            ChanceOutcome(probability=0.75, next_state=_State(competence=6, cost=2)),
            ChanceOutcome(probability=0.25, next_state=_State(competence=2, cost=2)),
        )

    shallow = solve_expectimax(
        state=_State(competence=2),
        horizon=1,
        stop_value=_stop_value,
        actions=actions,
        outcomes=outcomes,
    )
    deep = solve_expectimax(
        state=_State(competence=2),
        horizon=2,
        stop_value=_stop_value,
        actions=actions,
        outcomes=outcomes,
    )
    assert shallow.action is None
    assert deep.action == "setup"
    assert deep.value == pytest.approx(3.0)


def test_stop_wins_an_exact_tie() -> None:
    result = solve_expectimax(
        state=_State(competence=2),
        horizon=1,
        stop_value=_stop_value,
        actions=lambda *, state: ("neutral",),
        outcomes=lambda *, state, action: (ChanceOutcome(probability=1.0, next_state=state),),
    )
    assert result.action is None


@pytest.mark.parametrize(
    "branches",
    [
        (),
        (
            ChanceOutcome(probability=0.4, next_state=_State(competence=1)),
            ChanceOutcome(probability=0.4, next_state=_State(competence=2)),
        ),
        (
            ChanceOutcome(probability=0.0, next_state=_State(competence=1)),
            ChanceOutcome(probability=1.0, next_state=_State(competence=2)),
        ),
    ],
)
def test_rejects_malformed_chance_distributions(
    *,
    branches: tuple[ChanceOutcome[_State], ...],
) -> None:
    with pytest.raises(ValueError):
        solve_expectimax(
            state=_State(competence=0),
            horizon=1,
            stop_value=_stop_value,
            actions=lambda *, state: ("bad",),
            outcomes=lambda *, state, action: branches,
        )


def test_shared_successor_is_evaluated_once_at_each_depth() -> None:
    visits: dict[_State, int] = {}
    initial = _State(competence=0)
    successor = _State(competence=2)

    def stop_value(*, state: _State) -> float:
        visits[state] = visits.get(state, 0) + 1
        return _stop_value(state=state)

    result = solve_expectimax(
        state=initial,
        horizon=1,
        stop_value=stop_value,
        actions=lambda *, state: ("first", "second"),
        outcomes=lambda *, state, action: (ChanceOutcome(probability=1.0, next_state=successor),),
    )
    assert result.action == "first"
    assert result.value == 2.0
    assert visits == {initial: 1, successor: 1}


@pytest.mark.parametrize("probability", [-0.1, float("nan"), float("inf")])
def test_rejects_invalid_probabilities(*, probability: float) -> None:
    with pytest.raises(ValueError, match="chance probability must be finite and positive"):
        solve_expectimax(
            state=_State(competence=0),
            horizon=1,
            stop_value=_stop_value,
            actions=lambda *, state: ("invalid",),
            outcomes=lambda *, state, action: (
                ChanceOutcome(probability=probability, next_state=state),
            ),
        )


def test_rejects_negative_horizon_before_evaluating_model() -> None:
    def stop_value(*, state: _State) -> float:
        del state
        pytest.fail("invalid horizon must be rejected before the model is called")

    with pytest.raises(ValueError, match="horizon must be non-negative"):
        solve_expectimax(
            state=_State(competence=0),
            horizon=-1,
            stop_value=stop_value,
            actions=lambda *, state: (),
            outcomes=lambda *, state, action: (),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_stop_value(*, value: float) -> None:
    with pytest.raises(ValueError, match="stop value must be finite"):
        solve_expectimax(
            state=_State(competence=0),
            horizon=0,
            stop_value=lambda *, state: value,
            actions=lambda *, state: (),
            outcomes=lambda *, state, action: (),
        )
