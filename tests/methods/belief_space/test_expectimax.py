import pytest
from pydantic import BaseModel, ConfigDict

from hitl_pmp.methods.belief_space.expectimax import ChanceOutcome, solve_expectimax


class _State(BaseModel):
    model_config = ConfigDict(frozen=True)

    competence: int
    cost: int = 0


def _stop_value(state: _State) -> float:  # noqa: PLR0917 (solver callback)
    return float(state.competence - state.cost)


def test_horizon_zero_stops() -> None:
    result = solve_expectimax(
        state=_State(competence=1),
        horizon=0,
        stop_value=_stop_value,
        actions=lambda _state: ("practice",),
        outcomes=lambda _state, _action: (
            ChanceOutcome(probability=1.0, next_state=_State(competence=10)),
        ),
    )
    assert result.value == 1.0
    assert result.action is None


def test_looks_past_an_initially_unhelpful_setup_action() -> None:
    def actions(state: _State) -> tuple[str, ...]:  # noqa: PLR0917 (solver callback)
        return ("setup",) if state.cost == 0 else ("practice",)

    def outcomes(  # noqa: PLR0917 (solver callback)
        state: _State, action: str
    ) -> tuple[ChanceOutcome[_State], ...]:
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
        actions=lambda _state: ("neutral",),
        outcomes=lambda state, _action: (ChanceOutcome(probability=1.0, next_state=state),),
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
def test_rejects_malformed_chance_distributions(  # noqa: PLR0917 (pytest parametrization)
    branches: tuple[ChanceOutcome[_State], ...],
) -> None:
    with pytest.raises(ValueError):
        solve_expectimax(
            state=_State(competence=0),
            horizon=1,
            stop_value=_stop_value,
            actions=lambda _state: ("bad",),
            outcomes=lambda _state, _action: branches,
        )
