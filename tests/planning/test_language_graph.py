"""Real Fast Downward tests for frozen learned-state deployment."""

import pytest

from hitl_pmp.planning.language_graph import LanguageGraphPlanner


def make_planner(*, edges, competences):
    return LanguageGraphPlanner(
        clusters=("start", "middle", "goal", "dead"),
        goals=("goal",),
        edges=edges,
        competences=competences,
        human_actions=frozenset({"human"}),
    )


def test_reliable_long_route_beats_unreliable_shortcut():
    planner = make_planner(
        edges=[
            dict(source="start", action="shortcut", success="goal"),
            dict(source="start", action="prepare", success="middle"),
            dict(source="middle", action="finish", success="goal"),
        ],
        competences={"shortcut": 0.1, "prepare": 0.9, "finish": 0.9},
    )
    assert planner.plan(cluster="start") == ("prepare", "finish")
    assert planner.action(cluster="middle") == "finish"


def test_conditional_destinations_are_not_free_choices():
    planner = make_planner(
        edges=[
            dict(source="start", action="risky", success={"goal": 0.01, "dead": 0.99}),
            dict(source="start", action="prepare", success="middle"),
            dict(source="middle", action="finish", success="goal"),
        ],
        competences={"risky": 0.99, "prepare": 0.8, "finish": 0.8},
    )
    assert planner.action(cluster="start") == "prepare"


def test_human_is_excluded_and_no_plan_is_explicit():
    planner = make_planner(
        edges=[
            dict(source="start", action="human", success="goal"),
            dict(source="middle", action="finish", success="goal"),
        ],
        competences={"finish": 0.9},
    )
    assert planner.action(cluster="start") is None
    assert planner.action(cluster=None) is None
    assert planner.action(cluster="goal") is None


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_invalid_competence_is_rejected(*, probability):
    with pytest.raises(ValueError):
        make_planner(
            edges=[dict(source="start", action="finish", success="goal")],
            competences={"finish": probability},
        )


def test_missing_competence_cannot_silently_fall_back_to_unit_cost():
    with pytest.raises(KeyError):
        make_planner(edges=[dict(source="start", action="finish", success="goal")], competences={})
