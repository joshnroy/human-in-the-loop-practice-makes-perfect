from hitl_pmp.environments.tworoutes.environment import TwoRoutesEnvironment
from hitl_pmp.environments.tworoutes.predicates import TwoRoutesPredicates


def test_each_predicate_reads_its_own_state_flag() -> None:
    env = TwoRoutesEnvironment()
    state = env.build_state(done=True, via_b=True, after_b=True)
    holding = {
        predicate.name
        for predicate in TwoRoutesPredicates.all()
        if predicate.holds(state, (env.robot,))
    }
    assert holding == {"Done", "ViaB", "AfterB"}


def test_the_predicates_are_the_six_flags_in_order() -> None:
    assert [p.name for p in TwoRoutesPredicates.all()] == [
        "Ready",
        "Done",
        "ViaA",
        "ViaB",
        "AfterA",
        "AfterB",
    ]
    assert TwoRoutesPredicates.get(name="Ready").types == (TwoRoutesEnvironment.robot_type,)
