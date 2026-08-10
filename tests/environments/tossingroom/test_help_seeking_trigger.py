from hitl_pmp.core.method.method import HumanHelpRequested
from hitl_pmp.core.method.types import LabeledAction, Policy
from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.skill_provider import TossingRoomSkillProvider
from hitl_pmp.methods.help_seeking import HelpSeekingPolicy, HelpSeekingTrigger


def _acting_policy() -> Policy:
    return lambda state: LabeledAction(action=TossingRoomEnvironment().noop_action(), label="acted")


def test_on_no_applicable_skill_essentially_never_fires_on_tossing_room() -> None:
    """The defect this baseline documents, asserted rather than merely observed: the
    deleted `agent-signal` arm hooked `InteractionComplete` directly and measured 0
    interventions on 0/10 seeds here, byte-identical to no-human, because `MoveRoom`
    requires only `RobotInRoom` + `CanMoveRoom` -- nothing about holding an item or a
    bin being full -- and every room has at least one legal `CanMoveRoom` grounding
    (the one-way ledge blocks only the single rightward step out of
    `blocked_right_from`; every other adjacent step, in either direction, is legal).
    So a ground skill is applicable in every room the robot can be in, and this
    trigger -- which checks exactly that condition -- should ask (essentially) never.

    Swept over every room the robot can occupy, holding every other feature of the
    task's own initial state fixed, rather than only the canonical start room: the
    claim is about the trigger's condition given `MoveRoom`'s preconditions, not about
    one particular state."""
    env = TossingRoomEnvironment()
    provider = TossingRoomSkillProvider(env=env)
    policy = HelpSeekingPolicy(
        trigger=HelpSeekingTrigger.ON_NO_APPLICABLE_SKILL, skill_provider=provider
    )

    base_state = env.build_initial_state(weight_seed=0)
    env.set_state(state=base_state)

    asks = 0
    for room_index in range(env.num_rooms):
        state = env.get_current_state().model_copy(deep=True)
        state.set(obj=env.robot, feature_name="room", feature_val=float(room_index))
        try:
            policy.step(state=state, inner_policy=_acting_policy())
        except HumanHelpRequested:
            asks += 1

    assert asks == 0, f"expected the trigger to fire in 0/{env.num_rooms} rooms, got {asks}"
