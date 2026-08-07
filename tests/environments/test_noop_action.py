"""Every domain's `noop_action()` must genuinely do nothing.

Cross-domain rather than per-domain (like test_operator_dynamics_fidelity.py beside
it) because the property is a claim about the `Environment` interface, not about any
one domain: a `Method` that cannot plan degrades to `noop_action()`, and if that
action turns out to *do* something the run silently reports whatever it caused.

Tossing3D is absent by necessity -- one `take_action` there is hundreds of MuJoCo
ticks and needs KINDER installed. Its own decode is pinned in
tests/environments/tossing3d/test_environment.py instead, with no simulator.
"""

import numpy as np
import pytest

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.environments.ballring.environment import BallRingEnvironment
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.tossingroomsplitpickupweight.environment import (
    TossingRoomSplitPickupWeightEnvironment,
)

# Classes, not instances: each test builds its own and resets it, so no test can
# depend on the state another one happened to leave behind. A module-level list of
# live environments made these order-dependent, and one of them passed only because
# pytest runs a file top-down.
_ENVIRONMENT_TYPES: list[type[Environment]] = [
    LightSwitchEnvironment,
    BallRingEnvironment,
    TossingRoomSplitPickupWeightEnvironment,
]

# One entry since the three frozen-weight forks were retired. Kept as a list, and kept
# parametrized, because the property is a claim about the Tossing Room *shape* rather
# than about one class -- a second Tossing Room domain joins here, not by copying tests.
_TOSSING_ROOM_TYPES = [
    TossingRoomSplitPickupWeightEnvironment,
]


def _snapshot(*, state: State) -> dict[str, tuple[float, ...]]:
    """A comparable copy of a State: `State` wraps mutable numpy arrays and has no
    usable `==`, and `take_action` hands back a state the environment keeps
    mutating, so the comparison has to be against a value taken before the step."""
    return {obj.name: tuple(float(v) for v in features) for obj, features in state.data.items()}


def _reset(*, env_type: type[Environment]) -> Environment:
    env = env_type()
    env.hard_reset()
    return env


@pytest.mark.parametrize("env_type", _ENVIRONMENT_TYPES, ids=lambda t: t.__name__)
def test_noop_action_leaves_the_state_untouched(*, env_type: type[Environment]) -> None:
    env = _reset(env_type=env_type)
    before = _snapshot(state=env.get_current_state())
    env.take_action(action=env.noop_action())
    assert _snapshot(state=env.get_current_state()) == before


@pytest.mark.parametrize("env_type", _ENVIRONMENT_TYPES, ids=lambda t: t.__name__)
def test_noop_action_is_inside_the_domains_own_action_space(*, env_type: type[Environment]) -> None:
    """A no-op is still an action, so it has to be one the domain would accept.
    Ball-Ring's Box is genuinely bounded, so an out-of-range sentinel would be a
    value no `Method` could legally emit.

    Bounds rather than a bare `action_space.contains(...)`: every action in this repo
    is float64 and Ball-Ring's Box is float32, for which `contains` is False whatever
    the values are. That mismatch is repo-wide and predates this test -- no skill's
    `compute_action` would pass either -- so casting first is what makes the
    assertion about the *numbers*, which is the part that matters here."""
    env = _reset(env_type=env_type)
    space = env.action_space
    assert space.contains(env.noop_action().astype(space.dtype))


@pytest.mark.parametrize("env_type", _TOSSING_ROOM_TYPES, ids=lambda t: t.__name__)
def test_the_tossing_rooms_noop_id_is_not_a_real_skill_id(
    *, env_type: type[TossingRoomSplitPickupWeightEnvironment]
) -> None:
    """State-invariance alone does not discriminate `noop_action()` from
    `np.zeros(3)` on these domains -- both are inert today (see the next test) --
    so the property worth pinning is that the id names no skill at all."""
    env = env_type()
    assert env.SKILL_NOOP not in {
        env.SKILL_PICKUP,
        env.SKILL_MOVE_ROOM,
        env.SKILL_THROW,
        env.SKILL_PRESS,
    }


def test_a_zero_vector_is_not_a_no_op_on_ball_ring() -> None:
    """The defect this interface exists to close, pinned as behaviour rather than
    left as prose: `np.zeros(action_space.shape)` decodes to "navigate to (0, 0)" on
    Ball-Ring (`move_or_pickplace == 0`), which really moves the robot. The other
    broken domain is Tossing3D, where zero is `pick_shelf` at distance 0.0 -- pinned
    in that domain's own tests, since it needs no simulator to show."""
    env = _reset(env_type=BallRingEnvironment)
    before = _snapshot(state=env.get_current_state())
    env.take_action(action=np.zeros(env.action_space.shape))
    assert _snapshot(state=env.get_current_state()) != before


@pytest.mark.parametrize("env_type", _TOSSING_ROOM_TYPES, ids=lambda t: t.__name__)
def test_a_zero_vector_is_inert_on_the_tossing_rooms_only_by_coincidence(
    *, env_type: type[TossingRoomSplitPickupWeightEnvironment]
) -> None:
    """Recorded so the claim is measured rather than assumed, because the natural
    reading of `SKILL_PICKUP == 0` is that zeros was broken here too -- and it was
    not, so no archived Tossing Room result is affected by this change.

    Zeros survives only because slot 1 also rounds to 0 and `_apply_pickup` requires
    a real item kind there. The control is the same action with a valid kind, which
    does change the state -- so the skill id really is being dispatched, and the
    inertness lives one field over."""
    env = _reset(env_type=env_type)
    before = _snapshot(state=env.get_current_state())
    env.take_action(action=np.zeros(3))
    assert _snapshot(state=env.get_current_state()) == before

    env = _reset(env_type=env_type)
    env.take_action(action=np.array([float(env.SKILL_PICKUP), float(env.TRASH_KIND), 0.0]))
    assert _snapshot(state=env.get_current_state()) != before
