"""ReturnWiper to the wiper's home on top of the cupboard, from a captured state
(`data/returnwiper_states.jsonl`): seed 0 after PickWiper and SweepShelf(0.005, 0.05),
the wiper held at its one grasp with every cube in the box.

The checks come from the 2026-10-21 ReturnWiper diagnosis of launch ea9b45b15, whose
states had the wiper's old home in a slot on the third shelf and so cannot be restored
on this scene: there the drive to the slot base carried the wiper 7.7 mm into the back
panel unchecked, the grasp slipped, and the put-down re-aimed from it stalled."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.environment import (
    SKILL_IDS,
    SweepOffShelf3DEnvironment,
)
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import StateCapture, StateSet

STATES = Path(__file__).resolve().parent / "data/returnwiper_states.jsonl"

# Module-scoped fixtures: keep them on one xdist worker (see tests/test_xdist_groups.py).
pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("sweepoffshelf3d_test_return_wiper"),
]

AFTER_SWEEP = "returnwiper-edge-s0-after-sweep"


@pytest.fixture(name="env", scope="module")
def _env():
    # The states were captured in the lab2 scene; a state restores only there.
    env = SweepOffShelf3DEnvironment(scene_bg=True)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture(name="states", scope="module")
def _states():
    return {state.state_id: state for state in StateSet.load(path=STATES)}


def _restore(*, env, states, state_id):
    StateCapture.restore(env=env, state=states[state_id])
    return env.controllers()._skills  # noqa: SLF001


def _return_wiper(*, env):
    env.take_action(action=np.array([float(SKILL_IDS["ReturnWiper"]), 0.0, 0.0, 0.0]))
    return env.get_current_state().abstract_atoms


def _returned(*, env, atoms):
    shelf = env.controllers()._skills  # noqa: SLF001
    home = float(np.linalg.norm(shelf.wiper_pose()[:3, 3] - shelf.wiper_start[:3, 3]))
    return ("WiperHome", ("wiper_0",)) in atoms and ("HandEmpty", ("robot",)) in atoms, home


def test_a_wiper_lying_level_at_home_counts_as_put_down(*, env):
    """The fallback that lets go of a put-down stopped short measures the head against
    the board the wiper rests on. Measured from the wiper's start less the head's
    half-height, the support sat 7.5 mm below the board, so no head could come within
    the 3 mm gap and the fallback could never fire."""
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SLOT_PUT_DOWN_GAP

    env.hard_reset()
    shelf = env.controllers()._skills  # noqa: SLF001
    assert shelf.put_down_reached(support=shelf.wiper_home_support(), gap=SLOT_PUT_DOWN_GAP)


def test_a_base_whose_drive_cuts_the_carried_wiper_into_the_cupboard_is_refused(*, env, states):
    """From the carry pose, a drive to x 0.90 puts the carried wiper into the cupboard;
    the base ReturnWiper's route goes from, (0.72, 0.087), is clear."""
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import CARRY_Z

    shelf = _restore(env=env, states=states, state_id=AFTER_SWEEP)
    shelf.raise_hand(z=CARRY_Z, label="test", holding=True)
    shelf.recalibrate_grasp()
    assert shelf.carried_wiper_drive_hits(base=np.array([0.90, 0.087, 0.0]))
    assert not shelf.carried_wiper_drive_hits(base=np.array([0.72, 0.087, 0.0]))


def test_the_wiper_is_returned_home_at_the_grasp_it_swept_with(*, env, states):
    _restore(env=env, states=states, state_id=AFTER_SWEEP)
    atoms = _return_wiper(env=env)
    done, home = _returned(env=env, atoms=atoms)
    assert done, env.last_skill_error()
    assert home < 0.01


def test_a_put_down_that_collides_after_the_way_in_is_refused_and_backed_out(
    *, env, states, monkeypatch
):
    """The grasp slips 3 cm on the way in (as the drive into the back panel made it
    slip in the run): the put-down, re-aimed from the grasp as it now is, is re-checked
    and refused, and the wiper is backed out still in hand -- instead of the hand
    stalling against the board."""
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import ShelfSkills

    recalibrate = ShelfSkills.recalibrate_grasp
    calls = []

    def slipped_on_the_way_in(self):  # noqa: PLR0917 -- stands in for a method
        recalibrate(self)
        calls.append(None)
        if len(calls) == 2:  # the re-aim after the way in
            shift = np.eye(4)
            shift[2, 3] = 0.03
            self.in_hand = self.in_hand @ shift

    monkeypatch.setattr(ShelfSkills, "recalibrate_grasp", slipped_on_the_way_in)
    _restore(env=env, states=states, state_id=AFTER_SWEEP)
    atoms = _return_wiper(env=env)
    assert "put-down collides after the way in" in (env.last_skill_error() or "")
    assert ("HoldingWiper", ("robot", "wiper_0")) in atoms
