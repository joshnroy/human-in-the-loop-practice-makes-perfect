"""ReturnWiper from the 2026-10-21 ReturnWiper diagnosis's captured states (launch
ea9b45b15, `data/returnwiper_states.jsonl`):

- pomdp0 (cccc-determinized-always-ask/pomdp/0, step 9598): the wiper held with the
  grasp slipped 22 mm / 19.5 deg off the shelf grasp, still booked as the shelf grasp;
  refused 857 times;
- pomdp0 before (step 9347): the straight drive to the slot base put the carried
  wiper 7.7 mm into the cupboard's back panel, unchecked, and the grasp slipped there;
- det4 (cccc-determinized/pomdp/4, step 5090): the floor set-down before the regrasp
  stalled 9-15 mm above the floor, the bracelet link on the floor."""

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

SLIPPED = "returnwiper-pomdp0-slipped-grasp-refused"
BEFORE_SLOT = "returnwiper-pomdp0-before-slot-collision"
BEFORE_FLOOR = "returnwiper-det4-before-floor-stall"


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


def _restore(*, env, states, state_id, stale_shelf_grasp=False):
    StateCapture.restore(env=env, state=states[state_id])
    shelf = env.controllers()._skills  # noqa: SLF001
    if stale_shelf_grasp:
        # The run's own bookkeeping: `ShelfSkills` lived across skills and kept the
        # grasp kind its PickWiper booked, whatever the wiper did in the hand since.
        shelf.grasp_kind = "shelf"
    return shelf


def _return_wiper(*, env):
    env.take_action(action=np.array([float(SKILL_IDS["ReturnWiper"]), 0.0, 0.0, 0.0]))
    return env.get_current_state().abstract_atoms


def _returned(*, env, atoms):
    shelf = env.controllers()._skills  # noqa: SLF001
    home = float(np.linalg.norm(shelf.wiper_pose()[:3, 3] - shelf.wiper_start[:3, 3]))
    return ("WiperHome", ("wiper_0",)) in atoms and ("HandEmpty", ("robot",)) in atoms, home


def test_a_slipped_grasp_measures_as_no_fixed_grasp(*, env, states):
    shelf = _restore(env=env, states=states, state_id=SLIPPED, stale_shelf_grasp=True)
    assert shelf.grasp_kind == "shelf"
    assert shelf.measured_grasp_kind() is None


def test_the_wiper_held_by_a_slipped_grasp_is_returned(*, env, states):
    """Refused 857/857 times in the run; the diagnosis's live probe, with the grasp kind
    re-derived, put it home through the floor regrasp in 1334 control steps."""
    _restore(env=env, states=states, state_id=SLIPPED, stale_shelf_grasp=True)
    atoms = _return_wiper(env=env)
    done, home = _returned(env=env, atoms=atoms)
    assert done, env.last_skill_error()
    assert home < 0.01


def test_a_slot_base_whose_drive_cuts_the_carried_wiper_into_the_cupboard_is_refused(
    *, env, states
):
    """The run drove to (0.80, 0.065), whose arm route is clear, but whose straight
    drive put the carried wiper 7.7 mm into the back panel; (0.64, 0.065) is clean."""
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import CARRY_Z

    shelf = _restore(env=env, states=states, state_id=BEFORE_SLOT)
    shelf.raise_hand(z=CARRY_Z, label="test", holding=True)
    shelf.recalibrate_grasp()
    assert shelf.carried_wiper_drive_hits(base=np.array([0.80, 0.065, 0.0]))
    assert not shelf.carried_wiper_drive_hits(base=np.array([0.64, 0.065, 0.0]))


def test_the_wiper_is_returned_without_cutting_into_the_cupboard_on_the_way(*, env, states):
    """In the run the wiper slipped on the drive and the put-down stalled in the slot;
    with the drive checked, the diagnosis's probe returned it in 350 control steps."""
    _restore(env=env, states=states, state_id=BEFORE_SLOT)
    atoms = _return_wiper(env=env)
    done, home = _returned(env=env, atoms=atoms)
    assert done, env.last_skill_error()
    assert home < 0.01


def test_the_wiper_set_down_on_the_floor_is_let_go_once_level_and_near_the_floor(*, env, states):
    """det/4: the hand stalled 9-15 mm above the floor (`Physical move failed`). Let go
    level and under 2 cm up, the diagnosis's probe returned the wiper in 1858 steps."""
    _restore(env=env, states=states, state_id=BEFORE_FLOOR)
    atoms = _return_wiper(env=env)
    done, home = _returned(env=env, atoms=atoms)
    assert done, env.last_skill_error()
    assert home < 0.01


def test_a_put_down_that_collides_after_the_way_in_is_backed_out_and_regrasped(
    *, env, states, monkeypatch
):
    """With the drive check switched off, the run's own drive happens: the carried
    wiper cuts the back panel and the grasp slips. The put-down, re-aimed from the
    grasp as it now is, is re-checked, refused, and the wiper is taken again on the
    floor before going in once more -- instead of stalling against the board."""
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import ShelfSkills

    monkeypatch.setattr(ShelfSkills, "carried_wiper_drive_hits", lambda self, *, base: False)
    _restore(env=env, states=states, state_id=BEFORE_SLOT)
    atoms = _return_wiper(env=env)
    done, home = _returned(env=env, atoms=atoms)
    assert done, env.last_skill_error()
    assert home < 0.01
    notes = [n.reason for n in env.controllers().notes()]
    assert any("put-down collides after the way in" in r for r in notes), notes
