"""The shipped harvested states: failures from launch shelf-20261023b-1f209df65's
practice, in the wiper-on-top scene, each one loading on the live scene with its case's
preconditions holding. They replace the 2026-10-14 old-scene harvest (Josh, 2026-10-23).

Only the seven runs that kept a `recording.jsonl` (seed 0 of all six arms, and EES seed
5) can be harvested: the other 29 kept step events only, which hold no object poses."""

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import (
    StateCapture,
    StateSet,
)

SHIPPED = Path(__file__).resolve().parents[4] / "benchmarks/sweepoffshelf3d/states.jsonl"
LAUNCH = "shelf-launch-20261023b"


def _harvested():
    return [state for state in StateSet.load(path=SHIPPED) if state.family == "harvested"]


def test_every_harvested_state_comes_from_launch_shelf_20261023b() -> None:
    harvested = _harvested()
    assert harvested
    stale = [state.state_id for state in harvested if LAUNCH not in state.provenance]
    assert not stale


def test_the_harvested_set_is_the_failures_that_replayed_on_the_launch_commit() -> None:
    """Of 26 deduplicated candidates (13 shelf picks, 13 box returns), the ones whose
    rebuilt state failed again on 1f209df65; PickWiper never failed empty-handed."""
    counts = Counter(case.skill for state in _harvested() for case in state.cases)
    assert counts == {"PickCubeFromShelf": 5, "ReturnCubeToShelfFromBox": 3}
    assert all(len(state.cases) == 1 for state in _harvested())


@pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="needs KINDER",
)
@pytest.mark.xdist_group("sweepoffshelf3d_skill_benchmark_harvested_states")
def test_every_harvested_state_loads_with_its_case_applicable() -> None:
    env = SweepOffShelf3DEnvironment(scene_bg=True)
    try:
        refused = []
        for state in _harvested():
            StateCapture.restore(env=env, state=state)
            atoms = StateCapture.atoms(env=env)
            refused += [
                f"{state.state_id}: {case.skill}({case.cube})"
                for case in state.cases
                if not StateCapture.applicable(case=case, atoms=atoms)
            ]
        assert not refused
    finally:
        env.close()
