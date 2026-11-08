"""The human batch return succeeds while the robot holds the scraper over the rack.

`batch_return_curtis1_held_scraper_on_rack_front.json` is the state Della run
rack-prior2x2-b406ad8-20261106 curtis-p025/pomdp/1 crashed from at practice step
16,364: a Sweep failed at its fixture check and left the held scraper's blade resting
on the rack's front edge, centred on y = 0. The batch return keeps a held scraper in
the hand, and that episode's start draw was the lined-up "stack", whose five centres
sit against the front edge on the plate's centre line, so all of its draws met the
blade and the return raised "no collision-free food return placement".
"""

import importlib.util
import json
from pathlib import Path

import pytest

from .test_welded_food import restored

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator"
)

FIXTURE = (
    Path(__file__).parent / "fixtures" / "batch_return_curtis1_held_scraper_on_rack_front.json"
)


def _restored(*, tmp_path: Path, start_layout: str):
    plain = json.loads(FIXTURE.read_text())
    plain["config"]["start_layout"] = start_layout
    plain["target"] = None
    path = tmp_path / FIXTURE.name
    path.write_text(json.dumps(plain))
    env, _ = restored(fixture=path)
    return env


@pytest.mark.parametrize("start_layout", ["stack", "scattered"])
def test_batch_return_places_the_food_around_a_held_scraper_on_the_rack(*, tmp_path, start_layout):
    env = _restored(tmp_path=tmp_path, start_layout=start_layout)
    try:
        backend = env.backend()
        assert backend.held(name=backend.wiper_name)
        env.reset_movables(destination="human_batch_return")
        assert backend.held(name=backend.wiper_name)
        atoms = env.get_current_state().abstract_atoms
        for name in backend.cube_names:
            assert ("OnRack", (name, "rack_0")) in atoms
        # The lined-up pile cannot fit past the blade, so the person returns the five
        # scattered over the free rack instead, and the episode's start kind says so.
        assert backend.start_kind() == "scattered"
    finally:
        env.close()
