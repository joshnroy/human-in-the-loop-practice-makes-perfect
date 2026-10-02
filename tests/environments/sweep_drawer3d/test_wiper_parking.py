"""A closed gripper is not a held wiper."""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("shapely") is None, reason="shapely ships with KINDER"
)


def _hold():
    from hitl_pmp.environments.sweep_drawer3d.primitives import WiperHold

    return WiperHold


def test_a_wiper_three_quarters_of_a_metre_from_the_gripper_is_not_in_the_hand() -> None:
    """Seed 28: the environment finds no free spot for the wiper and leaves it on the
    floor behind the island. The stock PickWiper reports success there with the fingers
    shut on nothing and the wiper 0.758 m away."""
    gripper = np.array([-0.297, 0.322, 0.61])
    wiper = np.array([0.073, -0.11, 0.109])
    assert not _hold().in_hand(gripper=gripper, wiper=wiper)


@pytest.mark.parametrize(
    ("gripper", "wiper"),
    [
        ((1.033, 0.076, 0.494), (1.008, 0.076, 0.458)),
        ((1.058, 0.009, 0.494), (1.038, 0.009, 0.448)),
        ((1.306, -0.331, 0.609), (1.275, -0.331, 0.566)),
    ],
)
def test_a_wiper_taken_by_its_handle_is_in_the_hand(
    *, gripper: tuple[float, ...], wiper: tuple[float, ...]
) -> None:
    """As measured after the stock attempt on seeds 0, 5 and 12: 4 to 5 cm apart."""
    assert _hold().in_hand(gripper=np.array(gripper), wiper=np.array(wiper))
