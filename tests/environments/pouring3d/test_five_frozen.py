"""Five-object count goals and privileged reset packing."""

from types import SimpleNamespace

import numpy as np

from hitl_pmp.environments.pouring3d.five_frozen import FiveFrozenShrimpScene


def test_five_object_goal_requires_exact_split_and_settling() -> None:
    scene = object.__new__(FiveFrozenShrimpScene)
    scene.requested_count = 2
    scene.cube_locations = lambda: ["tray"] * 2 + ["pitcher"] * 3
    scene.cubes_settled = lambda: True
    assert scene._check_goals()
    scene.cube_locations = lambda: ["tray"] * 2 + ["pitcher"] * 2 + ["floor"]
    assert not scene._check_goals()
    scene.cube_locations = lambda: ["tray"] * 2 + ["pitcher"] * 3
    scene.cubes_settled = lambda: False
    assert not scene._check_goals()


def test_reset_layout_has_separated_layers_and_five_positions() -> None:
    points = FiveFrozenShrimpScene.reset_offsets()
    assert points.shape == (5, 3)
    assert np.all(np.linalg.norm(points[:, :2], axis=1) <= 0.0241)
    assert points[3:, 2].min() - points[:3, 2].max() > 0.011


def test_ready_checks_all_five_pieces() -> None:
    from hitl_pmp.environments.pouring3d.five_frozen import FiveFrozenController

    controller = SimpleNamespace(
        held=None,
        scene=SimpleNamespace(
            cube_locations=lambda: ["pitcher"] * 5,
            cubes_settled=lambda: True,
        ),
        vessel_home=lambda **kwargs: True,
    )
    assert FiveFrozenController.ready(controller)
    controller.scene.cube_locations = lambda: ["pitcher"] * 3
    assert not FiveFrozenController.ready(controller)
