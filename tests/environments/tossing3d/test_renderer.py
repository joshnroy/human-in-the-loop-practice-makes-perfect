"""Tests for the one thing in `renderer.py` that is not pixels: the caption.

`render_frame` draws the live MuJoCo scene, so it needs the optional dependency and is
covered from `test_kinder_fidelity.py`'s side. `caption` is a pure function of `State`,
it is the part a reader actually verifies the success claim from, and
`scripts/render_tossing3d_demo.py` reuses it for the smooth clip -- so it is worth
pinning on its own, in CI, with no simulator.
"""

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.renderer import Tossing3DRenderer

from .conftest import BIN_X, build_state


def test_the_caption_reports_the_cube_against_the_goal_bounds_and_the_bin() -> None:
    """All three numbers, because any two of them are misleading on their own.

    This scene's whole trap is that KINDER's goal region stops short of the bin, so a
    solved episode ends with the cube on the floor and the bin untouched. A caption
    showing only the cube reads as a near miss; one showing only the region hides that
    the bin is scenery. Together they are checkable.
    """
    env = Tossing3DEnvironment()
    caption = Tossing3DRenderer.caption(
        state=build_state(env=env, cube=(1.9139, 0.0116, 0.0249)), env=env, label="Toss"
    )
    assert caption.startswith("Toss")
    assert "cube x=1.91 z=0.02" in caption
    assert "goal x in [1.85, 2.15]" in caption
    assert f"bin x={BIN_X:.2f}" in caption


def test_the_caption_reports_height_rather_than_the_holding_flag() -> None:
    """`holding` is the height proxy `z > 0.2`, which reads as a grasp. On a per-tick
    clip that is false for the whole flight of a toss, so the caption prints z instead
    -- the quantity the flag is derived from -- and never claims a released cube is
    still in the gripper."""
    env = Tossing3DEnvironment()
    airborne = Tossing3DRenderer.caption(
        state=build_state(env=env, cube=(1.4, 0.0, 0.52), holding=1.0), env=env, label="Toss"
    )
    assert "z=0.52" in airborne
    assert "holding" not in airborne


def test_the_caption_calls_an_unlabelled_frame_the_start() -> None:
    """`render_frame` passes `label=None` for the pre-rollout frame, and the smooth clip
    does the same for its first captured tick."""
    env = Tossing3DEnvironment()
    caption = Tossing3DRenderer.caption(
        state=build_state(env=env, cube=(0.65, 0.0, 0.59)), env=env, label=None
    )
    assert caption.startswith("start")


def test_the_widest_caption_fits_inside_the_storyboards_figure() -> None:
    """Matplotlib silently draws text off the canvas rather than warning, and this
    caption is already 584 of the figure's 640 px at its widest -- the longest skill name
    carrying the cube. Anything added to `caption` has to be measured, so measure it."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    env = Tossing3DEnvironment()
    widest = Tossing3DRenderer.caption(
        state=build_state(env=env, cube=(0.75, 0.0, 0.59), holding=1.0),
        env=env,
        label="MoveToThrowPose",
    )
    figure = plt.figure(figsize=Tossing3DRenderer.figure_size, dpi=Tossing3DRenderer.dpi)
    canvas = FigureCanvasAgg(figure)
    artist = figure.text(0.5, 0.95, widest, ha="center", va="center", fontsize=11)
    canvas.draw()
    extent = artist.get_window_extent(renderer=canvas.get_renderer())
    plt.close(figure)
    assert extent.x0 > 0, f"caption overflows the figure on the left: {widest}"
    assert extent.x1 < figure.bbox.width, f"caption overflows on the right: {widest}"
