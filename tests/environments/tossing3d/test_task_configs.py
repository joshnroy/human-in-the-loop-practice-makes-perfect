"""The tests for this repo's own Tossing3D task config, `task_configs/`.

KINDER's shipped `Tossing3D-o1` has a bin its goal region no longer describes: upstream
commit `1183de7` moved `bin_init_region` from x = 2.0 to x = 2.23 and left
`blocks_goal_region` where it was, so the scored region sits on the open floor and the
bin is scenery. `o2` still has the bin at 2.0, where it matches. This repo ships a task
config that puts o1's bin back at o2's placement, and these tests pin what that buys.

Two kinds here, deliberately separated:

* **Provenance**, which needs no simulator: our JSON is upstream's o1 with exactly one
  key changed. This runs in CI, and is what stops the file silently drifting into a
  bespoke scene nobody validated.
* **The coincidence property**, which genuinely opens MuJoCo. These assert against the
  *live* `Region.bbox` and the bin's *actual* MuJoCo geoms -- never against numbers
  copied out of the JSON. That is the whole point: `MujocoGround._create_regions`
  inflates the JSON range by `ground_placement_threshold` (0.05 m) per side at load, so
  a test written against the JSON literal would re-encode the very mismatch this config
  exists to remove. Skipped where `kindergarden` is absent (CI included).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
from hitl_pmp.environments.tossing3d.skill_oracle_policy import SkillOraclePolicy

from .conftest import BIN_X, GOAL_REGION, kinder_available

_ENV = Tossing3DEnvironment

# How closely two independently-built boxes have to agree to count as "the same box".
# The bin is placed by sampling a 1 mm-wide `bin_init_region`, so its footprint lands
# within a millimetre of nominal rather than exactly on it; 2 mm is comfortably inside
# that and nowhere near the 5 cm inflation or the 8 cm mismatch this config removes.
COINCIDENCE_TOL = 0.002

# Where a cube comes to rest on the bin's interior floor, as opposed to 0.025 on the
# ground. Measured, and documented on `KinderBackend.held_height`; asserted to lie
# inside the live bin's z extent by the tests below rather than trusted blindly.
CUBE_REST_Z_IN_BIN = 0.044


# --------------------------------------------------------------------------------------
# Provenance -- no simulator, so CI runs these.
# --------------------------------------------------------------------------------------


def upstream_task_config(*, name: str) -> dict:
    """A stock KINDER task config, read from the installed package."""
    import kinder  # noqa: PLC0415 (optional dependency)

    tasks = Path(kinder.__file__).parent / "envs/dynamic3d/tasks/Tossing3D"
    return json.loads((tasks / f"{name}.json").read_text())


def test_the_shipped_config_says_where_it_came_from_and_what_it_changed() -> None:
    """Runs without the simulator, so CI checks the file at all.

    The literals here are pins on *our own* file -- "we changed the bin and nothing
    else" -- not a model of KINDER's scoring. That distinction matters: the goal box
    KINDER actually tests is the raw range below inflated by 0.05 m per side, and it is
    read from the live `Region.bbox` by the tests further down, never from here.
    """
    config = json.loads(KinderBackend.COINCIDENT_BIN_TASK_CONFIG.read_text())

    provenance = config["_provenance"]
    assert provenance["derived_from"].endswith("Tossing3D-o1.json")
    # The exact commit the derivation was taken against, which has to stay the commit
    # `pyproject.toml` pins the simulator to.
    pyproject = (Path(__file__).parents[3] / "pyproject.toml").read_text()
    assert provenance["upstream_commit"] in pyproject

    # The one thing this config changes...
    assert config["regions"]["bin_init_region"]["ranges"] == [[2.0, -0.0005, 2.001, 0.0005]]
    # ...and the things it deliberately does not. Moving the goal region to the bin was
    # the alternative; leaving it exactly as upstream wrote it is what makes this a
    # revert of a bin move rather than a redefinition of the task.
    assert config["regions"]["blocks_goal_region"]["ranges"] == [
        [1.90, -0.10, 0.0, 2.10, 0.10, 0.10]
    ]
    assert config["goal_state"] == [["on", "cube_0", "blocks_goal_region"]]


@kinder_available
def test_the_shipped_config_changes_exactly_one_region_of_upstreams_o1() -> None:
    """The file is a derivative and has to stay a *minimal* one, checked against the
    installed upstream rather than against a copy that could drift from it. An upstream
    bump that edits o1 fails here instead of silently widening this diff."""
    ours = json.loads(KinderBackend.COINCIDENT_BIN_TASK_CONFIG.read_text())
    provenance = ours.pop("_provenance")
    upstream = upstream_task_config(name="Tossing3D-o1")

    differing = [key for key in set(ours) | set(upstream) if ours.get(key) != upstream.get(key)]
    assert differing == ["regions"], f"expected only `regions` to differ, got {differing}"

    differing_regions = [
        name
        for name in set(ours["regions"]) | set(upstream["regions"])
        if ours["regions"].get(name) != upstream["regions"].get(name)
    ]
    assert differing_regions == ["bin_init_region"], differing_regions
    assert provenance["upstream_commit"]


@kinder_available
def test_the_bin_is_placed_where_upstreams_own_o2_puts_it() -> None:
    """The justification for this file rests on it reproducing a placement upstream
    itself still ships, rather than one invented here. If that stops being true the file
    loses its warrant, so it is asserted rather than only claimed in prose."""
    ours = json.loads(KinderBackend.COINCIDENT_BIN_TASK_CONFIG.read_text())
    o1 = upstream_task_config(name="Tossing3D-o1")
    o2 = upstream_task_config(name="Tossing3D-o2")

    assert ours["regions"]["bin_init_region"] == o2["regions"]["bin_init_region"]
    # ...and o2's goal region is byte-identical to o1's, which is *why* moving the bin
    # rather than the region is the change that makes the two coincide: upstream already
    # ships this exact goal region paired with this exact bin placement.
    assert o2["regions"]["blocks_goal_region"] == o1["regions"]["blocks_goal_region"]


# --------------------------------------------------------------------------------------
# The coincidence property -- against the live simulator.
# --------------------------------------------------------------------------------------

_ENVS: dict[bool, Tossing3DEnvironment] = {}


def shared_env(*, coincident: bool) -> Tossing3DEnvironment:
    """One simulator per configuration for the whole module. Opening one compiles a
    MuJoCo model and connects a PyBullet client, so they are built once and reused; a
    dict rather than a pytest fixture because this project's lint bans positional
    parameters outright (ruff PLR0917) and a fixture argument is one."""
    if coincident not in _ENVS:
        _ENVS[coincident] = Tossing3DEnvironment(coincident_bin_goal=coincident)
    return _ENVS[coincident]


def bin_footprint(*, env: Tossing3DEnvironment) -> tuple[np.ndarray, np.ndarray]:
    """The bin's real axis-aligned extent, unioned over its actual MuJoCo geoms.

    Read from the compiled model rather than from the task JSON's `length`/`width`/
    `wall_thickness`, so this measures the bin the simulator built -- the same standard
    `goal_region_bounds()` applies to the goal box.
    """
    import mujoco  # noqa: PLC0415 (optional dependency, ships with kindergarden)

    unwrapped = env.backend()._ensure_env().unwrapped._object_centric_env
    sim = unwrapped._robot_env.sim
    model = getattr(sim.model, "mj_model", sim.model)
    data = getattr(sim.data, "mj_data", sim.data)

    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    found = 0
    for geom in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom]) or ""
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if "bin_0" not in body and "bin_0" not in name:
            continue
        found += 1
        low = np.minimum(low, data.geom_xpos[geom] - model.geom_size[geom])
        high = np.maximum(high, data.geom_xpos[geom] + model.geom_size[geom])
    assert found, "no MuJoCo geom belongs to bin_0 -- the bin's naming has changed"
    return low, high


def in_goal_region(*, env: Tossing3DEnvironment, position: tuple[float, float, float]) -> bool:
    """KINDER's own containment test, which is what `_check_goals` delegates to."""
    unwrapped = env.backend()._ensure_env().unwrapped._object_centric_env
    return bool(
        unwrapped._ground_fixture.check_in_region(
            np.array(position, dtype=np.float32), "blocks_goal_region", unwrapped._robot_env
        )
    )


@kinder_available
def test_stock_o1_is_untouched_by_shipping_the_variant() -> None:
    """Every number measured on this domain was measured against stock o1, so the
    default construction has to keep producing exactly the scene those numbers came
    from. This is the guard on that, and it is the reason the variant is opt-in."""
    env = shared_env(coincident=False)
    assert env.coincident_bin_goal is False
    assert env.backend().task_config_path() is None
    assert env.goal_region_bounds() == pytest.approx(GOAL_REGION)

    low, high = bin_footprint(env=env)
    centre_x = float((low[0] + high[0]) / 2)
    assert centre_x == pytest.approx(BIN_X, abs=COINCIDENCE_TOL)
    # The stock mismatch itself: the bin starts 23 cm past where the scored region does.
    assert low[0] > GOAL_REGION[0] + 0.2


@kinder_available
def test_the_variant_makes_the_goal_box_and_the_bin_footprint_coincide() -> None:
    """The property this config exists for, against the live sim on both sides.

    The goal box comes from `Region.bbox` -- already inflated by
    `ground_placement_threshold` -- and the bin's extent from its compiled geoms. Neither
    is a JSON literal, which matters: the mismatch this fixes is invisible to any test
    that reads the raw range.
    """
    env = shared_env(coincident=True)
    goal = env.goal_region_bounds()
    low, high = bin_footprint(env=env)

    # The goal region is *unchanged* from stock -- only the bin moved.
    assert goal == pytest.approx(GOAL_REGION)

    for axis, name in ((0, "x"), (1, "y")):
        assert low[axis] == pytest.approx(goal[axis], abs=COINCIDENCE_TOL), (
            f"{name}_min: bin {low[axis]:.4f} vs goal {goal[axis]:.4f}"
        )
        assert high[axis] == pytest.approx(goal[axis + 3], abs=COINCIDENCE_TOL), (
            f"{name}_max: bin {high[axis]:.4f} vs goal {goal[axis + 3]:.4f}"
        )


@kinder_available
def test_the_variant_scores_a_cube_resting_in_the_bin_as_being_in_the_goal_region() -> None:
    """Coinciding boxes would be an empty win if the height a cube actually rests at
    inside the bin fell outside the region's z range, so that is checked at the position
    a cube really occupies rather than at the footprint's centre in the abstract."""
    env = shared_env(coincident=True)
    low, high = bin_footprint(env=env)
    assert low[2] <= CUBE_REST_Z_IN_BIN <= high[2], "the bin's z extent has moved"

    centre = (float((low[0] + high[0]) / 2), float((low[1] + high[1]) / 2), CUBE_REST_Z_IN_BIN)
    assert in_goal_region(env=env, position=centre), (
        f"a cube resting in the middle of the bin at {centre} is not in the goal region"
    )

    # Not just the centre: the bin's four interior corners, inset by the cube's own
    # half-extent, are all scored too -- i.e. the *whole* reachable interior counts.
    inset = 0.02 + 0.025  # wall thickness plus the cube's half-extent
    for x in (low[0] + inset, high[0] - inset):
        for y in (low[1] + inset, high[1] - inset):
            assert in_goal_region(env=env, position=(float(x), float(y), CUBE_REST_Z_IN_BIN)), (
                f"a cube resting in the bin's corner at {(x, y)} is not in the goal region"
            )


@kinder_available
def test_the_coincidence_test_is_not_vacuous_at_the_bins_near_wall() -> None:
    """Guards the tests above against passing on easy points only.

    Each scene is probed at *its own* bin's near wall, which is the line that separates
    "landed in the bin" from "landed short of it":

    * Under the variant that wall is also the scored region's near edge, so a band
      straddling it must **split** -- the classifier flips exactly at the bin. That is
      what makes "the boxes coincide" a claim about a boundary rather than an interior.
    * Under stock o1 the same probe must **not** split: the bin's near wall sits 23 cm
      inside the region, so both sides score alike and the wall is invisible to the goal
      check. That is the defect this config exists to remove, pinned as a fact.
    """
    stock, variant = shared_env(coincident=False), shared_env(coincident=True)
    offsets = (-0.03, -0.01, 0.01, 0.03)

    variant_wall = float(bin_footprint(env=variant)[0][0])
    variant_band = [(variant_wall + d, 0.0, CUBE_REST_Z_IN_BIN) for d in offsets]
    variant_verdicts = [in_goal_region(env=variant, position=p) for p in variant_band]
    assert variant_verdicts == [False, False, True, True], (
        f"the variant's bin wall is not the region's edge: "
        f"{list(zip(variant_band, variant_verdicts, strict=True))}"
    )

    stock_wall = float(bin_footprint(env=stock)[0][0])
    stock_band = [(stock_wall + d, 0.0, CUBE_REST_Z_IN_BIN) for d in offsets]
    stock_verdicts = [in_goal_region(env=stock, position=p) for p in stock_band]
    assert stock_verdicts == [True, True, True, True], (
        f"stock o1's bin wall has become visible to the goal check: "
        f"{list(zip(stock_band, stock_verdicts, strict=True))}"
    )
    # The two probes really are at different places -- i.e. the contrast above is between
    # two scenes, not an artefact of probing the same line twice.
    assert stock_wall - variant_wall > 0.2


@kinder_available
def test_a_toss_that_lands_in_the_variants_bin_passes_kinders_own_goal_check() -> None:
    """The empirical claim, end to end through upstream's `_check_goals()` rather than
    through any arithmetic of ours: under this config, landing in the bin is a *pass*.

    Under stock o1 the opposite holds -- a cube in the bin rests at x ~ 2.22, past the
    region's 2.15 edge -- which is what makes the demo read as a miss while the scored
    outcome is a pass on the open floor short of it.

    The swings below bracket the step in the landing point: 0.75 lands the cube on the
    floor short of the bin, and >= 0.96 lands it inside. Both sides are kept so that a
    physics change which stopped any swing reaching the bin fails loudly on the "proves
    nothing" assertion rather than passing vacuously.
    """
    env = shared_env(coincident=True)
    backend = env.backend()
    low, high = bin_footprint(env=env)

    landed_in_bin = []
    for swing in (0.75, 0.96, 1.0, 1.25):
        env.reset_to_seed(seed=0)
        for skill, param in (
            (_ENV.SKILL_PICK, SkillOraclePolicy.ORACLE_PICK_DISTANCE),
            (_ENV.SKILL_MOVE_TO_THROW_POSE, 0.0),
            (_ENV.SKILL_TOSS, swing),
        ):
            state = env.take_action(action=np.array([float(skill), float(param), 0.0]))
        x = state.get(obj=_ENV.cube, feature_name="x")
        z = state.get(obj=_ENV.cube, feature_name="z")
        # "In the bin" means resting above the floor inside the footprint, not merely
        # having an x that happens to fall in range.
        if low[0] <= x <= high[0] and z > 0.03:
            landed_in_bin.append((swing, x, z, backend.check_goals()))

    assert landed_in_bin, "no swing put the cube in the bin -- this test proves nothing"
    misscored = [entry for entry in landed_in_bin if not entry[3]]
    assert not misscored, f"a cube resting in the bin was scored a failure: {misscored}"
