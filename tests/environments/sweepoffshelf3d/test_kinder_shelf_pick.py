"""PickCubeFromShelf from the skill-benchmark states it failed on (kindergarden 0dac89e,
kinder-models 427ad6c), restored exactly: a grasp whose open jaws straddle a side
panel, an approach that stalled short and closed the gripper on air, and retries that
re-ran the candidate that had just failed."""

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import (
    JAW_HALF_WIDTH,
    SHELF_PICK_SAME_APPROACH,
    KinderSkills,
)
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import CUBE_STANDOFF, SkillFailed
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import StateCapture, StateSet
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.types import BenchmarkCase

STATES = Path(__file__).resolve().parents[3] / "benchmarks/sweepoffshelf3d/states.jsonl"
SIDE_PANELS = ("cupboard_1_left_panel", "cupboard_1_right_panel")
# (state, cube) of the benchmark's three execution failures.
BESIDE_PANEL = [("shelf-shelf-left-front", "cube_0"), ("shelf-shelf-right-front", "cube_2")]
STALLED_APPROACH = ("layout-56", "cube_1")
# One clear grasp, whose first plans the MuJoCo re-check refused.
FEW_CLEAR = ("layout-44", "cube_4")
# A cube in a back corner of the shelf, 6 mm from two panels.
BACK_CORNER = [("shelf-shelf-left-back", "cube_1"), ("shelf-shelf-right-back", "cube_3")]
# One clear grasp with a pre-grasp from one base only, whose plans the re-check refused.
ONE_ENTRY = ("layout-5", "cube_0")

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("sweepoffshelf3d_test_kinder_shelf_pick"),
]


@pytest.fixture(name="env", scope="module")
def _env():
    env = SweepOffShelf3DEnvironment(scene_bg=True)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture(name="states", scope="module")
def _states():
    return {state.state_id: state for state in StateSet.load(path=STATES)}


def _restore(*, env, states, state_id) -> KinderSkills:
    StateCapture.restore(env=env, state=states[state_id])
    return env.controllers()._kinder  # noqa: SLF001


def _pick(*, env, cube) -> set:
    case = BenchmarkCase(skill="PickCubeFromShelf", cube=cube)
    env.take_action(action=StateCapture.action(case=case))
    return set(env.get_current_state().abstract_atoms)


def _side_panel_between_pads(*, kinder: KinderSkills, hand: np.ndarray) -> str | None:
    """A side panel inside the space between the open pads at grasp `hand`, from the
    pads' own depth down to the cube's centre: where the jaws would close on it."""
    shelf = kinder.shelf
    m, d = shelf.model, shelf.data
    here = np.linalg.inv(shelf.site_pose())
    pads = [
        np.mean([(here @ np.r_[d.geom_xpos[g], 1.0])[:3] for g in side], axis=0)
        for side in shelf.pad_sides
    ]
    half_gap = 0.5 * abs(float(pads[0][0] - pads[1][0]))
    grid = np.stack(
        np.meshgrid(
            np.linspace(-half_gap, half_gap, 41),
            np.linspace(-JAW_HALF_WIDTH, JAW_HALF_WIDTH, 9),
            np.linspace(0.0, CUBE_STANDOFF, 15),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    world = (hand[:3, :3] @ grid.T).T + hand[:3, 3]
    for name in SIDE_PANELS:
        g = m.geom(name).id
        local = (d.geom_xmat[g].reshape(3, 3).T @ (world - d.geom_xpos[g]).T).T
        if np.all(np.abs(local) <= m.geom_size[g], axis=1).any():
            return name
    return None


@pytest.mark.parametrize(("state_id", "cube"), BESIDE_PANEL)
def test_no_grasp_offered_beside_a_side_panel_closes_the_jaws_on_it(*, env, states, state_id, cube):
    """The open outer pad passed outside the 1 cm panel, so the close clamped the
    panel and the lift stalled on it, every attempt."""
    kinder = _restore(env=env, states=states, state_id=state_id)
    centre = kinder.shelf.cube_position(cube=cube)
    candidates = kinder.shelf_candidates(cube=cube, centre=centre, tried=[])
    for offered, (hand, *_) in enumerate(candidates):
        assert _side_panel_between_pads(kinder=kinder, hand=hand) is None, np.round(hand, 3)
        if offered >= 7:
            break


@pytest.mark.parametrize(("state_id", "cube"), BESIDE_PANEL)
def test_pick_cube_from_shelf_takes_a_cube_beside_a_side_panel(*, env, states, state_id, cube):
    _restore(env=env, states=states, state_id=state_id)
    atoms = _pick(env=env, cube=cube)
    assert ("HoldingCube", ("robot", cube)) in atoms, env.last_skill_error()


def test_pick_cube_from_shelf_tries_the_bases_beyond_a_run_of_unreachable_ones(*, env, states):
    """layout-44: the one clear grasp's plans from the first base that has a
    pre-grasp were refused (pre-grasp planning failed; the bracelet hit the board
    above), and the three bases after it were unreachable, so the grasp was given up
    with twelve bases left that have pre-grasps of their own."""
    state_id, cube = FEW_CLEAR
    _restore(env=env, states=states, state_id=state_id)
    atoms = _pick(env=env, cube=cube)
    assert ("HoldingCube", ("robot", cube)) in atoms, env.last_skill_error()


def test_a_cube_by_a_side_panel_has_a_way_in_from_one_base_only(*, env, states):
    """layout-5, still not picked: of every drivable base, its one clear grasp has a
    pre-grasp from base 0 alone, so more bases cannot help it; both plans from there
    are refused by the MuJoCo re-check, and so were BiRRT seeds 1-6 (2026-10-19)."""
    state_id, cube = ONE_ENTRY
    kinder = _restore(env=env, states=states, state_id=state_id)
    centre = kinder.shelf.cube_position(cube=cube)
    bases = [
        b
        for b in kinder.shelf_pick_bases(centre=centre)
        if kinder.base_clear(base=b) and kinder.base_drivable(base=b)
    ]
    clear = [
        (hand, pres)
        for hand, pres in kinder.shelf_hands(cube=cube, centre=centre)
        if kinder.shelf_grasp_conf(hand=hand, base=bases[0], cube=cube) is not None
    ]
    assert len(clear) == 1
    (hand, pres), entered = clear[0], []
    for i, base in enumerate(bases):
        grasp_conf = kinder.shelf_grasp_conf(hand=hand, base=base, cube=cube)
        if grasp_conf is not None and any(
            kinder.shelf_pre_conf(pre=pre, base=base, grasp_conf=grasp_conf, cube=cube) is not None
            for pre in pres
        ):
            entered.append(i)
    assert entered == [0], entered


@pytest.mark.parametrize(("state_id", "cube"), BACK_CORNER)
def test_every_grasp_on_a_cube_in_a_back_corner_closes_the_jaws_on_a_panel(
    *, env, states, state_id, cube
):
    """A cube 6 mm from the back and a side panel: the pads close across a face
    normal, so one pair straddles the side panel and the other the back panel. No base
    or lean changes that; the candidate family has no grasp here."""
    kinder = _restore(env=env, states=states, state_id=state_id)
    centre = kinder.shelf.cube_position(cube=cube)
    hands = kinder.shelf_hands(cube=cube, centre=centre)
    assert hands
    assert all(kinder.cupboard_between_pads(hand=hand, cube=cube) for hand, _ in hands)


class _CloseLog:
    """A recorder: the cube's distance from the pinch site whenever the gripper
    command goes from open to closed."""

    def __init__(self, *, cube: str) -> None:
        self.cube = cube
        self.last_grip = 0.0
        self.closes: list[tuple[str, float]] = []

    def on_step(self, *, skills: Any, label: str) -> None:
        if skills.grip >= 0.5 > self.last_grip:
            gap = skills.cube_position(cube=self.cube) - skills.site_pose()[:3, 3]
            self.closes.append((label, float(np.linalg.norm(gap))))
        self.last_grip = skills.grip

    def mark(self, *, skills: Any, label: str) -> None:
        pass


def test_a_shelf_pick_whose_approach_stalls_never_closes_on_air(*, env, states):
    """layout-56: the straight-line approach stopped 17 cm short of the cube (wrist on
    the right panel, forearm on the robot's base); the failure was swallowed and the
    gripper closed on nothing."""
    state_id, cube = STALLED_APPROACH
    kinder = _restore(env=env, states=states, state_id=state_id)
    log = _CloseLog(cube=cube)
    kinder.shelf.recorder = log
    try:
        _pick(env=env, cube=cube)
    finally:
        kinder.shelf.recorder = None
    closes = [distance for label, distance in log.closes if label == "pick_cube_from_shelf"]
    # The pads close CUBE_STANDOFF (3.5 cm) ahead of the site, the cube between them.
    assert all(distance < CUBE_STANDOFF + 0.015 for distance in closes), log.closes


def test_a_shelf_pick_never_executes_the_same_approach_twice(*, env, states, monkeypatch):
    """layout-56 on `main`: three executions, one approach, three identical failures."""
    state_id, cube = STALLED_APPROACH
    _restore(env=env, states=states, state_id=state_id)
    made: list[np.ndarray] = []
    executed: list[np.ndarray] = []
    settler, drive = KinderSkills.settler, KinderSkills.drive

    def recording_settler(self: KinderSkills, **kwargs: Any) -> Any:  # noqa: PLR0917
        made.append(kwargs["hand"][:3, 2].copy())
        return settler(self, **kwargs)

    def recording_drive(self: KinderSkills, **kwargs: Any) -> str:  # noqa: PLR0917
        try:
            outcome = drive(self, **kwargs)
        except SkillFailed:
            executed.append(made[-1])
            raise
        if outcome == "done":
            executed.append(made[-1])
        return outcome

    monkeypatch.setattr(KinderSkills, "settler", recording_settler)
    monkeypatch.setattr(KinderSkills, "drive", recording_drive)
    _pick(env=env, cube=cube)
    assert executed
    for i, a in enumerate(executed):
        for b in executed[i + 1 :]:
            assert float(a @ b) < SHELF_PICK_SAME_APPROACH, (np.round(a, 3), np.round(b, 3))


class _Controller:
    """Stands in for the adapted `PickShelfController`; `drive` is stubbed."""

    def __init__(self, **_: Any) -> None:
        pass


class _Shelf:
    @staticmethod
    def cube_position(**_: Any) -> np.ndarray:
        return np.zeros(3)


def test_a_candidate_whose_execution_raises_is_still_recorded_as_tried(*, monkeypatch):
    """`tried` grew only when `drive` returned "done", so a candidate whose execution
    raised was offered again by the next attempt."""
    from hitl_pmp.environments.sweepoffshelf3d import kinder_skills

    kinder = object.__new__(KinderSkills)
    kinder.shelf = _Shelf()  # type: ignore[assignment]
    hand = np.eye(4)
    hand[:3, 2] = [0.5, 0.0, -np.sqrt(0.75)]
    pose, conf = np.eye(4), np.zeros(7)
    candidate = (hand, (pose,), np.zeros(3), conf, conf)

    def raises(**_: Any) -> str:
        raise SkillFailed("pick_cube_from_shelf: execution failed: the lift stopped")

    stubs = {
        "objects": lambda *names: [None] * len(names),
        "cube_pose": lambda **_: np.eye(4),
        "shelf_candidates": lambda **_: iter([candidate]),
        "shelf_exit": lambda **_: (pose, None),
        "pose_in": lambda **_: None,
        "fixed_ik": lambda **_: None,
        "planning_sim": lambda: None,
        "settler": lambda **_: None,
        "lifter": lambda **_: None,
        "facing_params": lambda **_: (0.6, 0.0),
        "drive": raises,
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(kinder, name, stub, raising=False)
    monkeypatch.setattr(
        kinder_skills.KinderControllers, "get", staticmethod(lambda **_: _Controller)
    )
    tried: list[np.ndarray] = []
    with pytest.raises(SkillFailed):
        kinder.shelf_pick_attempt(cube="cube_0", tried=tried)
    assert len(tried) == 1, tried
    assert np.allclose(tried[0], hand[:3, 2])


def _stub_candidates(*, monkeypatch, clear: dict[int, set[int]], bases: int) -> KinderSkills:
    """A `KinderSkills` whose grasp `h` (tagged in its bottom row) is clear from
    exactly the bases in `clear[h]`, each with one pre-grasp, and nothing crowded."""
    kinder = object.__new__(KinderSkills)
    hands = []
    for h in range(len(clear)):
        hand = np.eye(4)
        angle = np.deg2rad(30.0 * h)
        hand[:3, 2] = [np.cos(angle), np.sin(angle), 0.0]
        hand[3, 0] = h
        hands.append((hand, (hand,)))
    stubs = {
        "shelf_pick_bases": lambda **_: [np.array([0.0, float(b), 0.0]) for b in range(bases)],
        "base_clear": lambda **_: True,
        "base_drivable": lambda **_: True,
        "shelf_hands": lambda **_: hands,
        "cupboard_between_pads": lambda **_: None,
        "cube_between_pads": lambda **_: None,
        "touches_other_cube": lambda **_: False,
        "shelf_grasp_conf": lambda *, hand, base, **_: (
            np.zeros(7) if int(base[1]) in clear[int(hand[3, 0])] else None
        ),
        "shelf_pre_conf": lambda **_: np.zeros(7),
        "shelf_exit": lambda **_: (None, None),
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(kinder, name, stub, raising=False)
    return kinder


def test_a_grasp_given_up_after_unreachable_bases_is_offered_its_remaining_bases_last(
    *, monkeypatch
):
    """Grasp 0 is clear from base 0, unreachable from bases 1-3 and clear again from
    base 5: the first pass gives it up after base 3 and goes on to grasp 1 (clear from
    base 0); only once the first pass is exhausted does grasp 0 get bases 4 onward.
    Grasp 2, clear from none of the bases the first pass tried, is not searched again:
    that a grasp's clearance barely depends on the base still holds for it."""
    kinder = _stub_candidates(monkeypatch=monkeypatch, clear={0: {0, 5}, 1: {0}, 2: {6}}, bases=8)
    candidates = kinder.shelf_candidates(cube="cube_0", centre=np.zeros(3), tried=[])
    offered = [(int(hand[3, 0]), int(base[1])) for hand, _, base, *_ in candidates]
    assert offered == [(0, 0), (1, 0), (0, 5)], offered
