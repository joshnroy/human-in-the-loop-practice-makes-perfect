"""PickCubeFromShelf from the skill-benchmark states it failed on (kindergarden 0dac89e,
kinder-models 427ad6c), restored exactly: a grasp whose open jaws straddle a side
panel, an approach that stalled short and closed the gripper on air, and retries that
re-ran the candidate that had just failed."""

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.backend import WIDE_BOX_TASK_ID
from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import (
    JAW_HALF_WIDTH,
    SHELF_PICK_NEIGHBOUR_CLEARANCE,
    SHELF_PICK_SAME_APPROACH,
    KinderSkills,
    PadSweep,
)
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import CUBE_STANDOFF, SkillFailed
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import StateCapture, StateSet
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.types import BenchmarkCase

STATES = Path(__file__).resolve().parent / "data/wide_box_benchmark_states.jsonl"
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
    env = SweepOffShelf3DEnvironment(scene_bg=True, task_id=WIDE_BOX_TASK_ID)
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
        "neighbour_in_sweep": lambda **_: None,
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


def test_a_grasp_whose_pads_sweep_a_neighbour_is_refused_not_tried_last(*, monkeypatch):
    """Grasp 1's pads would sweep another cube as they come down or close, and grasp
    2's hand touches one at the grasp: on `main` both were offered after every other
    grasp, and in a cluster they were the only grasps left, so the pick shoved the
    neighbour. Now neither is offered at all."""
    kinder = _stub_candidates(monkeypatch=monkeypatch, clear={0: {0}, 1: {0}, 2: {0}}, bases=1)
    stubs = {
        "shelf_grasp_conf": lambda *, hand, **_: np.full(7, float(hand[3, 0])),
        "neighbour_in_sweep": lambda *, path, **_: "cube_1" if path[-1][3, 0] == 1 else None,
        "touches_other_cube": lambda *, arm, **_: bool(arm[0] == 2),
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(kinder, name, stub, raising=False)
    refused: list[str] = []
    candidates = kinder.shelf_candidates(
        cube="cube_0", centre=np.zeros(3), tried=[], refused=refused
    )
    offered = [int(hand[3, 0]) for hand, *_ in candidates]
    assert offered == [0], offered
    assert "cube_1" in refused, refused


NEIGHBOUR_STATES = Path(__file__).resolve().parent / "fixtures/neighbour_strike_states.jsonl"
# The four executed PickCubeFromShelf failures in the new scene (867abd9e6, kindergarden
# 490831fa5) where the closing pads struck a neighbour 3-3.5 cm away, rebuilt in this
# scene from their cube poses: each strikes again here before the fix.
NEIGHBOUR_STRIKES = [
    "neighbour-layout-49",
    "neighbour-start-s11",
    "neighbour-wiper-container-1",
    "neighbour-harvest-PickCubeFromShelf-1",
]
# How far (m) a cube may move during a pick that does not touch it: settling only.
UNTOUCHED = 0.005


@pytest.fixture(name="neighbour_states", scope="module")
def _neighbour_states():
    return {state.state_id: state for state in StateSet.load(path=NEIGHBOUR_STATES)}


@pytest.mark.parametrize("state_id", NEIGHBOUR_STRIKES)
def test_a_shelf_pick_never_shoves_a_neighbouring_cube(*, env, neighbour_states, state_id):
    """Before: the pads struck the neighbour on the way down or as they closed --
    a miss with both cubes shoved 2-6 cm, or the neighbour carried off in the jaws
    beside the target. After: the target alone is picked, or the pick refuses
    without having touched anything."""
    state = neighbour_states[state_id]
    (case,) = state.cases
    cube = case.cube
    kinder = _restore(env=env, states=neighbour_states, state_id=state_id)
    shelf = kinder.shelf
    names = [f"cube_{i}" for i in range(5)]
    before = {name: shelf.cube_position(cube=name).copy() for name in names}
    atoms = _pick(env=env, cube=cube)
    moved = {
        name: float(np.linalg.norm(shelf.cube_position(cube=name) - before[name])) for name in names
    }
    held = ("HoldingCube", ("robot", cube)) in atoms
    others = {name: round(1000 * d, 1) for name, d in moved.items() if name != cube}
    assert max(others.values()) < 1000 * UNTOUCHED, (others, env.last_skill_error())
    if not held:
        assert moved[cube] < UNTOUCHED, (moved, env.last_skill_error())
        assert "no grasp" in (env.last_skill_error() or ""), env.last_skill_error()


# The live 2F-85's pads in the pinch-site frame (m), each 8 mm thick and 22 mm wide,
# at four stages of closing: open, a quarter, half, and three quarters of the way,
# where their inner faces near the 2 cm cube's. They come in on an arc, riding forward.
_STAGES_MM = [((42.7, 50.7), (-8.1, 29.4)), ((33.2, 41.4), (-1.8, 35.7))]
_STAGES_MM += [((22.6, 31.0), (2.6, 40.2)), ((11.4, 20.0), (4.9, 42.5))]
TRAVEL = [
    [
        (np.array([x0, -11.0, z0]) / 1000, np.array([x1, 11.0, z1]) / 1000),
        (np.array([-x1, -11.0, z0]) / 1000, np.array([-x0, 11.0, z1]) / 1000),
    ]
    for (x0, x1), (z0, z1) in _STAGES_MM
]


def _neighbour_at(*, hand: np.ndarray, local_mm: tuple[float, float, float]) -> dict:
    """A 2 cm cube aligned with the hand, centred at `local_mm` in the hand's frame."""
    centre = (hand @ np.r_[np.asarray(local_mm) / 1000.0, 1.0])[:3]
    return {"cube_1": (centre, hand[:3, :3], np.full(3, O.cube_half_size))}


def _straight_down_grasp() -> np.ndarray:
    hand = np.eye(4)
    hand[:3, :3] = np.diag([1.0, -1.0, -1.0])  # closing along x, approach straight down
    hand[:3, 3] = [1.5, 0.0, 0.32]
    return hand


@pytest.mark.parametrize(
    ("local_mm", "struck"),
    [
        # The layout-49 neighbour: 30 mm along the closing axis, at the cube's depth.
        ((30.0, 0.0, 1000 * CUBE_STANDOFF), True),
        # Beyond the open pad's outer face.
        ((65.0, 0.0, 1000 * CUBE_STANDOFF), False),
        # The same 30 mm, but across the pads' width: 9 mm from the pad's edge.
        ((0.0, 30.0, 1000 * CUBE_STANDOFF), False),
        # Beyond where the closing pads' tips reach.
        ((30.0, 0.0, 85.0), False),
        # Just outside the open pad and ahead of it: the pads ride forward only as they
        # come in, so they pass it -- inside one box bounding open and closed pads.
        ((55.0, 0.0, 46.0), False),
    ],
)
def test_the_closing_sweep_reaches_a_neighbour_along_the_closing_axis(*, local_mm, struck):
    hand = _straight_down_grasp()
    hit = PadSweep.first_hit(
        sweep=PadSweep.closing(hand=hand, travel=TRAVEL),
        others=_neighbour_at(hand=hand, local_mm=local_mm),
        clearance=SHELF_PICK_NEIGHBOUR_CLEARANCE,
    )
    assert (hit == "cube_1") is struck, (local_mm, hit)


def test_the_descent_sweep_reaches_a_neighbour_the_closing_pads_do_not():
    """The start-s11 strike: the right pad met the neighbour on the way down, above
    where it closes. A neighbour in line with the open pad, 50 mm back along the
    approach, is clear of the close but in the way of a 60 mm descent."""
    hand = _straight_down_grasp()
    pre = hand.copy()
    pre[:3, 3] -= 0.06 * hand[:3, 2]
    others = _neighbour_at(hand=hand, local_mm=(46.6, 0.0, -50.0))
    clearance = SHELF_PICK_NEIGHBOUR_CLEARANCE
    closing = PadSweep.closing(hand=hand, travel=TRAVEL)
    descent = PadSweep.descent(path=[pre, hand], pads=TRAVEL[0])
    assert PadSweep.first_hit(sweep=closing, others=others, clearance=clearance) is None
    assert PadSweep.first_hit(sweep=descent, others=others, clearance=clearance) == "cube_1"


def test_a_grasp_the_sweep_refuses_is_offered_shifted_off_centre_after_every_centred_one(
    *, monkeypatch
):
    """Grasp 1's pads sweep a neighbour unless the hand is shifted at least 8 mm, along
    its closing axis or the pads' width; grasp 2 is clear of every neighbour but
    unreachable. After every centred grasp, grasp 1 is offered shifted, the smallest
    shift that clears first; grasp 0, offered centred, and grasp 2, which the sweep
    never refused, get no shifted copies."""
    kinder = _stub_candidates(monkeypatch=monkeypatch, clear={0: {0}, 1: {0}, 2: set()}, bases=1)

    def sweep(*, path, **_):
        hand = path[-1]
        swept = int(hand[3, 0]) == 1 and float(np.linalg.norm(hand[:3, 3])) < 0.0079
        return "cube_1" if swept else None

    monkeypatch.setattr(kinder, "neighbour_in_sweep", sweep, raising=False)
    hands = {int(hand[3, 0]): hand for hand, _ in kinder.shelf_hands()}
    offered = [
        (int(hand[3, 0]), hand[:3, 3].copy())
        for hand, *_ in kinder.shelf_candidates(cube="cube_0", centre=np.zeros(3), tried=[])
    ]
    assert [g for g, _ in offered[:2]] == [0, 1], offered
    assert np.allclose(offered[0][1], 0.0), offered
    assert all(g == 1 for g, _ in offered[1:]), offered
    shifts = [float(np.linalg.norm(shift)) for _, shift in offered[1:]]
    assert shifts[0] == pytest.approx(0.008), shifts
    assert shifts == sorted(shifts), shifts
    axes = hands[1][:3, :2]
    for _, shift in offered[1:]:
        along = axes.T @ shift
        assert np.count_nonzero(np.abs(along) > 1e-9) == 1, along
        assert np.allclose(axes @ along, shift), (along, shift)


# PickCubeFromShelf cases `main` lost to the neighbour sweep (#539), in this scene too:
# no grasp's pads clear the neighbour by SHELF_PICK_NEIGHBOUR_CLEARANCE (layout-29,
# layout-35), or only grasps no base reaches do (layout-0, layout-41). Each misses by
# 0.3-8 mm with the 2F-85 fully open, its inner pad faces 42.7 mm from the centre.
NEIGHBOUR_TIGHT = [
    ("layout-29", "cube_4"),
    ("layout-35", "cube_0"),
    ("layout-0", "cube_0"),
    ("layout-41", "cube_1"),
]


@pytest.mark.parametrize(("state_id", "cube"), NEIGHBOUR_TIGHT)
def test_a_cube_with_a_tight_neighbour_has_a_grasp_whose_pads_clear_it(
    *, env, states, state_id, cube
):
    """Before: `shelf_candidates` offered nothing and the pick refused. After: an
    off-centre grasp, its whole way in and its close still clear of every other cube."""
    kinder = _restore(env=env, states=states, state_id=state_id)
    centre = kinder.shelf.cube_position(cube=cube)
    hand, entry, *_ = next(
        iter(kinder.shelf_candidates(cube=cube, centre=centre, tried=[])), (None, ())
    )
    assert hand is not None, f"{state_id}/{cube}: no grasp offered"
    assert kinder.neighbour_in_sweep(path=[*entry, hand], cube=cube) is None


# The lift (`lifter`) backs the held cube out along the leaned approach and, on these
# two, shoves the neighbour the way in cleared: 10.7 mm on layout-0, 54 mm on
# layout-35. The grasp is not the cause; a lift that clears the neighbours is a
# separate change to how the pick executes.
_LIFT_SHOVES = pytest.mark.xfail(strict=True, reason="the lift shoves the neighbour")
# The same on layout-29 in the edge-home scene, whose regenerated layout is not the one
# the pick was measured on: the lift moves cube_1 5.3 mm against UNTOUCHED's 5 mm.
_LIFT_SHOVES_EDGE_HOME = pytest.mark.xfail(
    strict=True,
    reason="edge-home scene: the lift moves cube_1 5.3 mm against the 5 mm UNTOUCHED "
    "limit; awaits #556's lift-clear follow-up",
)


@pytest.mark.parametrize(
    ("state_id", "cube"),
    [
        pytest.param(*NEIGHBOUR_TIGHT[0], marks=_LIFT_SHOVES_EDGE_HOME),
        pytest.param(*NEIGHBOUR_TIGHT[1], marks=_LIFT_SHOVES),
        pytest.param(*NEIGHBOUR_TIGHT[2], marks=_LIFT_SHOVES),
    ],
)
def test_a_cube_with_a_tight_neighbour_is_picked_without_moving_it(*, env, states, state_id, cube):
    """Before: refused ("no grasp ... planned; refused grasps sweeping ..."). After:
    the target is held and every other cube is where it was."""
    kinder = _restore(env=env, states=states, state_id=state_id)
    shelf = kinder.shelf
    names = [f"cube_{i}" for i in range(5)]
    before = {name: shelf.cube_position(cube=name).copy() for name in names}
    atoms = _pick(env=env, cube=cube)
    assert ("HoldingCube", ("robot", cube)) in atoms, env.last_skill_error()
    moved = {
        name: round(1000 * float(np.linalg.norm(shelf.cube_position(cube=name) - before[name])), 1)
        for name in names
        if name != cube
    }
    assert max(moved.values()) < 1000 * UNTOUCHED, moved
