"""RobotBatchReturn: the robot puts every dumpling back on its start slot itself.

The robot counterpart of the person's batch return (Josh, 2026-11-13), opt-in by
`RackServingConfig.robot_batch_return`. One skill rather than a plan of single-dumpling
skills: a per-dumpling plan is up to ten actions, the whole of a planner's horizon.

What it does, in order:

1. A scraper in the hand is put home (ReturnScraper's own controller), since a hand
   holds one thing.
2. One draw of the start layout names each dumpling's slot, exactly as the person's
   return draws it. A dumpling that rests on the rack within the layout's own jitter of
   its slot is left alone. Every other one is picked up (PickFood's controller) and set
   down on its slot with the slot's yaw, a dumpling already in the hand first. One whose
   slot another displaced dumpling still covers waits until that one has moved. So does
   one that cannot be taken without its snap into the hand landing inside a neighbour
   (a pile on the plate): the pick refuses, nothing is disturbed, and it is tried again
   once another dumpling has been taken away.
3. If the scraper was in the hand at the start, it is picked up again, so a successful
   return leaves the hand as it found it.

One abstraction, isolated and switchable (`RackServingConfig.robot_batch_return_yaw`):
by default the set-down writes the slot's yaw, as welded handling already writes the
spot (`WeldedFood.set_down`'s `yaw`, the only place). The hand's reach to the slot, the
base and arm motion and every refusal are real. With "hand" nothing is written and the
hand must turn the dumpling to the slot's yaw itself.

Unlike the person's return it moves only what is out of place, takes thousands of
control steps, and can fail: a refused reach or a slot it cannot clear ends the skill
there with `SkillFailed`. Nothing is undone and nothing is hidden: the dumplings already
returned stay on the rack, the rest stay where they lie, and a scraper already put home
stays home. The measured state then says what is true, and this skill and the person's
return both remain applicable.
"""

import contextlib
from typing import Any, ClassVar

import numpy as np
from spatialmath import SE2

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

from .food import SNAP_INTO_FOOD, RackFoodController, WeldedFoodSkills
from .layout import RackLayout

Slot = tuple[float, float, float]


class RackRobotBatchReturn:
    # A dumpling is in place where a draw of the layout could have put it: within the
    # layout's jitter of its slot's centre. The remainder is what a set-down dumpling
    # moves as it settles on the wires.
    SLOT_TOLERANCE_M: ClassVar[float] = RackLayout.LINED_UP_JITTER + 0.002
    SLOT_YAW_TOLERANCE: ClassVar[float] = RackLayout.LINED_UP_YAW_JITTER + 0.02
    MAX_SLOT_DRAWS: ClassVar[int] = 300

    def __init__(self, *, controllers: Any) -> None:
        self.controllers = controllers
        self.backend = controllers.backend

    def in_place(self, *, name: str) -> bool:
        """Resting on the rack within the layout's own jitter of its slot's centre."""
        backend = self.backend
        if backend.location(name=name) != "OnRack":
            return False
        slot = backend.rack_slots(jitter=False)[name]
        x, y = backend.food_centre(name=name)
        turn = (backend.food_yaw(name=name) - slot[2] + np.pi) % (2 * np.pi) - np.pi
        return (
            abs(x - slot[0]) <= self.SLOT_TOLERANCE_M
            and abs(y - slot[1]) <= self.SLOT_TOLERANCE_M
            and abs(turn) <= self.SLOT_YAW_TOLERANCE
        )

    def put_back(self, *, name: str, slot: Slot) -> None:
        """Pick `name` up unless it is already in the hand, and set it down on `slot`,
        every base path both plan kept `BASE_CLEARANCE_M` from the counter where such a
        path exists (`ScraperMotionPlanner.clearance_guard`). Only the food motions: the
        scraper's own return and pick plan exactly as the skills ReturnScraper and
        PickScraper do, since a scraper picked from another stance is held differently
        and changes what the next GatherAndSweep delivers."""
        from .planning import BASE_CLEARANCE_M

        food = WeldedFoodSkills(food=RackFoodController(controllers=self.controllers))
        shelf = self.controllers.shelf
        shelf.base_clearance = BASE_CLEARANCE_M
        try:
            if not self.backend.held(name=name):
                food.pick(name=name, clear_of_food=True)
            food.place(name=name, destination="rack_0", slot=slot)
        finally:
            shelf.base_clearance = 0.0

    def draw_slots(self) -> tuple[dict[str, Slot], list[str]]:
        """A draw of the start layout and the dumplings out of place under it, redrawn
        until those, on their slots, are clear of one another and of the dumplings left
        alone: the person's return redraws a layout that collides in the same way."""
        backend = self.backend
        pending = [name for name in backend.cube_names if not self.in_place(name=name)]
        for _ in range(self.MAX_SLOT_DRAWS):
            slots = backend.rack_slots()
            if backend.slots_clear(slots={name: slots[name] for name in pending}):
                return slots, pending
        raise SkillFailed("RobotBatchReturn returned 0: no draw of the start layout is clear")

    def run(self) -> None:
        """Return the food, then leave the robot where its skills work: the arm at its
        episode-start pose (`park`). An abort first backs off to a standard stance
        (`back_off`), then fails as it would have."""
        try:
            self.return_food()
        except SkillFailed as error:
            try:
                self.back_off()
            except SkillFailed as refused:
                raise SkillFailed(f"{error}; back-off refused: {refused}") from error
            raise
        if not self.hand_full():
            self.park()

    def return_food(self) -> None:
        backend, controllers = self.backend, self.controllers
        carried = backend.held(name="wiper_0")
        report: dict[str, Any] = {
            "scraper_carried": carried,
            "moved": [],
            "out_of_place": [],
            "placed": {},
            "buried": [],
        }
        controllers.metrics = {"robot_batch_return": report}
        if carried:
            controllers.return_scraper()
        slots, pending = self.draw_slots()
        # A dumpling in the hand goes first: the hand can pick nothing else up.
        pending.sort(key=lambda name: not backend.held(name=name))
        report["out_of_place"] = list(pending)
        total = len(pending)
        # Dumplings whose pick was refused because the snap would put them inside a
        # neighbour: tried again once another one has been taken away.
        buried: list[str] = []
        while pending:
            name = next(
                (
                    name
                    for name in pending
                    if name not in buried and backend.slots_clear(slots={name: slots[name]})
                ),
                None,
            )
            try:
                if name is None:
                    raise SkillFailed(
                        f"{', '.join(buried)} cannot be taken clear of the other food"
                        if buried
                        else f"the slots of {', '.join(pending)} are covered by one another"
                    )
                try:
                    self.put_back(name=name, slot=slots[name])
                except SkillFailed as refused:
                    if SNAP_INTO_FOOD not in str(refused):
                        raise
                    buried.append(name)
                    report["buried"].append(name)
                    continue
                if not self.in_place(name=name):
                    raise SkillFailed(f"{name} did not come to rest on its slot")
            except SkillFailed as error:
                raise SkillFailed(
                    f"RobotBatchReturn returned {total - len(pending)} of {total}: {error}"
                ) from error
            buried.clear()
            pending.remove(name)
            report["moved"].append(name)
            x, y = backend.food_centre(name=name)
            turn = backend.food_yaw(name=name) - slots[name][2]
            report["placed"][name] = {
                "position_error_m": float(np.hypot(x - slots[name][0], y - slots[name][1])),
                "yaw_error_rad": float(abs((turn + np.pi) % (2 * np.pi) - np.pi)),
                "yaw_written_by_set_down": backend.config.robot_batch_return_yaw == "snap",
            }
        if carried:
            controllers.pick_scraper()

    def hand_full(self) -> bool:
        backend = self.backend
        return any(backend.held(name=name) for name in ("wiper_0", *backend.cube_names))

    def standard_stances(self) -> list[np.ndarray]:
        """The stances the gather and sweep strokes drive from (`StrokeStances`): the
        rear stance and the two side stances, nearest the base first."""
        config, shelf, stances = (
            self.backend.config,
            self.controllers.shelf,
            self.controllers.stances,
        )
        out = [
            stances.rear_base(y=config.rack_center_y, config=config),
            stances.side_base(side=-1, config=config),
            stances.side_base(side=1, config=config),
        ]
        base = shelf.data.qpos[shelf.ba].copy()
        out.sort(key=lambda stance: float(np.linalg.norm(stance[:2] - base[:2])))
        return out

    def park(self) -> None:
        """The empty arm, planned and driven, to its episode-start pose (`home_arm`)
        where the base stands; from a base where that is refused, by way of `back_off`.

        PickScraper seeds its hover IK from the arm as it finds it. From the arm the
        last set-down left, the hover lands on another IK branch, and gather(0.05,
        0.015) failed 5/5 starts that succeeded 5/5 from this pose; where the base
        stood made no difference (2026-11-25)."""
        try:
            with self.controllers.feedback():
                self.tuck()
        except SkillFailed:
            self.back_off()

    def tuck(self) -> None:
        """Plan the empty arm to `home_arm` at the base as it stands, and follow it."""
        from hitl_pmp.environments.pouring3d.kitchen.planning.parameterized_skills import (
            KitchenMotionPlanner,
        )

        from .planning import PlanningRefused, RackPyBulletSim, ScraperMotionPlanner

        tool, shelf = self.controllers.tool, self.controllers.shelf
        base = shelf.data.qpos[shelf.ba].copy()
        start = self.unwound(arm=shelf.data.qpos[shelf.qa].copy())
        home = np.asarray(shelf.home_arm, dtype=float)
        if float(np.max(np.abs((start - home + np.pi) % (2 * np.pi) - np.pi))) < 0.01:
            return
        ok = ScraperMotionPlanner.mujoco_guard(tool=tool, holding=False)
        sim = RackPyBulletSim.from_shelf(shelf=shelf)
        try:
            sim.set_robot_base(SE2(*(float(v) for v in base)))
            path = ScraperMotionPlanner.plan_arm_to(
                sim=sim,
                start=start,
                goal=KitchenMotionPlanner.with_fingers(sim=sim, arm=home),
                what="the episode-start arm",
                guard=ScraperMotionPlanner.arm_guard(ok=ok, base=base),
            )
        except PlanningRefused as exc:
            raise SkillFailed(f"RobotBatchReturn park: {exc}") from exc
        finally:
            sim.close()
        tool.follow_arm_path(path=[start, *path[1:]], label="robot_return_park")

    def unwound(self, *, arm: np.ndarray) -> np.ndarray:
        """`arm` with each continuous joint within a half turn of zero: the same pose,
        in the range the planner checks. Long practice winds them tens of turns."""
        from .planning import ScraperMotionPlanner

        lower, upper = ScraperMotionPlanner.limits(shelf=self.controllers.shelf)
        continuous = (lower <= -2 * np.pi) & (upper >= 2 * np.pi)
        return np.where(continuous, (arm + np.pi) % (2 * np.pi) - np.pi, arm)

    def back_off(self) -> None:
        """After an abort: out of wherever it stopped to the nearest standard stance it
        has a base path to, then, with an empty hand, the arm home there.

        A food lift refused mid-carry left the hand out over the counter beside the
        plate (B=120 ees05 seed 0, 2026-11-23). No later skill planned a base path out
        with the arm carried as it was: every move brought the hand within the
        planner's margin of the plate, and BlockedStart escapes only a start already
        inside it. So the hand is first lifted along the escape's own checked line,
        then the base is driven with the arm as it is, or else tucked first. Whatever
        the hand holds stays in it: nothing is put down or hidden."""
        from .tool import ESCAPE_UP_M, BlockedStart

        controllers, backend = self.controllers, self.backend
        food_name = next((n for n in backend.cube_names if backend.held(name=n)), None)
        holding_scraper = backend.held(name="wiper_0")
        motion: Any = controllers.tool
        if food_name is not None:
            food = RackFoodController(controllers=controllers)
            food.carried = food_name
            motion = food.motion
        with controllers.feedback():
            # No checked line up, nor back: drive from where the hand is.
            with contextlib.suppress(SkillFailed):
                BlockedStart.escape(
                    tool=motion,
                    label="robot_return_back_off_lift",
                    holding=holding_scraper,
                    distance=ESCAPE_UP_M,
                    back_first=False,
                    reason="RobotBatchReturn back-off",
                )
            self.drive_to_stance(
                motion=motion,
                held_name=food_name or ("wiper_0" if holding_scraper else None),
                holding=holding_scraper,
            )
            if not self.hand_full():
                self.tuck()

    def drive_to_stance(self, *, motion: Any, held_name: str | None, holding: bool) -> None:
        """A base path to the nearest standard stance that has one, carrying the arm as
        it is or, with an empty hand and no path that way, tucked home first."""
        from hitl_pmp.environments.pouring3d.kitchen.planning.parameterized_skills import (
            HeldObject,
            KitchenMotionPlanner,
        )

        from .planning import PlanningRefused, RackPyBulletSim, ScraperMotionPlanner

        shelf = self.controllers.shelf
        base = shelf.data.qpos[shelf.ba].copy()
        arm = self.unwound(arm=shelf.data.qpos[shelf.qa].copy())
        ignores_food = bool(getattr(motion, "ignores_food", False))
        sim = RackPyBulletSim.from_shelf(shelf=shelf, ignores_food=ignores_food)
        state = shelf.core._get_object_centric_state()  # noqa: SLF001
        ok = ScraperMotionPlanner.mujoco_guard(tool=motion, holding=holding)
        failures: list[str] = []
        try:
            sim.set_robot_base(SE2(*(float(v) for v in base)))
            held = (
                None
                if held_name is None
                else HeldObject.grasped(sim=sim, arm=arm, name=held_name, riders=())
            )
            ignore: tuple[str, ...] | None = None if held_name is None else (held_name,)
            if ignores_food:
                ignore = tuple(
                    name
                    for name in state.get_object_names()
                    if name.startswith(RackPyBulletSim.movable_prefix)
                )
            carries: list[list[np.ndarray]] = [[arm]]
            if held is None:
                try:
                    carries.append(
                        ScraperMotionPlanner.plan_arm_to(
                            sim=sim,
                            start=arm,
                            goal=KitchenMotionPlanner.with_fingers(
                                sim=sim, arm=np.asarray(shelf.home_arm, dtype=float)
                            ),
                            what="the episode-start arm",
                            guard=ScraperMotionPlanner.arm_guard(ok=ok, base=base),
                        )
                    )
                except PlanningRefused as exc:
                    failures.append(str(exc))
            for stance in self.standard_stances():
                for carry in carries:
                    try:
                        sim.set_robot_base(SE2(*(float(v) for v in base)))
                        path = ScraperMotionPlanner.plan_base(
                            sim=sim,
                            state=state,
                            base=base,
                            target=stance,
                            arm=carry[-1],
                            what="a standard stance",
                            guard=ScraperMotionPlanner.base_guard(ok=ok, arm=carry[-1]),
                            held=held,
                            ignore=ignore,
                        )
                    except PlanningRefused as exc:
                        failures.append(str(exc))
                        continue
                    if len(carry) > 1:
                        motion.follow_arm_path(
                            path=[arm, *carry[1:]], label="robot_return_back_off_tuck"
                        )
                    motion.follow_base_path(path=path, label="robot_return_back_off")
                    return
        finally:
            sim.close()
        raise SkillFailed(
            "RobotBatchReturn back-off: no base path to a standard stance "
            f"({ScraperMotionPlanner.tally(failures=failures)})"
        )
