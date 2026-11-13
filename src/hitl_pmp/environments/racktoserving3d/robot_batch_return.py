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
   slot another displaced dumpling still covers waits until that one has moved.
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

from typing import Any, ClassVar

import numpy as np

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

from .food import RackFoodController, WeldedFoodSkills
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
        """Pick `name` up unless it is already in the hand, and set it down on `slot`."""
        food = WeldedFoodSkills(food=RackFoodController(controllers=self.controllers))
        if not self.backend.held(name=name):
            food.pick(name=name)
        food.place(name=name, destination="rack_0", slot=slot)

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
        backend, controllers = self.backend, self.controllers
        carried = backend.held(name="wiper_0")
        report: dict[str, Any] = {
            "scraper_carried": carried,
            "moved": [],
            "out_of_place": [],
            "placed": {},
        }
        controllers.metrics = {"robot_batch_return": report}
        if carried:
            controllers.return_scraper()
        slots, pending = self.draw_slots()
        # A dumpling in the hand goes first: the hand can pick nothing else up.
        pending.sort(key=lambda name: not backend.held(name=name))
        report["out_of_place"] = list(pending)
        total = len(pending)
        while pending:
            name = next(
                (name for name in pending if backend.slots_clear(slots={name: slots[name]})),
                None,
            )
            try:
                if name is None:
                    raise SkillFailed(
                        f"the slots of {', '.join(pending)} are covered by one another"
                    )
                self.put_back(name=name, slot=slots[name])
                if not self.in_place(name=name):
                    raise SkillFailed(f"{name} did not come to rest on its slot")
            except SkillFailed as error:
                raise SkillFailed(
                    f"RobotBatchReturn returned {total - len(pending)} of {total}: {error}"
                ) from error
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
