"""Sweep simulator adapter: one chosen skill per action, no automatic reset.

This adapter never calls the prototype's scripted self-reset run(). The planner
chooses each primitive, whose observed effects are measured after execution.
"""

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from gymnasium.spaces import Box
from pydantic import PrivateAttr

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, State

from .motion import ExecutionError
from .primitives import DrawerStroke, WiperHold
from .self_reset import SweepDrawerSelfReset
from .session import SweepDrawerSession
from .start_regions import StartValidation, SweepRegions
from .stock_skills import StockSweepSkills
from .symbolic import SweepSymbols
from .types import SweepDrawerScene as S
from .upstream import SweepUpstreamFacts


class InvalidSweepStart(ValueError):
    """A rejected initial seed; callers must record it, never silently resample."""


class SweepDrawerEnvironment(Environment):
    canonical_seed: int = 0
    evaluation: bool = False
    output_dir: Path | None = None
    require_valid_start: bool = True
    action_space: ClassVar[Box] = Box(
        low=np.array([-1.0, -100.0, -100.0]), high=np.array([100.0, 100.0, 100.0]), dtype=np.float64
    )
    ACTION_NAMES: ClassVar[tuple[str, ...]] = tuple(s.name for s in SweepSymbols.skills())
    _upstream: Any = PrivateAttr(default=None)
    _session: SweepDrawerSession | None = PrivateAttr(default=None)
    _reset: SweepDrawerSelfReset | None = PrivateAttr(default=None)
    _held: dict[str, Any] = PrivateAttr(default_factory=dict)
    _initial_state: Any = PrivateAttr(default=None)
    _initial_validation: StartValidation | None = PrivateAttr(default=None)
    _action_count: int = PrivateAttr(default=0)
    _hard_reset_count: int = PrivateAttr(default=0)
    _human_reset_count: int = PrivateAttr(default=0)

    def session(self) -> SweepDrawerSession:
        assert self._session is not None, "hard_reset must initialize the simulator"
        return self._session

    def recovery(self) -> SweepDrawerSelfReset:
        if self._reset is None:
            self._reset = SweepDrawerSelfReset(session=self.session())
        return self._reset

    def hard_reset(self) -> None:
        if self._action_count and not self.evaluation:
            raise RuntimeError("Automatic reset of continuous Sweep practice is forbidden")
        self.reset_to_seed(seed=self.canonical_seed)

    def reset_to_seed(self, *, seed: int) -> State:
        if self._action_count and not self.evaluation:
            raise RuntimeError("Replacing the live Sweep practice scene is forbidden")
        self.close()
        episode_dir = (
            None
            if self.output_dir is None
            else self.output_dir / "episodes" / f"{self._hard_reset_count:06d}-seed{seed}"
        )
        log_path = None if episode_dir is None else episode_dir / "state.jsonl"
        replay_path = None if episode_dir is None else episode_dir / "replay.jsonl"
        self._session = SweepDrawerSession(seed=seed, log_path=log_path, replay_path=replay_path)
        self._write_event(
            event={
                "kind": "episode_start",
                "seed": seed,
                "episode_index": self._hard_reset_count,
                "state_log": None if log_path is None else str(log_path),
                "replay_log": None if replay_path is None else str(replay_path),
                "evaluation": self.evaluation,
            }
        )
        self._initial_validation = SweepRegions.validate(session=self.session())
        self._write_event(
            event={"kind": "initial_validation", **self._initial_validation.model_dump()}
        )
        if self.require_valid_start and not self._initial_validation.valid:
            raise InvalidSweepStart(f"seed {seed}: {self._initial_validation.reasons}")
        self._upstream = SweepUpstreamFacts.create(state=self.session().state)
        self._initial_state = self.session().state.copy()
        self._held.clear()
        self._hard_reset_count += 1
        self.current_state = self.observe()
        return self.current_state

    def set_state(self, *, state: State) -> None:
        # Task snapshots identify their original scene seed. This entry point is
        # evaluation-only once practice has begun; humans use reset_movables.
        seed = int(state.get(obj=SweepSymbols.SCENE, feature_name="seed"))
        self.reset_to_seed(seed=seed)

    def noop_action(self) -> Action:
        return np.array([-1.0, 0.0, 0.0])

    def get_valid_actions(self) -> list[Action]:
        return [np.array([float(i), 0.0, 0.0]) for i in range(len(self.ACTION_NAMES))]

    def take_action(self, *, action: Action) -> State:
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("Sweep action must be [skill_id, distance, rotation]")
        index = int(action[0])
        if index != action[0] or not -1 <= index < len(self.ACTION_NAMES):
            raise ValueError(f"Invalid Sweep action id {action[0]}")
        self._action_count += 1
        if index == -1:
            self._write_event(event={"kind": "action", "name": "NoOp", "index": self._action_count})
            return self.get_current_state()
        name = self.ACTION_NAMES[index]
        before = self.session().ticks
        error = ""
        if name in SweepSymbols.TRAINABLE:
            key = dict(zip(SweepSymbols.TRAINABLE, StockSweepSkills.SKILLS, strict=True))[name]
            record = StockSweepSkills.run(
                session=self.session(),
                skill=key,
                label=name,
                phase="evaluation" if self.evaluation else "practice",
                params=action[1:],
            )
            error = record.note
        else:
            self.session().begin(
                name=name, kind=name, phase="evaluation" if self.evaluation else "practice"
            )
            try:
                self._execute_recovery(name=name)
            except ExecutionError as exc:
                error = str(exc)
        self.current_state = self.observe()
        symbolic = GroundSkill(skill=SweepSymbols.skills()[index], objects=(SweepSymbols.SCENE,))
        effects_hold = all(
            a.predicate.holds(self.current_state, a.objects) for a in symbolic.add_effects
        )
        if name not in SweepSymbols.TRAINABLE:
            if not effects_hold and not error:
                error = "Controller returned without achieving declared observed effects"
            self.session().end(success=not error and effects_hold, note=error)
        self._write_event(
            event={
                "kind": "action",
                "name": name,
                "index": self._action_count,
                "params": action[1:].tolist(),
                "error": error,
                "symbolic_success": effects_hold,
                "declared_start_target_satisfied": all(
                    atom.predicate.holds(self.current_state, atom.objects)
                    for atom in SweepSymbols.human_reset(cost=0.0).add_effects
                ),
                "ticks": self.session().ticks - before,
                "facts": {
                    n: bool(self.current_state.get(obj=SweepSymbols.SCENE, feature_name=n))
                    for n in SweepSymbols.FACT_NAMES
                },
            }
        )
        return self.current_state

    def _execute_recovery(self, *, name: str) -> None:
        reset = self.recovery()
        primitives = reset.primitives
        if name == "OpenGripper":
            reset.primitives.motion.set_gripper(command=0.0)
            self._held.clear()
        elif name == "RecoverWiper":
            primitives.recover_wiper()
        elif name.startswith("WiggleDrawerCube"):
            reset.repositioning.wiggle_drawer()
        elif name.startswith("PickGroup"):
            self._pick_group(group=tuple(map(int, name.removeprefix("PickGroup"))))
        elif name.startswith("PlaceGroup"):
            self._place_group(group=tuple(map(int, name.removeprefix("PlaceGroup"))))
        elif name == "ParkWiper":
            primitives.park_wiper()
        elif name == "OpenResetDrawer":
            primitives.move_drawer(target=reset.open_to)
        elif name == "CloseDrawer":
            primitives.move_drawer(target=0.0)
        elif name == "ParkRobot":
            reset.park()
        elif name.startswith("PickCube"):
            cube = S.CUBES[int(name.removeprefix("PickCube"))]
            transform, _ = primitives.pick(cube=cube)
            self._held[cube] = transform
        elif name.startswith("PlaceCube"):
            cube = S.CUBES[int(name.removeprefix("PlaceCube"))]
            if cube not in self._held:
                raise ExecutionError(f"No executed grasp transform for {cube}")
            target, yaw = reset._target(cube=cube)
            primitives.place(
                cube=cube, target_xy=target, target_yaw=yaw, ee_to_cube=self._held[cube]
            )
            self._held.pop(cube)
        elif name.startswith("NudgeCube"):
            reset.repositioning.nudge(cube=S.CUBES[int(name.removeprefix("NudgeCube"))])
        elif name.startswith("PushCube"):
            primitives.push(cube=S.CUBES[int(name.removeprefix("PushCube"))])
        else:
            raise ValueError(f"Recovery dispatcher missing {name}")

    def _group_endpoints(self, *, group: tuple[int, ...]) -> tuple[str, str] | None:
        primitives = self.recovery().primitives
        members = {S.CUBES[i] for i in group}
        for i in group:
            for j in group:
                if i == j:
                    continue
                cube, partner = S.CUBES[i], S.CUBES[j]
                middle = primitives.row_between(cube=cube, partner=partner)
                if middle is None or {cube, *middle, partner} != members:
                    continue
                try:
                    primitives.find_pick(cube=cube, partner=partner)
                except ExecutionError:
                    continue
                return cube, partner
        return None

    def _pick_group(self, *, group: tuple[int, ...]) -> None:
        from pybullet_helpers.geometry import Pose, multiply_poses

        endpoints = self._group_endpoints(group=group)
        if endpoints is None:
            raise ExecutionError(f"No feasible exact row grasp for {group}")
        primitives = self.recovery().primitives
        primitives.pick(cube=endpoints[0], partner=endpoints[1])
        inverse = primitives.scene.ee_now().invert()
        for i in group:
            cube = S.CUBES[i]
            pose = Pose(
                tuple(self.session().position(name=cube)), self.session().quaternion(name=cube)
            )
            self._held[cube] = multiply_poses(inverse, pose)

    def _place_group(self, *, group: tuple[int, ...]) -> None:
        reset = self.recovery()
        members = tuple(S.CUBES[i] for i in group)
        if any(cube not in self._held for cube in members):
            raise ExecutionError(f"No executed grasp for all row members {group}")
        anchor = members[0]
        positions = np.asarray([self.session().position(name=cube)[:2] for cube in members])
        center, _ = reset._target(cube=anchor, spacing=0.07, margin=0.035, free_spot=True)
        target = np.asarray(center) + positions[0] - positions.mean(axis=0)
        reset.primitives.place(
            cube=anchor, target_xy=tuple(target), ee_to_cube=self._held[anchor], release=0.0
        )
        for cube in members:
            self._held.pop(cube, None)

    def observe(self) -> State:
        session = self.session()
        values = dict.fromkeys(SweepSymbols.SCENE_TYPE.feature_names, 0.0)
        values.update(
            seed=float(session.seed),
            base_x=session.base()[0],
            base_y=session.base()[1],
            base_yaw=session.base()[2],
            drawer_pos=session.drawer_pos(),
            gripper=session.gripper(),
        )
        reset = self.recovery()
        ee = np.array(reset.primitives.scene.ee_now().position)
        holding_wiper = session.gripper() > 0.2 and WiperHold.in_hand(
            gripper=ee, wiper=session.position(name=S.WIPER)
        )
        # Verify a stored grasp against observed geometry. A dropped cube must
        # never retain the symbolic HoldingCube fact just because pick returned.
        self._held = {
            cube: transform
            for cube, transform in self._held.items()
            if session.gripper() > 0.2 and np.linalg.norm(session.position(name=cube) - ee) < 0.15
        }
        empty = not holding_wiper and not self._held and session.gripper() < 0.2
        values.update(
            RecoveryHandEmpty=float(empty),
            RecoveryArmHome=float(
                np.max(np.abs((session.arm() - np.asarray(S.HOME) + np.pi) % (2 * np.pi) - np.pi))
                < 0.01
            ),
            RecoveryGripperClosed=float(session.gripper() >= 0.2),
            WiperLow=float(
                not holding_wiper and session.position(name=S.WIPER)[2] < S.COUNTER_TOP - 0.02
            ),
            ResetDrawerNotOpen=float(
                not DrawerStroke.reached(end=session.drawer_pos(), target=reset.open_to)
            ),
            ClosedEmpty=float(not holding_wiper and not self._held and not empty),
            PhysicallyHoldingWiper=float(holding_wiper),
            ResetDrawerOpen=float(
                DrawerStroke.reached(end=session.drawer_pos(), target=reset.open_to)
            ),
            ResetDrawerClosed=float(session.drawer_pos() < 0.01),
            DrawerNotOpen=float(session.drawer_pos() < 0.15),
            DrawerNotClosed=float(session.drawer_pos() >= 0.01),
        )
        validation = SweepRegions.validate(session=session)
        values["WiperHome"] = float(
            all(validation.checks.get(f"{S.WIPER}:{k}", False) for k in ("region", "yaw"))
        )
        values["RobotHome"] = float(
            all(validation.checks.get(f"{S.ROBOT}:{k}", False) for k in ("region", "yaw"))
        )
        values["RobotAway"] = 1.0 - values["RobotHome"]
        for i, cube in enumerate(S.CUBES):
            in_pile = all(validation.checks.get(f"{cube}:{k}", False) for k in ("region", "yaw"))
            held = cube in self._held
            loose = not in_pile and not held
            pickable = empty and loose and reset.primitives.pick_feasible(cube=cube)
            in_drawer = SweepRegions.in_goal(session=session, cube=cube)
            values.update({
                f"InPile{i}": float(in_pile),
                f"PhysicallyInDrawer{i}": float(in_drawer),
                f"SweepReachable{i}": float(in_pile or in_drawer),
                f"HoldingCube{i}": float(held),
                f"Loose{i}": float(loose),
                f"Pickable{i}": float(pickable),
                f"Blocked{i}": float(loose and not pickable),
            })
        for group in SweepSymbols.GROUPS:
            suffix = "".join(map(str, group))
            values[f"GroupPickable{suffix}"] = float(
                empty
                and all(values[f"Loose{i}"] for i in group)
                and self._group_endpoints(group=group) is not None
            )
        assert self._upstream is not None
        for fact in SweepUpstreamFacts.observed(abstractor=self._upstream, state=session.state):
            values[fact] = 1.0
        values["AnyCubeInPile"] = float(any(values[f"InPile{i}"] for i in range(5)))
        for short, physical in (
            ("wiper", S.WIPER),
            *((f"cube{i}", c) for i, c in enumerate(S.CUBES)),
        ):
            for axis, value in zip(
                ("x", "y", "z", "qx", "qy", "qz", "qw"),
                (*session.position(name=physical), *session.quaternion(name=physical)),
                strict=True,
            ):
                values[f"{short}_{axis}"] = float(value)
        return State(
            data={
                SweepSymbols.SCENE: np.array([
                    values[n] for n in SweepSymbols.SCENE_TYPE.feature_names
                ])
            }
        )

    def reset_movables(self, *, destination: str | None = None) -> bool:
        """Explicit charged human reset to this run's validated start sample.

        Sweep's approved intervention includes the robot, unlike the historical
        cube-bin-only intervention in Toss. No automatic practice/eval path calls
        this method. Restoring a fixed valid sample introduces no resampling.
        """
        if self.evaluation:
            raise RuntimeError("Human intervention is unavailable during evaluation")
        if destination is not None:
            raise ValueError("Sweep has one declared start destination")
        if self._initial_state is None or self._initial_validation is None:
            raise RuntimeError("No validated initial sample for human reset")
        if not self._initial_validation.valid:
            raise RuntimeError("Human reset cannot repair an invalid initializer")
        self.session().begin(name="HumanReset", kind="HumanReset", phase="practice")
        self.session().restore(state=self._initial_state.copy())
        self._held.clear()
        self.current_state = self.observe()
        validation = SweepRegions.validate(session=self.session())
        reset = SweepSymbols.human_reset(cost=0.0)
        effects_hold = all(
            a.predicate.holds(self.current_state, a.objects) for a in reset.add_effects
        )
        succeeded = validation.valid and effects_hold
        self._human_reset_count += 1
        self.session().end(success=succeeded, note="explicit human start-distribution restoration")
        self._write_event(
            event={
                "kind": "human_reset",
                "index": self._human_reset_count,
                "seed": self.session().seed,
                "success": succeeded,
                "validation": validation.model_dump(),
                "effects_hold": effects_hold,
            }
        )
        if not succeeded:
            raise RuntimeError("Human reset failed its shared declared-start contract")
        return True

    def _write_event(self, *, event: dict[str, Any]) -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "sweep_events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    def close(self) -> None:
        if self._upstream is not None:
            SweepUpstreamFacts.close(abstractor=self._upstream)
            self._upstream = None
        if self._reset is not None:
            self._reset.primitives.scene._sim.close()
            self._reset = None
        if self._session is not None:
            self._session.close()
            self._session = None
