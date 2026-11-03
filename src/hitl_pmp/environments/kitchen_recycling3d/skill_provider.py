"""Robot action encoding and explicit, training-only human request groundings."""

from functools import cached_property

import numpy as np

from hitl_pmp.core.method.skill_provider import OraclePolicyProvider, SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate

from .environment import KitchenRecyclingEnvironment
from .predicates import HAND_EMPTY, HOLDING, IN_BIN, PICKABLE, KitchenRecyclingPredicates
from .skills import KitchenRecyclingSkills
from .types import KitchenObjects


class KitchenRecyclingSkillProvider(SkillProvider):
    env: KitchenRecyclingEnvironment
    offer_human_reset: bool = True

    @cached_property
    def _objects(self) -> KitchenObjects:
        return KitchenObjects(config=self.env.config)

    @cached_property
    def _robot_skills(self) -> tuple[Skill, ...]:
        return KitchenRecyclingSkills.robot_skills(config=self.env.config)

    @cached_property
    def _human_skills(self) -> tuple[GroundSkill, ...]:
        recovery_items = {
            f"Recover_{item.name}_{context}": item
            for item in self._objects.items
            for context in ("near", "far")
        }
        return tuple(
            GroundSkill(
                skill=skill,
                objects=(self._objects.robot, self._objects.bin)
                if len(skill.parameters) == 2
                else (self._objects.robot, recovery_items[skill.name], self._objects.bin),
            )
            for skill in KitchenRecyclingSkills.human_skills(config=self.env.config)
        )

    def skills(self) -> tuple[Skill, ...]:
        return self._robot_skills

    def predicates(self) -> tuple[Predicate, ...]:
        return KitchenRecyclingPredicates.all(config=self.env.config)

    def types(self) -> tuple[Type, ...]:
        return self._objects.all_types()

    def objects(self) -> tuple[Object, ...]:
        return self._objects.all_objects()

    def _robot_signature(self, *, ground_skill: GroundSkill) -> tuple[int, int, str]:
        if ground_skill.skill not in self._robot_skills:
            raise ValueError("Only registered robot skills have robot actions")
        context = ground_skill.skill.name.rsplit("_", 1)[1]
        if len(ground_skill.objects) == 2:
            return 2, -1, context
        item = ground_skill.objects[1]
        for operation, opcode in (("Pick", 0), ("Toss", 1), ("OpenGripper", 2)):
            if ground_skill.skill.name == f"{operation}_{item.name}_{context}":
                return opcode, self._objects.items.index(item), context
        raise ValueError("Skill identity does not match its bound object")

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        opcode, item_index, context = self._robot_signature(ground_skill=ground_skill)
        if opcode != 1:
            return np.zeros(0)
        calibration = self.env.config.calibration(
            name=self._objects.items[item_index].name, context=context
        )
        lower = np.array([
            max(0.3, calibration.toss_standoff - 0.5),
            max(20.0, calibration.toss_speed_deg_s - 70.0),
            max(0.0, calibration.release_ms - 250.0),
        ])
        upper = np.array([
            calibration.toss_standoff + 0.5,
            calibration.toss_speed_deg_s + 70.0,
            min(1000.0, calibration.release_ms + 250.0),
        ])
        return rng.uniform(lower, upper)

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        del state
        opcode, item_index, context = self._robot_signature(ground_skill=ground_skill)
        values = np.asarray(params, dtype=float)
        if values.shape != (ground_skill.skill.param_dim,) or not np.isfinite(values).all():
            raise ValueError("Wrong parameter dimension or nonfinite skill parameters")
        action = np.zeros(6)
        action[:2] = opcode, item_index
        if opcode != 2:
            calibration = self.env.config.calibration(
                name=self._objects.items[item_index].name, context=context
            )
            if opcode == 0:
                action[2] = calibration.grasp_offset
            else:
                action[2:] = values[0], calibration.toss_azimuth, values[1], values[2]
        return action

    def hand_selected_feature_transform(
        self, *, ground_skill: GroundSkill, state: State, params: np.ndarray
    ) -> list[float]:
        features = [1.0]
        omitted = {"seed", "skills_taken", "training"}
        for obj in ground_skill.objects:
            features.extend(
                float(state.get(obj=obj, feature_name=feature))
                for feature in obj.type.feature_names
                if feature not in omitted
            )
        return features + [float(value) for value in params]

    def human_cube_bin_reset_skills(self) -> tuple[GroundSkill, ...]:
        if not self.offer_human_reset or self.env.phase != "training":
            return ()
        return self._human_skills

    def human_cube_bin_reset_skill(self) -> GroundSkill | None:
        skills = self.human_cube_bin_reset_skills()
        return skills[0] if skills else None

    def movables_reset_destination(self, *, ground_skill: GroundSkill) -> str:
        if ground_skill not in self._human_skills:
            raise ValueError("Not a registered human skill grounding")
        if ground_skill.skill.name == "RelocateBinNear":
            return "relocate_bin_near"
        if ground_skill.skill.name == "RestoreBinTarget":
            return "restore_bin_target"
        return f"recover:{ground_skill.objects[1].name}"


class KitchenRecyclingOracle(OraclePolicyProvider):
    """A calibration-based robot policy, not an asserted performance upper bound."""

    env: KitchenRecyclingEnvironment

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        if goal.is_satisfied(state=state):
            return LabeledAction(action=self.env.noop_action(), label="no-op (goal reached)")
        provider = KitchenRecyclingSkillProvider(env=self.env, offer_human_reset=False)
        objects = provider._objects
        context = "near" if state.get(obj=objects.bin, feature_name="near") > 0.5 else "far"
        held = next(
            (item for item in objects.items if HOLDING.holds(state, (objects.robot, item))), None
        )
        bound: tuple[Object, ...]
        if held is not None:
            name = f"Toss_{held.name}_{context}"
            calibration = self.env.config.calibration(name=held.name, context=context)
            params = np.array([
                calibration.toss_standoff,
                calibration.toss_speed_deg_s,
                calibration.release_ms,
            ])
            bound = (objects.robot, held, objects.bin)
        elif not HAND_EMPTY.holds(state, (objects.robot,)):
            name = f"OpenGripperEmpty_{context}"
            params = np.zeros(0)
            bound = (objects.robot, objects.bin)
        else:
            candidate = next(
                (
                    objects.by_name[name]
                    for name in self.env.config.required_objects
                    if not IN_BIN.holds(state, (objects.by_name[name], objects.bin))
                    and PICKABLE.holds(state, (objects.by_name[name],))
                ),
                None,
            )
            if candidate is None:
                return LabeledAction(
                    action=self.env.noop_action(), label="no-op (no robot recovery)"
                )
            name = f"Pick_{candidate.name}_{context}"
            params = np.zeros(0)
            bound = (objects.robot, candidate, objects.bin)
        skill = next(skill for skill in provider.skills() if skill.name == name)
        return LabeledAction(
            action=provider.compute_action(
                ground_skill=GroundSkill(skill=skill, objects=bound), params=params, state=state
            ),
            label=name,
        )
