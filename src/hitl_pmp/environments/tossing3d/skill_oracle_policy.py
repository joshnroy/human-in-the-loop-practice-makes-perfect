"""Tossing3D's privileged solver: the two-skill sequence and its parameters.

The whole policy is a two-way branch on the symbolic state -- pick, then move-and-throw --
because the domain admits exactly one plan shape. It used to be a three-way branch; the
middle arm is gone because upstream fuses the base move and the throw into one controller.

## What makes it an oracle rather than a fixed script

The continuous parameters, which are the part a learner would have to find. **They are
now upstream's own draws rather than this repo's constants.** The oracle asks the
controller's own sampler, seeded deterministically, and takes the first draw.

That is a deliberate change of kind, and it is worth being explicit about what it costs
and buys:

- **It retires `ORACLE_THROW_STANDOFF = 1.35`, `ORACLE_RELEASE_SPEED_DEG_S = 140` and
  `ORACLE_GRIPPER_RELEASE_MS = 792`**, together with the `--oracle-throw-standoff` flag
  that drove the band-calibration sweeps. Those were "one operating point measured
  together" for a *split* move-and-throw; with the two fused, the standoff and the two
  toss dials are drawn as one four-vector by one sampler, and a hand-picked triple can no
  longer be substituted into it piecewise.
- **The sampler is the narrow one now.** `MoveToTossLocationAndTossController` documents
  its own bounds as measured: `SPEED_BOUNDS` from 115 deg/s to the ceiling and
  `RELEASE_MS_BOUNDS` of `(700, 840)`, narrowed from the originally-shipped
  `(600, 840)` because "every scoring draw fell in speed_deg [117.5, 140.0] and release_ms
  [710.4, 836.1]". So an arbitrary draw from it is a plausible throw, which is precisely
  what the old wide hitl bounds were not.
- **It is not a guaranteed solve.** The old oracle was a single measured point that scored
  `10/10` at its own standoff. This one draws from a band, so its success rate is the
  band's, not a point's. Nothing in this repo has measured that rate under the new pins,
  and this docstring does not claim one -- `test_the_oracle_solves_the_shipped_scene`
  exercises a fixed seed rather than asserting a rate.

## Every previously published Tossing3D number is stale

Every measured number in `docs/` was taken under the three-skill decomposition, the
pre-bump goal region, and hitl's own sampling bounds. None of the three still holds. The
affected entries carry staleness notes; nothing has been recomputed here.
"""

import numpy as np

from hitl_pmp.core.method.types import GroundSkill, LabeledAction
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Goal

from .environment import Tossing3DEnvironment
from .predicates import HOLDING, Tossing3DPredicates
from .skills import Tossing3DSkills

# The oracle's draws are reproducible, so a rollout is reproducible. Fixed rather than
# threaded from `--seed` because the oracle is a reference arm: it should behave the same
# whichever run is being compared against it.
ORACLE_PARAMETER_SEED = 123


class SkillOraclePolicy:
    """Picks the next skill from ground-truth state. A static-method container, never
    instantiated, same as every other business-logic class in this project."""

    @staticmethod
    def get_labeled_action(
        *,
        state: State,
        env: Tossing3DEnvironment,
        goal: Goal,
        parameter_seed: int = ORACLE_PARAMETER_SEED,
    ) -> LabeledAction:
        """The next privileged action toward the one goal this domain has.

        Goal-agnostic: unlike Tossing Room, whose state cannot distinguish a
        throw-recycling task from a throw-trash one, there is exactly one goal family here
        and the state says everything.

        The branch reads `Holding` off upstream's abstractor rather than a local
        classifier, so the oracle and the operator model cannot disagree about which
        skill's preconditions currently hold.
        """
        del goal
        abstraction = env.abstraction()
        holding = Tossing3DPredicates.get(abstraction=abstraction, name=HOLDING).holds(
            state, (env.robot, env.cube)
        )

        ground_skill: GroundSkill
        if holding:
            ground_skill = GroundSkill(
                skill=Tossing3DSkills.move_to_toss_location_and_toss(abstraction=abstraction),
                objects=(env.robot, env.cube, env.barrier),
            )
        else:
            # Covers both "hand empty, cube on the ground" and the unrecoverable state
            # after a missed toss. In the latter the grasp will fail to plan and
            # `take_action` records a no-op -- there is deliberately no fallback skill,
            # because there is nothing this domain can do to recover and pretending
            # otherwise would hide the irreversibility the domain exists to exhibit.
            ground_skill = GroundSkill(
                skill=Tossing3DSkills.pick_cube(abstraction=abstraction),
                objects=(env.robot, env.cube, env.barrier),
            )

        params = Tossing3DSkills.sample_params(
            ground_skill=ground_skill,
            rng=np.random.default_rng(parameter_seed),
            controllers=env.controllers(),
            state=state,
        )
        action = Tossing3DSkills.compute_action(
            ground_skill=ground_skill, params=params, state=state
        )
        objects_desc = ", ".join(obj.name for obj in ground_skill.objects)
        label = f"{ground_skill.skill.name}({objects_desc})"
        if params.size > 0:
            label += f", params={[round(float(value), 2) for value in params]}"
        return LabeledAction(action=action, label=label)
