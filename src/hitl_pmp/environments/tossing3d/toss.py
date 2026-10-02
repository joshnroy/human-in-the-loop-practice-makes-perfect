"""The composed toss's parameter pipeline, shared verbatim by both layouts.

`MoveToTossLocationAndToss` has three learned parameters, `[standoff, speed,
release]`. The fourth controller argument -- the rotation about the bin that decides
where the robot stands -- is chosen here per state by `TossDirectionSelector` and is
never sampled or learned. The barrier layout and the same-side layout both route the
toss through this one class (`SameSideSkills.TOSS is Tossing3DToss`), so their
dimension, bounds, proposal, direction selection, action encoding and classifier row
cannot drift apart.

## The classifier row is `[1, standoff, speed, release]`

It used to also carry the pre-move robot-frame bin displacement (forward, lateral)
and the yaw. The throw's physics depends only on the throw: the robot always ends up
exactly `standoff` from the bin, facing it, and the barrier was measured not to change
the throw's outcome by side. Where the robot happened to be *before* driving says
nothing about the landing, and it differs systematically between practice (same-side
bins) and evaluation (far-side bins), so it made evaluation rows look unlike practice
rows for no physical reason.
"""

from typing import Any, ClassVar

import numpy as np

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.types import Action, State

from .environment import Tossing3DEnvironment
from .kinder_backend import KinderBackend
from .toss_direction import (
    InfeasibleStandoffBandError,
    NoFeasibleTossDirectionError,
    TossDirectionChoice,
    TossDirectionSelector,
    TossStandoffBand,
)
from .wide_long_range_proposal import (
    WIDE_TOSS_RELEASE_MS_BOUNDS,
    WIDE_TOSS_SPEED_BOUNDS,
    WIDE_TOSS_STANDOFF_BOUNDS,
    WideLongRangeTossProposal,
)

# The name the chosen direction is logged under, beside the three learned parameters.
TOSS_DIRECTION_ANNOTATION = "toss_direction_deg"

# One key for every standoff no direction reaches, so the proposal log counts them
# together; the per-direction reasons are the selector's, not the sampler's.
NO_FEASIBLE_TOSS_DIRECTION_REJECTION = "no_feasible_toss_direction"

# A far bin whose corner-aware standoff band is empty or not one interval: no toss is
# available at that bin pose, so every proposal is rejected and the pool comes back
# empty -- the signal the practice planner replans around.
INFEASIBLE_STANDOFF_BAND_REJECTION = "infeasible_standoff_band"


class Tossing3DToss:
    """A static-method container, never instantiated."""

    PARAM_BOUNDS: ClassVar[tuple[tuple[float, float], ...]] = (
        WIDE_TOSS_STANDOFF_BOUNDS,
        WIDE_TOSS_SPEED_BOUNDS,
        WIDE_TOSS_RELEASE_MS_BOUNDS,
    )

    @staticmethod
    def sample_params(*, rng: np.random.Generator) -> np.ndarray:
        """The state-free draw: standoff over the full band."""
        return WideLongRangeTossProposal.sample(rng=rng)

    @staticmethod
    def sample_params_at_state(
        *, rng: np.random.Generator, ground_skill: GroundSkill, state: State
    ) -> np.ndarray:
        """The draw a decision uses: standoff over this bin's feasible band. A bin with
        no band still gets a draw, over the analytic far range, so the proposal is
        rejected by `rejection_reason` rather than raised out of the sampler."""
        try:
            bounds = Tossing3DToss.standoff_bounds(ground_skill=ground_skill, state=state)
        except InfeasibleStandoffBandError:
            bin_x = state.get(obj=ground_skill.objects[1], feature_name="x")
            bounds = WideLongRangeTossProposal.far_standoff_bounds(bin_x=bin_x)
        return WideLongRangeTossProposal.sample(rng=rng, standoff_bounds=bounds)

    @staticmethod
    def standoff_bounds(*, ground_skill: GroundSkill, state: State) -> tuple[float, float]:
        """The per-bin band at the bin's live pose, yaw included: the robot-side band
        for a bin on the robot's side (the full band unless the bin was moved), the
        far band for a bin across the barrier. A function of the state's geometry."""
        robot, bin_, _, barrier, *_ = ground_skill.objects
        bin_x = state.get(obj=bin_, feature_name="x")
        barrier_x = state.get(obj=barrier, feature_name="x")
        robot_x = state.get(obj=robot, feature_name="pos_base_x")
        far = (bin_x - barrier_x) * (robot_x - barrier_x) < 0.0
        snapshot = getattr(state, "object_centric", None)
        geometry = (
            None if snapshot is None else KinderBackend.toss_feasibility_geometry(snapshot=snapshot)
        )
        if geometry is None:
            # A hand-built state carries no collider geometry; only the analytic
            # stand line is knowable, and direction selection refuses such a state
            # anyway.
            if far:
                return WideLongRangeTossProposal.far_standoff_bounds(bin_x=bin_x)
            return WIDE_TOSS_STANDOFF_BOUNDS
        if far:
            return TossStandoffBand.far_bounds(geometry=geometry)
        return TossStandoffBand.robot_side_bounds(geometry=geometry)

    @staticmethod
    def choose_direction(*, state: State, params: np.ndarray) -> TossDirectionChoice:
        return TossDirectionSelector.select_for_state(state=state, standoff=float(params[0]))

    @staticmethod
    def rejection_reason(
        *, ground_skill: GroundSkill, state: State, params: np.ndarray
    ) -> str | None:
        """A standoff with a plannable direction is accepted; one with none is rejected
        like any other proposal. The band keeps these rare -- they are a moved bin's
        gaps and planner refusals -- and a pose with no feasible toss at all empties
        the pool, which the practice planner replans around. A far bin with no
        feasible band is such a pose, so it rejects every proposal."""
        try:
            Tossing3DToss.standoff_bounds(ground_skill=ground_skill, state=state)
        except InfeasibleStandoffBandError:
            return INFEASIBLE_STANDOFF_BAND_REJECTION
        try:
            Tossing3DToss.choose_direction(state=state, params=params)
        except NoFeasibleTossDirectionError:
            return NO_FEASIBLE_TOSS_DIRECTION_REJECTION
        return None

    @staticmethod
    def compute_action(*, params: np.ndarray, state: State) -> Action:
        standoff, speed, release = (float(value) for value in params)
        choice = Tossing3DToss.choose_direction(state=state, params=params)
        return np.array(
            [
                Tossing3DEnvironment.move_to_toss_location_and_toss_id,
                standoff,
                choice.rotation,
                speed,
                release,
            ],
            dtype=float,
        )

    @staticmethod
    def feature_row(*, params: np.ndarray) -> list[float]:
        return [1.0, *(float(value) for value in params)]

    @staticmethod
    def action_annotations(*, action: Any) -> dict[str, float]:
        return {TOSS_DIRECTION_ANNOTATION: float(round(float(np.degrees(action[2])), 6))}
