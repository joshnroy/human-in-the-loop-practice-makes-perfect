"""Proposal integration without a simulator or access to live practice state."""

import numpy as np

from hitl_pmp.cli import Cli
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.recovery_skills import SameSideSkills
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import (
    TOSS_DISTANCE_BOUNDS,
    TOSS_RELEASE_MS_BOUNDS,
    TOSS_ROTATION_BOUNDS,
    TOSS_SPEED_BOUNDS,
    Tossing3DSkills,
)


def test_proposal_is_reproducible_and_within_controller_action_bounds(*, no_kinder_import) -> None:
    del no_kinder_import
    env = Tossing3DEnvironment()
    skill = GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier),
    )
    provider = Tossing3DSkillProvider(env=env, toss_proposal="long-range")
    rng = np.random.default_rng(50)
    repeat_rng = np.random.default_rng(50)
    bounds = np.array([
        TOSS_DISTANCE_BOUNDS,
        TOSS_ROTATION_BOUNDS,
        TOSS_SPEED_BOUNDS,
        TOSS_RELEASE_MS_BOUNDS,
    ])
    samples = np.stack([provider.sample_params(ground_skill=skill, rng=rng) for _ in range(100)])
    repeated = np.stack([
        provider.sample_params(ground_skill=skill, rng=repeat_rng) for _ in range(100)
    ])
    assert np.array_equal(samples, repeated)
    assert np.all(np.isfinite(samples))
    assert np.all(samples >= bounds[:, 0])
    assert np.all(samples <= bounds[:, 1])
    assert np.ptp(samples[:, 2]) > 0.0
    assert np.ptp(samples[:, 3]) > 0.0


def test_default_and_same_side_keep_existing_sampling_contracts() -> None:
    for layout, mode in (
        (Tossing3DLayout.BARRIER, "independent"),
        (Tossing3DLayout.SAME_SIDE, "long-range"),
    ):
        env = Tossing3DEnvironment(layout=layout)
        provider = Tossing3DSkillProvider(env=env, toss_proposal=mode)
        skill = GroundSkill(
            skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
            objects=(env.robot, env.bin, env.cube, env.barrier),
        )
        old_sampler = SameSideSkills if layout == Tossing3DLayout.SAME_SIDE else Tossing3DSkills
        rng = np.random.default_rng(40)
        previous_rng = np.random.default_rng(40)
        for _ in range(10):
            assert np.array_equal(
                provider.sample_params(ground_skill=skill, rng=rng),
                old_sampler.sample_params(ground_skill=skill, rng=previous_rng),
            )


def test_cli_records_explicit_proposal_choice() -> None:
    args = Cli.parse_args(
        argv=["--env", "tossing3d", "--method", "ees", "--toss-proposal", "long-range"]
    )
    assert args.toss_proposal == "long-range"
    defaults = Cli.parse_args(argv=["--env", "tossing3d", "--method", "ees"])
    assert defaults.toss_proposal == "independent"
