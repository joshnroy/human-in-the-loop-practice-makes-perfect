import numpy as np

from hitl_pmp.environments.tossing3d.descriptions import Tossing3DDescriptions
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.toss import Tossing3DToss
from hitl_pmp.methods.practice_makes_perfect.coded_sampler.workspace import SamplerInputLayout


def test_every_described_type_lists_exactly_its_features_in_order() -> None:
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment())
    for type_ in (*provider.types(), Tossing3DEnvironment.scene_type):
        specs = Tossing3DDescriptions.FEATURES[type_.name]
        assert tuple(spec.name for spec in specs) == type_.feature_names


def test_toss_parameter_bounds_are_the_proposals_own() -> None:
    specs = Tossing3DDescriptions.TOSS_PARAMETERS
    assert [(s.lower, s.upper) for s in specs] == [tuple(b) for b in Tossing3DToss.PARAM_BOUNDS]
    assert len(specs) == Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS.param_dim


def test_the_described_toss_row_matches_the_row_the_method_builds() -> None:
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment())
    skill = Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS
    description = SamplerInputLayout.describe(provider=provider, skill=skill)
    params = np.array([1.5, 300.0, 600.0])
    row = Tossing3DToss.feature_row(params=params)
    fields = description["sampler_input"]
    assert len(fields) == len(row)
    assert [f["name"] for f in fields] == ["bias", "param.standoff", "param.speed", "param.release"]
    assert row[1:] == [float(p) for p in params]
    assert description["layout"] == "hand_selected"


def test_the_task_description_carries_the_codes_numbers() -> None:
    text = Tossing3DDescriptions.task()
    assert "InBin(cube_0, bin_0)" in text
    assert "one-way door" in text.lower()
