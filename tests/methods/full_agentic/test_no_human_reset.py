"""``--no-human-reset`` removes the human interventions from the Full-Agentic arm.

With the flag off, the world refuses (and never charges) a human request, and no text
the agent is shown names a human skill, prices one, or says a human can reposition
the cube or the bin. With the flag at its default every rendering is unchanged.
"""

import itertools
import json
import tempfile
from pathlib import Path

import pytest

from hitl_pmp.core.practice_costs import ChargeFunction, HumanCharge, PracticeCosts
from hitl_pmp.full_agentic.runner import HERE, FullAgenticRunner
from hitl_pmp.full_agentic.world import World

HUMAN_TOKENS = (
    "human",
    "reset_cube_far",
    "reset_cube_and_bin_near",
    "request_help",
    "Hₕ",
    "cₕ",
    "dₕ",
    "C_H",
    "w_H",
    "c_h",
    "d_h",
    "H_h",
)
REQUIRED_ARGS = ["--output", "out", "--sandbox-settings", "settings.json"]
CUSTOM_COSTS = PracticeCosts(
    robot_step=ChargeFunction(value=2),
    human_skills={"reset_cube_far": HumanCharge(cost=ChargeFunction(value=7))},
)
RENDERINGS = list(
    itertools.product([PracticeCosts(), CUSTOM_COSTS], ["original", "new-wording"], [False, True])
)


def human_mentions(*, text: str) -> list[str]:
    lowered = text.lower()
    return [token for token in HUMAN_TOKENS if token.lower() in lowered]


class Bridge:
    def observe(self):
        return {"objects": []}


class FakeBridge:
    def __init__(self):
        self.calls = self.helps = 0
        self.env = self

    def observe(self):
        return {"calls": self.calls, "helps": self.helps}

    def action_spec(self):
        return {"low": [-1] * 18, "high": [1] * 18, "schedule_rows": 10}

    def step(self, *, action):
        self.calls += 1
        return self.observe()

    def reset_movables(self, *, destination):
        self.helps += 1
        return True


def test_flag_defaults_to_offering_the_human_reset():
    parser = FullAgenticRunner.parser()
    assert parser.parse_args(REQUIRED_ARGS).human_reset is True
    assert parser.parse_args([*REQUIRED_ARGS, "--human-reset"]).human_reset is True
    assert parser.parse_args([*REQUIRED_ARGS, "--no-human-reset"]).human_reset is False


@pytest.mark.parametrize("human_reset", [True, False])
def test_protocol_records_the_flag(*, human_reset):
    parser = FullAgenticRunner.parser()
    args = parser.parse_args(REQUIRED_ARGS + ([] if human_reset else ["--no-human-reset"]))
    protocol = FullAgenticRunner.protocol(args=args, costs=PracticeCosts())
    assert protocol["human_reset"] is human_reset
    json.dumps(protocol)


@pytest.mark.parametrize(("costs", "variant", "fragile"), RENDERINGS)
def test_rendered_prompts_without_human_reset_never_mention_a_human(*, costs, variant, fragile):
    task, system = FullAgenticRunner.rendered_prompts(
        costs=costs,
        variant=variant,
        fragile=fragile,
        mat_size=1,
        practice_step_budget=85000,
        measurement_interval_steps=1700,
        human_reset=False,
    )
    assert human_mentions(text=task) == []
    assert human_mentions(text=system) == []
    # The parts that must survive the redaction.
    assert "C = C_R" in task
    assert "robot control periods" in task
    assert "Decision summary:" in task
    assert "class GeneratedApproach" in task
    assert "total budget of 85000 counted steps" in task
    assert "frozen controller runs autonomously" in system
    if fragile:
        assert "cube is fragile" in task and "C_D" in task
    if variant == "new-wording":
        assert "When choosing a practice action" in task
    if costs != PracticeCosts():
        assert '"value": 2.0' in task
        assert '"value": 7.0' not in task


@pytest.mark.parametrize(("costs", "variant", "fragile"), RENDERINGS)
def test_rendered_prompts_with_human_reset_still_describe_it(*, costs, variant, fragile):
    task, system = FullAgenticRunner.rendered_prompts(
        costs=costs,
        variant=variant,
        fragile=fragile,
        mat_size=1,
        practice_step_budget=85000,
        measurement_interval_steps=1700,
        human_reset=True,
    )
    assert "reset_cube_far" in task and "reset_cube_and_bin_near" in task
    assert "hint: in order to minimize human cost" in task
    assert "human-skill invocations incur the specified costs" in system
    if costs != PracticeCosts():
        assert '"value": 7.0' in task


def test_default_rendering_is_the_raw_prompt_files():
    task, system = FullAgenticRunner.rendered_prompts(
        costs=PracticeCosts(),
        variant="original",
        fragile=False,
        mat_size=1,
        practice_step_budget=85000,
        measurement_interval_steps=1700,
        human_reset=True,
    )
    assert task.startswith((HERE / "task_prompt.md").read_text())
    assert system == (HERE / "system_prompt.md").read_text()


def test_shipped_files_without_human_reset_carry_no_human_api(*, tmp_path):
    FullAgenticRunner.initial_files(workspace=tmp_path, bridge=Bridge(), human_reset=False)
    assert {p.name for p in tmp_path.iterdir()} == {
        "env_client.py",
        "robot_api.md",
        "initial_observation.json",
    }
    for name in ("env_client.py", "robot_api.md"):
        assert human_mentions(text=(tmp_path / name).read_text()) == [], name
    client = (tmp_path / "env_client.py").read_text()
    assert "def step(" in client and "def finish_adaptation(" in client
    compile(client, "env_client.py", "exec")


def test_shipped_files_with_human_reset_are_the_raw_files(*, tmp_path):
    FullAgenticRunner.initial_files(workspace=tmp_path, bridge=Bridge(), human_reset=True)
    for name in ("env_client.py", "robot_api.md"):
        assert (tmp_path / name).read_bytes() == (HERE / name).read_bytes()


def make_world(*, root: Path, bridge, human_reset: bool) -> World:
    workspace = root / "workspace"
    workspace.mkdir()
    (workspace / "approach.py").write_text("class GeneratedApproach: pass")
    return World(
        bridge=bridge,
        workspace=workspace,
        output=root,
        deadline=float("inf"),
        human_reset=human_reset,
    )


@pytest.mark.parametrize("intervention", ["reset_cube_far", "reset_cube_and_bin_near"])
def test_world_refuses_and_does_not_charge_a_human_request(*, intervention):
    with tempfile.TemporaryDirectory() as temporary:
        bridge = FakeBridge()
        world = make_world(root=Path(temporary), bridge=bridge, human_reset=False)
        try:
            world.configure_measurements(budget=10, interval=10)
            request = dict(
                operation="request_help",
                request_id="h1",
                expected_version=world.version,
                intervention_id=intervention,
            )
            with pytest.raises(ValueError, match="not offered"):
                world.dispatch(request=request)
            assert bridge.helps == 0
            assert world.help_requests == 0
            assert world.counted_steps == 0
            assert world.accounting.human_cost == 0
            assert world.accounting.human_invocations == {}
            assert world.human_by_side == {"robot_side": 0, "opposite_side": 0}
            assert not world.uncertain
            assert world.version == 0
            assert world.db.execute("SELECT COUNT(*) FROM receipts").fetchone()[0] == 0
            assert not (Path(temporary) / "practice.jsonl").exists()
            # Practice continues normally afterwards.
            step = dict(
                operation="step", request_id="s1", expected_version=world.version, action=[0] * 18
            )
            assert "result" in world.dispatch(request=step)
            assert world.observation()["human_skill_quotes"] == {}
        finally:
            world.db.close()


def test_world_with_human_reset_still_quotes_and_executes_the_skills():
    with tempfile.TemporaryDirectory() as temporary:
        bridge = FakeBridge()
        world = make_world(root=Path(temporary), bridge=bridge, human_reset=True)
        try:
            assert set(world.observation()["human_skill_quotes"]) == {
                "reset_cube_far",
                "reset_cube_and_bin_near",
            }
            request = dict(
                operation="request_help",
                request_id="h1",
                expected_version=world.version,
                intervention_id="reset_cube_far",
            )
            assert "result" in world.dispatch(request=request)
            assert bridge.helps == 1
        finally:
            world.db.close()
