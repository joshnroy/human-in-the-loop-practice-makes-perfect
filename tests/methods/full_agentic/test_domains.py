"""The Full-Agentic runner is selected by `--env`, and Tossing3D's run is unchanged.

The goldens under `golden/tossing3d/` are what the runner wrote *before* it was made
domain-generic (a dry run of `FullAgenticRunner.main` up to the coder transport, on the
pre-change commit), so these tests pin the Tossing3D prompts and protocol byte for byte.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from hitl_pmp.agentic_runtime.sandbox import SandboxSettings
from hitl_pmp.core.practice_costs import PracticeCosts
from hitl_pmp.full_agentic.domains import (
    FULL_AGENTIC_DOMAINS,
    SweepOffShelf3DFullAgentic,
    Tossing3DFullAgentic,
)
from hitl_pmp.full_agentic.runner import FullAgenticRunner

GOLDEN = Path(__file__).with_name("golden")
TOSSING = GOLDEN / "tossing3d"
HAS_KINDER = importlib.util.find_spec("kinder") is not None


def settings() -> SandboxSettings:
    return SandboxSettings(image="hitl-agentic-strict:20261002", robocode_checkout=Path("/rc"))


def parse(*extra: str) -> object:
    return FullAgenticRunner.parse_args(
        argv=[
            "--output",
            "/unused",
            "--sandbox-settings",
            "/unused.json",
            *extra,
        ]
    )


def rendered(*, args: object) -> tuple[str, str]:
    domain = FULL_AGENTIC_DOMAINS[args.env]
    costs = FullAgenticRunner.costs(args=args, domain=domain)
    protocol = FullAgenticRunner.protocol(
        args=args, costs=costs, settings=settings(), domain=domain
    )
    prompt = FullAgenticRunner.practice_prompt(
        domain=domain,
        costs=costs,
        budget=args.practice_step_budget,
        interval=args.measurement_interval_steps,
    )
    return json.dumps(protocol, indent=2, allow_nan=False), prompt


class TestDomainSelection:
    def test_env_is_a_required_choice_of_the_registered_domains(self) -> None:
        with pytest.raises(SystemExit):
            FullAgenticRunner.parse_args(argv=["--output", "o", "--sandbox-settings", "s"])
        with pytest.raises(SystemExit):
            parse("--env", "lightswitch")
        assert parse("--env", "tossing3d").env == "tossing3d"
        assert {
            "tossing3d": Tossing3DFullAgentic,
            "sweepoffshelf3d": SweepOffShelf3DFullAgentic,
        } == FULL_AGENTIC_DOMAINS

    def test_only_the_shelf_takes_the_humans_competence(self) -> None:
        with pytest.raises(SystemExit):
            parse("--env", "tossing3d", "--reset-cubes-success-probability", "0.5")
        args = parse("--env", "sweepoffshelf3d", "--reset-cubes-success-probability", "0.5")
        assert SweepOffShelf3DFullAgentic.environment_options(args=args) == {
            "reset_cubes_success_probability": 0.5
        }
        assert Tossing3DFullAgentic.environment_options(args=parse("--env", "tossing3d")) == {}

    def test_a_cost_config_must_price_exactly_the_domains_human_skills(
        self, *, tmp_path: Path
    ) -> None:
        path = tmp_path / "costs.json"
        path.write_text(PracticeCosts(human_skills={"reset_cube_far": {}}).model_dump_json())
        with pytest.raises(ValueError, match="human skills"):
            FullAgenticRunner.costs(
                args=parse("--env", "tossing3d", "--practice-cost-config", str(path)),
                domain=Tossing3DFullAgentic,
            )

    def test_every_domain_prices_each_of_its_interventions_by_default(self) -> None:
        for domain in FULL_AGENTIC_DOMAINS.values():
            costs = domain.default_costs()
            assert set(costs.human_skills) == set(domain.human_interventions)


class TestTossing3DUnchanged:
    def test_default_protocol_and_prompt_match_the_pre_generic_runner(self) -> None:
        protocol, prompt = rendered(args=parse("--env", "tossing3d"))
        assert protocol == (TOSSING / "protocol.json").read_text()
        assert prompt == (TOSSING / "task_prompt.md").read_text()

    def test_configured_costs_protocol_and_prompt_match_the_pre_generic_runner(
        self,
    ) -> None:
        protocol, prompt = rendered(
            args=parse(
                "--env",
                "tossing3d",
                "--practice-cost-config",
                str(GOLDEN / "costs_tossing.json"),
                "--hours",
                "2",
                "--seed",
                "3",
            )
        )
        assert protocol == (TOSSING / "protocol_costs.json").read_text()
        assert prompt == (TOSSING / "task_prompt_costs.md").read_text()

    def test_system_prompt_and_initial_files_match_the_pre_generic_runner(
        self, *, tmp_path: Path
    ) -> None:
        assert FullAgenticRunner.system_prompt() == (TOSSING / "system_prompt.md").read_text()

        class Bridge:
            def observe(self) -> dict[str, list[object]]:
                return {"objects": []}

        FullAgenticRunner.initial_files(
            workspace=tmp_path, bridge=Bridge(), domain=Tossing3DFullAgentic
        )
        assert (tmp_path / "robot_api.md").read_text() == (TOSSING / "robot_api.md").read_text()

    def test_human_counters_keep_their_order(self) -> None:
        assert list(Tossing3DFullAgentic.human_interventions.items()) == [
            ("reset_cube_and_bin_near", "robot_side"),
            ("reset_cube_far", "opposite_side"),
        ]


@pytest.mark.skipif(not HAS_KINDER, reason="needs KINDER")
@pytest.mark.xdist_group("full_agentic_tossing3d_sim")
class TestTossing3DBridgeUnchanged:
    def test_domain_bridge_observes_and_specifies_exactly_as_the_direct_bridge(
        self,
    ) -> None:
        from hitl_pmp.environments.tossing3d.agentic_bridge import Tossing3DAgenticBridge

        env = Tossing3DFullAgentic.make_environment(options={}, human_seed=0)
        try:
            env.hard_reset()
            direct = Tossing3DAgenticBridge(
                env=env, observation_mode="object_state", step_limit=2**63 - 1
            )
            via_domain = Tossing3DFullAgentic.make_bridge(env=env, practice=True)
            assert via_domain.step_limit == direct.step_limit
            assert json.dumps(via_domain.observe()) == json.dumps(direct.observe())
            assert json.dumps(via_domain.robot_spec()) == json.dumps(direct.robot_spec())
            evaluation = Tossing3DFullAgentic.make_bridge(env=env, practice=False)
            assert evaluation.step_limit == 1000  # the bridge's own default, as before
        finally:
            env.close()


class TestSweepOffShelf3DDomain:
    def test_interventions_are_the_shelf_humans_two_skills(self) -> None:
        assert SweepOffShelf3DFullAgentic.human_interventions == {
            "reset_cubes": "reset_cubes",
            "reset_wiper": "reset_wiper",
        }
        assert SweepOffShelf3DFullAgentic.default_costs() == PracticeCosts(
            human_skills={"reset_cubes": {}, "reset_wiper": {}}
        )

    def test_protocol_records_the_domain_after_every_shared_field(self) -> None:
        protocol = json.loads(rendered(args=parse("--env", "sweepoffshelf3d", "--seed", "4"))[0])
        tossing = json.loads(rendered(args=parse("--env", "tossing3d", "--seed", "4"))[0])
        assert list(protocol)[: len(tossing)] == list(tossing)
        assert protocol["env"] == "sweepoffshelf3d"
        assert protocol["environment"] == {"reset_cubes_success_probability": 1.0}
        assert set(protocol["cost_functions"]["human_skills"]) == {"reset_cubes", "reset_wiper"}

    def test_shelf_prompts_differ_from_tossing3d_only_where_listed(self) -> None:
        """Every line that differs is a line the PR lists for sign-off."""
        tossing = Tossing3DFullAgentic.task_prompt().splitlines()
        shelf = SweepOffShelf3DFullAgentic.task_prompt().splitlines()
        assert len(shelf) == len(tossing)
        assert [i for i, (a, b) in enumerate(zip(tossing, shelf, strict=True)) if a != b] == [
            0,
            1,
            36,
        ]
        tossing_api = Tossing3DFullAgentic.robot_api().read_text().splitlines()
        shelf_api = SweepOffShelf3DFullAgentic.robot_api().read_text().splitlines()
        assert [
            i for i, (a, b) in enumerate(zip(tossing_api, shelf_api, strict=True)) if a != b
        ] == [14]

    def test_shelf_prompt_names_only_its_own_human_skills_and_command(self) -> None:
        prompt = rendered(args=parse("--env", "sweepoffshelf3d"))[1]
        assert "`reset_cubes`" in prompt and "`reset_wiper`" in prompt
        assert "reset_cube_far" not in prompt and "barrier" not in prompt
        api = SweepOffShelf3DFullAgentic.robot_api().read_text()
        assert "11-element" in api and "schedule" not in api

    def test_configured_shelf_costs_replace_the_unit_equations(self, *, tmp_path: Path) -> None:
        path = tmp_path / "costs.json"
        path.write_text(
            PracticeCosts(
                human_skills={"reset_cubes": {"cost": {"value": 30}}, "reset_wiper": {}}
            ).model_dump_json()
        )
        prompt = rendered(
            args=parse("--env", "sweepoffshelf3d", "--practice-cost-config", str(path))
        )[1]
        assert '"value": 30.0' in prompt
        assert "both equal to 1 initially" not in prompt
        assert r"c_h(e) \equiv 1" not in prompt
