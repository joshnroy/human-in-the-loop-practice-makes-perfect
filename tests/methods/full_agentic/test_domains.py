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
from hitl_pmp.full_agentic.domains import FULL_AGENTIC_DOMAINS, Tossing3DFullAgentic
from hitl_pmp.full_agentic.runner import FullAgenticRunner

GOLDEN = Path(__file__).with_name("golden")
TOSSING = GOLDEN / "tossing3d"
HAS_KINDER = importlib.util.find_spec("kinder") is not None


def settings() -> SandboxSettings:
    return SandboxSettings(image="hitl-agentic-strict:20261002", robocode_checkout=Path("/rc"))


def parse(*extra: str) -> object:
    return FullAgenticRunner.parser().parse_args([
        "--output",
        "/unused",
        "--sandbox-settings",
        "/unused.json",
        *extra,
    ])


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
            FullAgenticRunner.parser().parse_args(["--output", "o", "--sandbox-settings", "s"])
        with pytest.raises(SystemExit):
            parse("--env", "lightswitch")
        assert parse("--env", "tossing3d").env == "tossing3d"
        assert FULL_AGENTIC_DOMAINS["tossing3d"] is Tossing3DFullAgentic

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

        env = Tossing3DFullAgentic.make_environment()
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
