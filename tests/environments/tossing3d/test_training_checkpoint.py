"""A real MuJoCo interrupted/resumed run, not just a serialization round trip."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.training_checkpoint import TrainingCheckpoint

pytestmark = pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KINDER")


def test_interrupted_resume_matches_uninterrupted_tossing3d(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flags = [
        "--env",
        "tossing3d",
        "--method",
        "pomdp",
        "--seed",
        "1",
        "--no-scene-bg",
        "--num-cycles",
        "2",
        "--num-test-tasks",
        "1",
        "--max-steps-per-interaction",
        "2",
        "--sampler-max-train-iters",
        "2",
        "--pomdp-horizon",
        "2",
    ]
    complete = tmp_path / "complete.pkl"
    interrupted = tmp_path / "interrupted.pkl"
    resumed = tmp_path / "resumed.pkl"
    Cli.main(argv=[*flags, "--checkpoint", str(complete)])
    original_save = TrainingCheckpoint.save

    def interrupt(self: TrainingCheckpoint, **kwargs: Any) -> None:  # noqa: PLR0917
        original_save(self, **kwargs)
        if kwargs["progress"].completed_cycles == 1:
            raise InterruptedError("simulate crash after one complete cycle")

    with monkeypatch.context() as patch:
        patch.setattr(TrainingCheckpoint, "save", interrupt)
        with pytest.raises(InterruptedError):
            Cli.main(argv=[*flags, "--checkpoint", str(interrupted)])
    Cli.main(argv=[*flags, "--checkpoint", str(resumed), "--resume", str(interrupted)])

    env = Tossing3DEnvironment(scene_bg=False)
    method = Tossing3DPomdpMethod(env=env, skill_provider=Tossing3DSkillProvider(env=env))
    args = Cli.parse_args(argv=flags)
    writer = TrainingCheckpoint(path=resumed, args=args)
    expected = writer.load(path=complete, method=method)
    actual = writer.load(path=resumed, method=method)
    assert actual["progress"].completed_cycles == expected["progress"].completed_cycles == 2
    assert actual["progress"].num_online_transitions == expected["progress"].num_online_transitions
    assert actual["metrics"].model_dump_json() == expected["metrics"].model_dump_json()
    assert actual["method"]["_pomdp_state"] == expected["method"]["_pomdp_state"]
    assert actual["method"]["_all_attempt_outcomes"] == expected["method"]["_all_attempt_outcomes"]
    assert actual["train_rng"] == expected["train_rng"]
    assert actual["test_rng"] == expected["test_rng"]
    assert (
        actual["method"]["_rng"].bit_generator.state
        == expected["method"]["_rng"].bit_generator.state
    )
