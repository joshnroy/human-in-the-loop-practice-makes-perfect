import argparse
import io
import json
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks
from hitl_pmp.loop_checkpoint import LoopCheckpoint
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_model import TOSS_SKILL
from hitl_pmp.planning.grounding import SkillGrounder
from hitl_pmp.training_checkpoint import SymbolPickler, SymbolUnpickler, TrainingCheckpoint


def test_learning_checkpoint_roundtrip_preserves_fitted_sampler_and_rng() -> None:
    env = Tossing3DEnvironment(scene_bg=False)
    method = Tossing3DPomdpMethod(
        env=env, skill_provider=Tossing3DSkillProvider(env=env), sampler_max_train_iters=2
    )
    atoms = SkillGrounder.all_possible_ground_atoms(
        objects=method.objects(), predicates=method.predicates()
    )
    toss = next(
        skill
        for skill in SkillGrounder.applicable_ground_skills(
            skills=method.skills(), objects=method.objects(), true_atoms=atoms
        )
        if skill.skill.name == TOSS_SKILL
    )
    for success in (True, False, True):
        method.observe_outcome(ground_skill=toss, success=success)
        method.observe_sampler_outcome(
            skill_name=TOSS_SKILL, param_dim=4, sampler_input=[float(success)], success=success
        )
    method.fit_samplers()
    buffer = io.BytesIO()
    SymbolPickler(buffer).dump(method.checkpoint_learning_state())
    buffer.seek(0)
    saved = SymbolUnpickler(file=buffer, method=method).load()
    restored = Tossing3DPomdpMethod(
        env=env, skill_provider=Tossing3DSkillProvider(env=env), sampler_max_train_iters=2
    )
    restored.restore_learning_state(state=saved)
    assert restored.pomdp_state == method.pomdp_state
    assert restored.total_observations() == method.total_observations()
    for _ in range(3):
        candidates = [np.zeros(4), np.ones(4)]
        assert restored.random_choice(ground_skills=[toss]) == method.random_choice(
            ground_skills=[toss]
        )
        actual = restored.sampler(skill_name=TOSS_SKILL, param_dim=4).sample(
            candidates=candidates, sampler_inputs=[[0.0], [1.0]], explore=True
        )
        expected = method.sampler(skill_name=TOSS_SKILL, param_dim=4).sample(
            candidates=candidates, sampler_inputs=[[0.0], [1.0]], explore=True
        )
        np.testing.assert_array_equal(actual.params, expected.params)


def _setup(*, path: Path) -> tuple[TrainingCheckpoint, Tossing3DPomdpMethod, Tossing3DProblem]:
    env = Tossing3DEnvironment(scene_bg=False)
    method = Tossing3DPomdpMethod(env=env, skill_provider=Tossing3DSkillProvider(env=env))
    problem = Tossing3DProblem(env=env, tasks=Tossing3DTasks(env=env, seed=7))
    writer = TrainingCheckpoint(
        path=path, args=argparse.Namespace(output_dir=None, seed=7, num_cycles=20)
    )
    return writer, method, problem


def _save(
    *, writer: TrainingCheckpoint, method: Tossing3DPomdpMethod, problem: Tossing3DProblem
) -> None:
    writer.save(
        progress=LoopCheckpoint(completed_cycles=0, num_online_transitions=0, test_tasks=[]),
        method=method,
        problem=problem,
        evaluation_problem=problem,
        metrics=Metrics(),
        runner_state={"planning": (4, 7), "practice": {}, "targets": {}},
    )


def test_atomic_checkpoint_preserves_previous_file_on_failed_replace(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer, method, problem = _setup(path=tmp_path / "checkpoint.pkl")
    _save(writer=writer, method=method, problem=problem)
    before = writer.path.read_bytes()

    def fail(*args: object) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr("hitl_pmp.training_checkpoint.os.replace", fail)
    with pytest.raises(OSError, match="disk failure"):
        _save(writer=writer, method=method, problem=problem)
    assert writer.path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [writer.path]
    assert writer.load(path=writer.path, method=method)["runner"]["planning"] == (4, 7)


@pytest.mark.parametrize("change", ["schema", "config", "versions", "sha256"])
def test_checkpoint_metadata_and_corruption_are_rejected_before_unpickling(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    writer, method, problem = _setup(path=tmp_path / "checkpoint.pkl")
    _save(writer=writer, method=method, problem=problem)
    header, payload = writer.path.read_bytes().split(b"\n", 1)
    metadata = json.loads(header)
    metadata[change] = "invalid"
    writer.path.write_bytes(json.dumps(metadata).encode() + b"\n" + payload)

    def fail(*args: object, **kwargs: object) -> None:
        pytest.fail("unpickler must not be reached with invalid metadata")

    monkeypatch.setattr("hitl_pmp.training_checkpoint.SymbolUnpickler", fail)
    with pytest.raises(ValueError):
        writer.load(path=writer.path, method=method)


def test_task_rng_and_learning_state_restore_into_fresh_instances(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer, method, problem = _setup(path=tmp_path / "checkpoint.pkl")
    problem.tasks.train_rng.random(5)
    problem.tasks.test_rng.random(3)
    _save(writer=writer, method=method, problem=problem)
    _, new_method, new_problem = _setup(path=writer.path)
    saved = writer.load(path=writer.path, method=new_method)
    # No simulator is needed to test task stream and learner restoration.
    monkeypatch.setattr(Tossing3DProblem, "hard_reset", lambda self: None)
    TrainingCheckpoint.restore(
        saved=saved,
        method=new_method,
        problem=new_problem,
        evaluation_problem=new_problem,
    )
    np.testing.assert_array_equal(
        problem.tasks.train_rng.random(20), new_problem.tasks.train_rng.random(20)
    )
    np.testing.assert_array_equal(
        problem.tasks.test_rng.random(20), new_problem.tasks.test_rng.random(20)
    )
    assert new_method.pomdp_state == method.pomdp_state


def test_resume_rejects_never_reset_without_starting_simulator(*, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scheduled"):
        Cli.main(
            argv=[
                "--env",
                "tossing3d",
                "--method",
                "pomdp",
                "--practice-reset-policy",
                "never",
                "--checkpoint",
                str(tmp_path / "checkpoint.pkl"),
            ]
        )


def test_resume_preserves_previous_output_directory(*, tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(b"sentinel")
    with pytest.raises(ValueError, match="new, empty"):
        Cli.main(
            argv=[
                "--env",
                "tossing3d",
                "--method",
                "pomdp",
                "--resume",
                str(checkpoint),
                "--output-dir",
                str(tmp_path),
            ]
        )
    assert checkpoint.read_bytes() == b"sentinel"
