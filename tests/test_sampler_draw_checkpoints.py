"""Checkpointed sampler observers preserve files and discard only rolled-back writes."""

import pickle

import pytest

from hitl_pmp.sampler_draws import SAMPLER_PROPOSALS_FILENAME, SamplerDrawRecorder


def test_open_sampler_files_checkpoint_and_resume_without_duplicate_rows(*, tmp_path):
    recorder = SamplerDrawRecorder(output_path=tmp_path / "sampler_draws.jsonl")
    recorder._open().write("first draw\n")
    proposal = tmp_path / SAMPLER_PROPOSALS_FILENAME
    recorder._proposal_handle = proposal.open("w", encoding="utf-8")
    recorder._proposal_handle.write("first proposal\n")
    recorder.end_cycle()
    raw = pickle.dumps(recorder)
    recorder._open().write("discarded draw\n")
    recorder._open().flush()
    recorder._proposal_handle.write("discarded proposal\n")
    recorder._proposal_handle.flush()
    recorder._handle.close()
    recorder._proposal_handle.close()
    restored = pickle.loads(raw)
    # Merely inspecting a checkpoint must never mutate external files.
    assert "discarded" in recorder.output_path.read_text()
    restored.restore_checkpoint_files()
    assert restored._cycle == 1
    restored._open().write("second draw\n")
    restored._open().flush()
    restored._proposal_handle.write("second proposal\n")
    restored._proposal_handle.flush()
    assert recorder.output_path.read_text() == "first draw\nsecond draw\n"
    assert proposal.read_text() == "first proposal\nsecond proposal\n"


def test_initial_checkpoint_truncates_files_created_in_discarded_cycle(*, tmp_path):
    recorder = SamplerDrawRecorder(output_path=tmp_path / "sampler_draws.jsonl")
    raw = pickle.dumps(recorder)
    recorder.output_path.write_text("discarded\n")
    (tmp_path / SAMPLER_PROPOSALS_FILENAME).write_text("discarded\n")
    restored = pickle.loads(raw)
    restored.restore_checkpoint_files()
    assert recorder.output_path.read_bytes() == b""
    assert (tmp_path / SAMPLER_PROPOSALS_FILENAME).read_bytes() == b""


def test_resume_refuses_missing_checkpointed_sampler_rows(*, tmp_path):
    recorder = SamplerDrawRecorder(output_path=tmp_path / "sampler_draws.jsonl")
    recorder._open().write("kept\n")
    raw = pickle.dumps(recorder)
    recorder._handle.close()
    recorder.output_path.write_text("")
    restored = pickle.loads(raw)
    with pytest.raises(ValueError, match="shorter"):
        restored.restore_checkpoint_files()


@pytest.mark.parametrize("arm", ["ees", "curtis", "expectimax", "dp"])
def test_actual_symbolic_method_graph_checkpoints_with_open_recorder(*, arm, tmp_path):
    from types import SimpleNamespace

    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
    from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
    from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
    from hitl_pmp.methods.practice_makes_perfect.dp_method import DpMethod
    from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
    from hitl_pmp.step_protocol import CHECKPOINT_FILENAME, PracticeCheckpoint, RngStates

    env = Tossing3DEnvironment(scene_bg=False)
    provider = Tossing3DSkillProvider(env=env)
    recorder = SamplerDrawRecorder(output_path=tmp_path / "sampler_draws.jsonl")
    recorder._open().write("kept\n")
    kwargs = {"env": env, "skill_provider": provider, "draw_recorder": recorder, "seed": 13}
    if arm in {"curtis", "expectimax"}:
        method = Tossing3DPomdpMethod(
            **kwargs,
            pomdp_num_particles=32,
            pomdp_solver="determinized_astar" if arm == "curtis" else "expectimax",
            pomdp_determinized_variant="curtis" if arm == "curtis" else "algorithm3",
        )
        method._decision_index = 7
        method._cycle_index = 2
    elif arm == "dp":
        method = DpMethod(**kwargs, dp_budget=10, dp_solver="enumeration")
        method._dp_episodes = 3
    else:
        method = EesMethod(**kwargs)
    problem = SimpleNamespace(env=env)
    PracticeCheckpoint.save(
        output=tmp_path,
        live=PracticeCheckpoint.live_objects(problem=problem, skill_provider=provider),
        header={"next_cycle": 2},
        payload={"method": method},
    )
    fresh_env = Tossing3DEnvironment(scene_bg=False)
    fresh_provider = Tossing3DSkillProvider(env=fresh_env)
    fresh_problem = SimpleNamespace(env=fresh_env)
    restored = PracticeCheckpoint.load(
        path=tmp_path / CHECKPOINT_FILENAME,
        live=PracticeCheckpoint.live_objects(problem=fresh_problem, skill_provider=fresh_provider),
    )["method"]
    assert type(restored) is type(method)
    assert restored.env is fresh_env
    assert restored.skill_provider.env is fresh_env
    assert RngStates.capture(owner=restored) == RngStates.capture(owner=method)
    assert restored.draw_recorder._handle is None
    restored.draw_recorder.restore_checkpoint_files()
    assert restored.draw_recorder._open().tell() == len("kept\n")
    if arm in {"curtis", "expectimax"}:
        assert restored._pomdp_state.model_dump_json() == method._pomdp_state.model_dump_json()
        assert restored._decision_index == 7 and restored._cycle_index == 2
        assert type(restored.pomdp_planner) is type(method.pomdp_planner)
        assert RngStates.capture(owner=restored._pomdp_model) == RngStates.capture(
            owner=method._pomdp_model
        )
    if arm == "dp":
        assert restored._dp_episodes == 3
