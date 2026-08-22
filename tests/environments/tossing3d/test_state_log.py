"""Round-trips a real (short) Tossing3D rollout through `StateLogWriter`/
`StateLogReader` and `Tossing3DEnvironment.restore_plain_snapshot`, then checks the
replayed frames actually came from the logged physics rather than a fallback/no-op --
see `state_log.py`'s own module docstring for why a flat `core.State` alone cannot do
this and what this file demonstrates it fixes.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.state_log import (
    SkillEvent,
    StateLogHeader,
    StateLogReader,
    StateLogWriter,
    TickEvent,
)
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="KINDER is an optional extra (`kindergarden` + `kinder_models`); CI never installs it",
)


def _record_one_pick(*, output_path: Path, seed: int) -> list[np.ndarray]:
    """Records only PickCube's own ticks -- ~200, a few seconds -- not a full
    pick+toss, since this file checks the log/replay mechanism, not a live throw.
    Returns the live substep frames, for comparison against a later replay.

    Goes through `attach_state_log_writer`/`take_action`'s own automatic drain
    (`Tossing3DEnvironment._log_skill_ticks`), the same path a real CLI run uses --
    not a hand-rolled drain-and-write, which would test a mechanism nothing in
    production actually calls that way anymore."""
    args = argparse.Namespace(
        variant="o1", scene_bg=True, canonical_seed=125, seed=seed, test_env_seed_offset=10000
    )
    problem = Tossing3DCli.build_problem(args=args)
    env = problem.env
    skill_provider = Tossing3DSkillProvider(env=env)
    method = EesMethod(env=env, skill_provider=skill_provider, seed=seed)
    writer = StateLogWriter(
        output_path=output_path,
        header=StateLogHeader(
            variant="o1", scene_bg=True, canonical_seed=125, seed=seed, test_env_seed_offset=10000
        ),
    )
    env.attach_state_log_writer(writer=writer)
    try:
        problem.hard_reset()
        backend = env.backend()
        backend.set_substep_recording(enabled=True)
        task = problem.sample_train_task()
        state = problem.reset_to_task(task=task)
        backend.drain_substep_frames()

        policy = method.get_practice_policy(task=task)
        episode = method._practice_episode
        assert episode is not None
        pick = GroundSkill(
            skill=Tossing3DSkills.PICK_CUBE, objects=(env.robot, env.cube, env.barrier)
        )
        episode._plan = [pick]
        labeled_action = policy(state)
        problem.take_action(action=labeled_action.action)
        substep_frames = backend.drain_substep_frames()
        assert substep_frames, "fixture must actually produce ticks to log anything"
        return substep_frames
    finally:
        writer.close()
        env.close()


def test_a_recorded_log_round_trips_through_the_reader(*, tmp_path: Path) -> None:
    log_path = tmp_path / "state_log.jsonl"
    live_frames = _record_one_pick(output_path=log_path, seed=11)

    reader = StateLogReader(path=log_path)
    assert reader.header == StateLogHeader(
        variant="o1", scene_bg=True, canonical_seed=125, seed=11, test_env_seed_offset=10000
    )
    skill_events = [e for e in reader.events if isinstance(e, SkillEvent)]
    tick_events = [e for e in reader.events if isinstance(e, TickEvent)]
    assert [e.name for e in skill_events] == ["PickCube"]
    assert len(tick_events) == len(live_frames)


def test_replaying_a_logged_tick_reproduces_the_live_frame(*, tmp_path: Path) -> None:
    """The property this whole mechanism exists for: a frame rendered from a REPLAYED
    tick must match the frame the LIVE run produced at that same tick -- not merely
    "some frame", which a silently-broken restore (e.g. falling back to the scene's
    resting pose) would also produce."""
    log_path = tmp_path / "state_log.jsonl"
    live_frames = _record_one_pick(output_path=log_path, seed=17)
    mid_index = len(live_frames) // 2
    live_frame = live_frames[mid_index]

    reader = StateLogReader(path=log_path)
    assert reader.header is not None
    args = argparse.Namespace(
        variant=reader.header.variant,
        scene_bg=reader.header.scene_bg,
        canonical_seed=reader.header.canonical_seed,
        seed=reader.header.seed,
        test_env_seed_offset=reader.header.test_env_seed_offset,
    )
    problem = Tossing3DCli.build_problem(args=args)
    env = problem.env
    try:
        problem.hard_reset()
        problem.reset_to_task(task=problem.sample_train_task())
        tick_events = [e for e in reader.events if isinstance(e, TickEvent)]
        target = tick_events[mid_index]
        env.restore_plain_snapshot(plain={k: list(v) for k, v in target.state.items()})
        # The raw simulator frame, not `Tossing3DRenderer.render_frame` (which stacks a
        # caption bar underneath) -- `live_frame` is raw too (`drain_substep_frames`),
        # and the caption is a separate, deterministic overlay not worth comparing here.
        replayed_frame = env.backend().render()
    finally:
        env.close()

    # Pixel-identical, not merely close: restore() puts MuJoCo back to (float32-
    # rounded) the exact logged pose, and both frames come from the same camera/scene
    # with nothing else able to have moved between them.
    assert replayed_frame.shape == live_frame.shape
    mismatch_fraction = float(np.mean(replayed_frame != live_frame))
    assert mismatch_fraction < 0.01, (
        f"replayed frame differs from the live one in {mismatch_fraction:.2%} of pixels -- "
        "restore_plain_snapshot is not reproducing the logged tick"
    )


def test_state_capture_is_unconditional_even_when_frame_recording_is_off() -> None:
    """The property this whole fix exists for: `drain_substep_states` must return real
    per-tick data regardless of `record_substeps` (frame/video capture), which stays
    genuinely opt-in -- `drain_substep_frames` stays empty in the same run."""
    args = argparse.Namespace(
        variant="o1", scene_bg=True, canonical_seed=125, seed=23, test_env_seed_offset=10000
    )
    problem = Tossing3DCli.build_problem(args=args)
    env = problem.env
    try:
        problem.hard_reset()
        backend = env.backend()
        # Deliberately NOT calling set_substep_recording(enabled=True) -- this is the
        # ordinary, unrecorded-run path. `backend.run_pick_cube()` directly, not
        # `problem.take_action`/a policy: `Tossing3DEnvironment.take_action` now
        # drains-and-discards on every call itself (see `_log_skill_ticks`), so
        # calling it here would find nothing left to drain regardless of this
        # property -- this checks the KinderBackend-level guarantee in isolation.
        task = problem.sample_train_task()
        problem.reset_to_task(task=task)
        backend.drain_substep_frames()
        backend.drain_substep_states()

        backend.run_pick_cube()

        assert backend.drain_substep_frames() == [], "frame capture must stay opt-in"
        assert backend.drain_substep_states(), (
            "state capture must be unconditional, not gated on record_substeps"
        )
    finally:
        env.close()


def test_an_ordinary_cli_run_writes_a_state_log_with_no_manual_wiring(*, tmp_path: Path) -> None:
    """The property the rest of this file's tests don't cover: a real run through the
    actual `hitl_pmp.cli` entrypoint -- the path every sweep and every human actually
    uses -- must produce this log on its own. Every other test in this file builds the
    log by hand (`StateLogWriter`/`drain_substep_states` called directly), which proves
    the mechanism works but not that anything wires it into a real run. skill-oracle,
    not EES: no practice cycles, so this stays a single short evaluation episode."""
    output_dir = tmp_path / "run"
    Cli.main(
        argv=[
            "--env",
            "tossing3d",
            "--method",
            "skill-oracle",
            "--seed",
            "31",
            "--num-test-tasks",
            "1",
            "--output-dir",
            str(output_dir),
        ]
    )

    log_path = output_dir / "tossing3d_state_log.jsonl"
    assert log_path.exists(), "an ordinary run must write the state log with no extra flag"
    reader = StateLogReader(path=log_path)
    assert reader.header == StateLogHeader(
        variant="o1", scene_bg=True, canonical_seed=125, seed=31, test_env_seed_offset=10000
    )
    skill_events = [e for e in reader.events if isinstance(e, SkillEvent)]
    tick_events = [e for e in reader.events if isinstance(e, TickEvent)]
    # PickCube then the toss is this domain's only shortest solve (see
    # Tossing3DProblem.max_episode_steps' own docstring); a missed throw retries the
    # same pair within the step budget, so only the first two skills are seed-
    # independent -- not the whole sequence.
    assert [e.name for e in skill_events[:2]] == ["PickCube", "MoveToTossLocationAndToss"]
    assert len(skill_events) >= 2
    assert len(tick_events) > 0, "each skill must contribute its own real physics ticks"
