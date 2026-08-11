"""`TraceLengths` turns `episode_traces.jsonl` sidecars into per-episode step counts, so
what is pinned here is the arithmetic that does that -- not matplotlib (same convention
`test_human_ladder_curves.py` states for its own sibling module).

Things that can go wrong silently, each with a test:

- **steps get grouped into the wrong episode.** Rows are keyed `(seed, checkpoint,
  task_index)`; a bug here would silently merge two different episodes' steps.
- **a non-contiguous step_index sequence is accepted.** A gap or duplicate means a line is
  missing or doubled, and must raise rather than silently under/over-count a trace length.
- **`solved` disagrees within one episode.** It is the whole episode's outcome and must be
  constant on every line (see `episode_traces.py`'s own docstring); a disagreement means
  two different episodes' rows were merged.
- **an unsolved episode leaks into the trace-length pool.** Its step count is
  `max_episode_steps()` by construction, not a real "steps to goal" -- `solved_steps` must
  exclude it.
- **a family with zero solved episodes crashes instead of reporting it.**
- **a floor cross-check that should fail silently passes**, and vice versa.

Expected values are derived on paper, not recorded from a run of the code.
"""

import json
from pathlib import Path

from analysis.practice_makes_perfect.episode_trace_lengths import TraceLengths

_TRASH = "TrashInBin(trash, trash_bin)"
_RECYCLING = "RecyclingInBin(recycling, recycling_bin)"


def _write_steps(*, path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _step(
    *,
    checkpoint: int = 0,
    task_index: int = 0,
    step_index: int,
    goal: str = _TRASH,
    solved: bool = True,
) -> dict:
    return {
        "checkpoint": checkpoint,
        "num_online_transitions": checkpoint * 150,
        "task_index": task_index,
        "goal": goal,
        "step_index": step_index,
        "solved": solved,
        "action_label": "Throw(robot, trash, trash_bin)",
        "action": [0.0, 0.0, 0.0],
        "state": {"trash.x": 1.0},
    }


def test_load_episode_lengths_groups_by_seed_checkpoint_and_task(*, tmp_path: Path) -> None:
    """Two episodes at different checkpoints must not be merged into one."""
    directory = tmp_path / "arm"
    _write_steps(
        path=directory / "0" / "episode_traces.jsonl",
        rows=[
            _step(checkpoint=0, task_index=0, step_index=0),
            _step(checkpoint=0, task_index=0, step_index=1),
            _step(checkpoint=5, task_index=0, step_index=0),
        ],
    )
    episodes = TraceLengths.load_episode_lengths(directory=directory)
    assert len(episodes) == 2
    by_checkpoint = {e["checkpoint"]: e["num_steps"] for e in episodes}
    assert by_checkpoint == {0: 2, 5: 1}


def test_load_episode_lengths_rejects_a_gap_in_step_index(*, tmp_path: Path) -> None:
    directory = tmp_path / "arm"
    _write_steps(
        path=directory / "0" / "episode_traces.jsonl",
        rows=[_step(step_index=0), _step(step_index=2)],
    )
    try:
        TraceLengths.load_episode_lengths(directory=directory)
        raise AssertionError("expected a ValueError for a non-contiguous step_index run")
    except ValueError:
        pass


def test_load_episode_lengths_rejects_solved_disagreeing_within_an_episode(
    *, tmp_path: Path
) -> None:
    directory = tmp_path / "arm"
    _write_steps(
        path=directory / "0" / "episode_traces.jsonl",
        rows=[
            _step(step_index=0, solved=False),
            _step(step_index=1, solved=True),
        ],
    )
    try:
        TraceLengths.load_episode_lengths(directory=directory)
        raise AssertionError("expected a ValueError for disagreeing solved values")
    except ValueError:
        pass


def test_solved_steps_excludes_unsolved_episodes(*, tmp_path: Path) -> None:
    directory = tmp_path / "arm"
    _write_steps(
        path=directory / "0" / "episode_traces.jsonl",
        rows=[
            # Solved in 3 steps -- every line of a solved episode carries solved=True
            # (record_episode's own contract, see episode_traces.py).
            _step(checkpoint=0, task_index=0, step_index=0, solved=True),
            _step(checkpoint=0, task_index=0, step_index=1, solved=True),
            _step(checkpoint=0, task_index=0, step_index=2, solved=True),
            # Unsolved, ran the full 12-step horizon.
            *[_step(checkpoint=0, task_index=1, step_index=i, solved=False) for i in range(12)],
        ],
    )
    episodes = TraceLengths.load_episode_lengths(directory=directory)
    steps = TraceLengths.solved_steps(episodes=episodes, family="TRASH")
    assert steps == [3], "the unsolved 12-step episode must not appear"
    solved, total = TraceLengths.solved_rate(episodes=episodes, family="TRASH")
    assert (solved, total) == (1, 2)


def test_solved_steps_is_empty_for_a_family_with_no_solved_episodes(*, tmp_path: Path) -> None:
    directory = tmp_path / "arm"
    _write_steps(
        path=directory / "0" / "episode_traces.jsonl",
        rows=[_step(goal=_RECYCLING, solved=False, step_index=0)],
    )
    episodes = TraceLengths.load_episode_lengths(directory=directory)
    assert TraceLengths.solved_steps(episodes=episodes, family="RECYCLING") == []
    assert TraceLengths.solved_rate(episodes=episodes, family="RECYCLING") == (0, 1)


def test_layout_is_two_way_only_for_the_two_way_ledge_arm() -> None:
    assert TraceLengths.layout(arm="two-way-ledge") == "two-way"
    assert TraceLengths.layout(arm="no-human") == "one-way"
    assert TraceLengths.layout(arm="N7") == "one-way"


def test_check_floors_against_data_flags_an_impossibly_short_solve() -> None:
    """A TRASH solve in 1 step is below the domain's real floor (5) -- this must be
    caught, not silently plotted as a real reference line violation."""
    episodes = [
        {
            "seed": 0,
            "checkpoint": 0,
            "task_index": 0,
            "family": "TRASH",
            "solved": True,
            "num_steps": 1,
        }
    ]
    problems = TraceLengths.check_floors_against_data(arms={"no-human": episodes})
    assert problems, "a below-floor observation must be reported"
    assert "no-human/TRASH" in problems[0]


def test_check_floors_against_data_passes_when_every_min_is_at_or_above_the_floor() -> None:
    episodes = [
        {
            "seed": 0,
            "checkpoint": 0,
            "task_index": 0,
            "family": "TRASH",
            "solved": True,
            "num_steps": 5,
        },
        {
            "seed": 0,
            "checkpoint": 1,
            "task_index": 0,
            "family": "TRASH",
            "solved": True,
            "num_steps": 8,
        },
    ]
    assert TraceLengths.check_floors_against_data(arms={"no-human": episodes}) == []


def test_render_smoke(*, tmp_path: Path) -> None:
    """Not pinning pixels (matching test_human_ladder_curves.py's own stated convention)
    -- just that a real render, across every family including one with zero solved
    episodes, does not raise."""
    directory = tmp_path / "no-human"
    _write_steps(
        path=directory / "0" / "episode_traces.jsonl",
        rows=[
            _step(checkpoint=0, task_index=0, step_index=0, goal=_TRASH, solved=True),
            _step(checkpoint=0, task_index=0, step_index=1, goal=_TRASH, solved=True),
            *[
                _step(checkpoint=0, task_index=1, step_index=i, goal=_RECYCLING, solved=False)
                for i in range(4)
            ],
        ],
    )
    arms = {"no-human": TraceLengths.load_episode_lengths(directory=directory)}
    output = tmp_path / "figure.png"
    TraceLengths.render(arms=arms, output=output, title="smoke test")
    assert output.exists()
