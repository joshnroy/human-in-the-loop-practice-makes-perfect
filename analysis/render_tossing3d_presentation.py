"""Re-render a retained Tossing3D run as the canonical presentation dashboard.

The output deliberately matches the established 1920x640 practice-video layout:
run status across the top, the simulator on the left, a rolling skill history, belief
bars, and the current planner decision on the right.  Both inputs are ordinary run
artifacts, so this does not repeat the experiment::

    scripts/with_env.sh python analysis/render_tossing3d_presentation.py \
        --run artifacts/my-run --output artifacts/my-run/presentation.mp4

Realistic scene assets are the default.  Parameterized human resets additionally show
their bound bin destination (robot side or opposite side) as a large, persistent label.
"""

# Drawing primitives are naturally called with their primary value positionally.
# ruff: noqa: PLR0917

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hitl_pmp.core.renderer.renderer import VideoStream
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.state_log import StateLogHeader

WIDTH = 1920
HEIGHT = 640
TOP_HEIGHT = 48
SCENE_WIDTH = 640
SCENE_HEIGHT = 480
SCENE_TOP = TOP_HEIGHT
HISTORY_LEFT = 656
COMPETENCE_LEFT = 960
LEARNING_RATE_LEFT = 1280
DECISION_LEFT = 1600
COLUMN_WIDTH = 288
FPS_HOLD_SECONDS = 1.0

BACKGROUND = "#0f0d1c"
PANEL = "#151225"
TRACK = "#2b2941"
TEXT = "#f2f0f7"
MUTED = "#a39eb2"
PURPLE = "#9a6be8"
BLUE = "#45a9e8"
GREEN = "#51e299"
AMBER = "#eda015"

SKILLS = (
    "PickCube",
    "MoveToTossLocationAndToss",
    "OpenGripper",
    "ask_for_reset_cube_bin_only",
)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / filename
    return ImageFont.truetype(str(path), size)


def _timestamp(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _short_skill(name: str) -> str:
    return {
        "MoveToTossLocationAndToss": "Toss",
        "ask_for_reset_cube_bin_only": "Human reset",
    }.get(name, name)


def reset_destination(decision: dict[str, Any] | None) -> str | None:
    """Return the human-readable bound bin destination from a logged action."""
    if decision is None:
        return None
    action = decision.get("action")
    if not isinstance(action, dict):
        return None
    skill = action.get("skill")
    if not isinstance(skill, dict) or skill.get("name") != "ask_for_reset_cube_bin_only":
        return None
    objects = action.get("objects")
    if not isinstance(objects, list) or not objects:
        return None
    destination = objects[-1]
    if not isinstance(destination, dict):
        return None
    return {
        "robot_side": "ROBOT SIDE",
        "opposite_side": "OPPOSITE SIDE",
    }.get(str(destination.get("name")))


def _mean(decision: dict[str, Any] | None, *, field: str, skill: str) -> float | None:
    if decision is None:
        return None
    values = decision.get(field)
    if not isinstance(values, dict):
        return None
    for key in (f"{skill} (belief mean)", skill):
        if key in values:
            return float(values[key])
    return None


def _action_name(decision: dict[str, Any] | None) -> str | None:
    if decision is None:
        return None
    action = decision.get("action")
    if not isinstance(action, dict):
        return None
    skill = action.get("skill")
    return str(skill["name"]) if isinstance(skill, dict) and "name" in skill else None


def _draw_bar(
    draw: ImageDraw.ImageDraw,
    *,
    left: int,
    top: int,
    value: float | None,
    color: str,
    maximum: float = 1.0,
) -> None:
    right = left + 224
    draw.rectangle((left, top, right, top + 17), fill=TRACK)
    if value is None:
        draw.text((left, top + 23), "N/A", font=_font(12), fill=MUTED)
        return
    fraction = min(1.0, max(0.0, value / maximum)) if maximum > 0 else 0.0
    draw.rectangle((left, top, left + round(224 * fraction), top + 17), fill=color)
    draw.text((left, top + 23), f"{value:.4f}", font=_font(12), fill=color)


def _draw_metric_column(
    draw: ImageDraw.ImageDraw,
    *,
    left: int,
    title: str,
    field: str,
    decision: dict[str, Any] | None,
    color: str,
) -> None:
    draw.text((left, 65), title, font=_font(15, bold=True), fill=PURPLE)
    top = 105
    for skill in SKILLS:
        draw.text((left, top), _short_skill(skill), font=_font(15), fill=TEXT)
        _draw_bar(
            draw,
            left=left,
            top=top + 27,
            value=_mean(decision, field=field, skill=skill),
            color=color,
        )
        top += 116


def _draw_decision_column(draw: ImageDraw.ImageDraw, *, decision: dict[str, Any] | None) -> None:
    left = DECISION_LEFT
    draw.text((left, 65), "DECISION VALUES", font=_font(15, bold=True), fill=PURPLE)
    selected = _action_name(decision)
    selected_value = float(decision["value"]) if decision is not None else None
    stop_value: float | None = None
    if decision is not None:
        for event in decision.get("search", []):
            if event.get("event") == "stop_value" and int(event.get("node", -1)) == 0:
                stop_value = float(event["value"])
                break
    draw.text((left, 105), "Selected action", font=_font(15), fill=TEXT)
    draw.text(
        (left, 133),
        "STOP" if selected is None else _short_skill(selected),
        font=_font(20, bold=True),
        fill=GREEN,
    )
    _draw_bar(draw, left=left, top=170, value=selected_value, color=GREEN)
    draw.text((left, 232), "STOP", font=_font(15), fill=TEXT)
    _draw_bar(draw, left=left, top=260, value=stop_value, color=PURPLE)


def _state_xyz(state: dict[str, list[float]], name: str) -> tuple[float, float, float]:
    values = state[name]
    return float(values[0]), float(values[1]), float(values[2])


def _compose(
    *,
    frame: np.ndarray,
    state: dict[str, list[float]],
    seed: int,
    cycle: int,
    step: int,
    transitions: int,
    current_skill: str,
    current_objects: tuple[str, ...],
    history: list[tuple[str, str | None]],
    decision: dict[str, Any] | None,
    samples: int,
    horizon: int,
) -> np.ndarray:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    scene = Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8)).resize(
        (SCENE_WIDTH, SCENE_HEIGHT), Image.Resampling.LANCZOS
    )
    image.paste(scene, (0, SCENE_TOP))
    draw = ImageDraw.Draw(image)

    destination = (
        reset_destination(decision) if current_skill == "ask_for_reset_cube_bin_only" else None
    )
    active_color = GREEN if destination else AMBER
    draw.rectangle((0, 0, WIDTH, TOP_HEIGHT - 1), fill="#10141a")
    draw.rectangle((0, 0, 108, TOP_HEIGHT - 1), fill=active_color)
    draw.text((54, 24), "PRACTICE", font=_font(17, bold=True), fill="white", anchor="mm")
    status = (
        f"SEED {seed}    SAMPLES {samples}    H {horizon if horizon else '—'}    "
        f"CYCLE {cycle + 1}/10    STEP {step}    TRANSITIONS {transitions}"
    )
    draw.text((122, 16), status, font=_font(13), fill=TEXT)
    draw.text((122, 32), f"TASK {_short_skill(current_skill)}", font=_font(13), fill=MUTED)
    if destination:
        label = f"HUMAN RESET → BIN TO {destination}"
        draw.rounded_rectangle((1240, 7, 1900, 41), radius=8, fill="#123c30")
        draw.text((1570, 24), label, font=_font(18, bold=True), fill=GREEN, anchor="mm")

    draw.rectangle((SCENE_WIDTH, TOP_HEIGHT, WIDTH, HEIGHT), fill=PANEL)
    draw.line((SCENE_WIDTH, TOP_HEIGHT, SCENE_WIDTH, HEIGHT), fill="#39344c", width=2)
    draw.text((HISTORY_LEFT, 65), "SKILL HISTORY", font=_font(15, bold=True), fill=PURPLE)
    visible = history[-13:]
    first = max(1, transitions - len(visible) + 1)
    y = 103
    for index, (skill, logged_destination) in enumerate(visible, start=first):
        current = index == transitions
        color = (
            GREEN
            if current and skill == "ask_for_reset_cube_bin_only"
            else TEXT
            if current
            else MUTED
        )
        marker = "▶" if current else "•"
        label = _short_skill(skill)
        if logged_destination:
            label += f" → {logged_destination.lower()}"
        draw.text(
            (HISTORY_LEFT, y),
            f"{marker} {index:03d}  {label}",
            font=_font(14, bold=current),
            fill=color,
        )
        y += 36

    _draw_metric_column(
        draw,
        left=COMPETENCE_LEFT,
        title="THETA 1: COMPETENCE",
        field="competences",
        decision=decision,
        color=BLUE,
    )
    _draw_metric_column(
        draw,
        left=LEARNING_RATE_LEFT,
        title="THETA 2: LEARNING RATE",
        field="learning_rates",
        decision=decision,
        color=GREEN,
    )
    _draw_decision_column(draw, decision=decision)

    footer_top = SCENE_TOP + SCENE_HEIGHT
    draw.rectangle((0, footer_top, SCENE_WIDTH, HEIGHT), fill="#10141a")
    draw.rectangle((0, footer_top, SCENE_WIDTH, footer_top + 5), fill=active_color)
    cube = _state_xyz(state, "cube_0")
    bin_ = _state_xyz(state, "bin_0")
    objects = ", ".join(current_objects)
    draw.text(
        (12, footer_top + 14),
        f"{_short_skill(current_skill)}({objects})",
        font=_font(15, bold=True),
        fill=TEXT,
    )
    draw.text(
        (12, footer_top + 42),
        f"cube x={cube[0]:.3f} y={cube[1]:.3f} z={cube[2]:.3f}  |  "
        f"bin x={bin_[0]:.3f} y={bin_[1]:.3f} z={bin_[2]:.3f}",
        font=_font(13),
        fill=MUTED,
    )
    if destination:
        draw.rectangle((0, HEIGHT - 35, SCENE_WIDTH, HEIGHT), fill="#123c30")
        draw.text(
            (SCENE_WIDTH // 2, HEIGHT - 18),
            f"RESET DESTINATION: {destination}",
            font=_font(19, bold=True),
            fill=GREEN,
            anchor="mm",
        )
    return np.asarray(image, dtype=np.uint8)


def render(*, run: Path, output: Path, realistic_background: bool) -> int:
    state_path = run / "tossing3d_state_log.jsonl"
    decision_path = run / "pomdp_decisions.jsonl"
    state_rows = [json.loads(line) for line in state_path.read_text().splitlines()]
    diagnostic_rows = [json.loads(line) for line in decision_path.read_text().splitlines()]
    decisions = {
        int(row["decision"]): row for row in diagnostic_rows if row.get("event") == "decision"
    }
    timeline = sorted(
        [row for row in diagnostic_rows if row.get("event") in {"session_start", "dispatch"}],
        key=lambda row: _timestamp(row["timestamp"]),
    )

    header_values = state_rows[0].copy()
    header_values.pop("kind")
    header_values.pop("timestamp", None)
    header_values.pop("elapsed_seconds", None)
    header = StateLogHeader(**header_values)
    problem = Tossing3DCli.build_problem(
        args=argparse.Namespace(**{**header.model_dump(), "scene_bg": realistic_background})
    )
    env = problem.env
    backend = env.backend()
    problem.hard_reset()
    problem.reset_to_task(task=problem.sample_train_task())
    output.parent.mkdir(parents=True, exist_ok=True)
    video = VideoStream(output_path=output, fps=backend.render_fps())

    timeline_index = 0
    cycle = 0
    step = 0
    transitions = 0
    history: list[tuple[str, str | None]] = []
    current_skill = "Ready"
    current_objects: tuple[str, ...] = ()
    current_decision: dict[str, Any] | None = None
    last_frame: np.ndarray | None = None
    last_state: dict[str, list[float]] | None = None
    samples = 0
    horizon = 0
    hold_frames = max(1, round(FPS_HOLD_SECONDS * backend.render_fps()))
    try:
        for row in state_rows[1:]:
            event_time = _timestamp(row["timestamp"])
            while (
                timeline_index < len(timeline)
                and _timestamp(timeline[timeline_index]["timestamp"]) <= event_time
            ):
                event = timeline[timeline_index]
                if event["event"] == "session_start":
                    cycle = int(event["cycle"])
                    step = 0
                else:
                    cycle = int(event["cycle"])
                    step += 1
                    transitions += 1
                    current_skill = str(event["skill"])
                    current_decision = decisions.get(int(event["decision"]))
                    history.append((current_skill, reset_destination(current_decision)))
                    if current_decision is not None:
                        samples = int(current_decision.get("num_samples") or samples)
                        horizon = int(current_decision.get("horizon") or horizon)
                    if (
                        reset_destination(current_decision)
                        and last_frame is not None
                        and last_state is not None
                    ):
                        held = _compose(
                            frame=last_frame,
                            state=last_state,
                            seed=header.seed,
                            cycle=cycle,
                            step=step,
                            transitions=transitions,
                            current_skill=current_skill,
                            current_objects=(),
                            history=history,
                            decision=current_decision,
                            samples=samples,
                            horizon=horizon,
                        )
                        for _ in range(hold_frames):
                            video.append(frame=held)
                timeline_index += 1

            if row["kind"] == "skill":
                current_skill = str(row["name"])
                current_objects = tuple(str(value) for value in row.get("objects", ()))
                continue
            if row["kind"] != "tick":
                continue
            last_state = {key: list(value) for key, value in row["state"].items()}
            env.restore_plain_snapshot(plain=last_state)
            last_frame = backend.render()
            video.append(
                frame=_compose(
                    frame=last_frame,
                    state=last_state,
                    seed=header.seed,
                    cycle=cycle,
                    step=step,
                    transitions=transitions,
                    current_skill=current_skill,
                    current_objects=current_objects,
                    history=history,
                    decision=current_decision,
                    samples=samples,
                    horizon=horizon,
                )
            )
    finally:
        video.close()
        env.close()
    return video.frames_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plain-background",
        action="store_true",
        help="Use the stripped simulator scene; realistic presentation background is default.",
    )
    args = parser.parse_args()
    frames = render(
        run=args.run,
        output=args.output,
        realistic_background=not args.plain_background,
    )
    print(f"wrote {args.output} ({frames} frames)")


if __name__ == "__main__":
    main()
