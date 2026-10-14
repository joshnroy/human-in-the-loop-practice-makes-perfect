from pathlib import Path
from typing import ClassVar

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hitl_pmp.recording.run_record.types import Phase

from .types import ReplayStatus


class PresentationStyle:
    """The Tossing3D presentation palette and type, shared by every replay video."""

    background: ClassVar[str] = "#0f0d1c"
    panel: ClassVar[str] = "#151225"
    track: ClassVar[str] = "#2b2941"
    text: ClassVar[str] = "#f2f0f7"
    muted: ClassVar[str] = "#a39eb2"
    purple: ClassVar[str] = "#9a6be8"
    blue: ClassVar[str] = "#45a9e8"
    green: ClassVar[str] = "#51e299"
    amber: ClassVar[str] = "#eda015"
    red: ClassVar[str] = "#e8575a"
    bar: ClassVar[str] = "#10141a"
    divider: ClassVar[str] = "#39344c"

    _fonts: ClassVar[dict[tuple[int, bool], ImageFont.FreeTypeFont]] = {}

    @staticmethod
    def font(*, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        cached = PresentationStyle._fonts.get((size, bold))
        if cached is None:
            name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
            path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name
            cached = ImageFont.truetype(str(path), size)
            PresentationStyle._fonts[size, bold] = cached
        return cached


class ReplayCompositor:
    """A fixed 1280x784 canvas: status bar, the scene, and a skill-history panel.

    Fixed, so recordings from different processes (practice, then each evaluation)
    append to one video stream; both sides are multiples of 16 for the encoder."""

    width: ClassVar[int] = 1280
    height: ClassVar[int] = 784
    top: ClassVar[int] = 64
    scene_width: ClassVar[int] = 960
    scene_height: ClassVar[int] = 720
    history_rows: ClassVar[int] = 16

    @staticmethod
    def compose(*, frame: np.ndarray, status: ReplayStatus) -> np.ndarray:
        style = PresentationStyle
        image = Image.new(
            "RGB", (ReplayCompositor.width, ReplayCompositor.height), style.background
        )
        scene = Image.fromarray(np.ascontiguousarray(frame[..., :3], dtype=np.uint8)).resize(
            (ReplayCompositor.scene_width, ReplayCompositor.scene_height),
            Image.Resampling.LANCZOS,
        )
        image.paste(scene, (0, ReplayCompositor.top))
        draw = ImageDraw.Draw(image)
        ReplayCompositor._status_bar(draw=draw, status=status)
        ReplayCompositor._history(draw=draw, status=status)
        if status.banner is not None:
            ReplayCompositor._banner(draw=draw, status=status)
        return np.asarray(image, dtype=np.uint8)

    @staticmethod
    def _status_bar(*, draw: ImageDraw.ImageDraw, status: ReplayStatus) -> None:
        style = PresentationStyle
        practice = status.phase is Phase.PRACTICE
        chip = style.amber if practice else style.blue
        draw.rectangle((0, 0, ReplayCompositor.width, ReplayCompositor.top - 1), fill=style.bar)
        draw.rectangle((0, 0, 176, ReplayCompositor.top - 1), fill=chip)
        draw.text(
            (88, 22),
            "PRACTICE" if practice else "EVALUATION",
            font=style.font(size=18, bold=True),
            fill="white",
            anchor="mm",
        )
        draw.text(
            (88, 46),
            f"{status.speedup}x speed",
            font=style.font(size=13, bold=True),
            fill="white",
            anchor="mm",
        )
        fields = [status.environment.upper()]
        if status.method:
            fields.append(status.method)
        if status.checkpoint is not None:
            fields.append(f"CHECKPOINT {status.checkpoint}")
        if status.task_index is not None:
            fields.append(f"TASK {status.task_index}")
        if status.cycle is not None:
            fields.append(f"CYCLE {status.cycle}")
        if status.end_step is not None:
            fields.append(f"STEP {status.end_step}")
        draw.text((192, 12), "    ".join(fields), font=style.font(size=15), fill=style.text)
        skill = status.skill or "initial state"
        actor = "HUMAN" if status.actor == "human" else "ROBOT" if status.actor else ""
        draw.text(
            (192, 36),
            f"{actor}  {skill}".strip(),
            font=style.font(size=15, bold=True),
            fill=style.green if status.actor == "human" else style.muted,
        )

    @staticmethod
    def _history(*, draw: ImageDraw.ImageDraw, status: ReplayStatus) -> None:
        style = PresentationStyle
        left = ReplayCompositor.scene_width
        draw.rectangle(
            (left, ReplayCompositor.top, ReplayCompositor.width, ReplayCompositor.height),
            fill=style.panel,
        )
        draw.line(
            (left, ReplayCompositor.top, left, ReplayCompositor.height),
            fill=style.divider,
            width=2,
        )
        draw.text(
            (left + 16, ReplayCompositor.top + 14),
            "SKILL HISTORY",
            font=style.font(size=15, bold=True),
            fill=style.purple,
        )
        visible = status.history[-ReplayCompositor.history_rows :]
        y = ReplayCompositor.top + 50 + (ReplayCompositor.history_rows - len(visible)) * 40
        for index, item in enumerate(visible):
            current = index == len(visible) - 1
            draw.text(
                (left + 16, y),
                f"{'▶' if current else '•'} {item}"[:38],
                font=style.font(size=13, bold=current),
                fill=style.text if current else style.muted,
            )
            y += 40

    @staticmethod
    def _banner(*, draw: ImageDraw.ImageDraw, status: ReplayStatus) -> None:
        style = PresentationStyle
        color = {"human": style.green, "success": style.green, "failure": style.red}.get(
            status.banner_tone or "human", style.green
        )
        middle = ReplayCompositor.scene_width // 2
        top = ReplayCompositor.top + ReplayCompositor.scene_height // 2 - 70
        draw.rounded_rectangle(
            (90, top, ReplayCompositor.scene_width - 90, top + 140),
            radius=16,
            fill=style.bar,
            outline=color,
            width=4,
        )
        assert status.banner is not None
        lines = status.banner.splitlines()
        draw.text(
            (middle, top + 48),
            lines[0],
            font=style.font(size=30, bold=True),
            fill=color,
            anchor="mm",
        )
        if len(lines) > 1:
            draw.text(
                (middle, top + 98),
                lines[1][:70],
                font=style.font(size=17, bold=True),
                fill=style.text,
                anchor="mm",
            )
