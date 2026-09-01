"""Practice diagnostics arranged beside an unobscured environment scene."""

import textwrap
from typing import ClassVar

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class SkillChatOverlay:
    """Compose four stable diagnostic columns to the right of the scene."""

    width: ClassVar[int] = 320
    max_entries: ClassVar[int] = 8
    background: ClassVar[tuple[int, int, int]] = (22, 20, 32)
    text: ClassVar[tuple[int, int, int]] = (235, 235, 245)
    muted: ClassVar[tuple[int, int, int]] = (190, 190, 200)
    skill_rows: ClassVar[tuple[tuple[str, str], ...]] = (
        ("PickCube", "PickCube"),
        ("MoveToTossLocationAndToss", "Toss"),
        ("OpenGripper", "OpenGripper"),
        ("ask_for_reset_cube_bin_only", "Human Reset"),
    )

    @staticmethod
    def compose(
        *,
        frame: np.ndarray,
        history: list[str],
        values: dict[str, float] | None = None,
        competences: dict[str, float] | None = None,
        learning_rates: dict[str, float] | None = None,
    ) -> np.ndarray:
        height, scene_width = frame.shape[:2]
        canvas = Image.new(
            "RGB", (scene_width + 4 * SkillChatOverlay.width, height), SkillChatOverlay.background
        )
        canvas.paste(Image.fromarray(frame), (0, 0))
        draw = ImageDraw.Draw(canvas)
        lefts = [scene_width + index * SkillChatOverlay.width + 16 for index in range(4)]
        SkillChatOverlay.draw_history(draw=draw, left=lefts[0], height=height, history=history)
        SkillChatOverlay.draw_competences(draw=draw, left=lefts[1], estimates=competences or {})
        SkillChatOverlay.draw_learning_rates(
            draw=draw, left=lefts[2], estimates=learning_rates or {}
        )
        SkillChatOverlay.draw_values(draw=draw, left=lefts[3], values=values or {})
        return np.asarray(canvas)

    @staticmethod
    def draw_history(
        *, draw: ImageDraw.ImageDraw, left: int, height: int, history: list[str]
    ) -> None:
        font = ImageFont.load_default(size=17)
        draw.text((left, 14), "SKILL HISTORY", font=font, fill=(190, 150, 255))
        bottom = height - 20
        for entry in reversed(history[-SkillChatOverlay.max_entries :]):
            lines = textwrap.wrap(entry, width=29)
            top = bottom - 23 * len(lines) - 12
            if top < 52:
                break
            color = (90, 230, 160) if "RESET" in entry else SkillChatOverlay.text
            draw.multiline_text((left, top), "\n".join(lines), font=font, fill=color, spacing=5)
            bottom = top - 12

    @staticmethod
    def draw_competences(
        *, draw: ImageDraw.ImageDraw, left: int, estimates: dict[str, float]
    ) -> None:
        SkillChatOverlay.draw_skill_chart(
            draw=draw,
            left=left,
            title="THETA 1: COMPETENCE",
            values=estimates,
            color=(70, 160, 230),
            fixed_range=(0.0, 1.0),
            precision=4,
        )

    @staticmethod
    def draw_learning_rates(
        *, draw: ImageDraw.ImageDraw, left: int, estimates: dict[str, float]
    ) -> None:
        SkillChatOverlay.draw_skill_chart(
            draw=draw,
            left=left,
            title="THETA 2: LEARNING RATE",
            values=estimates,
            color=(70, 200, 175),
            fixed_range=(0.0, 1.0),
            precision=4,
        )

    @staticmethod
    def draw_values(*, draw: ImageDraw.ImageDraw, left: int, values: dict[str, float]) -> None:
        SkillChatOverlay.draw_skill_chart(
            draw=draw,
            left=left,
            title="EXPECTIMAX VALUES v",
            values=values,
            color=(140, 110, 210),
            precision=6,
            include_stop=True,
        )

    @staticmethod
    def draw_skill_chart(
        *,
        draw: ImageDraw.ImageDraw,
        left: int,
        title: str,
        values: dict[str, float],
        color: tuple[int, int, int],
        precision: int,
        fixed_range: tuple[float, float] | None = None,
        include_stop: bool = False,
    ) -> None:
        font = ImageFont.load_default(size=17)
        small = ImageFont.load_default(size=13)
        draw.multiline_text((left, 14), title, font=font, fill=color, spacing=3)
        rows: list[tuple[str, float | None]] = [
            (
                label,
                next((value for name, value in values.items() if name.startswith(prefix)), None),
            )
            for prefix, label in SkillChatOverlay.skill_rows
        ]
        if include_stop:
            rows.append(("STOP (v_stop)", values.get("STOP")))
        present = [value for _label, value in rows if value is not None]
        if fixed_range is None:
            low = min([0.0, *present])
            high = max([0.0, *present])
        else:
            low, high = fixed_range
        span = max(high - low, 1e-12)
        zero = left + 280 * (0.0 - low) / span
        best = max(present) if present else None
        for index, (label, value) in enumerate(rows):
            top = 62 + index * 92
            displayed_label = f"{label} (v*)" if value is not None and value == best else label
            draw.text((left, top), displayed_label, font=small, fill=SkillChatOverlay.text)
            bar_top = top + 23
            draw.rectangle((left, bar_top, left + 280, bar_top + 14), fill=(45, 45, 65))
            if value is None:
                draw.text((left, bar_top + 18), "N/A", font=small, fill=SkillChatOverlay.muted)
                continue
            end = left + 280 * (value - low) / span
            item_color = (90, 230, 160) if include_stop and value == best else color
            draw.rectangle((min(zero, end), bar_top, max(zero, end), bar_top + 14), fill=item_color)
            draw.line((zero, bar_top - 2, zero, bar_top + 16), fill=(240, 240, 240))
            draw.text((left, bar_top + 18), f"{value:.{precision}f}", font=small, fill=item_color)
