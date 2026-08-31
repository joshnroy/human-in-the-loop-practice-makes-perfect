"""Practice diagnostics arranged beside an unobscured environment scene."""

import textwrap
from collections.abc import Mapping
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
        SkillChatOverlay.draw_chart(
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
        SkillChatOverlay.draw_chart(
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
        best = max(values, key=lambda name: values[name]) if values else None
        colors = {name: (90, 230, 160) if name == best else (140, 110, 210) for name in values}
        SkillChatOverlay.draw_chart(
            draw=draw,
            left=left,
            title="DECISION VALUES",
            values=values,
            color=(140, 110, 210),
            colors=colors,
            precision=6,
        )

    @staticmethod
    def draw_chart(
        *,
        draw: ImageDraw.ImageDraw,
        left: int,
        title: str,
        values: Mapping[str, float],
        color: tuple[int, int, int],
        precision: int,
        fixed_range: tuple[float, float] | None = None,
        colors: Mapping[str, tuple[int, int, int]] | None = None,
    ) -> None:
        font = ImageFont.load_default(size=17)
        small = ImageFont.load_default(size=13)
        draw.multiline_text((left, 14), title, font=font, fill=color, spacing=3)
        if not values:
            draw.text((left, 64), "No decision yet", font=small, fill=SkillChatOverlay.muted)
            return
        if fixed_range is None:
            low = min(0.0, *values.values())
            high = max(0.0, *values.values())
        else:
            low, high = fixed_range
        span = max(high - low, 1e-12)
        zero = left + 280 * (0.0 - low) / span
        top = 76
        for name, value in values.items():
            label = "\n".join(textwrap.wrap(name, width=35))
            draw.multiline_text((left, top), label, font=small, fill=SkillChatOverlay.text)
            label_lines = max(1, len(label.splitlines()))
            bar_top = top + 18 * label_lines + 6
            end = left + 280 * (value - low) / span
            item_color = color if colors is None else colors[name]
            draw.rectangle((left, bar_top, left + 280, bar_top + 14), fill=(45, 45, 65))
            draw.rectangle((min(zero, end), bar_top, max(zero, end), bar_top + 14), fill=item_color)
            draw.line((zero, bar_top - 2, zero, bar_top + 16), fill=(240, 240, 240))
            draw.text((left, bar_top + 18), f"{value:.{precision}f}", font=small, fill=item_color)
            top = bar_top + 48
