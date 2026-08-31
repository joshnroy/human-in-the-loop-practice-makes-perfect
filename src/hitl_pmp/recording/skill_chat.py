"""A bounded, scrolling practice-event panel beside the unobscured scene."""

import textwrap
from typing import ClassVar

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class SkillChatOverlay:
    """Newest events appear at the bottom; repeated skills remain separate entries."""

    width: ClassVar[int] = 320
    max_entries: ClassVar[int] = 8

    @staticmethod
    def compose(
        *, frame: np.ndarray, history: list[str], values: dict[str, float] | None = None
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        canvas = Image.new("RGB", (width + SkillChatOverlay.width, height), (22, 20, 32))
        canvas.paste(Image.fromarray(frame), (0, 0))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=17)
        chart_bottom = SkillChatOverlay.draw_values(
            draw=draw, left=width + 16, values=values or {}, font=font
        )
        draw.text(
            (width + 16, chart_bottom + 16),
            "PRACTICE / SKILL HISTORY",
            font=font,
            fill=(190, 150, 255),
        )
        bottom = height - 20
        for entry in reversed(history[-SkillChatOverlay.max_entries :]):
            lines = textwrap.wrap(entry, width=29)
            top = bottom - 23 * len(lines) - 12
            if top < chart_bottom + 52:
                break
            color = (90, 230, 160) if "RESET" in entry else (240, 235, 250)
            draw.multiline_text(
                (width + 16, top), "\n".join(lines), font=font, fill=color, spacing=5
            )
            bottom = top - 12
        return np.asarray(canvas)

    @staticmethod
    def draw_values(
        *,
        draw: ImageDraw.ImageDraw,
        left: int,
        values: dict[str, float],
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ) -> int:
        if not values:
            return 0
        draw.text((left, 14), "DECISION VALUES", font=font, fill=(190, 150, 255))
        small = ImageFont.load_default(size=13)
        draw.text((left, 37), "Expected utility, including cost", font=small, fill=(190, 190, 200))
        low = min(0.0, *values.values())
        high = max(0.0, *values.values())
        span = max(high - low, 1e-12)
        zero = left + 280 * (0.0 - low) / span
        best = max(values, key=lambda name: values[name])
        top = 62
        for name, value in values.items():
            label = "\n".join(textwrap.wrap(name, width=35))
            draw.multiline_text((left, top), label, font=small, fill=(235, 235, 245))
            bar_top = top + 34
            end = left + 280 * (value - low) / span
            color = (90, 230, 160) if name == best else (140, 110, 210)
            draw.rectangle((min(zero, end), bar_top, max(zero, end), bar_top + 14), fill=color)
            draw.line((zero, bar_top - 2, zero, bar_top + 16), fill=(240, 240, 240))
            draw.text((left, bar_top + 17), f"{value:.6f}", font=small, fill=color)
            top += 72
        return top
