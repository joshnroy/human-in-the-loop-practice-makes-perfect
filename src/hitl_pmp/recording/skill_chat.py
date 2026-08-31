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
    def compose(*, frame: np.ndarray, history: list[str]) -> np.ndarray:
        height, width = frame.shape[:2]
        canvas = Image.new("RGB", (width + SkillChatOverlay.width, height), (22, 20, 32))
        canvas.paste(Image.fromarray(frame), (0, 0))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=17)
        draw.text((width + 16, 16), "PRACTICE / SKILL HISTORY", font=font, fill=(190, 150, 255))
        bottom = height - 20
        for entry in reversed(history[-SkillChatOverlay.max_entries :]):
            lines = textwrap.wrap(entry, width=29)
            top = bottom - 23 * len(lines) - 12
            if top < 52:
                break
            color = (90, 230, 160) if "RESET" in entry else (240, 235, 250)
            draw.multiline_text(
                (width + 16, top), "\n".join(lines), font=font, fill=color, spacing=5
            )
            bottom = top - 12
        return np.asarray(canvas)
