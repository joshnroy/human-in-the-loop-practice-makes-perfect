from pathlib import Path
from typing import ClassVar

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import LoopPhase, LoopStatus, ResetKind


class StatusBarOverlay:
    """Draws a run's loop state *around* an already-rendered environment frame: a
    persistent status bar below it whose fields update, plus a marker over the frame
    itself whenever something discontinuous happened (a reset, an outcome).

    Deliberately separate from every `core.Renderer`, and composing after the fact
    rather than during: a domain renderer's job is to draw its own state, and
    teaching it about cycles, sweeps and reset kinds would put harness knowledge
    into every domain and make the annotation impossible to change without touching
    all of them. This takes whatever HxWx3 frame a `Renderer` produced and returns a
    new, taller one; the input frame is never modified.

    **Status indicators, not a scrolling log.** The same fields sit in the same
    places for the whole video and change value; a viewer reads the bar rather than
    following it. Phase is additionally a filled colour chip, so the
    practice/evaluate rhythm is legible while scrubbing, without reading anything.

    A static-method container, never instantiated, same as every other
    business-logic class in this project -- it has no state between calls beyond the
    font cache, which is memoization, not state anyone observes.
    """

    bar_height: ClassVar[int] = 96
    # A narrow domain render would otherwise leave no room for the bar's text. The
    # environment frame stays at its own size, left-aligned, and the canvas widens
    # around it.
    min_width: ClassVar[int] = 640
    # imageio-ffmpeg resizes (and warns) unless both dimensions are a multiple of
    # this, which would blur exactly the text this class exists to make readable.
    macro_block: ClassVar[int] = 16

    background: ClassVar[tuple[int, int, int]] = (16, 20, 24)
    text_color: ClassVar[tuple[int, int, int]] = (238, 242, 246)
    key_color: ClassVar[tuple[int, int, int]] = (150, 162, 174)

    # One colour per phase, chosen to stay distinguishable in a thumbnail-sized
    # scrub preview: violet (nothing practiced yet), amber (practicing), green
    # (measuring).
    phase_colors: ClassVar[dict[LoopPhase, tuple[int, int, int]]] = {
        LoopPhase.BASELINE_EVALUATION: (124, 92, 186),
        LoopPhase.PRACTICE: (214, 138, 30),
        LoopPhase.EVALUATION: (32, 138, 96),
    }
    phase_labels: ClassVar[dict[LoopPhase, str]] = {
        LoopPhase.BASELINE_EVALUATION: "BASELINE EVAL",
        LoopPhase.PRACTICE: "PRACTICE",
        LoopPhase.EVALUATION: "EVALUATION",
    }

    # Deliberately far apart from each other *and* from the phase colours: telling
    # the five resets apart is the reason this recording exists.
    reset_colors: ClassVar[dict[ResetKind, tuple[int, int, int]]] = {
        ResetKind.HARD: (206, 40, 56),
        ResetKind.PERIOD: (206, 52, 176),
        ResetKind.INTERVAL: (16, 168, 208),
        # Green, the one hue no other kind or phase uses -- the charged reset should be
        # the one a viewer can pick out while scrubbing.
        ResetKind.HUMAN: (40, 190, 110),
        ResetKind.EVALUATION_TASK: (226, 200, 40),
    }
    reset_labels: ClassVar[dict[ResetKind, str]] = {
        ResetKind.HARD: "HARD RESET (harness, once per run)",
        ResetKind.PERIOD: "PERIOD RESET (top of practice cycle)",
        ResetKind.INTERVAL: "INTERVAL RESET (mid-period)",
        ResetKind.HUMAN: "HUMAN RESET (mid-period, charged)",
        ResetKind.EVALUATION_TASK: "EVAL-TASK RESET (per test task)",
    }
    event_color: ClassVar[tuple[int, int, int]] = (240, 244, 248)

    _fonts: ClassVar[dict[tuple[str, int], ImageFont.FreeTypeFont]] = {}

    @staticmethod
    def compose(
        *,
        frame: np.ndarray,
        status: LoopStatus,
        bar_position: str = "bottom",
        extra_fields: tuple[tuple[str, str], ...] = (),
    ) -> np.ndarray:
        """The environment frame with the status bar below it and, when something
        discontinuous happened, a marker over the frame itself. Returns a new
        array; `frame` is only read."""
        height, width = int(frame.shape[0]), int(frame.shape[1])
        canvas_width = StatusBarOverlay._rounded_up(value=max(width, StatusBarOverlay.min_width))
        canvas_height = StatusBarOverlay._rounded_up(value=height + StatusBarOverlay.bar_height)

        assert bar_position in {"top", "bottom"}
        bar_top = 0 if bar_position == "top" else height
        frame_top = StatusBarOverlay.bar_height if bar_position == "top" else 0
        image = Image.new("RGB", (canvas_width, canvas_height), StatusBarOverlay.background)
        image.paste(Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8)), (0, frame_top))
        draw = ImageDraw.Draw(image)
        StatusBarOverlay._draw_marker(
            draw=draw,
            status=status,
            top=frame_top,
            height=height,
            width=canvas_width,
        )
        StatusBarOverlay._draw_bar(
            draw=draw,
            status=status,
            top=bar_top,
            width=canvas_width,
            bottom=bar_top + StatusBarOverlay.bar_height - 1,
            extra_fields=extra_fields,
        )
        return np.array(image, dtype=np.uint8)

    @staticmethod
    def format_fields(*, status: LoopStatus) -> tuple[tuple[str, str], ...]:
        """The bar's (label, value) pairs, in reading order -- pure, so what the bar
        *says* is testable without going through pixels.

        The counters are phase-dependent on purpose. A practice period is one of
        num_cycles cycles and is shown 1-indexed, the way a human counts them; an
        evaluation sweep is shown 0-indexed, because sweep 0 is a real sweep that
        happens before cycle 1 exists and renumbering it would hide that.

        RESET/EVENT come *before* the counters, and the free-text goal comes last,
        because the bar wraps and a narrow frame can run out of room: the fields
        that only appear on the handful of frames where something actually happened
        are the ones that must never be the ones dropped."""
        is_practice = status.phase is LoopPhase.PRACTICE
        fields: list[tuple[str, str]] = [("PHASE", StatusBarOverlay.phase_labels[status.phase])]
        if status.reset is not None:
            fields.append(("RESET", StatusBarOverlay.reset_labels[status.reset]))
        if status.event is not None:
            fields.append(("EVENT", status.event))
        if is_practice:
            fields.append(("CYCLE", f"{status.cycle_index + 1}/{status.num_cycles}"))
        else:
            fields.append(("SWEEP", f"{status.cycle_index}/{status.num_cycles}"))
            if status.task_index is not None:
                fields.append(("TEST TASK", f"{status.task_index + 1}/{status.num_tasks}"))
        if status.step_index is not None:
            total = "?" if status.num_steps is None else str(status.num_steps)
            fields.append(("STEP", f"{status.step_index + 1}/{total}"))
        fields.append(("TRANSITIONS", str(status.transitions)))
        if status.skill is not None:
            fields.append(("SKILL", status.skill))
        if status.task:
            fields.append(("TASK" if is_practice else "GOAL", status.task))
        return tuple(fields)

    @staticmethod
    def _draw_bar(
        *,
        draw: ImageDraw.ImageDraw,
        status: LoopStatus,
        top: int,
        width: int,
        bottom: int,
        extra_fields: tuple[tuple[str, str], ...] = (),
    ) -> None:
        margin = 10
        draw.rectangle([(0, top), (width, bottom)], fill=StatusBarOverlay.background)
        # A hairline in the phase colour along the seam, so the phase is readable
        # even in a scrub preview too small to show the chip's text.
        draw.rectangle(
            [(0, top), (width, top + 4)], fill=StatusBarOverlay.phase_colors[status.phase]
        )

        chip_font = StatusBarOverlay._font(name="DejaVuSans-Bold.ttf", size=22)
        label = StatusBarOverlay.phase_labels[status.phase]
        chip_width = int(draw.textlength(label, font=chip_font)) + 2 * margin
        chip_top = top + margin + 4
        chip_bottom = bottom - margin
        draw.rectangle(
            [(margin, chip_top), (margin + chip_width, chip_bottom)],
            fill=StatusBarOverlay.phase_colors[status.phase],
        )
        draw.text(
            (margin + chip_width / 2, (chip_top + chip_bottom) / 2),
            label,
            font=chip_font,
            fill=(255, 255, 255),
            anchor="mm",
        )

        # Everything but PHASE (already the chip), wrapped over the bar's two text
        # lines so a long goal description cannot push the counters off the frame.
        fields = [*extra_fields, *StatusBarOverlay.format_fields(status=status)[1:]]
        left = 2 * margin + chip_width
        StatusBarOverlay._draw_fields(
            draw=draw,
            fields=fields,
            left=left,
            right=width - margin,
            first_baseline=top + margin + 12,
            line_height=34,
            num_lines=2,
        )

    @staticmethod
    def _draw_fields(
        *,
        draw: ImageDraw.ImageDraw,
        fields: list[tuple[str, str]],
        left: int,
        right: int,
        first_baseline: int,
        line_height: int,
        num_lines: int,
    ) -> None:
        key_font = StatusBarOverlay._font(name="DejaVuSans.ttf", size=17)
        value_font = StatusBarOverlay._font(name="DejaVuSans-Bold.ttf", size=21)
        x = left
        line = 0
        for key, value in fields:
            key_width = draw.textlength(f"{key} ", font=key_font)
            value_width = draw.textlength(value, font=value_font)
            if x > left and x + key_width + value_width > right:
                line += 1
                if line >= num_lines:
                    return
                x = left
            y = first_baseline + line * line_height
            draw.text((x, y), f"{key} ", font=key_font, fill=StatusBarOverlay.key_color)
            draw.text(
                (x + key_width, y - 2), value, font=value_font, fill=StatusBarOverlay.text_color
            )
            x = int(x + key_width + value_width + 26)

    @staticmethod
    def _draw_marker(
        *, draw: ImageDraw.ImageDraw, status: LoopStatus, top: int, height: int, width: int
    ) -> None:
        """A reset is an instantaneous state jump and an outcome is a single frame,
        so both get marked where the viewer is already looking -- a thick border
        plus a captioned chip over the environment drawing itself, not only a field
        change in the bar underneath it."""
        if status.reset is not None:
            color = StatusBarOverlay.reset_colors[status.reset]
            caption = f"RESET — {StatusBarOverlay.reset_labels[status.reset]}"
        elif status.event is not None:
            color = StatusBarOverlay.event_color
            caption = status.event
        else:
            return

        thickness = 8
        bottom = top + height - 1
        draw.rectangle([(0, top), (width - 1, bottom)], outline=color, width=thickness)
        font = StatusBarOverlay._font(name="DejaVuSans-Bold.ttf", size=20)
        text_width = draw.textlength(caption, font=font)
        pad = 12
        # Bottom-LEFT, next to the bar. Not the top, because a domain renderer's own
        # titles and legends live there (Tossing Room's "Start" and one-way-ledge
        # labels do) and a caption would cover them on exactly the frames a viewer
        # is looking hardest at; not centred, because Renderer.render_frame draws
        # its own action label along the bottom centre.
        caption_bottom = max(top + thickness, bottom - thickness)
        caption_top = caption_bottom - 32
        left = thickness + pad
        draw.rectangle(
            [(left, caption_top), (left + text_width + 2 * pad, caption_bottom)], fill=color
        )
        draw.text(
            (left + pad, (caption_top + caption_bottom) / 2),
            caption,
            font=font,
            fill=StatusBarOverlay.background,
            anchor="lm",
        )

    @staticmethod
    def _font(*, name: str, size: int) -> ImageFont.FreeTypeFont:
        """DejaVu, taken from matplotlib's own bundled copy rather than from the
        system: matplotlib is already a hard dependency and ships these fonts, so
        the bar renders identically on a machine with no fonts installed at all
        (CI), which PIL's tiny built-in bitmap font would not do legibly."""
        cached = StatusBarOverlay._fonts.get((name, size))
        if cached is None:
            path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name
            cached = ImageFont.truetype(str(path), size)
            StatusBarOverlay._fonts[(name, size)] = cached
        return cached

    @staticmethod
    def _rounded_up(*, value: int) -> int:
        return value + (-value) % StatusBarOverlay.macro_block
