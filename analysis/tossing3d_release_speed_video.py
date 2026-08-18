"""Compose the toss-parameter clips into one video where the arcs accumulate.

Reads `scripts/tossing3d_release_speed_clips.py`'s output -- one `clip_<speed>_<ms>.mp4` per
cell plus a `tosses.json` -- and builds simulator footage on the left, a to-scale side
elevation on the right, and a status bar underneath.

> **The producer is gone; this reader is not.** The two-skill migration deleted
> `scripts/tossing3d_release_speed_clips.py` along with the three other scripts that drove
> the retired `Pick -> MoveToThrowPose -> Toss` decomposition. This module is post-run
> analysis (`analysis/` never drives a simulator), so it still composes footage already on
> disk exactly as before -- but nothing in the tree produces new input for it, and the
> footage it can read was recorded on a decomposition and a scene that no longer exist.
> Kept rather than deleted because the composition itself is reusable; the named script is
> readable in git history. **The right-hand panel never clears**:
each parabola stays, faint, under the next, so the last segment shows the whole family.
Never touches a simulator, so the visual design can be re-cut against footage that cost
minutes.

No goal box is drawn: the subject is how far the cube flies, and the bin is marked only as
the landmark that contaminates the resting position. Both distances are annotated **from the
robot base**, the quantity `analysis/tossing3d_toss_parameter_surface.py` computes on.

`recording.StatusBarOverlay` is the right shape but cannot carry these fields: `LoopStatus`
is frozen around the practice loop's vocabulary and `compose` always paints a phase chip, so
it would burn `PHASE EVALUATION` `SWEEP 0/0` onto a video with neither. `StatusBar` below is
a sibling of it.

Sequential ramp (viridis, dark = slow/early), not the project's `#0072B2`/`#D55E00`, which
encode assistance-available versus nothing-intervenes.

    scripts/with_env.sh python analysis/tossing3d_release_speed_video.py \
        --clips-dir /tmp/toss-clips --output-video out.mp4 --output-figure out.png

(JSON and PNG frames only, so this needs no worktree-shadowing `PYTHONPATH`.)
"""

import argparse
import json
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

# The footage is 200 Hz; every `PLAYBACK_STRIDE`-th frame at `PLAYBACK_FPS` is 3.3x slow
# motion, which makes a 0.39 s flight legible as a throw.
PLAYBACK_FPS = 30
PLAYBACK_STRIDE = 2
PRE_RELEASE_SECONDS = 0.8
POST_LANDING_SECONDS = 0.7
# The accumulated panel, held, so the family is the last thing on screen.
FINALE_SECONDS = 6.0
# Above this, a flight ended in mid-air and its arc needs the extrapolation drawn in.
# Between a resting cube's 0.025 m centre and the 0.20 m top of the bin's walls.
TAIL_MIN_HEIGHT = 0.06

BAR_HEIGHT = 116
MACRO_BLOCK = 16

BACKGROUND = (16, 20, 24)
PANEL_BACKGROUND = (10, 13, 16)
TEXT_COLOR = (238, 242, 246)
KEY_COLOR = (150, 162, 174)
GRID_COLOR = (44, 52, 60)
GROUND_COLOR = (96, 108, 120)
BIN_COLOR = (226, 200, 40)
BASE_COLOR = (150, 162, 174)


class TossPhase(str, Enum):
    """SWING includes the windup. FLIGHT is the contact-free parabola, the only stretch
    where the cube's position measures the release rather than a collision."""

    SWING = "swing"
    FLIGHT = "flight"
    LANDED = "landed"


class PanelTransform(BaseModel):
    """World metres (x forward, z up) to panel pixels, isotropically.

    `fit` treats the requested z span as a hard floor and widens x to absorb the difference.
    """

    x_min: float
    x_max: float
    z_min: float
    z_max: float
    width: int
    height: int

    @staticmethod
    def fit(  # noqa: PLR0917
        *,
        x_min: float,
        x_max: float,
        z_min: float,
        z_max: float,
        width: int,
        height: int,
        margin: float = 0.0,
    ) -> "PanelTransform":
        x_lo, x_hi = x_min - margin, x_max + margin
        z_lo, z_hi = z_min, z_max + margin
        # The coarser axis wins, so neither requested span is compressed.
        scale = max((x_hi - x_lo) / width, (z_hi - z_lo) / height)
        x_span, z_span = scale * width, scale * height
        x_slack = (x_span - (x_hi - x_lo)) / 2.0
        return PanelTransform(
            x_min=x_lo - x_slack,
            x_max=x_hi + x_slack,
            z_min=z_lo,
            z_max=z_lo + z_span,
            width=width,
            height=height,
        )

    @property
    def metres_per_pixel_x(self) -> float:
        return (self.x_max - self.x_min) / self.width

    @property
    def metres_per_pixel_z(self) -> float:
        return (self.z_max - self.z_min) / self.height

    def to_pixel(self, *, x: float, z: float) -> tuple[float, float]:
        """One world point as (column, row). Row grows downward, so z is flipped."""
        column = (x - self.x_min) / self.metres_per_pixel_x
        row = self.height - (z - self.z_min) / self.metres_per_pixel_z
        return column, row


def phase_at(*, t: float, release_t: float, land_t: float) -> TossPhase:
    if t < release_t:
        return TossPhase.SWING
    if t <= land_t:
        return TossPhase.FLIGHT
    return TossPhase.LANDED


def playback_indices(  # noqa: PLR0917
    *,
    times: np.ndarray,
    release_t: float,
    land_t: float,
    pre_seconds: float,
    post_seconds: float,
    stride: int,
) -> list[int]:
    """The recorded frames to play, bracketing the flight and clamped to what exists.

    Clamped rather than padded with the first or last frame, which reads as a hang.
    """
    first = int(np.searchsorted(times, release_t - pre_seconds, side="left"))
    last = int(np.searchsorted(times, land_t + post_seconds, side="right")) - 1
    first = max(0, min(first, len(times) - 1))
    last = max(first, min(last, len(times) - 1))
    return list(range(first, last + 1, stride))


def status_fields(  # noqa: PLR0917
    *,
    speed: float,
    release_ms: float,
    index: int,
    total: int,
    seed: int,
    standoff: float,
    base_x: float,
    phase: TossPhase,
    cube_x: float,
    impact_x: float | None,
    resting_x: float | None,
    solved: bool | None,
) -> tuple[tuple[str, str], ...]:
    """The bar's (label, value) pairs in reading order.

    The measured range is withheld until the cube has landed: printing it mid-flight would
    imply the number was observed at that instant.
    """
    fields: list[tuple[str, str]] = [
        ("SPEED", f"{speed:.0f} deg/s"),
        ("RELEASE", f"{release_ms:.0f} ms"),
        ("THROW", f"{index + 1}/{total}"),
        ("STANDOFF", f"{standoff:.2f} m"),
        ("SEED", str(seed)),
        ("PHASE", phase.value.upper()),
        ("CUBE x", f"{cube_x:.3f} m"),
    ]
    if phase is TossPhase.LANDED and impact_x is not None:
        fields.append(("BALLISTIC", f"{impact_x - base_x:.3f} m from base"))
        if resting_x is not None:
            fields.append(("RESTING", f"{resting_x - base_x:.3f} m from base"))
        fields.append(("SCORES", {True: "yes", False: "no", None: "?"}[solved]))
    return tuple(fields)


class TossPanel:
    """The side elevation: ground, bin, robot base, and every arc drawn so far.

    Stateless: the accumulation lives in the caller's list of completed arcs."""

    _fonts: ClassVar[dict[tuple[str, int], ImageFont.FreeTypeFont]] = {}

    @staticmethod
    def font(*, name: str, size: int) -> ImageFont.FreeTypeFont:
        cached = TossPanel._fonts.get((name, size))
        if cached is None:
            path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name
            cached = ImageFont.truetype(str(path), size)
            TossPanel._fonts[(name, size)] = cached
        return cached

    @staticmethod
    def draw(  # noqa: PLR0917
        *,
        image: Image.Image,
        origin: tuple[int, int],
        transform: PanelTransform,
        bin_x: float,
        base_x: float,
        completed: Sequence[dict[str, Any]],
        active: dict[str, Any] | None,
        title: str,
        legend: Sequence[tuple[tuple[int, int, int], str, bool]] = (),
    ) -> None:
        draw = ImageDraw.Draw(image)
        left, top = origin
        w, h = transform.width, transform.height
        draw.rectangle([(left, top), (left + w, top + h)], fill=PANEL_BACKGROUND)

        def px(*, x: float, z: float) -> tuple[float, float]:
            column, row = transform.to_pixel(x=x, z=z)
            return left + column, top + row

        # A half-metre grid on both axes, so the isotropy is checkable by eye.
        tick_font = TossPanel.font(name="DejaVuSans.ttf", size=12)
        tick = np.floor(transform.x_min * 2.0) / 2.0
        while tick <= transform.x_max:
            column, _ = px(x=tick, z=0.0)
            draw.line([(column, top), (column, top + h)], fill=GRID_COLOR, width=1)
            draw.text((column + 3, top + h - 16), f"{tick:.1f}", font=tick_font, fill=GRID_COLOR)
            tick += 0.5
        height_tick = 0.5
        while height_tick <= transform.z_max:
            _, row = px(x=0.0, z=height_tick)
            draw.line([(left, row), (left + w, row)], fill=GRID_COLOR, width=1)
            draw.text(
                (left + w - 46, row - 15), f"z={height_tick:.1f}", font=tick_font, fill=GRID_COLOR
            )
            height_tick += 0.5

        ground_left, ground_row = px(x=transform.x_min, z=0.0)
        ground_right, _ = px(x=transform.x_max, z=0.0)
        draw.line(
            [(ground_left, ground_row), (ground_right, ground_row)], fill=GROUND_COLOR, width=2
        )

        # A landmark only: no scoring box is drawn, since the subject is how far the cube
        # flies rather than whether the throw scored.
        bin_col, bin_row = px(x=bin_x, z=0.0)
        draw.line([(bin_col, bin_row), (bin_col, bin_row - 34)], fill=BIN_COLOR, width=2)
        draw.text(
            (bin_col + 4, bin_row - 50),
            "bin",
            font=TossPanel.font(name="DejaVuSans.ttf", size=12),
            fill=BIN_COLOR,
        )

        base_col, base_row = px(x=base_x, z=0.0)
        draw.line([(base_col, base_row), (base_col, base_row - 26)], fill=BASE_COLOR, width=2)
        draw.text(
            (base_col + 4, base_row - 40),
            "robot base",
            font=TossPanel.font(name="DejaVuSans.ttf", size=12),
            fill=BASE_COLOR,
        )

        for arc in completed:
            TossPanel._draw_arc(draw=draw, px=px, arc=arc, bold=False)
        if active is not None:
            TossPanel._draw_arc(draw=draw, px=px, arc=active, bold=True)

        draw.text(
            (left + 12, top + 10),
            title,
            font=TossPanel.font(name="DejaVuSans-Bold.ttf", size=14),
            fill=TEXT_COLOR,
        )
        # In the panel's headroom, which isotropy leaves empty. Doubles as a running tally:
        # an entry stays dim until that throw has landed.
        legend_font = TossPanel.font(name="DejaVuSans-Bold.ttf", size=14)
        for row_index, (color, text, measured) in enumerate(legend):
            y = top + 36 + row_index * 21
            draw.rectangle(
                [(left + 14, y + 4), (left + 34, y + 9)],
                fill=color if measured else tuple(c // 3 for c in color),
            )
            draw.text(
                (left + 42, y),
                text,
                font=legend_font,
                fill=TEXT_COLOR if measured else KEY_COLOR,
            )

    @staticmethod
    def _draw_arc(*, draw: ImageDraw.ImageDraw, px: Any, arc: dict[str, Any], bold: bool) -> None:
        color = tuple(arc["color"])
        points = [px(x=x, z=z) for x, z in zip(arc["xs"], arc["zs"], strict=True)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=4 if bold else 2, joint="curve")
        marker = arc.get("impact_x")
        if marker is not None:
            column, row = px(x=marker, z=0.0)
            radius = 6 if bold else 5
            draw.ellipse(
                [(column - radius, row - radius), (column + radius, row + radius)],
                fill=color,
                outline=BACKGROUND,
            )
            # Above the ground line: below it is off the bottom of the panel, and clipped.
            label_font = TossPanel.font(name="DejaVuSans-Bold.ttf", size=13)
            draw.text(
                (column - 0.5 * draw.textlength(f"{marker:.2f}", font=label_font), row - 26),
                f"{marker:.2f}",
                font=label_font,
                fill=color,
            )
        # The extrapolation the reported distance rests on, for a throw that stopped in
        # mid-air against the bin's wall. The fitted parabola, not a chord.
        tail = arc.get("tail_xs")
        if tail:
            tail_points = [px(x=x, z=z) for x, z in zip(tail, arc["tail_zs"], strict=True)]
            for start, stop in zip(tail_points[::4], tail_points[2::4], strict=False):
                draw.line([start, stop], fill=color, width=2)
        # Tracks the cube *now*, which before release is in the gripper and off the arc.
        # The polyline's last point would park it on the release position through the windup.
        live = arc.get("live")
        if live is not None:
            head = px(x=live[0], z=live[1])
            draw.ellipse(
                [(head[0] - 5, head[1] - 5), (head[0] + 5, head[1] + 5)],
                fill=(255, 255, 255),
                outline=color,
            )


class StatusBar:
    """The bar under the composed frame. A sibling of `recording.StatusBarOverlay`."""

    @staticmethod
    def draw(*, image: Image.Image, top: int, width: int, height: int, fields: Any) -> None:
        draw = ImageDraw.Draw(image)
        draw.rectangle([(0, top), (width, top + height)], fill=BACKGROUND)
        draw.rectangle([(0, top), (width, top + 3)], fill=BIN_COLOR)
        key_font = TossPanel.font(name="DejaVuSans.ttf", size=15)
        value_font = TossPanel.font(name="DejaVuSans-Bold.ttf", size=20)
        margin, line_height = 14, 36
        x, line = margin, 0
        for key, value in fields:
            key_w = draw.textlength(f"{key} ", font=key_font)
            value_w = draw.textlength(value, font=value_font)
            if x > margin and x + key_w + value_w > width - margin:
                line += 1
                if line >= 3:
                    return
                x = margin
            y = top + 14 + line * line_height
            draw.text((x, y + 3), f"{key} ", font=key_font, fill=KEY_COLOR)
            draw.text((x + key_w, y), value, font=value_font, fill=TEXT_COLOR)
            x = int(x + key_w + value_w + 28)


def speed_colors(*, speeds: Sequence[float]) -> list[tuple[int, int, int]]:
    """A sequential ramp, dark = slow."""
    cmap = matplotlib.colormaps["viridis"]
    positions = np.linspace(0.12, 0.92, len(speeds))
    return [tuple(int(round(255 * c)) for c in cmap(p)[:3]) for p in positions]


def _rounded_up(*, value: int) -> int:
    return value + (-value) % MACRO_BLOCK


def _flight_slice(*, clip: dict[str, Any]) -> dict[str, np.ndarray]:
    """The recorded frames between release and first contact. Every extent, arc and apex
    label reads this one slice."""
    times = np.array(clip["frame_times"], dtype=float)
    start = int(np.searchsorted(times, float(clip["release_t"]), side="left"))
    stop = int(np.searchsorted(times, float(clip["land_t"]), side="right"))
    return {
        "start": start,
        "stop": stop,
        "xs": np.array(clip["frame_cube_x"], dtype=float)[start:stop],
        "zs": np.array(clip["frame_cube_z"], dtype=float)[start:stop],
    }


def ballistic_tail(*, clip: dict[str, Any]) -> tuple[list[float], list[float]]:
    """The fitted flight parabola continued from first contact to the ground crossing.

    Empty when the cube reached open floor on its own, non-empty when the bin's wall
    intercepted it in mid-air. Decided on the **height** the footage ended at, not on
    whether the last frame precedes the crossing time -- the latter is always true by up to
    one frame interval, so it would draw a sub-pixel stub under every arc.
    """
    times = np.array(clip["frame_times"], dtype=float)
    flight = _flight_slice(clip=clip)
    span = times[int(flight["start"]) : int(flight["stop"])]
    if len(span) < 3 or float(flight["zs"][-1]) <= TAIL_MIN_HEIGHT:
        return [], []
    rel = span - span[0]
    z_coeffs = np.polyfit(rel, flight["zs"], 2)
    x_coeffs = np.polyfit(rel, flight["xs"], 1)
    end = float(clip["ballistic_impact_t"]) - float(span[0])
    if end <= float(rel[-1]):
        return [], []
    extra = np.linspace(float(rel[-1]), end, 48)
    return (
        np.polyval(x_coeffs, extra).tolist(),
        np.polyval(z_coeffs, extra).tolist(),
    )


def legend_rows(
    *, clips: Sequence[dict[str, Any]], colors: Sequence[tuple[int, int, int]], measured: int
) -> list[tuple[tuple[int, int, int], str, bool]]:
    """One row per `(speed, release ms)` cell, dim until that throw has landed on screen.

    `measured` is how many throws the video has finished: a running tally, not a spoiler."""
    rows: list[tuple[tuple[int, int, int], str, bool]] = []
    for index, clip in enumerate(clips):
        speed = float(clip["commanded_speed_deg"])
        release_ms = float(clip["commanded_release_ms"])
        head = f"{speed:3.0f} deg/s @ {release_ms:4.0f} ms"
        if index < measured:
            crossing = float(clip["ballistic_impact_x"])
            thrown = crossing - float(clip["base_x_before_toss"])
            resting = float(clip["cube_x_final"]) - float(clip["base_x_before_toss"])
            text = f"{head}   ballistic {thrown:.3f} m   resting {resting:.3f} m"
        else:
            text = f"{head}   not thrown yet"
        rows.append((colors[index], text, index < measured))
    return rows


def build_video(*, clips_dir: Path, output_video: Path, output_figure: Path) -> dict[str, Any]:
    """Read the probe's output back and write the composed video and the finale still."""
    import imageio.v2 as imageio

    payload = json.loads((clips_dir / "tosses.json").read_text())
    clips = payload["clips"]
    colors = speed_colors(speeds=[c["commanded_speed_deg"] for c in clips])

    base_x = float(clips[0]["base_x_before_toss"])
    bin_x = float(clips[0]["bin_x"])
    # From the flight only: the windup lifts the cube ~0.5 m above anything it reaches in
    # the air, and isotropy pays that inflation back in width, pushing every arc into a
    # corner.
    flights = [_flight_slice(clip=clip) for clip in clips]
    x_lo = min(base_x, min(float(np.min(f["xs"])) for f in flights)) - 0.15
    x_hi = (
        max(
            max(float(c["ballistic_impact_x"]) for c in clips),
            max(float(np.max(f["xs"])) for f in flights),
            bin_x,
        )
        + 0.20
    )
    z_hi = max(float(np.max(f["zs"])) for f in flights) + 0.12

    probe_frame = imageio.get_reader(clips_dir / clips[0]["clip_filename"]).get_data(0)
    frame_h, frame_w = int(probe_frame.shape[0]), int(probe_frame.shape[1])
    panel_w, panel_h = frame_w, frame_h
    canvas_w = _rounded_up(value=frame_w + panel_w)
    canvas_h = _rounded_up(value=frame_h + BAR_HEIGHT)
    transform = PanelTransform.fit(
        x_min=x_lo, x_max=x_hi, z_min=0.0, z_max=z_hi, width=panel_w, height=panel_h
    )

    completed: list[dict[str, Any]] = []
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output_video,
        fps=PLAYBACK_FPS,
        codec="libx264",
        macro_block_size=1,
        output_params=["-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p"],
    )
    written = 0
    try:
        for index, clip in enumerate(clips):
            times = np.array(clip["frame_times"], dtype=float)
            xs = np.array(clip["frame_cube_x"], dtype=float)
            zs = np.array(clip["frame_cube_z"], dtype=float)
            indices = playback_indices(
                times=times,
                release_t=float(clip["release_t"]),
                land_t=float(clip["land_t"]),
                pre_seconds=PRE_RELEASE_SECONDS,
                post_seconds=POST_LANDING_SECONDS,
                stride=PLAYBACK_STRIDE,
            )
            flight = _flight_slice(clip=clip)
            release_index, land_index = int(flight["start"]), int(flight["stop"])
            color = colors[index]
            tail_xs, tail_zs = ballistic_tail(clip=clip)

            reader = imageio.get_reader(clips_dir / clip["clip_filename"])
            try:
                for i in indices:
                    phase = phase_at(
                        t=float(times[i]),
                        release_t=float(clip["release_t"]),
                        land_t=float(clip["land_t"]),
                    )
                    trace_stop = max(release_index + 1, min(i + 1, land_index))
                    landed = phase is TossPhase.LANDED
                    airborne = phase is not TossPhase.SWING
                    active = {
                        "xs": xs[release_index:trace_stop].tolist() if airborne else [],
                        "zs": zs[release_index:trace_stop].tolist() if airborne else [],
                        "live": (float(xs[i]), float(zs[i])),
                        "color": color,
                        "tail_xs": tail_xs if landed else [],
                        "tail_zs": tail_zs if landed else [],
                        "impact_x": (float(clip["ballistic_impact_x"]) if landed else None),
                    }
                    canvas = Image.new("RGB", (canvas_w, canvas_h), BACKGROUND)
                    canvas.paste(
                        Image.fromarray(np.ascontiguousarray(reader.get_data(i), dtype=np.uint8)),
                        (0, 0),
                    )
                    TossPanel.draw(
                        image=canvas,
                        origin=(frame_w, 0),
                        transform=transform,
                        bin_x=bin_x,
                        base_x=base_x,
                        completed=completed,
                        active=active,
                        title="side elevation, to scale — arcs accumulate",
                        legend=legend_rows(clips=clips, colors=colors, measured=index),
                    )
                    StatusBar.draw(
                        image=canvas,
                        top=frame_h,
                        width=canvas_w,
                        height=canvas_h - frame_h,
                        fields=status_fields(
                            speed=float(clip["commanded_speed_deg"]),
                            release_ms=float(clip["commanded_release_ms"]),
                            index=index,
                            total=len(clips),
                            seed=int(clip["seed"]),
                            standoff=float(clip["standoff"]),
                            base_x=base_x,
                            phase=phase,
                            cube_x=float(xs[i]),
                            impact_x=float(clip["ballistic_impact_x"]),
                            resting_x=float(clip["cube_x_final"]),
                            solved=clip["solved"],
                        ),
                    )
                    writer.append_data(np.asarray(canvas, dtype=np.uint8))
                    written += 1
            finally:
                reader.close()

            completed.append({
                "xs": xs[release_index:land_index].tolist(),
                "zs": zs[release_index:land_index].tolist(),
                "color": color,
                "tail_xs": tail_xs,
                "tail_zs": tail_zs,
                "impact_x": float(clip["ballistic_impact_x"]),
            })

        finale = _finale_frame(
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            transform=transform,
            bin_x=bin_x,
            base_x=base_x,
            completed=completed,
            clips=clips,
            colors=colors,
        )
        for _ in range(int(FINALE_SECONDS * PLAYBACK_FPS)):
            writer.append_data(np.asarray(finale, dtype=np.uint8))
            written += 1
    finally:
        writer.close()

    output_figure.parent.mkdir(parents=True, exist_ok=True)
    finale.save(output_figure)
    return {
        "frames": written,
        "seconds": written / PLAYBACK_FPS,
        "video": str(output_video),
        "figure": str(output_figure),
    }


def _finale_frame(  # noqa: PLR0917
    *,
    canvas_w: int,
    canvas_h: int,
    transform: PanelTransform,
    bin_x: float,
    base_x: float,
    completed: Sequence[dict[str, Any]],
    clips: Sequence[dict[str, Any]],
    colors: Sequence[tuple[int, int, int]],
) -> Image.Image:
    """All the arcs on one full-width panel, with the numbers under them.

    Full width, since a frozen simulator frame would suggest one throw is the subject.
    """
    wide = PanelTransform.fit(
        x_min=transform.x_min,
        x_max=transform.x_max,
        z_min=transform.z_min,
        z_max=transform.z_max,
        width=canvas_w,
        height=canvas_h - BAR_HEIGHT,
    )
    canvas = Image.new("RGB", (canvas_w, canvas_h), BACKGROUND)
    TossPanel.draw(
        image=canvas,
        origin=(0, 0),
        transform=wide,
        bin_x=bin_x,
        base_x=base_x,
        completed=completed,
        active=None,
        title=(
            f"Tossing3D, seed {clips[0]['seed']}, standoff {clips[0]['standoff']:.2f} m — "
            f"{len(clips)} throws, to scale, labelled by both dials "
            "(commanded release speed, gripper release ms)"
        ),
        legend=legend_rows(clips=clips, colors=colors, measured=len(clips)),
    )
    draw = ImageDraw.Draw(canvas)
    top = canvas_h - BAR_HEIGHT
    draw.rectangle([(0, top), (canvas_w, canvas_h)], fill=BACKGROUND)
    draw.rectangle([(0, top), (canvas_w, top + 3)], fill=BIN_COLOR)
    key_font = TossPanel.font(name="DejaVuSans.ttf", size=14)
    value_font = TossPanel.font(name="DejaVuSans-Bold.ttf", size=17)
    base = float(clips[0]["base_x_before_toss"])
    distances = [float(c["ballistic_impact_x"]) - base for c in clips]
    draw.text(
        (14, top + 12),
        "ballistic distance from the robot base, in metres, per (speed, release ms) cell",
        font=key_font,
        fill=KEY_COLOR,
    )
    # One row per speed group: nine labels on one line are clipped at the right edge.
    per_row = max(1, len({c["commanded_speed_deg"] for c in clips}))
    x = 14
    row = 0
    for index, clip in enumerate(clips):
        if index and index % (len(clips) // per_row) == 0:
            row += 1
            x = 14
        text = (
            f"{clip['commanded_speed_deg']:.0f}@{clip['commanded_release_ms']:.0f}: "
            f"{distances[index]:.3f}"
        )
        draw.text((x, top + 34 + row * 20), text, font=value_font, fill=colors[index])
        x = int(x + draw.textlength(text, font=value_font) + 22)
    # The reach the two dials open up. Not "first minus last": the surface is not monotone
    # in either dial, so an endpoint difference is not the span.
    draw.text(
        (int(canvas_w * 0.62), top + 12),
        f"reach over these {len(clips)} cells: "
        f"{min(distances):.3f} — {max(distances):.3f} m "
        f"(span {max(distances) - min(distances):.3f} m)",
        font=key_font,
        fill=TEXT_COLOR,
    )
    return canvas


def _parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    result = build_video(
        clips_dir=args.clips_dir,
        output_video=args.output_video,
        output_figure=args.output_figure,
    )
    print(
        f"wrote {result['video']} ({result['frames']} frames, {result['seconds']:.1f}s) "
        f"and {result['figure']}"
    )


if __name__ == "__main__":
    main()
