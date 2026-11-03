"""Whether the dumplings are at rest, judged by where their centres have been."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# Float accumulation in MuJoCo's clock (64.90000000012786 after 649 ticks) must not
# push the sample exactly one window back out of it.
TIME_TOLERANCE_S = 1e-6


class PayloadRestWindow(BaseModel):
    """Settled iff no centre moved more than `max_displacement_m` over the trailing
    `window_seconds`, with recorded samples spanning that whole window.

    KinDER's instantaneous speed rule misjudges a flat dumpling balanced on a single
    mesh contact point: it rocks in place at up to ~2.2 rad/s while its centre stays
    within about 1 mm, so a correctly split pour was scored as unsettled.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Long enough to span several rocking periods and the 0.1 s control tick many times.
    window_seconds: float = 2.0
    # Above the ~1.9 mm worst rocking excursion measured, below a real slide.
    max_displacement_m: float = 0.005
    samples: list[tuple[float, np.ndarray]] = Field(default_factory=list)

    def record(self, *, time: float, positions: np.ndarray) -> None:
        if self.samples and time < self.samples[-1][0] - TIME_TOLERANCE_S:
            # A restored earlier state: the old trajectory is not this one's past.
            self.samples.clear()
        self.samples.append((time, np.array(positions, dtype=float)))
        horizon = time - self.window_seconds - TIME_TOLERANCE_S
        # Keep the newest sample at or before the window start so coverage is known.
        while len(self.samples) > 1 and self.samples[1][0] <= horizon:
            self.samples.pop(0)

    def clear(self) -> None:
        self.samples.clear()

    def settled(self, *, time: float, positions: np.ndarray) -> bool:
        return bool(self.resting(time=time, positions=positions).all())

    def resting(self, *, time: float, positions: np.ndarray) -> np.ndarray:
        """Per piece: did its centre stay within the threshold over the whole window?"""
        current = np.asarray(positions, dtype=float)
        unknown = np.zeros(len(current), dtype=bool)
        if any(t > time + TIME_TOLERANCE_S for t, _ in self.samples):
            return unknown  # rewound to an earlier state; this history is not its past
        start = time - self.window_seconds
        window = [
            p for t, p in self.samples if start - TIME_TOLERANCE_S <= t <= time + TIME_TOLERANCE_S
        ]
        covered = any(t <= start + TIME_TOLERANCE_S for t, _ in self.samples)
        if not window or not covered:
            return unknown
        # The live positions join the window, so an unrecorded jump still counts.
        track = np.stack([*window, current])
        moved = np.linalg.norm(track - track[0], axis=-1)
        return moved.max(axis=0) <= self.max_displacement_m
