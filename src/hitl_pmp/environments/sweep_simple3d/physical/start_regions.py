"""Start validation data and native yaw-range checks for Simple."""

import numpy as np
from pydantic import BaseModel


class StartValidation(BaseModel):
    seed: int
    valid: bool
    checks: dict[str, bool]
    poses: dict[str, list[float]]
    reasons: tuple[str, ...]


class SweepRegions:
    """Native orientation interval membership."""

    @staticmethod
    def yaw_matches(*, yaw: float, ranges: list[list[float]], tolerance_deg: float = 0.0) -> bool:
        degrees = float(np.degrees(yaw) % 360)
        for low, high in ranges:
            if high - low >= 360:
                return True
            for candidate in (degrees - 360, degrees, degrees + 360):
                if low - tolerance_deg <= candidate <= high + tolerance_deg:
                    return True
        return False
