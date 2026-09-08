"""Comparable write timestamps for event logs within one process."""

import json
import math
import time
from datetime import datetime, timezone
from typing import Any, ClassVar


class LogTiming:
    """Elapsed time starts when logging is imported; wall time correlates processes."""

    started_at: ClassVar[float] = time.monotonic()

    @staticmethod
    def fields() -> dict[str, str | float]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - LogTiming.started_at,
        }

    @staticmethod
    def encode(*, record: dict[str, Any]) -> str:
        payload = _replace_non_finite_floats(value={**record, **LogTiming.fields()})
        return json.dumps(payload, allow_nan=False) + "\n"


def _replace_non_finite_floats(*, value: Any) -> Any:
    """Represent non-finite diagnostic values without emitting invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {
            key: _replace_non_finite_floats(value=item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_non_finite_floats(value=item) for item in value]
    if isinstance(value, tuple):
        return [_replace_non_finite_floats(value=item) for item in value]
    return value
